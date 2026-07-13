"""
Unified API server for Doubao (Playwright browser-based).

Exposes OpenAI-compatible endpoints:
  POST /v1/chat/completions     (chat, streaming & non-streaming)
  GET  /v1/models               (list available models)
  GET  /health                  (health check)
  GET  /auth                    (QR login page)

Start with:
    python -m doubao2api
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import collections
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .account_manager import DoubaoAccountManager, account_data_root
from .browser_client import BrowserClient
from .qianwen_client import QianwenClient, QIANWEN_MODELS
from .video_tasks import VideoTaskStore, video_task_db_path
from .video_request import collect_video_reference_values, extract_video_prompt
from .tool_calling import (
    build_tool_system_prompt,
    convert_messages_with_tools,
    parse_tool_calls_xml,
    is_tool_call_start,
    has_complete_tool_calls,
    StreamingGuard,
    detect_truncated_tool_call,
    build_continuation_prompt,
    filter_history_by_topic,
    ToolNameObfuscator,
    coerce_tool_arguments,
    deduplicate_continuation,
)
from .token_counter import count_tokens, count_messages_tokens, SAFETY_FACTOR

log = logging.getLogger("doubao_unified")

# ── Tool Name Obfuscation (enabled via QIANWEN_OBFUSCATE_TOOLS=true) ──
_tool_obfuscator = ToolNameObfuscator(
    enabled=os.environ.get("QIANWEN_OBFUSCATE_TOOLS", "false").lower() == "true"
)

# ── Model definitions ────────────────────────────────────────

CHAT_MODELS: Dict[str, int] = {
    "doubao": 0,
    "doubao-pro": 0,
    "doubao-think": 1,
    "doubao-expert": 3,
}

VIDEO_MODEL_ALIASES: Dict[str, Optional[str]] = {
    "doubao-video": None,
    "seedance_v2.0": "seedance_v2.0",
    "seedance2.0": "seedance_v2.0",
    "seedance2.0fast": "seedance_v2.0",
    "seedance-2.0-fast": "seedance_v2.0",
    "seedance_2.0_fast": "seedance_v2.0",
    "seedance_v2.0_fast": "seedance_v2.0",
    "Seedance 2.0": "seedance_v2.0",
    "Seedance 2.0 Fast": "seedance_v2.0",
}
VIDEO_MODEL_ALIASES_LOWER = {model.lower() for model in VIDEO_MODEL_ALIASES}


def _is_video_model_name(model: Optional[str]) -> bool:
    normalized = (model or "").strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    compact = re.sub(r"[\s_-]+", "", lowered)
    return (
        normalized in VIDEO_MODEL_ALIASES
        or lowered in VIDEO_MODEL_ALIASES_LOWER
        or compact.startswith("seedance")
        or lowered.startswith("doubao-video")
    )

# Qianwen models (routed to QianwenClient)
QIANWEN_MODEL_NAMES = set(QIANWEN_MODELS.keys())

ALL_MODELS = [
    {"id": m, "object": "model", "owned_by": "doubao", "created": 0}
    for m in CHAT_MODELS
] + [
    {"id": m, "object": "model", "owned_by": "qianwen", "created": 0}
    for m in QIANWEN_MODELS
] + [
    {"id": "doubao-image", "object": "model", "owned_by": "doubao", "created": 0},
    {"id": "doubao-music", "object": "model", "owned_by": "doubao", "created": 0},
    *[
        {"id": m, "object": "model", "owned_by": "doubao", "created": 0}
        for m in VIDEO_MODEL_ALIASES
    ],
]


# ── Expert Mode Quota Tracker ──
class ExpertQuotaTracker:
    """Detects when expert mode is silently downgraded and falls back to think."""

    def __init__(self, consecutive_threshold: int = 2, retry_interval: int = 1800):
        self._no_reasoning_count = 0  # consecutive expert requests without reasoning
        self._threshold = consecutive_threshold  # how many before marking degraded
        self._degraded = False
        self._last_retry_time = 0.0
        self._retry_interval = retry_interval  # seconds before retrying expert (30 min)

    @property
    def is_degraded(self) -> bool:
        """True if expert mode appears to be quota-limited."""
        if not self._degraded:
            return False
        # Periodically retry
        import time
        if time.time() - self._last_retry_time > self._retry_interval:
            return False  # Allow a retry
        return True

    def report_response(self, had_reasoning: bool):
        """Call after each expert-mode request with whether reasoning was present."""
        import time
        if had_reasoning:
            self._no_reasoning_count = 0
            if self._degraded:
                log.info("Expert mode recovered (reasoning detected)")
            self._degraded = False
        else:
            self._no_reasoning_count += 1
            if self._no_reasoning_count >= self._threshold and not self._degraded:
                self._degraded = True
                self._last_retry_time = time.time()
                log.warning("Expert mode appears degraded (no reasoning for %d requests), falling back to think", self._threshold)

    def mark_retry(self):
        """Mark that we're doing a retry probe."""
        import time
        self._last_retry_time = time.time()

    def get_effective_mode(self, requested_deep_think: int) -> tuple[int, str]:
        """Return (deep_think_value, model_name) considering degradation.

        If expert (3) is degraded, falls back to think (1).
        """
        if requested_deep_think == 3 and self.is_degraded:
            return 1, "doubao-think"
        model_map = {0: "doubao", 1: "doubao-think", 3: "doubao-expert"}
        return requested_deep_think, model_map.get(requested_deep_think, "doubao")


_expert_tracker = ExpertQuotaTracker()


def _size_to_ratio(size):
    """Convert OpenAI size format to Doubao ratio."""
    if not size:
        return "1:1"
    size_map = {
        "1024x1024": "1:1",
        "1792x1024": "16:9",
        "1024x1792": "9:16",
        "1024x768": "4:3",
        "768x1024": "3:4",
    }
    if size in size_map:
        return size_map[size]
    if ":" in size:
        return size
    return "1:1"


# ── Request log ring buffer ───────────────────────────────────

_REQUEST_LOG: collections.deque = collections.deque(maxlen=100)
_SERVER_START_TIME: float = time.time()


# ── Rate limiter ─────────────────────────────────────────────


class _TokenBucket:
    """Simple async token-bucket rate limiter."""

    def __init__(self, rpm: float):
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = now + self._interval
                    return
                wait_time = self._next_allowed - now
            await asyncio.sleep(wait_time)


# ── Pydantic request models ──────────────────────────────────


class _Message(BaseModel):
    role: str
    content: Any  # str | list[dict]
    tool_calls: Optional[list] = None  # for assistant messages with tool calls
    tool_call_id: Optional[str] = None  # for role:tool messages
    name: Optional[str] = None  # tool name for role:tool messages


class ChatCompletionRequest(BaseModel):
    model: str = "doubao"
    messages: List[_Message]
    stream: bool = False
    account_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    conversation_id: Optional[str] = None
    bot_id: Optional[str] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Any] = None  # "auto" | "none" | {"type":"function","function":{"name":"..."}}
    enable_thinking: Optional[bool] = None  # triggers deep_search="1" for thinking mode
    reasoning_effort: Optional[str] = None  # "low"|"medium"|"high" — also triggers thinking



class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "doubao-image"
    n: int = 1
    size: Optional[str] = "1024x1024"
    ratio: Optional[str] = None
    ref_image_key: Optional[str] = None
    response_format: Optional[str] = "url"
    account_id: Optional[str] = None

# ── Application factory ──────────────────────────────────────


def is_quota_exhaustion_message(message: str) -> bool:
    text = (message or "").lower()
    if re.search(r"(今日|今天)?\s*(剩余|还剩)\s*[1-9]\d*\s*个?(视频生成额度|视频额度|生成额度|额度)", text):
        return False
    if re.search(r"(今日|今天)?\s*(剩余|还剩)\s*0\s*个?(视频生成额度|视频额度|生成额度|额度)?", text):
        return True
    if "视频生成" in text and any(
        marker in text
        for marker in (
            "今日视频生成免费次数已用完",
            "视频生成免费次数已用完",
            "免费次数已用完",
            "次数已用完",
            "即可继续使用视频生成",
            "继续使用视频生成",
            "开通豆包专业版",
            "开通加强套餐",
            "开通套餐",
        )
    ):
        return True
    if re.search(r"(今日|今天)?\s*(剩余|还剩)\s*[1-9]\d*\s*个?(视频生成额度|视频额度|生成额度|额度)", text):
        return False
    if re.search(r"(今日|今天)?\s*(剩余|还剩)\s*0\s*个?(视频生成额度|视频额度|生成额度|额度)?", text):
        return True
    markers = (
        "exceed",
        "exceeded",
        "insufficient",
        "not enough",
        "\u6b21\u6570",
        "\u7528\u5b8c",
        "\u4e0d\u8db3",
        "\u4e0a\u9650",
        "\u4eca\u65e5\u5269\u4f59 0",
        "\u5269\u4f590",
        "\u8fd8\u5269 0",
        "\u8fd8\u52690",
        "\u989d\u5ea6\u4e0d\u8db3",
        "\u989d\u5ea6\u5df2\u7528\u5b8c",
        "\u89c6\u9891\u751f\u6210\u989d\u5ea6\u5df2\u7528\u5b8c",
        "\u79ef\u5206\u4e0d\u8db3",
        "\u4f59\u989d\u4e0d\u8db3",
        "\u6743\u76ca\u4e0d\u8db3",
        "\u6ca1\u6709\u76f8\u5173\u6743\u76ca",
        "\u6ca1\u6709\u89c6\u9891\u751f\u6210\u6743\u76ca",
        "\u6b21\u6570\u4e0d\u8db3",
        "\u6b21\u6570\u5df2\u7528\u5b8c",
        "\u514d\u8d39\u6b21\u6570\u5df2\u7528\u5b8c",
        "次数",
        "用完",
        "不足",
        "上限",
        "今日剩余 0",
        "剩余0",
        "还剩 0",
        "还剩0",
        "额度不足",
        "额度已用完",
        "视频生成额度已用完",
        "quota exceeded",
        "quota exhausted",
        "quota insufficient",
        "quota limit",
        "rate limit",
        "积分不足",
        "余额不足",
        "权益不足",
        "没有相关权益",
        "没有视频生成权益",
        "次数不足",
        "次数已用完",
        "免费次数已用完",
        "额度不足",
        "额度已用完",
        "视频生成额度已用完",
    )
    return any(marker in text for marker in markers)


def is_video_acceptance_message(message: str) -> bool:
    if is_quota_exhaustion_message(message):
        return False
    text = message or ""
    markers = (
        "\u6b63\u5728\u4e3a\u60a8\u751f\u6210\u89c6\u9891",
        "\u89c6\u9891\u751f\u6210\u597d\u540e",
        "\u751f\u6210\u597d\u540e",
        "\u9884\u8ba1\u7b49\u5f85",
        "\u6b63\u5728\u751f\u6210",
        "正在为您生成视频",
        "预计等待",
        "生成好后",
        "视频生成好后",
        "will notify",
        "generating video",
    )
    return any(marker.lower() in text.lower() for marker in markers)


def create_app(
    *,
    api_key: Optional[str] = None,
    rpm_limit: float = 20.0,
) -> FastAPI:
    """Build and return a configured FastAPI application."""

    headless = os.environ.get("DOUBAO_HEADLESS", "true").lower() == "true"
    accounts = DoubaoAccountManager(headless=headless)
    _qianwen: Dict[str, Any] = {}  # holds QianwenClient instance

    async def _browser_watchdog():
        """Background task: check hot account browser health every 30s."""
        while True:
            await asyncio.sleep(30)
            for account_id, client in list(accounts.clients.items()):
                try:
                    alive = await client.is_alive()
                    if not alive:
                        log.error("Browser watchdog: account %s process dead, restarting...", account_id)
                        await accounts.restart_client(account_id)
                        fresh = accounts.clients.get(account_id)
                        if fresh and fresh.is_ready:
                            log.info("Browser watchdog: account %s restart successful", account_id)
                        else:
                            log.warning("Browser watchdog: account %s restarted but not logged in", account_id)
                except Exception as e:
                    log.error("Browser watchdog error for account %s: %s", account_id, e)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Ensure browser_client logs are visible
        logging.getLogger("doubao2api.browser_client").setLevel(logging.INFO)
        logging.getLogger("doubao2api.browser_client").addHandler(logging.StreamHandler())

        await accounts.start()
        ready_accounts = [a for a in accounts.list_accounts() if a.get("runtime", {}).get("ready")]
        if ready_accounts:
            log.info("Doubao account pool ready: %d account(s)", len(ready_accounts))
        else:
            log.warning("No logged-in Doubao account. Visit /admin to add or scan account.")

        # Start browser watchdog
        watchdog_task = asyncio.create_task(_browser_watchdog())
        recovery_task = asyncio.create_task(_video_recovery_worker())

        # Start Qianwen client (optional, enabled via env var)
        qw_client = None
        if os.environ.get("QIANWEN_ENABLED", "false").lower() == "true":
            qw_headless = os.environ.get("QIANWEN_HEADLESS", "true").lower() == "true"
            qw_data_dir = os.environ.get(
                "QIANWEN_BROWSER_DATA",
                os.path.join(os.path.expanduser("~"), ".qianwen_browser"),
            )
            qw_client = QianwenClient(headless=qw_headless, user_data_dir=qw_data_dir)
            try:
                await qw_client.start()
                _qianwen["client"] = qw_client
                log.info("Qianwen client ready")
            except Exception as e:
                log.warning("Qianwen client failed to start: %s", e)
                qw_client = None

        yield

        # Shutdown
        watchdog_task.cancel()
        recovery_task.cancel()
        await accounts.stop_all()
        qw = _qianwen.pop("client", None)
        if qw:
            await qw.stop()

    app = FastAPI(title="Doubao API", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "api_error", "code": exc.status_code}},
        )

    @app.exception_handler(Exception)
    async def _unhandled_exc(request: Request, exc: Exception):
        log.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "internal_error", "code": 500}},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    bucket = _TokenBucket(rpm_limit)
    video_tasks = VideoTaskStore(video_task_db_path())
    video_tasks.mark_interrupted()
    video_tasks.normalize_completed()
    video_tasks.cleanup()
    _video_recovery_locks: Dict[str, asyncio.Lock] = {}
    _video_account_locks: Dict[str, asyncio.Lock] = {}

    def _video_lock(task_id: str) -> asyncio.Lock:
        lock = _video_recovery_locks.get(task_id)
        if lock is None:
            lock = asyncio.Lock()
            _video_recovery_locks[task_id] = lock
        return lock

    def _video_account_lock(account_id: str) -> asyncio.Lock:
        lock = _video_account_locks.get(account_id)
        if lock is None:
            lock = asyncio.Lock()
            _video_account_locks[account_id] = lock
        return lock

    def _accepted_video_result(result: Dict[str, Any]) -> bool:
        has_binding_id = bool(
            str(result.get("provider_task_id") or "").strip()
            or str(result.get("conversation_id") or "").strip()
            or str(result.get("conversation_url") or "").strip()
        )
        if not has_binding_id:
            return False
        if result.get("accepted") or result.get("pending"):
            return True
        return is_video_acceptance_message(str(result.get("message") or ""))

    def _video_task_update_fields(result: Dict[str, Any], *, accepted: bool = False) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        for key in ("provider_task_id", "conversation_id", "local_conversation_id"):
            if result.get(key):
                fields[key] = result.get(key)
        if accepted:
            fields["accepted_at"] = int(time.time())
        return fields

    def _video_task_binding_ids(task: Dict[str, Any]) -> tuple[str, str, str, str]:
        provider_task_id = str(task.get("provider_task_id") or "")
        conversation_id = str(task.get("conversation_id") or "")
        local_conversation_id = str(task.get("local_conversation_id") or "")
        conversation_url = ""
        if task.get("result_json"):
            try:
                result = json.loads(str(task["result_json"]))
            except json.JSONDecodeError:
                result = {}
            if isinstance(result, dict):
                provider_task_id = provider_task_id or str(result.get("provider_task_id") or "")
                conversation_id = conversation_id or str(result.get("conversation_id") or "")
                local_conversation_id = local_conversation_id or str(result.get("local_conversation_id") or "")
                conversation_url = str(result.get("conversation_url") or "")
        return provider_task_id, conversation_id, local_conversation_id, conversation_url

    def _video_task_provider_task_candidates(task: Dict[str, Any]) -> list[str]:
        candidates: list[str] = []

        def add(value: Any) -> None:
            text = str(value or "").strip()
            if text and text != "0" and text not in candidates:
                candidates.append(text)

        add(task.get("provider_task_id"))
        if task.get("result_json"):
            try:
                result = json.loads(str(task["result_json"]))
            except json.JSONDecodeError:
                result = {}
            if isinstance(result, dict):
                add(result.get("provider_task_id"))
                raw_candidates = result.get("provider_task_candidates") or []
                if isinstance(raw_candidates, list):
                    for value in raw_candidates:
                        add(value)
        return candidates

    # ── Auth helper ──

    def _effective_api_key() -> str:
        value = accounts.store.get_setting("api_key", "").strip()
        return value or (api_key or "").strip()

    def _check_auth(request: Request) -> None:
        active_api_key = _effective_api_key()
        if not active_api_key:
            return
        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else auth.strip()
        if not token:
            token = request.query_params.get("key", "").strip()
        if active_api_key == "any":
            if not token:
                raise HTTPException(status_code=401, detail="API key required")
            return
        if token != active_api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    async def _get_account_client(
        request: Optional[Request] = None,
        body: Optional[Dict[str, Any]] = None,
        account_id: Optional[str] = None,
        quota_kind: Optional[str] = None,
        quota_units: int = 1,
    ) -> tuple[Dict[str, Any], BrowserClient]:
        preferred = account_id
        if not preferred and body is not None:
            headers = dict(request.headers) if request else {}
            preferred = accounts.pick_account_id_from_request(headers, body)
        elif not preferred and request is not None:
            preferred = request.query_params.get("account_id") or request.query_params.get("doubao_account_id")
            if not preferred:
                preferred = accounts.pick_account_id_from_request(dict(request.headers), None)
        try:
            return await accounts.get_ready_client(
                preferred,
                quota_kind=quota_kind,
                quota_units=quota_units,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def _looks_quota_error(message: str) -> bool:
        return is_quota_exhaustion_message(message)

    def _looks_account_state_error(message: str) -> bool:
        text = (message or "").lower()
        if not text:
            return False
        business_markers = (
            "no images generated",
            "no videos generated",
            "no music tracks generated",
            "please upload",
            "请上传",
            "服务过载",
            "service overloaded",
            "生成失败",
            "invalid image data",
            "missing prompt",
        )
        if any(marker in text for marker in business_markers):
            return False
        account_markers = (
            "not logged in",
            "login",
            "未登录",
            "cookie",
            "session expired",
            "invalid session",
            "auth error",
            "unauthorized",
            "forbidden",
            "captcha",
            "验证码",
            "risk",
            "风控",
            "temporarily blocked",
            "browser not ready",
            "browser not available",
            "page crashed",
            "context closed",
            "target closed",
            "401",
            "403",
        )
        return any(marker in text for marker in account_markers)

    def _record_provider_failure(account_id: str, message: str) -> None:
        if _looks_account_state_error(message):
            accounts.mark_failure(account_id, message)
        else:
            accounts.record_task_failure(account_id, message)

    class _VideoAttemptFailed(RuntimeError):
        def __init__(self, account_id: str, message: str, retry_next_account: bool = False):
            super().__init__(message)
            self.account_id = account_id
            self.retry_next_account = retry_next_account

    def _request_has_account_selector(request: Optional[Request], body: Optional[Dict[str, Any]] = None) -> bool:
        body = body or {}
        if any(body.get(key) for key in ("account_id", "doubao_account_id", "account")):
            return True
        if not request:
            return False
        if request.query_params.get("account_id") or request.query_params.get("doubao_account_id"):
            return True
        headers = dict(request.headers)
        return bool(accounts.pick_account_id_from_request(headers, None))

    def _image_quota_units(body: ImageGenerationRequest) -> int:
        try:
            return max(1, int(body.n or 1))
        except (TypeError, ValueError):
            return 1

    def _video_quota_units(params: Dict[str, Any]) -> int:
        duration = params.get("duration")
        if duration is None:
            return 1
        try:
            duration_value = int(duration)
        except (TypeError, ValueError):
            return 1
        # Doubao currently presents short video quota roughly as 5s=1, 10s=2.
        return max(1, (duration_value + 4) // 5)

    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return False

    def _normalize_video_duration(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            duration = int(float(value))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="duration must be a number")
        if duration <= 0:
            raise HTTPException(status_code=400, detail="duration must be positive")
        return duration

    def _parse_video_request(body: Dict[str, Any]) -> Dict[str, Any]:
        prompt = extract_video_prompt(body)
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing prompt")

        ratio = body.get("ratio") or body.get("aspect_ratio") or body.get("size")
        if ratio and "x" in str(ratio):
            ratio = _size_to_ratio(str(ratio))
        if ratio is not None:
            ratio = str(ratio)

        model_name = str(body.get("model") or "doubao-video")
        requested_model = body.get("video_model") or body.get("provider_model")
        if not requested_model:
            requested_model = VIDEO_MODEL_ALIASES.get(model_name)
            if not requested_model and model_name.startswith("seedance"):
                requested_model = model_name
        if requested_model is not None:
            requested_model = str(requested_model)

        duration = _normalize_video_duration(
            body.get("duration", body.get("duration_seconds", body.get("seconds")))
        )
        reference_images = collect_video_reference_values(body)
        ref_image_key = reference_images[0] if reference_images else None

        return {
            "prompt": prompt,
            "model": model_name,
            "provider_model": requested_model,
            "ratio": ratio,
            "duration": duration,
            "ref_image_key": ref_image_key,
            "reference_images": reference_images,
            "reference_image_keys": [],
            "image_url": body.get("image_url"),
            "image": body.get("image"),
            "account_id": str(body.get("account_id") or body.get("doubao_account_id") or body.get("account") or "").strip() or None,
            "explicit_account_id": bool(body.get("account_id") or body.get("doubao_account_id") or body.get("account")),
        }

    def _wants_sync_video(body: Dict[str, Any]) -> bool:
        if body.get("async") is False:
            return True
        return any(_truthy(body.get(name)) for name in ("wait", "sync", "blocking"))

    async def _probe_after_video_failure(account_id: str, client: BrowserClient, message: str) -> bool:
        try:
            result = await client.chat("1+1=?只回答数字", use_deep_think=0)
            text = str(result.get("text") or "")
            if text.strip():
                return True
        except Exception as exc:
            log.warning("video failure probe failed for account %s: %s", account_id, exc)
            accounts.mark_failure(account_id, f"video failed, probe failed: {exc}")
            return False
        log.warning("video failure probe returned empty response for account %s: %s", account_id, message)
        accounts.mark_failure(account_id, "video failed, probe returned empty response")
        return False

    def _zero_video_quota(account_id: str, source: str, message: str) -> None:
        accounts.mark_quota_exhausted(account_id, "video", message)
        accounts.store.update_provider_quota(
            account_id,
            "video",
            remaining=0,
            source=source,
            message=message,
        )

    async def _handle_video_attempt_failure(
        account: Dict[str, Any],
        client: BrowserClient,
        reservation_id: str,
        message: str,
        quota_units: int = 1,
    ) -> bool:
        accounts.release_quota(reservation_id)
        message = message or "No videos generated"
        retry_next = False
        refreshed = accounts.update_provider_quota_from_text(
            account["id"],
            "video",
            message,
            units_completed=0,
        )
        provider_remaining = None
        if refreshed:
            provider_remaining = accounts.store.provider_quota_snapshot(refreshed, "video").get("remaining")
        if provider_remaining is not None:
            try:
                retry_next = int(provider_remaining) < max(1, int(quota_units))
            except (TypeError, ValueError):
                retry_next = False
            accounts.record_task_failure(account["id"], message)
            return retry_next
        if _looks_quota_error(message):
            _zero_video_quota(account["id"], "quota_error", message)
            retry_next = True
        else:
            probe_ok = await _probe_after_video_failure(account["id"], client, message)
            if not probe_ok:
                _zero_video_quota(account["id"], "probe_failed_after_video_error", message)
                retry_next = True
                return retry_next
        accounts.record_task_failure(account["id"], message)
        return retry_next

    async def _materialize_video_image_reference_info(
        client: BrowserClient,
        image_value: Any,
        index: int = 0,
    ) -> Optional[Dict[str, Any]]:
        image_value = str(image_value or "").strip()
        if not image_value:
            return None

        filename = f"video-reference-{index + 1}.png"
        image_bytes: bytes
        if image_value.startswith("data:"):
            try:
                header, encoded = image_value.split(",", 1)
                mime_type = header[5:].split(";", 1)[0]
                ext = mimetypes.guess_extension(mime_type) or ".png"
                filename = f"video-reference-{index + 1}{ext}"
                image_bytes = base64.b64decode(encoded)
            except (ValueError, TypeError, binascii.Error) as exc:
                raise HTTPException(status_code=400, detail="Invalid image data URI") from exc
        elif image_value.startswith("http://") or image_value.startswith("https://"):
            parsed_name = image_value.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
            if parsed_name:
                filename = parsed_name
            async with httpx.AsyncClient(timeout=120) as image_client:
                resp = await image_client.get(image_value)
                resp.raise_for_status()
                image_bytes = resp.content
        else:
            # Treat opaque values as an already uploaded upstream image/file key.
            return {
                "uri": image_value,
                "name": filename,
                "format": filename.rsplit(".", 1)[-1] if "." in filename else "png",
                "width": 64,
                "height": 64,
            }

        uploaded = await client.upload_video_reference_image(
            image_bytes=image_bytes,
            filename=filename,
        )
        uri = uploaded.get("uri") or uploaded.get("cdn_url")
        if not uri:
            return None
        uploaded["uri"] = uri
        uploaded.setdefault("name", filename)
        uploaded.setdefault("format", filename.rsplit(".", 1)[-1] if "." in filename else "png")
        uploaded.setdefault("width", 64)
        uploaded.setdefault("height", 64)
        return uploaded

    async def _materialize_video_image_reference(
        client: BrowserClient,
        image_value: Any,
        index: int = 0,
    ) -> Optional[str]:
        info = await _materialize_video_image_reference_info(client, image_value, index)
        if not info:
            return None
        return str(info.get("uri") or info.get("cdn_url") or "")

    async def _materialize_video_reference_image_infos(
        client: BrowserClient,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        refs = list(params.get("reference_images") or [])
        if not refs and params.get("ref_image_key"):
            refs = [params["ref_image_key"]]

        infos: List[Dict[str, Any]] = []
        seen = set()
        for index, ref in enumerate(refs):
            info = await _materialize_video_image_reference_info(client, ref, index)
            if not info:
                continue
            uri = str(info.get("uri") or info.get("cdn_url") or "")
            if not uri or uri in seen:
                continue
            seen.add(uri)
            info["uri"] = uri
            infos.append(info)
        return infos

    async def _materialize_video_reference_image_keys(
        client: BrowserClient,
        params: Dict[str, Any],
    ) -> List[str]:
        return [str(info["uri"]) for info in await _materialize_video_reference_image_infos(client, params)]

    async def _execute_video_generation_once(params: Dict[str, Any], request: Optional[Request] = None) -> Dict[str, Any]:
        await bucket.acquire()
        quota_units = int(params.get("quota_units") or _video_quota_units(params))
        account, client = await _get_account_client(
            request,
            params,
            account_id=params.get("account_id"),
            quota_kind=None if params.get("quota_reservation_id") else "video",
            quota_units=quota_units,
        )
        params["account_id"] = account["id"]
        reservation_id = str(params.get("quota_reservation_id") or "")
        if not reservation_id:
            reservation_id = accounts.reserve_quota(
                account["id"],
                "video",
                quota_units,
                request_id=f"video-sync-{uuid.uuid4().hex}",
                meta={
                    "model": params.get("model"),
                    "provider_model": params.get("provider_model"),
                    "duration": params.get("duration"),
                    "ratio": params.get("ratio"),
                    "reference_image_count": len(params.get("reference_images") or []),
                },
            )
            params["quota_reservation_id"] = reservation_id
            params["quota_units"] = quota_units
        try:
            async with _video_account_lock(account["id"]):
                reference_image_infos = await _materialize_video_reference_image_infos(client, params)
                reference_image_keys = [str(info["uri"]) for info in reference_image_infos if info.get("uri")]
                ref_image_key = reference_image_keys[0] if reference_image_keys else None
                params["ref_image_key"] = ref_image_key
                params["reference_image_keys"] = reference_image_keys
                params["reference_image_infos"] = reference_image_infos
                if params.get("task_id"):
                    video_tasks.update(
                        str(params["task_id"]),
                        "in_progress",
                        ref_image_key=ref_image_key,
                        reference_image_keys=json.dumps(reference_image_keys, ensure_ascii=False),
                    )
                result = await client.generate_video_web(
                    prompt=params["prompt"],
                    ratio=params.get("ratio"),
                    ref_image_key=ref_image_key,
                    reference_image_keys=reference_image_keys,
                    reference_image_infos=reference_image_infos,
                    model=params.get("provider_model"),
                    duration=params.get("duration"),
                    wait_for_result=False,
                )
                if not result.get("videos") and not _accepted_video_result(result):
                    log.info("video web ability did not return a bindable pending task; trying Samantha fallback")
                    result = await client.submit_video(
                        prompt=params["prompt"],
                        ratio=params.get("ratio"),
                        ref_image_key=ref_image_key,
                        reference_image_keys=reference_image_keys,
                        reference_image_infos=reference_image_infos,
                        model=params.get("provider_model"),
                        duration=params.get("duration"),
                    )
        except HTTPException:
            accounts.release_quota(reservation_id)
            raise
        except RuntimeError as exc:
            retry_next = await _handle_video_attempt_failure(account, client, reservation_id, str(exc), quota_units)
            raise _VideoAttemptFailed(account["id"], str(exc), retry_next)
        except Exception as exc:
            accounts.release_quota(reservation_id)
            raise

        videos = result.get("videos", [])
        msg = result.get("message", "")
        if _looks_quota_error(msg):
            retry_next = await _handle_video_attempt_failure(account, client, reservation_id, msg, quota_units)
            raise _VideoAttemptFailed(account["id"], msg, retry_next)
        if not videos:
            if _accepted_video_result(result):
                accounts.update_provider_quota_from_text(
                    account["id"],
                    "video",
                    msg,
                    units_completed=0,
                )
                accounts.mark_success(account["id"])
                refreshed_account = accounts.store.get(account["id"]) or account
                response = {
                    "created": int(time.time()),
                    "data": [],
                    "pending": True,
                    "accepted": True,
                    "status": "in_progress",
                    "account_id": account["id"],
                    "quota": accounts.store.quota_snapshot(refreshed_account, "video"),
                }
                for key in ("provider_task_id", "conversation_id", "local_conversation_id", "conversation_url"):
                    if result.get(key):
                        response[key] = result.get(key)
                for key in (
                    "provider_task_candidates",
                    "message_id_candidates",
                    "local_message_id",
                    "user_message_id",
                    "bot_message_id",
                    "reply_id",
                ):
                    if result.get(key):
                        response[key] = result.get(key)
                if params.get("reference_image_keys"):
                    response["reference_image_keys"] = params["reference_image_keys"]
                if msg:
                    response["message"] = msg
                if params.get("task_id"):
                    video_tasks.update(
                        str(params["task_id"]),
                        "in_progress",
                        result_json=json.dumps(response, ensure_ascii=False),
                        message=msg,
                        account_id=account["id"],
                        quota_reservation_id=reservation_id,
                        quota_units=quota_units,
                        **_video_task_update_fields(response, accepted=True),
                    )
                return response
            message = msg or "Video generation submit did not return task_id or conversation_id"
            if is_video_acceptance_message(message):
                message = (
                    "Provider accepted the video request but did not expose a "
                    f"retrievable task_id or conversation_id. Last provider message: {message}"
                )
            retry_next = await _handle_video_attempt_failure(account, client, reservation_id, message, quota_units)
            raise _VideoAttemptFailed(account["id"], message, retry_next)
        accounts.complete_quota(reservation_id)
        if msg:
            accounts.update_provider_quota_from_text(
                account["id"],
                "video",
                msg,
                units_completed=quota_units,
            )
        accounts.mark_success(account["id"])
        refreshed_account = accounts.store.get(account["id"]) or account

        response = {
            "created": int(time.time()),
            "data": videos,
            "account_id": account["id"],
            "quota": accounts.store.quota_snapshot(refreshed_account, "video"),
        }
        for key in ("provider_task_id", "conversation_id", "local_conversation_id", "conversation_url"):
            if result.get(key):
                response[key] = result.get(key)
        for key in (
            "provider_task_candidates",
            "message_id_candidates",
            "local_message_id",
            "user_message_id",
            "bot_message_id",
            "reply_id",
        ):
            if result.get(key):
                response[key] = result.get(key)
        if params.get("reference_image_keys"):
            response["reference_image_keys"] = params["reference_image_keys"]
        if msg:
            response["message"] = msg
        return response

    async def _execute_video_generation(params: Dict[str, Any], request: Optional[Request] = None) -> Dict[str, Any]:
        explicit_account = bool(params.get("explicit_account_id")) or _request_has_account_selector(request, None)
        attempt_errors: list[str] = []
        attempts = 0
        while True:
            attempts += 1
            attempt_params = dict(params)
            if attempts > 1:
                attempt_params.pop("account_id", None)
                attempt_params.pop("quota_reservation_id", None)
            try:
                result = await _execute_video_generation_once(attempt_params, request)
                if attempt_errors:
                    result["account_retry"] = {
                        "attempts": attempts,
                        "failed_accounts": attempt_errors,
                    }
                return result
            except HTTPException as exc:
                if attempt_errors:
                    raise RuntimeError(
                        "; ".join(attempt_errors) + f"; next account unavailable: {exc.detail}"
                    ) from exc
                raise
            except _VideoAttemptFailed as exc:
                attempt_errors.append(f"{exc.account_id}: {str(exc)[:220]}")
                if explicit_account or not exc.retry_next_account:
                    raise RuntimeError(str(exc))
                params.pop("account_id", None)
                params.pop("quota_reservation_id", None)
                quota_units = int(params.get("quota_units") or _video_quota_units(params))
                has_next_account = any(
                    account.get("enabled") and accounts.store.has_quota(account, "video", quota_units)
                    for account in accounts.store.list_accounts()
                )
                if not has_next_account:
                    raise RuntimeError("; ".join(attempt_errors))
                log.warning(
                    "video generation failed on account %s; trying next available account",
                    exc.account_id,
                )

    def _is_openai_video_request(request: Request) -> bool:
        path = request.url.path
        return path == "/v1/videos" or (
            path.startswith("/v1/videos/")
            and not path.startswith("/v1/videos/generations")
        )

    def _openai_video_progress(status: str) -> int:
        if status in {"completed", "failed", "cancelled"}:
            return 100
        if status == "in_progress":
            return 30
        return 0

    def _format_video_task(task: Dict[str, Any], *, openai: bool = False) -> Dict[str, Any]:
        if openai:
            status_map = {
                "queued": "queued",
                "in_progress": "in_progress",
                "running": "in_progress",
                "completed": "completed",
                "failed": "failed",
                "cancelled": "cancelled",
                "canceled": "cancelled",
            }
        else:
            status_map = {
                "queued": "queued",
                "in_progress": "running",
                "running": "running",
                "completed": "completed",
                "failed": "failed",
                "cancelled": "cancelled",
                "canceled": "cancelled",
            }
        status = status_map.get(str(task["status"]), str(task["status"]))
        response: Dict[str, Any] = {
            "id": task["task_id"],
            "task_id": task["task_id"],
            "object": "video" if openai else "video.generation.task",
            "created": task["created"],
            "updated": task["updated"],
            "status": status,
            "model": task.get("model") or "doubao-video",
            "provider": "DOUBAO-WEB-01",
            "prompt": task["prompt"],
            "poll_url": f"/v1/videos/{task['task_id']}" if openai else f"/v1/video/generations/{task['task_id']}",
        }
        if openai:
            response["created_at"] = task["created"]
            response["completed_at"] = task["updated"]
            response["progress"] = _openai_video_progress(status)
        if task.get("provider_model"):
            response["provider_model"] = task["provider_model"]
        for key in ("provider_task_id", "conversation_id", "local_conversation_id", "conversation_url"):
            if task.get(key):
                response[key] = task[key]
        if task.get("account_id"):
            response["account_id"] = task["account_id"]
        if task.get("ratio"):
            response["ratio"] = task["ratio"]
        if task.get("duration"):
            response["duration"] = task["duration"]
            if openai:
                response["seconds"] = str(task["duration"])
        if openai and task.get("ratio"):
            response["size"] = str(task["ratio"])
        if task.get("ref_image_key"):
            response["ref_image_key"] = task["ref_image_key"]
        if task.get("reference_image_keys"):
            try:
                reference_image_keys = json.loads(task["reference_image_keys"])
            except (TypeError, json.JSONDecodeError):
                reference_image_keys = []
            if reference_image_keys:
                response["reference_image_keys"] = reference_image_keys
        if task.get("message") and not (status == "completed" and task.get("result_json")):
            response["message"] = task["message"]
        if task.get("error"):
            err_code = "quota_exhausted" if _looks_quota_error(str(task["error"])) else "video_generation_failed"
            response["error"] = {
                "message": task["error"],
                "type": "provider_quota_exhausted" if err_code == "quota_exhausted" else "api_error",
                "code": err_code,
            }
        if task.get("result_json") and status not in {"failed", "cancelled"}:
            try:
                result = json.loads(task["result_json"])
            except json.JSONDecodeError:
                result = {}
            data = result.get("data") if isinstance(result, dict) else None
            if data is not None:
                response["data"] = data
                response["output"] = data
                response["result"] = {"data": data}
                if result.get("pending"):
                    response["pending"] = True
                    response["accepted"] = bool(result.get("accepted", True))
                for key in ("provider_task_id", "conversation_id", "local_conversation_id", "conversation_url"):
                    if result.get(key):
                        response[key] = result.get(key)
                for key in (
                    "provider_task_candidates",
                    "message_id_candidates",
                    "local_message_id",
                    "user_message_id",
                    "bot_message_id",
                    "reply_id",
                ):
                    if result.get(key):
                        response[key] = result.get(key)
                if result.get("account_id"):
                    response["account_id"] = result.get("account_id")
                if result.get("account_retry"):
                    response["account_retry"] = result.get("account_retry")
                if data and isinstance(data, list) and isinstance(data[0], dict):
                    first_url = data[0].get("video_url") or data[0].get("url")
                    if first_url:
                        response["url"] = first_url
                        if openai:
                            response["metadata"] = {"url": first_url}
        return response

    def _format_doubao_official_task(task: Dict[str, Any]) -> Dict[str, Any]:
        status_map = {
            "queued": "pending",
            "in_progress": "processing",
            "running": "processing",
            "completed": "succeeded",
            "failed": "failed",
            "cancelled": "failed",
            "canceled": "failed",
        }
        status = status_map.get(str(task["status"]), "processing")
        response: Dict[str, Any] = {
            "id": task["task_id"],
            "model": task.get("model") or "doubao-video",
            "status": status,
            "content": {"video_url": ""},
            "created_at": task["created"],
            "updated_at": task["updated"],
        }
        if task.get("ratio"):
            response["ratio"] = task["ratio"]
        if task.get("duration"):
            response["duration"] = int(task["duration"])
        if task.get("error"):
            response["error"] = {
                "code": "video_generation_failed",
                "message": task["error"],
            }
        if task.get("result_json") and status != "failed":
            try:
                result = json.loads(task["result_json"])
            except json.JSONDecodeError:
                result = {}
            if isinstance(result, dict):
                data = result.get("data")
                if data and isinstance(data, list) and isinstance(data[0], dict):
                    response["content"]["video_url"] = data[0].get("video_url") or data[0].get("url") or ""
                if result.get("message") and status != "succeeded":
                    response["error"] = {
                        "code": "video_generation_failed",
                        "message": result.get("message"),
                    }
        return response

    async def _run_video_task(task_id: str, params: Dict[str, Any]) -> None:
        existing = video_tasks.get(task_id)
        if existing and existing.get("status") == "cancelled":
            reservation_id = params.get("quota_reservation_id")
            if reservation_id:
                accounts.release_quota(str(reservation_id))
            return
        video_tasks.update(task_id, "in_progress")
        try:
            result = await _execute_video_generation(params)
        except HTTPException as exc:
            message = str(exc.detail)
            log.warning("video task %s failed: %s", task_id, message)
            video_tasks.update(task_id, "failed", error=message, message=message)
            return
        except Exception as exc:
            message = str(exc)
            log.warning("video task %s failed: %s", task_id, message)
            video_tasks.update(task_id, "failed", error=message, message=message)
            return
        message = result.get("message", "")
        current = video_tasks.get(task_id)
        if current and current.get("status") == "cancelled":
            return
        if result.get("pending"):
            video_tasks.update(
                task_id,
                "in_progress",
                result_json=json.dumps(result, ensure_ascii=False),
                message=message,
                account_id=result.get("account_id"),
                **_video_task_update_fields(result, accepted=True),
            )
            return
        video_tasks.update(
            task_id,
            "completed",
            result_json=json.dumps(result, ensure_ascii=False),
            error=None,
            message=message,
            account_id=result.get("account_id"),
            **_video_task_update_fields(result),
        )

    async def _recover_video_task_if_possible(
        task_id: str,
        task: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        task = task or video_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Video task not found")
        if task.get("status") in {"completed", "cancelled", "failed"}:
            return task
        if _looks_quota_error(str(task.get("message") or task.get("error") or "")):
            return task
        provider_task_id, conversation_id, _local_conversation_id, conversation_url = _video_task_binding_ids(task)
        if (
            task.get("status") == "in_progress"
            and not provider_task_id
            and not conversation_id
            and not conversation_url
            and (task.get("result_json") or task.get("message") or task.get("accepted_at"))
        ):
            message = "Video generation submit did not return task_id or conversation_id"
            accounts.release_quota(str(task.get("quota_reservation_id") or ""))
            video_tasks.update(task_id, "failed", error=message, message=message)
            if task.get("account_id"):
                accounts.record_task_failure(str(task["account_id"]), message)
            return video_tasks.get(task_id) or task
        if not VideoTaskStore._is_accepted_pending_result(
            task.get("result_json"),
            task.get("message"),
            provider_task_id=provider_task_id,
            conversation_id=conversation_id,
        ):
            return task

        async with _video_lock(task_id):
            task = video_tasks.get(task_id) or task
            if task.get("status") in {"completed", "cancelled", "failed"}:
                return task
            if _looks_quota_error(str(task.get("message") or task.get("error") or "")):
                return task
            provider_task_id, conversation_id, _local_conversation_id, conversation_url = _video_task_binding_ids(task)
            if (
                task.get("status") == "in_progress"
                and not provider_task_id
                and not conversation_id
                and not conversation_url
                and (task.get("result_json") or task.get("message") or task.get("accepted_at"))
            ):
                message = "Video generation submit did not return task_id or conversation_id"
                accounts.release_quota(str(task.get("quota_reservation_id") or ""))
                video_tasks.update(task_id, "failed", error=message, message=message)
                if task.get("account_id"):
                    accounts.record_task_failure(str(task["account_id"]), message)
                return video_tasks.get(task_id) or task
            if not VideoTaskStore._is_accepted_pending_result(
                task.get("result_json"),
                task.get("message"),
                provider_task_id=provider_task_id,
                conversation_id=conversation_id,
            ):
                return task

            now = int(time.time())
            max_age = int(os.environ.get("DOUBAO_VIDEO_MAX_WAIT_SECONDS", os.environ.get("DOUBAO_VIDEO_ACCEPTED_MAX_AGE_SECONDS", "600")))
            if max_age > 0 and task.get("status") != "failed" and now - int(task.get("created") or now) > max_age:
                message = f"Accepted video was not recoverable within {max_age} seconds"
                accounts.release_quota(str(task.get("quota_reservation_id") or ""))
                video_tasks.update(task_id, "failed", error=message, message=message)
                if task.get("account_id"):
                    accounts.record_task_failure(str(task["account_id"]), message)
                return video_tasks.get(task_id) or task
            min_interval = int(os.environ.get("DOUBAO_VIDEO_POLL_INTERVAL_SECONDS", os.environ.get("DOUBAO_VIDEO_GET_RECOVERY_MIN_INTERVAL_SECONDS", "15")))
            last_recovery_at = int(task.get("last_recovery_at") or 0)
            if not force and last_recovery_at and now - last_recovery_at < max(1, min_interval):
                return task

            video_tasks.mark_recovery_attempt(task_id)
            account_id = str(task.get("account_id") or accounts.default_account_id)
            try:
                account, client = await accounts.get_ready_client(account_id)
            except Exception as exc:
                log.warning("video recovery skipped for %s: account %s not ready: %s", task_id, account_id, exc)
                return video_tasks.get(task_id) or task

            recover_timeout = float(timeout if timeout is not None else os.environ.get("DOUBAO_VIDEO_RECOVERY_POLL_SECONDS", "20"))
            try:
                async with _video_account_lock(account["id"]):
                    provider_task_candidates = _video_task_provider_task_candidates(task)
                    if conversation_id or conversation_url:
                        result = await client.recover_video_result(
                            str(task.get("prompt") or ""),
                            conversation_id=conversation_id or None,
                            conversation_url=conversation_url or None,
                            timeout=recover_timeout,
                        )
                    elif provider_task_candidates:
                        result = {"videos": [], "prompt": str(task.get("prompt") or "")}
                        for candidate in provider_task_candidates:
                            try:
                                result = await client.poll_video_result(
                                    candidate,
                                    str(task.get("prompt") or ""),
                                    timeout=recover_timeout,
                                )
                            except Exception as exc:
                                log.warning(
                                    "video recovery task %s candidate %s failed: %s",
                                    task_id,
                                    candidate,
                                    exc,
                                )
                                continue
                            result_msg = str(result.get("message") or "")
                            if result.get("videos") or _looks_quota_error(result_msg):
                                result.setdefault("provider_task_id", candidate)
                                break
                    else:
                        result = {"videos": [], "prompt": str(task.get("prompt") or "")}
            except Exception as exc:
                log.warning("video recovery failed for %s: %s", task_id, exc)
                return video_tasks.get(task_id) or task

            videos = result.get("videos") or []
            result_msg = str(result.get("message") or "")
            quota_units = int(task.get("quota_units") or 1)
            if _looks_quota_error(result_msg):
                await _handle_video_attempt_failure(
                    account,
                    client,
                    str(task.get("quota_reservation_id") or ""),
                    result_msg,
                    quota_units,
                )
                video_tasks.update(task_id, "failed", error=result_msg, message=result_msg)
                return video_tasks.get(task_id) or task
            if not videos:
                return video_tasks.get(task_id) or task

            msg = str(task.get("message") or result.get("message") or "completed")
            accounts.complete_quota(str(task.get("quota_reservation_id") or ""))
            accounts.update_provider_quota_from_text(
                account["id"],
                "video",
                msg,
                units_completed=quota_units,
            )
            accounts.mark_success(account["id"])
            refreshed_account = accounts.store.get(account["id"]) or account
            response = {
                "created": int(time.time()),
                "data": videos,
                "account_id": account["id"],
                "quota": accounts.store.quota_snapshot(refreshed_account, "video"),
                "message": msg,
                "recovered": True,
            }
            for key in ("provider_task_id", "conversation_id", "local_conversation_id", "conversation_url"):
                value = result.get(key) or task.get(key)
                if value:
                    response[key] = value
            video_tasks.update(
                task_id,
                "completed",
                result_json=json.dumps(response, ensure_ascii=False),
                error=None,
                message=msg,
                account_id=account["id"],
                **_video_task_update_fields(response),
            )
            return video_tasks.get(task_id) or task

    async def _video_recovery_worker() -> None:
        interval = int(os.environ.get("DOUBAO_VIDEO_POLL_INTERVAL_SECONDS", os.environ.get("DOUBAO_VIDEO_RECOVERY_INTERVAL_SECONDS", "15")))
        limit = int(os.environ.get("DOUBAO_VIDEO_RECOVERY_BATCH_SIZE", "10"))
        while True:
            await asyncio.sleep(max(1, interval))
            try:
                for task in video_tasks.recovery_candidates(
                    min_interval_seconds=max(1, interval),
                    limit=limit,
                ):
                    try:
                        await _recover_video_task_if_possible(
                            str(task["task_id"]),
                            task,
                            timeout=float(os.environ.get("DOUBAO_VIDEO_SINGLE_POLL_TIMEOUT_SECONDS", os.environ.get("DOUBAO_VIDEO_BACKGROUND_RECOVERY_SECONDS", "180"))),
                        )
                    except Exception as exc:
                        log.warning("video recovery worker task %s failed: %s", task.get("task_id"), exc)
            except Exception as exc:
                log.warning("video recovery worker loop failed: %s", exc)

    def _get_qianwen_client() -> QianwenClient:
        client = _qianwen.get("client")
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="Qianwen client not available. Set QIANWEN_ENABLED=true.",
            )
        if not client.is_ready:
            raise HTTPException(status_code=503, detail="Qianwen client not ready")
        return client

    # ── Prompt extraction ──

    def _extract_prompt(messages: List[_Message]) -> str:
        """Extract text prompt from OpenAI-format messages."""
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg.content, str):
                if len(messages) == 1:
                    parts.append(msg.content)
                else:
                    parts.append(f"[{msg.role}]: {msg.content}")
            elif isinstance(msg.content, list):
                for p in msg.content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        text = p.get("text", "")
                        if text:
                            if len(messages) == 1:
                                parts.append(text)
                            else:
                                parts.append(f"[{msg.role}]: {text}")
        return "\n".join(parts)

    def _extract_prompt_and_file_refs(messages: List[_Message]) -> tuple[str, list[dict[str, Any]]]:
        """Extract text prompt and OpenAI-style file_url references."""
        parts: list[str] = []
        file_refs: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg.content, str):
                if len(messages) == 1:
                    parts.append(msg.content)
                else:
                    parts.append(f"[{msg.role}]: {msg.content}")
                continue
            if not isinstance(msg.content, list):
                continue
            for part in msg.content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text = part.get("text", "")
                    if text:
                        if len(messages) == 1:
                            parts.append(text)
                        else:
                            parts.append(f"[{msg.role}]: {text}")
                elif part.get("type") == "file_url":
                    file_url = part.get("file_url", {})
                    if isinstance(file_url, str):
                        file_refs.append({"url": file_url})
                    elif isinstance(file_url, dict):
                        file_refs.append(file_url)
        return "\n".join(parts), file_refs

    async def _materialize_file_refs(client: BrowserClient, file_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resolve TOS/data/http file_url references to uploaded file metadata."""
        import base64
        import mimetypes
        from urllib.parse import urlparse
        files: list[dict[str, Any]] = []
        for file_ref in file_refs:
            url = str(file_ref.get("url", "")).strip()
            if not url:
                raise HTTPException(status_code=400, detail="file_url.url is required")
            name = file_ref.get("name") or "file"
            size = int(file_ref.get("size") or 0)
            if url.startswith("tos-"):
                files.append({"uri": url, "name": name, "size": size})
                continue
            if url.startswith("data:"):
                try:
                    header, encoded = url.split(",", 1)
                    file_data = base64.b64decode(encoded)
                except (ValueError, TypeError) as exc:
                    raise HTTPException(status_code=400, detail="Invalid data URI") from exc
                if name == "file":
                    mime_type = header[5:].split(";", 1)[0]
                    ext = mimetypes.guess_extension(mime_type) or ".txt"
                    name = f"upload{ext}"
                uploaded = await client.upload_file(file_data=file_data, filename=name)
                files.append({"uri": uploaded["uri"], "name": uploaded["name"], "size": uploaded["size"]})
                continue
            if url.startswith("http://") or url.startswith("https://"):
                parsed = urlparse(url)
                inferred_name = parsed.path.rsplit("/", 1)[-1] or "downloaded_file"
                if name == "file":
                    name = inferred_name
                async with httpx.AsyncClient(timeout=120) as http_client:
                    response = await http_client.get(url)
                    response.raise_for_status()
                    file_data = response.content
                uploaded = await client.upload_file(file_data=file_data, filename=name)
                files.append({"uri": uploaded["uri"], "name": uploaded["name"], "size": uploaded["size"]})
                continue
            raise HTTPException(status_code=400, detail=f"Unsupported file_url: {url[:80]}")
        return files

    # ── Request logging middleware ──

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        path = request.url.path
        if path.startswith("/auth") or path.startswith("/admin"):
            return await call_next(request)
        start = time.time()
        response = await call_next(request)
        elapsed = round((time.time() - start) * 1000)
        _REQUEST_LOG.append({
            "ts": time.time(),
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "ms": elapsed,
        })
        return response

    # ── Endpoints ──

    @app.get("/health")
    async def health():
        account_rows = accounts.list_accounts()
        ready = any(a.get("runtime", {}).get("ready") for a in account_rows)
        result = {
            "status": "ok" if ready else "not_ready",
            "logged_in": ready,
            "accounts": {
                "total": len(account_rows),
                "ready": sum(1 for a in account_rows if a.get("runtime", {}).get("ready")),
                "hot": sum(1 for a in account_rows if a.get("runtime", {}).get("hot")),
                "counts": accounts.counts(),
            },
        }
        result["expert_degraded"] = _expert_tracker.is_degraded
        result["video_tasks"] = video_tasks.counts()
        # Qianwen status
        qw = _qianwen.get("client")
        result["qianwen_ready"] = qw.is_ready if qw else False
        return result

    @app.get("/v1/models")
    async def list_models(request: Request):
        _check_auth(request)
        return {"object": "list", "data": ALL_MODELS}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        _check_auth(request)

        # ── Route to Qianwen if model matches ──
        if body.model in QIANWEN_MODEL_NAMES:
            return await _handle_qianwen_chat(body, request)

        # ── Tool calling mode ──
        has_tools = bool(body.tools)
        if has_tools:
            # Use expert model for tool calling, with auto-fallback to think if degraded
            requested_deep_think = CHAT_MODELS["doubao-expert"]
            use_deep_think, model_name = _expert_tracker.get_effective_mode(requested_deep_think)
            # Convert messages with tool definitions injected
            messages_raw = [m.model_dump(exclude_none=True) for m in body.messages]
            prompt = convert_messages_with_tools(messages_raw, body.tools)
        else:
            use_deep_think = CHAT_MODELS.get(body.model)
            if use_deep_think is None:
                all_models = list(CHAT_MODELS.keys()) + list(QIANWEN_MODEL_NAMES)
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown model '{body.model}'. Available: {', '.join(all_models)}",
                )
            model_name = body.model
            prompt, file_refs = _extract_prompt_and_file_refs(body.messages)
            if not prompt:
                raise HTTPException(status_code=400, detail="No text content")

        await bucket.acquire()
        account, client = await _get_account_client(
            request,
            {"account_id": body.account_id} if body.account_id else None,
        )
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if body.stream:
            if not has_tools:
                _, file_refs_check = _extract_prompt_and_file_refs(body.messages)
                if file_refs_check:
                    raise HTTPException(
                        status_code=400,
                        detail="file_url attachments are currently supported for non-streaming requests only",
                    )
            return StreamingResponse(
                _stream_chat(client, prompt, use_deep_think, request_id, model_name,
                             conversation_id=body.conversation_id, bot_id=body.bot_id,
                             has_tools=has_tools,
                             messages_for_counting=body.messages,
                             account_id=account["id"]),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming: collect all chunks with thinking state machine
        try:
            if has_tools:
                # Tool calling non-streaming path
                message = await _collect_chat_response(
                    client, prompt, use_deep_think,
                    conversation_id=body.conversation_id, bot_id=body.bot_id,
                )
                # Report to expert tracker (detect silent downgrade)
                had_reasoning = bool(message.get("reasoning_content"))
                if use_deep_think >= 1:
                    _expert_tracker.report_response(had_reasoning)
                # Check if response contains tool calls
                content = message.get("content", "")
                parsed_tools = parse_tool_calls_xml(content) if content else None
                if parsed_tools:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": parsed_tools,
                    }
                    finish_reason = "tool_calls"
                else:
                    finish_reason = "stop"
            elif file_refs:
                files = await _materialize_file_refs(client, file_refs)
                result = await client.chat_with_file(
                    text=prompt,
                    file_uri=files,
                    file_name=files[0]["name"],
                    file_size=files[0]["size"],
                    use_deep_think=use_deep_think,
                )
                message = {"role": "assistant", "content": result["text"]}
                finish_reason = "stop"
            else:
                message = await _collect_chat_response(
                    client, prompt, use_deep_think,
                    conversation_id=body.conversation_id, bot_id=body.bot_id,
                )
                finish_reason = "stop"
        except RuntimeError as exc:
            _record_provider_failure(account["id"], str(exc))
            raise HTTPException(status_code=502, detail=str(exc))

        # max_tokens truncation (non-streaming only)
        content = message.get("content") or ""
        if body.max_tokens and content and not message.get("tool_calls"):
            max_chars = int(body.max_tokens * 2.5)  # rough tokens->chars
            if len(content) > max_chars:
                message["content"] = content[:max_chars]
                finish_reason = "length"

        # Token counting
        prompt_tokens = count_messages_tokens(
            [m.model_dump(exclude_none=True) for m in body.messages]
        )
        completion_content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content") or ""
        completion_tokens = count_tokens(completion_content + reasoning_content)

        resp_data = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        if message.get("conversation_id"):
            resp_data["conversation_id"] = message["conversation_id"]
        accounts.mark_success(account["id"])
        return JSONResponse(resp_data)

    # ------------------------------------------------------------------
    # Qianwen chat handler
    # ------------------------------------------------------------------

    async def _handle_qianwen_chat(body: ChatCompletionRequest, request: Request):
        """Handle chat completions routed to Qianwen."""
        qw_client = _get_qianwen_client()
        model_config = QIANWEN_MODELS.get(body.model, {"model": "Qwen", "deep_search": "0"})
        qw_model = model_config["model"]
        deep_search = model_config["deep_search"]
        # Support enable_thinking parameter (like official API)
        if body.enable_thinking or (body.reasoning_effort and body.reasoning_effort != "none"):
            deep_search = "1"
        # NOTE: tools + thinking mode IS supported — tool output appears in think_content
        # Do NOT force deep_search="0" when tools are present
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        messages_raw = [m.model_dump(exclude_none=True) for m in body.messages]

        # ── Topic isolation: discard irrelevant history on topic change ──
        messages_raw = filter_history_by_topic(messages_raw)

        # ── Tool calling: inject tool definitions into prompt ──
        has_tools = bool(body.tools)
        if has_tools:
            # Log input breakdown for debugging
            num_tools = len(body.tools) if body.tools else 0
            sys_msg = next((m for m in messages_raw if m.get("role") == "system"), None)
            sys_len = len(sys_msg.get("content", "")) if sys_msg else 0
            tool_msgs = [m for m in messages_raw if m.get("role") == "tool"]
            log.info("Qianwen input: %d msgs, %d tools, sys=%d chars, %d tool_results",
                     len(messages_raw), num_tools, sys_len, len(tool_msgs))

            # Obfuscate tool names if enabled (avoids Qwen built-in validation)
            tools_for_prompt = _tool_obfuscator.obfuscate_tools(body.tools)
            prompt = convert_messages_with_tools(messages_raw, tools_for_prompt)
            log.info("Qianwen prompt after flatten: %d chars (%dKB)",
                     len(prompt), len(prompt) // 1024)
            if len(prompt) > 50000:
                log.warning("Qianwen prompt OVER LIMIT: %d chars — truncation active", len(prompt))
            # Wrap as single user message for Qianwen
            messages_for_qw = [{"role": "user", "content": prompt}]
        else:
            messages_for_qw = messages_raw

        if body.stream:
            return StreamingResponse(
                _stream_qianwen_chat(
                    qw_client, messages_for_qw, qw_model, deep_search,
                    request_id, body.model, has_tools=has_tools,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            # Non-streaming
            try:
                result = await qw_client.chat(messages_for_qw, qw_model, deep_search)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

            import re as _re
            _think_prefix_re = _re.compile(r"^\[?\(multimodal_chat_think_\d+\)\]?\s*")

            content = result["content"]
            think_content = result.get("think_content", "")
            usage = result.get("usage", {})

            # Strip thinking prefix from content
            content = _think_prefix_re.sub("", content).strip()

            # Check for tool calls in think_content first, then content
            if has_tools:
                source = think_content if think_content else content
                parsed = parse_tool_calls_xml(source)
                if not parsed and content:
                    parsed = parse_tool_calls_xml(content)
                
                # Auto-continue if tool call was truncated
                if not parsed and detect_truncated_tool_call(source or content):
                    log.info("Detected truncated tool_call, attempting continuation...")
                    cont_prompt = build_continuation_prompt(source or content)
                    cont_messages = [{"role": "user", "content": cont_prompt}]
                    try:
                        cont_result = await qw_client.chat(cont_messages, qw_model, deep_search)
                        cont_content = cont_result["content"]
                        cont_content = _think_prefix_re.sub("", cont_content).strip()
                        # Deduplicate overlap before combining
                        combined = deduplicate_continuation(source or content, cont_content)
                        parsed = parse_tool_calls_xml(combined)
                        if parsed:
                            log.info("Continuation successful, got %d tool calls", len(parsed))
                    except Exception as e:
                        log.warning("Continuation failed: %s", e)
                
                if parsed:
                    # Deobfuscate tool names back to original
                    parsed = _tool_obfuscator.deobfuscate_tool_calls(parsed)
                    # Coerce parameter names to match schema
                    parsed = coerce_tool_arguments(parsed)
                    return JSONResponse({
                        "id": request_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": result.get("model", body.model),
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": parsed,
                            },
                            "finish_reason": "tool_calls",
                        }],
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    })

            # Build message with optional reasoning_content
            message: Dict[str, Any] = {"role": "assistant", "content": content}
            if think_content and deep_search == "1":
                message["reasoning_content"] = think_content

            return JSONResponse({
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result.get("model", body.model),
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            })

    async def _stream_qianwen_chat(
        qw_client: QianwenClient,
        messages: list,
        model: str,
        deep_search: str,
        request_id: str,
        model_name: str,
        *,
        has_tools: bool = False,
    ):
        """Generate OpenAI-compatible SSE stream from Qianwen's cumulative format.

        Handles:
        - Normal content streaming (delta computation from cumulative)
        - Thinking mode: emits reasoning_content deltas from think_content
        - Tool calling: detects <tool_call> in content OR think_content
        """
        import re as _re

        prev_content = ""
        prev_think = ""
        tool_mode = False
        is_thinking = (deep_search == "1")
        # Regex to strip [(multimodal_chat_think_N)] prefix
        _think_prefix_re = _re.compile(r"^\[?\(multimodal_chat_think_\d+\)\]?\s*")

        def _make_chunk(delta: dict, finish_reason=None):
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }],
            }

        # First chunk: role
        chunk = _make_chunk({"role": "assistant", "content": ""})
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        full_content = ""
        full_think = ""

        try:
            async for event in qw_client.chat_stream(messages, model, deep_search):
                if event.get("error"):
                    err_chunk = _make_chunk(
                        {"content": f"[Error: {event.get('message', 'unknown')}]"}
                    )
                    yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                    break

                data = event.get("data", {})
                msgs = data.get("messages", [])
                for msg in msgs:
                    if msg.get("mime_type") != "multi_load/iframe":
                        continue
                    current = msg.get("content", "")
                    if not current:
                        continue

                    # Extract think_content from meta_data if present
                    think_content = ""
                    meta = msg.get("meta_data", {})
                    multi_load = meta.get("multi_load", [])
                    if multi_load and isinstance(multi_load, list):
                        ml_content = multi_load[0].get("content", {})
                        if isinstance(ml_content, dict):
                            think_content = ml_content.get("think_content", "")

                    # Strip thinking prefix from main content
                    clean_content = _think_prefix_re.sub("", current).strip()
                    full_content = clean_content

                    if tool_mode:
                        # Buffering for tool call completion
                        full_think = think_content
                        continue

                    # Check for tool calls in think_content or main content
                    if has_tools:
                        check_text = think_content or clean_content
                        if is_tool_call_start(check_text.strip()):
                            tool_mode = True
                            full_think = think_content
                            continue

                    # Emit reasoning_content delta (thinking mode)
                    if is_thinking and think_content:
                        if len(think_content) > len(prev_think):
                            think_delta = think_content[len(prev_think):]
                            prev_think = think_content
                            full_think = think_content
                            delta_chunk = _make_chunk({
                                "reasoning_content": think_delta
                            })
                            yield f"data: {json.dumps(delta_chunk, ensure_ascii=False)}\n\n"

                    # Emit content delta
                    if len(clean_content) > len(prev_content):
                        delta_text = clean_content[len(prev_content):]
                        prev_content = clean_content
                        delta_chunk = _make_chunk({"content": delta_text})
                        yield f"data: {json.dumps(delta_chunk, ensure_ascii=False)}\n\n"

        except Exception as e:
            log.error("Qianwen stream error: %s", e)
            err_chunk = _make_chunk({"content": f"[Stream error: {e}]"})
            yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"

        # After stream ends: check for tool calls in both content and think_content
        if has_tools and (tool_mode or full_think or full_content):
            # Try think_content first (thinking mode puts tool calls there)
            source = full_think if full_think else full_content
            parsed = parse_tool_calls_xml(source)
            # Also try main content if think didn't have it
            if not parsed and full_content:
                parsed = parse_tool_calls_xml(full_content)
            
            # Auto-continue if truncated
            if not parsed and detect_truncated_tool_call(source or full_content):
                log.info("Stream: detected truncated tool_call, attempting continuation...")
                cont_prompt = build_continuation_prompt(source or full_content)
                try:
                    cont_messages = [{"role": "user", "content": cont_prompt}]
                    cont_result = await qw_client.chat(cont_messages, model, deep_search)
                    cont_content = cont_result.get("content", "")
                    # Deduplicate overlap before combining
                    combined = deduplicate_continuation(source or full_content, cont_content)
                    parsed = parse_tool_calls_xml(combined)
                    if parsed:
                        log.info("Stream continuation got %d tool calls", len(parsed))
                except Exception as e:
                    log.warning("Stream continuation failed: %s", e)
            
            if parsed:
                # Deobfuscate tool names back to original
                parsed = _tool_obfuscator.deobfuscate_tool_calls(parsed)
                # Coerce parameter names to match schema
                parsed = coerce_tool_arguments(parsed)
                for idx, tc in enumerate(parsed):
                    tc_delta = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "index": idx,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }],
                    }
                    yield f"data: {json.dumps(_make_chunk(tc_delta), ensure_ascii=False)}\n\n"
                final_chunk = _make_chunk({}, finish_reason="tool_calls")
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

        # Normal finish
        final_chunk = _make_chunk({}, finish_reason="stop")
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/images/generations")
    async def image_generations(body: ImageGenerationRequest, request: Request):
        _check_auth(request)
        if _is_video_model_name(body.model):
            raise HTTPException(
                status_code=400,
                detail=(
                    f'Model "{body.model}" is a video model. '
                    "Use /v1/video/generations or /v1/videos/generations."
                ),
            )
        await bucket.acquire()
        quota_units = _image_quota_units(body)
        account, client = await _get_account_client(
            request,
            {"account_id": body.account_id} if body.account_id else None,
            quota_kind="image",
            quota_units=quota_units,
        )
        reservation_id = accounts.reserve_quota(
            account["id"],
            "image",
            quota_units,
            request_id=f"image-{uuid.uuid4().hex}",
            meta={"model": body.model, "n": body.n, "size": body.size, "ratio": body.ratio},
        )

        ratio = body.ratio or _size_to_ratio(body.size)

        try:
            result = await client.generate_image(
                prompt=body.prompt,
                ratio=ratio,
                ref_image_key=body.ref_image_key,
            )
        except RuntimeError as exc:
            accounts.release_quota(reservation_id)
            if _looks_quota_error(str(exc)):
                accounts.mark_quota_exhausted(account["id"], "image", str(exc))
            _record_provider_failure(account["id"], str(exc))
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception:
            accounts.release_quota(reservation_id)
            raise

        images = result.get("images", [])
        if not images:
            accounts.release_quota(reservation_id)
            accounts.record_task_failure(account["id"], "No images generated")
            raise HTTPException(
                status_code=502, detail="No images generated"
            )

        data = []
        for img in images:
            data.append({
                "url": img["url"],
                "revised_prompt": body.prompt,
            })

        accounts.complete_quota(reservation_id)
        accounts.mark_success(account["id"])
        refreshed_account = accounts.store.get(account["id"]) or account
        return JSONResponse({
            "created": int(time.time()),
            "account_id": account["id"],
            "quota": accounts.store.quota_snapshot(refreshed_account, "image"),
            "data": data,
        })


    @app.post("/v1/audio/generations")
    async def audio_generations(request: Request):
        _check_auth(request)
        await bucket.acquire()

        body = await request.json()
        account, client = await _get_account_client(request, body)
        prompt = body.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing prompt")

        try:
            result = await client.generate_music(
                prompt=prompt,
                lyric=body.get("lyric"),
                genre=body.get("genre"),
            )
        except RuntimeError as exc:
            _record_provider_failure(account["id"], str(exc))
            raise HTTPException(status_code=502, detail=str(exc))

        tracks = result.get("tracks", [])
        if not tracks:
            accounts.record_task_failure(account["id"], "No music tracks generated")
            raise HTTPException(
                status_code=502, detail="No music tracks generated"
            )

        accounts.mark_success(account["id"])
        return JSONResponse({
            "created": int(time.time()),
            "account_id": account["id"],
            "data": tracks,
        })

    async def _video_generations_sync(body: Dict[str, Any], request: Optional[Request] = None) -> JSONResponse:
        params = _parse_video_request(body)
        try:
            result = await _execute_video_generation(params, request)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return JSONResponse(result)

    @app.post("/v1/video/generations/sync")
    async def video_generations_sync(request: Request):
        _check_auth(request)
        body = await request.json()
        return await _video_generations_sync(body, request)

    @app.post("/v1/videos")
    @app.post("/v1/videos/generations")
    @app.post("/v1/video/generations")
    async def video_generations(request: Request):
        _check_auth(request)
        body = await request.json()

        if _wants_sync_video(body):
            return await _video_generations_sync(body, request)

        params = _parse_video_request(body)
        quota_units = _video_quota_units(params)
        account, _ = await _get_account_client(
            request,
            body,
            quota_kind="video",
            quota_units=quota_units,
        )
        params["account_id"] = account["id"]
        task_id = f"video-{uuid.uuid4().hex}"
        params["task_id"] = task_id
        params["quota_units"] = quota_units
        params["quota_reservation_id"] = accounts.reserve_quota(
            account["id"],
            "video",
            quota_units,
            request_id=task_id,
            meta={
                "model": params.get("model"),
                "provider_model": params.get("provider_model"),
                "duration": params.get("duration"),
                "ratio": params.get("ratio"),
                "async": True,
                "reference_image_count": len(params.get("reference_images") or []),
            },
        )
        try:
            task = video_tasks.create(task_id, params, body)
        except Exception:
            accounts.release_quota(params["quota_reservation_id"])
            raise
        asyncio.create_task(_run_video_task(task_id, params))
        return JSONResponse(_format_video_task(task, openai=_is_openai_video_request(request)))

    @app.get("/v1/videos/{task_id}")
    @app.get("/v1/videos/generations/{task_id}")
    @app.get("/v1/video/generations/{task_id}")
    async def get_video_generation(task_id: str, request: Request):
        _check_auth(request)
        task = video_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Video task not found")
        if task.get("status") in {"queued", "in_progress"}:
            task = await _recover_video_task_if_possible(
                task_id,
                task,
                timeout=float(os.environ.get("DOUBAO_VIDEO_GET_RECOVERY_SECONDS", "4")),
                force=_truthy(request.query_params.get("recover")),
            )
        return JSONResponse(_format_video_task(task, openai=_is_openai_video_request(request)))

    @app.post("/v1/videos/{task_id}/cancel")
    @app.post("/v1/videos/generations/{task_id}/cancel")
    @app.post("/v1/video/generations/{task_id}/cancel")
    async def cancel_video_generation(task_id: str, request: Request):
        _check_auth(request)
        task = video_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Video task not found")
        if task.get("status") not in {"completed", "failed", "cancelled"}:
            video_tasks.update(
                task_id,
                "cancelled",
                error="cancelled",
                message="Task was cancelled locally.",
            )
            task = video_tasks.get(task_id) or task
        return JSONResponse(_format_video_task(task, openai=_is_openai_video_request(request)))

    @app.post("/api/v3/contents/generations/tasks")
    async def doubao_official_video_task_create(request: Request):
        _check_auth(request)
        body = await request.json()
        params = _parse_video_request(body)
        quota_units = _video_quota_units(params)
        account, _ = await _get_account_client(
            request,
            body,
            quota_kind="video",
            quota_units=quota_units,
        )
        params["account_id"] = account["id"]
        task_id = f"video-{uuid.uuid4().hex}"
        params["task_id"] = task_id
        params["quota_units"] = quota_units
        params["quota_reservation_id"] = accounts.reserve_quota(
            account["id"],
            "video",
            quota_units,
            request_id=task_id,
            meta={
                "model": params.get("model"),
                "provider_model": params.get("provider_model"),
                "duration": params.get("duration"),
                "ratio": params.get("ratio"),
                "api": "doubao_official_compat",
                "reference_image_count": len(params.get("reference_images") or []),
            },
        )
        try:
            task = video_tasks.create(task_id, params, body)
        except Exception:
            accounts.release_quota(params["quota_reservation_id"])
            raise
        asyncio.create_task(_run_video_task(task_id, params))
        return JSONResponse({"id": task["task_id"]})

    @app.get("/api/v3/contents/generations/tasks/{task_id}")
    async def doubao_official_video_task_get(task_id: str, request: Request):
        _check_auth(request)
        task = video_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Video task not found")
        if task.get("status") in {"queued", "in_progress"}:
            task = await _recover_video_task_if_possible(
                task_id,
                task,
                timeout=float(os.environ.get("DOUBAO_VIDEO_GET_RECOVERY_SECONDS", "4")),
                force=_truthy(request.query_params.get("recover")),
            )
        return JSONResponse(_format_doubao_official_task(task))

    @app.post("/v1/files")
    async def upload_file(request: Request):
        """Upload a file. Returns file metadata for use in chat."""
        _check_auth(request)
        await bucket.acquire()
        account, client = await _get_account_client(request)

        form = await request.form()
        file_field = form.get("file")
        if not file_field:
            raise HTTPException(status_code=400, detail="Missing file field")

        file_data = await file_field.read()
        filename = file_field.filename or "file.txt"

        try:
            result = await client.upload_file(file_data, filename)
        except RuntimeError as exc:
            _record_provider_failure(account["id"], str(exc))
            raise HTTPException(status_code=502, detail=str(exc))

        accounts.mark_success(account["id"])
        return JSONResponse({
            "id": result["uri"],
            "object": "file",
            "filename": result["name"],
            "bytes": result["size"],
            "uri": result["uri"],
            "file_type": result.get("file_type", ""),
            "purpose": "assistants",
        })


    @app.get("/v1/files/download")
    async def file_download(request: Request, uri: str, expire: int = 3600):
        _check_auth(request)
        await bucket.acquire()
        account, client = await _get_account_client(request)
        try:
            url = await client.get_file_download_url(uri=uri, expire_seconds=expire)
        except RuntimeError as exc:
            _record_provider_failure(account["id"], str(exc))
            raise HTTPException(status_code=502, detail=str(exc))
        accounts.mark_success(account["id"])
        return JSONResponse({"url": url, "uri": uri, "expires_in": expire})

    @app.post("/v1/images/upload")
    async def upload_image(request: Request):
        _check_auth(request)
        await bucket.acquire()
        account, client = await _get_account_client(request)
        form = await request.form()
        upload = form.get("file") or form.get("image")
        if not upload:
            raise HTTPException(status_code=400, detail="Missing file field")
        image_data = await upload.read()
        filename = upload.filename or "image.png"
        try:
            result = await client.upload_image(image_bytes=image_data, filename=filename)
        except RuntimeError as exc:
            _record_provider_failure(account["id"], str(exc))
            raise HTTPException(status_code=502, detail=str(exc))
        accounts.mark_success(account["id"])
        return JSONResponse({
            "uri": result["uri"],
            "cdn_url": result["cdn_url"],
            "url": result["cdn_url"],
            "name": result["name"],
            "format": result["format"],
            "width": result["width"],
            "height": result["height"],
        })

    @app.post("/v1/chat/completions/with-file")
    async def chat_with_file(request: Request):
        """Chat with file attachment. Body: {file_id, prompt, model}."""
        _check_auth(request)
        await bucket.acquire()

        body = await request.json()
        account, client = await _get_account_client(request, body)
        file_id = body.get("file_id", "")
        prompt = body.get("prompt", "")
        file_name = body.get("file_name", "file.txt")
        file_size = body.get("file_size", 0)
        model = body.get("model", "doubao")

        if not file_id or not prompt:
            raise HTTPException(status_code=400, detail="Missing file_id or prompt")

        use_deep_think = CHAT_MODELS.get(model, 0)

        try:
            result = await client.chat_with_file(
                text=prompt,
                file_uri=file_id,
                file_name=file_name,
                file_size=file_size,
                use_deep_think=use_deep_think,
            )
        except RuntimeError as exc:
            _record_provider_failure(account["id"], str(exc))
            raise HTTPException(status_code=502, detail=str(exc))

        accounts.mark_success(account["id"])
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    async def _collect_chat_response(
        client: BrowserClient,
        prompt: str,
        use_deep_think: int,
        *,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
    ) -> dict:
        """Collect full chat response with thinking separation.

        Returns an OpenAI message dict:
        {"role": "assistant", "content": "...", "reasoning_content": "..."}
        reasoning_content is only present when thinking was detected.
        """
        thinking_count = 0
        in_thinking = False
        thinking_parts: list = []
        content_parts: list = []
        result_conversation_id: Optional[str] = None

        def _iter_blocks(data: dict):
            for patch in data.get("patch_op", []):
                pv = patch.get("patch_value", {})
                yield from pv.get("content_block", [])
            dc = data.get("content", {})
            if isinstance(dc, dict):
                yield from dc.get("content_block", [])

        async for event in client.chat_completion(
            prompt, use_deep_think=use_deep_think,
            conversation_id=conversation_id or None,
            bot_id=bot_id or None,
        ):
            if event.get("error"):
                raise RuntimeError(
                    f"API error {event.get('status')}: "
                    f"{event.get('body', '')[:200]}"
                )
            if event.get("error_code"):
                code = event.get("error_code", 0)
                msg = event.get("error_msg", "")
                client.record_failure(code)
                raise RuntimeError(f"Error code={code}: {msg}")

            # Extract conversation_id for multi-turn
            if not result_conversation_id:
                cid = client.extract_conversation_id(event)
                if cid and cid != "0":
                    result_conversation_id = cid

            event_type = event.get("_event", "")

            # CHUNK_DELTA
            if (
                event_type == "CHUNK_DELTA"
                and "text" in event
                and isinstance(event.get("text"), str)
                and event["text"]
            ):
                if in_thinking:
                    thinking_parts.append(event["text"])
                else:
                    content_parts.append(event["text"])
                continue

            # content_block
            has_content_block = False
            for cb in _iter_blocks(event):
                has_content_block = True
                bt = cb.get("block_type", 0)
                block_content = cb.get("content", {})

                if bt == 10040:
                    thinking_count += 1
                    in_thinking = (thinking_count == 1)
                elif bt == 10000:
                    tb = block_content.get("text_block", {})
                    if isinstance(tb, dict) and tb.get("text"):
                        if in_thinking:
                            thinking_parts.append(tb["text"])
                        else:
                            content_parts.append(tb["text"])

            # patch_op content string fallback
            if not has_content_block:
                for patch in event.get("patch_op", []):
                    pv = patch.get("patch_value", {})
                    if isinstance(pv, dict) and "content" in pv:
                        content_str = pv.get("content", "")
                        if isinstance(content_str, str) and content_str:
                            try:
                                obj = json.loads(content_str)
                                t = obj.get("text", "")
                                if t:
                                    if in_thinking:
                                        thinking_parts.append(t)
                                    else:
                                        content_parts.append(t)
                            except (json.JSONDecodeError, TypeError):
                                pass

        message: dict = {"role": "assistant", "content": "".join(content_parts)}
        if thinking_parts:
            message["reasoning_content"] = "".join(thinking_parts)
        if result_conversation_id:
            message["conversation_id"] = result_conversation_id
        client.record_success()
        return message

    async def _stream_chat(
        client: BrowserClient,
        prompt: str,
        use_deep_think: int,
        request_id: str,
        model: str,
        *,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        has_tools: bool = False,
        messages_for_counting: Optional[list] = None,
        account_id: Optional[str] = None,
    ):
        """Generate real-time SSE stream in OpenAI format via httpx streaming.

        Thinking state machine (mirrors old client.py logic):
        - block_type=10040 toggles thinking mode (1st=enter, 2nd=exit)
        - Text between markers -> delta.reasoning_content
        - Text after exit -> delta.content
        - block_type=10025 -> delta.search_results (incremental)
        - error_code in event -> emit error and stop
        """
        thinking_count = 0
        in_thinking = False
        had_reasoning_content = False  # Track if any reasoning was emitted
        stream_content_chars = 0  # Track total output chars for token estimation
        # Track last emitted result count per block_id for incremental updates
        search_last_count: dict = {}
        result_conversation_id: Optional[str] = None
        # Tool calling state
        tool_buffer = ""  # accumulates text when tool call detected
        tool_mode = False  # True when we're buffering potential tool call XML
        emitted_tool_calls = False  # True once we've emitted tool_calls chunks

        def _make_chunk(delta: dict, finish_reason=None):
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }],
            }

        def _iter_blocks(data: dict):
            """Yield content_block dicts from patch_op or top-level content."""
            for patch in data.get("patch_op", []):
                pv = patch.get("patch_value", {})
                yield from pv.get("content_block", [])
            dc = data.get("content", {})
            if isinstance(dc, dict):
                yield from dc.get("content_block", [])

        try:
            async for event in client.chat_completion(
                prompt, use_deep_think=use_deep_think,
                conversation_id=conversation_id or None,
                bot_id=bot_id or None,
            ):
                if event.get("error"):
                    chunk = _make_chunk(
                        {"content": f"[Error {event.get('status')}]"}
                    )
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                event_type = event.get("_event", "")

                # --- Extract conversation_id for multi-turn ---
                if not result_conversation_id:
                    cid = client.extract_conversation_id(event)
                    if cid and cid != "0":
                        result_conversation_id = cid

                # --- error_code handling (risk control, session expired) ---
                if event_type == "STREAM_ERROR" or event.get("error_code"):
                    code = event.get("error_code", 0)
                    msg = event.get("error_msg", "unknown error")
                    client.record_failure(code)
                    chunk = _make_chunk(
                        {"content": f"[Error code={code}: {msg}]"}
                    )
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # --- CHUNK_DELTA: compact {"text": "..."} (highest priority) ---
                if (
                    event_type == "CHUNK_DELTA"
                    and "text" in event
                    and isinstance(event.get("text"), str)
                    and event["text"]
                ):
                    t = event["text"]
                    # Tool calling: buffer text to detect XML tool_calls
                    if has_tools and not in_thinking:
                        tool_buffer += t
                        if not tool_mode and is_tool_call_start(tool_buffer):
                            tool_mode = True
                        if tool_mode:
                            # Check if we have complete tool calls
                            if has_complete_tool_calls(tool_buffer):
                                # Parse and emit as tool_calls
                                parsed = parse_tool_calls_xml(tool_buffer)
                                if parsed:
                                    # Emit tool_calls in OpenAI streaming format
                                    for idx, tc in enumerate(parsed):
                                        # First chunk: role + tool_call with function name
                                        delta_tc = {
                                            "role": "assistant",
                                            "content": None,
                                            "tool_calls": [{
                                                "index": idx,
                                                "id": tc["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": tc["function"]["name"],
                                                    "arguments": "",
                                                },
                                            }],
                                        }
                                        chunk = _make_chunk(delta_tc)
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                        # Second chunk: arguments content
                                        delta_args = {
                                            "tool_calls": [{
                                                "index": idx,
                                                "function": {
                                                    "arguments": tc["function"]["arguments"],
                                                },
                                            }],
                                        }
                                        chunk = _make_chunk(delta_args)
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    tool_buffer = ""
                                    tool_mode = False
                                    emitted_tool_calls = True
                                else:
                                    # XML complete but parse failed — flush as content
                                    log.warning("Tool call XML parse failed, flushing as content")
                                    delta = {"role": "assistant", "content": tool_buffer}
                                    chunk = _make_chunk(delta)
                                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    tool_buffer = ""
                                    tool_mode = False
                            continue  # don't emit raw text while in tool mode
                        else:
                            # Not a tool call start — flush buffer as normal content
                            if len(tool_buffer) > 20 and not is_tool_call_start(tool_buffer):
                                delta = {"role": "assistant", "content": tool_buffer}
                                chunk = _make_chunk(delta)
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                tool_buffer = ""
                            elif not tool_buffer.strip().startswith("<"):
                                # Definitely not XML, flush immediately
                                delta = {"role": "assistant", "content": tool_buffer}
                                chunk = _make_chunk(delta)
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                tool_buffer = ""
                            continue
                    # Normal (non-tool) path
                    if in_thinking:
                        delta = {"reasoning_content": t}
                        had_reasoning_content = True
                    else:
                        delta = {"role": "assistant", "content": t}
                    stream_content_chars += len(t)
                    chunk = _make_chunk(delta)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    continue

                # --- Process content_block arrays for markers & search ---
                has_content_block = False
                for cb in _iter_blocks(event):
                    has_content_block = True
                    bt = cb.get("block_type", 0)
                    block_content = cb.get("content", {})

                    if bt == 10040:
                        thinking_count += 1
                        in_thinking = (thinking_count == 1)
                        continue

                    if bt == 10025:
                        sqrb = block_content.get(
                            "search_query_result_block", {}
                        )
                        if sqrb:
                            block_id = cb.get("block_id", "")
                            queries = sqrb.get("queries", [])
                            results = sqrb.get("results", [])
                            parsed = [
                                {
                                    "title": r.get("text_card", {}).get("title", ""),
                                    "url": r.get("text_card", {}).get("url", ""),
                                    "summary": r.get("text_card", {}).get("summary", ""),
                                    "source": r.get("text_card", {}).get("source_name", ""),
                                }
                                for r in results if r.get("text_card")
                            ]
                            prev = search_last_count.get(block_id, 0)
                            if (parsed and len(parsed) > prev) or (queries and prev == 0):
                                search_last_count[block_id] = len(parsed)
                                chunk = _make_chunk({
                                    "search_results": {
                                        "queries": queries,
                                        "results": parsed,
                                        "summary": f"搜索 {len(queries)} 个关键词，参考 {len(parsed)} 篇资料",
                                    },
                                })
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        continue

                    if bt == 10000:
                        tb = block_content.get("text_block", {})
                        if isinstance(tb, dict) and tb.get("text"):
                            t = tb["text"]
                            # Tool calling: buffer text for XML detection
                            if has_tools and not in_thinking:
                                tool_buffer += t
                                if not tool_mode and is_tool_call_start(tool_buffer):
                                    tool_mode = True
                                if tool_mode:
                                    if has_complete_tool_calls(tool_buffer):
                                        parsed = parse_tool_calls_xml(tool_buffer)
                                        if parsed:
                                            for idx, tc in enumerate(parsed):
                                                delta_tc = {
                                                    "role": "assistant", "content": None,
                                                    "tool_calls": [{"index": idx, "id": tc["id"], "type": "function",
                                                        "function": {"name": tc["function"]["name"], "arguments": ""}}],
                                                }
                                                chunk = _make_chunk(delta_tc)
                                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                                delta_args = {"tool_calls": [{"index": idx,
                                                    "function": {"arguments": tc["function"]["arguments"]}}]}
                                                chunk = _make_chunk(delta_args)
                                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                            tool_buffer = ""
                                            tool_mode = False
                                            emitted_tool_calls = True
                                        else:
                                            # XML complete but parse failed
                                            delta = {"role": "assistant", "content": tool_buffer}
                                            chunk = _make_chunk(delta)
                                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                            tool_buffer = ""
                                            tool_mode = False
                                elif len(tool_buffer) > 20 and not is_tool_call_start(tool_buffer):
                                    delta = {"role": "assistant", "content": tool_buffer}
                                    chunk = _make_chunk(delta)
                                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                    tool_buffer = ""
                                continue
                            if in_thinking:
                                delta = {"reasoning_content": t}
                                had_reasoning_content = True
                            else:
                                delta = {"role": "assistant", "content": t}
                            stream_content_chars += len(t)
                            chunk = _make_chunk(delta)
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        continue

                # --- patch_op content string (only if no content_block found) ---
                if not has_content_block:
                    for patch in event.get("patch_op", []):
                        pv = patch.get("patch_value", {})
                        if isinstance(pv, dict) and "content" in pv:
                            content_str = pv.get("content", "")
                            if isinstance(content_str, str) and content_str:
                                try:
                                    content_obj = json.loads(content_str)
                                    t = content_obj.get("text", "")
                                    if t:
                                        if in_thinking:
                                            delta = {"reasoning_content": t}
                                            had_reasoning_content = True
                                        else:
                                            delta = {"role": "assistant", "content": t}
                                        chunk = _make_chunk(delta)
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                                except (json.JSONDecodeError, TypeError):
                                    pass

        except Exception as exc:
            log.error("Stream error: %s", exc)
            if account_id:
                accounts.mark_failure(account_id, str(exc))
            chunk = _make_chunk({"content": f"[Error: {exc}]"})
            yield f"data: {json.dumps(chunk)}\n\n"

        # Flush any remaining tool buffer
        if tool_buffer:
            if tool_mode and has_complete_tool_calls(tool_buffer):
                parsed = parse_tool_calls_xml(tool_buffer)
                if parsed:
                    for idx, tc in enumerate(parsed):
                        delta_tc = {
                            "role": "assistant", "content": None,
                            "tool_calls": [{"index": idx, "id": tc["id"], "type": "function",
                                "function": {"name": tc["function"]["name"], "arguments": ""}}],
                        }
                        chunk = _make_chunk(delta_tc)
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        delta_args = {"tool_calls": [{"index": idx,
                            "function": {"arguments": tc["function"]["arguments"]}}]}
                        chunk = _make_chunk(delta_args)
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    emitted_tool_calls = True
                else:
                    # Parse failed — flush as content
                    delta = {"role": "assistant", "content": tool_buffer}
                    chunk = _make_chunk(delta)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif tool_buffer.strip():
                # Emit as regular content
                delta = {"role": "assistant", "content": tool_buffer}
                chunk = _make_chunk(delta)
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # Final chunk with usage
        client.record_success()
        if account_id:
            accounts.mark_success(account_id)
        # Report to expert tracker for degradation detection
        if has_tools and use_deep_think >= 1:
            _expert_tracker.report_response(had_reasoning_content)

        # Estimate token usage
        prompt_tokens = 0
        if messages_for_counting:
            prompt_tokens = count_messages_tokens(
                [m.model_dump(exclude_none=True) for m in messages_for_counting]
            )
        completion_tokens = int(stream_content_chars / 2.5 * SAFETY_FACTOR) if stream_content_chars else 0

        final_delta: dict = {}
        if result_conversation_id:
            final_delta["conversation_id"] = result_conversation_id
        final_chunk = _make_chunk(final_delta, 'tool_calls' if emitted_tool_calls else 'stop')
        final_chunk["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # ── Admin Dashboard & Auth ──

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_dashboard(request: Request):
        """Serve the admin dashboard (QR login + system + API test + logs)."""
        _check_auth(request)
        novnc_url = os.environ.get("DOUBAO_NOVNC_URL", "").strip()
        if not novnc_url:
            scheme = request.url.scheme
            host = request.url.hostname or "localhost"
            novnc_url = f"{scheme}://{host}:6080/vnc.html"
        novnc_password = os.environ.get("DOUBAO_NOVNC_PASSWORD", "").strip()
        if novnc_password and "password=" not in novnc_url:
            sep = "&" if "?" in novnc_url else "?"
            novnc_url = f"{novnc_url}{sep}password={novnc_password}"
        from pathlib import Path
        html_path = Path(__file__).parent / "static" / "admin.html"
        html = html_path.read_text(encoding="utf-8")
        return html.replace("{{NOVNC_URL}}", novnc_url)

    @app.get("/auth")
    async def auth_redirect(request: Request):
        """Redirect /auth to /admin for backwards compatibility."""
        from fastapi.responses import RedirectResponse
        key = request.query_params.get("key", "")
        url = "/admin" + (f"?key={key}" if key else "")
        return RedirectResponse(url=url)

    @app.get("/admin/api/system")
    async def admin_system(request: Request):
        """Return system information."""
        _check_auth(request)
        import platform
        import sys
        uptime = int(time.time() - _SERVER_START_TIME)
        return JSONResponse({
            "python_version": sys.version,
            "platform": platform.platform(),
            "uptime_seconds": uptime,
            "rpm_limit": rpm_limit,
            "host": os.environ.get("DOUBAO_HOST", "0.0.0.0"),
            "port": int(os.environ.get("DOUBAO_PORT", "9090")),
            "account_data_root": account_data_root(),
            "account_db": accounts.store.path,
            "default_account_id": accounts.default_account_id,
            "max_hot_accounts": accounts.max_hot_accounts,
            "quota": {
                "window_seconds": accounts.store.quota_window_seconds(),
                "image_24h_default": accounts.store.quota_limit({"id": "_default", "quota": {}}, "image"),
                "video_24h_default": accounts.store.quota_limit({"id": "_default", "quota": {}}, "video"),
            },
            "models": {
                "chat": list(CHAT_MODELS.keys()),
                "image": ["doubao-image"],
                "video": list(VIDEO_MODEL_ALIASES.keys()),
                "audio": ["doubao-music"],
            },
        })

    @app.get("/admin/api/service/api-key")
    async def admin_service_api_key(request: Request):
        _check_auth(request)
        return JSONResponse({
            "provider": "DOUBAO-WEB-01",
            "api_key": _effective_api_key(),
            "api_base_path": "/v1",
            "header": "Authorization: Bearer",
        })

    @app.put("/admin/api/service/api-key")
    async def admin_update_service_api_key(request: Request):
        _check_auth(request)
        body = await _json_or_empty(request)
        new_key = str(body.get("api_key") or "").strip()
        if not new_key:
            raise HTTPException(status_code=400, detail="api_key is required")
        accounts.store.set_setting("api_key", new_key)
        return JSONResponse({
            "ok": True,
            "provider": "DOUBAO-WEB-01",
            "api_key": _effective_api_key(),
            "api_base_path": "/v1",
            "header": "Authorization: Bearer",
        })

    @app.post("/admin/api/service/api-key")
    async def admin_post_service_api_key(request: Request):
        return await admin_update_service_api_key(request)

    @app.get("/admin/api/logs")
    async def admin_logs(request: Request):
        """Return recent request logs from ring buffer."""
        _check_auth(request)
        return JSONResponse(list(_REQUEST_LOG))

    async def _json_or_empty(request: Request) -> Dict[str, Any]:
        try:
            data = await request.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _admin_account_id(request: Request, body: Optional[Dict[str, Any]] = None, fallback: Optional[str] = None) -> str:
        body = body or {}
        return (
            fallback
            or str(body.get("account_id") or body.get("doubao_account_id") or "")
            or request.query_params.get("account_id")
            or request.query_params.get("doubao_account_id")
            or accounts.default_account_id
        )

    @app.get("/admin/api/accounts")
    async def admin_accounts(request: Request):
        _check_auth(request)
        return JSONResponse({
            "accounts": accounts.list_accounts(),
            "default_account_id": accounts.default_account_id,
            "max_hot_accounts": accounts.max_hot_accounts,
        })

    @app.post("/admin/api/accounts")
    async def admin_create_account(request: Request):
        _check_auth(request)
        body = await _json_or_empty(request)
        account = accounts.store.create_account(
            name=str(body.get("name") or "").strip(),
            account_id=str(body.get("id") or body.get("account_id") or "").strip(),
        )
        if body.get("start"):
            try:
                await accounts.ensure_client(account["id"])
            except Exception as exc:
                accounts.store.mark_failure(account["id"], str(exc), "browser_error")
        return JSONResponse(account)

    @app.patch("/admin/api/accounts/{account_id}")
    async def admin_update_account(account_id: str, request: Request):
        _check_auth(request)
        body = await _json_or_empty(request)
        fields: Dict[str, Any] = {}
        for key in ("name", "enabled", "proxy_url", "tags_json", "models_json", "quota_json"):
            if key in body:
                value = body[key]
                if key.endswith("_json") and not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                fields[key] = value
        account = accounts.store.update_account(account_id, **fields)
        if account is None:
            raise HTTPException(status_code=404, detail="Account not found")
        if "enabled" in fields and not fields["enabled"]:
            await accounts.stop_client(account_id)
            account = accounts.store.get(account_id) or account
        return JSONResponse(account)

    @app.delete("/admin/api/accounts/{account_id}")
    async def admin_delete_account(account_id: str, request: Request):
        _check_auth(request)
        if account_id in {"default", accounts.default_account_id}:
            raise HTTPException(status_code=400, detail="Default account cannot be deleted. Disable or reset it instead.")
        try:
            if accounts.store.is_in_maintenance(account_id) or account_id in accounts.maintenance_clients:
                await accounts.stop_maintenance(account_id)
            await accounts.stop_client(account_id, update_status=False)
            result = accounts.store.delete_account(account_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not result.get("deleted"):
            raise HTTPException(status_code=404, detail="Account not found")
        return JSONResponse({"status": "deleted", "account_id": account_id, "cleanup": result.get("cleanup", [])})

    @app.post("/admin/api/accounts/{account_id}/start")
    async def admin_start_account(account_id: str, request: Request):
        _check_auth(request)
        try:
            account, client = await accounts.ensure_client(account_id)
            return JSONResponse({
                "status": "ready" if client.is_ready else "not_logged_in",
                "account": accounts.store.get(account_id) or account,
            })
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            return JSONResponse({"status": "error", "message": str(exc)}, status_code=502)

    @app.post("/admin/api/accounts/{account_id}/stop")
    async def admin_stop_account(account_id: str, request: Request):
        _check_auth(request)
        await accounts.stop_client(account_id)
        return JSONResponse({"status": "stopped", "account_id": account_id})

    @app.post("/admin/api/accounts/{account_id}/restart")
    async def admin_restart_account(account_id: str, request: Request):
        _check_auth(request)
        try:
            account, client = await accounts.restart_client(account_id)
            return JSONResponse({
                "status": "ready" if client.is_ready else "not_logged_in",
                "account": accounts.store.get(account_id) or account,
            })
        except Exception as exc:
            return JSONResponse({"status": "error", "message": str(exc)}, status_code=502)

    @app.get("/admin/api/accounts/{account_id}/maintenance")
    async def admin_account_maintenance_status(account_id: str, request: Request):
        _check_auth(request)
        try:
            return JSONResponse(accounts.maintenance_status(account_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/admin/api/accounts/{account_id}/maintenance/start")
    async def admin_account_maintenance_start(account_id: str, request: Request):
        _check_auth(request)
        try:
            return JSONResponse(await accounts.start_maintenance(account_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            return JSONResponse({"status": "error", "message": str(exc)[:500]}, status_code=409)

    @app.post("/admin/api/accounts/{account_id}/maintenance/heartbeat")
    async def admin_account_maintenance_heartbeat(account_id: str, request: Request):
        _check_auth(request)
        body = await _json_or_empty(request)
        try:
            owner = str(body.get("lease_owner") or accounts.maintenance_owners.get(account_id) or "")
            if not owner:
                raise RuntimeError("Maintenance lease owner is required")
            return JSONResponse(accounts.store.heartbeat_maintenance(account_id, owner))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            return JSONResponse({"status": "error", "message": str(exc)[:500]}, status_code=409)

    @app.post("/admin/api/accounts/{account_id}/maintenance/stop")
    async def admin_account_maintenance_stop(account_id: str, request: Request):
        _check_auth(request)
        try:
            return JSONResponse(await accounts.stop_maintenance(account_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            return JSONResponse({"status": "error", "message": str(exc)[:500]}, status_code=409)

    @app.post("/admin/api/accounts/{account_id}/maintenance/validate")
    async def admin_account_maintenance_validate(account_id: str, request: Request):
        _check_auth(request)
        if accounts.store.is_in_maintenance(account_id) or account_id in accounts.maintenance_clients:
            return JSONResponse(
                {"status": "error", "message": "Stop the maintenance browser before validation."},
                status_code=409,
            )
        try:
            status = await accounts.login_status(account_id)
            return JSONResponse({"status": "valid" if status.get("logged_in") else "invalid", "data": status})
        except Exception as exc:
            return JSONResponse({"status": "invalid", "message": str(exc)[:500]}, status_code=424)

    @app.post("/admin/api/accounts/{account_id}/quota/sync")
    async def admin_sync_account_quota(account_id: str, request: Request):
        _check_auth(request)
        try:
            result = await accounts.sync_provider_credit(account_id)
            return JSONResponse({"status": "synced", **result})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            return JSONResponse(
                {"status": "error", "account_id": account_id, "message": str(exc)[:500]},
                status_code=502,
            )

    @app.get("/admin/api/cookies")
    async def admin_cookies(request: Request):
        """Return browser cookies for one account."""
        _check_auth(request)
        account_id = _admin_account_id(request)
        try:
            cookies = await accounts.cookies(account_id)
            return JSONResponse({"account_id": account_id, "cookies": cookies, "total": len(cookies)})
        except Exception as exc:
            return JSONResponse({"account_id": account_id, "cookies": [], "total": 0, "message": str(exc)})

    @app.get("/admin/api/accounts/{account_id}/cookies")
    async def admin_account_cookies(account_id: str, request: Request):
        _check_auth(request)
        try:
            cookies = await accounts.cookies(account_id)
            return JSONResponse({"account_id": account_id, "cookies": cookies, "total": len(cookies)})
        except Exception as exc:
            return JSONResponse({"account_id": account_id, "cookies": [], "total": 0, "message": str(exc)})

    @app.post("/admin/api/probe")
    async def admin_probe(request: Request):
        """Probe session by making a real chat request."""
        _check_auth(request)
        body = await _json_or_empty(request)
        account_id = _admin_account_id(request, body)
        return await _probe_account(account_id)

    @app.post("/admin/api/accounts/{account_id}/probe")
    async def admin_account_probe(account_id: str, request: Request):
        _check_auth(request)
        return await _probe_account(account_id)

    async def _probe_account(account_id: str):
        try:
            account, client = await accounts.get_ready_client(account_id)
        except Exception as exc:
            return JSONResponse({"status": "error", "account_id": account_id, "message": str(exc)[:300]})
        try:
            t0 = time.time()
            result = await client.chat("1+1=?只回答数字", use_deep_think=0)
            ms = int((time.time() - t0) * 1000)
            content = result.get("text", "")
            accounts.mark_success(account["id"])
            return JSONResponse({"status": "healthy", "account_id": account["id"], "ms": ms, "response": content[:200]})
        except Exception as exc:
            accounts.mark_failure(account["id"], str(exc))
            return JSONResponse({"status": "error", "account_id": account["id"], "message": str(exc)[:300]})

    @app.post("/auth/login")
    async def auth_login(request: Request):
        """Trigger browser QR login flow for one account."""
        _check_auth(request)
        body = await _json_or_empty(request)
        account_id = _admin_account_id(request, body)
        try:
            account, client = await accounts.ensure_client(account_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if client.is_ready:
            return {"status": "already_logged_in", "account_id": account["id"]}
        asyncio.create_task(_do_login(account["id"], client))
        return {"status": "login_started", "account_id": account["id"], "message": "QR code displayed in browser. Scan to login."}

    @app.post("/auth/reset_captcha")
    async def reset_captcha(request: Request):
        """Reset captcha flag after manual verification via VNC."""
        _check_auth(request)
        body = await _json_or_empty(request)
        account_id = _admin_account_id(request, body)
        account, client = await accounts.ensure_client(account_id)
        client.record_success()
        if not client.is_ready:
            client._ready = True
        accounts.mark_success(account["id"])
        return {"status": "ok", "account_id": account["id"], "message": "Captcha flag cleared, service resumed."}

    async def _do_login(account_id: str, client: BrowserClient):
        """Background login task."""
        try:
            ok = await client.wait_for_login(timeout=120)
            if ok:
                accounts.mark_success(account_id)
                log.info("QR login successful via /auth for account %s", account_id)
            else:
                accounts.store.mark_failure(account_id, "QR login timed out", "not_logged_in")
                log.warning("QR login timed out for account %s", account_id)
        except Exception as exc:
            accounts.mark_failure(account_id, str(exc))
            log.error("QR login error for account %s: %s", account_id, exc)

    @app.get("/auth/status")
    async def auth_status(request: Request):
        return await _get_login_status(request)

    @app.get("/admin/api/status")
    async def admin_api_status(request: Request):
        return await _get_login_status(request)

    @app.get("/admin/api/accounts/{account_id}/status")
    async def admin_account_status(account_id: str, request: Request):
        _check_auth(request)
        try:
            return JSONResponse(await accounts.login_status(account_id))
        except Exception as exc:
            return JSONResponse({"logged_in": False, "account_id": account_id, "browser": "not_started", "message": str(exc)})

    async def _get_login_status(request: Request):
        _check_auth(request)
        account_id = _admin_account_id(request)
        try:
            status = await accounts.login_status(account_id)
        except Exception as exc:
            return {"logged_in": False, "account_id": account_id, "browser": "not_started", "message": str(exc)}
        status["accounts"] = accounts.counts()
        return status

    @app.post("/auth/eval")
    async def auth_eval(request: Request):
        """Evaluate JS on the browser page (debug only)."""
        _check_auth(request)
        body = await _json_or_empty(request)
        account_id = _admin_account_id(request, body)
        _, client = await accounts.ensure_client(account_id)
        if client.page is None:
            raise HTTPException(status_code=503, detail="Browser not available")
        js = body.get("js", "")
        if not js:
            raise HTTPException(status_code=400, detail="Missing 'js' field")
        try:
            result = await client.page.evaluate(js)
            return {"account_id": account_id, "result": result}
        except Exception as e:
            return {"account_id": account_id, "error": str(e)}

    @app.get("/auth/screenshot")
    async def auth_screenshot(request: Request):
        """Return a screenshot of the selected browser page."""
        _check_auth(request)
        account_id = _admin_account_id(request)
        _, client = await accounts.ensure_client(account_id)
        if client.page is None:
            raise HTTPException(status_code=503, detail="Browser not available")
        png_bytes = await client.page.screenshot()
        from fastapi.responses import Response
        return Response(content=png_bytes, media_type="image/png")

    @app.get("/admin/api/accounts/{account_id}/screenshot")
    async def admin_account_screenshot(account_id: str, request: Request):
        _check_auth(request)
        _, client = await accounts.ensure_client(account_id)
        if client.page is None:
            raise HTTPException(status_code=503, detail="Browser not available")
        png_bytes = await client.page.screenshot()
        from fastapi.responses import Response
        return Response(content=png_bytes, media_type="image/png")

    # ── QR Login (pure HTTP, no VNC needed) ──

    _qr_login_states: Dict[str, Dict[str, Any]] = {}

    @app.post("/admin/api/accounts/{account_id}/qr-login")
    async def admin_account_qr_login_start(account_id: str, request: Request):
        return await _start_qr_login(request, account_id)

    @app.get("/admin/api/accounts/{account_id}/qr-login")
    async def admin_account_qr_login_poll(account_id: str, request: Request):
        return await _poll_qr_login(request, account_id)

    @app.post("/v1/session/qr-login")
    async def session_qr_login_start(request: Request):
        body = await _json_or_empty(request)
        account_id = _admin_account_id(request, body)
        return await _start_qr_login(request, account_id)

    @app.get("/v1/session/qr-login")
    async def session_qr_login_poll(request: Request):
        account_id = _admin_account_id(request)
        return await _poll_qr_login(request, account_id)

    async def _start_qr_login(request: Request, account_id: str):
        """Start QR login flow. Returns base64 QR code PNG."""
        _check_auth(request)
        from .qr_login import QRLogin, QRStatus

        if not accounts.store.get(account_id):
            raise HTTPException(status_code=404, detail="Account not found")
        await accounts.ensure_client(account_id)

        state = _qr_login_states.setdefault(account_id, {})
        if state.get("instance"):
            state["instance"].cancel()

        qr = QRLogin()
        state.clear()
        state["instance"] = qr
        state["status"] = "starting"
        state["error"] = ""
        state["account_id"] = account_id

        loop = asyncio.get_event_loop()

        def on_status(status: QRStatus, msg: str):
            state["status"] = status.value
            if msg == "qr_ready":
                state["qr_ready"] = True

        def on_done(result):
            if result.status == QRStatus.CONFIRMED:
                state["status"] = "success"
                state["cookies"] = result.cookies
                loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(
                        _inject_qr_cookies(account_id, result.cookies)
                    )
                )
                log.info("QR login success for account %s: %d cookies", account_id, len(result.cookies))
            else:
                state["status"] = "failed"
                state["error"] = result.error
                accounts.store.mark_failure(account_id, result.error or "QR login failed", "not_logged_in")

        qr.start(on_status=on_status, on_done=on_done)

        for _ in range(20):
            await asyncio.sleep(0.1)
            if qr.qrcode_data:
                break

        if qr.qrcode_data:
            import base64 as b64
            qr_b64 = b64.b64encode(qr.qrcode_data).decode()
            return JSONResponse({
                "status": "qr_ready",
                "account_id": account_id,
                "qr_image_base64": qr_b64,
                "message": "请用豆包 App 扫码。轮询 GET /v1/session/qr-login 获取状态。",
            })
        return JSONResponse({
            "status": state.get("status", "error"),
            "account_id": account_id,
            "error": state.get("error", "生成二维码失败"),
        }, status_code=502)

    async def _poll_qr_login(request: Request, account_id: str):
        _check_auth(request)
        state = _qr_login_states.get(account_id, {})
        status = state.get("status", "idle")
        resp: Dict[str, Any] = {"status": status, "account_id": account_id}
        if status == "success":
            resp["message"] = "登录成功，session 已更新"
            resp["cookies_count"] = len(state.get("cookies", {}))
            resp["browser_ready"] = state.get("browser_ready", False)
        elif status == "failed":
            resp["error"] = state.get("error", "未知错误")
        elif status == "idle":
            resp["message"] = "无进行中的登录。POST /v1/session/qr-login 开始。"
        return JSONResponse(resp)

    async def _inject_qr_cookies(account_id: str, cookies: Dict[str, str]):
        """Inject QR login cookies into the selected Playwright account and verify."""
        state = _qr_login_states.setdefault(account_id, {"account_id": account_id})
        try:
            _, client = await accounts.ensure_client(account_id)
            ok = await client.inject_cookies_and_reload(cookies)
            if ok:
                accounts.mark_success(account_id)
                log.info("QR cookies injected successfully for account %s", account_id)
                state["browser_ready"] = True
            else:
                accounts.store.mark_failure(account_id, "QR cookies injected but login check failed", "not_logged_in")
                log.warning("QR cookies injected but login check failed for account %s", account_id)
                state["browser_ready"] = False
        except Exception as e:
            accounts.mark_failure(account_id, str(e))
            log.error("Failed to inject QR cookies for account %s: %s", account_id, e)
            state["browser_ready"] = False

    return app




# ── Server runner ──


def run_server():
    """Start the uvicorn server with env-based configuration."""
    import uvicorn

    host = os.environ.get("DOUBAO_HOST", "0.0.0.0")
    port = int(os.environ.get("DOUBAO_PORT", "9090"))
    api_key = os.environ.get("DOUBAO_API_KEY", "")
    rpm = float(os.environ.get("DOUBAO_RPM_LIMIT", "20"))
    novnc_url = os.environ.get("DOUBAO_NOVNC_URL", "")

    app = create_app(api_key=api_key or None, rpm_limit=rpm)

    print(f"\n  Doubao API Server (Playwright)")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Admin page: http://{host}:{port}/admin")
    if novnc_url:
        print(f"  noVNC: {novnc_url}")
    if api_key:
        print(f"  API Key: {api_key[:4]}{'*' * (len(api_key) - 4)}")
    print()

    uvicorn.run(app, host=host, port=port, log_level="info")
