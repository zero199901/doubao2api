"""Playwright-based Doubao client with in-browser fetch.

Architecture:
- Playwright: Login (QR scan via noVNC) + page session
- In-browser fetch(): API requests go through ByteDance's fetch hook which
  automatically injects a_bogus/msToken signatures with real browser fingerprint
- httpx: Only used for file upload (TOS/ImageX flow, no fetch hook needed)
- expose_function bridge: Streams SSE chunks from browser JS back to Python

ByteDance's frontend exposes window.bdms.frontierSign() which generates
X-Bogus signatures. We use Playwright only to maintain a logged-in page
and call this signing function. All actual API traffic goes through httpx.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any, List
from urllib.parse import urlencode

import httpx
from playwright.async_api import async_playwright, BrowserContext, Page
from playwright_stealth import Stealth

log = logging.getLogger(__name__)

DOUBAO_URL = "https://www.doubao.com"
CHAT_URL = f"{DOUBAO_URL}/chat/"
COMPLETION_URL = f"{DOUBAO_URL}/chat/completion"
SAMANTHA_COMPLETION_URL = f"{DOUBAO_URL}/samantha/chat/completion"
DEFAULT_BOT_ID = "7338286299411103781"
DEFAULT_PC_VERSION = os.environ.get("DOUBAO_PC_VERSION", "3.22.5")
DEFAULT_VERSION_CODE = os.environ.get("DOUBAO_VERSION_CODE", "20800")
DEFAULT_TIMEZONE = os.environ.get("DOUBAO_TIMEZONE", "Asia/Tokyo")
DEFAULT_USER_AGENT = os.environ.get(
    "DOUBAO_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36",
)
DEFAULT_SESSION_FILE = os.environ.get("DOUBAO_SESSION_FILE", "/app/data/.doubao_session.json")
DEFAULT_CHROMIUM_EXECUTABLE_PATH = (
    os.environ.get("DOUBAO_CHROMIUM_EXECUTABLE_PATH")
    or os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
)


class BrowserClient:
    """Manages Playwright for login and in-browser fetch for API calls."""

    def __init__(
        self,
        headless: bool = True,
        user_data_dir: Optional[str] = None,
        session_file: Optional[str] = None,
        cookie_header: Optional[str] = None,
    ):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.session_file = session_file or DEFAULT_SESSION_FILE
        self.cookie_header = os.environ.get("DOUBAO_COOKIE", "").strip() if cookie_header is None else cookie_header.strip()
        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._ready = False
        self._device_id: Optional[str] = None
        self._web_id: Optional[str] = None
        self._fp: Optional[str] = None
        self._region: str = ""
        self._sys_region: str = ""
        self._is_old_user: bool = True
        # msToken rotation: updated from x-ms-token response header
        self._ms_token: str = ""
        # Robustness: failure tracking
        self._consecutive_failures: int = 0
        self._last_error_code: int = 0
        self._needs_captcha: bool = False
        # Stream bridge: request_id -> asyncio.Queue for SSE chunks
        self._stream_queues: Dict[str, asyncio.Queue] = {}
        self._bridge_ready: bool = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def page(self) -> Optional[Page]:
        return self._page

    @property
    def needs_captcha(self) -> bool:
        return self._needs_captcha

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def last_error_code(self) -> int:
        return self._last_error_code

    def record_success(self):
        """Reset failure counters on successful request."""
        self._consecutive_failures = 0
        self._last_error_code = 0
        self._needs_captcha = False

    def record_failure(self, error_code: int = 0):
        """Track consecutive failures. Mark captcha-needed on 710022004."""
        self._consecutive_failures += 1
        self._last_error_code = error_code
        if error_code == 710022004:
            self._needs_captcha = True
            log.warning("Captcha required (710022004) - marking needs_captcha=True")
        if self._consecutive_failures >= 5:
            log.error("5 consecutive failures - marking not ready")
            self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Launch browser, navigate to Doubao, init httpx client."""
        log.info("Starting BrowserClient (headless=%s)", self.headless)
        self._playwright = await async_playwright().start()

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-sandbox",
        ]
        launch_options: Dict[str, Any] = {
            "headless": self.headless,
            "args": launch_args,
        }
        if DEFAULT_CHROMIUM_EXECUTABLE_PATH:
            launch_options["executable_path"] = DEFAULT_CHROMIUM_EXECUTABLE_PATH

        if self.user_data_dir:
            self._clear_stale_profile_locks()
            self._context = await self._playwright.chromium.launch_persistent_context(
                self.user_data_dir,
                **launch_options,
                viewport={"width": 1280, "height": 720},
                locale="zh-CN",
                timezone_id=DEFAULT_TIMEZONE,
                user_agent=DEFAULT_USER_AGENT,
            )
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        else:
            browser = await self._playwright.chromium.launch(
                **launch_options,
            )
            self._context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                locale="zh-CN",
                timezone_id=DEFAULT_TIMEZONE,
                user_agent=DEFAULT_USER_AGENT,
            )
            self._page = await self._context.new_page()

        # Stealth patches
        stealth = Stealth(navigator_languages_override=("zh-CN", "zh"))
        await stealth.apply_stealth_async(self._page)

        await self._load_saved_cookies()

        # Navigate
        log.info("Navigating to %s", CHAT_URL)
        await self._page.goto(CHAT_URL, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(3)

        # Init httpx
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10))

        await self._check_login_state()

    def _clear_stale_profile_locks(self):
        """Remove Chromium profile singleton locks left by a crashed/replaced container."""
        if os.environ.get("DOUBAO_CLEAR_STALE_PROFILE_LOCKS", "false").lower() != "true":
            return
        if not self.user_data_dir:
            return
        profile = Path(self.user_data_dir)
        for name in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
            path = profile / name
            try:
                if path.exists() or path.is_symlink():
                    path.unlink()
                    log.info("Removed stale Chromium profile lock: %s", path)
            except Exception as exc:
                log.warning("Failed to remove Chromium profile lock %s: %s", path, exc)

    async def stop(self):
        """Close browser and httpx client."""
        if self._http:
            await self._http.aclose()
            self._http = None
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._context = None
        self._playwright = None
        self._page = None
        self._ready = False
        log.info("BrowserClient stopped")

    async def is_alive(self) -> bool:
        """Check if browser process is still responsive."""
        if not self._page or not self._context:
            return False
        try:
            result = await asyncio.wait_for(
                self._page.evaluate("1+1"), timeout=5
            )
            return result == 2
        except Exception as e:
            if "Execution context was destroyed" in str(e):
                return True
            log.warning("Browser health check failed: %s", e)
            return False

    async def restart(self):
        """Stop and restart the browser client."""
        log.info("Restarting BrowserClient...")
        await self.stop()
        await asyncio.sleep(2)
        await self.start()
        log.info("BrowserClient restarted. ready=%s", self._ready)

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    async def _check_login_state(self):
        """Check if logged in by looking for login button."""
        login_btn = self._page.locator('button:has-text("登录")')
        btn_count = await login_btn.count()
        log.info("Login check: login_button_count=%d", btn_count)

        if btn_count > 0:
            log.info("Not logged in - login button visible")
            self._ready = False
            return

        self._ready = True
        await self._extract_params()
        await self._seed_ms_token()
        await self._setup_fetch_bridge()
        await self._verify_fetch_hook()
        await self._wait_for_signing()  # still needed for upload endpoints
        log.info("Ready! device_id=%s, fetch_hook=%s", self._device_id, self._bridge_ready)

    async def _extract_params(self):
        """Extract live device/web params from localStorage/cookies.

        Doubao Web rotates several identifiers after login and after page-side
        experiments load. Keeping startup values around causes stale requests,
        so callers refresh these before every browser fetch.
        """
        for _ in range(5):
            params = await self._page.evaluate("""() => {
                const result = {};
                try {
                    const samWeb = JSON.parse(localStorage.getItem('samantha_web_web_id') || '{}');
                    result.device_id = samWeb.web_id || '';
                } catch(e) {}
                try {
                    const tea = JSON.parse(localStorage.getItem('__tea_cache_tokens_497858') || '{}');
                    result.web_id = tea.web_id || '';
                } catch(e) {}
                const fpCookie = document.cookie.split(';')
                    .map(c => c.trim())
                    .find(c => c.startsWith('s_v_web_id='));
                result.fp = fpCookie ? fpCookie.split('=')[1] : '';
                result.region = localStorage.getItem('flow_user_country') || '';
                result.sys_region = result.region || '';
                const userId = localStorage.getItem('flow_tea_user_id') || '';
                result.is_old_user = userId
                    ? localStorage.getItem(`ug_attribution_is_old_user_${userId}`) === 'true'
                    : true;
                return result;
            }""")
            self._device_id = params.get("device_id", "")
            self._web_id = params.get("web_id", "")
            self._fp = params.get("fp", "")
            self._region = params.get("region", "") or ""
            self._sys_region = params.get("sys_region", "") or self._region
            self._is_old_user = bool(params.get("is_old_user", True))
            if self._device_id and self._web_id:
                break
            await asyncio.sleep(1)
        log.info("Params: device_id=%s, web_id=%s, fp=%s",
                 self._device_id, self._web_id, self._fp[:20] if self._fp else "")

    async def _refresh_runtime_params(self):
        """Refresh volatile browser params before a request."""
        if not self._page:
            return
        await self._extract_params()
        await self._seed_ms_token()

    async def _load_saved_cookies(self):
        """Load cookies from the configured cookie header or session file into the context."""
        if not self._context:
            return
        cookies: Dict[str, str] = {}
        cookie_header = self.cookie_header
        if cookie_header:
            for part in cookie_header.split(";"):
                if "=" not in part:
                    continue
                name, value = part.split("=", 1)
                cookies[name.strip()] = value.strip()
        else:
            path = Path(self.session_file)
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    raw = data.get("cookies", {})
                    if isinstance(raw, dict):
                        cookies = {str(k): str(v) for k, v in raw.items() if v}
                except Exception as exc:
                    log.warning("Failed to load session file %s: %s", path, exc)
        if not cookies:
            return
        await self._context.add_cookies([
            {"name": name, "value": value, "domain": ".doubao.com", "path": "/"}
            for name, value in cookies.items()
        ])
        log.info("Loaded %d cookies into browser context", len(cookies))

    async def _save_current_cookies(self):
        """Persist browser cookies for restart recovery."""
        if not self._context:
            return
        path = Path(self.session_file)
        try:
            cookies = await self._context.cookies("https://www.doubao.com")
            data = {
                "cookies": {c["name"]: c["value"] for c in cookies},
                "saved_at": int(time.time()),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("Saved %d cookies to %s", len(cookies), path)
        except Exception as exc:
            log.warning("Failed to save cookies to %s: %s", path, exc)

    async def _wait_for_signing(self):
        """Wait for bdms.frontierSign to become available (legacy, kept for upload signing)."""
        for i in range(12):  # up to 60s
            has_sign = await self._page.evaluate(
                "() => typeof window.bdms?.frontierSign === 'function'"
            )
            if has_sign:
                log.info("bdms.frontierSign available after %ds", (i + 1) * 5)
                return
            await asyncio.sleep(5)
        log.warning("bdms.frontierSign not available after 60s - signing may fail")

    async def _setup_fetch_bridge(self):
        """Register expose_function callback for streaming data from browser to Python."""
        if self._bridge_ready:
            return

        async def _on_stream_chunk(request_id: str, chunk_json: str):
            """Called from browser JS for each SSE chunk or completion signal."""
            queue = self._stream_queues.get(request_id)
            if queue:
                await queue.put(chunk_json)

        try:
            await self._page.expose_function("__doubaoStreamChunk", _on_stream_chunk)
            self._bridge_ready = True
            log.info("Fetch bridge registered (expose_function ready)")
        except Exception as e:
            # May already be registered if page didn't navigate
            if "already been registered" in str(e).lower():
                self._bridge_ready = True
                log.info("Fetch bridge already registered")
            else:
                log.error("Failed to register fetch bridge: %s", e)
                raise

    async def _verify_fetch_hook(self):
        """Verify ByteDance's fetch interceptor is active (adds a_bogus)."""
        for i in range(15):  # up to 30s
            hooked = await self._page.evaluate("""() => {
                try {
                    const s = window.fetch.toString();
                    return !s.includes('native code');
                } catch(e) { return false; }
            }""")
            if hooked:
                log.info("Fetch hook verified active after %ds", (i + 1) * 2)
                return True
            await asyncio.sleep(2)
        log.warning("Fetch hook NOT detected after 30s - requests may fail")
        return False

    async def wait_for_login(self, timeout: int = 120) -> bool:
        """Wait for user to scan QR code via noVNC."""
        await self._trigger_login_dialog()
        log.info("Waiting for QR scan login (timeout=%ds)...", timeout)
        try:
            login_btn = self._page.locator('button:has-text("登录")')
            await login_btn.wait_for(state="hidden", timeout=timeout * 1000)
            await asyncio.sleep(2)
            if await login_btn.count() == 0:
                self._ready = True
                await self._extract_params()
                await self._seed_ms_token()
                await self._setup_fetch_bridge()
                await self._verify_fetch_hook()
                await self._wait_for_signing()
                log.info("Login successful!")
                return True
            return False
        except Exception as e:
            log.error("Login timeout: %s", e)
            return False

    async def _trigger_login_dialog(self):
        """Click login button to show QR code."""
        btn = self._page.locator('button:has-text("登录")')
        if await btn.count() > 0:
            await btn.click()
            await asyncio.sleep(2)


    async def inject_cookies_and_reload(self, cookies: Dict[str, str]) -> bool:
        """Inject cookies from QR login into browser context and reload.

        After qr_login.py obtains session cookies via pure HTTP,
        this method injects them into Playwright so that bdms.frontierSign
        becomes available.

        Returns True if login state is confirmed after reload.
        """
        if not self._context or not self._page:
            log.error("inject_cookies: browser not started")
            return False

        # Build cookie list for Playwright
        pw_cookies = []
        for name, value in cookies.items():
            pw_cookies.append({
                "name": name,
                "value": value,
                "domain": ".doubao.com",
                "path": "/",
            })

        await self._context.add_cookies(pw_cookies)
        log.info("Injected %d cookies into browser context", len(pw_cookies))

        # Reload page to pick up new session
        await self._page.reload(wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(3)

        # Re-check login state
        await self._check_login_state()
        if self._ready:
            await self._save_current_cookies()
        return self._ready
    # ------------------------------------------------------------------
    # Signing & Cookies
    # ------------------------------------------------------------------

    async def _get_cookies_string(self) -> str:
        """Get full cookie string including httpOnly cookies."""
        cookies = await self._context.cookies("https://www.doubao.com")
        return "; ".join(f"{c['name']}={c['value']}" for c in cookies)

    async def _get_csrf_token(self) -> str:
        """Get passport_csrf_token from browser cookies."""
        cookies = await self._context.cookies("https://www.doubao.com")
        for c in cookies:
            if c["name"] == "passport_csrf_token":
                return c["value"]
            if c["name"] == "passport_csrf_token_default":
                return c["value"]
        return ""

    async def _seed_ms_token(self):
        """Seed initial msToken from browser cookies."""
        cookies = await self._context.cookies("https://www.doubao.com")
        for c in cookies:
            if c["name"] == "msToken":
                self._ms_token = c["value"]
                log.info("Seeded msToken from cookies (%d chars)", len(c["value"]))
                return
        log.warning("No msToken cookie found - first request may trigger rate limit")

    async def _sign_url(self, base_url: str, params: Dict[str, str]) -> str:
        """Sign a URL using bdms.frontierSign with retry on failure."""
        sorted_params = dict(sorted(params.items()))
        query_string = urlencode(sorted_params)

        last_error = None
        for attempt in range(3):
            try:
                sig = await self._page.evaluate(
                    f'window.bdms.frontierSign("{query_string}")'
                )

                x_bogus = ""
                if isinstance(sig, dict):
                    x_bogus = sig.get("X-Bogus") or sig.get("a_bogus", "")
                elif isinstance(sig, str):
                    x_bogus = sig

                if x_bogus:
                    return f"{base_url}?{query_string}&X-Bogus={x_bogus}"

                last_error = f"empty signature: {sig}"
            except Exception as e:
                last_error = str(e)
                log.warning("frontierSign attempt %d failed: %s", attempt + 1, e)

            if attempt < 2:
                await asyncio.sleep(1)

        log.error("frontierSign failed after 3 attempts: %s", last_error)
        raise RuntimeError(f"Failed to generate X-Bogus signature: {last_error}")

    def _build_query_params(self) -> Dict[str, str]:
        """Build the standard query parameters for API calls."""
        params = {
            "aid": "497858",
            "device_id": self._device_id or "",
            "device_platform": "web",
            "fp": self._fp or "",
            "language": "zh",
            "pc_version": DEFAULT_PC_VERSION,
            "pkg_type": "release_version",
            "real_aid": "497858",
            "region": self._region or "",
            "samantha_web": "1",
            "sys_region": self._sys_region or self._region or "",
            "tea_uuid": self._web_id or "",
            "use-olympus-account": "1",
            "version_code": DEFAULT_VERSION_CODE,
            "web_id": self._web_id or "",
            "web_platform": "browser",
            "web_tab_id": str(uuid.uuid4()),
        }
        if self._ms_token and os.environ.get("DOUBAO_INCLUDE_MSTOKEN", "false").lower() == "true":
            params["msToken"] = self._ms_token
        return params

    def _build_headers(self, cookie_str: str, csrf_token: str = "") -> Dict[str, str]:
        """Build request headers."""
        # Extract CSRF token from cookie string if not provided
        if not csrf_token:
            for part in cookie_str.split("; "):
                if part.startswith("passport_csrf_token="):
                    csrf_token = part.split("=", 1)[1]
                    break
        headers = {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Content-Type": "application/json",
            "Cookie": cookie_str,
            "Origin": DOUBAO_URL,
            "Referer": CHAT_URL,
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "agw-js-conv": "str, str",
        }
        if csrf_token:
            headers["x-tt-passport-csrf-token"] = csrf_token
        return headers

    # ------------------------------------------------------------------
    # Chat Completion (streaming via in-browser fetch)
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: int = 0,
        chat_ability: Optional[Dict[str, Any]] = None,
        image_attachments: Optional[List[Dict[str, Any]]] = None,
        split_image_attachments: bool = False,
        attachment_type: int = 2,
        stream_timeout: float = 180,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send a chat message and yield SSE events via in-browser fetch."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        await self._refresh_runtime_params()
        need_create = conversation_id is None or conversation_id == ""
        effective_bot_id = bot_id or DEFAULT_BOT_ID
        msg_uuid = str(uuid.uuid4())
        local_conv_id = f"local_{uuid.uuid4().int % 10**16}"
        now_ms = int(time.time() * 1000)
        now_sec = int(time.time())

        messages: List[Dict[str, Any]] = []
        if split_image_attachments and image_attachments:
            attachment_blocks = self._build_chat_attachment_blocks(
                image_attachments,
                attachment_type=attachment_type,
            )
            if attachment_blocks:
                messages.append({
                    "local_message_id": msg_uuid,
                    "content_block": attachment_blocks,
                    "message_status": 0,
                })
        text_message_id = str(uuid.uuid4()) if messages else msg_uuid
        messages.append({
            "local_message_id": text_message_id,
            "content_block": self._build_chat_content_blocks(
                text,
                image_attachments=None if split_image_attachments else image_attachments,
                attachment_type=attachment_type,
            ),
            "message_status": 0,
        })

        payload = {
            "client_meta": {
                "local_conversation_id": local_conv_id if need_create else "",
                "conversation_id": conversation_id or "",
                "bot_id": effective_bot_id,
                "last_section_id": "",
                "last_message_index": None,
            },
            "messages": messages,
            "option": {
                "send_message_scene": "",
                "create_time_ms": now_ms,
                "collect_id": "",
                "is_audio": False,
                "answer_with_suggest": False,
                "tts_switch": False,
                "need_deep_think": use_deep_think,
                "click_clear_context": False,
                "from_suggest": False,
                "is_regen": False,
                "is_replace": False,
                "is_from_click_option": False,
                "disable_sse_cache": False,
                "select_text_action": "",
                "is_select_text": False,
                "resend_for_regen": False,
                "scene_type": 0,
                "unique_key": str(uuid.uuid4()),
                "start_seq": 0,
                "need_create_conversation": need_create,
                "conversation_init_option": {"need_ack_conversation": True},
                "regen_query_id": [],
                "edit_query_id": [],
                "regen_instruction": "",
                "no_replace_for_regen": False,
                "message_from": 0,
                "shared_app_name": "",
                "shared_app_id": "",
                "sse_recv_event_options": {"support_chunk_delta": True},
                "is_ai_playground": False,
                "is_old_user": self._is_old_user,
                "recovery_option": {
                    "is_recovery": False,
                    "req_create_time_sec": now_sec,
                    "append_sse_event_scene": 0,
                },
                "message_storage_type": 0,
            },
            "user_context": [],
            "ext": {
                "use_deep_think": str(use_deep_think),
                "fp": self._fp or "",
                "sub_conv_firstmet_type": "1" if need_create else "0",
                "collection_id": "",
                "conversation_init_option": json.dumps({"need_ack_conversation": True}),
                "commerce_credit_config_enable": "0",
            },
        }
        if chat_ability:
            payload["chat_ability"] = chat_ability
            payload["ext"]["answer_with_suggest"] = "0"

        # Build URL with query params (fetch hook will add a_bogus/msToken)
        query_params = self._build_query_params()
        query_string = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))
        url = f"/chat/completion?{query_string}"

        request_id = f"req_{uuid.uuid4().hex[:16]}"
        queue: asyncio.Queue = asyncio.Queue()
        self._stream_queues[request_id] = queue

        log.info("POST %s (conv=%s, deep_think=%s) [browser fetch]",
                 url.split("?")[0], conversation_id or "new", use_deep_think)

        # Launch browser fetch in background
        eval_task = asyncio.create_task(
            self._browser_fetch_stream(url, payload, request_id)
        )

        # Yield parsed SSE events from queue
        try:
            while True:
                chunk_json = await asyncio.wait_for(queue.get(), timeout=stream_timeout)
                if chunk_json is None:
                    # Stream complete
                    break
                if chunk_json.startswith("__ERROR__:"):
                    error_msg = chunk_json[10:]
                    log.error("Browser fetch error: %s", error_msg[:200])
                    yield {"error": True, "status": 0, "body": error_msg}
                    break
                if chunk_json.startswith("__HTTP_ERROR__:"):
                    status = int(chunk_json[15:].split(":", 1)[0])
                    body = chunk_json[15:].split(":", 1)[1] if ":" in chunk_json[15:] else ""
                    log.error("API error %d: %s", status, body[:200])
                    yield {"error": True, "status": status, "body": body}
                    break
                # Parse SSE line
                try:
                    data = json.loads(chunk_json)
                    yield data
                except json.JSONDecodeError:
                    continue
        except asyncio.TimeoutError:
            log.error("Stream timeout (%ss) for request %s", stream_timeout, request_id)
            yield {"error": True, "status": 0, "body": "Stream timeout"}
        finally:
            self._stream_queues.pop(request_id, None)
            if not eval_task.done():
                eval_task.cancel()
            else:
                # Check for exceptions
                try:
                    eval_task.result()
                except Exception:
                    pass

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return default
        return number if number > 0 else default

    @staticmethod
    def _build_chat_attachment_blocks(
        image_attachments: Optional[List[Dict[str, Any]]] = None,
        attachment_type: int = 2,
    ) -> List[Dict[str, Any]]:
        """Build Doubao chat attachment blocks with visible image metadata."""
        attachments: List[Dict[str, Any]] = []
        for image in image_attachments or []:
            uri = str(
                image.get("uri")
                or image.get("key")
                or image.get("fileKey")
                or image.get("file_key")
                or ""
            ).strip()
            if not uri:
                continue
            name = str(image.get("name") or image.get("fileName") or "image.png")
            fmt = str(image.get("format") or (name.rsplit(".", 1)[-1] if "." in name else "png"))
            cdn_url = str(
                image.get("cdn_url")
                or image.get("url")
                or image.get("main_url")
                or ""
            )
            width = BrowserClient._safe_int(image.get("width"), 64)
            height = BrowserClient._safe_int(image.get("height"), 64)
            attachments.append({
                "type": attachment_type,
                "identifier": str(image.get("identifier") or uuid.uuid4()),
                "image": {
                    "name": name,
                    "uri": uri,
                    "key": uri,
                    "image_ori": {
                        "url": cdn_url,
                        "width": width,
                        "height": height,
                        "format": fmt,
                        "url_formats": {},
                    },
                },
                "parse_state": BrowserClient._safe_int(image.get("parse_state"), 0),
                "review_state": BrowserClient._safe_int(image.get("review_state"), 0),
                "upload_status": 1,
                "progress": 100,
                "src": cdn_url,
            })
            if attachment_type != 1:
                attachments[-1]["extra"] = {"refer_types": "overall"}

        if not attachments:
            return []
        return [{
                "block_type": 10052,
                "content": {
                    "attachment_block": {"attachments": attachments},
                    "pc_event_block": "",
                },
                "block_id": str(uuid.uuid4()),
                "parent_id": "",
                "meta_info": [],
                "append_fields": [],
                "is_finish": True,
                "patch_type": 2,
        }]

    @staticmethod
    def _build_chat_text_blocks(text: str) -> List[Dict[str, Any]]:
        return [{
            "block_type": 10000,
            "content": {
                "text_block": {
                    "text": text,
                    "icon_url": "",
                    "icon_url_dark": "",
                    "summary": "",
                },
                "pc_event_block": "",
            },
            "block_id": str(uuid.uuid4()),
            "parent_id": "",
            "meta_info": [],
            "append_fields": [],
            "is_finish": True,
            "patch_type": 2,
        }]

    @staticmethod
    def _build_chat_content_blocks(
        text: str,
        image_attachments: Optional[List[Dict[str, Any]]] = None,
        attachment_type: int = 2,
    ) -> List[Dict[str, Any]]:
        """Build Doubao chat content blocks with visible image attachments.

        Doubao's current web UI persists images as BLOCK_ATTACHMENT entries.
        Passing only Samantha ``attachments`` or only a hidden ``ref_image_key``
        can make the downstream video skill receive a text-only message.
        """
        return (
            BrowserClient._build_chat_attachment_blocks(
                image_attachments,
                attachment_type=attachment_type,
            )
            + BrowserClient._build_chat_text_blocks(text)
        )

    async def _browser_fetch_stream(
        self, url: str, payload: Dict[str, Any], request_id: str
    ):
        """Execute fetch() inside browser page and push parsed SSE chunks.

        The original implementation relied on ``page.expose_function`` and a
        window callback. Doubao's SPA can replace the execution world after
        login or route changes, leaving that callback undefined. Reading the
        upstream stream directly inside ``page.evaluate`` is less real-time, but
        it is much more reliable and still preserves event order.
        """
        js_code = """
        async ([url, payloadJson, requestId]) => {
            const chunks = [];
            try {
                const csrf = document.cookie.match(/passport_csrf_token=([^;]+)/);
                const csrfToken = csrf ? csrf[1] : '';
                const headers = {
                    'Content-Type': 'application/json',
                    'agw-js-conv': 'str, str',
                };
                if (csrfToken) {
                    headers['x-tt-passport-csrf-token'] = csrfToken;
                }
                const res = await fetch(url, {
                    method: 'POST',
                    headers: headers,
                    body: payloadJson,
                    credentials: 'include',
                });
                if (!res.ok) {
                    const errBody = await res.text();
                    chunks.push('__HTTP_ERROR__:' + res.status + ':' + errBody.slice(0, 500));
                    return chunks;
                }
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let currentEvent = '';
                let buffer = '';
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, {stream: true});
                    const lines = buffer.split('\\n');
                    buffer = lines.pop();
                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed) continue;
                        if (trimmed.startsWith('event: ')) {
                            currentEvent = trimmed.slice(7);
                            continue;
                        }
                        if (trimmed.startsWith('id: ')) continue;
                        if (!trimmed.startsWith('data: ')) continue;
                        const dataStr = trimmed.slice(6);
                        if (!dataStr || dataStr === '{}') continue;
                        try {
                            const obj = JSON.parse(dataStr);
                            obj._event = currentEvent;
                            chunks.push(JSON.stringify(obj));
                        } catch(e) {}
                    }
                }
                // Process remaining buffer
                if (buffer.trim()) {
                    const trimmed = buffer.trim();
                    if (trimmed.startsWith('data: ')) {
                        const dataStr = trimmed.slice(6);
                        if (dataStr && dataStr !== '{}') {
                            try {
                                const obj = JSON.parse(dataStr);
                                obj._event = currentEvent;
                                chunks.push(JSON.stringify(obj));
                            } catch(e) {}
                        }
                    }
                }
                return chunks;
            } catch(e) {
                chunks.push('__ERROR__:' + e.message);
                return chunks;
            }
        }
        """
        payload_json = json.dumps(payload, ensure_ascii=False)
        try:
            chunks = await self._page.evaluate(js_code, [url, payload_json, request_id])
            if chunks:
                for chunk in chunks:
                    await self._stream_queues[request_id].put(chunk)
        finally:
            queue = self._stream_queues.get(request_id)
            if queue:
                await queue.put(None)

    # ------------------------------------------------------------------
    # High-level chat helper
    # ------------------------------------------------------------------

    async def chat(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        """Send message, collect full response. Returns {text, conversation_id}."""
        full_text = ""
        result_conv_id = conversation_id
        events = []

        async for event in self.chat_completion(
            text, conversation_id=conversation_id,
            bot_id=bot_id, use_deep_think=use_deep_think
        ):
            events.append(event)
            if event.get("error"):
                raise RuntimeError(
                    f"API error {event.get('status')}: {event.get('body', '')[:200]}"
                )
            if not result_conv_id:
                cid = self.extract_conversation_id(event)
                if cid and cid != "0":
                    result_conv_id = cid
            full_text += self._extract_text(event)

        return {"text": full_text, "conversation_id": result_conv_id}

    # ------------------------------------------------------------------
    # SSE parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(event: Dict[str, Any]) -> str:
        """Extract text content from a SSE event."""
        event_type = event.get("_event", "")

        if event_type == "CHUNK_DELTA" and "text" in event:
            return event["text"]

        if "patch_op" in event:
            for op in event["patch_op"]:
                pv = op.get("patch_value", {})
                for block in pv.get("content_block", []):
                    content = block.get("content", {})
                    tb = content.get("text_block", {})
                    if tb.get("text"):
                        return tb["text"]
                if op.get("patch_object") == 102:
                    raw = pv.get("content", "")
                    if raw:
                        try:
                            parsed = json.loads(raw)
                            if parsed.get("text"):
                                return parsed["text"]
                        except (json.JSONDecodeError, TypeError):
                            pass

        if event_type == "STREAM_MSG_NOTIFY":
            content = event.get("content", {})
            if isinstance(content, dict):
                for block in content.get("content_block", []):
                    tb = block.get("content", {}).get("text_block", {})
                    if tb.get("text"):
                        return tb["text"]

        return ""

    @staticmethod
    def extract_conversation_id(event: Dict[str, Any]) -> Optional[str]:
        """Extract conversation_id from SSE events."""
        ack = event.get("ack_client_meta", {})
        if ack.get("conversation_id"):
            return ack["conversation_id"]
        meta = event.get("meta", {})
        if meta.get("conversation_id"):
            return meta["conversation_id"]
        found = BrowserClient._find_samantha_conversation_id(event)
        return found or None

    # ------------------------------------------------------------------
    # Samantha endpoint (image/video/music generation)
    # ------------------------------------------------------------------

    async def _samantha_request(
        self,
        payload: Dict[str, Any],
        timeout: float = 120,
    ) -> str:
        """Send a request to /samantha/chat/completion via in-browser fetch."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        query_params = self._build_query_params()
        query_string = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))
        url = f"/samantha/chat/completion?{query_string}"

        js_code = """
        async ([url, payloadJson, timeoutMs]) => {
            const csrf = document.cookie.match(/passport_csrf_token=([^;]+)/);
            const csrfToken = csrf ? csrf[1] : '';
            const headers = {
                'Content-Type': 'application/json',
                'agw-js-conv': 'str',
            };
            if (csrfToken) {
                headers['x-tt-passport-csrf-token'] = csrfToken;
            }
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), timeoutMs);
            try {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: headers,
                    body: payloadJson,
                    credentials: 'include',
                    signal: controller.signal,
                });
                clearTimeout(timer);
                if (!res.ok) {
                    const errBody = await res.text();
                    return {error: true, status: res.status, body: errBody.slice(0, 500)};
                }
                const body = await res.text();
                return {error: false, body: body};
            } catch(e) {
                clearTimeout(timer);
                return {error: true, status: 0, body: e.message};
            }
        }
        """
        payload_json = json.dumps(payload, ensure_ascii=False)
        timeout_ms = int(timeout * 1000)

        log.info("POST %s [browser fetch, timeout=%ds]", url.split("?")[0], timeout)
        result = await self._page.evaluate(
            js_code, [url, payload_json, timeout_ms]
        )

        if result.get("error"):
            status = result.get("status", 0)
            body = result.get("body", "")
            raise RuntimeError(
                f"samantha/chat/completion failed ({status}): {body[:500]}"
            )

        body = result.get("body", "")
        debug_dir = os.environ.get("DOUBAO_SAMANTHA_DEBUG_DIR", "").strip()
        if debug_dir:
            try:
                path = Path(debug_dir)
                path.mkdir(parents=True, exist_ok=True)
                name = f"samantha-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.json"
                (path / name).write_text(
                    json.dumps(
                        {
                            "url": url,
                            "payload": payload,
                            "body": body,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                log.warning("failed to write Samantha debug dump: %s", exc)
        if body.lstrip().startswith("{"):
            try:
                err = json.loads(body)
                if isinstance(err, dict) and "code" in err:
                    raise RuntimeError(
                        f"samantha auth error: code={err.get('code')} "
                        f"msg={err.get('msg') or err.get('message', '')}"
                    )
            except json.JSONDecodeError:
                pass
        return body

    async def _samantha_async_stream_request(
        self,
        payload: Dict[str, Any],
        timeout: float = 120,
    ) -> str:
        """Send a request to /samantha/chat/async/stream via in-browser fetch."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        query_params = self._build_query_params()
        query_params.pop("fp", None)
        query_string = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))
        url = f"/samantha/chat/async/stream?{query_string}"

        js_code = """
        async ([url, payloadJson, timeoutMs]) => {
            const csrf = document.cookie.match(/passport_csrf_token=([^;]+)/);
            const csrfToken = csrf ? csrf[1] : '';
            const headers = {
                'Accept': 'text/event-stream',
                'Content-Type': 'application/json',
                'Agw-Js-Conv': 'str',
            };
            if (csrfToken) {
                headers['x-tt-passport-csrf-token'] = csrfToken;
            }
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), timeoutMs);
            try {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: headers,
                    body: payloadJson,
                    credentials: 'include',
                    signal: controller.signal,
                });
                clearTimeout(timer);
                const body = await res.text();
                if (!res.ok) {
                    return {error: true, status: res.status, body: body.slice(0, 500)};
                }
                return {error: false, body: body};
            } catch(e) {
                clearTimeout(timer);
                return {error: true, status: 0, body: e.message};
            }
        }
        """
        payload_json = json.dumps(payload, ensure_ascii=False)
        timeout_ms = int(timeout * 1000)

        log.info("POST %s [browser fetch, timeout=%ds]", url.split("?")[0], timeout)
        result = await self._page.evaluate(js_code, [url, payload_json, timeout_ms])

        if result.get("error"):
            status = result.get("status", 0)
            body = result.get("body", "")
            raise RuntimeError(
                f"samantha/chat/async/stream failed ({status}): {body[:500]}"
            )

        body = result.get("body", "")
        debug_dir = os.environ.get("DOUBAO_SAMANTHA_DEBUG_DIR", "").strip()
        if debug_dir:
            try:
                path = Path(debug_dir)
                path.mkdir(parents=True, exist_ok=True)
                name = f"samantha-async-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.json"
                (path / name).write_text(
                    json.dumps(
                        {
                            "url": url,
                            "payload": payload,
                            "body": body,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                log.warning("failed to write Samantha async debug dump: %s", exc)
        return body

    async def _browser_post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        timeout: float = 30,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """POST JSON from inside the logged-in browser context."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")
        if not self._page:
            raise RuntimeError("Browser page not ready")

        query_params = self._build_query_params()
        separator = "&" if "?" in path else "?"
        url = f"{path}{separator}{urlencode(query_params)}"
        timeout_ms = int(timeout * 1000)
        payload_json = json.dumps(payload, ensure_ascii=False)
        headers_json = json.dumps(extra_headers or {}, ensure_ascii=False)

        js_code = """
        async ([url, payloadJson, timeoutMs, headersJson]) => {
            const csrf = document.cookie.match(/passport_csrf_token=([^;]+)/);
            const csrfToken = csrf ? csrf[1] : '';
            const extraHeaders = JSON.parse(headersJson || '{}');
            const headers = {
                'Content-Type': 'application/json',
                'Agw-Js-Conv': 'str',
                ...extraHeaders,
            };
            if (csrfToken) {
                headers['x-tt-passport-csrf-token'] = csrfToken;
            }
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), timeoutMs);
            try {
                const res = await fetch(url, {
                    method: 'POST',
                    headers,
                    body: payloadJson,
                    credentials: 'include',
                    signal: controller.signal,
                });
                clearTimeout(timer);
                const text = await res.text();
                let body = null;
                try { body = text ? JSON.parse(text) : {}; }
                catch { body = {raw: text}; }
                return {ok: res.ok, status: res.status, body, text};
            } catch (e) {
                clearTimeout(timer);
                return {ok: false, status: 0, body: {message: String(e && e.message || e)}, text: ''};
            }
        }
        """
        log.info("POST %s [browser json, timeout=%ds]", path.split("?")[0], timeout)
        result = await self._page.evaluate(
            js_code, [url, payload_json, timeout_ms, headers_json]
        )
        body = result.get("body") or {}
        if not result.get("ok"):
            raise RuntimeError(
                f"{path} failed ({result.get('status', 0)}): "
                f"{(result.get('text') or json.dumps(body, ensure_ascii=False))[:500]}"
            )
        if isinstance(body, dict) and int(body.get("code") or 0) != 0:
            raise RuntimeError(
                f"{path} returned code={body.get('code')}: "
                f"{body.get('message') or body.get('msg') or body.get('error') or ''}"
            )
        return body

    async def get_credit_quota(self) -> Dict[str, Any]:
        """Fetch Doubao's own credit panel data for the current account."""
        credit_body = await self._browser_post_json(
            "/commerce/benefit_supply/credit/get_credit_num_optional_tasks",
            {"need_tasks": True},
            timeout=30,
        )
        history_body = await self._browser_post_json(
            "/commerce/benefit_supply/credit/get_credit_usage_history",
            {"size": 20, "transaction_type": 0},
            timeout=30,
        )
        credit_data = credit_body.get("data") or {}
        history_data = history_body.get("data") or {}
        return {
            "source": "doubao_credit_api",
            "synced_at": int(time.time()),
            "credit_info": credit_data.get("credit_info") or {},
            "optional_tasks": credit_data.get("optional_tasks") or [],
            "credit_rule_link": credit_data.get("credit_rule_link") or {},
            "usage_history": history_data,
        }

    @staticmethod
    def _parse_samantha_sse(raw: str) -> List[Dict[str, Any]]:
        """Parse samantha SSE body into list of event dicts."""
        events = []
        for block in raw.split("\n\n"):
            if not block.strip():
                continue
            data_str = ""
            for line in block.strip().split("\n"):
                if line.startswith("data:"):
                    data_str = line[5:].strip()
            if not data_str:
                continue
            try:
                events.append(json.loads(data_str))
            except json.JSONDecodeError:
                continue
        return events

    @staticmethod
    def _find_samantha_conversation_id(value: Any) -> str:
        """Find a conversation_id anywhere in a Samantha SSE event payload."""
        return BrowserClient._find_samantha_id(
            value,
            exact_keys={"conversation_id", "conversationid", "thread_id", "threadid"},
            fuzzy_terms=("conversation", "thread"),
            skip_terms=("local",),
            url_patterns=(r"/chat/(\d{8,})", r"conversation_id[=:\"']+(\d{8,})"),
        )

    @staticmethod
    def _find_samantha_task_id(value: Any) -> str:
        """Find an async task ID anywhere in a Samantha SSE event payload."""
        return BrowserClient._find_samantha_id(
            value,
            exact_keys={"task_id", "taskid", "async_task_id", "asynctaskid", "video_task_id", "videotaskid"},
            fuzzy_terms=("task",),
            skip_terms=("local",),
            url_patterns=(r"(?:task_id|taskId)[=:\"']+([A-Za-z0-9_-]{8,})",),
        )

    @staticmethod
    def _find_samantha_id(
        value: Any,
        *,
        exact_keys: set[str],
        fuzzy_terms: tuple[str, ...],
        skip_terms: tuple[str, ...] = (),
        url_patterns: tuple[str, ...] = (),
    ) -> str:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                for pattern in url_patterns:
                    match = re.search(pattern, value)
                    if match:
                        return str(match.group(1))
                return ""
        if isinstance(value, dict):
            for key, candidate in value.items():
                key_norm = str(key).replace("-", "_").replace(" ", "_").lower()
                compact = key_norm.replace("_", "")
                if any(term in key_norm or term in compact for term in skip_terms):
                    continue
                key_matches = compact in exact_keys or key_norm in exact_keys
                key_matches = key_matches or (
                    "id" in compact
                    and any(term in compact for term in fuzzy_terms)
                )
                if key_matches and candidate and str(candidate) != "0":
                    return str(candidate)
            for item in value.values():
                found = BrowserClient._find_samantha_id(
                    item,
                    exact_keys=exact_keys,
                    fuzzy_terms=fuzzy_terms,
                    skip_terms=skip_terms,
                    url_patterns=url_patterns,
                )
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = BrowserClient._find_samantha_id(
                    item,
                    exact_keys=exact_keys,
                    fuzzy_terms=fuzzy_terms,
                    skip_terms=skip_terms,
                    url_patterns=url_patterns,
                )
                if found:
                    return found
        return ""

    async def generate_image(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        ref_image_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate images using /samantha/chat/completion.

        Args:
            prompt: Text description of the image to generate.
            ratio: Aspect ratio ("1:1", "16:9", "9:16", "4:3", "3:4").
            ref_image_key: Optional uploaded image key for reference.

        Returns:
            Dict with 'images' list, each having url/width/height/key.
        """
        content_data: Dict[str, Any] = {"text": prompt}
        if ratio:
            content_data["ratio"] = ratio

        message: Dict[str, Any] = {
            "content": json.dumps(content_data, ensure_ascii=False),
            "content_type": 2009,
            "attachments": [],
            "references": [],
            "skill": {
                "skill_type": 3,
                "skill_type_no_default": 3,
                "skill_id": "3",
                "skill_id_no_default": "3",
            },
        }

        if ref_image_key:
            message["attachments"] = [
                {"type": "image", "key": ref_image_key,
                 "extra": {"refer_types": "overall"}}
            ]

        payload = {
            "messages": [message],
            "completion_option": {
                "is_regen": False,
                "with_suggest": True,
                "need_create_conversation": True,
                "launch_stage": 1,
                "is_replace": False,
                "is_delete": False,
                "is_ai_playground": False,
                "memory_type": 2,
                "message_from": 0,
                "use_deep_think": False,
                "use_auto_cot": False,
                "resend_for_regen": False,
                "enable_commerce_credit": False,
                "action_bar_skill_id": 3,
            },
            "evaluate_option": {"web_ab_params": ""},
            "local_conversation_id": str(uuid.uuid4()),
            "local_message_id": str(uuid.uuid4()),
        }

        log.info("generate_image: prompt=%s, ratio=%s", prompt[:50], ratio)
        raw = await self._samantha_request(payload, timeout=120)

        # Parse response - look for content_type=2010 (image output)
        images = []
        for data in self._parse_samantha_sse(raw):
            et = data.get("event_type")
            if et == 2005:
                detail = data.get("event_data", "")
                raise RuntimeError(f"generate_image error: {str(detail)[:500]}")
            if et != 2001:
                continue

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue

            if msg.get("content_type") != 2010:
                continue

            content_raw = msg.get("content", "")
            if isinstance(content_raw, str):
                try:
                    content = json.loads(content_raw)
                except json.JSONDecodeError:
                    continue
            else:
                content = content_raw

            for item in content.get("data", []):
                if not isinstance(item, dict):
                    continue
                ori = item.get("image_ori", {}) or {}
                raw_img = item.get("image_raw", {}) or {}
                thumb = item.get("image_thumb", {}) or {}
                images.append({
                    "key": item.get("key", ""),
                    "url": ori.get("url") or raw_img.get("url") or thumb.get("url", ""),
                    "width": ori.get("width") or thumb.get("width", 0),
                    "height": ori.get("height") or thumb.get("height", 0),
                    "format": ori.get("format") or thumb.get("format", ""),
                })

        log.info("generate_image: got %d images", len(images))
        return {"images": images, "prompt": prompt}

    async def generate_music(
        self,
        prompt: str,
        lyric: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate music using /samantha/chat/completion.

        Args:
            prompt: Text description of the music to generate.
            lyric: Explicit lyrics (optional).
            genre: Music genre (optional).

        Returns:
            Dict with 'tracks' list, each having audio_url/title/lyrics/duration.
        """
        import base64

        content_data: Dict[str, Any] = {"text": prompt}
        if lyric:
            content_data["lyric"] = lyric
        if genre:
            content_data["genre"] = genre

        message: Dict[str, Any] = {
            "content": json.dumps(content_data, ensure_ascii=False),
            "content_type": 2005,
            "attachments": [],
            "references": [],
            "skill": {
                "skill_type": 9,
                "skill_type_no_default": 9,
                "skill_id": "9",
                "skill_id_no_default": "9",
            },
        }

        payload = {
            "messages": [message],
            "completion_option": {
                "is_regen": False,
                "with_suggest": True,
                "need_create_conversation": True,
                "launch_stage": 1,
                "is_replace": False,
                "is_delete": False,
                "is_ai_playground": False,
                "memory_type": 2,
                "message_from": 0,
                "use_deep_think": False,
                "use_auto_cot": False,
                "resend_for_regen": False,
                "enable_commerce_credit": False,
                "action_bar_skill_id": 9,
            },
            "evaluate_option": {"web_ab_params": ""},
            "local_conversation_id": str(uuid.uuid4()),
            "local_message_id": str(uuid.uuid4()),
        }

        log.info("generate_music: prompt=%s", prompt[:50])
        raw = await self._samantha_request(payload, timeout=300)

        # Parse: find last content_type=2006 with video_model
        tracks = []
        final_content = None
        for data in self._parse_samantha_sse(raw):
            et = data.get("event_type")
            if et == 2005:
                detail = data.get("event_data", "")
                raise RuntimeError(f"generate_music error: {str(detail)[:500]}")
            if et != 2001:
                continue

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue

            if msg.get("content_type") not in (2006, 2004):
                continue

            content_raw = msg.get("content", "")
            if isinstance(content_raw, str):
                try:
                    content = json.loads(content_raw)
                except json.JSONDecodeError:
                    continue
            else:
                content = content_raw

            # Keep updating - we want the final (most complete) version
            final_content = content

        if not final_content:
            log.warning("generate_music: no content_type=2006 found")
            return {"tracks": [], "prompt": prompt}

        # Parse tasks
        tasks = final_content.get("tasks", {})
        if isinstance(tasks, dict):
            tasks_list = list(tasks.values())
        elif isinstance(tasks, list):
            tasks_list = tasks
        else:
            tasks_list = []

        for task in tasks_list:
            if not isinstance(task, dict):
                continue

            audio_url = ""
            duration = 0.0
            vm_str = task.get("video_model", "")
            if vm_str:
                try:
                    vm = json.loads(vm_str) if isinstance(vm_str, str) else vm_str
                    duration = vm.get("video_duration", 0.0)
                    vlist = vm.get("video_list", {})
                    for _q, vinfo in vlist.items():
                        main_url_b64 = vinfo.get("main_url", "")
                        if main_url_b64:
                            audio_url = base64.b64decode(main_url_b64).decode(
                                "utf-8", errors="replace"
                            )
                            break
                except (json.JSONDecodeError, Exception):
                    pass

            cover_url = ""
            cover = task.get("cover", {})
            if isinstance(cover, dict):
                cover_ori = cover.get("image_ori", {}) or {}
                cover_url = cover_ori.get("url", "")

            if audio_url or task.get("title"):
                tracks.append({
                    "audio_url": audio_url,
                    "title": task.get("title", ""),
                    "lyrics": task.get("lyric", ""),
                    "duration": duration,
                    "cover_url": cover_url,
                })

        log.info("generate_music: got %d tracks", len(tracks))
        return {"tracks": tracks, "prompt": prompt}

    @staticmethod
    def _normalize_reference_image_keys(
        ref_image_key: Optional[str],
        reference_image_keys: Optional[List[str]],
    ) -> List[str]:
        keys: List[str] = []
        if ref_image_key:
            keys.append(str(ref_image_key))
        for key in reference_image_keys or []:
            if key:
                keys.append(str(key))
        seen = set()
        result: List[str] = []
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    @staticmethod
    def _normalize_reference_image_attachments(
        ref_image_key: Optional[str],
        reference_image_keys: Optional[List[str]],
        reference_image_infos: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize uploaded image metadata for visible chat attachments."""
        attachments: List[Dict[str, Any]] = []
        seen = set()
        for info in reference_image_infos or []:
            if not isinstance(info, dict):
                continue
            uri = str(
                info.get("uri")
                or info.get("key")
                or info.get("fileKey")
                or info.get("file_key")
                or ""
            ).strip()
            if not uri or uri in seen:
                continue
            seen.add(uri)
            attachment = dict(info)
            attachment["uri"] = uri
            attachments.append(attachment)

        for key in BrowserClient._normalize_reference_image_keys(ref_image_key, reference_image_keys):
            if key in seen:
                continue
            seen.add(key)
            attachments.append({
                "uri": key,
                "name": "reference.png",
                "format": "png",
                "width": 64,
                "height": 64,
            })
        return attachments

    @staticmethod
    def _build_video_message(
        prompt: str,
        ratio: Optional[str],
        image_keys: List[str],
        model: Optional[str],
        duration: Optional[int],
        image_attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        content_data: Dict[str, Any] = {"text": prompt}
        if ratio:
            content_data["ratio"] = ratio
        if image_keys:
            content_data["ref_image_key"] = image_keys[0]
            content_data["ref_image_keys"] = image_keys
            content_data["reference_image_keys"] = image_keys
            content_data["reference_images"] = [{"key": key, "type": "image"} for key in image_keys]
            ref_images: List[Dict[str, Any]] = []
            by_uri: Dict[str, Dict[str, Any]] = {}
            for item in image_attachments or []:
                uri = str(item.get("uri") or item.get("key") or "").strip()
                if uri:
                    by_uri[uri] = item
            for key in image_keys:
                info = by_uri.get(key, {})
                ref_images.append({
                    "image_token": key,
                    "url": info.get("cdn_url") or info.get("url") or info.get("main_url") or "",
                    "refer_types": "overall",
                    "review_status": BrowserClient._safe_int(info.get("review_status"), 1),
                    "identifier": str(info.get("identifier") or uuid.uuid4()),
                    "file_name": info.get("name") or "reference.png",
                    "image_ori": {
                        "url": info.get("cdn_url") or info.get("url") or info.get("main_url") or "",
                        "width": BrowserClient._safe_int(info.get("width"), 64),
                        "height": BrowserClient._safe_int(info.get("height"), 64),
                        "format": info.get("format") or "png",
                    },
                })
            content_data["samantha_context"] = {
                "query_context": {
                    "ref_images": ref_images,
                }
            }
        if model:
            content_data["model"] = model
        if duration:
            content_data["duration"] = int(duration)

        message: Dict[str, Any] = {
            "content": json.dumps(content_data, ensure_ascii=False),
            "content_type": 2020,
            "attachments": [],
            "references": [],
            "skill": {
                "skill_type": 17,
                "skill_type_no_default": 17,
                "skill_id": "17",
                "skill_id_no_default": "17",
            },
        }
        if image_keys:
            message["attachments"] = [
                {
                    "type": "image",
                    "key": key,
                    "url": (by_uri.get(key, {}).get("cdn_url") or by_uri.get(key, {}).get("url") or ""),
                    "extra": {"refer_types": "overall"},
                    "identifier": str(by_uri.get(key, {}).get("identifier") or uuid.uuid4()),
                }
                for key in image_keys
            ]
        return message

    async def generate_video_web(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        ref_image_key: Optional[str] = None,
        reference_image_keys: Optional[List[str]] = None,
        reference_image_infos: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        duration: Optional[int] = None,
        wait_for_result: bool = True,
        ui_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate video through Doubao's current web chat ability path."""
        import base64
        import re

        image_attachments = self._normalize_reference_image_attachments(
            ref_image_key,
            reference_image_keys,
            reference_image_infos,
        )

        ability_param: Dict[str, Any] = {
            "model": model or os.environ.get("DOUBAO_VIDEO_MODEL", "seedance_v2.0"),
            "duration": int(duration or os.environ.get("DOUBAO_VIDEO_DURATION", "10")),
        }
        if ratio:
            ability_param["ratio"] = ratio

        chat_ability = {
            "ability_type": 17,
            "ability_param": json.dumps(ability_param, ensure_ascii=False),
        }
        video_prefix = "\u751f\u6210\u89c6\u9891"
        text_prompt = prompt if prompt.strip().startswith(video_prefix) else f"{video_prefix}\uff1a{prompt}"

        videos: List[Dict[str, Any]] = []
        text_parts: List[str] = []

        def maybe_json(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

        def maybe_b64_url(value: Any) -> str:
            if not isinstance(value, str) or len(value) < 16:
                return ""
            try:
                decoded = base64.b64decode(value + "=" * (-len(value) % 4)).decode("utf-8", errors="ignore")
            except Exception:
                return ""
            return decoded if decoded.startswith(("http://", "https://")) else ""

        def add_video(url: str, item: Optional[Dict[str, Any]] = None) -> None:
            if not url or not url.startswith(("http://", "https://")):
                return
            if re.search(r"\.(png|jpe?g|webp|gif)(~|\?|$)", url, re.I):
                return
            if not re.search(r"(\.mp4|\.m3u8|/video/|video)", url, re.I):
                return
            if any(v["video_url"] == url for v in videos):
                return
            item = item or {}
            cover = item.get("cover", {})
            videos.append({
                "video_url": url,
                "cover_url": item.get("cover_url", "") or item.get("poster", "") or (cover.get("url", "") if isinstance(cover, dict) else ""),
                "width": item.get("width", 0) or item.get("video_width", 0),
                "height": item.get("height", 0) or item.get("video_height", 0),
                "duration": item.get("duration", 0.0) or item.get("video_duration", 0.0),
            })

        def walk(value: Any, parent: Optional[Dict[str, Any]] = None) -> None:
            value = maybe_json(value)
            if isinstance(value, dict):
                text_block = value.get("text_block")
                if isinstance(text_block, dict) and isinstance(text_block.get("text"), str):
                    text_parts.append(text_block["text"])
                content = maybe_json(value.get("content"))
                if isinstance(content, dict):
                    if isinstance(content.get("text"), str):
                        text_parts.append(content["text"])
                    walk(content, value)
                for key in ("video_url", "url", "main_url", "play_url", "download_url"):
                    raw = value.get(key)
                    if isinstance(raw, str):
                        add_video(raw, value)
                        add_video(maybe_b64_url(raw), value)
                vm = maybe_json(value.get("video_model"))
                if isinstance(vm, dict):
                    walk(vm, value)
                for item in value.values():
                    if item is not content and item is not vm:
                        walk(item, value)
            elif isinstance(value, list):
                for item in value:
                    walk(item, parent)

        def fix_mojibake(text: str) -> str:
            if not text or not any(marker in text for marker in ("鐢", "瑙", "鏈", "杩", "鍔", "棰")):
                return text
            if "鐢熸垚瑙嗛" in text and any(marker in text for marker in ("鍔¤繃杞", "璇风", "鍚庨噸")):
                return "豆包视频生成已触发，但上游返回：服务过载，请稍后重试。"
            for encoding in ("gb18030", "gbk"):
                try:
                    fixed = text.encode(encoding).decode("utf-8")
                except Exception:
                    continue
                if any(marker in fixed for marker in ("生成", "视频", "服务", "重试", "额度")):
                    return fixed
            if any(marker in text for marker in ("杩囪", "繃杞", "璇风", "重璇")):
                return "豆包视频生成已触发，但上游返回：服务过载，请稍后重试。"
            return text

        log.info("generate_video_web: prompt=%s, ability=%s", prompt[:50], ability_param)
        conversation_id = ""
        async for event in self.chat_completion(
            text_prompt,
            use_deep_think=0,
            chat_ability=chat_ability,
            image_attachments=image_attachments,
            split_image_attachments=bool(image_attachments),
            attachment_type=1 if image_attachments else 2,
            stream_timeout=float(os.environ.get("DOUBAO_VIDEO_TIMEOUT", "420")),
        ):
            if event.get("error"):
                raise RuntimeError(
                    f"API error {event.get('status')}: {event.get('body', '')[:500]}"
                )
            if not conversation_id:
                cid = self.extract_conversation_id(event)
                if cid and cid != "0":
                    conversation_id = cid
            if isinstance(event.get("text"), str):
                text_parts.append(event["text"])
            walk(event)

        full_text = fix_mojibake("".join(text_parts).strip())
        conversation_url = self._page.url if self._page else ""
        if not conversation_id and conversation_url:
            match = re.search(r"/chat/(\d{8,})", conversation_url)
            if match:
                conversation_id = match.group(1)
        log.info("generate_video_web: got %d videos; text=%s", len(videos), full_text[:120])
        if not videos and self._is_video_acceptance_text(full_text):
            if wait_for_result:
                ui_result = await self._wait_for_video_result_from_ui(
                    prompt,
                    conversation_id=conversation_id or None,
                    conversation_url=conversation_url or None,
                    timeout=float(ui_timeout if ui_timeout is not None else os.environ.get("DOUBAO_VIDEO_UI_WAIT_SECONDS", "360")),
                )
                videos.extend(ui_result.get("videos", []))
                conversation_id = conversation_id or str(ui_result.get("conversation_id") or "")
                conversation_url = conversation_url or str(ui_result.get("conversation_url") or "")
        result = {
            "videos": videos,
            "prompt": prompt,
            "message": full_text,
            "conversation_id": conversation_id,
            "conversation_url": conversation_url,
        }
        if not videos and self._is_video_acceptance_text(full_text):
            result["pending"] = True
            result["accepted"] = True
        return result

    @staticmethod
    def _is_video_terminal_failure_text(text: str) -> bool:
        lowered = (text or "").lower()
        if "视频生成" in lowered and any(marker in lowered for marker in (
            "今日视频生成免费次数已用完",
            "视频生成免费次数已用完",
            "免费次数已用完",
            "次数已用完",
            "即可继续使用视频生成",
            "继续使用视频生成",
            "开通豆包专业版",
            "开通加强套餐",
            "开通套餐",
        )):
            return True
        return any(marker in lowered for marker in (
            "\u79ef\u5206\u4e0d\u8db3",
            "\u4f59\u989d\u4e0d\u8db3",
            "\u6743\u76ca\u4e0d\u8db3",
            "\u6ca1\u6709\u76f8\u5173\u6743\u76ca",
            "\u6ca1\u6709\u89c6\u9891\u751f\u6210\u6743\u76ca",
            "\u989d\u5ea6\u4e0d\u8db3",
            "\u989d\u5ea6\u5df2\u7528\u5b8c",
            "\u89c6\u9891\u751f\u6210\u989d\u5ea6\u5df2\u7528\u5b8c",
            "积分不足",
            "余额不足",
            "权益不足",
            "没有相关权益",
            "没有视频生成权益",
            "额度不足",
            "额度已用完",
            "视频生成额度已用完",
            "quota exceeded",
            "quota exhausted",
            "quota insufficient",
            "quota limit",
        ))

    @staticmethod
    def _is_video_acceptance_text(text: str) -> bool:
        if BrowserClient._is_video_terminal_failure_text(text):
            return False
        lowered = (text or "").lower()
        return any(marker in lowered for marker in (
            "\u6b63\u5728\u4e3a\u60a8\u751f\u6210\u89c6\u9891",
            "\u89c6\u9891\u751f\u6210\u597d\u540e",
            "\u751f\u6210\u597d\u540e",
            "\u9884\u8ba1\u7b49\u5f85",
            "\u6b63\u5728\u751f\u6210",
            "正在为您生成视频",
            "视频生成好后",
            "生成好后",
            "预计等待",
            "generating video",
            "will notify",
        ))

    async def _wait_for_video_result_from_ui(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        conversation_url: Optional[str] = None,
        timeout: float = 360,
    ) -> Dict[str, Any]:
        """Wait for Doubao's web UI notification card and extract its video URL."""
        import re

        if not self._page:
            return {
                "videos": [],
                "prompt": prompt,
                "conversation_id": conversation_id,
                "conversation_url": conversation_url,
            }
        bound_url = conversation_url or self._page.url
        if not conversation_id and not bound_url:
            log.info("generate_video_web: skip UI recovery without conversation binding")
            return {
                "videos": [],
                "prompt": prompt,
                "conversation_id": conversation_id,
                "conversation_url": conversation_url,
            }

        video_regex = re.compile(r"(mp4|m3u8|douyinvod|mime_type=video_mp4|video_gen)", re.I)
        prompt_snippet = (prompt or "").strip()[:48]
        require_prompt_match = not bool(conversation_id)
        deadline = time.time() + timeout
        visited: set[str] = set()

        async def extract_current_page() -> List[Dict[str, Any]]:
            result = await self._page.evaluate(
                """async () => {
                    const videoPattern = /mp4|m3u8|douyinvod|mime_type=video_mp4|video_gen/i;
                    const coverPattern = /video_dsz|video.*watermark|tos-cn-p|douyinvod/i;
                    const collect = () => {
                      const urls = new Set();
                      for (const el of document.querySelectorAll('video, source, a, [src], [href]')) {
                        for (const attr of ['src', 'href', 'currentSrc']) {
                          const value = el[attr] || (el.getAttribute && el.getAttribute(attr));
                          if (typeof value === 'string' && videoPattern.test(value)) urls.add(value);
                        }
                      }
                      const html = document.body ? document.body.innerHTML : '';
                      for (const match of html.matchAll(/https?:\\/\\/[^\\s'"<>]+/g)) {
                        if (videoPattern.test(match[0])) urls.add(match[0]);
                      }
                      return Array.from(urls);
                    };
                    try { window.scrollTo(0, document.body ? document.body.scrollHeight : 0); } catch (_) {}
                    await new Promise(resolve => setTimeout(resolve, 600));
                    let urls = collect();
                    if (!urls.length) {
                      const covers = Array.from(document.querySelectorAll('img')).filter(img => coverPattern.test(img.src || img.getAttribute('src') || ''));
                      for (const img of covers.slice(-5)) {
                        const target = img.closest('button,[role=button],a,div') || img;
                        try {
                          target.scrollIntoView({block: 'center', inline: 'center'});
                          target.click();
                        } catch (_) {}
                      }
                      await new Promise(resolve => setTimeout(resolve, 1800));
                      urls = collect();
                    }
                    return {
                      href: location.href,
                      text: document.body ? document.body.innerText : '',
                      urls,
                    };
                }"""
            )
            page_text = str(result.get("text") or "")
            if require_prompt_match and prompt_snippet and prompt_snippet not in page_text:
                return []
            videos: List[Dict[str, Any]] = []
            for url in result.get("urls") or []:
                if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    continue
                if not video_regex.search(url):
                    continue
                if any(video["video_url"] == url for video in videos):
                    continue
                videos.append({
                    "video_url": url,
                    "cover_url": "",
                    "width": 0,
                    "height": 0,
                    "duration": 0.0,
                })
            return videos

        async def candidate_urls() -> List[str]:
            if conversation_id:
                return [f"{DOUBAO_URL}/chat/{conversation_id}"]
            return [bound_url]
            current = self._page.url
            links = await self._page.evaluate(
                """() => Array.from(document.querySelectorAll('a[href*="/chat/"]'))
                    .map(a => ({ text: (a.innerText || a.textContent || '').trim(), href: a.href }))
                    .filter(x => /\\/chat\\/\\d+/.test(x.href))
                    .slice(0, 30)"""
            )
            candidates = []
            if conversation_id:
                candidates.append(f"{DOUBAO_URL}/chat/{conversation_id}")
            candidates.append(current)
            for item in links or []:
                text = str(item.get("text") or "")
                href = str(item.get("href") or "")
                if not href:
                    continue
                if "视频" in text or "生成" in text or "video" in text.lower():
                    candidates.append(href)
            for item in links or []:
                href = str(item.get("href") or "")
                if href:
                    candidates.append(href)
            unique: List[str] = []
            for url in candidates:
                if url and url not in unique:
                    unique.append(url)
            return unique

        while time.time() < deadline:
            if not self._page:
                return {
                    "videos": [],
                    "prompt": prompt,
                    "conversation_id": conversation_id,
                    "conversation_url": conversation_url or bound_url,
                }
            try:
                for url in await candidate_urls():
                    if not self._page:
                        return {
                            "videos": [],
                            "prompt": prompt,
                            "conversation_id": conversation_id,
                            "conversation_url": conversation_url or bound_url,
                        }
                    if url not in visited or time.time() + 20 > deadline:
                        visited.add(url)
                    if self._page.url != url:
                        await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        try:
                            await self._page.wait_for_load_state("networkidle", timeout=5000)
                        except Exception:
                            pass
                    videos = await extract_current_page()
                    if videos:
                        log.info("generate_video_web: collected %d video(s) from UI", len(videos))
                        return {
                            "videos": videos,
                            "prompt": prompt,
                            "conversation_id": conversation_id,
                            "conversation_url": conversation_url or bound_url,
                        }
            except Exception as exc:
                log.warning("generate_video_web: UI video polling attempt failed: %s", exc)
            await asyncio.sleep(10)

        log.info("generate_video_web: UI polling timed out without video URL")
        return {
            "videos": [],
            "prompt": prompt,
            "conversation_id": conversation_id,
            "conversation_url": conversation_url or bound_url,
        }

    async def recover_video_result(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        conversation_url: Optional[str] = None,
        timeout: float = 30,
    ) -> Dict[str, Any]:
        """Recover an already accepted video from a Doubao conversation."""
        return await self._wait_for_video_result_from_ui(
            prompt,
            conversation_id=conversation_id,
            conversation_url=conversation_url,
            timeout=timeout,
        )

    async def poll_video_result(
        self,
        provider_task_id: str,
        prompt: str,
        timeout: float = 30,
    ) -> Dict[str, Any]:
        """Poll an already submitted Samantha video task by provider task ID."""
        return await self._poll_video_result(provider_task_id, prompt, timeout=timeout)

    async def submit_video(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        ref_image_key: Optional[str] = None,
        reference_image_keys: Optional[List[str]] = None,
        reference_image_infos: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        duration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Submit a video job and return provider binding IDs only.

        Args:
            prompt: Text description of the video to generate.
            ratio: Aspect ratio ("16:9", "9:16", "1:1").
            ref_image_key: Optional uploaded image key to use as the first/reference frame.
            reference_image_keys: Optional list of uploaded image keys for multi-image reference.

        Returns:
            Dict describing an accepted pending job. Completion is intentionally
            handled by a separate polling step because Doubao Web always queues
            video generation before exposing a video URL.
        """
        import base64

        image_attachments = self._normalize_reference_image_attachments(
            ref_image_key,
            reference_image_keys,
            reference_image_infos,
        )
        image_keys = [str(item["uri"]) for item in image_attachments if item.get("uri")]

        message = self._build_video_message(
            prompt,
            ratio,
            image_keys,
            model,
            duration,
            image_attachments=image_attachments,
        )

        local_conversation_id = str(uuid.uuid4())
        local_message_id = str(uuid.uuid4())
        payload = {
            "messages": [message],
            "completion_option": {
                "is_regen": False,
                "with_suggest": True,
                "need_create_conversation": True,
                "launch_stage": 1,
                "is_replace": False,
                "is_delete": False,
                "is_ai_playground": False,
                "memory_type": 2,
                "message_from": 0,
                "use_deep_think": False,
                "use_auto_cot": False,
                "resend_for_regen": False,
                "enable_commerce_credit": False,
                "action_bar_skill_id": 17,
                "conversation_init_option": {"need_ack_conversation": True},
            },
            "evaluate_option": {"web_ab_params": ""},
            "ext": {
                "conversation_init_option": json.dumps(
                    {"need_ack_conversation": True},
                    ensure_ascii=False,
                ),
            },
            "local_conversation_id": local_conversation_id,
            "local_message_id": local_message_id,
        }

        log.info("submit_video: prompt=%s, ratio=%s", prompt[:50], ratio)
        raw = await self._samantha_request(payload, timeout=60)

        # Phase 1: Extract async task_id from fin_reason
        task_id = None
        conversation_id = ""
        user_message_id = ""
        bot_message_id = ""
        reply_id = ""
        text_parts = []
        for data in self._parse_samantha_sse(raw):
            if not conversation_id:
                conversation_id = self._find_samantha_conversation_id(data)
            et = data.get("event_type")
            if et == 2005:
                detail = data.get("event_data", "")
                raise RuntimeError(f"generate_video error: {str(detail)[:500]}")

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            event_message_id = str(ed.get("message_id") or "")
            event_reply_id = str(ed.get("reply_id") or "")
            if et == 2002 and event_message_id and not user_message_id:
                user_message_id = event_message_id
            if event_reply_id and not reply_id:
                reply_id = event_reply_id
            if et != 2001:
                continue

            # Check for async task
            fin_reason = ed.get("fin_reason", {})
            if fin_reason and fin_reason.get("reason") == 1:
                async_task = fin_reason.get("async_task", {})
                task_id = async_task.get("id", "")

            # Collect text for error messages
            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue
            if msg.get("content_type") == 2001:
                if event_message_id and not bot_message_id:
                    bot_message_id = event_message_id
                content_raw = msg.get("content", "")
                if isinstance(content_raw, str):
                    try:
                        c = json.loads(content_raw)
                        text_parts.append(c.get("text", ""))
                    except json.JSONDecodeError:
                        pass

        full_text = "".join(text_parts)
        if not task_id:
            for data in self._parse_samantha_sse(raw):
                task_id = self._find_samantha_task_id(data)
                if task_id:
                    break
        provider_task_candidates: List[str] = []
        for candidate in (task_id,):
            if candidate and candidate not in provider_task_candidates and str(candidate) != "0":
                provider_task_candidates.append(str(candidate))
        message_id_candidates: List[str] = []
        for candidate in (user_message_id, reply_id, bot_message_id):
            if candidate and candidate not in message_id_candidates and str(candidate) != "0":
                message_id_candidates.append(str(candidate))
        if "服务过载" in full_text or "重试" in full_text:
            raise RuntimeError("视频生成服务过载，请稍后重试")

        if not task_id and not conversation_id:
            if full_text:
                return {
                    "videos": [],
                    "prompt": prompt,
                    "message": full_text,
                    "conversation_id": conversation_id,
                    "local_conversation_id": local_conversation_id,
                    "local_message_id": local_message_id,
                }
            raise RuntimeError("Video generation submit did not return task_id or conversation_id")

        accepted = bool(task_id) or bool(
            conversation_id and (not full_text or self._is_video_acceptance_text(full_text))
        )
        if not accepted:
            return {
                "videos": [],
                "prompt": prompt,
                "message": full_text,
                "provider_task_id": task_id,
                "provider_task_candidates": provider_task_candidates,
                "conversation_id": conversation_id,
                "local_conversation_id": local_conversation_id,
                "local_message_id": local_message_id,
                "user_message_id": user_message_id,
                "bot_message_id": bot_message_id,
                "reply_id": reply_id,
                "message_id_candidates": message_id_candidates,
            }

        result = {
            "videos": [],
            "prompt": prompt,
            "message": full_text,
            "provider_task_id": task_id,
            "provider_task_candidates": provider_task_candidates,
            "conversation_id": conversation_id,
            "local_conversation_id": local_conversation_id,
            "local_message_id": local_message_id,
            "user_message_id": user_message_id,
            "bot_message_id": bot_message_id,
            "reply_id": reply_id,
            "message_id_candidates": message_id_candidates,
            "pending": True,
            "accepted": True,
        }
        return result

    async def generate_video(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        ref_image_key: Optional[str] = None,
        reference_image_keys: Optional[List[str]] = None,
        reference_image_infos: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        duration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate video using /samantha/chat/completion (legacy wait path)."""
        submitted = await self.submit_video(
            prompt=prompt,
            ratio=ratio,
            ref_image_key=ref_image_key,
            reference_image_keys=reference_image_keys,
            reference_image_infos=reference_image_infos,
            model=model,
            duration=duration,
        )
        task_id = str(submitted.get("provider_task_id") or "")
        if not task_id:
            return submitted

        # Phase 2: Poll for result
        log.info("generate_video: polling task_id=%s", task_id)
        result = await self._poll_video_result(task_id, prompt)
        full_text = str(submitted.get("message") or "")
        if full_text and not result.get("message"):
            result["message"] = full_text
        result.setdefault("provider_task_id", task_id)
        result.setdefault("conversation_id", submitted.get("conversation_id") or "")
        result.setdefault("local_conversation_id", submitted.get("local_conversation_id") or "")
        result.setdefault("local_message_id", submitted.get("local_message_id") or "")
        if not result.get("videos"):
            result.setdefault("pending", True)
            result.setdefault("accepted", True)
        return result

    async def _poll_video_result(
        self, task_id: str, prompt: str, timeout: float = 300
    ) -> Dict[str, Any]:
        """Poll /samantha/chat/async/stream with task_id for video result."""
        import base64

        poll_payload = {"task_id": task_id, "event_id": 0}
        raw = await self._samantha_async_stream_request(poll_payload, timeout=timeout)

        videos = []
        text_parts: List[str] = []
        error_parts: List[str] = []
        for data in self._parse_samantha_sse(raw):
            et = data.get("event_type")
            if et == 2005:
                detail = data.get("event_data", "")
                if detail:
                    error_parts.append(str(detail))
                continue
            if et != 2001:
                continue

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue

            if msg.get("content_type") == 2001:
                content_raw = msg.get("content", "")
                if isinstance(content_raw, str):
                    try:
                        content = json.loads(content_raw)
                        if isinstance(content, dict) and content.get("text"):
                            text_parts.append(str(content.get("text")))
                    except json.JSONDecodeError:
                        pass
                continue

            if msg.get("content_type") != 2021:
                continue

            content_raw = msg.get("content", "")
            if isinstance(content_raw, str):
                try:
                    content = json.loads(content_raw)
                except json.JSONDecodeError:
                    continue
            else:
                content = content_raw

            for item in content.get("data", [content]):
                if not isinstance(item, dict):
                    continue
                video_url = item.get("video_url", "") or item.get("url", "")
                if not video_url:
                    vm_str = item.get("video_model", "")
                    if vm_str:
                        try:
                            vm = json.loads(vm_str) if isinstance(vm_str, str) else vm_str
                            vlist = vm.get("video_list", {})
                            for _q, vinfo in vlist.items():
                                main_b64 = vinfo.get("main_url", "")
                                if main_b64:
                                    video_url = base64.b64decode(main_b64).decode(
                                        "utf-8", errors="replace"
                                    )
                                    break
                        except (json.JSONDecodeError, Exception):
                            pass

                cover_url = item.get("cover_url", "") or item.get("cover", {}).get("url", "")
                if video_url:
                    videos.append({
                        "video_url": video_url,
                        "cover_url": cover_url,
                        "width": item.get("width", 0),
                        "height": item.get("height", 0),
                        "duration": item.get("duration", 0.0),
                    })

        log.info("generate_video: got %d videos", len(videos))
        result: Dict[str, Any] = {"videos": videos, "prompt": prompt}
        message = "".join(text_parts).strip() or "\n".join(error_parts).strip()
        if message:
            result["message"] = message
        return result

    # ------------------------------------------------------------------
    # File upload (TOS / ImageX flow)
    # ------------------------------------------------------------------

    @staticmethod
    def _image_dimensions(image_bytes: bytes) -> tuple[int, int]:
        """Read common image dimensions without adding heavyweight deps."""
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") and len(image_bytes) >= 24:
            width = int.from_bytes(image_bytes[16:20], "big")
            height = int.from_bytes(image_bytes[20:24], "big")
            if width > 0 and height > 0:
                return width, height
        if image_bytes.startswith(b"\xff\xd8"):
            idx = 2
            while idx + 9 < len(image_bytes):
                if image_bytes[idx] != 0xFF:
                    idx += 1
                    continue
                marker = image_bytes[idx + 1]
                idx += 2
                if marker in (0xD8, 0xD9):
                    continue
                if idx + 2 > len(image_bytes):
                    break
                block_len = int.from_bytes(image_bytes[idx:idx + 2], "big")
                if block_len < 2:
                    break
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    height = int.from_bytes(image_bytes[idx + 3:idx + 5], "big")
                    width = int.from_bytes(image_bytes[idx + 5:idx + 7], "big")
                    if width > 0 and height > 0:
                        return width, height
                    break
                idx += block_len
        return 64, 64

    async def upload_file(
        self,
        file_data: bytes,
        filename: str,
        resource_type: int = 1,
    ) -> Dict[str, Any]:
        """Upload a file to Doubao's storage (ByteDance TOS via ImageX proxy).

        4-step flow:
          1. POST /alice/resource/prepare_upload -> STS credentials
          2. GET  /top/v1?Action=ApplyImageUpload -> upload address
          3. POST https://{tos_host}/upload/v1/{store_uri} -> upload binary
          4. POST /top/v1?Action=CommitImageUpload -> confirm

        Returns:
            Dict with uri, name, size, file_type.
        """
        import zlib
        import hashlib
        import hmac as hmac_mod
        from datetime import datetime, timezone
        from urllib.parse import urlparse, parse_qs, quote as url_quote

        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        file_size = len(file_data)
        crc32 = format(zlib.crc32(file_data) & 0xFFFFFFFF, "08x")

        query_params = self._build_query_params()
        signed_url = await self._sign_url(
            f"{DOUBAO_URL}/alice/resource/prepare_upload", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)

        # Step 1: prepare_upload
        resp = await self._http.post(
            signed_url, headers=headers,
            json={"tenant_id": "5", "scene_id": "5", "resource_type": int(resource_type)},
            timeout=30,
        )
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"prepare_upload failed: {body.get('msg', body)}")
        data = body["data"]
        service_id = data["service_id"]
        auth_token = data["upload_auth_token"]
        ak = auth_token["access_key"]
        sk = auth_token["secret_key"]
        st = auth_token["session_token"]

        # AWS V4 signing helper
        def _aws_sign_v4(method, url, req_body):
            parsed = urlparse(url)
            host = parsed.hostname or ""
            path = parsed.path or "/"
            now = datetime.now(timezone.utc)
            amz_date = now.strftime("%Y%m%dT%H%M%SZ")
            date_stamp = now.strftime("%Y%m%d")
            qparams = parse_qs(parsed.query, keep_blank_values=True)
            sorted_qp = sorted((k, v[0] if v else "") for k, v in qparams.items())
            canonical_qs = "&".join(
                f"{url_quote(k, safe='~')}={url_quote(v, safe='~')}" for k, v in sorted_qp
            )
            h2s = {"host": host, "x-amz-date": amz_date}
            if st:
                h2s["x-amz-security-token"] = st
            signed_h = ";".join(sorted(h2s.keys()))
            canonical_h = "".join(f"{k}:{v}\n" for k, v in sorted(h2s.items()))
            body_b = req_body if isinstance(req_body, bytes) else req_body.encode()
            payload_hash = hashlib.sha256(body_b).hexdigest()
            cr = f"{method}\n{path}\n{canonical_qs}\n{canonical_h}\n{signed_h}\n{payload_hash}"
            scope = f"{date_stamp}/cn-north-1/imagex/aws4_request"
            cr_hash = hashlib.sha256(cr.encode()).hexdigest()
            sts = f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{cr_hash}"
            def _s(key, msg):
                return hmac_mod.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
            k_d = _s(f"AWS4{sk}".encode("utf-8"), date_stamp)
            k_r = _s(k_d, "cn-north-1")
            k_sv = _s(k_r, "imagex")
            k_sg = _s(k_sv, "aws4_request")
            sig = hmac_mod.new(k_sg, sts.encode("utf-8"), hashlib.sha256).hexdigest()
            auth_str = f"AWS4-HMAC-SHA256 Credential={ak}/{scope}, SignedHeaders={signed_h}, Signature={sig}"
            result = {"Authorization": auth_str, "x-amz-date": amz_date, "x-amz-content-sha256": payload_hash}
            if st:
                result["x-amz-security-token"] = st
            return result

        # Step 2: ApplyImageUpload
        file_ext = f".{ext}" if ext else ""
        apply_url = (
            f"{DOUBAO_URL}/top/v1?"
            f"Action=ApplyImageUpload&Version=2018-08-01"
            f"&ServiceId={service_id}&NeedFallback=true"
            f"&FileSize={file_size}&FileExtension={file_ext}"
            f"&s=jdnfglwfkl"
        )
        sign_h = _aws_sign_v4("GET", apply_url, "")
        sign_h["Cookie"] = cookie_str
        resp = await self._http.get(apply_url, headers=sign_h, timeout=30)
        result_data = resp.json().get("Result")
        if not result_data:
            raise RuntimeError(f"ApplyImageUpload failed: {resp.json()}")
        upload_addr = result_data["UploadAddress"]
        store_info = upload_addr["StoreInfos"][0]
        store_uri = store_info["StoreUri"]
        tos_auth = store_info["Auth"]
        session_key = upload_addr["SessionKey"]
        upload_hosts = upload_addr.get("UploadHosts", [])

        # Step 3: Upload binary to TOS
        tos_host = upload_hosts[0] if upload_hosts else "tos-mya2lf.vodupload.com"
        upload_url = f"https://{tos_host}/upload/v1/{store_uri}"
        resp = await self._http.post(
            upload_url, content=file_data,
            headers={"Authorization": tos_auth, "Content-CRC32": crc32},
            timeout=120,
        )
        tos_resp = resp.json()
        if tos_resp.get("code") != 2000:
            raise RuntimeError(f"TOS upload failed: {tos_resp}")

        # Step 4: CommitImageUpload
        commit_url = (
            f"{DOUBAO_URL}/top/v1?"
            f"Action=CommitImageUpload&Version=2018-08-01"
            f"&ServiceId={service_id}"
        )
        commit_body = json.dumps({"SessionKey": session_key})
        sign_h2 = _aws_sign_v4("POST", commit_url, commit_body)
        sign_h2["Content-Type"] = "application/json"
        sign_h2["Cookie"] = cookie_str
        resp = await self._http.post(commit_url, content=commit_body, headers=sign_h2, timeout=30)
        body = resp.json()
        results = body.get("Result", {}).get("Results", [])
        if not results or results[0].get("UriStatus") != 2000:
            raise RuntimeError(f"CommitImageUpload failed: {body}")

        log.info("File uploaded: %s -> %s", filename, store_uri)
        return {"uri": store_uri, "name": filename, "size": file_size, "file_type": ext}

    async def upload_video_reference_image(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
    ) -> Dict[str, Any]:
        """Upload an image through the same ImageX path used by Doubao video UI."""
        width, height = self._image_dimensions(image_bytes)
        uploaded = await self.upload_file(
            image_bytes,
            filename,
            resource_type=2,
        )
        ext = str(
            uploaded.get("file_type")
            or (filename.rsplit(".", 1)[-1] if "." in filename else "png")
        ).lower()
        return {
            "uri": uploaded["uri"],
            "name": filename,
            "format": ext,
            "width": width,
            "height": height,
            "cdn_url": "",
            "parse_state": 0,
            "review_state": 1,
            "upload_status": 1,
            "progress": 100,
        }

    async def get_file_download_url(
        self,
        uri: str,
        expire_seconds: int = 3600,
    ) -> Dict[str, Any]:
        """Get a temporary CDN URL for a previously uploaded file."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")
        query_params = self._build_query_params()
        signed_url = await self._sign_url(
            f"{DOUBAO_URL}/alice/message/get_file_url", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)
        ext = uri.rsplit(".", 1)[-1] if "." in uri else ""
        resp = await self._http.post(
            signed_url,
            headers=headers,
            json={
                "uris": [uri],
                "type": "file",
                "format": ext,
                "expire_second": expire_seconds,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"get_file_url failed ({resp.status_code}): {resp.text[:500]}")
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"get_file_url error: {body.get('msg', body)}")
        file_urls = body.get("data", {}).get("file_urls", [])
        if not file_urls:
            raise RuntimeError("get_file_url returned no file_urls")
        return file_urls[0].get("main_url", "")

    async def upload_image(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
    ) -> Dict[str, Any]:
        """Upload an image and return metadata usable by chat/image generation."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
        query_params = self._build_query_params()
        signed_url = await self._sign_url(
            f"{DOUBAO_URL}/samantha/pages/upload_image", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)
        headers.pop("Content-Type", None)
        files = {
            "data": (filename, image_bytes, f"image/{ext}"),
            "file_type": (None, ext),
        }
        resp = await self._http.post(signed_url, headers=headers, files=files, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Image upload failed ({resp.status_code}): {resp.text[:500]}")
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"Image upload error: {body.get('msg', body)}")
        uri = body.get("data", {}).get("uri", "")
        if not uri:
            raise RuntimeError(f"Image upload returned no uri: {body}")
        query_params = self._build_query_params()
        file_url = await self._sign_url(
            f"{DOUBAO_URL}/alice/message/get_file_url", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)
        resp = await self._http.post(
            file_url,
            headers=headers,
            json={
                "uris": [uri],
                "type": "image",
                "format": ext,
                "expire_second": 3600,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"get_file_url failed ({resp.status_code}): {resp.text[:500]}")
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"get_file_url error: {body.get('msg', body)}")
        file_urls = body.get("data", {}).get("file_urls", [])
        if not file_urls:
            raise RuntimeError("get_file_url returned no file_urls")
        info = file_urls[0]
        return {
            "uri": info.get("uri", uri),
            "cdn_url": info.get("main_url", ""),
            "name": filename,
            "format": ext,
            "width": "64",
            "height": "64",
        }

    async def chat_with_file(
        self,
        text: str,
        file_uri: str,
        file_name: str,
        file_size: int,
        use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        """Chat with a file attachment. The AI will read the file and answer.

        Args:
            text: Question about the file.
            file_uri: URI from upload_file().
            file_name: Original filename.
            file_size: File size in bytes.
            use_deep_think: 0=quick, 1=think, 3=expert.

        Returns:
            Dict with 'text' and 'conversation_id'.
        """
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        msg_uuid = str(uuid.uuid4())
        local_conv_id = f"local_{uuid.uuid4().int % 10**16}"
        now_ms = int(time.time() * 1000)
        now_sec = int(time.time())

        if isinstance(file_uri, list):
            file_refs = file_uri
        else:
            file_refs = [{"uri": file_uri, "name": file_name, "size": file_size}]
        file_attachments = []
        for file_ref in file_refs:
            file_attachments.append({
                "type": 3,
                "identifier": str(uuid.uuid4()),
                "file": {
                    "uri": file_ref.get("uri", ""),
                    "url": "",
                    "file_type": 0,
                    "name": file_ref.get("name", "file.txt"),
                    "size": int(file_ref.get("size") or 0),
                },
                "parse_state": 1,
                "review_state": 1,
                "upload_status": 1,
                "progress": 100,
                "src": "",
            })

        payload = {
            "client_meta": {
                "local_conversation_id": local_conv_id,
                "conversation_id": "",
                "bot_id": DEFAULT_BOT_ID,
                "last_section_id": "",
                "last_message_index": None,
            },
            "messages": [{
                "local_message_id": msg_uuid,
                "content_block": [
                    {
                        "block_type": 10052,
                        "content": {
                            "attachment_block": {
                                "attachments": file_attachments
                            },
                            "pc_event_block": "",
                        },
                        "block_id": str(uuid.uuid4()),
                        "parent_id": "",
                        "meta_info": [],
                        "append_fields": [],
                    },
                    {
                        "block_type": 10000,
                        "content": {
                            "text_block": {"text": text, "icon_url": "", "icon_url_dark": "", "summary": ""},
                            "pc_event_block": "",
                        },
                        "block_id": str(uuid.uuid4()),
                        "parent_id": "",
                        "meta_info": [],
                        "append_fields": [],
                    },
                ],
                "message_status": 0,
            }],
            "option": {
                "send_message_scene": "", "create_time_ms": now_ms, "collect_id": "",
                "is_audio": False, "answer_with_suggest": False, "tts_switch": False,
                "need_deep_think": use_deep_think, "click_clear_context": False,
                "from_suggest": False, "is_regen": False, "is_replace": False,
                "disable_sse_cache": False, "select_text_action": "",
                "resend_for_regen": False, "scene_type": 0,
                "unique_key": str(uuid.uuid4()), "start_seq": 0,
                "need_create_conversation": True, "regen_query_id": [],
                "edit_query_id": [], "regen_instruction": "",
                "no_replace_for_regen": False, "message_from": 0,
                "shared_app_name": "", "shared_app_id": "",
                "sse_recv_event_options": {"support_chunk_delta": True},
                "is_ai_playground": False,
                "recovery_option": {"is_recovery": False, "req_create_time_sec": now_sec, "append_sse_event_scene": 0},
                "message_storage_type": 0,
            },
            "ext": {
                "use_deep_think": str(use_deep_think), "fp": self._fp or "",
                "collection_id": "", "commerce_credit_config_enable": "0",
                "sub_conv_firstmet_type": "1",
            },
        }

        query_params = self._build_query_params()
        query_string = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))
        url = f"/chat/completion?{query_string}"

        # Use browser fetch (non-streaming, collect full response)
        js_code = """
        async ([url, payloadJson]) => {
            const csrf = document.cookie.match(/passport_csrf_token=([^;]+)/);
            const csrfToken = csrf ? csrf[1] : '';
            const headers = {
                'Content-Type': 'application/json',
                'agw-js-conv': 'str',
            };
            if (csrfToken) headers['x-tt-passport-csrf-token'] = csrfToken;
            const res = await fetch(url, {
                method: 'POST',
                headers: headers,
                body: payloadJson,
                credentials: 'include',
            });
            if (!res.ok) {
                const errBody = await res.text();
                return {error: true, status: res.status, body: errBody.slice(0, 500)};
            }
            const body = await res.text();
            return {error: false, body: body};
        }
        """
        payload_json = json.dumps(payload, ensure_ascii=False)
        log.info("POST /chat/completion [chat_with_file, browser fetch]")
        result = await self._page.evaluate(js_code, [url, payload_json])

        if result.get("error"):
            raise RuntimeError(
                f"chat_with_file error {result.get('status')}: {result.get('body', '')[:200]}"
            )

        full_text = ""
        conv_id = None
        raw_body = result.get("body", "")
        for block in raw_body.split("\n"):
            line = block.strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if not data_str or data_str == "{}":
                continue
            try:
                data = json.loads(data_str)
                full_text += self._extract_text(data)
                if not conv_id:
                    cid = self.extract_conversation_id(data)
                    if cid and cid != "0":
                        conv_id = cid
            except json.JSONDecodeError:
                continue

        return {"text": full_text, "conversation_id": conv_id}
