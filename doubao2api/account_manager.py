from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python always has zoneinfo in supported runtimes.
    ZoneInfo = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .browser_client import BrowserClient

DEFAULT_SESSION_FILE = os.environ.get("DOUBAO_SESSION_FILE", "/app/data/.doubao_session.json")


ACCOUNT_STATUSES = {
    "new",
    "starting",
    "ready",
    "not_logged_in",
    "captcha_required",
    "error",
    "browser_error",
    "temporarily_blocked",
    "maintenance_pending_validation",
    "stopped",
    "disabled",
}

UNAVAILABLE_ACCOUNT_STATUSES = {
    "not_logged_in",
    "captcha_required",
    "browser_error",
    "temporarily_blocked",
    "maintenance_pending_validation",
    "disabled",
}

QUOTA_KINDS = {"image", "video"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def account_data_root() -> str:
    explicit = os.environ.get("DOUBAO_ACCOUNT_DATA_DIR") or os.environ.get("DOUBAO_DATA_DIR")
    if explicit:
        return explicit

    session_file = os.environ.get("DOUBAO_SESSION_FILE")
    if session_file:
        return os.path.dirname(session_file)

    browser_data = os.environ.get("DOUBAO_BROWSER_DATA")
    if browser_data:
        return os.path.dirname(browser_data)

    return "/app/data" if os.path.isdir("/app") else os.path.join(os.getcwd(), "data")


def account_db_path() -> str:
    explicit = os.environ.get("DOUBAO_ACCOUNT_DB")
    if explicit:
        return explicit
    return os.path.join(account_data_root(), "doubao_accounts.sqlite3")


def _now() -> int:
    return int(time.time())


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _quota_timezone():
    name = (
        os.environ.get("DOUBAO_PROVIDER_QUOTA_TIMEZONE")
        or os.environ.get("DOUBAO_QUOTA_TIMEZONE")
        or "Asia/Shanghai"
    )
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception:
            pass
    return timezone.utc


def _next_provider_quota_reset_at(now: Optional[int] = None) -> int:
    now_ts = int(now if now is not None else _now())
    reset_hour = max(0, min(23, _env_int("DOUBAO_PROVIDER_QUOTA_RESET_HOUR", 0)))
    reset_minute = max(0, min(59, _env_int("DOUBAO_PROVIDER_QUOTA_RESET_MINUTE", 0)))
    tz = _quota_timezone()
    current = datetime.fromtimestamp(now_ts, tz=tz)
    reset = current.replace(
        hour=reset_hour,
        minute=reset_minute,
        second=0,
        microsecond=0,
    )
    if reset <= current:
        reset += timedelta(days=1)
    return int(reset.timestamp())


def _safe_id(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = value.strip("-_")
    return value or f"acct-{uuid.uuid4().hex[:10]}"


class DoubaoAccountStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or account_db_path()
        self._lock = threading.RLock()
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()
        self.ensure_default_account()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS doubao_accounts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'new',
                    session_file TEXT NOT NULL,
                    user_data_dir TEXT NOT NULL,
                    proxy_url TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    models_json TEXT NOT NULL DEFAULT '[]',
                    quota_json TEXT NOT NULL DEFAULT '{}',
                    last_used_at INTEGER,
                    last_validated_at INTEGER,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            for column, ddl in {
                "proxy_url": "ALTER TABLE doubao_accounts ADD COLUMN proxy_url TEXT NOT NULL DEFAULT ''",
                "tags_json": "ALTER TABLE doubao_accounts ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]'",
                "models_json": "ALTER TABLE doubao_accounts ADD COLUMN models_json TEXT NOT NULL DEFAULT '[]'",
                "quota_json": "ALTER TABLE doubao_accounts ADD COLUMN quota_json TEXT NOT NULL DEFAULT '{}'",
                "last_used_at": "ALTER TABLE doubao_accounts ADD COLUMN last_used_at INTEGER",
                "last_validated_at": "ALTER TABLE doubao_accounts ADD COLUMN last_validated_at INTEGER",
                "last_error": "ALTER TABLE doubao_accounts ADD COLUMN last_error TEXT NOT NULL DEFAULT ''",
            }.items():
                cols = {row["name"] for row in conn.execute("PRAGMA table_info(doubao_accounts)").fetchall()}
                if column not in cols:
                    conn.execute(ddl)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doubao_accounts_enabled ON doubao_accounts(enabled)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doubao_accounts_status ON doubao_accounts(status)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS doubao_account_maintenance (
                    account_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL DEFAULT 'active',
                    lease_owner TEXT NOT NULL DEFAULT '',
                    lease_expires_at INTEGER,
                    profile_path TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS doubao_account_usage (
                    id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    units INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'reserved',
                    request_id TEXT,
                    meta_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doubao_account_usage_window ON doubao_account_usage(account_id, kind, status, created_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS doubao_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )

    def get_setting(self, key: str, default: str = "") -> str:
        with self._lock, self._connection() as conn:
            row = conn.execute("SELECT value FROM doubao_settings WHERE key = ?", (key,)).fetchone()
            return str(row["value"]) if row else default

    def set_setting(self, key: str, value: str) -> None:
        now = _now()
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO doubao_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, now),
            )

    def ensure_default_account(self) -> Dict[str, Any]:
        existing = self.get("default")
        if existing:
            return existing

        root = account_data_root()
        session_file = os.environ.get("DOUBAO_SESSION_FILE") or DEFAULT_SESSION_FILE
        user_data_dir = os.environ.get(
            "DOUBAO_BROWSER_DATA",
            os.path.join(os.path.expanduser("~"), ".doubao_browser"),
        )
        now = _now()
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO doubao_accounts (
                    id, name, enabled, status, session_file, user_data_dir,
                    models_json, created_at, updated_at
                ) VALUES (?, ?, 1, 'new', ?, ?, ?, ?, ?)
                """,
                (
                    "default",
                    "默认豆包账号",
                    session_file or os.path.join(root, ".doubao_session.json"),
                    user_data_dir,
                    json.dumps(["chat", "image", "video", "audio"], ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return self.get("default") or {}

    def create_account(self, name: str = "", account_id: str = "") -> Dict[str, Any]:
        root = account_data_root()
        raw_id = account_id or name or f"acct-{uuid.uuid4().hex[:10]}"
        account_id = _safe_id(raw_id)
        if account_id == "default" or self.get(account_id):
            account_id = f"{account_id}-{uuid.uuid4().hex[:6]}"
        if not name:
            name = f"豆包账号 {account_id[-6:]}"

        base = os.path.join(root, "accounts", account_id)
        session_file = os.path.join(base, "session.json")
        user_data_dir = os.path.join(base, "profile")
        now = _now()
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO doubao_accounts (
                    id, name, enabled, status, session_file, user_data_dir,
                    models_json, created_at, updated_at
                ) VALUES (?, ?, 1, 'new', ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    name,
                    session_file,
                    user_data_dir,
                    json.dumps(["chat", "image", "video", "audio"], ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return self.get(account_id) or {}

    def list_accounts(self) -> list[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM doubao_accounts ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get(self, account_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM doubao_accounts WHERE id = ?",
                (account_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def begin_maintenance(self, account_id: str, owner: str, ttl_seconds: int = 900) -> Dict[str, Any]:
        account = self.get(account_id)
        if not account:
            raise KeyError(f"Account not found: {account_id}")
        owner = str(owner or "").strip()
        if not owner:
            raise ValueError("Maintenance lease owner is required")
        now = _now()
        expires_at = now + max(60, int(ttl_seconds))
        with self._lock, self._connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO doubao_account_maintenance (
                    account_id, state, lease_owner, lease_expires_at, profile_path,
                    last_error, created_at, updated_at
                ) VALUES (?, 'maintenance', ?, ?, ?, '', ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    state='maintenance', lease_owner=excluded.lease_owner,
                    lease_expires_at=excluded.lease_expires_at,
                    profile_path=excluded.profile_path, last_error='', updated_at=excluded.updated_at
                WHERE doubao_account_maintenance.state NOT IN ('maintenance', 'validating')
                   OR doubao_account_maintenance.lease_expires_at < ?
                   OR doubao_account_maintenance.lease_owner = excluded.lease_owner
                """,
                (account_id, owner, expires_at, account["user_data_dir"], now, now, now),
            )
            if cur.rowcount == 0:
                raise RuntimeError(f"Account {account_id} already has an active maintenance lease")
        return self.maintenance_status(account_id)

    def heartbeat_maintenance(self, account_id: str, owner: str, ttl_seconds: int = 900) -> Dict[str, Any]:
        now = _now()
        with self._lock, self._connection() as conn:
            cur = conn.execute(
                """
                UPDATE doubao_account_maintenance
                   SET lease_expires_at = ?, updated_at = ?
                 WHERE account_id = ? AND lease_owner = ? AND state IN ('maintenance', 'validating')
                   AND lease_expires_at > ?
                """,
                (now + max(60, int(ttl_seconds)), now, account_id, owner, now),
            )
            if cur.rowcount == 0:
                raise RuntimeError("Maintenance lease is not owned by this session")
        return self.maintenance_status(account_id)

    def end_maintenance(self, account_id: str, owner: str, last_error: str = "") -> Dict[str, Any]:
        now = _now()
        with self._lock, self._connection() as conn:
            cur = conn.execute(
                """
                UPDATE doubao_account_maintenance
                   SET state='active', lease_owner='', lease_expires_at=NULL,
                       last_error=?, updated_at=?
                 WHERE account_id=? AND (lease_owner=? OR lease_expires_at < ?)
                """,
                (str(last_error or "")[:500], now, account_id, owner, now),
            )
            if cur.rowcount == 0:
                raise RuntimeError("Maintenance lease is not owned by this session")
        return self.maintenance_status(account_id)

    def maintenance_status(self, account_id: str) -> Dict[str, Any]:
        account = self.get(account_id)
        if not account:
            raise KeyError(f"Account not found: {account_id}")
        with self._lock, self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM doubao_account_maintenance WHERE account_id = ?",
                (account_id,),
            ).fetchone()
        if not row:
            return {
                "account_id": account_id,
                "state": "active",
                "lease_owner": "",
                "lease_expires_at": None,
                "profile_path": account["user_data_dir"],
                "last_error": "",
            }
        result = dict(row)
        if result.get("state") in {"maintenance", "validating"} and int(result.get("lease_expires_at") or 0) <= _now():
            previous_expiry = result.get("lease_expires_at")
            with self._lock, self._connection() as conn:
                cur = conn.execute(
                    """
                    UPDATE doubao_account_maintenance
                       SET state='active', lease_owner='', lease_expires_at=NULL, updated_at=?
                     WHERE account_id=? AND lease_expires_at=?
                       AND state IN ('maintenance', 'validating')
                    """,
                    (_now(), account_id, previous_expiry),
                )
            if cur.rowcount == 0:
                return self.maintenance_status(account_id)
            result.update(state="active", lease_owner="", lease_expires_at=None)
        return result

    def is_in_maintenance(self, account_id: str) -> bool:
        try:
            return self.maintenance_status(account_id).get("state") in {"maintenance", "validating"}
        except KeyError:
            return False

    def update_account(self, account_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
        allowed = {
            "name",
            "enabled",
            "status",
            "session_file",
            "user_data_dir",
            "proxy_url",
            "tags_json",
            "models_json",
            "quota_json",
            "last_used_at",
            "last_validated_at",
            "last_error",
        }
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "status" and value not in ACCOUNT_STATUSES:
                continue
            if key == "enabled":
                value = 1 if value else 0
            assignments.append(f"{key} = ?")
            values.append(value)
        if not assignments:
            return self.get(account_id)
        assignments.append("updated_at = ?")
        values.append(_now())
        values.append(account_id)
        with self._lock, self._connection() as conn:
            conn.execute(
                f"UPDATE doubao_accounts SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
        return self.get(account_id)

    def mark_success(self, account_id: str) -> None:
        self.update_account(
            account_id,
            status="ready",
            last_error="",
            last_used_at=_now(),
            last_validated_at=_now(),
        )

    def mark_failure(self, account_id: str, message: str, status: str = "error") -> None:
        self.update_account(
            account_id,
            status=status if status in ACCOUNT_STATUSES else "error",
            last_error=message[:500],
            last_validated_at=_now(),
        )

    def record_task_failure(self, account_id: str, message: str, *, keep_ready: bool = False) -> None:
        account = self.get(account_id)
        if not account:
            return
        fields: Dict[str, Any] = {
            "last_error": message[:500],
            "last_validated_at": _now(),
        }
        current_status = str(account.get("status") or "").lower()
        if keep_ready and current_status not in UNAVAILABLE_ACCOUNT_STATUSES:
            fields["status"] = "ready"
        self.update_account(account_id, **fields)

    def quota_limit(self, account: Dict[str, Any], kind: str) -> int:
        quota = account.get("quota") or {}
        if not isinstance(quota, dict):
            quota = {}
        aliases = (
            f"{kind}_24h_limit",
            f"{kind}_daily_quota",
            f"{kind}_quota",
            f"{kind}_limit",
        )
        for key in aliases:
            value = quota.get(key)
            if value not in (None, ""):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        if kind == "video":
            return _env_int("DOUBAO_VIDEO_24H_QUOTA", _env_int("DOUBAO_VIDEO_DAILY_QUOTA", 10))
        if kind == "image":
            return _env_int("DOUBAO_IMAGE_24H_QUOTA", _env_int("DOUBAO_IMAGE_DAILY_QUOTA", 30))
        return 0

    def quota_window_seconds(self) -> int:
        return max(60, _env_int("DOUBAO_QUOTA_WINDOW_SECONDS", 24 * 3600))

    def quota_used(self, account_id: str, kind: str) -> int:
        cutoff = _now() - self.quota_window_seconds()
        with self._lock, self._connection() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(units), 0) AS used
                  FROM doubao_account_usage
                 WHERE account_id = ?
                   AND kind = ?
                   AND status IN ('reserved', 'completed')
                   AND created_at >= ?
                """,
                (account_id, kind, cutoff),
            ).fetchone()
        return int(row["used"] or 0) if row else 0

    def provider_sync_status(self, account: Dict[str, Any]) -> Dict[str, Any]:
        quota = account.get("quota") or {}
        if not isinstance(quota, dict):
            quota = {}
        return {
            "credit": quota.get("provider_credit") or {},
            "quota": quota.get("provider_quota") or {},
        }

    def provider_quota_snapshot(self, account: Dict[str, Any], kind: str) -> Dict[str, Any]:
        quota = account.get("quota") or {}
        if not isinstance(quota, dict):
            quota = {}
        provider_quota = quota.get("provider_quota") or {}
        if not isinstance(provider_quota, dict):
            provider_quota = {}
        raw = provider_quota.get(kind) or {}
        if not isinstance(raw, dict):
            return {}

        remaining = _safe_int(raw.get("remaining"))
        limit = _safe_int(raw.get("limit"))
        synced_at = _safe_int(raw.get("synced_at"))
        reset_at = _safe_int(raw.get("reset_at"))
        source = str(raw.get("source") or "")
        ttl = _env_int("DOUBAO_PROVIDER_QUOTA_TTL_SECONDS", 24 * 3600)
        now = _now()
        stale = bool(
            (reset_at and now >= reset_at)
            or (ttl > 0 and synced_at and now - synced_at > ttl)
        )
        return {
            "kind": kind,
            "remaining": remaining,
            "limit": limit,
            "synced_at": synced_at,
            "reset_at": reset_at,
            "source": source,
            "stale": stale,
            "message": str(raw.get("message") or "")[:500],
        }

    def quota_snapshot(self, account: Dict[str, Any], kind: str) -> Dict[str, Any]:
        limit = self.quota_limit(account, kind)
        used = self.quota_used(account["id"], kind)
        remaining = None if limit <= 0 else max(limit - used, 0)
        provider = self.provider_quota_snapshot(account, kind)
        provider_remaining = provider.get("remaining")
        provider_stale = bool(provider.get("stale"))
        effective_remaining = remaining
        if provider_remaining is not None and not provider_stale:
            effective_remaining = (
                provider_remaining
                if effective_remaining is None
                else min(effective_remaining, provider_remaining)
            )
        reset_at = None
        cutoff = _now() - self.quota_window_seconds()
        with self._lock, self._connection() as conn:
            row = conn.execute(
                """
                SELECT MIN(created_at) AS oldest
                  FROM doubao_account_usage
                 WHERE account_id = ?
                   AND kind = ?
                   AND status IN ('reserved', 'completed')
                   AND created_at >= ?
                """,
                (account["id"], kind, cutoff),
            ).fetchone()
        if row and row["oldest"]:
            reset_at = int(row["oldest"]) + self.quota_window_seconds()
        return {
            "kind": kind,
            "limit": limit,
            "used": used,
            "remaining": remaining,
            "effective_remaining": effective_remaining,
            "provider": provider,
            "window_seconds": self.quota_window_seconds(),
            "reset_at": reset_at,
            "exhausted": bool(
                (limit > 0 and remaining is not None and remaining <= 0)
                or (provider_remaining is not None and not provider_stale and provider_remaining <= 0)
            ),
        }

    def quota_status(self, account: Dict[str, Any]) -> Dict[str, Any]:
        return {kind: self.quota_snapshot(account, kind) for kind in sorted(QUOTA_KINDS)}

    def has_quota(self, account: Dict[str, Any], kind: Optional[str], units: int = 1) -> bool:
        if not kind:
            return True
        if kind not in QUOTA_KINDS:
            return True
        snapshot = self.quota_snapshot(account, kind)
        effective_remaining = snapshot.get("effective_remaining")
        if effective_remaining is None:
            return True
        return int(effective_remaining) >= max(1, int(units))

    def reserve_quota(
        self,
        account_id: str,
        kind: str,
        units: int,
        *,
        request_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        if kind not in QUOTA_KINDS:
            return ""
        account = self.get(account_id)
        if not account:
            raise RuntimeError(f"Account not found: {account_id}")
        units = max(1, int(units))
        if not self.has_quota(account, kind, units):
            snapshot = self.quota_snapshot(account, kind)
            raise RuntimeError(
                f"Account {account_id} {kind} quota exhausted: "
                f"{snapshot['used']}/{snapshot['limit']} used in 24h"
            )
        usage_id = f"usage-{uuid.uuid4().hex}"
        now = _now()
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO doubao_account_usage (
                    id, account_id, kind, units, status, request_id, meta_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'reserved', ?, ?, ?, ?)
                """,
                (
                    usage_id,
                    account_id,
                    kind,
                    units,
                    request_id,
                    json.dumps(meta or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return usage_id

    def complete_quota(self, usage_id: str) -> None:
        if not usage_id:
            return
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE doubao_account_usage SET status = 'completed', updated_at = ? WHERE id = ?",
                (_now(), usage_id),
            )

    def release_quota(self, usage_id: str) -> None:
        if not usage_id:
            return
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE doubao_account_usage SET status = 'released', updated_at = ? WHERE id = ?",
                (_now(), usage_id),
            )

    def release_reserved_usage(self) -> None:
        with self._lock, self._connection() as conn:
            conn.execute(
                "UPDATE doubao_account_usage SET status = 'released', updated_at = ? WHERE status = 'reserved'",
                (_now(),),
            )

    def mark_quota_exhausted(self, account_id: str, kind: str, message: str = "") -> None:
        account = self.get(account_id)
        if not account or kind not in QUOTA_KINDS:
            return
        if os.environ.get("DOUBAO_MARK_QUOTA_EXHAUSTED_AS_USAGE", "false").lower() not in {"1", "true", "yes", "on"}:
            return
        snapshot = self.quota_snapshot(account, kind)
        remaining = snapshot.get("remaining")
        if remaining is None:
            remaining = 1
        units = max(1, int(remaining))
        now = _now()
        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO doubao_account_usage (
                    id, account_id, kind, units, status, request_id, meta_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'completed', ?, ?, ?, ?)
                """,
                (
                    f"quota-exhausted-{uuid.uuid4().hex}",
                    account_id,
                    kind,
                    units,
                    "provider-quota-exhausted",
                    json.dumps({"message": message[:500]}, ensure_ascii=False),
                    now,
                    now,
                ),
            )

    def update_provider_credit(self, account_id: str, sync_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        account = self.get(account_id)
        if not account:
            return None
        quota = account.get("quota") or {}
        if not isinstance(quota, dict):
            quota = {}

        credit_info = sync_data.get("credit_info") or {}
        usage_history = sync_data.get("usage_history") or {}
        provider_credit = {
            "source": sync_data.get("source") or "doubao_credit_api",
            "synced_at": _safe_int(sync_data.get("synced_at"), _now()),
            "total_credit_num": _safe_int(credit_info.get("total_credit_num"), 0),
            "credit_text": credit_info.get("credit_text") or "",
            "credit_desc_text": credit_info.get("credit_desc_text") or "",
            "history_has_more": bool(usage_history.get("has_more")),
            "history_items": (usage_history.get("items") or [])[:20],
        }
        quota["provider_credit"] = provider_credit
        return self.update_account(account_id, quota_json=json.dumps(quota, ensure_ascii=False))

    def update_provider_quota(
        self,
        account_id: str,
        kind: str,
        *,
        remaining: Optional[int],
        limit: Optional[int] = None,
        source: str = "message",
        message: str = "",
    ) -> Optional[Dict[str, Any]]:
        if kind not in QUOTA_KINDS or remaining is None:
            return self.get(account_id)
        account = self.get(account_id)
        if not account:
            return None
        quota = account.get("quota") or {}
        if not isinstance(quota, dict):
            quota = {}
        provider_quota = quota.get("provider_quota") or {}
        if not isinstance(provider_quota, dict):
            provider_quota = {}
        current = provider_quota.get(kind) or {}
        if not isinstance(current, dict):
            current = {}
        current_limit = _safe_int(current.get("limit"))
        remaining_value = max(0, int(remaining))
        entry = {
            **current,
            "remaining": remaining_value,
            "limit": _safe_int(limit, current_limit),
            "source": source,
            "synced_at": _now(),
            "message": message[:500],
        }
        if remaining_value <= 0:
            entry["reset_at"] = _next_provider_quota_reset_at()
        else:
            entry.pop("reset_at", None)
        provider_quota[kind] = entry
        quota["provider_quota"] = provider_quota
        return self.update_account(account_id, quota_json=json.dumps(quota, ensure_ascii=False))

    def update_provider_quota_from_text(
        self,
        account_id: str,
        kind: str,
        text: str,
        *,
        units_completed: int = 0,
    ) -> Optional[Dict[str, Any]]:
        if not text:
            return self.get(account_id)
        patterns = [
            r"(?:今日|今天)?\s*(?:剩余|还剩)\s*(\d+)\s*个?(?:视频生成额度|视频额度|生成额度|额度)",
            r"(?:视频生成额度|视频额度|生成额度|额度)\s*(?:剩余|还剩)\s*(\d+)\s*个?",
            r"(?:今日|今天)?\s*(?:剩余|还剩)\s*(\d+)\s*个?(?:视频生成|视频|生成)",
            r"(?:视频生成|视频|生成)\s*(?:剩余|还剩)\s*(\d+)\s*个?",
            r"(?:今日|今天)?\s*(?:剩余|还剩)\s*(\d+)\s*个?(?:视频生成额度|视频额度|生成额度|额度)",
            r"(?:视频生成额度|视频额度|生成额度|额度)\s*(?:剩余|还剩)\s*(\d+)\s*个?",
            r"(?:今日|今天)?\s*(?:剩余|还剩)\s*(\d+)\s*个?(?:视频生成额度|视频额度|生成额度)",
            r"(?:视频生成额度|视频额度|生成额度)\s*(?:剩余|还剩)\s*(\d+)\s*个?",
        ]
        remaining: Optional[int] = None
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                remaining = _safe_int(match.group(1))
                break
        if remaining is None:
            return self.get(account_id)

        limit = None
        account = self.get(account_id)
        if account:
            current = self.provider_quota_snapshot(account, kind)
            current_limit = _safe_int(current.get("limit"))
            if current_limit is not None:
                limit = max(current_limit, remaining)
        if limit is None and units_completed:
            limit = remaining + max(0, int(units_completed))

        return self.update_provider_quota(
            account_id,
            kind,
            remaining=remaining,
            limit=limit,
            source="generation_message",
            message=text,
        )

    def delete_account(self, account_id: str) -> Dict[str, Any]:
        if account_id == "default":
            raise ValueError("Default account cannot be deleted")
        account = self.get(account_id)
        if not account:
            return {"deleted": False, "account_id": account_id, "cleanup": []}
        if self.is_in_maintenance(account_id):
            raise RuntimeError(f"Account {account_id} has an active maintenance lease")

        with self._lock, self._connection() as conn:
            conn.execute("DELETE FROM doubao_account_usage WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM doubao_account_maintenance WHERE account_id = ?", (account_id,))
            cur = conn.execute("DELETE FROM doubao_accounts WHERE id = ?", (account_id,))
            deleted = cur.rowcount > 0
        cleanup = self._cleanup_account_files(account) if deleted else []
        return {"deleted": deleted, "account_id": account_id, "cleanup": cleanup}

    def _cleanup_account_files(self, account: Dict[str, Any]) -> list[Dict[str, str]]:
        account_id = str(account.get("id") or "")
        base_dir = Path(account_data_root()).joinpath("accounts", account_id).expanduser().resolve()
        results: list[Dict[str, str]] = []
        candidates = [
            ("session_file", account.get("session_file")),
            ("user_data_dir", account.get("user_data_dir")),
            ("account_dir", str(base_dir)),
        ]
        for kind, raw_path in candidates:
            if not raw_path:
                continue
            target = Path(str(raw_path)).expanduser()
            try:
                resolved = target.resolve(strict=False)
                if resolved != base_dir and base_dir not in resolved.parents:
                    results.append({"kind": kind, "path": str(target), "status": "skipped_outside_account_dir"})
                    continue
                if not target.exists() and not target.is_symlink():
                    results.append({"kind": kind, "path": str(target), "status": "missing"})
                    continue
                if target.is_symlink() or target.is_file():
                    target.unlink()
                elif target.is_dir():
                    shutil.rmtree(target)
                results.append({"kind": kind, "path": str(target), "status": "deleted"})
            except Exception as exc:
                results.append({"kind": kind, "path": str(target), "status": "error", "message": str(exc)[:300]})
        return results

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["enabled"] = bool(data.get("enabled"))
        for key in ("tags_json", "models_json", "quota_json"):
            raw = data.get(key)
            try:
                parsed = json.loads(raw) if raw else ([] if key != "quota_json" else {})
            except json.JSONDecodeError:
                parsed = [] if key != "quota_json" else {}
            data[key.replace("_json", "")] = parsed
        return data


class DoubaoAccountManager:
    def __init__(
        self,
        *,
        headless: bool = True,
        max_hot_accounts: Optional[int] = None,
        idle_ttl_seconds: int = 600,
    ):
        self.store = DoubaoAccountStore()
        self.headless = headless
        self.max_hot_accounts = max_hot_accounts or int(os.environ.get("DOUBAO_MAX_HOT_ACCOUNTS", "2"))
        self.idle_ttl_seconds = idle_ttl_seconds
        self.clients: Dict[str, BrowserClient] = {}
        self.maintenance_clients: Dict[str, BrowserClient] = {}
        self.maintenance_owners: Dict[str, str] = {}
        self._maintenance_heartbeats: Dict[str, asyncio.Task] = {}
        self.last_touch: Dict[str, float] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self.default_account_id = os.environ.get("DOUBAO_DEFAULT_ACCOUNT_ID", "default")

    def _lock_for(self, account_id: str) -> asyncio.Lock:
        if account_id not in self._locks:
            self._locks[account_id] = asyncio.Lock()
        return self._locks[account_id]

    async def start(self) -> None:
        self.store.ensure_default_account()
        self.store.release_reserved_usage()
        if os.environ.get("DOUBAO_AUTOSTART_DEFAULT_ACCOUNT", "true").lower() != "false":
            try:
                await self.ensure_client(self.default_account_id)
            except Exception:
                # Startup should not fail just because the web session needs login.
                pass

    async def stop_all(self) -> None:
        for account_id in list(self.maintenance_clients):
            await self.stop_maintenance(account_id)
        for account_id in list(self.clients):
            await self.stop_client(account_id, update_status=False)

    async def ensure_client(
        self,
        account_id: str,
        *,
        allow_pending_validation: bool = False,
    ) -> tuple[Dict[str, Any], BrowserClient]:
        account = self.store.get(account_id)
        if not account:
            raise KeyError(f"Account not found: {account_id}")
        if not account.get("enabled"):
            raise RuntimeError(f"Account disabled: {account_id}")
        if account.get("status") == "maintenance_pending_validation" and not allow_pending_validation:
            raise RuntimeError(f"Account requires validation after maintenance: {account_id}")
        if self.store.is_in_maintenance(account_id):
            raise RuntimeError(f"Account is in maintenance: {account_id}")

        async with self._lock_for(account_id):
            account = self.store.get(account_id) or account
            if account.get("status") == "maintenance_pending_validation" and not allow_pending_validation:
                raise RuntimeError(f"Account requires validation after maintenance: {account_id}")
            if self.store.is_in_maintenance(account_id):
                raise RuntimeError(f"Account is in maintenance: {account_id}")
            client = self.clients.get(account_id)
            if client and client.page and client._context:
                self.last_touch[account_id] = time.time()
                return account, client

            self.store.update_account(account_id, status="starting", last_error="")
            cookie_header = os.environ.get("DOUBAO_COOKIE", "") if account_id == "default" else ""
            from .browser_client import BrowserClient
            client = BrowserClient(
                headless=self.headless,
                user_data_dir=account["user_data_dir"],
                session_file=account["session_file"],
                cookie_header=cookie_header,
            )
            try:
                await client.start()
            except Exception as exc:
                self.store.mark_failure(account_id, str(exc), "browser_error")
                try:
                    await client.stop()
                except Exception:
                    pass
                raise

            self.clients[account_id] = client
            self.last_touch[account_id] = time.time()
            self.store.update_account(
                account_id,
                status="ready" if client.is_ready else "not_logged_in",
                last_error="" if client.is_ready else "未登录",
                last_validated_at=_now(),
            )
            await self.prune_idle(exclude={account_id})
            return self.store.get(account_id) or account, client

    async def get_ready_client(
        self,
        preferred_account_id: Optional[str] = None,
        *,
        quota_kind: Optional[str] = None,
        quota_units: int = 1,
    ) -> tuple[Dict[str, Any], BrowserClient]:
        if preferred_account_id:
            account, client = await self.ensure_client(preferred_account_id)
            self._ensure_ready(account, client)
            if not self.store.has_quota(account, quota_kind, quota_units):
                snapshot = self.store.quota_snapshot(account, str(quota_kind))
                raise RuntimeError(
                    f"Account {account['id']} {quota_kind} quota exhausted: "
                    f"{snapshot['used']}/{snapshot['limit']} used in 24h"
                )
            self.last_touch[account["id"]] = time.time()
            return account, client

        accounts = [
            a
            for a in self.store.list_accounts()
            if a.get("enabled")
            and str(a.get("status") or "").lower() not in UNAVAILABLE_ACCOUNT_STATUSES
            and not self.store.is_in_maintenance(a["id"])
            and self.store.has_quota(a, quota_kind, quota_units)
        ]
        if not accounts:
            raise RuntimeError(f"No enabled Doubao accounts with available {quota_kind or 'general'} quota")

        hot_ready = []
        for account in accounts:
            client = self.clients.get(account["id"])
            if client and client.is_ready and not client.needs_captcha:
                hot_ready.append(account)
        if hot_ready:
            hot_ready.sort(key=lambda a: a.get("last_used_at") or 0)
            account = hot_ready[0]
            client = self.clients[account["id"]]
            self.last_touch[account["id"]] = time.time()
            return account, client

        last_error = ""
        for account in sorted(accounts, key=lambda a: a.get("last_used_at") or 0):
            try:
                account, client = await self.ensure_client(account["id"])
                self._ensure_ready(account, client)
                self.last_touch[account["id"]] = time.time()
                return account, client
            except Exception as exc:
                last_error = str(exc)
                continue
        raise RuntimeError(last_error or f"No ready Doubao accounts with available {quota_kind or 'general'} quota")

    def _ensure_ready(self, account: Dict[str, Any], client: BrowserClient) -> None:
        if not client.is_ready:
            raise RuntimeError(f"Account {account['id']} is not logged in")
        if client.needs_captcha:
            self.store.update_account(account["id"], status="captcha_required", last_error="Captcha required")
            raise RuntimeError(f"Account {account['id']} requires captcha")

    async def stop_client(self, account_id: str, *, update_status: bool = True) -> None:
        async with self._lock_for(account_id):
            client = self.clients.pop(account_id, None)
            self.last_touch.pop(account_id, None)
            if client:
                await client.stop()
            if update_status and self.store.get(account_id):
                self.store.update_account(account_id, status="stopped")

    async def restart_client(self, account_id: str) -> tuple[Dict[str, Any], BrowserClient]:
        await self.stop_client(account_id, update_status=False)
        return await self.ensure_client(account_id)

    async def start_maintenance(self, account_id: str) -> Dict[str, Any]:
        account = self.store.get(account_id)
        if not account:
            raise KeyError(f"Account not found: {account_id}")
        owner = uuid.uuid4().hex
        self.store.begin_maintenance(account_id, owner)
        client: Optional[BrowserClient] = None
        try:
            await self.stop_client(account_id, update_status=False)
            async with self._lock_for(account_id):
                from .browser_client import BrowserClient

                client = BrowserClient(
                    headless=False,
                    user_data_dir=account["user_data_dir"],
                    session_file=account["session_file"],
                    cookie_header=os.environ.get("DOUBAO_COOKIE", "") if account_id == "default" else "",
                )
                await client.start()
                self.maintenance_clients[account_id] = client
                self.maintenance_owners[account_id] = owner
                self._maintenance_heartbeats[account_id] = asyncio.create_task(
                    self._maintenance_heartbeat_loop(account_id, owner)
                )
        except Exception as exc:
            self.maintenance_clients.pop(account_id, None)
            self.maintenance_owners.pop(account_id, None)
            task = self._maintenance_heartbeats.pop(account_id, None)
            if task:
                task.cancel()
            if client is not None:
                try:
                    await client.stop()
                except Exception:
                    pass
            try:
                self.store.end_maintenance(account_id, owner, str(exc))
            except Exception:
                pass
            raise
        status = self.store.maintenance_status(account_id)
        status["novnc_url"] = os.environ.get("DOUBAO_NOVNC_URL", "http://127.0.0.1:6080/vnc.html?autoconnect=1&resize=scale")
        return status

    async def _maintenance_heartbeat_loop(self, account_id: str, owner: str) -> None:
        try:
            while True:
                await asyncio.sleep(300)
                self.store.heartbeat_maintenance(account_id, owner)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.stop_maintenance(account_id, owner=owner, last_error=str(exc))

    async def stop_maintenance(
        self,
        account_id: str,
        *,
        owner: str = "",
        last_error: str = "",
    ) -> Dict[str, Any]:
        current = self.store.maintenance_status(account_id)
        lease_owner = str(current.get("lease_owner") or "")
        local_owner = self.maintenance_owners.get(account_id, "")
        foreign_lease = bool(lease_owner and lease_owner != local_owner)
        if foreign_lease and not local_owner:
            raise RuntimeError("Active maintenance lease is owned by another process")
        if owner and local_owner and owner != local_owner:
            raise RuntimeError("Maintenance lease owner does not match this process")
        async with self._lock_for(account_id):
            client = self.maintenance_clients.pop(account_id, None)
            if client:
                await client.stop()
            task = self._maintenance_heartbeats.pop(account_id, None)
            self.maintenance_owners.pop(account_id, None)
            if task and task is not asyncio.current_task():
                task.cancel()
            if lease_owner:
                if foreign_lease:
                    raise RuntimeError("Local maintenance browser was stopped, but the active lease belongs to another process")
                self.store.end_maintenance(account_id, local_owner, last_error)
            if local_owner:
                self.store.update_account(
                    account_id,
                    status="browser_error" if last_error else "maintenance_pending_validation",
                    last_error=last_error,
                )
        return self.store.maintenance_status(account_id)

    def maintenance_status(self, account_id: str) -> Dict[str, Any]:
        status = self.store.maintenance_status(account_id)
        account = self.store.get(account_id)
        if status.get("state") == "active" and account and account.get("status") == "maintenance_pending_validation":
            status["state"] = "validating"
        status["browser_running"] = account_id in self.maintenance_clients
        status["novnc_url"] = os.environ.get("DOUBAO_NOVNC_URL", "http://127.0.0.1:6080/vnc.html?autoconnect=1&resize=scale")
        return status

    async def prune_idle(self, exclude: Optional[set[str]] = None) -> None:
        exclude = exclude or set()
        if self.max_hot_accounts <= 0:
            return
        hot_ids = list(self.clients)
        if len(hot_ids) <= self.max_hot_accounts:
            return
        candidates = [aid for aid in hot_ids if aid not in exclude]
        candidates.sort(key=lambda aid: self.last_touch.get(aid, 0))
        while len(self.clients) > self.max_hot_accounts and candidates:
            await self.stop_client(candidates.pop(0))

    def list_accounts(self) -> list[Dict[str, Any]]:
        rows = []
        for account in self.store.list_accounts():
            client = self.clients.get(account["id"])
            runtime = {
                "hot": bool(client),
                "ready": bool(client and client.is_ready),
                "needs_captcha": bool(client and client.needs_captcha),
                "consecutive_failures": client.consecutive_failures if client else 0,
                "last_error_code": client.last_error_code if client else 0,
                "page_url": client.page.url if client and client.page else "",
            }
            account["runtime"] = runtime
            account["maintenance"] = self.maintenance_status(account["id"])
            account["quota_status"] = self.store.quota_status(account)
            account["provider_sync"] = self.store.provider_sync_status(account)
            rows.append(account)
        return rows

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for account in self.list_accounts():
            status = account.get("status") or "unknown"
            counts[status] = counts.get(status, 0) + 1
        return counts

    def mark_success(self, account_id: str) -> None:
        client = self.clients.get(account_id)
        if client:
            client.record_success()
        self.store.mark_success(account_id)

    def mark_failure(self, account_id: str, message: str, error_code: int = 0) -> None:
        client = self.clients.get(account_id)
        status = "browser_error"
        if client:
            client.record_failure(error_code)
            if client.needs_captcha:
                status = "captcha_required"
        self.store.mark_failure(account_id, message, status)

    def record_task_failure(self, account_id: str, message: str) -> None:
        client = self.clients.get(account_id)
        if client and client.needs_captcha:
            self.store.mark_failure(account_id, message, "captcha_required")
            return
        self.store.record_task_failure(
            account_id,
            message,
            keep_ready=bool(client and client.is_ready),
        )

    def reserve_quota(
        self,
        account_id: str,
        kind: str,
        units: int,
        *,
        request_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.store.reserve_quota(account_id, kind, units, request_id=request_id, meta=meta)

    def complete_quota(self, usage_id: str) -> None:
        self.store.complete_quota(usage_id)

    def release_quota(self, usage_id: str) -> None:
        self.store.release_quota(usage_id)

    def mark_quota_exhausted(self, account_id: str, kind: str, message: str = "") -> None:
        self.store.mark_quota_exhausted(account_id, kind, message)

    def update_provider_quota_from_text(
        self,
        account_id: str,
        kind: str,
        text: str,
        *,
        units_completed: int = 0,
    ) -> Optional[Dict[str, Any]]:
        return self.store.update_provider_quota_from_text(
            account_id,
            kind,
            text,
            units_completed=units_completed,
        )

    async def sync_provider_credit(self, account_id: str) -> Dict[str, Any]:
        account, client = await self.ensure_client(account_id)
        self._ensure_ready(account, client)
        sync_data = await client.get_credit_quota()
        account = self.store.update_provider_credit(account_id, sync_data) or account
        account["quota_status"] = self.store.quota_status(account)
        account["provider_sync"] = self.store.provider_sync_status(account)
        return {"account_id": account_id, "sync": sync_data, "account": account}

    def pick_account_id_from_request(self, headers: Dict[str, str], body: Optional[Dict[str, Any]] = None) -> Optional[str]:
        body = body or {}
        for key in ("account_id", "doubao_account_id", "account"):
            value = body.get(key)
            if value:
                return str(value)
        for key in ("x-doubao-account-id", "x-account-id"):
            value = headers.get(key) or headers.get(key.title())
            if value:
                return str(value)
        return None

    async def cookies(self, account_id: str) -> list[Dict[str, Any]]:
        _, client = await self.ensure_client(account_id)
        if client._context is None:
            return []
        cookies = await client._context.cookies("https://www.doubao.com")
        return [{"name": c["name"], "value": c["value"], "length": len(c["value"])} for c in cookies]

    async def login_status(self, account_id: str) -> Dict[str, Any]:
        account, client = await self.ensure_client(account_id, allow_pending_validation=True)
        page_url = client.page.url if client.page else ""
        login_btn_count = 0
        if client.page:
            try:
                login_btn_count = await client.page.locator('button:has-text("登录")').count()
            except Exception:
                pass
        actual_logged_in = client.is_ready and login_btn_count == 0
        status = "ready" if actual_logged_in else "not_logged_in"
        self.store.update_account(account_id, status=status, last_error="" if actual_logged_in else "未登录")
        return {
            "account_id": account_id,
            "account_name": account.get("name", account_id),
            "logged_in": actual_logged_in,
            "is_ready_flag": client.is_ready,
            "login_button_visible": login_btn_count > 0,
            "page_url": page_url,
            "device_id": client._device_id or "",
            "web_id": client._web_id or "",
            "needs_captcha": client.needs_captcha,
            "consecutive_failures": client.consecutive_failures,
            "last_error_code": client.last_error_code,
        }

    def export_public_account(self, account: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(account)
        return data
