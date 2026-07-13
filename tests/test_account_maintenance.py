import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "doubao2api" / "account_manager.py"
SPEC = importlib.util.spec_from_file_location("account_manager_maintenance", MODULE_PATH)
account_manager = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(account_manager)
DoubaoAccountStore = account_manager.DoubaoAccountStore
DoubaoAccountManager = account_manager.DoubaoAccountManager


class AccountMaintenanceTest(unittest.TestCase):
    def test_lease_is_exclusive_and_released_by_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DoubaoAccountStore(str(Path(tmp) / "accounts.sqlite3"))

            started = store.begin_maintenance("default", "owner-a", ttl_seconds=900)
            self.assertEqual(started["state"], "maintenance")
            self.assertEqual(started["lease_owner"], "owner-a")
            self.assertTrue(store.is_in_maintenance("default"))

            with self.assertRaises(RuntimeError):
                store.begin_maintenance("default", "owner-b", ttl_seconds=900)

            heartbeat = store.heartbeat_maintenance("default", "owner-a", ttl_seconds=900)
            self.assertEqual(heartbeat["lease_owner"], "owner-a")

            stopped = store.end_maintenance("default", "owner-a")
            self.assertEqual(stopped["state"], "active")
            self.assertFalse(store.is_in_maintenance("default"))

    def test_expired_lease_can_be_taken_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DoubaoAccountStore(str(Path(tmp) / "accounts.sqlite3"))
            store.begin_maintenance("default", "owner-a", ttl_seconds=900)
            with store._connect() as conn:
                conn.execute(
                    "UPDATE doubao_account_maintenance SET lease_expires_at = ? WHERE account_id = ?",
                    (0, "default"),
                )

            started = store.begin_maintenance("default", "owner-b", ttl_seconds=900)
            self.assertEqual(started["lease_owner"], "owner-b")

    def test_transitional_statuses_are_not_schedulable(self):
        self.assertIn("starting", account_manager.UNAVAILABLE_ACCOUNT_STATUSES)
        self.assertIn("maintenance_pending_validation", account_manager.UNAVAILABLE_ACCOUNT_STATUSES)


class AccountMaintenanceDrainTest(unittest.IsolatedAsyncioTestCase):
    async def test_drain_does_not_wait_for_calling_task(self):
        manager = object.__new__(DoubaoAccountManager)
        manager._active_operations = {}
        manager._operation_idle_events = {}
        manager.maintenance_drain_timeout_seconds = 1
        manager._track_current_operation("default")

        await manager._wait_for_active_operations("default")

    async def test_waits_for_active_account_operation(self):
        manager = object.__new__(DoubaoAccountManager)
        manager._active_operations = {}
        manager._operation_idle_events = {}
        manager.maintenance_drain_timeout_seconds = 1
        started = asyncio.Event()
        release = asyncio.Event()

        async def operation():
            manager._track_current_operation("default")
            started.set()
            await release.wait()

        operation_task = asyncio.create_task(operation())
        await started.wait()
        drain_task = asyncio.create_task(manager._wait_for_active_operations("default"))
        await asyncio.sleep(0)
        self.assertFalse(drain_task.done())

        release.set()
        await operation_task
        await drain_task

    async def test_validation_keeps_account_unschedulable_until_final_probe(self):
        class Store:
            def __init__(self):
                self.account = {
                    "id": "default",
                    "enabled": True,
                    "status": "maintenance_pending_validation",
                }

            def get(self, account_id):
                return dict(self.account) if account_id == "default" else None

            def is_in_maintenance(self, account_id):
                return False

        class Client:
            page = object()
            _context = object()

        manager = object.__new__(DoubaoAccountManager)
        manager.store = Store()
        manager.clients = {"default": Client()}
        manager.last_touch = {}
        manager._locks = {}
        manager._active_operations = {}
        manager._operation_idle_events = {}

        account, _ = await manager.ensure_client(
            "default",
            allow_pending_validation=True,
            track_operation=True,
        )
        self.assertEqual(account["status"], "maintenance_pending_validation")
        self.assertEqual(manager.store.get("default")["status"], "maintenance_pending_validation")


if __name__ == "__main__":
    unittest.main()
