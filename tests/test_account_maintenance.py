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


if __name__ == "__main__":
    unittest.main()
