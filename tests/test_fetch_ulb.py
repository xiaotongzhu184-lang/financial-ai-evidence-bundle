import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_ulb.py"
SPEC = importlib.util.spec_from_file_location("fetch_ulb", SCRIPT)
fetch_ulb = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(fetch_ulb)


class FetchUlbTests(unittest.TestCase):
    def test_fixed_download_origin_is_accepted(self):
        self.assertEqual(fetch_ulb.checked_download_url(), fetch_ulb.URL)

    def test_changed_download_origin_is_rejected(self):
        original = fetch_ulb.URL
        try:
            fetch_ulb.URL = "https://example.com/creditcard.csv"
            with self.assertRaisesRegex(ValueError, "unexpected dataset mirror URL"):
                fetch_ulb.checked_download_url()
        finally:
            fetch_ulb.URL = original


if __name__ == "__main__":
    unittest.main()
