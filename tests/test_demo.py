import json
import http.client
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from yk_evidence.demo import build_server
from yk_evidence.pipeline import contract_run, integrated_run
from yk_evidence.review import append_review, read_reviews, review_path


class DemoTest(unittest.TestCase):
    def test_review_journal_uses_manifest_identity_after_bundle_directory_rename(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = contract_run(Path(root), "stable-review", seed=42)
            append_review(bundle, "Risk lead", "request_changes",
                          "Checked the evidence before renaming the delivery directory.")
            journal = review_path(bundle)

            renamed = Path(root) / "renamed-delivery-directory"
            bundle.rename(renamed)

            self.assertEqual(journal, review_path(renamed))
            records = read_reviews(renamed)
            self.assertEqual(1, len(records))
            self.assertEqual("stable-review", records[0]["bundle_run_id"])

    def test_review_form_accepts_2000_chinese_characters_and_explains_character_limit(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = contract_run(Path(root), "review-unicode", seed=42)
            server = build_server(bundle, host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                connection.request("GET", "/")
                response = connection.getresponse()
                page = response.read().decode("utf-8")
                self.assertEqual(200, response.status)
                token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
                headers = {"Content-Type": "application/x-www-form-urlencoded",
                           "Origin": f"http://127.0.0.1:{server.server_port}"}

                rationale = "核" * 2000
                payload = urlencode({"csrf": token, "reviewer": "审查负责人",
                                     "decision": "request_changes", "rationale": rationale})
                self.assertGreater(len(payload.encode("ascii")), 4096)
                connection.request("POST", "/api/reviews", body=payload, headers=headers)
                response = connection.getresponse()
                response.read()
                self.assertEqual(303, response.status)
                self.assertEqual(rationale, read_reviews(bundle)[0]["rationale"])

                payload = urlencode({"csrf": token, "reviewer": "审查负责人",
                                     "decision": "request_changes", "rationale": "核" * 2001})
                connection.request("POST", "/api/reviews", body=payload, headers=headers)
                response = connection.getresponse()
                error = response.read().decode("utf-8")
                self.assertEqual(400, response.status)
                self.assertIn("10-2000 printable characters", error)
                self.assertEqual(1, len(read_reviews(bundle)))
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_review_form_records_manifest_bound_decision_and_rejects_bad_token(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = contract_run(Path(root), "review-demo", seed=42)
            server = build_server(bundle, host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                with urlopen(url) as response:
                    page = response.read().decode()
                token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
                fields = {"csrf": token, "reviewer": "Risk lead", "decision": "request_changes",
                          "rationale": "Checked raw scores and the blocked release gate."}
                request = Request(url + "/api/reviews", data=urlencode(fields).encode(),
                                  headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
                with urlopen(request) as response:
                    self.assertEqual(200, response.status)  # urllib follows the 303
                    self.assertIn("Checked raw scores", response.read().decode())
                records = read_reviews(bundle)
                self.assertEqual(1, len(records))
                self.assertEqual("request_changes", records[0]["decision"])
                self.assertNotIn("review_records", json.loads((bundle / "bundle_manifest.json").read_text())["artifacts"])
                self.assertTrue(review_path(bundle).exists())
                fields["csrf"] = "wrong"
                request = Request(url + "/api/reviews", data=urlencode(fields).encode(),
                                  headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(400, error.exception.code)
                self.assertEqual(1, len(read_reviews(bundle)))
                journal = review_path(bundle)
                journal.write_text(journal.read_text().replace("request_changes", "reject_evidence"))
                from yk_evidence.pipeline import verify
                self.assertTrue(any("review journal" in item for item in verify(bundle)))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_contract_demo_exposes_verified_new_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = contract_run(Path(root), "contract-demo", seed=42)
            server = build_server(bundle, host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                with urlopen(url + "/api/manifest") as response:
                    self.assertEqual("0.3", json.load(response)["schema_version"])
                with urlopen(url + "/api/artifacts/risk_scores.csv") as response:
                    self.assertIn("model_id,transaction_id", response.read().decode())
                with urlopen(url + "/") as response:
                    self.assertIn("Decision impact at a fixed threshold", response.read().decode())
                with urlopen(url + "/api/artifacts/decision_impact.json") as response:
                    self.assertEqual(3, len(json.load(response)["policy_scenarios"]))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_contract_get_rejects_foreign_host(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = contract_run(Path(root), "contract-host", seed=42)
            server = build_server(bundle, host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                with urlopen(url + "/api/artifacts/finance_memo.json") as response:
                    self.assertEqual(200, response.status)
                for path in ("/", "/api/manifest", "/api/artifacts/finance_memo.json"):
                    with self.subTest(path=path):
                        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                        try:
                            connection.putrequest("GET", path, skip_host=True)
                            connection.putheader("Host", f"unrelated.example:{server.server_port}")
                            connection.endheaders()
                            response = connection.getresponse()
                            response.read()
                            self.assertEqual(403, response.status)
                        finally:
                            connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_contract_get_rechecks_bundle_after_start(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = contract_run(Path(root), "contract-tamper", seed=42)
            server = build_server(bundle, host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                with urlopen(url + "/api/artifacts/finance_memo.json") as response:
                    self.assertEqual(200, response.status)
                artifact = bundle / "finance_memo.json"
                artifact.write_bytes(artifact.read_bytes() + b" ")
                for path in ("/", "/api/manifest", "/api/artifacts/finance_memo.json"):
                    with self.subTest(path=path):
                        with self.assertRaises(HTTPError) as error:
                            urlopen(url + path)
                        self.assertEqual(409, error.exception.code)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_cli_exposes_local_serve_command(self):
        result = subprocess.run([sys.executable, "-m", "yk_evidence", "serve", "--help"], capture_output=True, text=True)
        self.assertEqual(0, result.returncode)
        self.assertIn("bundle", result.stdout)

    def test_read_only_demo_serves_verified_bundle_and_blocks_other_paths(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = integrated_run(Path(root), "demo", seed=42)
            server = build_server(bundle, host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                with urlopen(url + "/") as response:
                    page = response.read().decode()
                    self.assertIn("Governance Evidence Bundle", page)
                    self.assertIn("Synthetic benchmark", page)
                with urlopen(url + "/api/manifest") as response:
                    manifest = json.load(response)
                    self.assertEqual("0.2", manifest["schema_version"])
                with self.assertRaises(HTTPError) as error:
                    urlopen(url + "/api/artifacts/../VERSION")
                self.assertEqual(404, error.exception.code)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_tampered_bundle_cannot_start_demo(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = integrated_run(Path(root), "demo", seed=42)
            (bundle / "summary.md").write_text("tampered")
            with self.assertRaises(ValueError):
                build_server(bundle, host="127.0.0.1", port=0)


if __name__ == "__main__":
    unittest.main()
