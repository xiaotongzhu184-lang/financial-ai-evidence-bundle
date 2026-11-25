"""Integration tests use fitted smoke models, real HTTP and adversarial bundles."""
import hashlib
import csv
import http.client
import io
import json
import re
import shutil
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from urllib.parse import urlencode

from yk_evidence.pipeline import verify
from yk_evidence.demo import build_server
from yk_evidence.research import research_run, research_smoke


def rehash(folder, name):
    path = folder / 'bundle_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['artifacts'][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


class ResearchIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.reference = research_smoke(cls.base, 'reference', seed=42, load_rows=400)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.bundle = Path(self.work.name) / 'reference'
        shutil.copytree(self.reference, self.bundle)

    def tearDown(self):
        self.work.cleanup()

    def test_completed_smoke_is_semantically_verified_without_raw_data(self):
        self.assertEqual([], verify(self.bundle))
        gate = json.loads((self.bundle / 'release_gate.json').read_text())
        self.assertEqual('pass', gate['evidence_completeness']['status'])
        self.assertEqual('blocked', gate['production_approval']['status'])
        self.assertEqual('smoke_only', gate['research_suitability']['status'])
        self.assertFalse((self.bundle / 'creditcard.csv').exists())
        self.assertIn('synthetic_smoke_fixture', (self.bundle / 'review_report.html').read_text())

    def test_source_error_has_no_completed_or_partial_bundle(self):
        root = Path(self.work.name) / 'invalid'
        with self.assertRaises((ValueError, OSError)):
            research_run(root, 'bad', Path('absent.csv'),
                         json.loads((self.reference / 'source_metadata.json').read_text()),
                         json.loads((self.reference / 'corpus.json').read_text()), ['revenue'])
        self.assertFalse((root / 'bad').exists())
        self.assertEqual([], list(root.iterdir()) if root.exists() else [])

    def test_no_overwrite_and_safe_run_id(self):
        with self.assertRaises(FileExistsError):
            research_smoke(self.base, 'reference', load_rows=400)
        with self.assertRaises(ValueError):
            research_smoke(self.base, '../unsafe', load_rows=400)

    def test_generated_csv_cannot_claim_known_public_ulb_source(self):
        source = Path(self.work.name) / 'fake.csv'
        with source.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['Time', *(f'V{i}' for i in range(1, 29)), 'Amount', 'Class'])
            for index in range(400):
                label = int(index % 8 == 0)
                writer.writerow([index // 2, *[label + index % 7 / 10] * 28, index % 100, label])
        metadata = {'classification': 'public_ulb_creditcard', 'dataset_name': 'Claimed ULB',
                    'source_url': 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud',
                    'license_status': 'claimed ODbL/DbCL',
                    'csv_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        root = Path(self.work.name) / 'forged'
        with self.assertRaisesRegex(ValueError, 'ULB'):
            research_run(root, 'false-public', source, metadata,
                         json.loads((self.reference / 'corpus.json').read_text()), ['revenue'], load_rows=400)
        self.assertFalse((root / 'false-public').exists())

    def test_public_identity_rejects_wrong_url_and_conflicting_hash_aliases(self):
        from yk_evidence.research import validate_source_identity
        digest = '76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89'
        profile = {'source_sha256': digest, 'source_bytes': 150828752,
                   'row_count': 284807, 'positive_count': 492,
                   'time_range_seconds': [0.0, 172792.0]}
        metadata = {'classification': 'public_ulb_creditcard',
                    'source_url': 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'}
        for field in ('csv_sha256', 'source_sha256'):
            validate_source_identity({**metadata, field: digest}, profile)
        with self.assertRaisesRegex(ValueError, 'ULB'):
            validate_source_identity({**metadata, 'source_sha256': digest,
                                      'source_url': 'https://example.org/pretend-ulb'}, profile)
        with self.assertRaisesRegex(ValueError, 'hash'):
            validate_source_identity({**metadata, 'source_sha256': digest, 'csv_sha256': '0' * 64}, profile)
        with self.assertRaisesRegex(ValueError, 'ULB'):
            validate_source_identity({**metadata, 'source_sha256': digest}, {**profile, 'row_count': 400})

    def test_reclassified_smoke_cannot_pass_after_all_manifests_and_report_are_rewritten(self):
        from yk_evidence.research import policy_for
        from yk_evidence.research_view import render_report
        source = json.loads((self.bundle / 'source_metadata.json').read_text())
        source['classification'] = 'public_ulb_creditcard'
        source['source_url'] = 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'
        (self.bundle / 'source_metadata.json').write_text(json.dumps(source))
        summary = json.loads((self.bundle / 'evaluation_summary.json').read_text())
        summary['dataset']['metadata'] = source
        (self.bundle / 'evaluation_summary.json').write_text(json.dumps(summary))
        metadata = json.loads((self.bundle / 'run_metadata.json').read_text())
        metadata['source_classification'] = 'public_ulb_creditcard'
        (self.bundle / 'run_metadata.json').write_text(json.dumps(metadata))
        (self.bundle / 'release_gate.json').write_text(json.dumps(policy_for('public_ulb_creditcard')))
        (self.bundle / 'review_report.html').write_bytes(render_report(self.bundle))
        inner_path = self.bundle / 'evaluation_manifest.json'
        inner = json.loads(inner_path.read_text())
        inner['artifacts'] = {name: hashlib.sha256((self.bundle / name).read_bytes()).hexdigest() for name in inner['artifacts']}
        inner_path.write_text(json.dumps(inner))
        outer_path = self.bundle / 'bundle_manifest.json'
        outer = json.loads(outer_path.read_text()); outer['source_classification'] = 'public_ulb_creditcard'
        outer['artifacts'] = {name: hashlib.sha256((self.bundle / name).read_bytes()).hexdigest() for name in outer['artifacts']}
        outer_path.write_text(json.dumps(outer))
        self.assertTrue(any('ULB' in error for error in verify(self.bundle)))

    def test_rehashed_retrieval_and_policy_claims_are_rejected(self):
        path = self.bundle / 'retrieval_evidence.json'
        obj = json.loads(path.read_text())
        obj['queries'][0]['answer_excerpt'] = 'fabricated financial result'
        path.write_text(json.dumps(obj))
        rehash(self.bundle, path.name)
        self.assertTrue(verify(self.bundle))
        shutil.copyfile(self.reference / path.name, path)
        rehash(self.bundle, path.name)
        gate = self.bundle / 'release_gate.json'
        obj = json.loads(gate.read_text())
        obj['production_approval']['status'] = 'approved'
        gate.write_text(json.dumps(obj))
        rehash(self.bundle, gate.name)
        self.assertTrue(verify(self.bundle))

    def test_rehashed_metric_and_profile_mismatch_are_rejected(self):
        summary = self.bundle / 'evaluation_summary.json'
        original = summary.read_bytes()
        obj = json.loads(original)
        obj['models']['logistic_regression']['metrics']['test']['recall'] = .123456789
        summary.write_text(json.dumps(obj))
        evaluation_manifest = self.bundle / 'evaluation_manifest.json'
        inner = json.loads(evaluation_manifest.read_text())
        inner['artifacts'][summary.name] = hashlib.sha256(summary.read_bytes()).hexdigest()
        evaluation_manifest.write_text(json.dumps(inner))
        rehash(self.bundle, summary.name); rehash(self.bundle, evaluation_manifest.name)
        self.assertTrue(verify(self.bundle))
        shutil.copyfile(self.reference / summary.name, summary)
        shutil.copyfile(self.reference / evaluation_manifest.name, evaluation_manifest)
        rehash(self.bundle, summary.name); rehash(self.bundle, evaluation_manifest.name)
        profile = self.bundle / 'source_profile.json'
        obj = json.loads(profile.read_text()); obj['positive_count'] += 1
        profile.write_text(json.dumps(obj)); rehash(self.bundle, profile.name)
        self.assertTrue(verify(self.bundle))

    def test_changed_review_journal_fails_verification(self):
        from yk_evidence.review import append_review, review_path
        append_review(self.bundle, 'Engineer', 'request_changes', 'Please verify the calibration assumptions.')
        path = review_path(self.bundle)
        obj = json.loads(path.read_text()); obj['decision'] = 'accept_demo_evidence'
        path.write_text(json.dumps(obj) + '\n')
        self.assertTrue(verify(self.bundle))

    def test_unsafe_extra_and_unknown_schema_fail_closed(self):
        for extra in ('secret.txt', 'directory'):
            path = self.bundle / extra
            path.mkdir() if extra == 'directory' else path.write_text('secret')
            self.assertTrue(verify(self.bundle))
            path.rmdir() if path.is_dir() else path.unlink()
        p = self.bundle / 'source_profile.json'
        p.unlink()
        p.symlink_to(self.reference / 'source_profile.json')
        self.assertTrue(verify(self.bundle))

        manifest_path = self.bundle / 'bundle_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['schema_version'] = '9.9'
        manifest_path.write_text(json.dumps(manifest))
        self.assertTrue(verify(self.bundle))

    def test_nonobject_manifest_is_rejected_without_crashing(self):
        (self.bundle / 'bundle_manifest.json').write_text('[]')
        self.assertTrue(verify(self.bundle))

    def test_http_review_csrf_export_and_post_start_tamper(self):
        server = build_server(self.bundle, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = http.client.HTTPConnection('127.0.0.1', server.server_port)
            client.request('GET', '/')
            response = client.getresponse(); page = response.read().decode()
            self.assertEqual(200, response.status)
            self.assertIn('Production approval', page)
            self.assertIn('Alerts per 10,000', page)
            self.assertIn('Missed cases', page)
            self.assertIn('Review minutes and monetary costs were not measured', page)
            self.assertIn('No institution-specific handling time was supplied', page)
            summary = json.loads((self.bundle / 'evaluation_summary.json').read_text())
            body = re.search(r'<tbody>(.*?)</tbody>', page, re.S).group(1)
            for row in re.findall(r'<tr>(.*?)</tr>', body, re.S):
                values = re.findall(r'<td>(.*?)</td>', row, re.S)
                counts = summary['models'][values[0]]['metrics']['test']['counts']
                alerts = counts['tp'] + counts['fp']
                self.assertEqual(str(alerts), values[-3])
                self.assertEqual(f'{alerts / counts["rows"] * 10000:.5f}', values[-2])
                self.assertEqual(str(counts['fn']), values[-1])
            token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
            payload = urlencode({'csrf': token, 'reviewer': 'Engineer',
                                 'decision': 'accept_demo_evidence',
                                 'rationale': 'Checked saved evidence and its limitations.'})
            headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Origin': 'https://evil.example'}
            client.request('POST', '/api/reviews', body=payload, headers=headers)
            response = client.getresponse(); response.read()
            self.assertEqual(403, response.status)
            headers['Origin'] = f'http://127.0.0.1:{server.server_port}'
            client.request('POST', '/api/reviews', body=payload.replace(token, 'wrong'), headers=headers)
            response = client.getresponse(); response.read()
            self.assertEqual(400, response.status)
            client.request('POST', '/api/reviews', body=payload, headers=headers)
            response = client.getresponse(); response.read()
            self.assertEqual(303, response.status)
            self.assertEqual([], verify(self.bundle))
            client.request('GET', '/api/export')
            response = client.getresponse(); payload = response.read()
            self.assertEqual(200, response.status)
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                self.assertIn('bundle_manifest.json', archive.namelist())
                self.assertNotIn('creditcard.csv', archive.namelist())
                self.assertFalse(any('reviews' in name for name in archive.namelist()))
            (self.bundle / 'source_profile.json').write_text('{}')
            client.request('GET', '/api/export')
            response = client.getresponse(); response.read()
            self.assertEqual(409, response.status)
            client.close()
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_http_review_accepts_long_chinese_rationale_and_rejects_character_overflow(self):
        from yk_evidence.review import read_reviews
        server = build_server(self.bundle, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
            client.request('GET', '/')
            response = client.getresponse(); page = response.read().decode('utf-8')
            self.assertEqual(200, response.status)
            token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
            headers = {'Content-Type': 'application/x-www-form-urlencoded',
                       'Origin': f'http://127.0.0.1:{server.server_port}'}

            rationale = '核' * 2000
            payload = urlencode({'csrf': token, 'reviewer': '审查负责人',
                                 'decision': 'request_changes', 'rationale': rationale})
            self.assertGreater(len(payload.encode('ascii')), 8192)
            client.request('POST', '/api/reviews', body=payload, headers=headers)
            response = client.getresponse(); response.read()
            self.assertEqual(303, response.status)
            self.assertEqual(rationale, read_reviews(self.bundle)[0]['rationale'])

            payload = urlencode({'csrf': token, 'reviewer': '审查负责人',
                                 'decision': 'request_changes', 'rationale': '核' * 2001})
            client.request('POST', '/api/reviews', body=payload, headers=headers)
            response = client.getresponse(); error = response.read().decode('utf-8')
            self.assertEqual(400, response.status)
            self.assertIn('10-2000 printable characters', error)
            self.assertEqual(1, len(read_reviews(self.bundle)))
            client.close()
        finally:
            server.shutdown(); server.server_close(); thread.join()
