"""Unified runs must bind real component outputs, not merely list their names."""
import http.client
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
import zipfile

from yk_evidence.algorithm import algorithm_smoke
from yk_evidence.pipeline import verify
from yk_evidence.project import assemble_project, project_smoke
from yk_evidence.research import read, sha, write

CORPUS = {'schema_version': '0.4', 'documents': [
    {'id': 'synthetic', 'title': 'Generated finance example',
     'source_url': 'https://example.org/fixture', 'classification': 'synthetic_smoke_fixture',
     'text': 'Acme revenue was USD 100 in fiscal 2025.'}]}


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.risk = algorithm_smoke(cls.root, 'risk', load_rows=16)
        cls.reference = assemble_project(cls.root, 'project', cls.risk, CORPUS,
                                         ['Acme revenue', 'unicorn nebula'])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.temp_case = tempfile.TemporaryDirectory()
        self.root_case = Path(self.temp_case.name)
        self.bundle = self.root_case / 'project'
        shutil.copytree(self.reference, self.bundle)

    def tearDown(self):
        self.temp_case.cleanup()

    def rehash(self, name):
        manifest = read(self.bundle / 'bundle_manifest.json')
        manifest['artifacts'][name] = sha(self.bundle / name)
        write(self.bundle / 'bundle_manifest.json', manifest)

    def test_binds_components_and_explicitly_records_extractive_limit(self):
        self.assertEqual([], verify(self.bundle))
        self.assertEqual((self.risk / 'algorithm_scores.json').read_bytes(),
                         (self.bundle / 'risk/algorithm_scores.json').read_bytes())
        lineage = read(self.bundle / 'lineage.json')
        self.assertEqual(sha(self.risk / 'bundle_manifest.json'), lineage['risk_manifest_sha256'])
        queries = read(self.bundle / 'retrieval_evidence.json')['queries']
        self.assertEqual(['supported', 'insufficient_evidence'], [q['status'] for q in queries])
        gate = read(self.bundle / 'release_gate.json')
        self.assertEqual('blocked', gate['production_approval']['status'])
        self.assertEqual('extractive_only', gate['financial_analysis']['status'])

    def test_missing_component_and_rehashed_false_citation_are_rejected(self):
        (self.bundle / 'risk/algorithm_scores.json').unlink()
        self.assertTrue(verify(self.bundle))
        shutil.copyfile(self.risk / 'algorithm_scores.json', self.bundle / 'risk/algorithm_scores.json')
        result = read(self.bundle / 'retrieval_evidence.json')
        result['queries'][0]['answer_excerpt'] = 'Invented revenue USD 999.'
        write(self.bundle / 'retrieval_evidence.json', result)
        self.rehash('retrieval_evidence.json')
        self.assertTrue(verify(self.bundle))

    def test_rehashed_wrong_lineage_and_extra_files_fail(self):
        lineage = read(self.bundle / 'lineage.json')
        lineage['risk_manifest_sha256'] = '0' * 64
        write(self.bundle / 'lineage.json', lineage)
        self.rehash('lineage.json')
        self.assertTrue(verify(self.bundle))
        shutil.copyfile(self.reference / 'lineage.json', self.bundle / 'lineage.json')
        self.rehash('lineage.json')
        (self.bundle / 'unlisted.txt').write_text('must not export')
        self.assertTrue(verify(self.bundle))

    def test_rehashed_metadata_cannot_claim_joint_feature_training(self):
        metadata = read(self.bundle / 'run_metadata.json')
        metadata['execution'] = 'trained_with_financial_source_as_predictive_feature'
        metadata['code_sha256'] = 'not-a-hash'
        write(self.bundle / 'run_metadata.json', metadata)
        self.rehash('run_metadata.json')
        self.assertTrue(verify(self.bundle))

    def test_failure_is_atomic_and_existing_runs_preserved(self):
        with self.assertRaises(ValueError):
            assemble_project(self.root_case, 'invalid', self.risk, {}, ['revenue'])
        self.assertFalse((self.root_case / 'invalid').exists())
        with self.assertRaises(FileExistsError):
            assemble_project(self.root_case, 'project', self.risk, CORPUS, ['revenue'])
        with self.assertRaises(ValueError):
            assemble_project(self.root_case, '../bad', self.risk, CORPUS, ['revenue'])
        self.assertEqual([], list(self.root_case.glob('.*')))

    def test_web_exports_nested_components_and_rejects_later_corruption(self):
        from yk_evidence.demo import build_server
        server = build_server(self.bundle, port=0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=30)
            conn.request('GET', '/')
            response = conn.getresponse(); body = response.read().decode()
            self.assertEqual(200, response.status)
            self.assertIn('Acme revenue', body)
            self.assertIn('Frozen transaction models', body)
            conn.request('GET', '/api/export')
            response = conn.getresponse(); raw = response.read()
            self.assertEqual(200, response.status)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                self.assertEqual(set(read(self.bundle / 'bundle_manifest.json')['artifacts']) |
                                 {'bundle_manifest.json'}, set(archive.namelist()))
                self.assertIsNone(archive.testzip())
            (self.bundle / 'risk/source_profile.json').write_text('{}')
            conn.request('GET', '/')
            response = conn.getresponse(); response.read()
            self.assertEqual(409, response.status)
            conn.close()
        finally:
            server.shutdown(); server.server_close(); worker.join()

    def test_one_command_smoke_executes_both_paths(self):
        bundle = project_smoke(self.root_case, 'new-smoke', load_rows=16)
        self.assertEqual([], verify(bundle))
        self.assertTrue((bundle / 'risk/model_xgboost.joblib').is_file())
