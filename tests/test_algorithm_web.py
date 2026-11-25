"""Real HTTP acceptance for algorithm evidence; no mocked model fitting."""
import http.client
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from urllib.parse import urlencode
import zipfile

class AlgorithmWebTests(unittest.TestCase):
    def test_verified_algorithm_can_be_reviewed_and_exported_without_changing_evidence(self):
        from yk_evidence.algorithm import algorithm_smoke
        from yk_evidence.demo import build_server
        from yk_evidence.pipeline import verify
        with tempfile.TemporaryDirectory() as tmp:
            bundle = algorithm_smoke(Path(tmp), 'web-smoke', load_rows=32)
            manifest_before=(bundle/'bundle_manifest.json').read_bytes()
            self.assertEqual([],verify(bundle))
            server=build_server(bundle,port=0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                c=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=90)
                c.request('GET','/');r=c.getresponse();page=r.read().decode();self.assertEqual(200,r.status)
                self.assertIn('Prediction delivery recovery',page)
                self.assertIn('Individual model delivery attempts',page)
                self.assertLess(page.index('Prediction delivery recovery'),
                                page.index('Unprotected 0.5 control probes (before recovery)'))
                for artifact in ('resilience_policy.json','resilience_evidence.json'):
                    c.request('GET','/api/artifacts/'+artifact)
                    r=c.getresponse();downloaded=r.read();self.assertEqual(200,r.status)
                    self.assertEqual((bundle/artifact).read_bytes(),downloaded)
                token=re.search(r'name="csrf" value="([^"]+)"',page).group(1)
                action=re.search(r'<form[^>]*action="([^"]+)"',page).group(1)
                self.assertEqual('/api/reviews',action)
                self.assertIn('href="/api/export"',page)
                body=urlencode({'csrf':token,'reviewer':'Automated web test','decision':'request_changes','rationale':'Human engineering acceptance is required after automated review.'})
                c.request('POST',action,body,{'Content-Type':'application/x-www-form-urlencoded','Origin':f'http://127.0.0.1:{server.server_port}'})
                r=c.getresponse();r.read();self.assertEqual(303,r.status)
                c.request('GET','/api/export');r=c.getresponse();raw=r.read();self.assertEqual(200,r.status)
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    expected=set(json.loads(manifest_before)['artifacts'])|{'bundle_manifest.json'}
                    self.assertEqual(expected,set(z.namelist()));self.assertIsNone(z.testzip())
                c.close()
            finally:
                server.shutdown();server.server_close();thread.join()
            self.assertEqual(manifest_before,(bundle/'bundle_manifest.json').read_bytes())
            self.assertEqual([],verify(bundle))
