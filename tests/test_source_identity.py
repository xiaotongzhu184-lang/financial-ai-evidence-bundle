"""Vendored Python affects the identity of an installed source tree."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yk_evidence import research


class SourceIdentityTests(unittest.TestCase):
    def test_nested_runtime_changes_affect_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'research.py'
            source.write_text('# runtime\n')
            vendor = root / '_vendor'
            vendor.mkdir()
            module = vendor / 'psi.py'
            module.write_text('value = 1\n')
            with patch.object(research, '__file__', str(source)), patch.object(research, 'ROOT', root):
                before = research.code_identity()
                module.write_text('value = 2\n')
                after = research.code_identity()
                self.assertNotEqual(before['code_sha256'], after['code_sha256'])
                self.assertEqual('source-sha256:' + after['code_sha256'], after['code_revision'])
                (vendor / '__pycache__').mkdir()
                (vendor / '__pycache__' / 'psi.pyc').write_bytes(b'cache')
                self.assertEqual(after, research.code_identity())
