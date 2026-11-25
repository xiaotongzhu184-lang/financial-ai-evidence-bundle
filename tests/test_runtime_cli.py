"""The installed entry point must run without a source-tree helper or env setup."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest


class RuntimeCliTests(unittest.TestCase):
    def test_test_launcher_prepares_native_runtime_without_shell_configuration(self):
        launcher = Path(__file__).resolve().parents[1] / 'scripts/run_tests.py'
        environment = {k: v for k, v in os.environ.items()
                       if k not in ('DYLD_LIBRARY_PATH', 'YK_EVIDENCE_OPENMP_READY')}
        result = subprocess.run([sys.executable, str(launcher),
                                 '--pattern', 'native_xgboost_probe.py'],
                                env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn('OK', result.stderr)

    def test_compatibility_launcher_propagates_verification_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(__file__).resolve().parents[1] / 'scripts/run_local.py'
            result = subprocess.run([sys.executable, str(launcher), 'verify', tmp],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(1, result.returncode, result.stderr)
            self.assertIn('missing or invalid manifest', result.stderr)

    def test_direct_cli_runs_project_and_verifies_without_dyld_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = {k: v for k, v in os.environ.items()
                           if k not in ('DYLD_LIBRARY_PATH', 'YK_EVIDENCE_OPENMP_READY')}
            result = subprocess.run([sys.executable, '-m', 'yk_evidence', 'project-smoke',
                '--output', tmp, '--run-id', 'direct', '--load-rows', '16'],
                env=environment, capture_output=True, text=True, timeout=90)
            self.assertEqual(0, result.returncode, result.stderr)
            result = subprocess.run([sys.executable, '-m', 'yk_evidence', 'verify', str(Path(tmp) / 'direct')],
                env=environment, capture_output=True, text=True, timeout=90)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn('bundle verified', result.stdout)
            if os.name != 'nt':
                launcher = Path(__file__).resolve().parents[1] / 'scripts/run_local.py'
                process = subprocess.Popen([sys.executable, str(launcher), 'serve',
                    str(Path(tmp) / 'direct'), '--port', '0'], env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
                try:
                    self.assertIn('http://127.0.0.1:', process.stdout.readline())
                    os.killpg(process.pid, signal.SIGINT)
                    _, stderr = process.communicate(timeout=20)
                    self.assertEqual(0, process.returncode, stderr)
                    self.assertNotIn('Traceback', stderr)
                finally:
                    if process.poll() is None:
                        process.kill(); process.communicate()
