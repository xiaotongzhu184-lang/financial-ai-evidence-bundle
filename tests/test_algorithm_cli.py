"""CLI contract independent of fitting or optional runtime loading."""
import subprocess
import sys
import unittest


class AlgorithmCliTests(unittest.TestCase):
    def test_algorithm_commands_have_reproducible_controls(self):
        p = subprocess.run([sys.executable, '-m', 'yk_evidence', 'algorithm-run', '--help'], capture_output=True, text=True)
        self.assertEqual(0, p.returncode, p.stderr)
        for flag in ('--csv', '--metadata', '--seed', '--load-rows', '--review-budget'):
            self.assertIn(flag, p.stdout)

    def test_algorithm_smoke_can_be_discovered_without_loading_model_runtime(self):
        p = subprocess.run([sys.executable, '-m', 'yk_evidence', 'algorithm-smoke', '--help'], capture_output=True, text=True)
        self.assertEqual(0, p.returncode, p.stderr)
        self.assertIn('--run-id', p.stdout)
