import tempfile
import unittest
import importlib.util
from pathlib import Path

from yk_evidence.pipeline import integrated_run


@unittest.skipUnless(importlib.util.find_spec("matplotlib") is not None, "optional Matplotlib not installed")
class FigureTest(unittest.TestCase):
    def test_figures_are_rendered_only_from_verified_bundle(self):
        from scripts.render_validation_figures import render_figures
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bundle = integrated_run(root, "figure-source", seed=42)
            output = root / "figures"
            names = render_figures(bundle, output)
            self.assertEqual({"precision_recall.png", "stability.png", "stress_replay.png"}, set(names))
            self.assertTrue(all((output / name).stat().st_size > 5000 for name in names))
            (bundle / "benchmark.json").write_text("tampered")
            with self.assertRaises(ValueError):
                render_figures(bundle, root / "tampered-figures")


if __name__ == "__main__":
    unittest.main()
