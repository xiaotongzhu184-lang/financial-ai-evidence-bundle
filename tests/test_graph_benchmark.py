import unittest
import importlib.util
import hashlib
import json
import tempfile
from pathlib import Path

from yk_evidence.risk_data import generate_transactions

GRAPH_STACK_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torch_geometric") is not None


@unittest.skipUnless(GRAPH_STACK_AVAILABLE, "optional PyTorch Geometric stack not installed")
class GraphBenchmarkTest(unittest.TestCase):
    def test_graphsage_can_join_contract_bundle_and_metric_tamper_fails(self):
        from yk_evidence.pipeline import contract_run, verify
        with tempfile.TemporaryDirectory() as root:
            bundle = contract_run(Path(root), "graph-contract", seed=42, graph_epochs=2)
            self.assertEqual([], verify(bundle))
            graph_path = bundle / "graphsage_evidence.json"
            graph = json.loads(graph_path.read_text())
            self.assertEqual(100, len(graph["predictions"]))
            graph["metrics"]["pr_auc"] = 0.0
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"][graph_path.name] = hashlib.sha256(graph_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(any("GraphSAGE metrics mismatch" in error for error in verify(bundle)))

    def test_graphsage_emits_test_edge_scores_with_frozen_train_context(self):
        from yk_evidence.graph_benchmark import run_graph_benchmark
        rows = generate_transactions(seed=42, count=400)
        result = run_graph_benchmark(rows, seed=42, epochs=2)
        self.assertEqual("GraphSAGE", result["model_family"])
        self.assertEqual(300, result["training_edge_count"])
        self.assertEqual(100, len(result["predictions"]))
        self.assertEqual([row["transaction_id"] for row in rows[300:]], [row["transaction_id"] for row in result["predictions"]])
        self.assertTrue(all(0 <= item["score"] <= 1 for item in result["predictions"]))
        self.assertIsInstance(result["metrics"]["pr_auc"], float)

    def test_future_holdout_edge_cannot_change_earlier_score(self):
        from yk_evidence.graph_benchmark import run_graph_benchmark
        rows = generate_transactions(seed=42, count=400)
        original = run_graph_benchmark(rows, seed=42, epochs=2)
        modified = [dict(row) for row in rows]
        modified[-1]["amount"] *= 10
        modified[-1]["merchant_id"] = "new-future-merchant"
        changed = run_graph_benchmark(modified, seed=42, epochs=2)
        self.assertAlmostEqual(original["predictions"][0]["score"], changed["predictions"][0]["score"], places=10)


if __name__ == "__main__":
    unittest.main()
