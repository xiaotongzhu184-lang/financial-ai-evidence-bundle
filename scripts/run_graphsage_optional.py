"""Run the optional GraphSAGE comparison on a verified integrated bundle."""

import argparse
import json
from pathlib import Path

from yk_evidence.graph_benchmark import run_graph_benchmark
from yk_evidence.pipeline import verify


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    errors = verify(args.bundle)
    if errors:
        parser.error("bundle verification failed: " + "; ".join(errors))
    source = json.loads((args.bundle / "synthetic_transactions.json").read_text(encoding="utf-8"))
    result = run_graph_benchmark(source["rows"], seed=source["seed"], epochs=args.epochs)
    result["input_bundle_run_id"] = args.bundle.name
    result["input_manifest_sha256"] = __import__("hashlib").sha256((args.bundle / "manifest.json").read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("output already exists")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
