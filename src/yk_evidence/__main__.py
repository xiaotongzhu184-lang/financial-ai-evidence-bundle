"""Command-line entry point for research and synthetic evidence bundles."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import contract_run, integrated_run, run, verify


def main() -> int:
    from .runtime import prepare_cli_runtime
    prepare_cli_runtime()
    parser = argparse.ArgumentParser(description="Financial research evidence and local model review")
    sub = parser.add_subparsers(dest="command", required=True)
    scoring = sub.add_parser('score-batch', help='Use trusted frozen 0.6 models on an unlabeled batch in the SAME feature coordinates')
    scoring.add_argument('--model-bundle', type=Path, required=True)
    scoring.add_argument('--csv', type=Path, required=True)
    scoring.add_argument('--output', type=Path, default=Path('runs'))
    scoring.add_argument('--run-id', required=True)
    scoring.add_argument('--trust-models', action='store_true', help='Allow loading model files from a trusted source; pickle can execute code')
    scoring.add_argument('--same-feature-space', action='store_true', help='Attest that V1..V28 use the training data transformation, not unrelated or raw features')
    project = sub.add_parser('project-run', help='Train the four-model path and bind financial retrieval in one review bundle')
    project.add_argument('--csv', type=Path, required=True)
    project.add_argument('--metadata', type=Path, required=True)
    project.add_argument('--review-budget', type=float, default=.005)
    assembled = sub.add_parser('project-assemble', help='Bind an existing verified 0.6 model run and financial retrieval; no retraining')
    assembled.add_argument('--risk-bundle', type=Path, required=True)
    project_fixture = sub.add_parser('project-smoke', help='Run both project components offline on generated test data')
    for command in (project, assembled):
        command.add_argument('--corpus', type=Path, required=True)
        command.add_argument('--questions', type=Path, required=True)
    for command in (project, project_fixture):
        command.add_argument('--seed', type=int, default=42)
        command.add_argument('--load-rows', type=int, default=100_000)
    for command in (project, assembled, project_fixture):
        command.add_argument('--output', type=Path, default=Path('runs'))
        command.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('project-%Y%m%dT%H%M%SZ'))
    algorithm = sub.add_parser("algorithm-run", help="Run the four-model exploratory ensemble and capped review experiment")
    algorithm.add_argument("--csv", type=Path, required=True)
    algorithm.add_argument("--metadata", type=Path, required=True)
    algorithm.add_argument("--review-budget", type=float, default=.005,
                           help="Review-capacity fraction; scenario assumption, default 0.005")
    algorithm_smoke_parser = sub.add_parser("algorithm-smoke", help="Run the actual algorithm path on a generated fixture")
    for command in (algorithm, algorithm_smoke_parser):
        command.add_argument("--output", type=Path, default=Path("runs"))
        command.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("algorithm-%Y%m%dT%H%M%SZ"))
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--load-rows", type=int, default=100_000)
    research = sub.add_parser("research-run", help="Evaluate a declared ULB-schema source and financial corpus")
    research.add_argument("--csv", type=Path, required=True)
    research.add_argument("--metadata", type=Path, required=True)
    research.add_argument("--corpus", type=Path, required=True)
    research.add_argument("--questions", type=Path, required=True, help="JSON list of query strings")
    smoke = sub.add_parser("research-smoke", help="Fit actual models on a generated schema 0.4 fixture offline")
    for command in (research, smoke):
        command.add_argument("--output", type=Path, default=Path("runs"))
        command.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("research-%Y%m%dT%H%M%SZ"))
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--load-rows", type=int, default=100_000)
    research.add_argument("--fpr-ceiling", type=float, default=.01)
    create = sub.add_parser("run")
    create.add_argument("--output", type=Path, default=Path("runs"))
    create.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ"))
    integrated = sub.add_parser("integrated-run", help="Run SEC finance and synthetic risk benchmark")
    integrated.add_argument("--output", type=Path, default=Path("runs"))
    integrated.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("integrated-%Y%m%dT%H%M%SZ"))
    integrated.add_argument("--seed", type=int, default=42)
    contract = sub.add_parser("contract-run", help="Run the integrated demo and write a verified 0.3 evidence bundle")
    contract.add_argument("--output", type=Path, default=Path("runs"))
    contract.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("contract-%Y%m%dT%H%M%SZ"))
    contract.add_argument("--seed", type=int, default=42)
    contract.add_argument("--graph-epochs", type=int, default=0,
                          help="Include an optional GraphSAGE experiment in the verified bundle")
    contract.add_argument("--transactions-csv", type=Path,
                          help="Import a rights-declared synthetic transaction CSV")
    contract.add_argument("--transactions-metadata", type=Path,
                          help="JSON with source, license, classification and exact CSV SHA-256")
    check = sub.add_parser("verify")
    check.add_argument("bundle", type=Path)
    serve = sub.add_parser("serve", help="Review a verified bundle locally and save demo decisions")
    serve.add_argument("bundle", type=Path)
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        if args.command == 'score-batch':
            from .inference import run_inference
            bundle = run_inference(args.model_bundle, args.csv, args.output, args.run_id,
                                   trusted_model_load=args.trust_models,
                                   feature_space_attested=args.same_feature_space)
            print(bundle)
            return 0
        if args.command in ('project-run', 'project-assemble', 'project-smoke'):
            from .project import project_run, assemble_project, project_smoke
            from .research import read
            if args.command == 'project-smoke':
                bundle = project_smoke(args.output, args.run_id, args.seed, args.load_rows)
            elif args.command == 'project-assemble':
                bundle = assemble_project(args.output, args.run_id, args.risk_bundle,
                                          read(args.corpus), read(args.questions))
            else:
                bundle = project_run(args.output, args.run_id, args.csv, read(args.metadata),
                                     read(args.corpus), read(args.questions), args.seed,
                                     args.load_rows, args.review_budget)
            print(bundle)
            return 0
        if args.command in ("algorithm-run", "algorithm-smoke"):
            from .algorithm import algorithm_run, algorithm_smoke
            if args.command == "algorithm-smoke":
                bundle = algorithm_smoke(args.output, args.run_id, args.seed, args.load_rows)
            else:
                metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
                bundle = algorithm_run(args.output, args.run_id, args.csv, metadata,
                                       args.seed, args.load_rows, args.review_budget)
            print(bundle)
            return 0
        if args.command in ("research-run", "research-smoke"):
            from .research import research_run, research_smoke
            if args.command == "research-smoke":
                bundle = research_smoke(args.output, args.run_id, args.seed, args.load_rows)
            else:
                def load(path):
                    return json.loads(path.read_text(encoding="utf-8"))
                bundle = research_run(args.output, args.run_id, args.csv, load(args.metadata),
                                      load(args.corpus), load(args.questions), args.seed,
                                      args.load_rows, args.fpr_ceiling)
            print(bundle)
            return 0
        if args.command == "run":
            bundle = run(args.output, args.run_id)
            print(bundle)
            return 0
        if args.command == "integrated-run":
            bundle = integrated_run(args.output, args.run_id, seed=args.seed)
            print(bundle)
            return 0
        if args.command == "contract-run":
            bundle = contract_run(args.output, args.run_id, seed=args.seed, graph_epochs=args.graph_epochs,
                                  transactions_csv=args.transactions_csv,
                                  transactions_metadata=args.transactions_metadata)
            print(bundle)
            return 0
        if args.command == "serve":
            from .demo import build_server
            server = build_server(args.bundle, port=args.port)
            print(f"Local review demo: http://127.0.0.1:{server.server_port}/", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
            return 0
        errors = verify(args.bundle)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print("bundle verified")
        return 0
    except (FileExistsError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
