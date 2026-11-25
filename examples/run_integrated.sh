#!/usr/bin/env sh
set -eu
RUN_ID="${1:-contract-example}"
python3 -m yk_evidence contract-run --output runs --run-id "$RUN_ID" --seed 42
python3 -m yk_evidence verify "runs/$RUN_ID"
