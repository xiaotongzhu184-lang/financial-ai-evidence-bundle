# Governance-Grade Evidence Bundle Demo

[![QA](https://github.com/xiaotongzhu184-lang/financial-ai-evidence-bundle/actions/workflows/qa.yml/badge.svg)](https://github.com/xiaotongzhu184-lang/financial-ai-evidence-bundle/actions/workflows/qa.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-24476B.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-24476B.svg)](LICENSE)

Local research software for examining financial-source extracts and transaction-model experiments. A researcher or model-risk reviewer can run an experiment, inspect its inputs and limits, compare saved scores, replay controlled failures, export the evidence, and record a review. The software does not make a bank's release decision or process payments.

“Governance-grade” is the historical project name and a design objective, not a certification. The current software release candidate is **0.7.2rc1**.

## Quick start

Use Python 3.12. From the extracted source directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps .
yk-evidence project-smoke --output runs --run-id first-review --load-rows 10000
yk-evidence verify runs/first-review
yk-evidence serve runs/first-review --port 8765
```

On Windows, create the environment with `py -3.12 -m venv .venv`, activate it with `.\.venv\Scripts\Activate.ps1`, and then run the remaining commands. Open **http://127.0.0.1:8765/** while `serve` is running. Press Ctrl-C to stop the local server. Each completed run needs a new run ID.

The smoke command fits real models on generated data and runs literal retrieval on generated text. It verifies the software path, including recovery, export and review. Its metrics are not evidence of real fraud-detection performance.

Run the full local acceptance path with:

```bash
python scripts/project_acceptance.py --output acceptance-run
python scripts/run_tests.py
```

`run_tests.py` prepares the process-local OpenMP library path needed by the packaged macOS command-line workflow. Directly importing `xgboost` from a separate macOS Python process can still require an installed OpenMP runtime; see the upstream XGBoost installation guide.

## Inputs and outputs

| Command | Input | Output and intended use |
|---|---|---|
| `project-run` | A separately obtained, hash-checked ULB-format credit-card snapshot and its metadata; a declared financial text corpus and questions | Four calibrated models, a validation-frozen fusion policy, scores, model comparisons, review-capacity calculations, drift and recovery evidence, and cited literal passages in one verified review bundle. |
| `project-assemble` | An existing verified 0.6 algorithm bundle, a corpus and questions | The same top-level bundle without retraining. The original training files, run ID and hashes remain intact. |
| `score-batch` | A trusted saved-model bundle and an unlabeled batch in the same transformed feature coordinates | Per-model and fused scores plus a capacity-limited review queue. The command does not retrain, choose a new threshold or calculate accuracy. |
| `verify` / `serve` | A saved evidence bundle | Recalculated consistency checks; a loopback-only HTML review; an allowlisted ZIP export; and a separate, manifest-bound review journal. |

The financial passages and transaction experiment have independent sources. They share one evidence review; the project does not invent a relationship between a public-company filing and anonymous card transactions. The financial path uses TF-IDF retrieval with exact citations and an insufficient-evidence response. It is not a generative RAG system.

## Historical benchmark path

The raw ULB/Worldline CSV is not distributed in this repository. Publisher metadata declares ODbL 1.0 for the database and DbCL 1.0 for its contents. The helper downloads a known TensorFlow tutorial mirror and verifies the exact SHA-256 used by the project. Read the [data-access note](docs/project/ULB_DATA_ACCESS.md) and [evaluation protocol](docs/project/TABULAR_EVALUATION_PROTOCOL.md) before using it.

```bash
python scripts/fetch_ulb.py --output /path/to/ulb-data
yk-evidence project-run --csv /path/to/ulb-data/creditcard.csv \
  --metadata /path/to/ulb-data/metadata.json \
  --corpus src/yk_evidence/assets/data/financial/corpus_v04_sec_companyfacts.json \
  --questions examples/financial_questions.json \
  --output runs --run-id historical-review --seed 42 --load-rows 100000
yk-evidence verify runs/historical-review
```

The adapter recognizes the inspected ULB snapshot and schema; it is not a general corporate-data importer. [Batch-scoring instructions](docs/INFERENCE_PROTOCOL.md) explain why an unrelated institution's raw transaction table cannot be substituted for the anonymous `V1`–`V28` feature coordinates.

## Architecture

- `project.py` assembles the top-level evidence envelope and verifies source relationships.
- `algorithm.py`, `fusion.py` and `resilience.py` implement training, validation-frozen selection, hard-cap review queues and controlled score-delivery recovery.
- `inference.py` performs explicitly trusted saved-model loading and separate unlabeled batch scoring.
- `retrieval.py` performs literal financial-text retrieval and reconstructs citation scores.
- `research_view.py`, `demo.py` and `review.py` provide loopback-only viewing, verified downloads and local review records.

The final output directory appears only after the required checks pass. A missing, modified or semantically inconsistent artifact fails verification. `verify` does not deserialize executable model files. Hashes establish internal consistency; they do not prove source authenticity, reviewer identity or a trusted timestamp.

## Limits

- Real-data results concern an anonymized historical benchmark. There is no institution-specific pilot, current-stream validation or production authorization. `production_approval: blocked` records this evidence boundary; it does not mean the command failed.
- The bundled corpus is a small SEC Company Facts-derived extract. General PDF ingestion, generative financial advice and broad filing-quality evaluation are outside the implemented path.
- The graph experiment is a separately labelled synthetic path. The ULB table has no customer or merchant network identifiers.
- Controlled recovery tests do not cover every upstream feature or model failure. Synthetic load repeats features to measure execution load; it does not create independent accuracy observations.
- Saved model files can execute code when deserialized. `score-batch --trust-models --same-feature-space` therefore requires an explicit trust decision. A checksum and matching column names do not make an unknown model safe.
- The release does not claim a new algorithmic accuracy gain. Stronger individual baselines and negative findings remain visible in the evidence.

## Reproducibility and security

The [0.7 evidence contract](docs/project/PROJECT_BUNDLE_V07.md), [model card](docs/project/MODEL_CARD.md), [data card](docs/project/DATA_CARD.md), [provenance record](PROVENANCE.md), [third-party notices](THIRD_PARTY_NOTICES.md), and [CycloneDX software bill of materials](SBOM.cdx.json) describe the implemented boundaries. Dependencies are pinned in `requirements.lock`. Linux, macOS and Windows jobs execute the test and acceptance paths in GitHub Actions; a passing job applies only to its exact commit and environment.

Security-sensitive reports should follow [SECURITY.md](SECURITY.md). This source release does not establish a DOI, copyright registration, regulatory certification or production approval.

## Citation and license

Xiaotong Zhu maintains this release. Citation metadata, including ORCID [0009-0000-4680-095X](https://orcid.org/0009-0000-4680-095X), is provided in [`CITATION.cff`](CITATION.cff).

Original project code and documentation are released under the Apache License 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Third-party components, public-source extracts and separately obtained datasets retain their own terms as described in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`PROVENANCE.md`](PROVENANCE.md), and the relevant data cards.
