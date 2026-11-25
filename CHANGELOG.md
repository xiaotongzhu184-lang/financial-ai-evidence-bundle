# Changelog

## 0.7.2rc1 — public-source hardening, 2026-11-25

- Added Apache-2.0 project licensing, citation metadata, a security policy, contribution guidance and monthly dependency updates.
- Updated Certifi to 2026.11.20 after dependency auditing identified `PYSEC-2026-4320`; updated DuckDB to 2.0.1 and Cloudpickle to 3.1.3 after compatibility testing.
- Pinned GitHub Actions and the Docker base image by full cryptographic revision.
- Replaced an optimization-sensitive `assert` with an explicit runtime check and documented the fixed-origin dataset downloader guard.
- Removed account, publication-planning and non-technical delivery records from the public source tree. Raw third-party transaction data, fitted models, run outputs and private review records remain excluded.
- Locate review journals by the immutable manifest run ID so a renamed evidence-bundle directory keeps its review history.
- Accept the documented 2,000-character review rationale when it contains URL-encoded Unicode text, while retaining a bounded 32 KiB request limit and explicit over-limit errors.
- Add file-level and real HTTP regressions for renamed bundles and long UTF-8 review text.
- Keep the evidence schema, estimator, threshold and historical benchmark results unchanged.

## 0.7.1rc1 — portable engineer verification

- Added one test command for macOS, Linux and Windows: `python scripts/run_tests.py`.
- Reused scikit-learn's installed OpenMP runtime for the macOS test subprocess, matching the existing CLI behavior without a persistent shell setting.
- Added a regression probe for the native XGBoost import and retained the project, tamper, recovery, retrieval and batch-scoring suite.

## 0.7.0rc1 — integrated review candidate

- Added one project envelope for the transaction experiment and independent financial retrieval, including joint semantic verification and web export.
- Added explicit trusted frozen-model batch scoring on unlabeled, same-coordinate input.
- Fixed stale verification and foreign-Host access in the legacy viewer.
- Moved macOS native runtime setup into the installed CLI and added end-to-end project acceptance.

## 0.6.0rc1 — bounded prediction recovery

- Added bounded Tenacity retries, immutable producer receipts, row-ID alignment and per-scenario model quarantine.
- Froze healthy-subset fallback policies on validation data and withheld decisions when fewer than two models remained healthy.
- Added controlled delivery scenarios, semantic transcript reconstruction and an independent result audit.

## 0.5.0rc1 — algorithm experiment extension

- Added an XGBoost CPU challenger alongside three calibrated scikit-learn baselines.
- Added validation-only convex fusion selection, capacity-limited review queues, ablations and score-space fault probes.
- Adapted the pinned Evidently PSI numerical core while retaining its Apache-2.0 license and modification notice.
- Kept ULB comparisons explicitly retrospective and exploratory; no novel-algorithm or production-approval claim was introduced.

## 0.4.0rc1 — historical-data research and review

- Added typed DuckDB source profiling and a hash-pinned ULB downloader.
- Added chronological model evaluation, policy thresholds, saved calibration transforms, stress probes and bounded load measurements.
- Added multi-issuer SEC Company Facts excerpts and deterministic cited retrieval with abstention for unsupported questions.

## 0.3.0rc1 — evidence contract

- Added the 0.3 schema, four-baseline evidence export, held-out metric recomputation and negative tamper tests.
- Made source and wheel runs record an exact Git revision or source SHA-256.
- Kept the model-release gate blocked for generated data and absent formal RAG evaluation.

## 0.2.0.dev0 — first integrated development slice

- Added a fixed SEC financial-fact example with accession, unit and source trace.
- Added a generated transaction benchmark, chronological split, saved holdout predictions and explicit failing gates.
- Added a local read-only demo, SHA-256 evidence manifest and optional GraphSAGE experiment.
