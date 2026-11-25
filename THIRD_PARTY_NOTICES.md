# Third-party software and data notices

The project-specific work is the orchestration, evidence contracts, source adapters, semantic checks, review workflow and evaluation policy. Algorithms and database engines are attributed to their upstream maintainers. Originality is not measured by renamed upstream files, repository size or added line count.

| Component | Version / use | Upstream license and attribution |
|---|---|---|
| scikit-learn | 1.9.1; logistic regression, random forest, histogram gradient boosting, scaling/imputation pipelines, frozen-estimator sigmoid calibration, metrics and TF-IDF | BSD-3-Clause; scikit-learn developers. Full installed distribution notice: `docs/third_party/SCIKIT_LEARN_COPYING.txt`; https://github.com/scikit-learn/scikit-learn |
| DuckDB | 2.0.1; typed SQL scan of original ULB schema before training, finite-value/count/range checks, configured buffer budget | MIT; DuckDB Foundation. Full notice: `docs/third_party/DUCKDB_LICENSE.txt`; https://github.com/duckdb/duckdb |
| jsonschema | 4.26.0; legacy evidence schema validation | MIT; Julian Berman. Full notice: `docs/third_party/JSONSCHEMA_COPYING.txt`; https://github.com/python-jsonschema/jsonschema |
| NumPy / SciPy / joblib / threadpoolctl | Exact versions in requirements.lock; array operations, numerical routines, trusted fitted-model persistence and bounded native threads | Respective upstream distribution licenses remain applicable; wheels include their own notices |
| PyTorch / PyTorch Geometric | Existing optional synthetic GraphSAGE path; not used to invent an entity graph for ULB | See PROVENANCE.md and optional dependency documentation |

Research alone does not constitute adoption. XGBoost adoption is recorded below; LightGBM, MLflow and Great Expectations remain research candidates. The Evidently numerical core is source-derived code as recorded below; its product framework is not installed.

## Source data

ULB/Worldline Credit Card Fraud Detection has its own database and contents licenses. See `docs/project/ULB_DATA_ACCESS.md`; no raw CSV is included in the source/wheel archive. Prediction evidence is a derived research artifact and remains subject to a separate distribution review. SEC Company Facts extracts record source URLs, accessions, raw response hashes and selected fact records. Financial retrieval text is explicitly derived from structured facts; it is not presented as a verbatim quotation of an entire filing.

The original project code and documentation are licensed under Apache License 2.0 by the copyright holder identified in `NOTICE`. This license does not replace the separate terms that apply to the components and data sources listed here.

## Certifi 2026.11.20

Mozilla trust-store bundle, distributed under MPL-2.0. Used only to validate TLS certificates for the public dataset download; certificate verification remains enabled. Retained notice: docs/third_party/CERTIFI_LICENSE.txt. Source: https://github.com/certifi/python-certifi.

## XGBoost 3.4.1 (0.5 algorithm path)
Official upstream: https://github.com/dmlc/xgboost . Apache-2.0 notice retained in `docs/third_party/XGBOOST_LICENSE.txt` from the installed official wheel. Installed as `xgboost-cpu==3.4.1` on Linux (no NCCL dependency) and `xgboost==3.4.1` elsewhere. No upstream implementation copied or modified. The model configuration, fusion search, review-budget policy and evidence integration are project code. macOS `scripts/run_local.py` may use the already-installed scikit-learn wheel's OpenMP library for that child process; it neither copies that binary nor modifies system settings.

## Evidently PSI source adaptation (0.5 monitoring extension)

Copyright 2021-2023 Evidently AI, Inc. Licensed under Apache License 2.0. The complete upstream license is retained at `src/yk_evidence/_vendor/EVIDENTLY_LICENSE.txt`, beside the derived implementation `src/yk_evidence/_vendor/evidently_psi.py`.

Upstream: https://github.com/evidentlyai/evidently ; pinned commit `34771fe41c08ca074d96a52aaf4a45cd640cbaf5`; original file `src/evidently/legacy/calculations/stattests/psi.py`, function `_psi`; original file SHA-256 `11c5131df994f19e0131c56e883d9fc966087745aa143c91cca71a5a87f83426`.

Modification notice: the NumPy PSI expression was extracted into `psi_components`, with explicit aligned-positive-vector validation and per-bin contribution output. Upstream pandas/registry integration, pooled reference/current binning, zero replacement and the default threshold decision were not copied. Reference-frozen bins, smoothing, chronological window diagnostics and evidence verification are separate project integration work. No full Evidently service or UI is included, and no claim to invent PSI is made.

## Fraud Detection Handbook — methodological reference only

Source: https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook at commit `81cf7d1714bb7b2f5b496407d9055d91dc68dc25`, especially `Chapter_References/shared_functions.py`. Its code, including notebook code, is licensed under GNU GPL version 3. No handbook code is included in the runtime or source distribution. Time-aware assessment and review-budget evaluation were examined as methods; its card-level aggregation and compromised-card removal require entity identifiers and operational feedback absent from ULB. Transaction-row IDs are not substituted for card IDs. This notice records the reference, not an adopted implementation.

## Tenacity 9.2.1 (0.6 resilient scoring)

Upstream: https://github.com/jd/tenacity ; fixed tag 9.2.1 / commit `4410996dd9a96864a2733ce0778bcaa112b72811`. Apache-2.0; complete installed distribution license retained at `docs/third_party/TENACITY_LICENSE.txt` (SHA-256 `58d1e17ffe5109a7ae296caafcadfdbe6a7d176f0bc4ab01e12a689b0499d8bd`). The runtime uses the public bounded retry API; no upstream code is copied or renamed. Local receipts, transaction-ID validation, fallback selection, quarantine state and evidence semantics are project integration code. PyBreaker and BentoML were design references only and are not installed dependencies.
