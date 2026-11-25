# Upstream adoption and modification boundaries

This note records the upstream code and package interfaces that are actually used by the software. It separates adopted components from projects that were examined but not incorporated. The package-level inventory and license notices are in [`PROVENANCE.md`](../../PROVENANCE.md) and [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

## Adopted components

### Evidently PSI calculation

The project contains a small derivative of Evidently's Population Stability Index calculation in `src/yk_evidence/_vendor/evidently_psi.py`.

- Upstream repository: [evidentlyai/evidently](https://github.com/evidentlyai/evidently)
- Pinned commit: `34771fe41c08ca074d96a52aaf4a45cd640cbaf5`
- Upstream source: `src/evidently/legacy/calculations/stattests/psi.py`
- Upstream source SHA-256: `11c5131df994f19e0131c56e883d9fc966087745aa143c91cca71a5a87f83426`
- License: Apache-2.0
- Local license copy: `src/yk_evidence/_vendor/EVIDENTLY_LICENSE.txt`

The derivative keeps the PSI expression and adds strict input validation and per-bin output. Project-owned logic in `src/yk_evidence/monitoring.py` fits reference-only bins, applies additive smoothing, groups later observations into time-ordered windows, and records the counts needed for semantic verification. This is not an integration of the complete Evidently platform. PSI is reported as a distribution-shift diagnostic and is not treated as proof that model accuracy or financial loss changed.

### Tenacity retry policy

`src/yk_evidence/resilience.py` uses the public Tenacity 9.2.1 API for bounded retries around explicitly classified temporary failures.

- Upstream repository: [jd/tenacity](https://github.com/jd/tenacity)
- Pinned research commit: `4410996dd9a96864a2733ce0778bcaa112b72811`
- Integration form: installed dependency; no Tenacity source is copied
- License: Apache-2.0
- Local license copy: `docs/third_party/TENACITY_LICENSE.txt`

Retries are limited to two attempts with no wait. Other exceptions remain visible to the caller. Evidence receipts, row identifiers, model-subset decisions, failure isolation, and evidence recomputation are project-specific logic around the upstream API.

### XGBoost model API

`src/yk_evidence/algorithm.py` uses the XGBoost 3.4.1 package API as one of the benchmark models. The Linux lock uses `xgboost-cpu`; other supported systems use `xgboost`. No XGBoost source code is copied into this repository.

## Examined but not incorporated

The [Fraud Detection Handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook) was reviewed at commit `81cf7d1714bb7b2f5b496407d9055d91dc68dc25` for time-aware evaluation and review-budget ideas. Its card-level functions depend on customer or card identifiers and delayed feedback semantics that the ULB transaction dataset does not provide. The project therefore uses transaction-row evaluation and does not copy the handbook's GPL-3.0 code.

PyBreaker and BentoML were also examined as architecture references. Neither package is installed or represented as an adopted subsystem.

## Verification boundaries

The local test suite checks the modified PSI arithmetic, frozen-bin behavior, window counts, model integration, evidence regeneration, and tamper rejection. These tests validate this project's integration. They do not claim to run or certify any upstream project's full test suite.

Any future upstream adoption must record the exact version or commit, license, adopted files or API, local modifications, and an integration test before the component is described as part of the software.
