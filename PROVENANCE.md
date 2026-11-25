# Project Provenance

The first executable slice contains original project code and original synthetic fixture data. The integrated development branch additionally carries a two-fact extract from SEC Company Facts for Microsoft 2025 Form 10-K, accession `0000950170-25-100235`. The extract is in `src/yk_evidence/assets/data/financial/msft_2025_10k_revenue.json`; it records its source URL, retrieval time, raw response SHA-256, concept, unit and both periods. It does not include the 4.8 MB raw response. The [SEC reuse FAQ](https://www.sec.gov/files/about/webmaster-faq.htm) says EDGAR public filing content is free to access and reuse. The original project code parses and cites the extract; no SEC code is imported.

Candidate research and adopted-component decisions are summarized in `docs/project/UPSTREAM_REUSE_MAP.md`. A candidate is not an adopted dependency. An adoption record requires a pinned version, exact license, source URL, notice obligations and local modification list.

## Adopted package: scikit-learn

- Purpose: tabular and graph-history logistic baselines, graph-history random forest, and held-out evaluation metrics in `src/yk_evidence/benchmark.py`.
- Official source: [scikit-learn GitHub](https://github.com/scikit-learn/scikit-learn), [release 1.9.1](https://github.com/scikit-learn/scikit-learn/releases/tag/1.9.1), source commit `866c0f51e7560ef0303cbcc5f159df5382ea9e3f`.
- Code license: [BSD-3-Clause at the adopted tag](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/COPYING). Package is installed from a wheel via `requirements.lock`; no upstream code is copied or modified.
- Local integration: original synthetic generator, chronological split, prior-history features, raw prediction records and evidence schema. Runtime lock lists scikit-learn and its installed transitive packages. No upstream source code is vendored. The lock pins versions, not wheel hashes; wheel hashes, transitive notices and distribution terms require review before any public archive or dependency redistribution.

## Adopted package: jsonschema

- Purpose: validate ten JSON artifacts against the project's Draft 2020-12 evidence schema in `src/yk_evidence/contract.py` before semantic checks.
- Official source: [python-jsonschema](https://github.com/python-jsonschema/jsonschema), version `4.26.0` in `requirements.lock`; [MIT license](https://github.com/python-jsonschema/jsonschema/blob/v4.26.0/COPYING).
- Local integration: original schema in `src/yk_evidence/assets/schemas/contract_0_3.json`; no upstream source code copied or modified. Schema validation alone does not establish authenticity.

## Optional adopted packages: PyTorch and PyTorch Geometric

- Purpose: run the original two-layer GraphSAGE edge classifier in `src/yk_evidence/graph_benchmark.py`, included in the same evidence bundle only when `--graph-epochs` is requested.
- Official sources: [PyTorch 2.9.1](https://github.com/pytorch/pytorch/releases/tag/v2.9.1) and [PyTorch Geometric 2.8.0](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.8.0). Their respective [PyTorch license](https://github.com/pytorch/pytorch/blob/v2.9.1/LICENSE) and [PyG MIT license](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/LICENSE) apply; no upstream code is copied or modified.
- Local integration and verification on 2026-10-09: isolated Python 3.12 environment with `torch==2.9.1` and `torch-geometric==2.8.0`; the graph tests, including a future-edge perturbation test, passed. A 20-epoch run on the generated fixture emitted raw edge scores and was linked into a 0.3 evidence bundle. The synthetic test split has only six positive transactions, and the model's training-only graph context differs from the history-feature baseline. The optional result does not establish superiority or real-world fraud performance.
- Distribution limit: the core package does not bundle these large wheels. Before redistributing them, record exact wheel hashes and review their bundled third-party notices. The current version-only install instructions do not pin wheel bytes.

## 0.4.0rc1 adopted components

scikit-learn 1.9.1 supplies the actual LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier, FrozenEstimator/CalibratedClassifierCV and TfidfVectorizer implementations. DuckDB 2.0.1 performs the typed CSV quality scan under a declared memory budget. Certifi 2026.11.20 supplies Mozilla CA roots for the authenticated HTTPS download path. These are package dependencies, not copied repositories or original algorithm claims. Their licenses and retained notices are in THIRD_PARTY_NOTICES.md and docs/third_party/.

The new original project work is the evaluation partition/operating-policy protocol, evidence contract and semantic checks, failure handling, bounded load measurement integration, conservative financial-source retrieval and review/export workflow. This does not establish patent novelty. At the 0.4.0rc1 milestone XGBoost and LightGBM were studied but not adopted; the subsequent 0.5.0rc1 section records XGBoost adoption.

The public historical ULB CSV is retrieved separately from the TensorFlow tutorial mirror, matched by SHA-256 and attributed to its ULB/Worldline publisher page. It is excluded from the software repository. SEC Company Facts derived revenue text retains the selected original structured fact, accession, issuer URL and fiscal dates; it is not represented as a whole filing or verbatim narrative.

## 0.5.0rc1 algorithm extension

XGBoost 3.4.1 supplies an actual CPU histogram boosted-tree challenger; the official Linux CPU distribution avoids GPU runtime dependencies. The fixed challenger configuration, chronological fusion-selection protocol, 35-candidate simplex search, queue-capacity arbitration and evidence reconstruction are recorded in `docs/project/ALGORITHM_FUSION_SPEC.md`. Existing sklearn components provide the three baselines and sigmoid calibration. These combinations are project engineering choices, not a claim to have invented boosting, convex mixtures, robust selection or top-k ranking. All ULB final-period results in this extension are explicitly retrospective exploratory comparisons because that period was inspected in v0.4.

## Concrete reuse distinction for the 0.5 extension

| Upstream | Reuse type | Local implementation / role | Adopted boundary |
|---|---|---|---|
| XGBoost 3.4.1 | Installed package API | `src/yk_evidence/algorithm.py` creates the real `XGBClassifier`; package variants are pinned in `requirements.lock` | Library implementation is not copied or relabeled as original |
| Evidently commit `34771fe41c08ca074d96a52aaf4a45cd640cbaf5` | Source-derived numerical core | `src/yk_evidence/_vendor/evidently_psi.py::psi_components`; Apache-2.0 license alongside it | Numerical PSI expression adapted; full monitoring product/framework not bundled |
| Fraud Detection Handbook commit `81cf7d1714bb7b2f5b496407d9055d91dc68dc25` | Methodological reference | Time-aware and capacity-limited assessment examined; existing project transaction-level policy retained | No code copied; card-level semantics cannot be inferred from ULB transaction rows |

The Evidently original is `src/evidently/legacy/calculations/stattests/psi.py::_psi`, SHA-256 `11c5131df994f19e0131c56e883d9fc966087745aa143c91cca71a5a87f83426`. The derived file retains Copyright 2021-2023 Evidently AI, Inc., Apache-2.0 and a modification notice. Original inputs, source hashes and licenses were retrieved and checked on 2026-10-11; the source research is in `docs/project/CONCRETE_UPSTREAM_REUSE.md`.

Local changes extract the calculation from the pandas/stat-test registry, validate finite positive aligned vectors, and return per-bin contributions. The original helper's pooled reference/current Sturges binning and adaptive zero replacement are not used. Project integration freezes bins on the policy reference, applies declared additive smoothing and supplies chronological windows and queue diagnostics. Consequently output is not claimed to match Evidently's full original helper where binning or smoothing differs. The full license is in `src/yk_evidence/_vendor/EVIDENTLY_LICENSE.txt`; archive/wheel inclusion and executable integration require the corresponding final build checks.

Verification record (2026-10-11): monitoring tests 6/6 and algorithm integration tests 12/12 passed; the main task then ran all 130 project tests in 27.979 seconds with no skips. The retained Evidently license was byte-compared with the pinned upstream file and matched SHA-256 `379054a067cbe8e134e084287550e9391d2804636687c4723f80a2c58a463f0e`. These are project integration checks, not a claim that the complete upstream Evidently test suite was run. Final distributable verification is recorded by the release build.

## 0.6 prediction delivery resilience

Tenacity 9.2.1 is a direct runtime dependency for at most two prediction attempts on explicit delivery-integrity/model-unavailable exceptions. Tag commit: `4410996dd9a96864a2733ce0778bcaa112b72811`, source https://github.com/jd/tenacity . Its Apache-2.0 notice is retained in docs/third_party/TENACITY_LICENSE.txt. The implementation adds immutable producer receipts, row-ID joins, per-run quarantine, validation-frozen healthy-subset fallback and a reconstructable event transcript. PyBreaker and BentoML informed design choices only; no complete serving platform is included. See `docs/project/RESILIENCE_SPEC.md`. This correction addresses post-production score-delivery faults; it does not make scores intrinsically true or authenticate a remote producer.

## 0.7 integration and batch reuse (2026-10-14)

The project envelope and same-coordinate batch scorer reuse the pinned scikit-learn, joblib and XGBoost public APIs already listed above. No new upstream implementation was copied and no new package was added. The original work is the envelope contract, preserved child provenance, trusted-loader input/policy checks, JSON scoring receipts and verifier, and the repaired local viewing/installed entry path. Upstream training, calibration, vectorization and serialization remain upstream capabilities.

MLflow was evaluated as an experiment-tracking alternative but was not installed or adopted. Saved-model loading follows the [scikit-learn persistence constraints](https://scikit-learn.org/stable/model_persistence.html) and [joblib trust boundary](https://joblib.readthedocs.io/en/stable/generated/joblib.load.html). The CLI requires explicit trust and matching-feature-space flags before deserializing a model.

## Project license and citation

Original project code and documentation are licensed under Apache License 2.0. `LICENSE`, `NOTICE` and `CITATION.cff` identify the license, copyright holder, release version and citation metadata. The project license does not override the separate data and third-party terms listed above.
