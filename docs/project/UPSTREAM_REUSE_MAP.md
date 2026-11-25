# Upstream Research and Reuse Map

> Historical scope note (2026-10-11): this map records candidate decisions made for the earlier synthetic path. Current adopted components and exact license notices are maintained in the root `PROVENANCE.md` and `THIRD_PARTY_NOTICES.md`.

Research snapshot: 2026-10-05. Scikit-learn 1.9.1 is adopted as a package dependency on the integrated development branch; SEC Company Facts is adopted as a two-fact public filing snapshot. Other entries remain candidates. The observed `HEAD` hashes below are discovery references, not release pins. Recheck exact tags, license files and dependency terms at adoption.

| Layer | Candidate and official source | Observed HEAD | License evidence | Fit and decision |
| --- | --- | --- | --- | --- |
| 1 | [FAISS](https://github.com/facebookresearch/faiss) | `b0074a3fa426027d9ff8575b44386d5922859ac9` | [MIT](https://github.com/facebookresearch/faiss/blob/main/LICENSE) | Retrieval candidate; native dependency unnecessary for first slice. |
| 2 | [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) | `79d33965a40b7fa83616a9f598a0f8619f25d939` | [MIT](https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE) | GCN candidate; verify PyTorch/extension environment. |
| 2 | [scikit-learn](https://github.com/scikit-learn/scikit-learn) | Adopted release `1.9.1`, commit `866c0f51e7560ef0303cbcc5f159df5382ea9e3f` | [BSD-3-Clause at tag](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/COPYING) | Installed package API for risk baselines; no source code copied. |
| 3 | [MLflow](https://github.com/mlflow/mlflow) | `c6a8c9a5adc71519a60cf1d71b4dfaae5b2372ed` | [Apache-2.0](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt) | Run/model registry candidate; service state must be reproducible. |
| 3 | [Dagster](https://github.com/dagster-io/dagster) | `54577dad9a95610029c2a3a2e5c1a02cb540d9da` | [Apache-2.0](https://github.com/dagster-io/dagster/blob/master/LICENSE) | Orchestration candidate; heavy for first slice. |
| 2 | [IBM AMLSim](https://github.com/IBM/AMLSim) | `7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca` | [Apache-2.0 code](https://github.com/IBM/AMLSim/blob/master/LICENSE) | Synthetic generator candidate; AML domain fit and output terms need review. |
| 2 data | [IBM AML-Data](https://github.com/IBM/AML-Data/blob/main/README.md) | `366436e605ac7713bf4eee8b4fbc45984d64427c` | Code Apache-2.0; **data [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/)** | Candidate only; preserve data attribution and sharing obligations if adopted. |
| 1 data | [SEC EDGAR developer resources](https://www.sec.gov/about/developer-resources) | Adopted accession `0000950170-25-100235` | [SEC reuse FAQ](https://www.sec.gov/files/about/webmaster-faq.htm) | Two-fact Company Facts snapshot, with URL, retrieval timestamp and raw SHA-256 in `data/financial`; no unrelated art/exhibits. |

No upstream source code is copied into this repository. The original synthetic fixture remains available as a smoke test. Actual adoption details are recorded in `PROVENANCE.md`.
