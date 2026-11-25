# Development Model Card

> Historical scope note (2026-10-11): this document records the earlier 0.1–0.3 synthetic path. For current 0.7 onboarding, use the repository README and `PROJECT_BUNDLE_V07.md`. The 0.4 ULB path is documented in `TABULAR_EVALUATION_PROTOCOL.md` and `RETRIEVAL_PROTOCOL_V04.md`.

## Intended use

Compare four baselines: a fixed amount rule, a scikit-learn balanced logistic regression on amount/hour, a logistic regression enriched with prior customer/merchant transaction-history features, and a random forest on those same history features. The latter two use graph-aware feature engineering, **not** a GCN or GraphSAGE model. Outputs are demonstration risk scores on generated transactions, not calibrated fraud probabilities.

## Evaluation

The benchmark saves per-transaction held-out IDs, labels, scores and decisions. Metrics are computed from those raw predictions and defined in `BENCHMARK_PROTOCOL.md`. The model card does not hard-code a performance number; each run's evidence bundle is the source of measured values. The reported recall at the training-selected 5% FPR target does **not** imply a 5% FPR on the held-out set. Review the separately reported held-out FPR and false-positive count alongside recall and average precision.

The default generated run saves five-seed stability metrics and confusion counts. An imported file gets confusion counts but no invented repeated-seed estimate. A separately frozen fitted graph-history logistic model is checked against the baseline scores, then applied to two changed holdout datasets. The parameter digest and each scenario's raw scores allow the reviewer to see that the stress pass did not refit the model. The synthetic link-suppression scenario may fail its illustrative threshold; such a failure is retained as evidence. The fixed amount rule uses 800 input units and is not calibrated to another source.

## Limitations and safeguards

The default generator assigns the label, the class prevalence and ring topology. No real transaction, protected attribute or externally verified distribution is represented. Do not use this system to deny a transaction, investigate a person, claim regulatory compliance, or cite a synthetic metric as field performance. An optional GraphSAGE experiment can be included in the same bundle; it shares test IDs but uses a stricter training-only graph context than the history-feature baseline. Controlled architecture ranking and independently licensed external-data validation require a separate evidence gate.
