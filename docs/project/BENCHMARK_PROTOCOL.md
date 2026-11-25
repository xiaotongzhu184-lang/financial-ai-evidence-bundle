# Synthetic Transaction Benchmark Protocol

> Historical scope note (2026-10-11): this protocol records the earlier 0.1–0.3 synthetic path. The current envelope is documented in `PROJECT_BUNDLE_V07.md`; the separate ULB benchmark is documented in `TABULAR_EVALUATION_PROTOCOL.md`.

Status: development protocol for a generated demonstration dataset. No metric from it estimates performance on real transactions.

The generator uses seed 42 by default, emits 400 or more ordered transactions, and injects a recurring customer/merchant ring every nineteenth transaction. Its label is known by construction. The chronological first 75% trains estimators; the last 25% is held out. Amount and hour are the tabular features. The graph-aware baseline adds counts of prior transactions sharing a customer or merchant and the prior merchant mean amount. Features use no labels or future rows.

The benchmark compares four baselines: a fixed amount rule, a balanced logistic regression on tabular features, the same estimator on tabular plus graph-history features, and a random forest on tabular plus graph-history features. Report average precision (PR-AUC), F1 at score 0.5, and recall at a threshold selected from training negatives to target FPR at most 0.05. This is a **training-set target**, not a guarantee on the held-out set. Report the actual held-out FPR and false-positive count beside recall; store every held-out ID, label and score. The gate treats a single-class holdout as undefined.

Limitations: this is a deliberately simplified synthetic pattern with labels assigned by the generator. A graph-aware feature model is not a GCN or GraphSAGE. Results cannot be used as claims about real fraud detection. A graph-neural benchmark and an independently sourced dataset require separate validation.

The integrated run additionally repeats generation and fitting over five neighboring seeds; it stores each run's metrics and the mean/population standard deviation, not a population confidence interval. The frozen graph-history logistic estimator is then rescored under two declared holdout perturbations: multiplying holdout amounts by 1.2, and replacing ring-merchant links with isolated merchant IDs. This intentionally measures dependence on observable links. The same fitted model, training split and training-selected threshold are used across scenarios. A drop in average precision over 0.10 is a demonstration stress failure, not a production policy. Raw scores and status are preserved in `stress_replay.json`.
