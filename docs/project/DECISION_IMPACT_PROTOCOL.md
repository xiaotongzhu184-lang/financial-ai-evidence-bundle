# Decision-impact protocol (0.3.0rc3)

> Historical scope note (2026-10-11): this protocol records the earlier 0.3 synthetic path. The current top-level evidence envelope is documented in `PROJECT_BUNDLE_V07.md`.

The intended reviewer is a research or model-risk lead deciding whether **demonstration evidence** merits acceptance, change requests or rejection. A ranked metric alone does not tell that reviewer how many alerts a team would have to examine or which errors an operating policy might trade off. `decision_impact.json` makes those consequences inspectable without turning a synthetic example into a production claim.

## Frozen inputs and calculations

For each of the four baselines, the code reads held-out transaction IDs, binary labels and scores from `benchmark.json`. It uses the model's saved threshold selected from **training negatives** for the 5% training-FPR target. A held-out row is flagged only when `score > threshold`; it never reselects a threshold from held-out labels. The artifact stores TP, FP, TN and FN, held-out positive rate, TPR and FPR, plus 95% Wilson score intervals for TPR and FPR. A single-class holdout has `availability=single_class_holdout`, undefined rates and no scenario outputs.

Three frozen example scenarios assume prevalence 1%, 5% or 10% across 1,000 transactions, a capacity of 100 reviews, one arbitrary cost unit per reviewed alert and 20 per missed positive. For an assumed prevalence `p`, volume `N`, observed held-out rates `TPR` and `FPR`:

- Expected true alerts = `N × p × TPR`; expected false alerts = `N × (1 − p) × FPR`.
- Expected missed positives = `N × p × (1 − TPR)`.
- Expected review load = true alerts + false alerts. It exceeds assumed capacity only if this value is greater than 100.
- Illustrative cost units = expected review load × 1 + expected missed positives × 20.

These are **scenario inputs**, not measured prevalence, staffing cost or deployment thresholds. Fractional expected counts are deliberately retained to show the calculation, not rounded into case records. The local review page shows the 1% case and links to all assumptions and raw evidence. No model is automatically approved by this layer; `release_gate.json` remains blocked for the synthetic example.

## Verification and limitations

`yk-evidence verify` requires this artifact for rc3, validates its structure, hashes it in the manifest and independently rebuilds the entire JSON from `benchmark.json` and the declared frozen scenario table. It rejects a changed figure even if the artifact hash is updated. A changed benchmark prediction or label is also cross-checked against source transaction rows and other metric outputs. The verifier cannot authenticate a wholly replaced, self-consistent source bundle, prove that training chose the threshold correctly, or infer real-world operating rates from a small synthetic holdout. Wilson intervals reflect binomial sampling uncertainty conditional on this holdout and do not account for dataset shift, model-selection bias or uncertain real prevalence.

This layer is original integration around the project's scikit-learn baselines and verified evidence contract. Evidently, Great Expectations and MLflow were assessed separately at that stage; no code from those products was copied into the rc3 decision-impact layer.
