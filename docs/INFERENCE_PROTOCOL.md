# Frozen-model batch scoring (0.6)

This workflow applies the four fitted models and the **validation-frozen** fusion weights, alert threshold and review budget from an existing 0.6 algorithm bundle to a separate, unlabeled CSV. It does not fit a model, tune a threshold, calculate accuracy, or make an operational approval decision. The output is a locally sealed research evidence bundle.

## Input contract and scope

The CSV header must be exactly:

```text
row_id,Time,V1,V2,...,V28,Amount
```

The actual file must contain each of `V1` through `V28` in numerical order. `row_id` is a unique nonnegative integer that remains stable across reruns. `Time` is a finite nonnegative **relative** value. Every feature must be finite and numeric. `Class`, another outcome label, missing columns, reordered columns and extra columns are rejected. The raw CSV remains outside the output bundle.

The source bundle's anonymized `V1`–`V28` columns are already transformed coordinates. Matching column names or plausible values cannot establish that another institution's data uses the same transform. The operator must affirm the **same feature coordinate system** before scoring. Do not put ordinary raw transaction fields under the `V1`–`V28` names, and do not interpret this path as a general bank-data interface. The synthetic example below checks software behavior only; it is not fraud-effectiveness evidence.

## Local smoke example

Install the project with Python 3.12 and the pinned dependency lock as in the main README. Then generate a source bundle using the software's synthetic fixture:

```bash
.venv/bin/yk-evidence algorithm-smoke --output runs --run-id trusted-smoke --load-rows 256
.venv/bin/yk-evidence verify runs/trusted-smoke
```

Create an **unlabeled** batch using the same generated feature formula. This is local test data; it contains no real transactions:

```bash
python3.12 - <<'PY'
import csv
from pathlib import Path

path = Path('unlabeled-smoke.csv')
with path.open('w', encoding='utf-8', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['row_id', 'Time', *(f'V{i}' for i in range(1, 29)), 'Amount'])
    for index in range(200):
        synthetic_pattern = int(index % 8 == 0)
        features = [round((index % (7 + j % 5)) * .1 + synthetic_pattern * (j % 3), 3)
                    for j in range(1, 29)]
        writer.writerow([1000 + index, index // 2, *features, index % 100])
print(path)
PY
```

Score only when you trust the producer of the model bundle and know that the incoming features share its coordinate system:

```bash
.venv/bin/yk-evidence score-batch \
  --model-bundle runs/trusted-smoke \
  --csv unlabeled-smoke.csv \
  --output runs \
  --run-id scored-smoke \
  --trust-models \
  --same-feature-space
.venv/bin/yk-evidence verify runs/scored-smoke
```

The frozen review capacity is `floor(budget_fraction × batch row count)`; it is computed for the **entire submitted batch**, not each row or a continuous live queue. A small batch can have zero review capacity. Use a fresh run ID for each batch; completed runs are not overwritten.

## Evidence and verification

`inference_metadata.json` binds the source manifest, four model files, frozen policy and input file by SHA-256. It records source/runtime versions, feature-column contract, row count, stable row-ID digest and the operator's coordinate attestation. `inference_scores.json` keeps, per row and model, the raw probability, calibration input and calibrated probability, followed by the fused score and three queue flags. `inference_queue.json` records the threshold-alerted, reviewed and deferred row IDs and hard-cap totals. `bundle_manifest.json` hashes the three JSON artifacts. The raw input CSV and source model binaries are not copied into the scored bundle.

The normal verifier reads JSON and **never loads a pickle**. It checks file hashes, row identities, the recorded calibration formula, fused-score arithmetic and the deterministic threshold/budget queue. When given the source bundle, the library verifier also checks that the recorded policy and model hashes point to that exact source bundle. This does not independently authenticate a malicious producer who can change records and rehash them. It also does not rerun saved models; model replay requires the explicitly trusted load path.

## Model loading boundary

The source must pass its existing 0.6 safe verifier before model loading. The loader then requires explicit `--trust-models`, explicit `--same-feature-space`, and exact installed scikit-learn, NumPy and XGBoost versions from the saved model record. Loaded calibrators must report the expected 29-feature shape. Errors leave no completed output directory.

The trust flags are substantive: [`joblib.load` can execute arbitrary Python code](https://joblib.readthedocs.io/en/stable/generated/joblib.load.html), and [scikit-learn does not support loading fitted models across versions](https://scikit-learn.org/stable/model_persistence.html). A hash in a self-contained bundle does not make an unknown producer trustworthy. Use only a source bundle whose producer and transfer path you already trust.

The output is research evidence from an anonymized historical benchmark or synthetic software fixture. It does not establish performance on new institutions, production suitability, or permission to publish derived data. Check source and output rights separately before distribution.
