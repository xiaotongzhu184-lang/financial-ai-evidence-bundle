# Independent transaction benchmark: data access and interpretation

The new tabular evaluation accepts the ULB/Worldline Credit Card Fraud Detection format. Its public release describes 284,807 anonymized European card transactions from two days in September 2013, including 492 fraud labels. V1–V28 are principal components; Amount and elapsed Time are also supplied. Customer, merchant and device identifiers are not supplied. No graph identity is reconstructed from row order or PCA values.

Primary publisher: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Official tutorial linking the same CSV: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

Download mirror: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

Evaluated SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`

Size: 150,828,752 bytes. The row count and 492 positive labels were independently checked locally on 11 October 2026. Zero Amount is valid: 1,825 records have zero Amount. Time remains relative seconds; the software does not invent calendar timestamps.

## License and distribution

Kaggle's publisher metadata labels the database Open Database License and the contents Database Contents License. Read both: https://opendatacommons.org/licenses/odbl/1-0/ and https://opendatacommons.org/licenses/dbcl/1-0/. These data terms are separate from the software's ownership/license decision and the upstream Python libraries' licenses. Retain source attribution when sharing results. A public or adapted database distribution requires its own review of the applicable database conditions.

The raw CSV is not bundled in GitHub source, wheels or the prepared Zenodo software archive. The fetch script downloads it locally and checks the evaluated fingerprint. Derived per-row prediction evidence preserves the source attribution and remains in the private review package pending archive rights review. A local manifest authenticates neither the publisher nor the original fraud adjudication.

## Reproduction

Run `python scripts/fetch_ulb.py --output external-data/ulb`. Read the displayed source and license information before redistribution. If the mirror bytes change, the command fails instead of silently accepting a new dataset. For a trusted local copy supply the CSV and metadata directly to `research-run`.

Chronological splitting, score calibration and model selection are specified in `TABULAR_EVALUATION_PROTOCOL.md`. The benchmark is an external historical sample, not a prospective institution trial. Temporal splitting here cannot undo transformations already performed by the original publisher; the PCA fitting scope is not independently known. Accuracy across this short historical period does not establish modern bank performance, realized cost savings or intervention effects.
