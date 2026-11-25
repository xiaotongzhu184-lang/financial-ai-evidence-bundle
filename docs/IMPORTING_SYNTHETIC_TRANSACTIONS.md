# Importing a rights-declared synthetic transaction file

The default `contract-run` uses the packaged project-generated example. To test the **same evidence contract** with a separate synthetic CSV, prepare the source file and metadata locally. The importer does not download a dataset, grant reuse rights, or verify an operator's license declaration. Keep private input files and the resulting run outside a public repository.

## Accepted CSV

UTF-8 CSV with these exact required columns (other columns are ignored):

```text
transaction_id,timestamp_utc,customer_id,merchant_id,amount,label
```

Each row is one customer-to-merchant transaction. `transaction_id` must be unique; `timestamp_utc` must be timezone-aware ISO 8601 and rows must be in chronological order; `amount` must be a finite positive number in one consistent unit; `label` must be exactly `0` or `1`. The importer derives `hour` and a contiguous sequence number. It accepts 80–200,000 rows, at most 150 MB, and requires both classes in the first 75% and final 25%. It rejects a timestamp tie crossing the train/test cutoff. No currency conversion, entity resolution, label verification or resampling is performed.

Create a JSON metadata file with actual facts for the exact CSV you hold:

```json
{
  "source_name": "Exact dataset and file name",
  "source_url": "https://publisher.example/exact-file-or-landing-page",
  "dataset_version": "Exact release or retrieval date",
  "license_id": "Exact data license, not the code license",
  "label_definition": "What label 1 actually means",
  "csv_sha256": "64 lowercase hex characters from the exact CSV bytes",
  "classification": "synthetic_public",
  "redistribution_permission": "permitted"
}
```

Use `synthetic_private` and `private_local_only` when distribution is not cleared. Calculate the hash using `shasum -a 256 path/to/input.csv` on macOS or `sha256sum path/to/input.csv` on Linux. The software checks that exact hash; it cannot determine whether the URL, label definition, license or permission is truthful. Review the publisher's data terms before sending a bundle to anyone.

```bash
.venv/bin/yk-evidence contract-run \
  --transactions-csv /absolute/path/input.csv \
  --transactions-metadata /absolute/path/input-metadata.json \
  --output runs --run-id imported-001 --seed 42
.venv/bin/yk-evidence verify runs/imported-001
.venv/bin/yk-evidence serve runs/imported-001 --port 8765
```

The bundle retains the exact CSV, metadata, normalized rows, a compact data-quality profile, raw model scores and an explicitly blocked release gate. It does not invent a five-seed stability estimate for a fixed imported file. The fixed SEC financial example remains a separate input; the two sources do not establish a causal link between a revenue fact and any transaction label. `private_local_only` bundles are local review material and must not be uploaded to GitHub or Zenodo.
