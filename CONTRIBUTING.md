# Contributing

Small, reviewable changes are welcome. Open an issue before changing evidence schemas, data-source rules or model-selection policy so that compatibility and claim boundaries can be discussed first.

## Development setup

Use Python 3.12 in a virtual environment:

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python scripts/run_tests.py
python scripts/project_acceptance.py --output acceptance-run
```

Add a regression test for a repaired defect. Preserve deterministic seeds and do not replace a failing check with a fixed result. Generated runs, raw third-party datasets, model files, credentials and private review records must not be committed.

## Documentation and provenance

Document new dependencies in `PROVENANCE.md` and `THIRD_PARTY_NOTICES.md`, including the exact version, official source, license and whether code was copied, adapted or only called through a package API. Experimental metrics must identify the data, split, seed and evidence file that produced them. A local or CI pass does not establish production performance, regulatory approval or data rights.

By submitting a contribution, you agree that your contribution is licensed under the Apache License 2.0 and that you have the right to provide it under that license.
