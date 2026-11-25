#!/usr/bin/env python3
"""Download the publisher-attributed TensorFlow tutorial mirror, then hash it.

Raw data is deliberately excluded from the software archive. See
docs/project/ULB_DATA_ACCESS.md for the independent database/content licenses.
"""
import argparse
import hashlib
import json
import os
import tempfile
import ssl
import certifi
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"


def checked_download_url() -> str:
    """Return the fixed publisher mirror only after validating its origin."""
    parsed = urlsplit(URL)
    if (parsed.scheme != "https" or parsed.hostname != "storage.googleapis.com"
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or parsed.query or parsed.fragment
            or parsed.path != "/download.tensorflow.org/data/creditcard.csv"):
        raise ValueError("unexpected dataset mirror URL")
    return URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "creditcard.csv"
    if target.exists():
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != SHA256:
            raise SystemExit("Existing CSV has a different hash; choose a new output directory.")
    else:
        fd, tmp = tempfile.mkstemp(prefix=".ulb-", dir=args.output)
        try:
            digest = hashlib.sha256()
            size = 0
            request = urllib.request.Request(checked_download_url(), headers={"User-Agent": "yk-evidence/0.7.2rc1"})
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(  # nosec B310: fixed HTTPS origin validated above
                    request, timeout=60,
                    context=ssl.create_default_context(cafile=certifi.where())) as response:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 200_000_000:
                        raise ValueError("Download exceeds the expected dataset size budget")
                    digest.update(chunk)
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            if digest.hexdigest() != SHA256:
                raise ValueError("Downloaded data differs from the evaluated source SHA-256")
            # Exclusive link prevents overwriting a source created during download.
            os.link(tmp, target)
        finally:
            Path(tmp).unlink(missing_ok=True)
    metadata = {
        "dataset_name": "ULB / Worldline Credit Card Fraud Detection, 2013",
        "classification": "public_ulb_creditcard",
        "source_url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "download_url": URL, "csv_sha256": SHA256,
        "license_status": "Database ODbL 1.0; contents DbCL 1.0, as declared by publisher; raw data not included in software archive",
        "license_urls": ["https://opendatacommons.org/licenses/odbl/1-0/", "https://opendatacommons.org/licenses/dbcl/1-0/"],
        "label_definition": "Publisher Class=1 fraud and Class=0 nonfraud; no independent adjudication",
        "dataset_version": "Kaggle version 3 / TensorFlow tutorial mirror, evaluated SHA-256",
    }
    destination = args.output / "metadata.json"
    if not destination.exists():
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
    print(f"Verified {target}: {SHA256}")


if __name__ == "__main__":
    main()
