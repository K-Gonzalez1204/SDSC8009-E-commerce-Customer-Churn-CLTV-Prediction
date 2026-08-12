import csv
import hashlib
import json
from pathlib import Path

import polars as pl


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_dataframe(path) -> pl.DataFrame:
    return pl.read_csv(path)


def validate_input(path, expected: str) -> str:
    actual = sha256_file(path)
    if actual.upper() != expected.upper():
        raise ValueError(f"input hash mismatch: expected {expected}, got {actual}")
    return actual


def write_json(path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_records_csv(path, records) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
