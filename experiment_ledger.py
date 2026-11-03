#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path


LEDGER_DIR = "experiments"
LEDGER_CSV = "experiments.csv"


LEDGER_FIELDS = [
    "run_ts",
    "project",
    "status",
    "return_code",
    "duration_sec",
    "runner",
    "train_script",
    "requirements_file",
    "python_version",
    "numpy_version",
    "pandas_version",
    "sklearn_version",
    "notes",
]


def _ensure_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writeheader()


def append_ledger_row(root: Path, row: dict) -> Path:
    ledger_path = root / LEDGER_DIR / LEDGER_CSV
    _ensure_header(ledger_path)
    safe_row = {k: row.get(k, "") for k in LEDGER_FIELDS}
    with open(ledger_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writerow(safe_row)
    return ledger_path

