#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from nyse_calendar_utils import generate_nyse_open_days_csv


def main():
    parser = argparse.ArgumentParser(description="Generate NYSE open-day calendar CSV.")
    parser.add_argument("--start_year", type=int, default=1990)
    parser.add_argument("--end_year", type=int, default=datetime.now().year + 1)
    parser.add_argument("--out", type=str, default=str(Path("data") / "nyse_open_days.csv"))
    args = parser.parse_args()

    out = generate_nyse_open_days_csv(
        csv_path=Path(args.out),
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(f"[OK] NYSE open days written: {out}")


if __name__ == "__main__":
    main()

