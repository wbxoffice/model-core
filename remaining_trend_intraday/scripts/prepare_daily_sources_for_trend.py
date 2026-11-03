#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
from pandas.tseries.offsets import BDay

from market_time_utils import current_nyse_trade_date


THIS_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = THIS_DIR.parent
PROJECT_ROOT = ARCHIVE_ROOT.parent

DAILY_FILES = [
    "SP_SPX, 1D.csv",
    "DIX.xlsx",
    "CBOE_DLY_VIX.csv",
    "event_calendar.csv",
]


def _find_spx_vdtr_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "spx_vdtr_v2",
        PROJECT_ROOT / "SPX VDTR model v2.0",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("spx_vdtr_v2 project directory not found.")


def _daily_features_have_target(target_date: pd.Timestamp) -> bool:
    spx_vdtr_dir = _find_spx_vdtr_dir()
    for p in [PROJECT_ROOT, spx_vdtr_dir]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from feature_engineering_v2_1 import build_feature_frame_v21
    from nyse_calendar_utils import (
        ensure_nyse_open_days_csv,
        load_nyse_open_days,
        align_series_to_next_open_day,
    )

    daily = build_feature_frame_v21(for_training=False).copy()
    if "date" not in daily.columns or "target_date" not in daily.columns:
        print("[INFO] daily feature check: missing date/target_date columns.")
        return False

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["target_date"] = pd.to_datetime(daily["target_date"], errors="coerce")
    mask = daily["target_date"].isna() & daily["date"].notna()
    if mask.any():
        daily.loc[mask, "target_date"] = daily.loc[mask, "date"] + BDay(1)

    nyse_cal_file = PROJECT_ROOT / "data" / "nyse_open_days.csv"
    ensure_nyse_open_days_csv(nyse_cal_file)
    nyse_open_days = load_nyse_open_days(nyse_cal_file)
    daily["target_date"] = align_series_to_next_open_day(daily["target_date"], nyse_open_days)

    target = pd.Timestamp(target_date).normalize()
    available = pd.to_datetime(daily["target_date"], errors="coerce").dt.normalize()
    max_target = available.max()
    ok = bool((available == target).any())
    print(
        "[INFO] daily feature check: "
        f"target={target.date().isoformat()} available={ok} max_target={max_target.date().isoformat() if pd.notna(max_target) else 'NA'}"
    )
    return ok


def _run_parent_update() -> None:
    updater = PROJECT_ROOT / "update_parent_data_from_latest_v2.py"
    if not updater.exists():
        raise FileNotFoundError(f"parent update script not found: {updater}")
    print(f"[INFO] Update parent daily data: {updater}")
    cp = subprocess.run([sys.executable, str(updater)], cwd=str(PROJECT_ROOT), check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"parent daily data update failed: rc={cp.returncode}")


def _find_spx_vdtr_data_dir() -> Path:
    p = _find_spx_vdtr_dir() / "data"
    if not p.exists():
        raise FileNotFoundError(f"spx_vdtr_v2 data directory not found: {p}")
    return p


def _sync_daily_files() -> None:
    src_dir = PROJECT_ROOT / "data"
    dst_dir = _find_spx_vdtr_data_dir()
    print(f"[INFO] Sync daily feature sources: {src_dir} -> {dst_dir}")
    for name in DAILY_FILES:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            print(f"[SKIP] missing source: {src}")
            continue
        if name != "event_calendar.csv" and not dst.exists():
            print(f"[SKIP] missing destination: {dst}")
            continue
        shutil.copy2(src, dst)
        print(f"[SYNC] {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-date", default=None, help="Target date to check, YYYY-MM-DD. Default: today.")
    ap.add_argument("--force-update", action="store_true", help="Always run parent update and sync.")
    ap.add_argument("--check-only", action="store_true", help="Only check daily feature availability; do not update.")
    args = ap.parse_args()

    target_date = (
        pd.Timestamp(args.target_date).normalize()
        if args.target_date
        else current_nyse_trade_date()
    )
    try:
        if not args.force_update and _daily_features_have_target(target_date):
            print("[INFO] daily feature already available. Skip parent update/sync.")
            return 0
        if args.check_only:
            print("[INFO] daily feature missing. check-only mode, no update performed.")
            return 2

        print("[INFO] daily feature missing or force-update enabled. Run parent update/sync.")
        _run_parent_update()
        _sync_daily_files()
        if not _daily_features_have_target(target_date):
            print("[WARN] daily feature still missing after update/sync.")
        return 0
    except Exception as exc:
        print(f"[ERROR] prepare daily sources failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
