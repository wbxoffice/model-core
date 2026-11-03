#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
One-click controller for multi-project SPX models.

What it does:
1) Sync shared data from ROOT/data/ into each project data/ (only if both sides have the file).
   - Strict rules: SPX files only overwrite SPX, US500 only overwrite US500 (no cross-over).
2) Run each project's "run_all_models..." script (name can vary).
3) Run each project's "daily_meta_strategy_engine_v3.py" if present.
4) Aggregate each project's log/prediction_log.csv (latest 2 rows) into ROOT/prediction_log_all_models.xlsx
   - One sheet per project, overwrite by 'date'.
5) Aggregate latest *_summary.txt and *_strategy.txt from each project log/ into ROOT summary/strategy txt.

Per-project Python runner selection priority:
A) <project>/PYTHON_ENV.txt (highest)
   - Absolute python path, e.g. C:\Users\...\miniconda3\envs\spx_model_env\python.exe
   - Or conda env: conda:spx_model_env  (uses `conda run -n <env> python`)
B) <project>/.venv or <project>/venv python
C) fallback python (from --python or sys.executable)

Usage:
  python sync_run_and_aggregate_projects.py
  python sync_run_and_aggregate_projects.py --dry_run
  python sync_run_and_aggregate_projects.py --python "D:\Python 3.11\python.exe"
  python sync_run_and_aggregate_projects.py --root "E:\...\Model projects"
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
from project_registry_utils import load_registry, registry_map_by_name, validate_registry
from shared.runner import pick_runner as shared_pick_runner


# ----------------------------- Settings -----------------------------

SHARED_DATA_DIRNAME = "data"          # shared master data lives here: ROOT/data/
LATEST_DATA_DIRNAME = "latest data"   # not used by this script; reserved for your update pipeline
LOG_DIRNAME = "log"
DATA_DIRNAME = "data"

MASTER_XLSX_NAME = "prediction_log_all_models.xlsx"
SUMMARY_ALL_NAME = "summary_all_projects.txt"
STRATEGY_ALL_NAME = "strategy_all_projects.txt"
UPDATE_PARENT_SCRIPT = "update_parent_data_from_latest_v2.py"

# Strict sync rules: only same-class overwrites (prevents SPX -> US500 accidents)
DATA_SYNC_RULES = [
    # DIX
    (["DIX.xlsx"], ["DIX.xlsx"]),

    # VIX
    (["CBOE_DLY_VIX.csv", "VIX.csv", "vix.csv"], ["CBOE_DLY_VIX.csv", "VIX.csv", "vix.csv"]),

    # SPX / SP_SPX
    (["SP_SPX, 1D.csv"], ["SP_SPX, 1D.csv"]),
    (["SPX.csv"], ["SPX.csv"]),
    (["SPX, 1D.csv"], ["SPX, 1D.csv"]),

    # US500
    (["US500_1D.csv"], ["US500_1D.csv"]),
    (["US500.csv"], ["US500.csv"]),
    (["US500, 1D.csv"], ["US500, 1D.csv"]),
]

# Scripts
META_STRATEGY_SCRIPT = "daily_meta_strategy_engine_v3.py"
RUN_ALL_CANDIDATES = [
    "run_all_models_and_log.py",
    "run_all_models_and_log_v2.py",
    "run_all_models_and_log_v3.py",
    "run_all_models_and_log_latest.py",
    "run_all_models_and_log_v21.py",
    "run_all_models_and_log_v2_1.py",
    "run_all_models_and_log_v2_2.py",
]

# Project discovery: ignore these folders
SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", ".idea", "data", "latest data"}


# ----------------------------- Helpers -----------------------------

def _pick_existing_file(folder: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = folder / name
        if p.exists():
            return p
    return None


def pick_runner_for_project(project_dir: Path, fallback_python: str) -> list[str]:
    """Return a subprocess runner prefix list.（实现已收敛到 shared/runner.py）"""
    return shared_pick_runner(project_dir, fallback_python)


def run_script(runner: list[str], script_path: Path, cwd: Path, dry_run: bool = False) -> int:
    cmd = list(runner) + [str(script_path)]
    print(f"[RUN] ({cwd.name}) {' '.join([str(x) for x in cmd])}")
    if dry_run:
        return 0
    try:
        cp = subprocess.run(cmd, cwd=str(cwd), check=False)
        return int(cp.returncode)
    except FileNotFoundError as e:
        print(f"[ERR] runner not found: {e}")
        return 2
    except Exception as e:
        print(f"[ERR] run failed: {e}")
        return 3


def find_run_all_script(project_dir: Path) -> Path | None:
    for name in RUN_ALL_CANDIDATES:
        p = project_dir / name
        if p.exists():
            return p
    for p in sorted(project_dir.glob("run_all*.py")):
        return p
    return None


def find_run_all_script_with_registry(project_dir: Path, row: dict | None) -> Path | None:
    if row:
        rs = row.get("run_script")
        if isinstance(rs, str) and rs.strip():
            p = project_dir / rs
            if p.exists():
                return p
    return find_run_all_script(project_dir)


def find_meta_script_with_registry(project_dir: Path, row: dict | None) -> Path | None:
    if row:
        ms = row.get("strategy_script")
        if isinstance(ms, str) and ms.strip():
            p = project_dir / ms
            if p.exists():
                return p
            return None
    p = project_dir / META_STRATEGY_SCRIPT
    return p if p.exists() else None


def _safe_sheet_name(project_name: str) -> str:
    s = project_name.strip()
    return s if len(s) <= 31 else s[:31]


def _merge_rows_by_date(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new_rows.copy()

    if "date" not in existing.columns or "date" not in new_rows.columns:
        return pd.concat([existing, new_rows], ignore_index=True).drop_duplicates()

    ex = existing.copy()
    nw = new_rows.copy()

    ex["date"] = pd.to_datetime(ex["date"], errors="coerce")
    nw["date"] = pd.to_datetime(nw["date"], errors="coerce")
    ex = ex.dropna(subset=["date"])
    nw = nw.dropna(subset=["date"])

    all_cols = sorted(set(ex.columns) | set(nw.columns))
    ex = ex.reindex(columns=all_cols).astype("object")
    nw = nw.reindex(columns=all_cols).astype("object")

    ex = ex.set_index("date")
    nw = nw.set_index("date")

    out = ex.copy()
    out.update(nw)  # overwrite same date
    missing_idx = nw.index.difference(out.index)
    if len(missing_idx) > 0:
        out = pd.concat([out, nw.loc[missing_idx]], axis=0)

    return out.reset_index().sort_values("date")


def update_master_excel(master_xlsx: Path, sheet_name: str, new_rows: pd.DataFrame) -> None:
    if master_xlsx.exists():
        try:
            existing = pd.read_excel(master_xlsx, sheet_name=sheet_name)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    merged = _merge_rows_by_date(existing, new_rows)

    if master_xlsx.exists():
        with pd.ExcelWriter(master_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            merged.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(master_xlsx, engine="openpyxl", mode="w") as writer:
            merged.to_excel(writer, sheet_name=sheet_name, index=False)


def sync_data_files(root: Path, project_dir: Path, dry_run: bool) -> None:
    shared_dir = root / SHARED_DATA_DIRNAME
    project_data_dir = project_dir / DATA_DIRNAME
    if not project_data_dir.exists():
        return

    any_synced = False
    for root_names, proj_names in DATA_SYNC_RULES:
        src = _pick_existing_file(shared_dir, root_names)
        dst = _pick_existing_file(project_data_dir, proj_names)
        if src is None or dst is None:
            continue

        any_synced = True
        rel = dst.relative_to(project_dir)
        print(f"[SYNC] {project_dir.name}: {src.name} -> {rel.as_posix()}")
        if not dry_run:
            dst.write_bytes(src.read_bytes())

    if not any_synced:
        print("[SYNC] (no applicable files)")


def aggregate_prediction_log(project_dir: Path, master_xlsx: Path) -> None:
    log_dir = project_dir / LOG_DIRNAME
    log_file = log_dir / "prediction_log.csv"
    if not log_file.exists():
        print(f"[AGG] prediction_log not found under: {log_dir}")
        return

    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"[AGG] failed to read prediction_log: {e}")
        return

    if "date" not in df.columns:
        print(f"[AGG] prediction_log missing 'date' column: {log_file}")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df_tail = df.tail(2).copy()
    sheet = _safe_sheet_name(project_dir.name)

    try:
        update_master_excel(master_xlsx, sheet, df_tail)
        print(f"[AGG] prediction_log ({log_file.name}) -> {master_xlsx.name} [sheet={sheet}] rows={len(df_tail)}")
    except Exception as e:
        print(f"[WARN] failed to aggregate prediction_log: {e}")


def _latest_txt(log_dir: Path, suffix: str) -> Path | None:
    files = list(log_dir.glob(f"*{suffix}.txt"))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _latest_txt_by_regex(log_dir: Path, pattern: re.Pattern[str]) -> Path | None:
    files = [p for p in log_dir.glob("*.txt") if pattern.match(p.name)]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _read_text_with_retry(path: Path, retries: int = 6, sleep_sec: float = 0.4) -> str:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except PermissionError as e:
            last_err = e
            if i < retries - 1:
                time.sleep(sleep_sec)
            continue
    if last_err is not None:
        raise last_err
    return path.read_text(encoding="utf-8", errors="replace")


def aggregate_txt(project_dir: Path, summary_out: Path, strategy_out: Path) -> None:
    log_dir = project_dir / LOG_DIRNAME
    if not log_dir.exists():
        return

    # V1 latest
    re_v1_summary = re.compile(r"^\d{4}-\d{2}-\d{2}_summary\.txt$")
    re_v1_strategy = re.compile(r"^\d{4}-\d{2}-\d{2}_strategy\.txt$")
    # V2 latest
    re_v2_summary = re.compile(r"^v2_ord_rf_aa_realtime_\d{4}-\d{2}-\d{2}_summary\.txt$")
    re_v2_strategy = re.compile(r"^v2_ord_rf_aa_realtime_\d{4}-\d{2}-\d{2}_strategy\.txt$")

    latest_summary_files = [
        _latest_txt_by_regex(log_dir, re_v1_summary),
        _latest_txt_by_regex(log_dir, re_v2_summary),
    ]
    latest_strategy_files = [
        _latest_txt_by_regex(log_dir, re_v1_strategy),
        _latest_txt_by_regex(log_dir, re_v2_strategy),
    ]

    if not any(latest_summary_files):
        print("[AGG-TXT] No V1/V2 summary txt found.")
    else:
        for latest_summary in [p for p in latest_summary_files if p is not None]:
            try:
                txt = _read_text_with_retry(latest_summary)
                with open(summary_out, "a", encoding="utf-8") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"[{project_dir.name}] {latest_summary.name}\n")
                    f.write("=" * 80 + "\n")
                    f.write(txt)
                    if not txt.endswith("\n"):
                        f.write("\n")
            except Exception as e:
                print(f"[WARN] failed to aggregate summary txt for {project_dir.name}: {e}")

    if not any(latest_strategy_files):
        print("[AGG-TXT] No V1/V2 strategy txt found.")
    else:
        for latest_strategy in [p for p in latest_strategy_files if p is not None]:
            try:
                txt = _read_text_with_retry(latest_strategy)
                with open(strategy_out, "a", encoding="utf-8") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"[{project_dir.name}] {latest_strategy.name}\n")
                    f.write("=" * 80 + "\n")
                    f.write(txt)
                    if not txt.endswith("\n"):
                        f.write("\n")
            except Exception as e:
                print(f"[WARN] failed to aggregate strategy txt for {project_dir.name}: {e}")


def find_projects(root: Path) -> list[Path]:
    projects: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.lower() in SKIP_DIRS:
            continue
        if (p / "config.py").exists() or (p / "data").exists() or (p / "log").exists():
            projects.append(p)
    return projects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None)
    ap.add_argument("--python", type=str, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_update", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    pyexe = args.python if args.python else sys.executable
    dry_run = bool(args.dry_run)

    shared_dir = root / SHARED_DATA_DIRNAME
    if not shared_dir.exists():
        print(f"[ERR] shared data dir not found: {shared_dir}")
        return

    master_xlsx = root / MASTER_XLSX_NAME
    summary_out = root / SUMMARY_ALL_NAME
    strategy_out = root / STRATEGY_ALL_NAME

    print(f"[INFO] root={root}")
    print(f"[INFO] shared_data={shared_dir}")
    print(f"[INFO] python={pyexe}")
    print(f"[INFO] dry_run={dry_run}")

    # Optional integrated data update step:
    # exported data -> data/latest data -> data
    if not args.skip_update:
        updater = root / UPDATE_PARENT_SCRIPT
        if not updater.exists():
            print(f"[ERR] update script not found: {updater}")
            return
        if dry_run:
            print(f"[STEP] Update parent data (dry-run skip execute): {updater.name}")
        else:
            print(f"[STEP] Update parent data: {updater.name}")
            rc = run_script([pyexe], updater, cwd=root, dry_run=False)
            if rc != 0:
                print(f"[ERR] parent data update failed, return_code={rc}")
                return
    else:
        print("[STEP] Update parent data skipped (--skip_update)")

    projects = find_projects(root)
    registry = load_registry(root)
    reg_map = registry_map_by_name(registry)
    print(f"[INFO] registry_loaded={'yes' if reg_map else 'no'}")
    validate_registry(root, reg_map)

    # Minimal integration:
    # - if include_in_daily_pipeline is false in registry, skip project.
    # - run/strategy script can be overridden by registry fields.
    selected_projects: list[Path] = []
    for p in projects:
        row = reg_map.get(p.name)
        if row and row.get("include_in_daily_pipeline") is False:
            print(f"[SKIP] registry include_in_daily_pipeline=false: {p.name}")
            continue
        selected_projects.append(p)

    print(f"[INFO] Found {len(selected_projects)} projects:")
    for p in selected_projects:
        print(f"  - {p.name}")

    # Reset aggregate txt each run (overwrite)
    if not dry_run:
        for fp in (summary_out, strategy_out):
            if fp.exists():
                fp.unlink()

    for proj in selected_projects:
        print("\n" + "#" * 80)
        print(f"[PROJECT] {proj.name}")

        row = reg_map.get(proj.name)
        runner = pick_runner_for_project(proj, pyexe)
        print(f"[INFO] ({proj.name}) runner={' '.join(runner)}")

        sync_data_files(root, proj, dry_run)

        run_all = find_run_all_script_with_registry(proj, row)
        if run_all is None:
            print("[RUN] run_all_models script not found, skip.")
        else:
            rc = run_script(runner, run_all, cwd=proj, dry_run=dry_run)
            if rc != 0:
                print(f"[WARN] run_all_models returned code={rc}")

        meta = find_meta_script_with_registry(proj, row)
        if meta is not None:
            rc = run_script(runner, meta, cwd=proj, dry_run=dry_run)
            if rc != 0:
                print(f"[WARN] daily_meta_strategy returned code={rc}")
        else:
            print("[RUN] daily_meta_strategy_engine_v3.py not found, skip.")

        if not dry_run:
            aggregate_prediction_log(proj, master_xlsx)
            aggregate_txt(proj, summary_out, strategy_out)

    print("\n" + "=" * 80)
    print("[DONE]")
    print(f"- Master Excel  : {master_xlsx}")
    print(f"- Summary TXT   : {summary_out}")
    print(f"- Strategy TXT  : {strategy_out}")


if __name__ == "__main__":
    main()
