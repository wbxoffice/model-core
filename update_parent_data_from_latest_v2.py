#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import json
from datetime import datetime
import pandas as pd

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
LATEST_DIR = DATA_DIR / "latest data"
EXPORTED_DIR = ROOT / "exported data"
MANIFEST_FILE = DATA_DIR / "update_parent_data_manifest.json"
KEEP_LATEST_DAYS = 300

# 文件名 -> 日期列名（强约束）
FILES = {
    "SP_SPX, 1D.csv": "time",
    "US500_1D.csv": "time",
    "CBOE_DLY_VIX.csv": "time",
    "DIX.xlsx": "date",
}

# exported data filename prefix -> latest data canonical filename
EXPORTED_TO_LATEST = {
    "SP_DLY_SPX, 1D": ("SP_SPX, 1D.csv", "time"),
    "SP_DLY_SPX,1D": ("SP_SPX, 1D.csv", "time"),
    "SPCFD_SPX, 1D": ("SP_SPX, 1D.csv", "time"),
    "SPCFD_SPX,1D": ("SP_SPX, 1D.csv", "time"),
    "SP_SPX, 1D": ("SP_SPX, 1D.csv", "time"),
    "SP_SPX,1D": ("SP_SPX, 1D.csv", "time"),
    "CBOE_DLY_SPX, 1D": ("SP_SPX, 1D.csv", "time"),
    "CBOE_DLY_SPX,1D": ("SP_SPX, 1D.csv", "time"),
    "CAPITALCOM_US500, 1D": ("US500_1D.csv", "time"),
    "CAPITALCOM_US500,1D": ("US500_1D.csv", "time"),
    "CAPITALCOM_SPX500, 1D": ("US500_1D.csv", "time"),
    "CAPITALCOM_SPX500,1D": ("US500_1D.csv", "time"),
    "US500, 1D": ("US500_1D.csv", "time"),
    "US500,1D": ("US500_1D.csv", "time"),
    "CBOE_DLY_VIX, 1D": ("CBOE_DLY_VIX.csv", "time"),
}

def _exported_prefix_priority(filename: str) -> int:
    """Higher value means higher priority when both sources exist for same timeframe."""
    if filename.startswith("SP_DLY_SPX, 1D") or filename.startswith("SP_DLY_SPX,1D"):
        return 4
    if filename.startswith("SPCFD_SPX, 1D") or filename.startswith("SPCFD_SPX,1D"):
        return 3
    if filename.startswith("SP_SPX, 1D") or filename.startswith("SP_SPX,1D"):
        return 2
    if filename.startswith("CBOE_DLY_SPX, 1D") or filename.startswith("CBOE_DLY_SPX,1D"):
        return 1
    return 0


def _infer_exported_date_col(filename: str) -> str | None:
    lname = filename.lower()
    if lname == "dix.csv":
        return "date"
    for prefix, (_, date_col) in EXPORTED_TO_LATEST.items():
        if filename.startswith(prefix) and filename.endswith(".csv"):
            return date_col
    return None


def _exported_file_latest_date(path: Path) -> pd.Timestamp:
    col = _infer_exported_date_col(path.name)
    if col is None:
        return pd.NaT
    try:
        df = pd.read_csv(path, usecols=lambda c: c.strip().lower() in {col.lower(), "time", "datetime", "date"})
        if df.empty:
            return pd.NaT
        pick = None
        for c in [col, "time", "datetime", "date"]:
            if c in df.columns:
                pick = c
                break
        if pick is None:
            return pd.NaT
        dt = pd.to_datetime(df[pick], errors="coerce")
        return dt.max()
    except Exception:
        return pd.NaT


def _latest_date_in_file(path: Path, date_col: str) -> pd.Timestamp:
    try:
        df = read_file(path)
        if date_col not in df.columns:
            lower = {str(c).strip().lower(): c for c in df.columns}
            date_col = lower.get(date_col.lower(), date_col)
        if date_col not in df.columns:
            return pd.NaT
        df = normalize_date(df, date_col)
        return df[date_col].max()
    except Exception:
        return pd.NaT


def _date_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return None


def read_file(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_csv(p)


def write_file(df: pd.DataFrame, p: Path):
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df.to_excel(p, index=False)
    else:
        df.to_csv(p, index=False)


def normalize_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    df[col] = df[col].dt.normalize()
    return df


def _merge_by_date(df_base: pd.DataFrame, df_new: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df_b = normalize_date(df_base, date_col)
    df_n = normalize_date(df_new, date_col)

    all_cols = sorted(set(df_b.columns) | set(df_n.columns))
    df_b = df_b.reindex(columns=all_cols)
    df_n = df_n.reindex(columns=all_cols)

    out = (
        pd.concat([df_b, df_n], ignore_index=True)
        .sort_values(date_col)
        .drop_duplicates(subset=[date_col], keep="last")
    )
    return out


def trim_latest_file(fname: str, date_col: str, keep_days: int):
    p = LATEST_DIR / fname
    rec = {
        "file": fname,
        "path": str(p),
        "status": "pending",
        "before_days": None,
        "after_days": None,
    }
    if not p.exists():
        print(f"[SKIP] latest trim target not found: {fname}")
        rec["status"] = "missing"
        return rec
    try:
        df = read_file(p)
        df = normalize_date(df, date_col).sort_values(date_col)
        uniq_days = sorted(df[date_col].dropna().unique())
        rec["before_days"] = int(len(uniq_days))
        if len(uniq_days) <= keep_days:
            print(f"[TRIM] {fname}: keep {len(uniq_days)} day(s), no trim.")
            rec["status"] = "no_trim"
            rec["after_days"] = int(len(uniq_days))
            return rec

        keep_set = set(uniq_days[-keep_days:])
        out = df[df[date_col].isin(keep_set)].copy().sort_values(date_col)
        write_file(out, p)
        rec["status"] = "trimmed"
        rec["after_days"] = int(out[date_col].nunique())
        print(
            f"[TRIM] {fname}: {len(uniq_days)} -> {out[date_col].nunique()} day(s) "
            f"(kept last {keep_days})"
        )
    except Exception as e:
        print(f"[ERR] trim latest {fname} failed: {e}")
        rec["status"] = "error"
        rec["error"] = str(e)
    return rec


def trim_latest_all(keep_days: int = KEEP_LATEST_DAYS):
    records = []
    for fname, date_col in FILES.items():
        records.append(trim_latest_file(fname, date_col, keep_days))
    return records


def ingest_exported_data_into_latest():
    records = []
    if not EXPORTED_DIR.exists():
        print(f"[INFO] EXPORTED_DIR not found: {EXPORTED_DIR}")
        return records

    exported_files = [p for p in EXPORTED_DIR.iterdir() if p.is_file()]
    # Process older files first, newer files last.
    # Tie-break: SP_DLY_SPX > SPCFD_SPX > SP_SPX > CBOE_DLY_SPX when recency is similar.
    scored = []
    for p in exported_files:
        latest_dt = _exported_file_latest_date(p)
        scored.append(
            (
                p,
                pd.notna(latest_dt),
                latest_dt if pd.notna(latest_dt) else pd.Timestamp.min,
                pd.Timestamp.fromtimestamp(p.stat().st_mtime),
                _exported_prefix_priority(p.name),
            )
        )
    scored.sort(key=lambda x: (x[1], x[2], x[3], x[4]))
    exported_files = [x[0] for x in scored]
    if not exported_files:
        print("[INFO] EXPORTED_DIR is empty.")
        return records

    print(f"[INFO] EXPORTED_DIR = {EXPORTED_DIR}")
    consumed: list[Path] = []

    for src in exported_files:
        sname = src.name
        lname = sname.lower()

        # DIX.csv -> latest data/DIX.xlsx
        if lname == "dix.csv":
            target = LATEST_DIR / "DIX.xlsx"
            rec = {
                "source_file": sname,
                "source_path": str(src),
                "target_file": "DIX.xlsx",
                "target_path": str(target),
                "date_col": "date",
                "source_latest": _date_text(_exported_file_latest_date(src)),
                "target_before_latest": _date_text(_latest_date_in_file(target, "date")) if target.exists() else None,
                "status": "pending",
                "removed_exported": False,
            }
            if not target.exists():
                print(f"[SKIP] latest target not found for DIX: {target}")
                rec["status"] = "target_missing"
                records.append(rec)
                continue

            try:
                df_src = pd.read_csv(src)
                df_tgt = pd.read_excel(target)
                out = _merge_by_date(df_tgt, df_src, "date")
                write_file(out, target)
                rec["status"] = "imported"
                rec["source_rows"] = int(len(df_src))
                rec["target_after_latest"] = _date_text(out["date"].max())
                rec["target_days"] = int(out["date"].nunique())
                print(
                    f"[IMPORT] {sname} -> {target.name}: "
                    f"rows={len(df_src)}, latest_days={out['date'].nunique()}"
                )
                consumed.append(src)
            except Exception as e:
                print(f"[ERR] import DIX.csv failed: {e}")
                rec["status"] = "error"
                rec["error"] = str(e)
            records.append(rec)
            continue

        # Pattern files with random suffix IDs, e.g. SP_DLY_SPX, 1D_xxxxx.csv
        matched = False
        for prefix, (target_name, date_col) in EXPORTED_TO_LATEST.items():
            if sname.startswith(prefix) and sname.endswith(".csv"):
                matched = True
                target = LATEST_DIR / target_name
                rec = {
                    "source_file": sname,
                    "source_path": str(src),
                    "target_file": target_name,
                    "target_path": str(target),
                    "date_col": date_col,
                    "source_latest": _date_text(_exported_file_latest_date(src)),
                    "target_before_latest": _date_text(_latest_date_in_file(target, date_col)) if target.exists() else None,
                    "status": "pending",
                    "removed_exported": False,
                }
                if not target.exists():
                    print(f"[SKIP] latest target not found: {target}")
                    rec["status"] = "target_missing"
                    records.append(rec)
                    break
                try:
                    df_src = pd.read_csv(src)
                    df_tgt = pd.read_csv(target)
                    out = _merge_by_date(df_tgt, df_src, date_col)
                    write_file(out, target)
                    rec["status"] = "imported"
                    rec["source_rows"] = int(len(df_src))
                    rec["target_after_latest"] = _date_text(out[date_col].max())
                    rec["target_days"] = int(out[date_col].nunique())
                    print(
                        f"[IMPORT] {sname} -> {target.name}: "
                        f"rows={len(df_src)}, latest_days={out[date_col].nunique()}"
                    )
                    consumed.append(src)
                except Exception as e:
                    print(f"[ERR] import {sname} failed: {e}")
                    rec["status"] = "error"
                    rec["error"] = str(e)
                records.append(rec)
                break

        if not matched:
            print(f"[SKIP] unrecognized exported file: {sname}")

    for p in consumed:
        try:
            p.unlink()
            print(f"[CLEAN] removed exported file: {p.name}")
            for rec in records:
                if rec.get("source_file") == p.name and rec.get("status") == "imported":
                    rec["removed_exported"] = True
        except Exception as e:
            print(f"[WARN] failed to remove exported file {p.name}: {e}")
            for rec in records:
                if rec.get("source_file") == p.name and rec.get("status") == "imported":
                    rec["removed_exported"] = False
                    rec["remove_error"] = str(e)

    if not consumed:
        print("[INFO] no exported files were consumed.")
    return records


def update_one(fname: str, date_col: str):
    master = DATA_DIR / fname
    latest = LATEST_DIR / fname
    rec = {
        "file": fname,
        "master_path": str(master),
        "latest_path": str(latest),
        "date_col": date_col,
        "status": "pending",
        "master_before_latest": None,
        "latest_latest": None,
        "master_after_latest": None,
        "incremental_days": 0,
    }

    if not master.exists():
        print(f"[SKIP] master not found: {fname}")
        rec["status"] = "master_missing"
        return rec
    if not latest.exists():
        print(f"[SKIP] latest not found: {fname}")
        rec["status"] = "latest_missing"
        return rec

    df_m = normalize_date(read_file(master), date_col)
    df_l = normalize_date(read_file(latest), date_col)

    last_dt = df_m[date_col].max()
    rec["master_before_latest"] = _date_text(last_dt)
    rec["latest_latest"] = _date_text(df_l[date_col].max())
    if pd.isna(last_dt):
        print(f"[SKIP] {fname}: no valid date in master")
        rec["status"] = "no_valid_master_date"
        return rec

    # 取 >= last_dt（覆盖最后一天）
    df_inc = df_l[df_l[date_col] >= last_dt]
    if df_inc.empty:
        print(f"[OK] {fname}: no new rows (last={last_dt.date()})")
        rec["status"] = "no_new_rows"
        rec["master_after_latest"] = rec["master_before_latest"]
        return rec

    # 列对齐
    all_cols = sorted(set(df_m.columns) | set(df_inc.columns))
    df_m = df_m.reindex(columns=all_cols)
    df_inc = df_inc.reindex(columns=all_cols)

    df_out = (
        pd.concat([df_m, df_inc], ignore_index=True)
        .sort_values(date_col)
        .drop_duplicates(subset=[date_col], keep="last")
    )

    write_file(df_out, master)
    rec["status"] = "updated"
    rec["incremental_days"] = int(df_inc[date_col].nunique())
    rec["incremental_min"] = _date_text(df_inc[date_col].min())
    rec["incremental_max"] = _date_text(df_inc[date_col].max())
    rec["master_after_latest"] = _date_text(df_out[date_col].max())

    print(
        f"[UPDATE] {fname}: "
        f"+{df_inc[date_col].nunique()} day(s) "
        f"({df_inc[date_col].min().date()} ~ {df_inc[date_col].max().date()})"
    )
    return rec


def _write_update_manifest(records: dict):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[MANIFEST] wrote {MANIFEST_FILE}")


def main():
    print(f"[INFO] DATA_DIR   = {DATA_DIR}")
    print(f"[INFO] LATEST_DIR = {LATEST_DIR}")
    print(f"[INFO] EXPORTED_DIR = {EXPORTED_DIR}")
    print(f"[INFO] KEEP_LATEST_DAYS = {KEEP_LATEST_DAYS}")
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "data_dir": str(DATA_DIR),
        "latest_dir": str(LATEST_DIR),
        "exported_dir": str(EXPORTED_DIR),
        "ingested_exports": [],
        "trim_latest": [],
        "master_updates": [],
    }

    # STEP 0: ingest exported raw files into latest data
    manifest["ingested_exports"] = ingest_exported_data_into_latest()
    # STEP 0.5: keep latest data window small
    manifest["trim_latest"] = trim_latest_all(KEEP_LATEST_DAYS)

    for fname, date_col in FILES.items():
        try:
            manifest["master_updates"].append(update_one(fname, date_col))
        except Exception as e:
            print(f"[ERR] {fname}: {e}")
            manifest["master_updates"].append(
                {
                    "file": fname,
                    "date_col": date_col,
                    "status": "error",
                    "error": str(e),
                }
            )
    _write_update_manifest(manifest)


if __name__ == "__main__":
    main()
