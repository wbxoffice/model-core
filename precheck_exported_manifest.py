#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pre-run completeness gate for the daily pipeline.

Checks `exported data/` against the expected manifest BEFORE update_parent runs.
If any required item is missing, unreadable, or has neither new exported rows nor
an already up-to-date local baseline, it prints the full checklist and asks the
user whether to continue with an incomplete manifest.

Designed to run as launcher step0 in runlog mode: because that mode redirects
stdout to the log file, the human-facing checklist + prompt are written to the
controlling terminal (/dev/tty) so they are actually visible on screen.

Exit codes (launch.py halts the chain on non-zero):
  0  manifest OK, or user chose to continue, or --allow-incomplete
  1  incomplete and user declined, or no TTY and not --allow-incomplete
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


UPDATE_MANIFEST_NAME = "update_parent_data_manifest.json"


# display name -> (accepted filename prefixes, date column, required?, baseline files)
MANIFEST = [
    ("SPX 日线 (1D)", ("SP_DLY_SPX, 1D", "SP_DLY_SPX,1D", "SPCFD_SPX, 1D", "SPCFD_SPX,1D", "SP_SPX, 1D", "SP_SPX,1D",
                       "CBOE_DLY_SPX, 1D", "CBOE_DLY_SPX,1D"), "time", True, ("SP_SPX, 1D.csv",)),
    ("SPX 30分钟",    ("SP_DLY_SPX, 30", "SP_DLY_SPX,30", "SPCFD_SPX, 30", "SPCFD_SPX,30", "SP_SPX, 30", "SP_SPX,30",
                       "CBOE_DLY_SPX, 30", "CBOE_DLY_SPX,30"), "time", True, ("SP_SPX, 30.csv",)),
    ("US500 日线",    ("CAPITALCOM_SPX500, 1D", "CAPITALCOM_SPX500,1D", "CAPITALCOM_US500, 1D",
                       "CAPITALCOM_US500,1D", "US500, 1D", "US500,1D"), "time", True, ("US500_1D.csv",)),
    ("VIX 日线",      ("CBOE_DLY_VIX, 1D", "CBOE_DLY_VIX,1D", "CBOE_DLY_VIX"), "time", True, ("CBOE_DLY_VIX.csv",)),
    ("DIX",          ("DIX.csv", "DIX,", "DIX "), "date", True, ("DIX.xlsx",)),
    ("SPX 5分钟 (开盘90分钟)", ("SP_DLY_SPX, 5", "SP_DLY_SPX,5", "SPCFD_SPX, 5", "SPCFD_SPX,5", "SP_SPX, 5", "SP_SPX,5"), "time", False, tuple()),
]


def _expected_latest_open_day(root: Path):
    cal = root / "data" / "nyse_open_days.csv"
    if not cal.exists():
        return None
    try:
        df = pd.read_csv(cal)
        col = "date" if "date" in df.columns else ("time" if "time" in df.columns else df.columns[0])
        days = pd.to_datetime(df[col], errors="coerce").dropna().dt.normalize()
        today = pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None).normalize()
        past = days[days <= today]
        return past.max() if not past.empty else None
    except Exception:
        return None


def _file_latest_date(path: Path, date_col: str):
    try:
        wanted = {date_col.lower(), "time", "date", "datetime"}
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path, usecols=lambda c: str(c).strip().lower() in wanted)
        else:
            df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in wanted)
        col = next((c for c in df.columns if str(c).strip().lower() == date_col), None)
        if col is None:
            col = df.columns[0]
        # Compare calendar dates only. Parse as UTC to handle tz-aware intraday
        # bars uniformly, then drop tz WITHOUT shifting (naive daily midnights
        # must not roll back a day). UTC calendar date == US session date for RTH.
        dt = pd.to_datetime(df[col], errors="coerce", utc=True)
        if dt.notna().any():
            return dt.max().tz_localize(None).normalize()
        dt = pd.to_datetime(df[col], errors="coerce")
        return dt.max().normalize() if dt.notna().any() else None
    except Exception:
        return None  # unreadable -> likely a cloud placeholder not yet downloaded


def _baseline_latest_date(root: Path, baseline_files: tuple[str, ...], date_col: str):
    latest = None
    for fname in baseline_files:
        for folder in (root / "data", root / "data" / "latest data"):
            p = folder / fname
            if not p.exists():
                continue
            dt = _file_latest_date(p, date_col)
            if dt is not None and (latest is None or dt > latest):
                latest = dt
    return latest


def _load_update_manifest(root: Path) -> dict | None:
    path = root / "data" / UPDATE_MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _manifest_is_recent(manifest: dict | None, max_age_hours: float = 18.0) -> bool:
    if not manifest:
        return False
    try:
        ts = pd.Timestamp(manifest.get("generated_at"))
        if pd.isna(ts):
            return False
        age_hours = (pd.Timestamp.now() - ts).total_seconds() / 3600.0
        return 0 <= age_hours <= max_age_hours
    except Exception:
        return False


def _manifest_processed_status(manifest: dict | None, baseline_files: tuple[str, ...]) -> tuple[bool, str]:
    """Return whether the latest parent update already handled this canonical file."""
    if not _manifest_is_recent(manifest):
        return False, ""
    targets = set(baseline_files)
    if not targets:
        return False, ""

    imported = []
    for rec in manifest.get("ingested_exports", []) or []:
        if rec.get("target_file") in targets and rec.get("status") == "imported":
            imported.append(rec)
    if imported:
        best = imported[-1]
        latest = best.get("target_after_latest") or best.get("source_latest") or "NA"
        src = best.get("source_file") or "unknown"
        removed = "已删除源文件" if best.get("removed_exported") else "源文件未删除"
        return True, f"OK 已由更新脚本导入 (到 {latest}, {src}, {removed})"

    updated = []
    for rec in manifest.get("master_updates", []) or []:
        if rec.get("file") in targets and rec.get("status") in {"updated", "no_new_rows"}:
            updated.append(rec)
    if updated:
        best = updated[-1]
        latest = best.get("master_after_latest") or best.get("master_before_latest") or "NA"
        status = "更新" if best.get("status") == "updated" else "检查无新增"
        return True, f"OK 更新脚本已{status}本地数据 (到 {latest})"

    return False, ""


def _scan(root: Path):
    ex_dir = root / "exported data"
    files = [p for p in ex_dir.iterdir() if p.is_file()] if ex_dir.exists() else []
    expected_latest = _expected_latest_open_day(root)
    update_manifest = _load_update_manifest(root)
    rows = []
    problems = []  # required items that are missing/stale/unreadable
    soft_stale = []  # stale only matters when nothing new is available at all
    has_required_increment = False
    for name, prefixes, date_col, required, baseline_files in MANIFEST:
        # 新旧前缀文件可能并存(如 SPCFD_SPX / SP_DLY_SPX)，取最近修改的那个
        matches = [p for p in files if p.name.startswith(prefixes)]
        match = max(matches, key=lambda p: p.stat().st_mtime) if matches else None
        baseline_latest = _baseline_latest_date(root, baseline_files, date_col)
        manifest_ok, manifest_status = _manifest_processed_status(update_manifest, baseline_files)
        if match is None:
            if manifest_ok:
                status = manifest_status
            elif expected_latest is not None and baseline_latest is not None and baseline_latest >= expected_latest:
                status = f"OK 本地已到 {baseline_latest.date()} (无 exported 源文件)"
            else:
                status = "缺失"
            latest = None
            local_ok = expected_latest is not None and baseline_latest is not None and baseline_latest >= expected_latest
            if required and not manifest_ok and not local_ok:
                problems.append(name)
        else:
            latest = _file_latest_date(match, date_col)
            if latest is None:
                status = "无法读取(可能云端未下载)"
                if required:
                    problems.append(name)
            elif baseline_latest is not None and latest > baseline_latest:
                has_required_increment = has_required_increment or required
                status = f"OK 有新增 (导出到 {latest.date()}, 本地到 {baseline_latest.date()})"
            elif expected_latest is not None and baseline_latest is not None and baseline_latest >= expected_latest:
                status = f"OK 本地已到 {baseline_latest.date()} (导出到 {latest.date()})"
            elif expected_latest is not None and latest < expected_latest:
                base_txt = f", 本地到 {baseline_latest.date()}" if baseline_latest is not None else ""
                if manifest_ok:
                    status = manifest_status + f"; 导出提示: 到 {latest.date()}{base_txt}, 参考 {expected_latest.date()}"
                else:
                    status = f"提示: 无新增/早于参考 (导出到 {latest.date()}{base_txt}, 参考 {expected_latest.date()})"
                if required and not manifest_ok:
                    soft_stale.append(name)
            else:
                status = f"OK (到 {latest.date()})"
        rows.append((name, required, match.name if match else "-", status))
    if not has_required_increment:
        problems.extend(soft_stale)
    return rows, problems, expected_latest


def _render(rows, expected_latest) -> str:
    lines = []
    el = expected_latest.date() if expected_latest is not None else "未知(无日历)"
    lines.append(f"[导出清单体检] exported data/  期望最新交易日={el}")
    lines.append(f"{'项目':<22} {'必需':<5} {'状态':<32} 文件")
    for name, required, fname, status in rows:
        req = "必需" if required else "可选"
        lines.append(f"{name:<22} {req:<5} {status:<32} {fname}")
    return "\n".join(lines)


def _ask_via_tty(prompt: str):
    """Write prompt to the controlling terminal and read the answer there.
    Returns the stripped lowercase answer, or None if no TTY is available."""
    try:
        tty = open("/dev/tty", "r+")
    except Exception:
        return None
    try:
        tty.write(prompt)
        tty.flush()
        ans = tty.readline()
        return (ans or "").strip().lower()
    finally:
        tty.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="清单不完整也不询问、直接继续(自动化/无终端场景)。")
    ap.add_argument("--strict", action="store_true",
                    help="清单不完整直接中止、不询问。")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    rows, problems, expected_latest = _scan(root)
    report = _render(rows, expected_latest)
    print(report)  # to runlog

    if not problems:
        print("[OK] 导出清单完整,继续日跑。")
        return 0

    warn = "\n[注意] 以下必需项缺失/过旧: " + ", ".join(problems)
    print(warn)

    if args.allow_incomplete:
        print("[--allow-incomplete] 跳过询问,以不完整清单继续。")
        return 0
    if args.strict:
        print("[--strict] 清单不完整,已中止日跑。")
        return 1

    # Interactive prompt on the real terminal (runlog redirects stdout to log).
    msg = (
        "\n" + report + warn +
        "\n\n上述必需项不完整(常见原因: OneDrive 尚未把导出文件同步到本地)。\n"
        "是否仍以不完整清单继续日跑? [y/N]: "
    )
    ans = _ask_via_tty(msg)
    if ans is None:
        print("[ABORT] 清单不完整且无可交互终端(自动化场景)。如需强制继续请加 --allow-incomplete。")
        return 1
    if ans in ("y", "yes"):
        print("[继续] 用户确认以不完整清单继续日跑。")
        return 0
    print("[ABORT] 用户选择不继续。请等 OneDrive 同步齐或补导出后重试。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
