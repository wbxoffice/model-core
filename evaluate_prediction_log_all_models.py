#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import re
import sys

import pandas as pd


def sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r'[\\/:*?"<>|]', "", s)
    return s[:80] if len(s) > 80 else s


def ask_int(prompt: str, default: int) -> int:
    while True:
        x = input(prompt).strip()
        if not x:
            return default
        try:
            n = int(x)
            if n <= 0:
                print("[ERR] Please enter an integer > 0.")
                continue
            return n
        except ValueError:
            print("[ERR] Please enter an integer.")


def ask_window_n(max_days: int, default_days: int = 30) -> int:
    while True:
        x = input(
            f"Enter evaluation window days (integer), 'max' for all, Enter={default_days}: "
        ).strip().lower()
        if not x:
            return default_days
        if x in {"max", "all", "m"}:
            return max_days
        try:
            n = int(x)
            if n <= 0:
                print("[ERR] Please enter an integer > 0, or 'max'.")
                continue
            return n
        except ValueError:
            print("[ERR] Invalid input. Use integer or 'max'.")


def ask_choice(max_choice: int) -> int:
    while True:
        x = input(f"Enter selection number (1-{max_choice}): ").strip()
        try:
            n = int(x)
            if 1 <= n <= max_choice:
                return n
            print("[ERR] Selection out of range.")
        except ValueError:
            print("[ERR] Please enter an integer.")


def coerce_date_col(df: pd.DataFrame) -> pd.DataFrame:
    # try common date column names
    for c in ["date", "Date", "dt", "Datetime", "datetime"]:
        if c in df.columns:
            df = df.copy()
            df[c] = pd.to_datetime(df[c], errors="coerce")
            # standardize to 'date'
            if c != "date":
                df.rename(columns={c: "date"}, inplace=True)
            return df
    # fallback: if first column looks like date
    if len(df.columns) > 0:
        c0 = df.columns[0]
        tmp = pd.to_datetime(df[c0], errors="coerce")
        if tmp.notna().mean() > 0.8:
            df = df.copy()
            df["date"] = tmp
            if c0 != "date":
                # keep original col too, but date will be used for sorting
                pass
            return df
    return df


def date_coverage_stats(df: pd.DataFrame) -> dict:
    out = {
        "rows_total": int(len(df)),
        "rows_with_date": 0,
        "unique_days": 0,
        "date_min": None,
        "date_max": None,
        "missing_bdays_count": None,
        "missing_bdays_preview": [],
    }

    if "date" not in df.columns:
        return out

    d = pd.to_datetime(df["date"], errors="coerce").dropna().dt.normalize()
    out["rows_with_date"] = int(len(d))
    if len(d) == 0:
        return out

    uniq = pd.DatetimeIndex(sorted(d.unique()))
    out["unique_days"] = int(len(uniq))
    out["date_min"] = str(uniq.min().date())
    out["date_max"] = str(uniq.max().date())

    expected = pd.bdate_range(uniq.min(), uniq.max())
    missing = expected.difference(uniq)
    out["missing_bdays_count"] = int(len(missing))
    out["missing_bdays_preview"] = [str(x.date()) for x in missing[:20]]
    return out


def basic_metrics_from_predlog(df: pd.DataFrame) -> dict:
    """
    Generic / robust metrics that don't assume exact column set.
    We look for common pairs:
    - vol_pred_regime vs vol_real_regime / vol_real / vol_regime
    - dir_flag/dir_pred vs real_dir / dir_real / dir_label
    - trend_flag vs real_trend
    - reversal flags
    """
    out = {}

    def acc(pred_col, real_col, name):
        if pred_col in df.columns and real_col in df.columns:
            s = df[[pred_col, real_col]].dropna()
            if len(s) == 0:
                out[name] = None
            else:
                out[name] = float((s[pred_col] == s[real_col]).mean())
        else:
            out[name] = None

    # candidate columns
    candidates = {
        "vol_pred": ["vol_pred_regime", "vol_pred", "pred_vol", "vol_regime_pred"],
        "vol_real": ["vol_real_regime", "vol_real", "real_vol", "vol_regime_real", "real_regime"],
        "dir_pred": ["dir_flag", "dir_pred", "pred_dir", "direction_pred"],
        "dir_real": ["dir_real", "real_dir", "direction_real", "direction", "dir_label"],
        "trend_pred": ["trend_flag", "trend_pred", "pred_trend"],
        "trend_real": ["trend_real", "real_trend", "trend_label"],
        "rev_pred": ["reversal_risk", "rev_pred", "pred_rev", "reversal_pred"],
        "rev_real": ["real_reversal", "rev_real", "reversal_real"],
    }

    def pick(keys):
        for k in keys:
            if k in df.columns:
                return k
        return None

    vol_p, vol_r = pick(candidates["vol_pred"]), pick(candidates["vol_real"])
    dir_p, dir_r = pick(candidates["dir_pred"]), pick(candidates["dir_real"])
    tr_p, tr_r = pick(candidates["trend_pred"]), pick(candidates["trend_real"])
    rv_p, rv_r = pick(candidates["rev_pred"]), pick(candidates["rev_real"])

    if vol_p and vol_r:
        acc(vol_p, vol_r, "vol_accuracy")
    if dir_p and dir_r:
        acc(dir_p, dir_r, "dir_accuracy")
    if tr_p and tr_r:
        acc(tr_p, tr_r, "trend_accuracy")
    if rv_p and rv_r:
        acc(rv_p, rv_r, "reversal_accuracy")

    out["rows_used"] = int(len(df))
    out["date_min"] = str(df["date"].min().date()) if "date" in df.columns and df["date"].notna().any() else None
    out["date_max"] = str(df["date"].max().date()) if "date" in df.columns and df["date"].notna().any() else None
    out["has_date_col"] = "date" in df.columns
    return out


def evaluate_one_sheet(master_xlsx: Path, sheet: str, n_last: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_excel(master_xlsx, sheet_name=sheet)
    df = coerce_date_col(df)

    if "date" in df.columns:
        df = df.sort_values("date")
        # keep last N rows with non-null date if possible
        df2 = df[df["date"].notna()].copy()
        if len(df2) > 0:
            df = df2.tail(n_last).copy()
        else:
            df = df.tail(n_last).copy()
    else:
        df = df.tail(n_last).copy()

    metrics = basic_metrics_from_predlog(df)

    # Summary row dataframe
    summary = pd.DataFrame([{
        "sheet": sheet,
        "rows_used": metrics.get("rows_used"),
        "date_min": metrics.get("date_min"),
        "date_max": metrics.get("date_max"),
        "vol_accuracy": metrics.get("vol_accuracy"),
        "dir_accuracy": metrics.get("dir_accuracy"),
        "trend_accuracy": metrics.get("trend_accuracy"),
        "reversal_accuracy": metrics.get("reversal_accuracy"),
    }])

    return df, summary, metrics


def main():
    root = Path(__file__).resolve().parent
    master_xlsx = root / "prediction_log_all_models.xlsx"
    outdir = root / "model evaluation"
    outdir.mkdir(parents=True, exist_ok=True)

    if not master_xlsx.exists():
        print("[ERR] Missing:", master_xlsx)
        sys.exit(2)

    xls = pd.ExcelFile(master_xlsx)
    sheets = xls.sheet_names

    print("\n===========================================")
    print(" Prediction Log ALL-MODELS Evaluator")
    print("===========================================")
    print("ROOT :", root)
    print("INPUT:", master_xlsx.name)
    print("")

    print("\n===========================================")
    print(" Data Coverage (full prediction log)")
    print("===========================================")
    sheet_stats: dict[str, dict] = {}
    for i, s in enumerate(sheets, start=1):
        df_s = pd.read_excel(master_xlsx, sheet_name=s)
        df_s = coerce_date_col(df_s)
        st = date_coverage_stats(df_s)
        sheet_stats[s] = st
        miss = st["missing_bdays_count"]
        miss_text = "N/A" if miss is None else str(miss)
        print(
            f"{i:2d}. {s} | days={st['unique_days']} | "
            f"range={st['date_min']} -> {st['date_max']} | missing_bdays={miss_text}"
        )
        if st["missing_bdays_preview"]:
            print(f"    missing preview: {', '.join(st['missing_bdays_preview'])}")

    all_idx = len(sheets) + 1
    print(f"{all_idx:2d}. ALL (evaluate all sheets)\n")

    sel = ask_choice(all_idx)
    if sel == all_idx:
        chosen = "ALL"
        chosen_sheets = sheets
    else:
        chosen = sheets[sel - 1]
        chosen_sheets = [chosen]

    if chosen == "ALL":
        max_available = max(sheet_stats[s]["unique_days"] for s in chosen_sheets)
        min_available = min(sheet_stats[s]["unique_days"] for s in chosen_sheets)
        print(
            f"\n[INFO] ALL mode: each sheet uses min(N, available days). "
            f"Available days range: {min_available} ~ {max_available}"
        )
    else:
        max_available = sheet_stats[chosen]["unique_days"]
        miss_cnt = sheet_stats[chosen]["missing_bdays_count"]
        miss_txt = "N/A" if miss_cnt is None else str(miss_cnt)
        print(
            f"\n[INFO] {chosen}: available days={max_available}, "
            f"date range={sheet_stats[chosen]['date_min']} -> {sheet_stats[chosen]['date_max']}, "
            f"missing_bdays={miss_txt}"
        )

    n_last = ask_window_n(max_days=max_available, default_days=30)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_name = f"eval_{ts}_N{n_last}_{sanitize_filename(chosen)}.xlsx"
    out_path = outdir / out_name

    all_summaries = []
    print("\n[RUN] N =", n_last, " SHEET =", chosen)
    print("[OUT]", out_path)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Summary will be written later after we collect all
        for sh in chosen_sheets:
            df_used, summary_df, _ = evaluate_one_sheet(master_xlsx, sh, n_last)
            all_summaries.append(summary_df)

            # write used window + full column set (for debugging)
            safe_sheet = sanitize_filename(sh)[:31] or "sheet"
            df_used.to_excel(writer, sheet_name=safe_sheet, index=False)

        summary_all = pd.concat(all_summaries, ignore_index=True)
        summary_all.to_excel(writer, sheet_name="Summary", index=False)

    # Print summary
    print("\n===========================================")
    print(" Summary")
    print("===========================================")
    with pd.option_context("display.width", 220, "display.max_columns", 200):
        print(pd.read_excel(out_path, sheet_name="Summary").to_string(index=False))

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
