#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from train_trend_after_open90_experiment import build_after_open90_table
import run_trend_modes_daily as daily


THIS_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = THIS_DIR.parent
RESULT_DIR = ARCHIVE_ROOT / "results"
EVAL_DIR = ARCHIVE_ROOT / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

PRED_FILE = RESULT_DIR / "trend_mode_daily_predictions.csv"


def _parse_eval_dates(pred: pd.DataFrame, window: str, n_days: int) -> pd.Series:
    dates = pd.to_datetime(pred["date"]).drop_duplicates().sort_values()
    if window == "all":
        return dates
    if n_days <= 0:
        raise ValueError("n_days must be > 0 for recent window.")
    return dates.tail(n_days)


def _load_realized_labels() -> pd.DataFrame:
    source = daily._resolve_prediction_intraday_source()
    real = build_after_open90_table(atr_window=10, atr_mult=0.8, spx5m_file=source).copy()
    real["date"] = pd.to_datetime(real["date"]).dt.normalize()
    return real[[
        "date",
        "real_trend_remaining_after_open90",
        "real_trend_close150_after_open90",
    ]]


def _label_to_real_col(label: str) -> str:
    if label == "real_trend_remaining_after_open90":
        return "real_trend_remaining_after_open90"
    if label == "real_trend_close150_after_open90":
        return "real_trend_close150_after_open90"
    raise ValueError(f"Unknown label: {label}")


def _evaluate_group(df: pd.DataFrame) -> dict:
    y_true = df["real_trend"].astype(int).values
    y_pred = df["trend_pred"].astype(int).values
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = 0.5 * (tpr + tnr)
    return {
        "samples": int(len(df)),
        "real_pos_ratio": float(np.mean(y_true)),
        "pred_pos_ratio": float(np.mean(y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(bacc),
        "trend_capture_correct_rate": float(p),  # precision
        "trend_capture_rate": float(r),          # recall
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=["all", "recent"], default="all", help="Evaluate all dates or recent n dates.")
    ap.add_argument("--n_days", type=int, default=30, help="Used when --window recent.")
    args = ap.parse_args()

    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Prediction file not found: {PRED_FILE}")

    pred = pd.read_csv(PRED_FILE)
    if pred.empty:
        raise ValueError("Prediction file is empty.")

    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    if "run_mode" not in pred.columns:
        pred["run_mode"] = "legacy"
    pred["run_mode"] = pred["run_mode"].fillna("legacy").astype(str)
    eval_dates = _parse_eval_dates(pred, args.window, args.n_days)
    pred_eval = pred[pred["date"].isin(eval_dates)].copy()
    if pred_eval.empty:
        raise ValueError("No predictions found for selected evaluation window.")

    real = _load_realized_labels()
    pred_eval["real_col"] = pred_eval["label"].apply(_label_to_real_col)

    merged_rows = []
    for _, r in pred_eval.iterrows():
        d = r["date"]
        real_row = real[real["date"] == d]
        if real_row.empty:
            continue
        rv = real_row.iloc[-1][r["real_col"]]
        if pd.isna(rv):
            continue
        rr = r.to_dict()
        rr["real_trend"] = int(rv)
        merged_rows.append(rr)

    merged = pd.DataFrame(merged_rows)
    if merged.empty:
        raise ValueError("No matched realized labels for selected window.")

    summaries = []
    for (run_mode, label, mode), g in merged.groupby(["run_mode", "label", "mode"]):
        m = _evaluate_group(g)
        m.update(
            {
                "label": label,
                "mode": mode,
                "run_mode": run_mode,
                "date_start": g["date"].min().date().isoformat(),
                "date_end": g["date"].max().date().isoformat(),
            }
        )
        summaries.append(m)
    out_summary = pd.DataFrame(summaries).sort_values(["run_mode", "label", "mode"]).reset_index(drop=True)

    # Also aggregate by run mode + signal mode across both labels.
    mode_rows = []
    for (run_mode, mode), g in merged.groupby(["run_mode", "mode"]):
        m = _evaluate_group(g)
        m.update(
            {
                "label": "ALL_LABELS",
                "mode": mode,
                "run_mode": run_mode,
                "date_start": g["date"].min().date().isoformat(),
                "date_end": g["date"].max().date().isoformat(),
            }
        )
        mode_rows.append(m)
    out_mode = pd.DataFrame(mode_rows)
    out_all = pd.concat([out_summary, out_mode], ignore_index=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    win_tag = "ALL" if args.window == "all" else f"RECENT_{args.n_days}"

    detail_path = EVAL_DIR / f"trend_mode_eval_detail_{win_tag}_{ts}.csv"
    summary_path = EVAL_DIR / f"trend_mode_eval_summary_{win_tag}_{ts}.csv"
    txt_path = EVAL_DIR / f"trend_mode_eval_report_{win_tag}_{ts}.txt"

    merged.to_csv(detail_path, index=False, encoding="utf-8-sig")
    out_all.to_csv(summary_path, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("Trend Mode Prediction Evaluation")
    lines.append(f"Window: {win_tag}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Prediction file: {PRED_FILE}")
    lines.append("")
    lines.append(out_all.to_string(index=False))
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("[SAVE]", detail_path)
    print("[SAVE]", summary_path)
    print("[SAVE]", txt_path)
    print(out_all.to_string(index=False))


if __name__ == "__main__":
    main()
