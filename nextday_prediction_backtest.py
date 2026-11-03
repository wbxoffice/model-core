#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest for similarity-based next-day prediction path.

For each source date in [start_date, end_date]:
1) Use source date as anchor window.
2) Match only historical windows strictly before source date.
3) Build weighted next-day predicted 30m return path.
4) Scale predicted path range to model expected next-day range.
5) Read actual next-day 30m return path.
6) Save per-target-date chart: predicted vs actual (% vs open).

Outputs:
- log/nextday_similarity_backtest/backtest_<start>_<end>.json
- log/nextday_similarity_backtest/plots/<target_date>_pred_vs_actual.png
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict_nextday_similarity_path import (
    _attach_calendar_next_dates,
    _build_window_vectors,
    _collect_candidates,
    _complete_day_path_returns,
    _expected_range_pct_from_model,
    _expected_session_bars,
    _is_early_close_day,
    _load_daily_features,
    _load_intraday_30m,
    _next_open_day_from_calendar,
    _pt_time_labels_for_target,
    _next_day_path_returns,
    _pattern_label,
    _softmax,
)


def _date_in_range(src_date: str, start_date: str, end_date: str) -> bool:
    d = pd.to_datetime(src_date, errors="coerce")
    s = pd.to_datetime(start_date, errors="coerce")
    e = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(d) or pd.isna(s) or pd.isna(e):
        return False
    return bool(s <= d <= e)


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--start_date", type=str, required=True, help="YYYY-MM-DD (source date)")
    ap.add_argument("--end_date", type=str, required=True, help="YYYY-MM-DD (source date)")
    ap.add_argument("--lookback_days", type=int, default=10)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_days", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    lookback = max(5, int(args.lookback_days))
    top_k = max(3, int(args.top_k))
    max_days = max(0, int(args.max_days))

    out_dir = root / "log" / "nextday_similarity_backtest"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_daily = _load_daily_features(root)
    windows, vecs = _build_window_vectors(df_daily, lookback=lookback)
    if windows.empty:
        raise ValueError("No valid daily windows.")
    windows = _attach_calendar_next_dates(windows, root)

    intra30 = _load_intraday_30m(root)

    run_records: List[dict] = []
    processed = 0

    for anchor_pos in range(len(windows)):
        source_date = str(windows.iloc[anchor_pos]["src_date"])
        if not _date_in_range(source_date, args.start_date, args.end_date):
            continue
        if max_days > 0 and processed >= max_days:
            break

        target_date = _next_open_day_from_calendar(root, pd.to_datetime(source_date)).date().isoformat()
        target_early_close = _is_early_close_day(root, target_date)
        target_session_bars = _expected_session_bars(root, target_date)

        candidates, match_mode = _collect_candidates(
            root=root,
            windows=windows,
            vecs=vecs,
            anchor_pos=anchor_pos,
            intra30=intra30,
            lookback=lookback,
            historical_only=True,
            target_early_close=target_early_close,
            top_k=top_k,
        )
        if len(candidates) < top_k:
            continue

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_k]
        sim_values = np.array([x[1] for x in top], dtype=float)
        conf = _softmax(sim_values, temp=0.08)

        min_len_pred = int(min(len(x[2]) for x in top))
        top_paths = [x[2][:min_len_pred] for x in top]
        pred_raw = np.zeros(min_len_pred, dtype=float)
        for w, p in zip(conf, top_paths):
            pred_raw += w * p

        actual = _complete_day_path_returns(intra30, root, target_date)
        if len(actual) != target_session_bars:
            continue

        expected_range_pct, range_meta = _expected_range_pct_from_model(root, target_date)
        raw_range = float(np.max(pred_raw) - np.min(pred_raw))
        scale = expected_range_pct / raw_range if raw_range > 1e-8 else 1.0
        pred_scaled = pred_raw * scale

        L = int(min(len(pred_scaled), len(actual)))
        pred_eval = pred_scaled[:L]
        act_eval = actual[:L]
        bars = np.arange(L)
        pt_labels = _pt_time_labels_for_target(intra30, target_date, L)

        mae_pct = float(np.mean(np.abs(pred_eval - act_eval)) * 100.0)
        range_actual_pct = float((np.max(act_eval) - np.min(act_eval)) * 100.0)
        range_pred_pct = float((np.max(pred_eval) - np.min(pred_eval)) * 100.0)
        close_pred_pct = float(pred_eval[-1] * 100.0)
        close_actual_pct = float(act_eval[-1] * 100.0)
        dir_hit = int(np.sign(close_pred_pct) == np.sign(close_actual_pct))

        # top labels
        label_scores: Dict[str, float] = {}
        top_rows = []
        for rank, (idx, sim, ret_path, _open_gap_ret) in enumerate(top, start=1):
            lbl = _pattern_label(ret_path[:min_len_pred])
            w = float(conf[rank - 1])
            label_scores[lbl] = label_scores.get(lbl, 0.0) + w
            top_rows.append(
                {
                    "rank": rank,
                    "similar_src_date": windows.iloc[idx]["src_date"],
                    "historical_next_date": windows.iloc[idx]["next_date"],
                    "similarity": round(float(sim), 6),
                    "confidence": round(w, 6),
                    "label": lbl,
                    "nextday_open_to_close_ret": round(float(ret_path[min_len_pred - 1]), 6),
                    "nextday_range_pct": round(float(np.max(ret_path[:min_len_pred]) - np.min(ret_path[:min_len_pred])), 6),
                }
            )

        # plot
        plot_path = plot_dir / f"{target_date}_pred_vs_actual.png"
        plt.figure(figsize=(12, 6))
        plt.plot(bars, pred_eval * 100.0, linewidth=2.0, label="Predicted (scaled, %)")
        plt.plot(bars, act_eval * 100.0, linewidth=2.0, label="Actual (% vs open)")
        plt.axhline(0.0, linestyle="--", linewidth=1.0, color="gray", label="Open = 0%")
        plt.title(f"Next-Day Backtest | src={source_date} -> target={target_date}")
        step = 2 if len(pt_labels) >= 10 else 1
        ticks = np.arange(0, len(pt_labels), step)
        plt.xticks(ticks=ticks, labels=[pt_labels[i] for i in ticks], rotation=35, ha="right")
        plt.xlabel("PT Time (America/Los_Angeles)")
        plt.ylabel("Return vs Open (%)")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        record = {
            "source_date": source_date,
            "target_date": target_date,
            "lookback_days": lookback,
            "top_k": top_k,
            "target_early_close": bool(target_early_close),
            "expected_session_bars": int(target_session_bars),
            "candidate_match_mode": match_mode,
            "mae_pct": round(mae_pct, 4),
            "close_pred_pct": round(close_pred_pct, 4),
            "close_actual_pct": round(close_actual_pct, 4),
            "direction_hit": dir_hit,
            "range_pred_pct": round(range_pred_pct, 4),
            "range_actual_pct": round(range_actual_pct, 4),
            "expected_range_pct_model": round(expected_range_pct * 100.0, 4),
            "scale_factor": round(_safe_float(scale, 1.0), 6),
            "model_range_components": {k: round(_safe_float(v, np.nan), 6) for k, v in range_meta.items()},
            "top5": top_rows,
            "label_confidence_sum": {k: round(float(v), 6) for k, v in sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)},
            "pred_path_pct": [round(float(x * 100.0), 6) for x in pred_eval],
            "actual_path_pct": [round(float(x * 100.0), 6) for x in act_eval],
            "plot_file": str(plot_path),
        }
        run_records.append(record)
        processed += 1

    out_json = out_dir / f"backtest_{args.start_date}_to_{args.end_date}.json"
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "lookback_days": lookback,
        "top_k": top_k,
        "max_days": max_days,
        "n_records": len(run_records),
        "records": run_records,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if run_records:
        mae = float(np.mean([r["mae_pct"] for r in run_records]))
        hit = float(np.mean([r["direction_hit"] for r in run_records]))
    else:
        mae = float("nan")
        hit = float("nan")

    print(f"[OK] records={len(run_records)}")
    print(f"[OK] avg_mae_pct={mae:.4f}, direction_hit_rate={hit:.4f}")
    print(f"[OK] saved json: {out_json}")
    print(f"[OK] saved plots dir: {plot_dir}")


if __name__ == "__main__":
    main()
