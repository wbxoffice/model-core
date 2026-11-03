#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import pickle
import subprocess
import sys
import re
import json
import math

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

THIS_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = THIS_DIR.parent
PROJECT_ROOT = ARCHIVE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Compatible with both old and renamed project folders.
for cand in [
    PROJECT_ROOT / "spx_vdtr_v2",
    PROJECT_ROOT / "SPX VDTR model v2.0",
]:
    if cand.exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from feature_engineering_v2_1 import build_feature_frame_v21
from train_trend_after_open90_experiment import (
    build_open90_feature_table,
    _add_open90_interaction_features,
    _discover_intraday_file,
)
from nyse_calendar_utils import (
    ensure_nyse_open_days_csv,
    load_nyse_open_days,
    align_series_to_next_open_day,
)
from market_time_utils import current_nyse_timestamp, current_nyse_trade_date


MODEL_DIR = ARCHIVE_ROOT / "models"
RESULT_DIR = ARCHIVE_ROOT / "results"
LOG_DIR = ARCHIVE_ROOT / "log"
PREPARED_5M_INPUT = ARCHIVE_ROOT / "data" / "SPX_5min_prediction_input.csv"
MERGE_5M_SCRIPT = THIS_DIR / "merge_spx_5min_sources_incremental.py"
for d in [RESULT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


LABEL_REMAIN = "real_trend_remaining_after_open90"
LABEL_CLOSE150 = "real_trend_close150_after_open90"
PRIMARY_MODE = "balanced"  # default: recall-priority

DEFAULT_THRESHOLDS = {
    LABEL_REMAIN: {"balanced": 0.27, "precision": 0.64},
    LABEL_CLOSE150: {"balanced": 0.24, "precision": 0.50},
}


def _is_30min_source(path: Path) -> bool:
    name = path.name.lower()
    return (", 30" in name) or (",30" in name) or ("_30" in name)


def _load_intraday_timestamps(path: Path) -> pd.Series:
    df = pd.read_csv(
        path,
        usecols=lambda c: c.strip().lower() in {"time", "datetime", "date"},
    )
    if df.empty or len(df.columns) == 0:
        return pd.Series(dtype="datetime64[ns]")
    col = df.columns[0]
    raw = df[col].astype(str)
    aware = raw.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True, na=False)
    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if aware.any():
        utc = pd.to_datetime(raw[aware], format="ISO8601", errors="coerce", utc=True)
        parsed.loc[aware] = utc.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    if (~aware).any():
        parsed.loc[~aware] = pd.to_datetime(raw[~aware], errors="coerce")
    return parsed.dropna().sort_values().reset_index(drop=True)


def _validate_open90_completeness(path: Path, trade_date: pd.Timestamp) -> dict:
    """Validate exact open90 timestamp coverage and describe the run stage."""
    timestamps = _load_intraday_timestamps(path)
    trade_date = pd.Timestamp(trade_date).normalize()
    day = timestamps[timestamps.dt.normalize() == trade_date].copy()
    freq_minutes = 30 if _is_30min_source(path) else 5
    expected_count = 3 if freq_minutes == 30 else 18
    start = trade_date + pd.Timedelta(hours=6, minutes=30)
    expected = pd.date_range(start=start, periods=expected_count, freq=f"{freq_minutes}min")
    window = day[(day >= start) & (day < trade_date + pd.Timedelta(hours=8))]
    actual_unique = pd.DatetimeIndex(window.drop_duplicates().sort_values())
    missing = expected.difference(actual_unique)
    extra = actual_unique.difference(expected)
    duplicate_count = int(window.duplicated().sum())
    has_trade_date_data = not day.empty
    complete = (
        has_trade_date_data
        and len(actual_unique) == expected_count
        and len(missing) == 0
        and len(extra) == 0
        and duplicate_count == 0
    )
    day_max = day.max() if has_trade_date_data else pd.NaT
    normal_close_last_bar = trade_date + (
        pd.Timedelta(hours=13) if freq_minutes == 30 else pd.Timedelta(hours=12, minutes=55)
    )
    inferred_run_mode = (
        "postclose"
        if has_trade_date_data and pd.notna(day_max) and day_max >= normal_close_last_bar
        else "intraday"
    )
    status = {
        "trade_date": trade_date,
        "frequency_minutes": freq_minutes,
        "expected_bars": expected_count,
        "actual_bars": int(len(actual_unique)),
        "duplicate_bars": duplicate_count,
        "missing_times": [x.strftime("%H:%M") for x in missing],
        "extra_times": [x.strftime("%H:%M") for x in extra],
        "has_trade_date_data": has_trade_date_data,
        "complete": complete,
        "source_max_timestamp_pt": day_max,
        "inferred_run_mode": inferred_run_mode,
    }
    if not complete:
        raise ValueError(
            "Incomplete open90 data: "
            f"trade_date={trade_date.date().isoformat()} freq={freq_minutes}min "
            f"expected={expected_count} actual={len(actual_unique)} duplicates={duplicate_count} "
            f"missing={status['missing_times']} extra={status['extra_times']} source={path}"
        )
    return status


def _select_prediction_trade_date(path: Path, run_ts_ny: pd.Timestamp) -> tuple[pd.Timestamp, str]:
    """Choose today's session, or the latest completed source day during NY pre-open."""
    timestamps = _load_intraday_timestamps(path)
    today = current_nyse_trade_date(run_ts_ny)
    today_rows = timestamps[timestamps.dt.normalize() == today]
    if not today_rows.empty:
        return today, "current_source_date"

    run_ts_ny = pd.Timestamp(run_ts_ny)
    if run_ts_ny.tzinfo is None:
        run_ts_ny = run_ts_ny.tz_localize("America/New_York")
    else:
        run_ts_ny = run_ts_ny.tz_convert("America/New_York")
    before_ny_open = (run_ts_ny.hour, run_ts_ny.minute) < (9, 30)
    prior_rows = timestamps[timestamps.dt.normalize() < today]
    if before_ny_open and not prior_rows.empty:
        return prior_rows.dt.normalize().max(), "ny_preopen_latest_source_date"
    return today, "current_date_required"


def _resolve_run_mode(requested: str, open90_status: dict) -> str:
    if requested not in {"auto", "intraday", "postclose"}:
        raise ValueError(f"Unknown run mode: {requested}")
    if requested == "auto":
        return str(open90_status["inferred_run_mode"])
    return requested


def _resolve_prediction_intraday_source(refresh_5m: bool = False) -> Path:
    discovered = _discover_intraday_file()
    if _is_30min_source(discovered):
        raise ValueError(
            "Daily trend prediction requires a 5min intraday source; "
            f"30min source is not accepted: {discovered}"
        )

    if refresh_5m:
        cp = subprocess.run([sys.executable, str(MERGE_5M_SCRIPT)], cwd=str(ARCHIVE_ROOT), check=False)
        if cp.returncode != 0:
            raise RuntimeError(f"5min prediction input refresh failed: rc={cp.returncode}")
    if not PREPARED_5M_INPUT.exists():
        raise FileNotFoundError(
            f"Prepared 5min prediction input missing: {PREPARED_5M_INPUT}. "
            "Run run_trend_modes_daily.bat so the merge step can create it."
        )
    return PREPARED_5M_INPUT


def _extract_trade_date_and_run_dt(log_file: Path) -> tuple[pd.Timestamp | None, pd.Timestamp, str]:
    txt = log_file.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Prediction date:\s*(\d{4}-\d{2}-\d{2})", txt)
    if not m:
        m = re.search(r"No open90 data for today \((\d{4}-\d{2}-\d{2})\)", txt)
    trade_date = pd.to_datetime(m.group(1)).normalize() if m else None

    m2 = re.search(r"run_trend_modes_(\d{8}_\d{6})\.txt$", log_file.name)
    if m2:
        run_dt = pd.to_datetime(m2.group(1), format="%Y%m%d_%H%M%S")
    else:
        run_dt = pd.Timestamp.fromtimestamp(log_file.stat().st_mtime)
    m3 = re.search(r"Run mode:\s*([A-Za-z0-9_-]+)", txt)
    run_mode = m3.group(1) if m3 else "legacy"
    return trade_date, run_dt, run_mode


def _append_archive_jsonl(rows: list[dict], archive_jsonl: Path):
    if not rows:
        return
    archive_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(archive_jsonl, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _archive_one_txt_to_json(src: Path, reason: str) -> dict | None:
    trade_date, run_dt, run_mode = _extract_trade_date_and_run_dt(src)
    if trade_date is None:
        return None
    content = src.read_text(encoding="utf-8", errors="replace")
    row = {
        "archived_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_date": trade_date.date().isoformat(),
        "run_ts": run_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "run_mode": run_mode,
        "source_file": src.name,
        "reason": reason,
        "content": content,
    }
    try:
        src.unlink()
    except Exception:
        return None
    return row


def _migrate_archive_txt_to_jsonl(archive_dir: Path):
    old_txts = sorted(archive_dir.glob("run_trend_modes_*.txt"))
    if not old_txts:
        return
    jsonl = archive_dir / "trend_mode_log_archive.jsonl"
    rows_jsonl: list[dict] = []
    for p in old_txts:
        rec = _archive_one_txt_to_json(p, "migrate_archive_txt_to_jsonl")
        if rec is None:
            continue
        rows_jsonl.append(rec)
    _append_archive_jsonl(rows_jsonl, jsonl)


def _maintain_daily_logs(log_dir: Path, keep_trade_days: int = 10):
    files = sorted(log_dir.glob("run_trend_modes_*.txt"))
    if len(files) <= 1:
        return

    records = []
    for f in files:
        trade_date, run_dt, run_mode = _extract_trade_date_and_run_dt(f)
        if trade_date is None:
            continue
        records.append({"file": f, "trade_date": trade_date, "run_dt": run_dt, "run_mode": run_mode})
    if not records:
        return

    # Keep the latest log for each trade_date + run_mode pair.
    latest_by_trade_mode: dict[tuple[pd.Timestamp, str], dict] = {}
    for r in records:
        key = (r["trade_date"], r["run_mode"])
        if key not in latest_by_trade_mode or r["run_dt"] > latest_by_trade_mode[key]["run_dt"]:
            latest_by_trade_mode[key] = r

    keep_dates = sorted({r["trade_date"] for r in latest_by_trade_mode.values()}, reverse=True)[:keep_trade_days]
    keep_files = {
        r["file"] for r in latest_by_trade_mode.values() if r["trade_date"] in keep_dates
    }

    archive_dir = log_dir / "archive"
    _migrate_archive_txt_to_jsonl(archive_dir)
    archive_jsonl = archive_dir / "trend_mode_log_archive.jsonl"
    archive_rows_jsonl: list[dict] = []

    for r in records:
        f = r["file"]
        if f in keep_files:
            continue
        if r["trade_date"] in keep_dates:
            reason = "duplicate_same_trade_date_and_mode_keep_latest"
        else:
            reason = "older_than_recent_10_trade_days"
        rec = _archive_one_txt_to_json(f, reason)
        if rec is None:
            continue
        archive_rows_jsonl.append(rec)

    _append_archive_jsonl(archive_rows_jsonl, archive_jsonl)


def _load_model_bundles() -> dict[str, dict]:
    bundles = {}
    for p in MODEL_DIR.glob("spx_trend_*_after_open90_model_exp_atr08_v1.pkl"):
        try:
            with open(p, "rb") as f:
                head = f.read(16)
        except OSError as exc:
            raise OSError(f"Cannot read model file: {p} ({exc})") from exc
        if not head and p.stat().st_size > 0:
            raise RuntimeError(
                "Model file appears to be a macOS/OneDrive dataless placeholder "
                f"(metadata size={p.stat().st_size}, readable bytes=0): {p}. "
                "Right-click the file/folder in Finder and choose 'Download Now' / "
                "'Always Keep on This Device', or retrain the trend models."
            )
        with open(p, "rb") as f:
            try:
                b = pickle.load(f)
            except EOFError as exc:
                raise RuntimeError(
                    f"Model file is unreadable or incomplete: {p}. "
                    "If `ls -lO` shows 'dataless', force OneDrive to download it or retrain."
                ) from exc
        label = b.get("label_name", "")
        if label:
            bundles[label] = b
    missing = [lb for lb in [LABEL_REMAIN, LABEL_CLOSE150] if lb not in bundles]
    if missing:
        raise FileNotFoundError(f"Missing model bundles for labels: {missing}. Please train models first.")
    return bundles


def _load_thresholds() -> dict[str, dict]:
    th = {k: v.copy() for k, v in DEFAULT_THRESHOLDS.items()}
    constrained_labels: set[str] = set()

    # Priority 1: constraint-optimized recommendations (recall mode)
    p_constraint = ARCHIVE_ROOT / "baseline" / "constraint_recommendation_latest.csv"
    if p_constraint.exists():
        try:
            c = pd.read_csv(p_constraint)
            for lb, setup in [
                (LABEL_REMAIN, "open90_recall"),
                (LABEL_CLOSE150, "close150_recall"),
            ]:
                sub = c[(c.get("setup") == setup) & (c.get("status") == "ok")]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                if pd.notna(r.get("threshold", np.nan)):
                    th[lb]["balanced"] = float(r["threshold"])
                    constrained_labels.add(lb)
        except Exception:
            pass

    # Priority 2: legacy optimization summary (fallback for precision / missing values)
    p = RESULT_DIR / "trend_after_open90_optimization_summary.csv"
    if not p.exists():
        return th
    try:
        s = pd.read_csv(p)
        for lb in [LABEL_REMAIN, LABEL_CLOSE150]:
            sub = s[s["label"] == lb]
            if sub.empty:
                continue
            r = sub.iloc[0]
            if lb not in constrained_labels and pd.notna(r.get("recommended_threshold_balanced", np.nan)):
                th[lb]["balanced"] = float(r["recommended_threshold_balanced"])
            if pd.notna(r.get("recommended_threshold_precision", np.nan)):
                th[lb]["precision"] = float(r["recommended_threshold_precision"])
    except Exception:
        pass
    return th


def _build_prediction_frame(spx_file: Path | None = None) -> pd.DataFrame:
    intraday = build_open90_feature_table(
        atr_window=10,
        atr_mult=0.8,
        spx5m_file=spx_file,
        require_future_windows=False,
    )
    if intraday.empty:
        raise ValueError("No intraday open90 features available.")

    daily = build_feature_frame_v21(for_training=False).copy()
    # Align with daily-run behavior: if the latest row has no next-day target_date yet,
    # use next business day as provisional target_date so open90(today) can still map.
    if "target_date" in daily.columns and "date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily["target_date"] = pd.to_datetime(daily["target_date"], errors="coerce")
        mask = daily["target_date"].isna() & daily["date"].notna()
        if mask.any():
            daily.loc[mask, "target_date"] = daily.loc[mask, "date"] + BDay(1)
        # NYSE calendar alignment: weekend-only BDay is not enough (e.g. Good Friday).
        nyse_cal_file = PROJECT_ROOT / "data" / "nyse_open_days.csv"
        ensure_nyse_open_days_csv(nyse_cal_file)
        nyse_open_days = load_nyse_open_days(nyse_cal_file)
        daily["target_date"] = align_series_to_next_open_day(daily["target_date"], nyse_open_days)

    daily["target_date"] = pd.to_datetime(daily["target_date"]).dt.normalize()
    intraday["date"] = pd.to_datetime(intraday["date"]).dt.normalize()
    df = daily.merge(intraday, left_on="target_date", right_on="date", how="inner")
    df = _add_open90_interaction_features(df)

    return df.sort_values("target_date").reset_index(drop=True)


def _predict_for_latest_date(df: pd.DataFrame, bundles: dict[str, dict], thresholds: dict[str, dict]) -> pd.DataFrame:
    pred_date = pd.to_datetime(df["target_date"]).max()
    row = df[df["target_date"] == pred_date].iloc[-1]

    outputs = []
    for label, bundle in bundles.items():
        model = bundle["model"]
        feats = list(bundle["feature_cols"])

        x = row.to_frame().T.copy()
        for c in feats:
            if c not in x.columns:
                x[c] = 0.0
        x = x[feats].apply(pd.to_numeric, errors="coerce").fillna(0)

        classes = list(model.classes_)
        pos_idx = classes.index(1) if 1 in classes else 0
        p_trend = float(model.predict_proba(x)[:, pos_idx][0])

        for mode in ["balanced", "precision"]:
            th = float(thresholds[label][mode])
            outputs.append(
                {
                    "date": pd.to_datetime(pred_date).date().isoformat(),
                    "label": label,
                    "mode": mode,
                    "threshold": th,
                    "p_trend": p_trend,
                    "trend_pred": int(p_trend >= th),
                    "model_name": bundle.get("model_name", ""),
                }
            )
    return pd.DataFrame(outputs)


def _attach_yesterday_feedback(new_rows: pd.DataFrame, df_all: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    out = new_rows.copy()
    out["yesterday_result"] = np.nan
    out["yesterday_is_correct"] = np.nan
    if out.empty:
        return out

    pred_date = pd.to_datetime(out["date"]).dt.normalize().max()
    all_dates = pd.to_datetime(df_all["target_date"]).dropna().dt.normalize().sort_values().unique()
    prev_dates = [d for d in all_dates if d < pred_date]
    if not prev_dates:
        return out
    prev_date = pd.Timestamp(prev_dates[-1]).normalize()

    prev_df = df_all[pd.to_datetime(df_all["target_date"]).dt.normalize() == prev_date]
    if prev_df.empty:
        return out
    prev_row = prev_df.iloc[-1]

    actual_by_label: dict[str, float] = {}
    for lb in [LABEL_REMAIN, LABEL_CLOSE150]:
        v = prev_row.get(lb, np.nan)
        if pd.notna(v):
            actual_by_label[lb] = float(v)

    pred_by_key: dict[tuple[str, str, str], float] = {}
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        if {"date", "label", "mode", "trend_pred"}.issubset(old.columns):
            old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.normalize()
            if "run_mode" not in old.columns:
                old["run_mode"] = "legacy"
            old["run_mode"] = old["run_mode"].fillna("legacy").astype(str)
            sub = old[old["date"] == prev_date]
            for _, r in sub.iterrows():
                pred_by_key[(str(r["label"]), str(r["mode"]), str(r["run_mode"]))] = float(r["trend_pred"])

    ys = []
    yc = []
    for _, r in out.iterrows():
        lb = str(r["label"])
        md = str(r["mode"])
        run_mode = str(r.get("run_mode", "legacy"))
        actual = actual_by_label.get(lb, np.nan)
        pred_prev = pred_by_key.get((lb, md, run_mode), np.nan)
        if pd.isna(pred_prev):
            for fallback_mode in ["intraday", "postclose", "legacy"]:
                pred_prev = pred_by_key.get((lb, md, fallback_mode), np.nan)
                if not pd.isna(pred_prev):
                    break
        if pd.isna(actual):
            ys.append(np.nan)
            yc.append(np.nan)
            continue
        ys.append(int(actual))
        if pd.isna(pred_prev):
            yc.append(np.nan)
        else:
            yc.append(int(int(pred_prev) == int(actual)))
    out["yesterday_result"] = ys
    out["yesterday_is_correct"] = yc
    return out


def _attach_historical_metrics(new_rows: pd.DataFrame, df_all: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    out = new_rows.copy()
    out["hist_precision"] = np.nan
    out["hist_recall"] = np.nan
    if out.empty:
        return out
    scan = _load_walkforward_scan()
    if scan.empty:
        return out

    hp, hr = [], []
    for _, r in out.iterrows():
        lb = str(r["label"])
        th = round(float(r["threshold"]), 2)
        sub = scan[scan["label"].astype(str) == lb].copy()
        if sub.empty:
            hp.append(np.nan)
            hr.append(np.nan)
            continue
        sub["threshold"] = pd.to_numeric(sub["threshold"], errors="coerce").round(2)
        row = sub[sub["threshold"] == th]
        if row.empty:
            # fallback to nearest threshold on scan grid
            idx = (sub["threshold"] - th).abs().idxmin()
            rr = sub.loc[idx]
        else:
            rr = row.iloc[0]
        hp.append(pd.to_numeric(rr.get("precision", np.nan), errors="coerce"))
        hr.append(pd.to_numeric(rr.get("recall", np.nan), errors="coerce"))

    out["hist_precision"] = hp
    out["hist_recall"] = hr
    return out


def _load_walkforward_scan() -> pd.DataFrame:
    # Preferred: explicit 30m scan snapshot (stable, threshold-grid ready)
    candidates = [
        RESULT_DIR / "trend_after_open90_walkforward_30m_atr_augmented_threshold_scan_30m.csv",
        RESULT_DIR / "trend_after_open90_walkforward_30m_atr_augmented_threshold_scan.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                s = pd.read_csv(p)
                needed = {"label", "threshold", "precision", "recall"}
                if needed.issubset(set(s.columns)):
                    s["threshold"] = pd.to_numeric(s["threshold"], errors="coerce").round(2)
                    return s.dropna(subset=["label", "threshold"]).copy()
            except Exception:
                pass

    # Fallback: build scan from probability history
    p_pred = RESULT_DIR / "trend_after_open90_walkforward_boosting_compare_predictions.csv"
    if not p_pred.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(p_pred)
    except Exception:
        return pd.DataFrame()
    need = {"label", "p_trend", "real_trend"}
    if not need.issubset(set(d.columns)):
        return pd.DataFrame()

    rows = []
    thresholds = np.round(np.arange(0.20, 0.80 + 1e-12, 0.01), 2)
    for lb, sub in d.groupby("label"):
        y = pd.to_numeric(sub["real_trend"], errors="coerce")
        p = pd.to_numeric(sub["p_trend"], errors="coerce")
        m = y.notna() & p.notna()
        y = y[m].astype(int).values
        p = p[m].astype(float).values
        if len(y) == 0:
            continue
        for th in thresholds:
            yhat = (p >= th).astype(int)
            tp = int(((yhat == 1) & (y == 1)).sum())
            fp = int(((yhat == 1) & (y == 0)).sum())
            fn = int(((yhat == 0) & (y == 1)).sum())
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else np.nan
            rows.append(
                {
                    "label": lb,
                    "threshold": float(round(th, 2)),
                    "precision": prec,
                    "recall": rec,
                }
            )
    return pd.DataFrame(rows)


def _attach_walkforward_metrics(new_rows: pd.DataFrame) -> pd.DataFrame:
    out = new_rows.copy()
    out["wf_ref_threshold"] = np.nan
    out["wf_precision"] = np.nan
    out["wf_recall"] = np.nan
    if out.empty:
        return out

    scan = _load_walkforward_scan()
    if scan.empty:
        return out

    # Thresholds are at 0.01 grid; choose highest threshold <= p_trend (still predicts 1).
    for i, r in out.iterrows():
        lb = str(r["label"])
        p = pd.to_numeric(r.get("p_trend", np.nan), errors="coerce")
        if pd.isna(p):
            continue
        sub = scan[scan["label"].astype(str) == lb].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("threshold")
        ths = pd.to_numeric(sub["threshold"], errors="coerce").dropna().values
        if len(ths) == 0:
            continue

        # "closest threshold that still keeps pred=1"
        p_floor = math.floor(float(p) * 100.0) / 100.0
        elig = ths[ths <= p_floor + 1e-12]
        if len(elig) > 0:
            ref_th = float(np.max(elig))
        else:
            # If p is below min scanned threshold, use nearest available for reference.
            ref_th = float(np.min(ths))

        row = sub[np.isclose(sub["threshold"].astype(float), ref_th)]
        if row.empty:
            row = sub.iloc[[int(np.argmin(np.abs(sub["threshold"].astype(float).values - ref_th)))]]
        rr = row.iloc[0]
        out.at[i, "wf_ref_threshold"] = float(ref_th)
        out.at[i, "wf_precision"] = pd.to_numeric(rr.get("precision", np.nan), errors="coerce")
        out.at[i, "wf_recall"] = pd.to_numeric(rr.get("recall", np.nan), errors="coerce")

    return out


def _append_results(new_rows: pd.DataFrame, out_csv: Path):
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        if "run_mode" not in old.columns:
            old["run_mode"] = "legacy"
        old["run_mode"] = old["run_mode"].fillna("legacy").astype(str)
        for c in new_rows.columns:
            if c not in old.columns:
                old[c] = np.nan
        for c in old.columns:
            if c not in new_rows.columns:
                new_rows[c] = np.nan
        new_rows = new_rows[old.columns]

        # Allow metadata backfill for existing immutable prediction rows.
        update_cols = [
            c
            for c in [
                "yesterday_result",
                "yesterday_is_correct",
                "hist_precision",
                "hist_recall",
                "wf_ref_threshold",
                "wf_precision",
                "wf_recall",
            ]
            if c in old.columns
        ]
        if update_cols:
            key_cols = ["date", "label", "mode", "run_mode"]
            upd = new_rows[key_cols + update_cols].copy()
            merged_old = old.merge(upd, on=key_cols, how="left", suffixes=("", "__new"))
            for c in update_cols:
                nc = f"{c}__new"
                if nc in merged_old.columns:
                    old[c] = merged_old[nc].combine_first(merged_old[c])

        keys = ["date", "label", "mode", "run_mode"]
        # Allow overwrite only for today's keys; historical rows remain immutable.
        today_str = current_nyse_trade_date().date().isoformat()
        if "date" in new_rows.columns:
            new_rows["date"] = pd.to_datetime(new_rows["date"], errors="coerce").dt.date.astype(str)
        if "date" in old.columns:
            old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.date.astype(str)
        today_rows = new_rows[new_rows["date"] == today_str].copy() if "date" in new_rows.columns else pd.DataFrame()
        if not today_rows.empty:
            old_with_flag = old.merge(today_rows[keys], on=keys, how="left", indicator=True)
            old = old_with_flag.loc[old_with_flag["_merge"] == "left_only", old.columns].copy()

        old_keys = old[keys].copy()
        new_keys = new_rows[keys].copy()
        # Append-only for history; today's rows can be replaced by latest run.
        merged = new_keys.merge(old_keys, on=keys, how="left", indicator=True)
        to_add = new_rows.loc[merged["_merge"] == "left_only"].copy()
        all_rows = pd.concat([old, to_add], ignore_index=True)
    else:
        all_rows = new_rows.copy()
    all_rows = all_rows.sort_values(["date", "run_mode", "label", "mode"]).reset_index(drop=True)
    all_rows.to_csv(out_csv, index=False, encoding="utf-8-sig")


def _append_primary_rows(new_rows: pd.DataFrame, out_csv: Path):
    pri = new_rows[new_rows["mode"].astype(str) == PRIMARY_MODE].copy().reset_index(drop=True)
    if pri.empty:
        return

    # Keep schema stable if historical file exists.
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        if "run_mode" not in old.columns:
            old["run_mode"] = "legacy"
        old["run_mode"] = old["run_mode"].fillna("legacy").astype(str)
        for c in pri.columns:
            if c not in old.columns:
                old[c] = np.nan
        for c in old.columns:
            if c not in pri.columns:
                pri[c] = np.nan
        pri = pri[old.columns]

        keys = ["date", "label", "mode", "run_mode"]
        today_str = current_nyse_trade_date().date().isoformat()
        if "date" in pri.columns:
            pri["date"] = pd.to_datetime(pri["date"], errors="coerce").dt.date.astype(str)
        if "date" in old.columns:
            old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.date.astype(str)

        # Today can be replaced; historical rows remain immutable.
        pri_today = pri[pri["date"] == today_str].copy() if "date" in pri.columns else pd.DataFrame()
        if not pri_today.empty:
            old_with_flag = old.merge(pri_today[keys], on=keys, how="left", indicator=True)
            old = old_with_flag.loc[old_with_flag["_merge"] == "left_only", old.columns].copy()

        old_keys = old[keys].copy().reset_index(drop=True)
        new_keys = pri[keys].copy().reset_index(drop=True)
        merged = new_keys.merge(old_keys, on=keys, how="left", indicator=True)
        to_add = pri.loc[merged["_merge"].to_numpy() == "left_only"].copy()
        all_rows = pd.concat([old, to_add], ignore_index=True)
    else:
        all_rows = pri.copy()

    all_rows = all_rows.sort_values(["date", "run_mode", "label"]).reset_index(drop=True)
    all_rows.to_csv(out_csv, index=False, encoding="utf-8-sig")


def _write_log(rows: pd.DataFrame, log_path: Path, source_file: Path | None = None):
    lines = []
    lines.append("Trend Daily Mode Prediction")
    lines.append(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Primary mode: {PRIMARY_MODE} (recall-priority)")
    lines.append("Reference mode: precision (ExtraTrees reference)")
    if not rows.empty:
        first = rows.iloc[0]
        lines.append(f"Run mode: {first.get('run_mode', 'legacy')}")
        lines.append(f"Run timestamp NY: {first.get('run_timestamp_ny', 'NA')}")
        lines.append(f"Run timestamp UTC: {first.get('run_timestamp_utc', 'NA')}")
        lines.append(
            "Open90 completeness: "
            f"{first.get('open90_actual_bars', 'NA')}/{first.get('open90_expected_bars', 'NA')} "
            f"complete={first.get('open90_complete', 'NA')}"
        )
        lines.append(f"Source max timestamp PT: {first.get('source_max_timestamp_pt', 'NA')}")
    if source_file is not None:
        lines.append(f"Intraday source: {source_file}")
    if rows.empty:
        lines.append("No prediction rows generated.")
    else:
        d = rows.iloc[0]["date"]
        lines.append(f"Prediction date: {d}")
        for _, r in rows.iterrows():
            yres = r["yesterday_result"] if "yesterday_result" in rows.columns else np.nan
            yok = r["yesterday_is_correct"] if "yesterday_is_correct" in rows.columns else np.nan
            yres_txt = "NA" if pd.isna(yres) else str(int(yres))
            yok_txt = "NA" if pd.isna(yok) else str(int(yok))
            hp = r["hist_precision"] if "hist_precision" in rows.columns else np.nan
            hr = r["hist_recall"] if "hist_recall" in rows.columns else np.nan
            hp_txt = "NA" if pd.isna(hp) else f"{float(hp):.4f}"
            hr_txt = "NA" if pd.isna(hr) else f"{float(hr):.4f}"
            wf_th = r["wf_ref_threshold"] if "wf_ref_threshold" in rows.columns else np.nan
            wf_p = r["wf_precision"] if "wf_precision" in rows.columns else np.nan
            wf_r = r["wf_recall"] if "wf_recall" in rows.columns else np.nan
            wf_th_txt = "NA" if pd.isna(wf_th) else f"{float(wf_th):.2f}"
            wf_p_txt = "NA" if pd.isna(wf_p) else f"{float(wf_p):.4f}"
            wf_r_txt = "NA" if pd.isna(wf_r) else f"{float(wf_r):.4f}"
            lines.append(f"[{r['label']}] mode={r['mode']} th={r['threshold']:.2f} p_trend={r['p_trend']:.4f} pred={int(r['trend_pred'])}")
            lines.append(f"  yesterday_result={yres_txt} yesterday_is_correct={yok_txt}")
            lines.append(f"  hist_precision={hp_txt} hist_recall={hr_txt}")
            lines.append(f"  wf_ref_th={wf_th_txt} wf_precision={wf_p_txt} wf_recall={wf_r_txt}")
            lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")


def main(run_mode_arg: str = "auto", skip_merge_refresh: bool = False):
    run_ts_ny = current_nyse_timestamp()
    run_ts_utc = run_ts_ny.tz_convert("UTC")
    bundles = _load_model_bundles()
    thresholds = _load_thresholds()
    src_file = _resolve_prediction_intraday_source(refresh_5m=not skip_merge_refresh)
    target_date, target_date_reason = _select_prediction_trade_date(src_file, run_ts_ny)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_trend_modes_{ts}.txt"
    open90_status = _validate_open90_completeness(src_file, target_date)

    if (
        target_date_reason == "ny_preopen_latest_source_date"
        and open90_status["inferred_run_mode"] != "postclose"
        and run_mode_arg != "postclose"
    ):
        raise ValueError(
            "NY pre-open fallback requires a completed prior session through the close; "
            f"source_max={open90_status['source_max_timestamp_pt']} target_date={target_date.date()} "
            "(use --run-mode postclose only for a verified early-close session)."
        )

    run_mode = _resolve_run_mode(run_mode_arg, open90_status)
    frame = _build_prediction_frame(spx_file=src_file)
    frame_today = frame[frame["target_date"] == target_date].copy()
    if frame_today.empty:
        raise RuntimeError(
            f"Open90 is complete for {target_date.date().isoformat()}, but no matching daily feature row was built."
        )

    out_csv = RESULT_DIR / "trend_mode_daily_predictions.csv"
    pred_rows = _predict_for_latest_date(frame_today, bundles, thresholds)
    source_max = open90_status["source_max_timestamp_pt"]
    pred_rows["run_mode"] = run_mode
    pred_rows["run_timestamp_ny"] = run_ts_ny.isoformat()
    pred_rows["run_timestamp_utc"] = run_ts_utc.isoformat()
    pred_rows["source_file"] = str(src_file)
    pred_rows["target_date_reason"] = target_date_reason
    pred_rows["source_max_timestamp_pt"] = (
        source_max.isoformat() if pd.notna(source_max) else ""
    )
    pred_rows["open90_frequency_minutes"] = int(open90_status["frequency_minutes"])
    pred_rows["open90_expected_bars"] = int(open90_status["expected_bars"])
    pred_rows["open90_actual_bars"] = int(open90_status["actual_bars"])
    pred_rows["open90_complete"] = bool(open90_status["complete"])
    pred_rows = _attach_yesterday_feedback(pred_rows, frame, out_csv)
    pred_rows = _attach_historical_metrics(pred_rows, frame, out_csv)
    pred_rows = _attach_walkforward_metrics(pred_rows)
    _append_results(pred_rows, out_csv)
    primary_csv = RESULT_DIR / "trend_mode_daily_primary.csv"
    _append_primary_rows(pred_rows, primary_csv)

    _write_log(pred_rows, log_path, source_file=src_file)
    _maintain_daily_logs(LOG_DIR, keep_trade_days=10)

    print(f"[INFO] Intraday source: {src_file}")
    print(f"[INFO] Prediction date: {target_date.date()}; reason={target_date_reason}")
    print(
        f"[INFO] Run mode: {run_mode}; open90="
        f"{open90_status['actual_bars']}/{open90_status['expected_bars']} complete"
    )
    print("[SAVE]", out_csv)
    print("[SAVE]", primary_csv)
    print("[SAVE]", log_path)
    print(pred_rows.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-mode",
        choices=["auto", "intraday", "postclose"],
        default="auto",
        help="auto infers from source session coverage; override when needed (for example early-close days).",
    )
    parser.add_argument(
        "--skip-merge-refresh",
        action="store_true",
        help="Use only when the launcher already refreshed the prepared 5min prediction input.",
    )
    args = parser.parse_args()
    main(run_mode_arg=args.run_mode, skip_merge_refresh=args.skip_merge_refresh)
