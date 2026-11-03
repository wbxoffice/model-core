#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train intraday trend models that predict *after open90 is known*.

Use case:
- At 08:00 (first 90 min finished), use:
  1) previous-day daily factors (VDTR v2.1 feature frame row whose target_date=today)
  2) realized open90 intraday features (06:30-08:00)
  to predict:
  - remaining session trend: 08:00-13:00
  - close150 trend: 10:30-13:00

Trend threshold:
- dynamic: 0.8 * ATR10_prev (ATR from daily TR ratio, shifted by 1 day)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys
import subprocess

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

THIS_DIR = Path(__file__).resolve().parent
ARCHIVE_ROOT = THIS_DIR.parent
PROJECT_ROOT = ARCHIVE_ROOT.parent
# Compatible with both old and renamed project folders.
for cand in [
    PROJECT_ROOT / "spx_vdtr_v2",
    PROJECT_ROOT / "SPX VDTR model v2.0",
]:
    if cand.exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

from config import BASE_DIR, DATA_DIR, MODEL_DIR, SPX_FILE, SKLEARN_VERSION, PYTHON_VERSION
from feature_engineering_v2_1 import build_feature_frame_v21


TRAIN_START_DATE = pd.Timestamp("2016-01-01")
DEFAULT_ATR_WINDOW = 10
DEFAULT_ATR_MULT = 0.8

OPEN90_START = "06:30"
OPEN90_END = "08:00"
REMAIN_START = "08:00"
REMAIN_END = "13:00"
CLOSE150_START = "10:30"
CLOSE150_END = "13:00"
DEFAULT_RETRAIN_30M_FILE = ARCHIVE_ROOT / "data" / "SP_SPX, 30_merged.csv"
MERGE_30M_SCRIPT = THIS_DIR / "merge_spx_30min_training_source.py"

# ====================== Feature Spec (standardized) ======================
EVENT_FLAGS = ["is_fomc", "is_fomc_window", "is_vix_expiry", "is_spy_opex_m"]

OPEN90_BASE_FEATURES = [
    "open90_range_ratio", "open90_ret", "open90_body_ratio", "open90_up_bar_ratio",
    "open90_volume_rel20", "open90_vwap_dev", "open90_close_pos_in_range",
    "open90_efficiency_ratio", "open90_bar_ret_std", "open90_bar_ret_mean",
    "open90_dir_switch_count", "open90_bar1_ret", "open90_bar2_ret", "open90_bar3_ret",
]

OPEN90_POSITION_FEATURES = [
    "open90_open_gap_prev_close", "open90_close_rel_prev_high", "open90_close_rel_prev_low",
    "open90_break_prev_high", "open90_break_prev_low", "open90_close_above_prev_close",
]

OPEN90_ATR_REL_FEATURES = [
    "open90_range_over_atr_threshold", "open90_ret_over_atr_threshold",
]

OPEN90_INTERACTION_FEATURES = [
    "open90_ret_x_is_fomc", "open90_range_x_is_fomc",
    "open90_ret_x_is_fomc_window", "open90_range_x_is_fomc_window",
    "open90_ret_x_is_vix_expiry", "open90_range_x_is_vix_expiry",
    "open90_ret_x_is_spy_opex_m", "open90_range_x_is_spy_opex_m",
    "open90_ret_x_vix_level", "open90_range_x_vixchg", "open90_ret_x_gexz",
]


def _ts(date_obj, hm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{pd.Timestamp(date_obj).date()} {hm}")


def _slice_window(df_day: pd.DataFrame, date_obj, start_hm: str, end_hm: str) -> pd.DataFrame:
    st = _ts(date_obj, start_hm)
    et = _ts(date_obj, end_hm)
    return df_day[(df_day["datetime"] >= st) & (df_day["datetime"] < et)].copy()


def _load_spx_daily() -> pd.DataFrame:
    d = pd.read_csv(SPX_FILE)
    if "time" in d.columns:
        d["date"] = pd.to_datetime(d["time"], format="ISO8601")
    elif "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"])
    else:
        raise ValueError("SPX daily file missing date/time")

    for c in ["open", "high", "low", "close"]:
        if c not in d.columns:
            raise ValueError(f"SPX daily file missing column: {c}")
    d = d[["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)
    return d


def _atr_threshold_table(atr_window: int, atr_mult: float) -> pd.DataFrame:
    d = _load_spx_daily()
    prev_close = d["close"].shift(1)
    tr = pd.concat(
        [
            (d["high"] - d["low"]).abs(),
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["tr_ratio"] = tr / d["open"].replace(0, np.nan)
    d["atr_prev"] = d["tr_ratio"].rolling(atr_window, min_periods=atr_window).mean().shift(1)
    d["trend_threshold"] = atr_mult * d["atr_prev"]
    return d[["date", "atr_prev", "trend_threshold"]]


def _discover_intraday_file() -> Path:
    """Discover intraday source. Priority: exported data > data/trend, and 5min > 30min."""
    root_trend = BASE_DIR.parent / "data" / "trend"
    root_exported = BASE_DIR.parent / "exported data"

    def _file_latest_date(p: Path) -> pd.Timestamp:
        try:
            df = pd.read_csv(p, usecols=lambda c: c.strip().lower() in {"time", "datetime", "date"})
            if df.empty:
                return pd.NaT
            col = None
            for c in df.columns:
                cc = c.strip().lower()
                if cc in {"time", "datetime", "date"}:
                    col = c
                    break
            if col is None:
                return pd.NaT
            dt = pd.to_datetime(df[col], format="ISO8601", errors="coerce", utc=True)
            if dt.notna().any():
                return dt.max().tz_convert("America/Los_Angeles").tz_localize(None)
            # fallback plain parse
            dt2 = pd.to_datetime(df[col], errors="coerce")
            return dt2.max()
        except Exception:
            return pd.NaT

    def _file_prefix_priority(path: Path) -> int:
        n = path.name
        if n.startswith("SP_DLY_SPX,"):
            return 4
        if n.startswith("SPCFD_SPX,"):
            return 3
        if n.startswith("SP_SPX,"):
            return 2
        if n.startswith("CBOE_DLY_SPX,"):
            return 1
        return 0

    def _latest_by_date_then_mtime(paths: list[Path]) -> Path | None:
        if not paths:
            return None
        scored = []
        for p in paths:
            scored.append((p, _file_prefix_priority(p), _file_latest_date(p), pd.Timestamp.fromtimestamp(p.stat().st_mtime)))
        # sort by latest data first; prefix priority only as tie-break
        scored.sort(
            key=lambda x: (
                pd.notna(x[2]),
                x[2] if pd.notna(x[2]) else pd.Timestamp.min,
                x[3],
                x[1],
            ),
            reverse=True,
        )
        return scored[0][0]

    five_patterns = [
        "SP_DLY_SPX, 5*.csv",
        "SPCFD_SPX, 5*.csv",
        "SP_DLY_SPX,5*.csv",
        "SPCFD_SPX,5*.csv",
        "SP_SPX, 5*.csv",
        "SP_SPX,5*.csv",
        "CBOE_DLY_SPX, 5*.csv",
        "CBOE_DLY_SPX,5*.csv",
        "*SP_DLY_SPX*5min*.csv",
        "*SPCFD_SPX*5min*.csv",
        "*SPCFD_SPX*5*.csv",
        "*SP_SPX*5min*.csv",
        "*CBOE_DLY_SPX*5*.csv",
        "SPX_5min*.csv",
        "*SPX*5min*.csv",
    ]
    thirty_patterns = [
        "SP_DLY_SPX, 30*.csv",
        "SPCFD_SPX, 30*.csv",
        "SP_DLY_SPX,30*.csv",
        "SPCFD_SPX,30*.csv",
        "SP_SPX, 30*.csv",
        "SP_SPX,30*.csv",
        "CBOE_DLY_SPX, 30*.csv",
        "CBOE_DLY_SPX,30*.csv",
        "*SP_DLY_SPX*30*.csv",
        "*SPCFD_SPX*30*.csv",
        "*SP_SPX*30*.csv",
        "*CBOE_DLY_SPX*30*.csv",
        "SPX_30min*.csv",
        "*SPX*30min*.csv",
    ]

    roots = [root_exported, root_trend]
    for root in roots:
        five = []
        thirty = []
        if root.exists():
            for pat in five_patterns:
                five.extend(root.glob(pat))
            for pat in thirty_patterns:
                thirty.extend(root.glob(pat))

        p5 = _latest_by_date_then_mtime(list(set(five)))
        if p5 is not None:
            return p5

        p30 = _latest_by_date_then_mtime(list(set(thirty)))
        if p30 is not None:
            return p30

    # Fallback for backward compatibility
    legacy = DATA_DIR / "SPX_5min.csv"
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        "No intraday SPX file found. Expected in exported data or data/trend (5min preferred, then 30min). "
        f"Checked: {root_exported} ; {root_trend}"
    )


def _run_30m_merge_for_retrain() -> None:
    if not MERGE_30M_SCRIPT.exists():
        raise FileNotFoundError(f"Missing merge script: {MERGE_30M_SCRIPT}")
    proc = subprocess.run(
        [sys.executable, str(MERGE_30M_SCRIPT)],
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(
            f"30min merge failed before retrain. script={MERGE_30M_SCRIPT}, returncode={proc.returncode}"
        )


def _resolve_retrain_intraday_source(spx5m_file: str | None, run_merge_if_default: bool = True) -> Path:
    """
    Retrain default source policy:
    - If --spx5m_file is explicitly provided, use it as-is.
    - Otherwise always use local 30min merged file in this project data folder.
      Before using it, run 30min merge script to refresh.
    """
    if spx5m_file:
        return Path(spx5m_file)

    if run_merge_if_default:
        _run_30m_merge_for_retrain()
    if not DEFAULT_RETRAIN_30M_FILE.exists():
        raise FileNotFoundError(
            f"Default retrain source missing: {DEFAULT_RETRAIN_30M_FILE}. "
            "Run merge_spx_30min_training_source.py first."
        )
    return DEFAULT_RETRAIN_30M_FILE


def _load_spx_intraday(path: Path | None = None, cleanup_files: bool = True) -> pd.DataFrame:
    p = path or _discover_intraday_file()
    if not p.exists():
        raise FileNotFoundError(f"SPX intraday file not found: {p}")
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]

    if "datetime" in df.columns:
        dt_raw = df["datetime"]
    elif "time" in df.columns:
        dt_raw = df["time"]
    elif "date" in df.columns:
        dt_raw = df["date"]
    else:
        raise ValueError("SPX intraday file missing datetime/time/date")

    # Parse timestamps safely:
    # - If string already has timezone suffix (Z / +hh:mm / -hh:mm), convert from UTC to PT.
    # - If no timezone suffix, treat as local exchange time directly.
    raw = dt_raw.astype(str)
    has_tz_mask = raw.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True, na=False)
    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    if has_tz_mask.any():
        dt_utc = pd.to_datetime(raw[has_tz_mask], format="ISO8601", utc=True, errors="coerce")
        dt.loc[has_tz_mask] = dt_utc.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)

    if (~has_tz_mask).any():
        dt.loc[~has_tz_mask] = pd.to_datetime(raw[~has_tz_mask], errors="coerce")

    if dt.isna().all():
        raise ValueError("Failed to parse intraday datetime column.")
    df["datetime"] = dt

    need = ["open", "high", "low", "close"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"SPX intraday file missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = np.nan

    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.normalize()
    if cleanup_files:
        _cleanup_older_same_prefix_files(p)
    return df


def _cleanup_older_same_prefix_files(used_file: Path) -> None:
    """
    Keep only the latest-date file within the same prefix family in the same folder.
    Prefix family:
      - 5min family: filenames containing 'SP_DLY_SPX'/'SPCFD_SPX'/'SP_SPX' and ('5' / '5min')
      - 30min family: filenames containing 'SP_DLY_SPX'/'SPCFD_SPX'/'SP_SPX' and ('30' / '30min')
    """
    parent = used_file.parent
    if not parent.exists():
        return
    trend_root = (BASE_DIR.parent / "data" / "trend").resolve()
    if parent.resolve() != trend_root:
        return

    name = used_file.name.lower()
    if not ("sp_dly_spx" in name or "spcfd_spx" in name or "sp_spx" in name or "cboe_dly_spx" in name):
        return
    if "5min" in name or ", 5" in name or ",5" in name:
        pats = [
            "SP_DLY_SPX, 5*.csv",
            "SPCFD_SPX, 5*.csv",
            "SP_DLY_SPX,5*.csv",
            "SPCFD_SPX,5*.csv",
            "SP_SPX, 5*.csv",
            "SP_SPX,5*.csv",
            "CBOE_DLY_SPX, 5*.csv",
            "CBOE_DLY_SPX,5*.csv",
            "*SP_DLY_SPX*5min*.csv",
            "*SPCFD_SPX*5min*.csv",
            "*SP_SPX*5min*.csv",
            "*CBOE_DLY_SPX*5min*.csv",
        ]
    elif "30min" in name or ", 30" in name or ",30" in name:
        pats = [
            "SP_DLY_SPX, 30*.csv",
            "SPCFD_SPX, 30*.csv",
            "SP_DLY_SPX,30*.csv",
            "SPCFD_SPX,30*.csv",
            "SP_SPX, 30*.csv",
            "SP_SPX,30*.csv",
            "CBOE_DLY_SPX, 30*.csv",
            "CBOE_DLY_SPX,30*.csv",
            "*SP_DLY_SPX*30*.csv",
            "*SPCFD_SPX*30*.csv",
            "*SP_SPX*30*.csv",
            "*CBOE_DLY_SPX*30*.csv",
        ]
    else:
        return

    cands = []
    for pat in pats:
        cands.extend(parent.glob(pat))
    cands = list(set([p for p in cands if p.resolve() != used_file.resolve()]))
    for old in cands:
        try:
            old.unlink()
        except Exception:
            pass


def _window_range_ratio(sub: pd.DataFrame) -> float:
    if sub.empty:
        return np.nan
    p0 = float(sub.iloc[0]["open"])
    if p0 == 0 or pd.isna(p0):
        return np.nan
    return (float(sub["high"].max()) - float(sub["low"].min())) / p0


def _window_features(sub: pd.DataFrame, prefix: str) -> dict:
    if sub.empty:
        return {}
    o = float(sub.iloc[0]["open"])
    c = float(sub.iloc[-1]["close"])
    h = float(sub["high"].max())
    l = float(sub["low"].min())
    v = float(sub["volume"].sum()) if "volume" in sub.columns else np.nan
    if o == 0 or pd.isna(o):
        return {}

    tp = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    vv = sub["volume"].fillna(0.0)
    vwap = np.nan
    if float(vv.sum()) > 0:
        vwap = float((tp * vv).sum() / vv.sum())

    bar_rets = (sub["close"] / sub["open"] - 1.0).astype(float).values
    bar_sign = np.sign(bar_rets)
    nz = bar_sign[bar_sign != 0]
    dir_switch = int(np.sum(nz[1:] != nz[:-1])) if len(nz) >= 2 else 0

    out = {
        f"{prefix}_open": o,
        f"{prefix}_close": c,
        f"{prefix}_high": h,
        f"{prefix}_low": l,
        f"{prefix}_range_ratio": (h - l) / o,
        f"{prefix}_ret": c / o - 1.0,
        f"{prefix}_body_ratio": abs(c - o) / o,
        f"{prefix}_up_bar_ratio": float((sub["close"] > sub["open"]).mean()),
        f"{prefix}_volume": v,
        f"{prefix}_vwap_dev": (c - vwap) / o if not pd.isna(vwap) else np.nan,
        f"{prefix}_close_pos_in_range": (c - l) / (h - l) if (h - l) > 0 else np.nan,
        f"{prefix}_efficiency_ratio": (abs(c - o) / o) / (((h - l) / o) if o else np.nan),
        f"{prefix}_bar_ret_std": float(np.nanstd(bar_rets)),
        f"{prefix}_bar_ret_mean": float(np.nanmean(bar_rets)),
        f"{prefix}_dir_switch_count": dir_switch,
    }
    for i in range(3):
        out[f"{prefix}_bar{i+1}_ret"] = float(bar_rets[i]) if i < len(bar_rets) else np.nan
    return out


def build_open90_feature_table(
    atr_window: int,
    atr_mult: float,
    spx5m_file: Path | None = None,
    require_future_windows: bool = True,
    cleanup_files: bool = True,
) -> pd.DataFrame:
    atr = _atr_threshold_table(atr_window=atr_window, atr_mult=atr_mult)
    m5 = _load_spx_intraday(spx5m_file, cleanup_files=cleanup_files)

    rows = []
    for d, day in m5.groupby("date"):
        day = day.sort_values("datetime").reset_index(drop=True)
        s_open90 = _slice_window(day, d, OPEN90_START, OPEN90_END)
        s_remain = _slice_window(day, d, REMAIN_START, REMAIN_END)
        s_close150 = _slice_window(day, d, CLOSE150_START, CLOSE150_END)
        if s_open90.empty:
            continue
        if require_future_windows and (s_remain.empty or s_close150.empty):
            continue

        feat = _window_features(s_open90, "open90")
        if not feat:
            continue

        rows.append(
            {
                "date": pd.to_datetime(d),
                **feat,
                "remain_range_ratio": _window_range_ratio(s_remain) if not s_remain.empty else np.nan,
                "close150_range_ratio": _window_range_ratio(s_close150) if not s_close150.empty else np.nan,
            }
        )

    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.merge(atr, on="date", how="left")

    # Reference to prior day levels (known before open)
    dref = _load_spx_daily().copy()
    dref["prev_high"] = dref["high"].shift(1)
    dref["prev_low"] = dref["low"].shift(1)
    dref["prev_close"] = dref["close"].shift(1)
    dref = dref[["date", "prev_high", "prev_low", "prev_close"]]
    t = t.merge(dref, on="date", how="left")
    t["open90_open_gap_prev_close"] = t["open90_open"] / t["prev_close"].replace(0, np.nan) - 1.0
    t["open90_close_rel_prev_high"] = (t["open90_close"] - t["prev_high"]) / t["open90_open"].replace(0, np.nan)
    t["open90_close_rel_prev_low"] = (t["open90_close"] - t["prev_low"]) / t["open90_open"].replace(0, np.nan)
    t["open90_break_prev_high"] = (t["open90_high"] >= t["prev_high"]).astype(float)
    t["open90_break_prev_low"] = (t["open90_low"] <= t["prev_low"]).astype(float)
    t["open90_close_above_prev_close"] = (t["open90_close"] > t["prev_close"]).astype(float)
    t["real_trend_remaining_after_open90"] = (
        t["remain_range_ratio"] > t["trend_threshold"]
    ).astype(float)
    t["real_trend_close150_after_open90"] = (
        t["close150_range_ratio"] > t["trend_threshold"]
    ).astype(float)
    bad = t["trend_threshold"].isna() | t["remain_range_ratio"].isna() | t["close150_range_ratio"].isna()
    t.loc[bad, "real_trend_remaining_after_open90"] = np.nan
    t.loc[bad, "real_trend_close150_after_open90"] = np.nan

    # Open90 volume normalized by trailing 20-day median (known by 08:00)
    t = t.sort_values("date").reset_index(drop=True)
    med20 = t["open90_volume"].rolling(20, min_periods=5).median().shift(1)
    t["open90_volume_rel20"] = t["open90_volume"] / med20.replace(0, np.nan)
    t["open90_range_over_atr_threshold"] = t["open90_range_ratio"] / t["trend_threshold"].replace(0, np.nan)
    t["open90_ret_over_atr_threshold"] = t["open90_ret"].abs() / t["trend_threshold"].replace(0, np.nan)
    return t


def build_after_open90_table(
    atr_window: int,
    atr_mult: float,
    spx5m_file: Path | None = None,
    cleanup_files: bool = True,
) -> pd.DataFrame:
    return build_open90_feature_table(
        atr_window=atr_window,
        atr_mult=atr_mult,
        spx5m_file=spx5m_file,
        require_future_windows=True,
        cleanup_files=cleanup_files,
    )


def _daily_feature_cols(df: pd.DataFrame) -> list[str]:
    lag_cols = [c for c in df.columns if any(c.endswith(f"_lag{i}") for i in range(5))]
    event_cols = [
        "is_fomc",
        "is_fomc_window",
        "days_to_next_fomc",
        "days_from_prev_fomc",
        "is_vix_expiry",
        "days_to_vix_expiry",
        "is_spy_opex_m",
        "days_to_spy_opex_m",
    ]
    out = lag_cols[:]
    for c in ["dow", "moy"]:
        if c in df.columns:
            out.append(c)
    out.extend([c for c in event_cols if c in df.columns])
    return out


def _fit_rf(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X, y)
    return clf


def _save_bundle(path: Path, model, feature_cols, label_name, as_of, atr_window, atr_mult):
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "label_name": label_name,
        "model_name": path.stem,
        "train_strategy": "after_open90_experiment_expanding",
        "train_start_date": TRAIN_START_DATE,
        "train_end_date": as_of,
        "atr_window": atr_window,
        "atr_multiplier": atr_mult,
        "sklearn_version": SKLEARN_VERSION,
        "python_version": PYTHON_VERSION,
        "version": "v2.0-after-open90-exp-v1",
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def _temporal_holdout_metrics(X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> dict:
    n = len(X)
    split = max(int(n * 0.8), 1)
    Xtr, ytr = X.iloc[:split], y.iloc[:split]
    Xte, yte = X.iloc[split:], y.iloc[split:]
    dte = dates.iloc[split:]

    clf = _fit_rf(Xtr, ytr)
    pred = clf.predict(Xte)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    pred_pos_ratio = float((pred == 1).mean()) if len(pred) else np.nan
    pred_pos_count = int((pred == 1).sum()) if len(pred) else 0
    return {
        "test_start": dte.min().date().isoformat() if len(dte) else "",
        "test_end": dte.max().date().isoformat() if len(dte) else "",
        "test_n": int(len(Xte)),
        "test_pos_ratio": float(yte.mean()) if len(yte) else np.nan,
        "test_pred_trend_count": pred_pos_count,
        "test_pred_trend_ratio": pred_pos_ratio,
        "test_acc": float(accuracy_score(yte, pred)) if len(yte) else np.nan,
        "test_bacc": float(balanced_accuracy_score(yte, pred)) if len(yte) else np.nan,
        "test_precision": float(p),
        "test_trend_capture_correct_rate": float(p),
        "test_recall": float(r),
        "test_trend_capture_rate": float(r),
        "test_f1": float(f1),
    }


def _add_open90_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # Event/state interactions (all available by 08:00)
    for ev in EVENT_FLAGS:
        if ev in df.columns:
            df[f"open90_ret_x_{ev}"] = df["open90_ret"] * df[ev]
            df[f"open90_range_x_{ev}"] = df["open90_range_ratio"] * df[ev]
    if "vix_level_lag0" in df.columns:
        df["open90_ret_x_vix_level"] = df["open90_ret"] * df["vix_level_lag0"]
    if "vix_5d_chg_z60_lag0" in df.columns:
        df["open90_range_x_vixchg"] = df["open90_range_ratio"] * df["vix_5d_chg_z60_lag0"]
    if "gex_scaled_z120_lag0" in df.columns:
        df["open90_ret_x_gexz"] = df["open90_ret"] * df["gex_scaled_z120_lag0"]
    return df


def _build_feature_cols(df: pd.DataFrame) -> list[str]:
    base_daily_cols = _daily_feature_cols(df)
    open90_cols = (
        OPEN90_BASE_FEATURES
        + OPEN90_POSITION_FEATURES
        + OPEN90_ATR_REL_FEATURES
        + OPEN90_INTERACTION_FEATURES
    )
    return base_daily_cols + [c for c in open90_cols if c in df.columns]


def get_feature_spec() -> dict[str, list[str]]:
    return {
        "event_flags": EVENT_FLAGS,
        "open90_base_features": OPEN90_BASE_FEATURES,
        "open90_position_features": OPEN90_POSITION_FEATURES,
        "open90_atr_relative_features": OPEN90_ATR_REL_FEATURES,
        "open90_interaction_features": OPEN90_INTERACTION_FEATURES,
    }


def build_after_open90_training_frame(
    atr_window: int,
    atr_mult: float,
    as_of_date: str | None = None,
    spx5m_file: str | None = None,
    cleanup_files: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Timestamp]:
    intraday_tbl = build_after_open90_table(
        atr_window=atr_window,
        atr_mult=atr_mult,
        spx5m_file=Path(spx5m_file) if spx5m_file else None,
        cleanup_files=cleanup_files,
    )
    if intraday_tbl.empty:
        raise ValueError("No after-open90 samples built.")

    daily = build_feature_frame_v21(for_training=True).copy()
    daily["target_date"] = pd.to_datetime(daily["target_date"]).dt.normalize()
    intraday_tbl["date"] = pd.to_datetime(intraday_tbl["date"]).dt.normalize()
    df = daily.merge(intraday_tbl, left_on="target_date", right_on="date", how="left")
    df = _add_open90_interaction_features(df)

    if as_of_date:
        as_of = pd.to_datetime(as_of_date).normalize()
    else:
        as_of = df["target_date"].max()

    feature_cols = _build_feature_cols(df)
    return df, intraday_tbl, feature_cols, as_of


def main(as_of_date: str | None, atr_window: int, atr_mult: float, spx5m_file: str | None):
    src = _resolve_retrain_intraday_source(spx5m_file, run_merge_if_default=True)
    df, intraday_tbl, feature_cols, as_of = build_after_open90_training_frame(
        atr_window=atr_window,
        atr_mult=atr_mult,
        as_of_date=as_of_date,
        spx5m_file=str(src),
    )

    mask = (df["target_date"] >= TRAIN_START_DATE) & (df["target_date"] <= as_of)
    d1 = df.loc[mask & df["real_trend_remaining_after_open90"].notna()].copy().sort_values("target_date")
    d2 = df.loc[mask & df["real_trend_close150_after_open90"].notna()].copy().sort_values("target_date")
    if d1.empty or d2.empty:
        raise ValueError("Training set empty for remaining/close150 labels.")

    X1 = d1[feature_cols].fillna(0)
    y1 = d1["real_trend_remaining_after_open90"].astype(int)
    X2 = d2[feature_cols].fillna(0)
    y2 = d2["real_trend_close150_after_open90"].astype(int)

    m1 = _fit_rf(X1, y1)
    m2 = _fit_rf(X2, y2)

    model_dir = ARCHIVE_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = ARCHIVE_ROOT / "results"
    feat_dir.mkdir(parents=True, exist_ok=True)
    atr_tag = str(atr_mult).replace(".", "")

    out1 = model_dir / f"spx_trend_remaining_after_open90_model_exp_atr{atr_tag}_v1.pkl"
    out2 = model_dir / f"spx_trend_close150_after_open90_model_exp_atr{atr_tag}_v1.pkl"
    _save_bundle(out1, m1, feature_cols, "real_trend_remaining_after_open90", as_of, atr_window, atr_mult)
    _save_bundle(out2, m2, feature_cols, "real_trend_close150_after_open90", as_of, atr_window, atr_mult)

    # diagnostics
    imp1 = pd.Series(m1.feature_importances_, index=feature_cols).sort_values(ascending=False)
    imp2 = pd.Series(m2.feature_importances_, index=feature_cols).sort_values(ascending=False)
    imp = pd.concat(
        [
            pd.DataFrame({"model": out1.stem, "feature": imp1.index, "importance": imp1.values}),
            pd.DataFrame({"model": out2.stem, "feature": imp2.index, "importance": imp2.values}),
        ],
        ignore_index=True,
    )
    imp_path = feat_dir / f"trend_after_open90_feature_importance_atr{atr_tag}.csv"
    imp.to_csv(imp_path, index=False, encoding="utf-8-sig")

    m1_hold = _temporal_holdout_metrics(X1, y1, d1["target_date"])
    m2_hold = _temporal_holdout_metrics(X2, y2, d2["target_date"])
    summary = pd.DataFrame(
        [
            {
                "model": out1.stem,
                "label": "real_trend_remaining_after_open90",
                "as_of_date": as_of.date().isoformat(),
                "samples": int(len(d1)),
                "positive_ratio": float(y1.mean()),
                "atr_window": atr_window,
                "atr_multiplier": atr_mult,
                **m1_hold,
            },
            {
                "model": out2.stem,
                "label": "real_trend_close150_after_open90",
                "as_of_date": as_of.date().isoformat(),
                "samples": int(len(d2)),
                "positive_ratio": float(y2.mean()),
                "atr_window": atr_window,
                "atr_multiplier": atr_mult,
                **m2_hold,
            },
        ]
    )
    sum_path = feat_dir / "trend_after_open90_experiment_summary.csv"
    summary.to_csv(sum_path, index=False, encoding="utf-8-sig")

    # latest available day prediction at/after 08:00
    latest_rem = d1.sort_values("target_date").iloc[-1]
    latest_clo = d2.sort_values("target_date").iloc[-1]
    x_rem = latest_rem[feature_cols].to_frame().T.fillna(0).infer_objects(copy=False)
    x_clo = latest_clo[feature_cols].to_frame().T.fillna(0).infer_objects(copy=False)
    p_rem = float(m1.predict_proba(x_rem)[0][1])
    p_clo = float(m2.predict_proba(x_clo)[0][1])
    pred_latest = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(latest_rem["target_date"]).date().isoformat(),
                "model": out1.stem,
                "label": "real_trend_remaining_after_open90",
                "p_trend": p_rem,
                "trend_pred": int(p_rem >= 0.5),
                "trend_capture_rate_test": float(m1_hold["test_trend_capture_rate"]),
            },
            {
                "date": pd.to_datetime(latest_clo["target_date"]).date().isoformat(),
                "model": out2.stem,
                "label": "real_trend_close150_after_open90",
                "p_trend": p_clo,
                "trend_pred": int(p_clo >= 0.5),
                "trend_capture_rate_test": float(m2_hold["test_trend_capture_rate"]),
            },
        ]
    )
    pred_path = feat_dir / "trend_after_open90_latest_prediction.csv"
    pred_latest.to_csv(pred_path, index=False, encoding="utf-8-sig")

    diag_cols = [
        "date",
        "atr_prev",
        "trend_threshold",
        "open90_range_ratio",
        "open90_ret",
        "open90_body_ratio",
        "open90_up_bar_ratio",
        "open90_volume",
        "open90_volume_rel20",
        "open90_vwap_dev",
        "remain_range_ratio",
        "close150_range_ratio",
        "real_trend_remaining_after_open90",
        "real_trend_close150_after_open90",
    ]
    diag_path = feat_dir / f"trend_after_open90_label_diagnostics_atr{atr_tag}.csv"
    intraday_tbl[diag_cols].to_csv(diag_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {out1}")
    print(f"[SAVE] {out2}")
    print(f"[SAVE] {sum_path}")
    print(f"[SAVE] {imp_path}")
    print(f"[SAVE] {diag_path}")
    print(f"[SAVE] {pred_path}")
    print("[INFO] remaining samples=", len(d1), "pos_ratio=", round(float(y1.mean()), 4))
    print("[INFO] close150 samples=", len(d2), "pos_ratio=", round(float(y2.mean()), 4))


def _safe_iso_ts(ts: pd.Timestamp | float | None) -> str:
    if ts is None or pd.isna(ts):
        return "NA"
    return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")


def _print_precheck(as_of_date: str | None, atr_window: int, atr_mult: float, spx5m_file: str | None):
    src = _resolve_retrain_intraday_source(spx5m_file, run_merge_if_default=True)
    m5 = _load_spx_intraday(src, cleanup_files=False)

    days_open90 = 0
    days_remain = 0
    days_close150 = 0
    full_window_days = 0
    for d, day in m5.groupby("date"):
        day = day.sort_values("datetime").reset_index(drop=True)
        s_open90 = _slice_window(day, d, OPEN90_START, OPEN90_END)
        s_remain = _slice_window(day, d, REMAIN_START, REMAIN_END)
        s_close150 = _slice_window(day, d, CLOSE150_START, CLOSE150_END)
        has_open90 = not s_open90.empty
        has_remain = not s_remain.empty
        has_close150 = not s_close150.empty
        days_open90 += int(has_open90)
        days_remain += int(has_remain)
        days_close150 += int(has_close150)
        full_window_days += int(has_open90 and has_remain and has_close150)

    df, intraday_tbl, feature_cols, as_of = build_after_open90_training_frame(
        atr_window=atr_window,
        atr_mult=atr_mult,
        as_of_date=as_of_date,
        spx5m_file=str(src),
        cleanup_files=False,
    )

    mask = (df["target_date"] >= TRAIN_START_DATE) & (df["target_date"] <= as_of)
    d1 = df.loc[mask & df["real_trend_remaining_after_open90"].notna()].copy().sort_values("target_date")
    d2 = df.loc[mask & df["real_trend_close150_after_open90"].notna()].copy().sort_values("target_date")

    y1 = d1["real_trend_remaining_after_open90"].astype(int) if not d1.empty else pd.Series(dtype=int)
    y2 = d2["real_trend_close150_after_open90"].astype(int) if not d2.empty else pd.Series(dtype=int)

    print("[PRECHECK] intraday_source:", src)
    print(
        "[PRECHECK] intraday_rows=",
        len(m5),
        "date_range=",
        _safe_iso_ts(m5["datetime"].min()),
        "to",
        _safe_iso_ts(m5["datetime"].max()),
        "unique_days=",
        int(m5["date"].nunique()),
    )
    print(
        "[PRECHECK] days_with_open90=",
        days_open90,
        "days_with_remain=",
        days_remain,
        "days_with_close150=",
        days_close150,
        "full_window_days=",
        full_window_days,
    )
    print(
        "[PRECHECK] intraday_table_rows=",
        len(intraday_tbl),
        "intraday_date_range=",
        _safe_iso_ts(intraday_tbl["date"].min() if not intraday_tbl.empty else None),
        "to",
        _safe_iso_ts(intraday_tbl["date"].max() if not intraday_tbl.empty else None),
    )
    print(
        "[PRECHECK] train_mask_range=",
        TRAIN_START_DATE.date().isoformat(),
        "to",
        pd.to_datetime(as_of).date().isoformat(),
        "feature_count=",
        len(feature_cols),
    )
    print(
        "[PRECHECK] remaining_samples=",
        len(d1),
        "positive_ratio=",
        round(float(y1.mean()), 4) if len(y1) else np.nan,
        "classes=",
        sorted(y1.unique().tolist()) if len(y1) else [],
    )
    print(
        "[PRECHECK] close150_samples=",
        len(d2),
        "positive_ratio=",
        round(float(y2.mean()), 4) if len(y2) else np.nan,
        "classes=",
        sorted(y2.unique().tolist()) if len(y2) else [],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as_of_date", type=str, default=None)
    parser.add_argument("--atr_window", type=int, default=DEFAULT_ATR_WINDOW)
    parser.add_argument("--atr_mult", type=float, default=DEFAULT_ATR_MULT)
    parser.add_argument(
        "--spx5m_file",
        type=str,
        default=None,
        help="Optional custom intraday file path. Default retrain source is local data/SP_SPX, 30_merged.csv (auto-merged before run).",
    )
    parser.add_argument("--precheck", action="store_true", help="Only print source and sample stats; do not retrain")
    args = parser.parse_args()
    if args.precheck:
        _print_precheck(args.as_of_date, args.atr_window, args.atr_mult, args.spx5m_file)
    else:
        main(args.as_of_date, args.atr_window, args.atr_mult, args.spx5m_file)
