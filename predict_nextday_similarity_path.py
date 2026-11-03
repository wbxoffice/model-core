#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Similarity-based next-day path projection.

Goal:
1) Use latest 10-day feature window (today and previous 9 trading days).
2) Find historical windows with highest similarity.
3) Take each matched day's NEXT DAY 30min path as scenario samples.
4) Output top-5 scenarios + label confidence.
5) Build weighted synthetic next-day path and stretch its realized range to
   model-expected next-day range amplitude.

Outputs (ROOT/log/nextday_similarity):
- top5_similar_days_YYYY-MM-DD.csv
- weighted_path_YYYY-MM-DD.csv
- weighted_path_YYYY-MM-DD.png
- summary_YYYY-MM-DD.txt
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_dt_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


_NYSE_OPEN_DAYS_CACHE: Dict[str, pd.Series] = {}


def _load_nyse_open_days(root: Path) -> pd.Series:
    key = str(root)
    if key in _NYSE_OPEN_DAYS_CACHE:
        return _NYSE_OPEN_DAYS_CACHE[key]
    s = pd.Series([], dtype="datetime64[ns]")
    cal_path = root / "data" / "nyse_open_days.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        col = "date" if "date" in cal.columns else ("time" if "time" in cal.columns else None)
        if col is not None:
            s = (
                pd.to_datetime(cal[col], errors="coerce")
                .dropna()
                .dt.normalize()
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True)
            )
    _NYSE_OPEN_DAYS_CACHE[key] = s
    return s


def _next_open_day_from_calendar(root: Path, d: pd.Timestamp) -> pd.Timestamp:
    s = _load_nyse_open_days(root)
    if s.empty:
        return d + pd.offsets.BDay(1)
    d_norm = pd.to_datetime(d).normalize()
    later = s[s > d_norm]
    if later.empty:
        return d + pd.offsets.BDay(1)
    return later.iloc[0]


# Scheduled NYSE early closes (13:00 ET = 10:00 PT -> 8 bars of 30m):
# Jul 3 (when it is a trading day), the Friday after Thanksgiving,
# and Dec 24 (when it is a trading day). Rule validated against all
# 12 early-close sessions present in 2020-2025 30m data.
_EARLY_CLOSE_BARS = 8


def _is_early_close_day(root: Path, d) -> bool:
    ts = pd.to_datetime(d).normalize()
    matched = False
    if (ts.month, ts.day) in {(7, 3), (12, 24)}:
        matched = True
    elif ts.month == 11 and ts.weekday() == 4:
        first = ts.replace(day=1)
        first_thu = 1 + (3 - first.weekday()) % 7
        matched = ts.day == first_thu + 22  # Friday after the 4th Thursday
    if not matched:
        return False
    open_days = _load_nyse_open_days(root)
    if open_days.empty:
        return ts.weekday() < 5
    return bool((open_days == ts).any())


def _expected_session_bars(root: Path, d) -> int:
    return _EARLY_CLOSE_BARS if _is_early_close_day(root, d) else 14


def _parse_30m_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt_utc = pd.to_datetime(out.get("time"), errors="coerce", utc=True)
    dt_local = dt_utc.dt.tz_convert("America/Los_Angeles")
    out["dt"] = dt_local
    out["date"] = dt_local.dt.tz_localize(None).dt.normalize()
    return out


def _normalize_fib_percent_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Fibonacci-like percent column names to 2-decimal canonical form.
    Examples:
      38% -> 38.00%
      38.2% -> 38.20%
      -300.0% -> -300.00%
    """
    pat = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%\s*$")
    rename_map: Dict[str, str] = {}
    used_targets = set(df.columns)

    for c in df.columns:
        m = pat.match(str(c))
        if not m:
            continue
        val = float(m.group(1))
        target = f"{val:.2f}%"
        if c == target:
            continue
        # If target already exists, keep existing target as source of truth.
        if target in used_targets:
            continue
        rename_map[c] = target
        used_targets.add(target)

    if rename_map:
        return df.rename(columns=rename_map)
    return df


def _find_latest_exported_30m(root: Path) -> Path | None:
    ex_dir = root / "exported data"
    if not ex_dir.exists():
        return None
    prefixes = ("SP_DLY_SPX, 30", "SP_DLY_SPX,30", "SPCFD_SPX, 30", "SPCFD_SPX,30", "SP_SPX, 30", "SP_SPX,30", "CBOE_DLY_SPX, 30", "CBOE_DLY_SPX,30")
    files = [
        p
        for p in ex_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv" and p.name.startswith(prefixes)
    ]
    if not files:
        return None

    def _file_latest_dt(path: Path) -> pd.Timestamp:
        try:
            d = pd.read_csv(path, usecols=lambda c: c.strip().lower() in {"time", "datetime", "date"})
            if d.empty:
                return pd.NaT
            col = next((c for c in ["time", "datetime", "date"] if c in d.columns), None)
            if col is None:
                return pd.NaT
            s = pd.to_datetime(d[col], format="ISO8601", errors="coerce", utc=True)
            if s.notna().any():
                return s.max().tz_convert("America/Los_Angeles").tz_localize(None)
            return pd.to_datetime(d[col], errors="coerce").max()
        except Exception:
            return pd.NaT

    def _prefix_priority(path: Path) -> int:
        n = path.name
        if n.startswith(("SP_DLY_SPX, 30", "SP_DLY_SPX,30")):
            return 4
        if n.startswith(("SPCFD_SPX, 30", "SPCFD_SPX,30")):
            return 3
        if n.startswith(("SP_SPX, 30", "SP_SPX,30")):
            return 2
        if n.startswith(("CBOE_DLY_SPX, 30", "CBOE_DLY_SPX,30")):
            return 1
        return 0
    scored = []
    for p in files:
        latest_dt = _file_latest_dt(p)
        scored.append(
            (
                p,
                pd.notna(latest_dt),
                latest_dt if pd.notna(latest_dt) else pd.Timestamp.min,
                pd.Timestamp.fromtimestamp(p.stat().st_mtime),
                _prefix_priority(p),
            )
        )
    scored.sort(key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
    return scored[0][0]


# Legal session-end bar start times (PT): 13:00 = normal close,
# 10:00/11:00/12:00 = early-close templates (aligned with STATUS_SYNC v2026.04.06.8).
_VALID_LAST_BAR_TIMES = {"10:00", "11:00", "12:00", "13:00"}

# Minimum 30m bars for a historical candidate's next-day path: full sessions
# have 14 bars (06:30-13:00); 13 tolerates a missing closing bar but rejects
# early-close days so they cannot truncate the weighted path.
_MIN_CANDIDATE_BARS = 13

# v2: predicted path anchored at previous close with no modeled open gap.
# The similarity-matched gap was statistically uncorrelated with the actual
# gap (~0.4% noise vs ~0.75% typical daily range), so it only shifted levels.
_CHART_DEFINITION = "spx_price_prev_close_anchor_v2"


def _is_complete_30m_session(day_df: pd.DataFrame) -> bool:
    """A day is complete when bars start at 06:30 PT, are gapless at 30min,
    and the last bar starts at a legal close time (accepts early-close days)."""
    dts = pd.to_datetime(day_df["dt"]).sort_values()
    if dts.empty:
        return False
    if dts.iloc[0].strftime("%H:%M") != "06:30":
        return False
    diffs = dts.diff().dropna()
    if not (diffs == pd.Timedelta(minutes=30)).all():
        return False
    return dts.iloc[-1].strftime("%H:%M") in _VALID_LAST_BAR_TIMES


def _ingest_latest_exported_30m(root: Path) -> Dict[str, object]:
    """
    Merge latest exported 30m into data/SP_SPX, 30.csv with strict rules:
    - only append dates strictly after current max date
    - only append complete trading-day dates (early-close sessions accepted)
    - never overwrite existing dates
    - delete processed exported source file
    """
    master_path = root / "data" / "SP_SPX, 30.csv"
    src_path = _find_latest_exported_30m(root)
    result: Dict[str, object] = {
        "source_file": str(src_path) if src_path else "",
        "source_deleted": False,
        "appended_rows": 0,
        "appended_dates": [],
        "skipped_reason": "",
        "expected_bars_per_day": None,
    }
    if src_path is None:
        result["skipped_reason"] = "no_exported_30m_file"
        return result
    if not master_path.exists():
        result["skipped_reason"] = "master_30m_missing"
        return result

    master = pd.read_csv(master_path)
    src = pd.read_csv(src_path)
    src = _normalize_fib_percent_column_names(src)
    if "time" not in master.columns or "time" not in src.columns:
        result["skipped_reason"] = "time_column_missing"
        return result

    master = _parse_30m_time_columns(master)
    src = _parse_30m_time_columns(src)
    master = master.dropna(subset=["dt", "date"]).drop_duplicates(subset=["time"]).sort_values("dt")
    src = src.dropna(subset=["dt", "date"]).drop_duplicates(subset=["time"]).sort_values("dt")

    if master.empty or src.empty:
        # Keep the source file: an empty parse may be a format change, not bad data.
        result["skipped_reason"] = "empty_master_or_src"
        return result

    max_date = master["date"].max()
    src_future = src[src["date"] > max_date].copy()
    if src_future.empty:
        result["skipped_reason"] = "no_future_dates"
        # Keep source file for retry (e.g., when master file was locked and
        # replacement of same-date rows is needed later).
        result["source_deleted"] = False
        return result

    counts_master = master.groupby("date").size()
    expected = int(counts_master.mode().iloc[0]) if not counts_master.empty else 14
    result["expected_bars_per_day"] = expected

    complete_dates = sorted([d for d, g in src_future.groupby("date") if _is_complete_30m_session(g)])
    if not complete_dates:
        result["skipped_reason"] = "no_complete_future_day"
        return result

    add = src_future[src_future["date"].isin(complete_dates)].copy()
    # keep same master schema; unknown source columns are ignored
    master_cols = [c for c in master.columns if c not in ("dt", "date")]
    for c in master_cols:
        if c not in add.columns:
            add[c] = np.nan
    add = add[master_cols]

    out = pd.concat([master[master_cols], add], ignore_index=True)
    out = out.drop_duplicates(subset=["time"], keep="first")
    out = out.sort_values("time")
    out.to_csv(master_path, index=False)

    result["appended_rows"] = int(len(add))
    result["appended_dates"] = [pd.Timestamp(d).date().isoformat() for d in complete_dates]
    try:
        src_path.unlink()
        result["source_deleted"] = True
    except Exception:
        result["source_deleted"] = False
    return result


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _softmax(x: np.ndarray, temp: float = 0.08) -> np.ndarray:
    t = max(temp, 1e-6)
    z = x / t
    z = z - np.max(z)
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    return e / s


def _pattern_label(path_ret: np.ndarray) -> str:
    if len(path_ret) < 4:
        return "unknown"
    close_ret = float(path_ret[-1])
    half_ret = float(path_ret[len(path_ret) // 2])
    max_ret = float(np.max(path_ret))
    min_ret = float(np.min(path_ret))
    swing = max_ret - min_ret

    if swing < 0.0035:
        return "range"
    if close_ret >= 0.0025 and half_ret >= 0:
        return "trend_up"
    if close_ret <= -0.0025 and half_ret <= 0:
        return "trend_down"
    if half_ret > 0 and close_ret < 0:
        return "up_then_down"
    if half_ret < 0 and close_ret > 0:
        return "down_then_up"
    return "mixed"


def _load_daily_features(root: Path) -> pd.DataFrame:
    spx_path = root / "data" / "SP_SPX, 1D.csv"
    dix_path = root / "data" / "DIX.xlsx"
    vix_path = root / "data" / "CBOE_DLY_VIX.csv"

    spx = pd.read_csv(spx_path)
    dix = pd.read_excel(dix_path)
    vix = pd.read_csv(vix_path)

    if "time" in spx.columns:
        spx["date"] = _to_dt_col(spx, "time")
    elif "date" in spx.columns:
        spx["date"] = _to_dt_col(spx, "date")
    else:
        raise ValueError("SP_SPX, 1D.csv missing time/date column")

    dix["date"] = _to_dt_col(dix, "date")
    if "time" in vix.columns:
        vix["date"] = _to_dt_col(vix, "time")
    elif "date" in vix.columns:
        vix["date"] = _to_dt_col(vix, "date")
    else:
        raise ValueError("CBOE_DLY_VIX.csv missing time/date column")

    for df in (spx, dix, vix):
        df["date"] = df["date"].dt.normalize()

    spx = spx[["date", "open", "high", "low", "close"]].copy()
    dix_cols = [c for c in ["date", "dix", "gex"] if c in dix.columns]
    vix_cols = [c for c in ["date", "close", "vix"] if c in vix.columns]

    dix = dix[dix_cols].copy()
    if "close" in vix.columns and "vix" not in vix.columns:
        vix = vix.rename(columns={"close": "vix"})
    vix = vix[["date", "vix"]].copy()

    df = spx.merge(dix, on="date", how="inner").merge(vix, on="date", how="left")
    df = df.sort_values("date").dropna(subset=["open", "high", "low", "close", "dix", "gex"])

    # daily features used in similarity window
    df["ret"] = df["close"].pct_change()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_ret"] = (df["close"] - df["open"]) / df["open"]
    df["gap_ret"] = df["open"] / df["close"].shift(1) - 1.0
    df["dix_chg"] = df["dix"].pct_change()
    df["gex_chg"] = df["gex"].pct_change()
    df["vix_chg"] = df["vix"].pct_change()

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna().reset_index(drop=True)


def _build_window_vectors(df: pd.DataFrame, lookback: int) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    feat_cols = ["ret", "hl_range", "oc_ret", "gap_ret", "dix", "dix_chg", "gex", "gex_chg", "vix", "vix_chg"]
    rows = []
    vecs: List[np.ndarray] = []

    for i in range(lookback - 1, len(df)):
        w = df.iloc[i - lookback + 1 : i + 1][feat_cols].copy()
        if w.isna().any().any():
            continue
        # z-score per-feature within window for shape similarity
        w = (w - w.mean()) / (w.std(ddof=0).replace(0, np.nan))
        w = w.fillna(0.0)
        vec = w.to_numpy().reshape(-1).astype(float)
        vecs.append(vec)
        rows.append(
            {
                "idx": i,
                "src_date": pd.Timestamp(df.iloc[i]["date"]).date().isoformat(),
                "close_src": float(df.iloc[i]["close"]),
            }
        )
    return pd.DataFrame(rows), vecs


def _attach_calendar_next_dates(windows: pd.DataFrame, root: Path) -> pd.DataFrame:
    """next_date must come from the NYSE calendar, never BDay(1): around holidays
    BDay points at non-trading days, whose empty paths silently drop candidates."""
    windows = windows.copy()
    windows["next_date"] = windows["src_date"].apply(
        lambda s: _next_open_day_from_calendar(root, pd.to_datetime(s)).date().isoformat()
    )
    return windows


def _load_intraday_30m(root: Path) -> pd.DataFrame:
    p = root / "data" / "SP_SPX, 30.csv"
    df = pd.read_csv(p)
    if "time" not in df.columns:
        raise ValueError("SP_SPX, 30.csv missing time column")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = _parse_30m_time_columns(df)
    df = df.dropna(subset=["dt", "open", "close"]).sort_values("dt")
    return df


def _next_day_path_returns(intra30: pd.DataFrame, trade_date: str) -> np.ndarray:
    """Return each 30m bar OPEN relative to the 06:30 session open."""
    d = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(d):
        return np.array([], dtype=float)
    day_df = intra30[intra30["date"] == d].copy()
    if day_df.empty:
        return np.array([], dtype=float)
    day_df = day_df.sort_values("dt")
    open0 = float(day_df.iloc[0]["open"])
    if not np.isfinite(open0) or open0 == 0:
        return np.array([], dtype=float)
    ret = day_df["open"].to_numpy(dtype=float) / open0 - 1.0
    return ret


def _day_open_gap_return(intra30: pd.DataFrame, trade_date: str) -> float:
    """Return the 06:30 open relative to the previous available session close."""
    d = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(d):
        return float("nan")
    day_df = intra30[intra30["date"] == d].copy().sort_values("dt")
    prev_df = intra30[intra30["date"] < d].copy().sort_values("dt")
    if day_df.empty or prev_df.empty:
        return float("nan")
    open0 = float(day_df.iloc[0]["open"])
    prev_close = float(prev_df.iloc[-1]["close"])
    if not np.isfinite(open0) or not np.isfinite(prev_close) or prev_close == 0:
        return float("nan")
    return open0 / prev_close - 1.0


def _collect_candidates(
    root: Path,
    windows: pd.DataFrame,
    vecs: List[np.ndarray],
    anchor_pos: int,
    intra30: pd.DataFrame,
    lookback: int,
    historical_only: bool,
    target_early_close: bool,
    top_k: int,
) -> Tuple[List[tuple], str]:
    """Collect (i, sim, ret_path, open_gap_ret) candidates for the anchor window.

    Normal targets match only normal-session historical days. Early-close targets
    prefer the (small) pool of historical early-close days; if that pool cannot
    fill top_k, fall back to normal sessions truncated to the early-close bar
    count so the shape source stays large while the horizon matches the session.
    """
    anchor_vec = vecs[anchor_pos]

    def _scan(match_early: bool, trim_to: int | None, min_bars: int) -> List[tuple]:
        found = []
        for i in range(len(windows)):
            if i == anchor_pos:
                continue
            if historical_only and i >= anchor_pos:
                continue
            # avoid immediate neighborhood to reduce leakage of near-identical rolling windows
            if abs(i - anchor_pos) < lookback:
                continue
            candidate_next = windows.iloc[i]["next_date"]
            if _is_early_close_day(root, candidate_next) != match_early:
                continue
            ret_path = _next_day_path_returns(intra30, candidate_next)
            if trim_to is not None:
                ret_path = ret_path[:trim_to]
            if len(ret_path) < min_bars:
                continue
            sim = _cosine_similarity(anchor_vec, vecs[i])
            open_gap_ret = _day_open_gap_return(intra30, candidate_next)
            found.append((i, sim, ret_path, open_gap_ret))
        return found

    if not target_early_close:
        return _scan(False, None, _MIN_CANDIDATE_BARS), "normal_pool"

    pool = _scan(True, _EARLY_CLOSE_BARS, _EARLY_CLOSE_BARS - 1)
    if len(pool) >= top_k:
        return pool, "early_close_pool"
    return (
        _scan(False, _EARLY_CLOSE_BARS, _EARLY_CLOSE_BARS - 1),
        "normal_pool_trimmed_to_early_close",
    )


def _pt_time_labels_for_target(intra30: pd.DataFrame, target_date: str, n: int) -> List[str]:
    d = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(d):
        return [f"bar{i+1}" for i in range(n)]
    day_df = intra30[intra30["date"] == d].copy().sort_values("dt")
    if len(day_df) >= n:
        dt_vals = pd.to_datetime(day_df["dt"], errors="coerce").head(n)
        return [x.strftime("%H:%M") if pd.notna(x) else f"bar{i+1}" for i, x in enumerate(dt_vals)]

    start = pd.Timestamp(d.date()) + pd.Timedelta(hours=6, minutes=30)
    return [(start + pd.Timedelta(minutes=30 * i)).strftime("%H:%M") for i in range(n)]


def _expected_range_pct_from_model(root: Path, target_date: str) -> Tuple[float, Dict[str, float]]:
    """
    expected_range_pct = sum(P(regime_i) * q50_i), from primary sheet + range quantiles json.
    """
    xlsx = root / "prediction_log_all_models.xlsx"
    qjson = root / "aio_v1" / "range_quantiles_by_regime.json"

    if not xlsx.exists() or not qjson.exists():
        return 0.012, {"fallback": 1.0}

    xls = pd.ExcelFile(xlsx)
    # use parent primary model sheet if exists, else first sheet
    sheet = "aio_v1" if "aio_v1" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xlsx, sheet_name=sheet)
    if "date" not in df.columns:
        return 0.012, {"fallback": 1.0}

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    td = pd.to_datetime(target_date).normalize()
    row = df[df["date"] == td].tail(1)
    if row.empty:
        row = df[df["date"] <= td].tail(1)
    if row.empty:
        row = df.tail(1)
    row = row.iloc[0]
    row_date = pd.Timestamp(row["date"]).normalize()
    if pd.notna(row_date) and row_date != td:
        print(
            f"[WARN] expected_range regime probs fall back to {row_date.date()} "
            f"(target {td.date()} not in prediction_log sheet)"
        )

    with open(qjson, "r", encoding="utf-8") as f:
        q = json.load(f)

    p0 = float(row.get("vol_P_regime0", 0.33))
    p1 = float(row.get("vol_P_regime1", 0.34))
    p2 = float(row.get("vol_P_regime2", 0.33))
    q50_0 = float(q.get("0", {}).get("q50", 0.006))
    q50_1 = float(q.get("1", {}).get("q50", 0.011))
    q50_2 = float(q.get("2", {}).get("q50", 0.021))
    expected = p0 * q50_0 + p1 * q50_1 + p2 * q50_2
    return max(expected, 1e-4), {"p0": p0, "p1": p1, "p2": p2, "q50_0": q50_0, "q50_1": q50_1, "q50_2": q50_2}


def _plot_prediction(
    fig_path: Path,
    target_date: str,
    source_date: str,
    pred_scaled_pct: np.ndarray,
    pt_labels: List[str],
    prev_close: float,
    actual_pct: np.ndarray | None = None,
    actual_open_gap_pct: float | None = None,
) -> None:
    bars = np.arange(len(pred_scaled_pct))
    if not np.isfinite(prev_close) or prev_close <= 0:
        raise ValueError(f"Invalid previous close for {target_date}: {prev_close}")
    # Anchor at previous close with no modeled gap: the similarity-matched gap
    # was uncorrelated noise and only shifted every predicted level.
    predicted_open = prev_close
    predicted_prices = predicted_open * (1.0 + pred_scaled_pct / 100.0)
    predicted_high = float(np.max(predicted_prices))
    predicted_low = float(np.min(predicted_prices))

    plt.figure(figsize=(12, 6))
    plt.plot(bars, predicted_prices, linewidth=2.2, label="Predicted SPX Path")
    has_actual = (
        actual_pct is not None
        and len(actual_pct) > 0
        and actual_open_gap_pct is not None
        and np.isfinite(actual_open_gap_pct)
    )
    if has_actual:
        L = min(len(pred_scaled_pct), len(actual_pct))
        actual_open = prev_close * (1.0 + float(actual_open_gap_pct) / 100.0)
        actual_prices = actual_open * (1.0 + actual_pct[:L] / 100.0)
        plt.plot(np.arange(L), actual_prices, linewidth=2.0, label="Actual SPX Open Path")
    plt.axhline(prev_close, color="gray", linestyle="--", linewidth=1.1, label=f"Previous Close: {prev_close:.2f}")
    plt.axhline(
        predicted_high,
        color="darkorange",
        linestyle="-.",
        linewidth=1.2,
        label=f"Predicted High: {predicted_high:.2f}",
    )
    plt.axhline(
        predicted_low,
        color="seagreen",
        linestyle="-.",
        linewidth=1.2,
        label=f"Predicted Low: {predicted_low:.2f}",
    )
    current_ticks = list(plt.yticks()[0])
    reference_ticks = [prev_close, predicted_high, predicted_low]
    normal_tick_gaps = np.diff(sorted(current_ticks))
    normal_spacing = float(np.median(normal_tick_gaps)) if len(normal_tick_gaps) else 0.0
    min_gap = normal_spacing * 0.55
    current_ticks = [
        tick
        for tick in current_ticks
        if all(abs(float(tick) - reference) >= min_gap for reference in reference_ticks)
    ]
    merged_ticks = sorted({round(float(tick), 2) for tick in current_ticks + reference_ticks})
    plt.yticks(merged_ticks, [f"{tick:.2f}" for tick in merged_ticks])
    plt.title(f"Next-Day Similarity SPX Price Path | target={target_date} | source={source_date}")
    ticks = np.arange(0, len(pt_labels))
    plt.xticks(ticks=ticks, labels=[pt_labels[i] for i in ticks], rotation=35, ha="right")
    plt.xlabel("PT Time (America/Los_Angeles)")
    plt.ylabel("SPX Price")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    notes = [
        "Anchor: previous close (open gap not modeled)",
        f"Predicted high / low: {predicted_high:.2f} / {predicted_low:.2f}",
    ]
    if has_actual:
        notes.extend(
            [
                f"Actual open gap: {float(actual_open_gap_pct):+.2f}%",
                f"Actual 06:30 open: {actual_open:.2f}",
            ]
        )
    plt.annotate(
        "\n".join(notes),
        xy=(0, predicted_open),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_scenarios(
    fig_path: Path,
    target_date: str,
    source_date: str,
    scenario_items: List[dict],
    weighted_scaled_pct: np.ndarray,
    pt_labels: List[str],
    actual_pct: np.ndarray | None = None,
) -> None:
    """Companion chart: the five matched historical paths shown individually
    (percent vs open), so the dispersion between scenarios stays visible."""
    plt.figure(figsize=(12, 6))
    for item in scenario_items:
        p = np.asarray(item["path_pct"], dtype=float)
        plt.plot(
            np.arange(len(p)),
            p,
            linewidth=1.2,
            alpha=0.55,
            label=f"#{item['rank']} {item['next_date']} (sim {item['similarity']:.2f}, {item['label']})",
        )
    plt.plot(
        np.arange(len(weighted_scaled_pct)),
        weighted_scaled_pct,
        linewidth=2.6,
        color="black",
        label="Weighted (scaled)",
    )
    if actual_pct is not None and len(actual_pct) > 0:
        plt.plot(
            np.arange(len(actual_pct)),
            np.asarray(actual_pct, dtype=float),
            linewidth=2.0,
            color="crimson",
            linestyle="--",
            label="Actual",
        )
    plt.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    plt.title(f"Top-5 Scenario Paths (% vs open) | target={target_date} | source={source_date}")
    ticks = np.arange(0, len(pt_labels))
    plt.xticks(ticks=ticks, labels=[pt_labels[i] for i in ticks], rotation=35, ha="right")
    plt.xlabel("PT Time (America/Los_Angeles)")
    plt.ylabel("Return vs Open (%)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _backfill_actual_overlays(
    store: Dict[str, dict],
    out_dir: Path,
    intra30: pd.DataFrame,
    daily: pd.DataFrame,
    root: Path,
) -> int:
    """
    For historical predictions already stored in JSON:
    if actual target-day path now exists and overlay is missing, regenerate chart with actual line.
    """
    updated = 0
    for target_date, rec in list(store.items()):
        # Legacy records used bar-close paths. Do not mix a new bar-open actual
        # overlay with an old bar-close prediction path.
        if rec.get("path_definition") != "30m_bar_open_relative_to_0630_open":
            continue
        weighted = rec.get("weighted_path", [])
        if not isinstance(weighted, list) or not weighted:
            continue
        pred_pct = np.array(
            [float(x.get("weighted_ret_scaled_pct", np.nan)) for x in weighted],
            dtype=float,
        )
        pred_pct = pred_pct[np.isfinite(pred_pct)]
        if len(pred_pct) < 3:
            continue

        expected_bars = _expected_session_bars(root, target_date)
        actual_ret = _next_day_path_returns(intra30, target_date)
        if len(actual_ret) < expected_bars - 1:
            continue
        actual_pct = actual_ret * 100.0

        fig_path = out_dir / f"weighted_path_{target_date}.png"
        # Daily backfill only manages charts still present in the active folder.
        # Archived or manually removed charts are intentionally left untouched.
        if not fig_path.exists():
            continue
        stored_actual = rec.get("actual_path_pct") or []
        # Re-process records frozen with a partial intraday actual, and records
        # rendered with an older chart definition (e.g. modeled-gap anchoring).
        if (
            bool(rec.get("has_actual_overlay", False))
            and rec.get("chart_definition") == _CHART_DEFINITION
            and len(stored_actual) >= expected_bars - 1
        ):
            continue

        source_date = str(rec.get("source_date", ""))
        pt_labels = _pt_time_labels_for_target(intra30, target_date, len(pred_pct))
        actual_gap_pct = _day_open_gap_return(intra30, target_date) * 100.0
        prev_rows = daily[daily["date"] < pd.Timestamp(target_date)]
        if prev_rows.empty:
            continue
        prev_close = float(prev_rows.iloc[-1]["close"])
        _plot_prediction(
            fig_path,
            target_date,
            source_date,
            pred_pct,
            pt_labels,
            prev_close,
            actual_pct,
            actual_open_gap_pct=actual_gap_pct,
        )
        rec["actual_path_pct"] = [round(float(x), 6) for x in actual_pct]
        rec["actual_open_gap_pct"] = round(float(actual_gap_pct), 6)
        rec["has_actual_overlay"] = True
        rec["chart_definition"] = _CHART_DEFINITION
        rec["actual_overlay_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        updated += 1
    return updated


def _rolling_evaluation(
    store: Dict[str, dict],
    df_daily: pd.DataFrame,
    root: Path,
    window: int = 30,
) -> Dict[str, object] | None:
    """Evaluate the last `window` stored predictions that have complete actuals.

    Direction is compared against the always-long baseline; range (model
    expected_range, daily H-L definition) against the 5-day-mean H-L baseline.
    """
    daily = df_daily[["date", "high", "low", "close"]].copy().sort_values("date")
    daily["hl_pct"] = (daily["high"] - daily["low"]) / daily["close"] * 100.0
    daily["baseline5_hl_pct"] = daily["hl_pct"].shift(1).rolling(5).mean()
    prev_close = daily["close"].shift(1)
    daily["high_vs_prev_pct"] = (daily["high"] / prev_close - 1.0) * 100.0
    daily["low_vs_prev_pct"] = (daily["low"] / prev_close - 1.0) * 100.0
    daily_key = daily["date"].dt.normalize().dt.date.astype(str)
    hl_map = dict(zip(daily_key, daily["hl_pct"]))
    base5_map = dict(zip(daily_key, daily["baseline5_hl_pct"]))
    high_map = dict(zip(daily_key, daily["high_vs_prev_pct"]))
    low_map = dict(zip(daily_key, daily["low_vs_prev_pct"]))

    rows = []
    for tgt, rec in store.items():
        if not isinstance(rec, dict):
            continue
        if rec.get("path_definition") != "30m_bar_open_relative_to_0630_open":
            continue
        if not rec.get("has_actual_overlay"):
            continue
        ap_ = rec.get("actual_path_pct") or []
        wp = rec.get("weighted_path") or []
        if not ap_ or not wp:
            continue
        if len(ap_) < _expected_session_bars(root, tgt) - 1:
            continue
        pred = [float(x.get("weighted_ret_scaled_pct", np.nan)) for x in wp]
        n = min(len(pred), len(ap_))
        if n < 4 or not np.isfinite(pred[n - 1]):
            continue
        pred_arr = np.array(pred[:n], dtype=float)
        act_arr = np.array(ap_[:n], dtype=float)
        dseg_p = np.diff(pred_arr)
        dseg_a = np.diff(act_arr)
        rows.append(
            {
                "tgt": tgt,
                "pred_close": float(pred_arr[-1]),
                "act_close": float(act_arr[-1]),
                "exp_range": float(rec.get("expected_range_pct_model", np.nan)),
                "act_hl": float(hl_map.get(tgt, np.nan)),
                "base5_hl": float(base5_map.get(tgt, np.nan)),
                "pred_high": float(pred_arr.max()),
                "pred_low": float(pred_arr.min()),
                "act_high": float(high_map.get(tgt, np.nan)),
                "act_low": float(low_map.get(tgt, np.nan)),
                "last30_hit": float(
                    np.sign(pred_arr[-1] - pred_arr[-2]) == np.sign(act_arr[-1] - act_arr[-2])
                ),
                "seg_hits": [float(np.sign(dseg_p[s]) == np.sign(dseg_a[s])) for s in range(len(dseg_p))],
            }
        )

    rows = sorted(rows, key=lambda r: r["tgt"])[-max(5, int(window)):]
    if len(rows) < 5:
        return None

    pred_close = np.array([r["pred_close"] for r in rows])
    act_close = np.array([r["act_close"] for r in rows])
    model_hit = float(np.mean(np.sign(pred_close) == np.sign(act_close)))
    long_hit = float(np.mean(act_close > 0))
    model_pnl = float(np.mean(act_close * np.sign(pred_close)))
    long_pnl = float(np.mean(act_close))

    rng = [r for r in rows if np.isfinite(r["exp_range"]) and np.isfinite(r["act_hl"]) and np.isfinite(r["base5_hl"])]
    range_metrics: Dict[str, float] = {}
    if len(rng) >= 5:
        exp_r = np.array([r["exp_range"] for r in rng])
        act_r = np.array([r["act_hl"] for r in rng])
        b5_r = np.array([r["base5_hl"] for r in rng])
        range_metrics = {
            "n_range": len(rng),
            "range_mae_model": float(np.mean(np.abs(exp_r - act_r))),
            "range_mae_baseline5": float(np.mean(np.abs(b5_r - act_r))),
            "range_corr_model": float(np.corrcoef(exp_r, act_r)[0, 1]) if np.std(exp_r) > 1e-9 else float("nan"),
            "range_corr_baseline5": float(np.corrcoef(b5_r, act_r)[0, 1]) if np.std(b5_r) > 1e-9 else float("nan"),
        }

    # High/low point accuracy (% vs previous close, matching the chart anchor)
    # and last-30min segment direction.
    hl_rows = [r for r in rows if np.isfinite(r["act_high"]) and np.isfinite(r["act_low"])]
    highlow_metrics: Dict[str, float] = {}
    if len(hl_rows) >= 5:
        h_err = np.array([r["pred_high"] - r["act_high"] for r in hl_rows])
        l_err = np.array([r["pred_low"] - r["act_low"] for r in hl_rows])
        h_broken = [r["act_high"] - r["pred_high"] for r in hl_rows if r["act_high"] > r["pred_high"]]
        l_broken = [r["pred_low"] - r["act_low"] for r in hl_rows if r["act_low"] < r["pred_low"]]
        highlow_metrics = {
            "n_highlow": len(hl_rows),
            "high_abs_err_median_pct": float(np.median(np.abs(h_err))),
            "low_abs_err_median_pct": float(np.median(np.abs(l_err))),
            "high_bias_pct": float(np.mean(h_err)),
            "low_bias_pct": float(np.mean(l_err)),
            "high_break_prob": float(len(h_broken) / len(hl_rows)),
            "low_break_prob": float(len(l_broken) / len(hl_rows)),
            "high_break_avg_overshoot_pct": float(np.mean(h_broken)) if h_broken else float("nan"),
            "low_break_avg_overshoot_pct": float(np.mean(l_broken)) if l_broken else float("nan"),
        }
    last30_metrics = {"direction_hit_last30m": float(np.mean([r["last30_hit"] for r in rows]))}

    # Per-30min-segment direction hit, ranked. Early-close days contribute to
    # the morning segments only (their bar times align with the full session).
    seg_labels = [
        "06:30->07:00", "07:00->07:30", "07:30->08:00", "08:00->08:30",
        "08:30->09:00", "09:00->09:30", "09:30->10:00", "10:00->10:30",
        "10:30->11:00", "11:00->11:30", "11:30->12:00", "12:00->12:30",
        "12:30->13:00",
    ]
    seg_ranked = []
    for s in range(len(seg_labels)):
        vals = [r["seg_hits"][s] for r in rows if len(r["seg_hits"]) > s]
        if len(vals) >= 5:
            seg_ranked.append({"segment": seg_labels[s], "hit": float(np.mean(vals)), "n": len(vals)})
    seg_ranked.sort(key=lambda x: -x["hit"])

    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": int(window),
        "n": len(rows),
        "first_target": rows[0]["tgt"],
        "last_target": rows[-1]["tgt"],
        "direction_hit_model": model_hit,
        "direction_hit_always_long": long_hit,
        "avg_open_to_close_pnl_model_pct": model_pnl,
        "avg_open_to_close_pnl_always_long_pct": long_pnl,
        "model_beats_long_direction": bool(model_hit > long_hit),
        **range_metrics,
        **highlow_metrics,
        **last30_metrics,
        "segment_direction_hit_ranked": seg_ranked,
    }


def _baseline_range_forecast(df_daily: pd.DataFrame, source_date: str, n_days: int = 5) -> float:
    """Tomorrow's baseline H-L range forecast: mean of the last n daily ranges."""
    daily = df_daily[["date", "high", "low", "close"]].copy().sort_values("date")
    daily = daily[daily["date"] <= pd.Timestamp(source_date)]
    if daily.empty:
        return float("nan")
    hl_pct = (daily["high"] - daily["low"]) / daily["close"] * 100.0
    return float(hl_pct.tail(n_days).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--lookback_days", type=int, default=10)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--eval_window", type=int, default=30, help="Rolling evaluation window (days with actual).")
    ap.add_argument("--anchor_date", type=str, default="", help="Feature-day (today) to anchor, format YYYY-MM-DD")
    ap.add_argument(
        "--historical_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only windows strictly before anchor_date (default on; "
        "--no-historical_only allows future windows and leaks when backfilling).",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    lookback = max(5, int(args.lookback_days))
    top_k = max(3, int(args.top_k))

    out_dir = root / "log" / "nextday_similarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    ingest_info = _ingest_latest_exported_30m(root)

    df_daily = _load_daily_features(root)
    daily_prices = df_daily[["date", "close"]].copy().sort_values("date")
    windows, vecs = _build_window_vectors(df_daily, lookback=lookback)
    if len(windows) < top_k + 20:
        raise ValueError("Not enough daily windows for similarity search")
    windows = _attach_calendar_next_dates(windows, root)

    if args.anchor_date:
        anchor_date = pd.to_datetime(args.anchor_date).date().isoformat()
        anchor_row = windows[windows["src_date"] == anchor_date]
        if anchor_row.empty:
            raise ValueError(f"anchor_date not found in windows: {anchor_date}")
        anchor_pos = int(anchor_row.index[-1])
    else:
        anchor_pos = len(windows) - 1
        anchor_date = str(windows.iloc[anchor_pos]["src_date"])

    latest_open_day = _load_nyse_open_days(root)
    if not latest_open_day.empty:
        today = pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None).normalize()
        available_days = latest_open_day[latest_open_day <= today]
        if not available_days.empty:
            expected_latest = available_days.iloc[-1]
            anchor_ts = pd.Timestamp(anchor_date).normalize()
            if anchor_ts < expected_latest:
                print(
                    f"[WARN] SPX daily features are stale: latest={anchor_ts.date()}, "
                    f"latest NYSE open day={expected_latest.date()}. "
                    "Prediction target may remain on an old date."
                )

    intra30 = _load_intraday_30m(root)

    target_date = _next_open_day_from_calendar(root, pd.to_datetime(anchor_date))
    target_date_str = target_date.date().isoformat()
    target_early_close = _is_early_close_day(root, target_date)
    target_session_bars = _expected_session_bars(root, target_date)

    sims, candidate_match_mode = _collect_candidates(
        root=root,
        windows=windows,
        vecs=vecs,
        anchor_pos=anchor_pos,
        intra30=intra30,
        lookback=lookback,
        historical_only=bool(args.historical_only),
        target_early_close=target_early_close,
        top_k=top_k,
    )
    if len(sims) < top_k:
        raise ValueError("Not enough historical candidates with valid next-day 30min path")

    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:top_k]
    sim_values = np.array([x[1] for x in top], dtype=float)
    conf = _softmax(sim_values, temp=0.08)

    # align by min length among top-k
    min_len = int(min(len(x[2]) for x in top))
    top_paths = [x[2][:min_len] for x in top]
    weighted_path = np.zeros(min_len, dtype=float)
    for w, p in zip(conf, top_paths):
        weighted_path += w * p
    gap_values = np.array([x[3] for x in top], dtype=float)
    valid_gap = np.isfinite(gap_values)
    expected_open_gap_ret = (
        float(np.average(gap_values[valid_gap], weights=conf[valid_gap]))
        if valid_gap.any()
        else float("nan")
    )

    expected_range_pct, range_meta = _expected_range_pct_from_model(root, target_date_str)
    raw_range_pct = float(np.max(weighted_path) - np.min(weighted_path))
    scale = expected_range_pct / raw_range_pct if raw_range_pct > 1e-8 else 1.0
    scaled_path = weighted_path * scale

    # top5 table
    recs = []
    label_scores: Dict[str, float] = {}
    for (rank, (idx, sim, ret_path, open_gap_ret)) in enumerate(top, start=1):
        w = float(conf[rank - 1])
        src_date = windows.iloc[idx]["src_date"]
        next_date = windows.iloc[idx]["next_date"]
        lbl = _pattern_label(ret_path[:min_len])
        label_scores[lbl] = label_scores.get(lbl, 0.0) + w
        recs.append(
            {
                "rank": rank,
                "similar_src_date": src_date,
                "historical_next_date": next_date,
                "similarity": round(float(sim), 6),
                "confidence": round(w, 6),
                "label": lbl,
                "nextday_open_gap_ret": round(float(open_gap_ret), 6) if np.isfinite(open_gap_ret) else None,
                "nextday_open_path_end_ret": round(float(ret_path[min_len - 1]), 6),
                "nextday_range_pct": round(float(np.max(ret_path[:min_len]) - np.min(ret_path[:min_len])), 6),
            }
        )
    top_df = pd.DataFrame(recs)

    # path output
    bars = np.arange(1, min_len + 1)
    pt_labels = _pt_time_labels_for_target(intra30, target_date_str, min_len)
    path_df = pd.DataFrame(
        {
            "bar_index": bars,
            "pt_time": pt_labels,
            "weighted_ret_raw_pct": weighted_path * 100.0,
            "weighted_ret_scaled_pct": scaled_path * 100.0,
            "target_date": target_date_str,
            "source_date": anchor_date,
            "expected_range_pct_model": expected_range_pct * 100.0,
            "raw_range_pct": raw_range_pct * 100.0,
            "scale_factor": scale,
        }
    )

    fig_png = out_dir / f"weighted_path_{target_date_str}.png"
    summary_txt = out_dir / f"summary_{target_date_str}.txt"
    json_store = out_dir / "nextday_similarity_results.json"

    # Upsert JSON store: one record per target_date, keep latest run only.
    store: Dict[str, dict] = {}
    if json_store.exists():
        try:
            loaded = json.loads(json_store.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                store = loaded
        except Exception:
            store = {}

    store[target_date_str] = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_date": anchor_date,
        "target_date": target_date_str,
        "lookback_days": lookback,
        "top_k": top_k,
        "expected_range_pct_model": expected_range_pct * 100.0,
        "raw_weighted_range_pct": raw_range_pct * 100.0,
        "scale_factor": float(scale),
        "target_early_close": bool(target_early_close),
        "expected_session_bars": int(target_session_bars),
        "candidate_match_mode": candidate_match_mode,
        "path_definition": "30m_bar_open_relative_to_0630_open",
        "chart_definition": _CHART_DEFINITION,
        "expected_open_gap_pct": expected_open_gap_ret * 100.0,
        "model_range_components": {k: float(v) for k, v in range_meta.items()},
        "top5": top_df.to_dict(orient="records"),
        "weighted_path": path_df.to_dict(orient="records"),
    }
    json_store.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")

    # Remove CSV artifacts (current target) to enforce JSON-only storage.
    for p in (
        out_dir / f"top5_similar_days_{target_date_str}.csv",
        out_dir / f"weighted_path_{target_date_str}.csv",
    ):
        if p.exists():
            p.unlink()

    actual_ret_now = _next_day_path_returns(intra30, target_date_str)
    # Only persist a complete-session actual: a partial intraday path written
    # here would otherwise be frozen forever by the backfill's overlay skip.
    actual_pct_now = actual_ret_now * 100.0 if len(actual_ret_now) >= target_session_bars - 1 else None
    actual_open_gap_pct_now = _day_open_gap_return(intra30, target_date_str) * 100.0
    prev_rows = daily_prices[daily_prices["date"] < target_date]
    if prev_rows.empty:
        raise ValueError(f"No previous SPX close available for target date: {target_date_str}")
    prev_close = float(prev_rows.iloc[-1]["close"])
    _plot_prediction(
        fig_path=fig_png,
        target_date=target_date_str,
        source_date=anchor_date,
        pred_scaled_pct=path_df["weighted_ret_scaled_pct"].to_numpy(dtype=float),
        pt_labels=pt_labels,
        prev_close=prev_close,
        actual_pct=actual_pct_now,
        actual_open_gap_pct=actual_open_gap_pct_now,
    )

    scenarios_png = out_dir / f"scenarios_path_{target_date_str}.png"
    scenario_items = [
        {
            "rank": r["rank"],
            "next_date": r["historical_next_date"],
            "similarity": float(r["similarity"]),
            "label": r["label"],
            "path_pct": (top[r["rank"] - 1][2][:min_len] * 100.0).tolist(),
        }
        for r in recs
    ]
    _plot_scenarios(
        fig_path=scenarios_png,
        target_date=target_date_str,
        source_date=anchor_date,
        scenario_items=scenario_items,
        weighted_scaled_pct=scaled_path * 100.0,
        pt_labels=pt_labels,
        actual_pct=actual_pct_now,
    )
    store[target_date_str]["scenarios_chart"] = scenarios_png.name
    if actual_pct_now is not None:
        store[target_date_str]["actual_path_pct"] = [round(float(x), 6) for x in actual_pct_now]
        store[target_date_str]["actual_open_gap_pct"] = round(float(actual_open_gap_pct_now), 6)
        store[target_date_str]["has_actual_overlay"] = True
    else:
        store[target_date_str]["has_actual_overlay"] = False

    backfill_count = _backfill_actual_overlays(
        store=store, out_dir=out_dir, intra30=intra30, daily=daily_prices, root=root
    )
    json_store.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")

    evaluation = _rolling_evaluation(store, df_daily, root, window=int(args.eval_window))
    if evaluation is not None:
        (out_dir / "rolling_evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    baseline_range_pred = _baseline_range_forecast(df_daily, anchor_date)

    lbl_df = pd.DataFrame(
        sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True),
        columns=["label", "confidence_sum"],
    )

    lines = []
    lines.append("Similarity-Based Next-Day Path Summary")
    lines.append(f"source_date={anchor_date}")
    lines.append(f"target_date={target_date_str}")
    lines.append(f"lookback_days={lookback} top_k={top_k}")
    lines.append(f"ingest_source_file={ingest_info.get('source_file','')}")
    lines.append(f"ingest_appended_dates={ingest_info.get('appended_dates', [])}")
    lines.append(f"ingest_appended_rows={ingest_info.get('appended_rows', 0)}")
    lines.append(f"ingest_source_deleted={ingest_info.get('source_deleted', False)}")
    lines.append(f"ingest_skipped_reason={ingest_info.get('skipped_reason', '')}")
    lines.append(f"actual_overlay_backfilled_count={backfill_count}")
    lines.append(f"target_early_close={target_early_close} (expected_session_bars={target_session_bars})")
    lines.append(f"candidate_match_mode={candidate_match_mode}")
    lines.append("path_definition=30m bar OPEN relative to 06:30 OPEN (06:30=0%)")
    lines.append(f"expected_open_gap_vs_prev_close={expected_open_gap_ret*100.0:+.4f}%")
    if np.isfinite(actual_open_gap_pct_now):
        lines.append(f"actual_open_gap_vs_prev_close={actual_open_gap_pct_now:+.4f}%")
    lines.append(f"chart_definition={_CHART_DEFINITION}")
    lines.append(f"expected_range_pct_model={expected_range_pct*100.0:.4f}%")
    lines.append(f"baseline_range_pred_5d_mean={baseline_range_pred:.4f}% (H-L definition, reference)")
    lines.append("    说明: 模型预期波幅(主模型regime概率加权) vs 朴素基线(前5日实际波幅均值); 两者差异大时次日实际波幅是检验哪边更准的机会。")
    lines.append(f"raw_weighted_range_pct={raw_range_pct*100.0:.4f}%")
    lines.append(f"scale_factor={scale:.4f}")
    lines.append("    说明: top5路径加权平均后的原始波幅及其拉伸倍数; scale_factor越大(>2)说明5个场景分歧越大互相抵消, 当天路径形状的可信度越低, 建议改看scenarios图。")
    lines.append("")
    if evaluation is not None:
        ev = evaluation
        lines.append(f"Rolling evaluation (last {ev['n']} targets with actual, {ev['first_target']} ~ {ev['last_target']}):")
        dir_flag = "" if ev["model_beats_long_direction"] else "  [below always-long baseline]"
        lines.append(
            f"- direction hit: model {ev['direction_hit_model']:.1%} vs always-long {ev['direction_hit_always_long']:.1%}{dir_flag}"
        )
        lines.append("    说明: 收盘方向(开盘->收盘)命中率, 基线=全做多(每天猜涨); 模型须持续高于基线, 方向输出才有参考价值。")
        lines.append(
            f"- avg open->close pnl: model {ev['avg_open_to_close_pnl_model_pct']:+.3f}% vs always-long {ev['avg_open_to_close_pnl_always_long_pct']:+.3f}%"
        )
        lines.append("    说明: 按预测方向开盘进场持有到收盘的日均收益 vs 全做多; 即使命中率打平, 此值更高说明模型在大波动日方向更准。")
        if "range_mae_model" in ev:
            rng_flag = "" if ev["range_mae_model"] <= ev["range_mae_baseline5"] else "  [below 5d-mean baseline]"
            lines.append(
                f"- range MAE (H-L): model {ev['range_mae_model']:.3f}% vs 5d-mean {ev['range_mae_baseline5']:.3f}%{rng_flag}"
            )
            lines.append(
                f"- range corr (H-L): model {ev['range_corr_model']:.3f} vs 5d-mean {ev['range_corr_baseline5']:.3f}"
            )
            lines.append("    说明: 预期波幅 vs 实际日高-日低的误差与相关性, 基线=前5日H-L均值; corr越高越能分辨大小波动日, 是当前模块最可信的成分, 适合做仓位/期权参考。")
        if "high_abs_err_median_pct" in ev:
            lines.append(
                f"- high/low point err (median, vs prev close): high {ev['high_abs_err_median_pct']:.2f}% / low {ev['low_abs_err_median_pct']:.2f}%"
            )
            lines.append("    说明: 图上预测高/低点与实际日高/日低的中位误差(%相对昨收); 需小于日波幅约1/3(~0.3%)点位才可用, 当前误差≈半个日波幅, 只能看宽度不能看具体价位。")
            lines.append(
                f"- high/low bias (mean, +=pred above actual): high {ev['high_bias_pct']:+.2f}% / low {ev['low_bias_pct']:+.2f}%"
            )
            lines.append("    说明: 带符号平均偏差, 负=预测区间整体偏低挂(模型偏空); 关注符号是否翻转, 翻转提示模型多空倾向变化, 修正数值本身意义不大(噪音远大于偏差)。")
            lines.append(
                f"- high/low break prob: high {ev['high_break_prob']:.0%} (avg overshoot {ev['high_break_avg_overshoot_pct']:+.2f}%) / "
                f"low {ev['low_break_prob']:.0%} (avg overshoot {ev['low_break_avg_overshoot_pct']:+.2f}%)"
            )
            lines.append("    说明: 实际突破预测高点/跌破预测低点的天数占比; ≈50%=纯随机(两条线不是边界), <30%开始像真支撑阻力, >70%可当突破触发参考; 偏离50%才说明点位变得可用。")
        lines.append(f"- last-30min segment direction hit: {ev['direction_hit_last30m']:.1%}")
        lines.append("    说明: 尾盘12:30->13:00这段30min涨跌方向命中; 持续<50%时应忽略图中尾盘形状。")
        seg_ranked = ev.get("segment_direction_hit_ranked") or []
        if seg_ranked:
            lines.append("- segment direction hit, ranked:")
            for i in range(0, len(seg_ranked), 4):
                chunk = " | ".join(f"{x['segment']} {x['hit']:.0%}" for x in seg_ranked[i : i + 4])
                lines.append(f"    {chunk}")
            lines.append("    说明: 各30min时段涨跌方向命中排序(滚动窗口重算); 用于决定哪些时段的图值得参考、哪些直接忽略; 13段同时比较属描述性统计, 单段高命中勿直接当交易信号。")
        lines.append("")
    lines.append("Model range components:")
    for k, v in range_meta.items():
        lines.append(f"- {k}={v:.6f}")
    lines.append("")
    lines.append("Top labels (confidence_sum):")
    for _, r in lbl_df.iterrows():
        lines.append(f"- {r['label']}: {float(r['confidence_sum']):.4f}")
    lines.append("")
    lines.append("Top-5 similar historical days:")
    for r in recs:
        lines.append(
            f"- #{r['rank']} src={r['similar_src_date']} next={r['historical_next_date']} "
            f"sim={r['similarity']:.6f} conf={r['confidence']:.6f} "
            f"label={r['label']} open_gap={r['nextday_open_gap_ret']} "
            f"open_path_end={r['nextday_open_path_end_ret']:.6f} "
            f"range={r['nextday_range_pct']:.6f}"
        )
    lines.append("")
    lines.append(f"json_store={json_store}")
    lines.append(f"path_png={fig_png}")
    lines.append(f"scenarios_png={scenarios_png}")
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] source_date={anchor_date}, target_date={target_date_str}")
    print(f"[OK] ingest_30m={ingest_info}")
    print(f"[OK] actual_overlay_backfilled_count={backfill_count}")
    if evaluation is not None:
        print(
            f"[EVAL] last {evaluation['n']}: dir model {evaluation['direction_hit_model']:.1%} "
            f"vs long {evaluation['direction_hit_always_long']:.1%}; "
            f"range MAE model {evaluation.get('range_mae_model', float('nan')):.3f}% "
            f"vs 5d-mean {evaluation.get('range_mae_baseline5', float('nan')):.3f}%"
        )
        if not evaluation["model_beats_long_direction"]:
            print("[EVAL][WARN] direction signal is below the always-long baseline in this window")
    print(f"[OK] saved: {json_store}")
    print(f"[OK] saved: {fig_png}")
    print(f"[OK] saved: {scenarios_png}")
    print(f"[OK] saved: {summary_txt}")


if __name__ == "__main__":
    main()
