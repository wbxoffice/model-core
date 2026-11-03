#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import sys

import pandas as pd
from pandas.errors import EmptyDataError

from market_time_utils import current_nyse_trade_date

THIS_DIR = Path(__file__).resolve().parent
TREND_ROOT = THIS_DIR.parent
PROJECT_ROOT = TREND_ROOT.parent

_TS_RE = re.compile(r"^\d{8}_\d{6}$")
FREQ_MINUTES = 5
CONTIGUITY_STEP = pd.Timedelta(minutes=FREQ_MINUTES)
OVERLAP_LOOKBACK_DAYS = 5


def _keep_latest_one_snapshot(out_dir: Path, prefix: str) -> int:
    candidates = []
    for p in out_dir.glob(f"{prefix}_*.csv"):
        ts = p.stem.replace(f"{prefix}_", "")
        if _TS_RE.fullmatch(ts):
            candidates.append((ts, p))
    candidates.sort(key=lambda x: x[0], reverse=True)
    to_delete = [p for _, p in candidates[1:]]
    for p in to_delete:
        p.unlink(missing_ok=True)
    return len(to_delete)


def _file_latest_date(path: Path) -> pd.Timestamp:
    try:
        df = pd.read_csv(path, usecols=lambda c: c.strip().lower() in {"time", "datetime", "date"})
        col = None
        for c in ["time", "datetime", "date"]:
            if c in df.columns:
                col = c
                break
        if col is None:
            return pd.NaT
        dt = pd.to_datetime(df[col], format="ISO8601", errors="coerce", utc=True)
        if dt.notna().any():
            return dt.max().tz_convert("America/Los_Angeles").tz_localize(None)
        dt2 = pd.to_datetime(df[col], errors="coerce")
        return dt2.max()
    except Exception:
        return pd.NaT


def _discover_latest_5_file_in_root(root: Path) -> Path | None:
    patterns = [
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
    cand = []
    for pat in patterns:
        cand.extend(root.glob(pat))
    if not cand:
        return None
    uniq = list({p.resolve(): p for p in cand}.values())
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
    scored = []
    for p in uniq:
        scored.append((p, _file_prefix_priority(p), _file_latest_date(p), pd.Timestamp.fromtimestamp(p.stat().st_mtime)))
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


def _discover_latest_preferred_5_file(project_root: Path) -> tuple[Path | None, str]:
    exported_root = project_root / "exported data"
    trend_root = project_root / "data" / "trend"

    p_exp = _discover_latest_5_file_in_root(exported_root) if exported_root.exists() else None
    if p_exp is not None:
        return p_exp, "exported"

    p_trend = _discover_latest_5_file_in_root(trend_root) if trend_root.exists() else None
    if p_trend is not None:
        return p_trend, "trend"

    return None, ""


def _cleanup_older_same_prefix_files(used_file: Path) -> int:
    """
    Keep only the latest-date file within the same 5min prefix family
    in the same folder as used_file.
    """
    parent = used_file.parent
    if not parent.exists():
        return 0

    name = used_file.name.lower()
    if not ("sp_dly_spx" in name or "spcfd_spx" in name or "sp_spx" in name or "cboe_dly_spx" in name):
        return 0
    if not ("5min" in name or ", 5" in name or ",5" in name):
        return 0

    # Keep this intentionally narrow. Broad patterns like "*SPCFD_SPX*5*.csv"
    # can match 30min files whose random suffix contains "5".
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
    cands = []
    for pat in pats:
        cands.extend(parent.glob(pat))
    cands = list(set([p for p in cands if p.resolve() != used_file.resolve()]))

    deleted = 0
    for old in cands:
        try:
            old.unlink()
            deleted += 1
        except Exception:
            pass
    return deleted


def _load_one(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    tcol = "time" if "time" in df.columns else ("datetime" if "datetime" in df.columns else "date")
    raw = df[tcol].astype(str)
    has_tz_mask = raw.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True, na=False)
    dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if has_tz_mask.any():
        dt_utc = pd.to_datetime(raw[has_tz_mask], format="ISO8601", utc=True, errors="coerce")
        dt.loc[has_tz_mask] = dt_utc.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    if (~has_tz_mask).any():
        dt.loc[~has_tz_mask] = pd.to_datetime(raw[~has_tz_mask], errors="coerce")

    out = pd.DataFrame(
        {
            "datetime": dt,
            "open": pd.to_numeric(df.get("open"), errors="coerce"),
            "high": pd.to_numeric(df.get("high"), errors="coerce"),
            "low": pd.to_numeric(df.get("low"), errors="coerce"),
            "close": pd.to_numeric(df.get("close"), errors="coerce"),
            "volume": pd.to_numeric(df.get("volume"), errors="coerce"),
            "source": source_name,
        }
    )
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    return out


def _empty_source_summary(source_name: str, path: Path, note: str) -> dict:
    return {
        "source": source_name,
        "path": str(path),
        "exists": 1,
        "rows": 0,
        "min_dt": "",
        "max_dt": "",
        "note": note,
    }


def _load_existing_merged(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Existing merged file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "time" not in df.columns:
        raise ValueError("Existing merged file missing 'time' column.")
    dt = pd.to_datetime(df["time"], errors="coerce")
    out = pd.DataFrame(
        {
            "datetime": dt,
            "open": pd.to_numeric(df.get("open"), errors="coerce"),
            "high": pd.to_numeric(df.get("high"), errors="coerce"),
            "low": pd.to_numeric(df.get("low"), errors="coerce"),
            "close": pd.to_numeric(df.get("close"), errors="coerce"),
            "volume": pd.to_numeric(df.get("volume"), errors="coerce"),
            "source": df.get("source", "existing_merged"),
        }
    )
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    if out.empty:
        raise ValueError("Existing merged file is empty after parsing.")
    return out


def _source_priority(sources: list[tuple[Path, str]]) -> dict[str, int]:
    priority = {
        "vdtr_spx_5min": 1,
        "daytrade_sp_spx_5": 2,
    }
    for _, name in sources:
        if name.startswith("exported_"):
            priority[name] = 4
        elif name.startswith("trend_"):
            priority[name] = 3
    return priority


def _write_prediction_input(
    history: pd.DataFrame,
    current_frames: list[pd.DataFrame],
    out_path: Path,
    priority: dict[str, int],
) -> pd.DataFrame:
    """Write the full-history input consumed by the daily predictor.

    The canonical merged history intentionally excludes the in-progress trade
    date. This companion file adds current-session bars back on top of that
    stable history, so rolling intraday features retain their history without
    contaminating the canonical training input.
    """
    parts = [history.copy()]
    parts.extend(d.copy() for d in current_frames if not d.empty)
    combined = pd.concat(parts, ignore_index=True)
    combined["priority"] = combined["source"].map(priority).fillna(0)
    combined = (
        combined.sort_values(["datetime", "priority"])
        .drop_duplicates(subset=["datetime"], keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    combined["time"] = combined["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    combined[["time", "open", "high", "low", "close", "volume", "source"]].to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )
    print(
        f"[INFO] prediction input rows={len(combined)} "
        f"max={combined['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Wrote: {out_path}")
    return combined


def main() -> None:
    today_ts = current_nyse_trade_date()

    out_dir = TREND_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_latest = out_dir / "SPX_5min_merged.csv"
    out_summary_latest = out_dir / "SPX_5min_merged_summary.csv"
    out_prediction = out_dir / "SPX_5min_prediction_input.csv"

    old_raw = _load_existing_merged(out_latest)
    had_today_rows = bool((old_raw["datetime"].dt.normalize() >= today_ts).any())
    old = old_raw.copy()
    # Keep merged data stable for model features: do not include current-date bars.
    old = old[old["datetime"].dt.normalize() < today_ts].copy()
    old = old.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    if old.empty:
        raise ValueError("Existing merged file has no rows before today after current-date exclusion.")
    old_max_dt = old["datetime"].max()
    old_times = set(old["datetime"])
    old_tail_cut = old_max_dt - pd.Timedelta(days=OVERLAP_LOOKBACK_DAYS)
    old_tail_times = set(old.loc[old["datetime"] >= old_tail_cut, "datetime"])

    latest_5, latest_5_source = _discover_latest_preferred_5_file(PROJECT_ROOT)
    if latest_5 is not None:
        removed = _cleanup_older_same_prefix_files(latest_5)
        if removed > 0:
            print(f"[INFO] cleaned old same-prefix 5min files: removed={removed}, kept={latest_5}")
    sources = [
        (PROJECT_ROOT / "spx_vdtr_v2" / "data" / "SPX_5min.csv", "vdtr_spx_5min"),
        (PROJECT_ROOT / "SPX VDTR model v2.0" / "data" / "SPX_5min.csv", "vdtr_spx_5min_legacy"),
        (PROJECT_ROOT / "daytrade_strength" / "data" / "SP_SPX, 5.csv", "daytrade_sp_spx_5"),
        (PROJECT_ROOT / "daytrade long short strength" / "data" / "SP_SPX, 5.csv", "daytrade_sp_spx_5_legacy"),
    ]
    if latest_5 is not None:
        prefix = "exported" if latest_5_source == "exported" else "trend"
        sources.append((latest_5, f"{prefix}_{latest_5.stem}"))

    frames = []
    current_frames = []
    summary = []
    candidate_max_dt = pd.NaT
    priority = _source_priority(sources)

    for p, name in sources:
        if not p.exists():
            summary.append({"source": name, "path": str(p), "exists": 0, "rows": 0, "min_dt": "", "max_dt": ""})
            continue
        try:
            d = _load_one(p, name)
        except (EmptyDataError, UnicodeDecodeError, OSError, ValueError) as exc:
            print(f"[WARN] skip unreadable 5min source: {p} ({type(exc).__name__}: {exc})")
            summary.append(_empty_source_summary(name, p, f"unreadable: {type(exc).__name__}: {exc}"))
            continue
        d_current = d[d["datetime"].dt.normalize() == today_ts].copy()
        if not d_current.empty:
            current_frames.append(d_current)
        d = d[d["datetime"].dt.normalize() < today_ts].copy()
        if d.empty:
            summary.append({"source": name, "path": str(p), "exists": 1, "rows": 0, "min_dt": "", "max_dt": ""})
            continue
        summary.append(
            {
                "source": name,
                "path": str(p),
                "exists": 1,
                "rows": int(len(d)),
                "min_dt": d["datetime"].min().isoformat(sep=" "),
                "max_dt": d["datetime"].max().isoformat(sep=" "),
            }
        )
        mx = d["datetime"].max()
        if pd.isna(candidate_max_dt) or mx > candidate_max_dt:
            candidate_max_dt = mx
        cutoff = old_max_dt - pd.Timedelta(days=OVERLAP_LOOKBACK_DAYS)
        d_inc = d[d["datetime"] >= cutoff].copy()
        if not d_inc.empty:
            frames.append(d_inc)

    if pd.isna(candidate_max_dt):
        _write_prediction_input(old, current_frames, out_prediction, priority)
        print("[INFO] No valid source data. Skip incremental merge.")
        return

    if candidate_max_dt <= old_max_dt:
        if had_today_rows:
            cleaned = old.copy()
            cleaned = cleaned.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
            cleaned["time"] = cleaned["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
            cleaned[["time", "open", "high", "low", "close", "volume", "source"]].to_csv(
                out_latest, index=False, encoding="utf-8-sig"
            )
            print(
                f"[INFO] No newer source data. cleaned current-date rows for {today_ts.date().isoformat()}."
            )
            print(f"[INFO] old_max(after_clean)={old_max_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            _write_prediction_input(old, current_frames, out_prediction, priority)
            return
        _write_prediction_input(old, current_frames, out_prediction, priority)
        print(
            f"[INFO] No newer source data. old_max={old_max_dt.strftime('%Y-%m-%d %H:%M:%S')} "
            f"candidate_max={candidate_max_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return

    if not frames:
        _write_prediction_input(old, current_frames, out_prediction, priority)
        print("[INFO] No incremental rows after lookback filter. Skip.")
        return

    incoming = pd.concat(frames, ignore_index=True)
    incoming = incoming.sort_values("datetime").drop_duplicates(subset=["datetime", "source"], keep="last").reset_index(drop=True)
    incoming_new = incoming[incoming["datetime"] > old_max_dt].copy()
    if incoming_new.empty:
        _write_prediction_input(old, current_frames, out_prediction, priority)
        print("[INFO] No rows newer than existing merged max. Skip.")
        return

    overlap_exists = bool(set(incoming["datetime"]).intersection(old_tail_times))
    next_bar_exists = (old_max_dt + CONTIGUITY_STEP) in set(incoming_new["datetime"])
    if not overlap_exists and not next_bar_exists:
        print("[ERROR] Discontinuity detected. Incremental merge aborted.")
        print(f"[ERROR] old_max={old_max_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[ERROR] incoming_min_new={incoming_new['datetime'].min().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[ERROR] incoming_max_new={incoming_new['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')}")
        print("[ERROR] Condition failed: no overlap with merged tail and no contiguous next 5min bar.")
        sys.exit(2)

    merged = pd.concat([old, incoming], ignore_index=True)
    merged["priority"] = merged["source"].map(priority).fillna(0)
    merged = merged.sort_values(["datetime", "priority"]).drop_duplicates(subset=["datetime"], keep="last")
    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged["time"] = merged["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"SPX_5min_merged_{ts}.csv"
    out_summary = out_dir / f"SPX_5min_merged_summary_{ts}.csv"

    merged[["time", "open", "high", "low", "close", "volume", "source"]].to_csv(out_csv, index=False, encoding="utf-8-sig")
    merged[["time", "open", "high", "low", "close", "volume", "source"]].to_csv(out_latest, index=False, encoding="utf-8-sig")
    _write_prediction_input(merged, current_frames, out_prediction, priority)

    s = pd.DataFrame(summary)
    s = pd.concat(
        [
            s,
            pd.DataFrame(
                [
                    {
                        "source": "merged_existing",
                        "path": str(out_latest),
                        "exists": 1,
                        "rows": int(len(old)),
                        "min_dt": old["datetime"].min().isoformat(sep=" "),
                        "max_dt": old_max_dt.isoformat(sep=" "),
                    },
                    {
                        "source": "merged_incremental_window",
                        "path": "in_memory_incremental",
                        "exists": 1,
                        "rows": int(len(incoming)),
                        "min_dt": incoming["datetime"].min().isoformat(sep=" "),
                        "max_dt": incoming["datetime"].max().isoformat(sep=" "),
                    },
                    {
                        "source": "merged",
                        "path": str(out_latest),
                        "exists": 1,
                        "rows": int(len(merged)),
                        "min_dt": merged["datetime"].min().isoformat(sep=" "),
                        "max_dt": merged["datetime"].max().isoformat(sep=" "),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    s.to_csv(out_summary, index=False, encoding="utf-8-sig")
    s.to_csv(out_summary_latest, index=False, encoding="utf-8-sig")

    removed_merged = _keep_latest_one_snapshot(out_dir, "SPX_5min_merged")
    removed_summary = _keep_latest_one_snapshot(out_dir, "SPX_5min_merged_summary")

    print(f"[OK] Incremental merge updated.")
    print(f"[INFO] excluded_current_date={today_ts.date().isoformat()}")
    print(f"[INFO] old_max={old_max_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] new_max={merged['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] appended_rows={len(incoming_new)}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_latest}")
    print(f"Wrote: {out_summary_latest}")
    print(f"Pruned old snapshots: merged={removed_merged}, summary={removed_summary}")


if __name__ == "__main__":
    main()
