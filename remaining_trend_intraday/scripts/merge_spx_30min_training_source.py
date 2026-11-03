#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import re
from datetime import time

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
TREND_ROOT = THIS_DIR.parent
PROJECT_ROOT = TREND_ROOT.parent

LOCAL_DIR = TREND_ROOT / "data"
MAIN_DATA_DIR = PROJECT_ROOT / "data"
OUT_FILE = LOCAL_DIR / "SP_SPX, 30_merged.csv"
SESSION_START = time(6, 30)
SESSION_END = time(13, 0)  # exclusive
ALLOWED_LAST_BAR_TIMES = {
    time(10, 0),  # early close 10:00 PT (with final short bar timestamp)
    time(11, 0),  # early close 11:00 PT
    time(12, 0),  # early close 12:00 PT
    time(13, 0),  # regular close 13:00 PT
}


def _discover_30m_files(root: Path) -> list[Path]:
    pats = [
        "SP_DLY_SPX, 30_*.csv",
        "SPCFD_SPX, 30_*.csv",
        "SP_DLY_SPX,30_*.csv",
        "SPCFD_SPX,30_*.csv",
        "SP_SPX, 30_*.csv",
        "SP_SPX,30_*.csv",
        "CBOE_DLY_SPX, 30_*.csv",
        "CBOE_DLY_SPX,30_*.csv",
        "*SP_DLY_SPX*30*.csv",
        "*SPCFD_SPX*30*.csv",
        "*SP_SPX*30*.csv",
        "*CBOE_DLY_SPX*30*.csv",
    ]
    out: list[Path] = []
    for pat in pats:
        out.extend(root.glob(pat))
    uniq = list({p.resolve(): p for p in out}.values())
    return sorted(uniq)


def _parse_dt(raw: pd.Series) -> pd.Series:
    s = raw.astype(str)
    has_tz_mask = s.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True, na=False)
    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    if has_tz_mask.any():
        dt_utc = pd.to_datetime(s[has_tz_mask], format="ISO8601", utc=True, errors="coerce")
        dt.loc[has_tz_mask] = dt_utc.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    if (~has_tz_mask).any():
        dt.loc[~has_tz_mask] = pd.to_datetime(s[~has_tz_mask], errors="coerce")
    return dt


def _file_latest_dt(path: Path) -> pd.Timestamp:
    try:
        d = pd.read_csv(path, usecols=lambda c: c.strip().lower() in {"time", "datetime", "date"})
        if d.empty:
            return pd.NaT
        dt = _parse_dt(d[d.columns[0]])
        return dt.max()
    except Exception:
        return pd.NaT


def _pick_latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
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
    for p in paths:
        scored.append((p, _file_prefix_priority(p), _file_latest_dt(p), pd.Timestamp.fromtimestamp(p.stat().st_mtime)))
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


def _load_one(path: Path, source: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    tcol = "time" if "time" in d.columns else ("datetime" if "datetime" in d.columns else "date")
    dt = _parse_dt(d[tcol])
    out = pd.DataFrame(
        {
            "datetime": dt,
            "open": pd.to_numeric(d.get("open"), errors="coerce"),
            "high": pd.to_numeric(d.get("high"), errors="coerce"),
            "low": pd.to_numeric(d.get("low"), errors="coerce"),
            "close": pd.to_numeric(d.get("close"), errors="coerce"),
            "volume": pd.to_numeric(d.get("volume"), errors="coerce"),
            "source": source,
        }
    )
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    return out.sort_values("datetime").reset_index(drop=True)


def _validate_overlap_trade_days(local_df: pd.DataFrame, main_df: pd.DataFrame):
    if local_df.empty or main_df.empty:
        return
    local_days = set(pd.to_datetime(local_df["datetime"]).dt.normalize().dropna().unique())
    main_days = set(pd.to_datetime(main_df["datetime"]).dt.normalize().dropna().unique())
    overlap = local_days.intersection(main_days)
    if len(overlap) == 0:
        lmin = pd.to_datetime(local_df["datetime"]).min()
        lmax = pd.to_datetime(local_df["datetime"]).max()
        mmin = pd.to_datetime(main_df["datetime"]).min()
        mmax = pd.to_datetime(main_df["datetime"]).max()
        raise RuntimeError(
            "No overlapping trade days between local and main 30min sources. "
            f"local_range={lmin.strftime('%Y-%m-%d')}~{lmax.strftime('%Y-%m-%d')}, "
            f"main_range={mmin.strftime('%Y-%m-%d')}~{mmax.strftime('%Y-%m-%d')}. "
            "Potential data gap detected; merge aborted."
        )


def _cleanup_old_merged_files(out_file: Path):
    for p in out_file.parent.glob("SP_SPX, 30_merged*.csv"):
        if p.resolve() == out_file.resolve():
            continue
        p.unlink(missing_ok=True)
    for p in out_file.parent.glob("SP_SPX,30_merged*.csv"):
        if p.resolve() == out_file.resolve():
            continue
        p.unlink(missing_ok=True)


def _cleanup_parent_30m_files(parent_dir: Path):
    if not parent_dir.exists():
        return
    pats = [
        "SP_DLY_SPX, 30_*.csv",
        "SPCFD_SPX, 30_*.csv",
        "SP_DLY_SPX,30_*.csv",
        "SPCFD_SPX,30_*.csv",
        "SP_SPX, 30_*.csv",
        "SP_SPX,30_*.csv",
        "CBOE_DLY_SPX, 30_*.csv",
        "CBOE_DLY_SPX,30_*.csv",
        "*SP_DLY_SPX*30*.csv",
        "*SPCFD_SPX*30*.csv",
        "*SP_SPX*30*.csv",
        "*CBOE_DLY_SPX*30*.csv",
    ]
    deleted = 0
    for pat in pats:
        for p in parent_dir.glob(pat):
            if "merged" in p.name.lower():
                continue
            try:
                p.unlink()
                deleted += 1
            except Exception:
                pass
    print("[INFO] parent_30m_deleted_files=", deleted, "from", parent_dir)


def _trim_if_latest_day_incomplete(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Guardrail for live-day partial data:
    - Check the latest trade day in this source.
    - If latest day does not have a full 30m regular session, drop that day from this source.
    """
    if df.empty or "datetime" not in df.columns:
        return df

    work = df.copy()
    work["trade_date"] = pd.to_datetime(work["datetime"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["trade_date"])
    if work.empty:
        return df

    latest_day = work["trade_date"].max()
    day = work[work["trade_date"] == latest_day].copy()
    if day.empty:
        return df

    tod = pd.to_datetime(day["datetime"], errors="coerce").dt.time
    in_session = tod.apply(lambda t: t is not None and (t >= SESSION_START and t <= SESSION_END))
    day_sess = day[in_session].copy()
    bars = pd.to_datetime(day_sess["datetime"], errors="coerce").dropna().drop_duplicates().sort_values()
    bar_cnt = int(len(bars))
    latest_bar = bars.max() if bar_cnt > 0 else pd.NaT

    is_complete = False
    if bar_cnt > 0 and pd.notna(latest_bar):
        first_bar = bars.iloc[0]
        last_bar = bars.iloc[-1]
        if first_bar.time() == SESSION_START and last_bar.time() in ALLOWED_LAST_BAR_TIMES:
            expected_index = pd.date_range(start=first_bar, end=last_bar, freq="30min")
            # Complete means no missing 30m slot from open to the provider-stamped close bar.
            is_complete = (len(expected_index) == bar_cnt) and (set(expected_index) == set(bars.tolist()))

    if is_complete:
        return work.drop(columns=["trade_date"])

    keep = work[work["trade_date"] < latest_day].drop(columns=["trade_date"]).copy()
    print(
        "[WARN] latest 30m trade day incomplete; dropping latest day from source:",
        source_name,
        "latest_day=", pd.to_datetime(latest_day).date().isoformat(),
        "bar_cnt=", bar_cnt,
        "latest_bar=", latest_bar.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(latest_bar) else "NA",
        "allowed_last_bars=", ",".join(sorted([t.strftime("%H:%M") for t in ALLOWED_LAST_BAR_TIMES])),
    )
    return keep


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    main_src = _discover_30m_files(MAIN_DATA_DIR)

    p_main = _pick_latest(main_src)
    p_existing = OUT_FILE if OUT_FILE.exists() else None

    main_df = _load_one(p_main, "main_30m") if (p_main is not None and p_main.exists()) else pd.DataFrame()
    existing_df = _load_one(p_existing, "existing_merged_30m") if (p_existing is not None and p_existing.exists()) else pd.DataFrame()
    main_df = _trim_if_latest_day_incomplete(main_df, "main_30m")
    existing_df = _trim_if_latest_day_incomplete(existing_df, "existing_merged_30m")

    # Baseline policy:
    # - Use existing merged file as baseline only.
    # - Do NOT use local raw 30m files (non-merged) as baseline/source.
    frames = []
    if not existing_df.empty:
        frames.append(existing_df)
    if not main_df.empty:
        frames.append(main_df)

    if not frames:
        raise FileNotFoundError(
            f"No available 30min files to merge. Checked local={LOCAL_DIR}, main={MAIN_DATA_DIR}"
        )

    merged = pd.concat(frames, ignore_index=True)
    # Keep existing merged as baseline, then override by local/main latest sources.
    priority = {"existing_merged_30m": 0, "local_30m": 1, "main_30m": 2}
    merged["priority"] = merged["source"].map(priority).fillna(0)
    merged = merged.sort_values(["datetime", "priority"]).drop_duplicates(subset=["datetime"], keep="last")
    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged["time"] = merged["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        merged[["time", "open", "high", "low", "close", "volume", "source"]].to_csv(
            OUT_FILE, index=False, encoding="utf-8-sig"
        )
    except PermissionError as e:
        raise PermissionError(
            f"Cannot write merged file (likely opened/locked by another program): {OUT_FILE}"
        ) from e
    _cleanup_old_merged_files(OUT_FILE)
    # Do NOT clean parent data source files when source is PROJECT_ROOT/data.
    # Parent `data/` is the long-lived master store and must be preserved.

    print("[INFO] existing_merged:", str(p_existing) if p_existing is not None else "None")
    print("[INFO] local_source(raw): disabled_by_policy")
    print("[INFO] main_source(data):", str(p_main) if p_main is not None else "None")
    print("[SAVE]", OUT_FILE)
    print(
        "[INFO] merged_rows=", len(merged),
        "date_range=", merged["datetime"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "to", merged["datetime"].max().strftime("%Y-%m-%d %H:%M:%S"),
    )


if __name__ == "__main__":
    main()
