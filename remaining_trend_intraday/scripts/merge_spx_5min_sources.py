#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
import re

import pandas as pd
from pandas.errors import EmptyDataError

THIS_DIR = Path(__file__).resolve().parent
TREND_ROOT = THIS_DIR.parent
PROJECT_ROOT = TREND_ROOT.parent


_TS_RE = re.compile(r"^\d{8}_\d{6}$")
SESSION_START = time(6, 30)
ALLOWED_LAST_BAR_TIMES_5M = {
    time(10, 0), time(10, 5),  # early close 10:00 PT (with/without short close bar)
    time(11, 0), time(11, 5),  # early close 11:00 PT
    time(12, 0), time(12, 5),  # early close 12:00 PT
    time(13, 0), time(13, 5),  # regular close 13:00 PT
}
FREQ_5M = "5min"


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
        # Treat timezone-naive timestamps as already local exchange time.
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


def _trim_if_latest_day_incomplete_5m(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    For each source, keep latest trade day only when it is complete on 5m grid.
    Completeness rules:
    - starts at 06:30
    - ends at one of allowed NYSE close templates (10:00/11:00/12:00/13:00, with or without short close bar)
    - no missing 5m slot between first and last bar
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
    bars = pd.to_datetime(day["datetime"], errors="coerce").dropna().drop_duplicates().sort_values()
    if bars.empty:
        keep = work[work["trade_date"] < latest_day].drop(columns=["trade_date"]).copy()
        return keep

    first_bar = bars.iloc[0]
    last_bar = bars.iloc[-1]
    is_complete = False
    if first_bar.time() == SESSION_START and last_bar.time() in ALLOWED_LAST_BAR_TIMES_5M:
        expected = pd.date_range(start=first_bar, end=last_bar, freq=FREQ_5M)
        is_complete = (len(expected) == len(bars)) and (set(expected) == set(bars.tolist()))

    if is_complete:
        return work.drop(columns=["trade_date"])

    print(
        "[WARN] latest 5m trade day incomplete; dropping latest day from source:",
        source_name,
        "latest_day=", pd.to_datetime(latest_day).date().isoformat(),
        "bar_cnt=", int(len(bars)),
        "first_bar=", first_bar.strftime("%Y-%m-%d %H:%M:%S"),
        "latest_bar=", last_bar.strftime("%Y-%m-%d %H:%M:%S"),
        "allowed_last_bars=", ",".join(sorted([t.strftime("%H:%M") for t in ALLOWED_LAST_BAR_TIMES_5M])),
    )
    keep = work[work["trade_date"] < latest_day].drop(columns=["trade_date"]).copy()
    return keep


def main() -> None:
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
    summary = []
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
        d = _trim_if_latest_day_incomplete_5m(d, name)
        if d.empty:
            summary.append({"source": name, "path": str(p), "exists": 1, "rows": 0, "min_dt": "", "max_dt": ""})
            continue
        frames.append(d)
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

    if not frames:
        raise RuntimeError("No available source data for merge.")

    merged = pd.concat(frames, ignore_index=True)

    # Priority: newer feeds override older when timestamps overlap.
    priority = {
        "vdtr_spx_5min": 1,
        "daytrade_sp_spx_5": 2,
    }
    merged["priority"] = merged["source"].map(priority).fillna(0)
    merged.loc[merged["source"].astype(str).str.startswith("trend_"), "priority"] = 3
    merged.loc[merged["source"].astype(str).str.startswith("exported_"), "priority"] = 4

    merged = merged.sort_values(["datetime", "priority"]).drop_duplicates(subset=["datetime"], keep="last")
    merged = merged.sort_values("datetime").reset_index(drop=True)
    merged["time"] = merged["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    merged["date"] = merged["datetime"].dt.normalize()

    out_dir = TREND_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"SPX_5min_merged_{ts}.csv"
    out_latest = out_dir / "SPX_5min_merged.csv"
    out_summary = out_dir / f"SPX_5min_merged_summary_{ts}.csv"
    out_summary_latest = out_dir / "SPX_5min_merged_summary.csv"

    merged[["time", "open", "high", "low", "close", "volume", "source"]].to_csv(out_csv, index=False, encoding="utf-8-sig")
    merged[["time", "open", "high", "low", "close", "volume", "source"]].to_csv(out_latest, index=False, encoding="utf-8-sig")

    s = pd.DataFrame(summary)
    s = pd.concat(
        [
            s,
            pd.DataFrame(
                [
                    {
                        "source": "merged",
                        "path": str(out_latest),
                        "exists": 1,
                        "rows": int(len(merged)),
                        "min_dt": merged["datetime"].min().isoformat(sep=" "),
                        "max_dt": merged["datetime"].max().isoformat(sep=" "),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    s.to_csv(out_summary, index=False, encoding="utf-8-sig")
    s.to_csv(out_summary_latest, index=False, encoding="utf-8-sig")

    removed_merged = _keep_latest_one_snapshot(out_dir, "SPX_5min_merged")
    removed_summary = _keep_latest_one_snapshot(out_dir, "SPX_5min_merged_summary")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_latest}")
    print(f"Wrote: {out_summary_latest}")
    print(f"Pruned old snapshots: merged={removed_merged}, summary={removed_summary}")


if __name__ == "__main__":
    main()
