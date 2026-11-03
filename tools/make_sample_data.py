#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用本机真实历史"校准"生成示例数据（不是真实报价），写入 examples/sample_data/。

方法：匹配式分段抽样（matched block bootstrap）
- 每个真实交易日拆成与价格水平无关的量：开/高/低/收 相对前收的比例、成交量、
  VIX 点位、DIX、GEX/价格²、US500 相对 SPX 的比例、30 分钟和 5 分钟 K 线相对前收的比例。
- 合成日历 = 真实 NYSE 开市日（2011-05 起，DIX 有数据起）。按 20 天一段抽取真实日，
  段首挑 VIX 与当前最接近的若干候选之一，保证波动环境连续。
- 2020 年起只从有完整 30 分钟线的真实日抽，2024-08 起只从同时有完整 5 分钟线的真实日抽，
  所以日线 / 30min / 5min 来自同一个真实日，彼此自洽。
- 价格从 1500 起按抽到的比例连乘，路径和真实 SPX 不同。

用法（需要本机有真实数据，即 data/ 和 remaining_trend_intraday/data/）：
    python tools/make_sample_data.py --root . [--seed 7]
新人不需要运行它，直接用 tools/install_sample_data.py 安装已生成的示例。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK = 20
K_NEAREST = 15
START = pd.Timestamp("2011-05-02")
START_PRICE = 1500.0
ERA_30M = pd.Timestamp("2020-01-02")
BARS_30M, BARS_5M = 14, 79  # 06:30..13:00（含 13:00 收盘那根）


def _daily(path: Path, cols=("open", "high", "low", "close")) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["time"].astype(str).str[:10], errors="coerce")
    keep = ["date", *cols] + (["Volume"] if "Volume" in d.columns else [])
    d = d[keep].dropna(subset=list(cols)).sort_values("date").drop_duplicates("date", keep="last")
    return d.set_index("date")


def _intraday(path: Path, aware: bool) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig")
    t = pd.to_datetime(d["time"], utc=True, errors="coerce") if aware else pd.to_datetime(d["time"], errors="coerce")
    if aware:
        t = t.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    vol = "Volume" if "Volume" in d.columns else "volume"
    out = pd.DataFrame({"ts": t, "open": d["open"], "high": d["high"], "low": d["low"], "close": d["close"],
                        "volume": d[vol] if vol in d.columns else np.nan}).dropna(subset=["ts", "close"])
    out = out.sort_values("ts").drop_duplicates("ts", keep="last")
    out["date"] = out["ts"].dt.normalize()
    out["tod"] = out["ts"].dt.strftime("%H:%M")
    return out


def build_pool(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    spx = _daily(root / "data" / "SP_SPX, 1D.csv")
    us = _daily(root / "data" / "US500_1D.csv")
    vix = _daily(root / "data" / "CBOE_DLY_VIX.csv")
    dix = pd.read_excel(root / "data" / "DIX.xlsx")
    dix["date"] = pd.to_datetime(dix["date"]).dt.normalize()
    dix = dix.drop_duplicates("date", keep="last").set_index("date")

    pc = spx["close"].shift()
    pool = pd.DataFrame({
        "r_open": spx["open"] / pc, "r_high": spx["high"] / pc, "r_low": spx["low"] / pc, "r_close": spx["close"] / pc,
        "spx_vol": spx.get("Volume"),
    })
    for c in ["open", "high", "low", "close"]:
        pool[f"u_{c}"] = us[c].reindex(pool.index) / spx["close"]
        pool[f"v_{c}"] = vix[c].reindex(pool.index)
    pool["v_prev"] = vix["close"].shift().reindex(pool.index)
    pool["dix"] = dix["dix"].reindex(pool.index)
    pool["gex_per_p2"] = (dix["gex"] / dix["price"] ** 2).reindex(pool.index)
    pool = pool.dropna(subset=[c for c in pool.columns if c != "spx_vol"])
    pool = pool[pool.index >= START]

    def shapes(df: pd.DataFrame, n_bars: int) -> dict:
        out = {}
        df = df[df["tod"] <= "13:00"]
        for d, g in df.groupby("date"):
            if len(g) != n_bars or d not in pool.index:
                continue
            prev = spx["close"].shift().get(d)
            if pd.isna(prev):
                continue
            out[d] = pd.DataFrame({"tod": g["tod"].values,
                                   "o": g["open"].values / prev, "h": g["high"].values / prev,
                                   "l": g["low"].values / prev, "c": g["close"].values / prev,
                                   "volume": g["volume"].values})
        return out

    s30 = shapes(_intraday(root / "data" / "SP_SPX, 30.csv", aware=True), BARS_30M)
    s5 = shapes(_intraday(root / "remaining_trend_intraday" / "data" / "SPX_5min_merged.csv", aware=False), BARS_5M)
    return pool, s30, s5


def synthesize(root: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pool, s30, s5 = build_pool(root)
    era_5m = min(s5) if s5 else None
    cal = pd.read_csv(root / "data" / "nyse_open_days.csv", encoding="utf-8-sig")
    days = pd.DatetimeIndex(pd.to_datetime(cal["date"]))
    days = days[(days >= START) & (days <= pool.index.max())]

    def era_pool(d):
        if era_5m is not None and d >= era_5m:
            idx = [x for x in pool.index if x in s5 and x in s30]
        elif d >= ERA_30M:
            idx = [x for x in pool.index if x in s30]
        else:
            idx = list(pool.index)
        return pd.DatetimeIndex(idx)

    picks, cur_vix, i = [], float(pool["v_prev"].iloc[0]), 0
    cache = {}
    while i < len(days):
        d = days[i]
        key = "5m" if era_5m is not None and d >= era_5m else ("30m" if d >= ERA_30M else "d")
        if key not in cache:
            cache[key] = era_pool(d)
        cand = cache[key]
        starts = np.arange(max(1, len(cand) - BLOCK))
        dist = np.abs(pool.loc[cand[starts], "v_prev"].values - cur_vix)
        best = starts[np.argsort(dist)[:K_NEAREST]]
        s = int(rng.choice(best))
        n = min(BLOCK, len(days) - i)
        for j in range(n):
            dj = days[i + j]
            kj = "5m" if era_5m is not None and dj >= era_5m else ("30m" if dj >= ERA_30M else "d")
            if kj != key:  # era boundary inside block: restart block
                n = j
                break
            picks.append(cand[min(s + j, len(cand) - 1)])
        i += n
        cur_vix = float(pool.loc[picks[-1], "v_close"])

    rows, bars30, bars5, prev = [], [], [], START_PRICE
    for d, src in zip(days, picks):
        p = pool.loc[src]
        o, h, l, c = (prev * p[k] for k in ("r_open", "r_high", "r_low", "r_close"))
        rows.append({"date": d, "open": o, "high": h, "low": l, "close": c,
                     "Volume": p["spx_vol"],
                     **{f"u_{k}": c * p[f"u_{k}"] for k in ("open", "high", "low", "close")},
                     **{f"v_{k}": p[f"v_{k}"] for k in ("open", "high", "low", "close")},
                     "dix": p["dix"], "gex": p["gex_per_p2"] * c * c})
        for store, shape in ((bars30, s30.get(src)), (bars5, s5.get(src))):
            if shape is not None and d >= ERA_30M and (store is bars30 or (era_5m is not None and d >= era_5m)):
                sh = shape.copy()
                for k in ("o", "h", "l", "c"):
                    sh[k] = sh[k] * prev
                sh["date"] = d
                store.append(sh)
        prev = c
    return {"daily": pd.DataFrame(rows),
            "bars30": pd.concat(bars30, ignore_index=True) if bars30 else pd.DataFrame(),
            "bars5": pd.concat(bars5, ignore_index=True) if bars5 else pd.DataFrame()}


def write_examples(out: dict, root: Path, dest: Path) -> None:
    (dest / "data").mkdir(parents=True, exist_ok=True)
    (dest / "intraday").mkdir(parents=True, exist_ok=True)
    d = out["daily"]
    ds = d["date"].dt.strftime("%Y-%m-%d")
    r2 = lambda s: s.round(2)  # noqa: E731
    pd.DataFrame({"time": ds, "open": r2(d["open"]), "high": r2(d["high"]), "low": r2(d["low"]),
                  "close": r2(d["close"]), "Volume": d["Volume"]}).to_csv(dest / "data" / "SP_SPX, 1D.csv", index=False)
    pd.DataFrame({"time": ds, **{k: r2(d[f"u_{k}"]) for k in ("open", "high", "low", "close")}, "Volume": 0}
                 ).to_csv(dest / "data" / "US500_1D.csv", index=False)
    pd.DataFrame({"time": ds, **{k: d[f"v_{k}"].round(2) for k in ("open", "high", "low", "close")}}
                 ).to_csv(dest / "data" / "CBOE_DLY_VIX.csv", index=False)
    pd.DataFrame({"date": ds, "price": r2(d["close"]), "dix": d["dix"].round(6), "gex": d["gex"].round(1)}
                 ).to_csv(dest / "data" / "DIX.csv", index=False)
    for name, key in (("SPX_30min.csv", "bars30"), ("SPX_5min.csv", "bars5")):
        b = out[key]
        pd.DataFrame({"date": b["date"].dt.strftime("%Y-%m-%d"), "tod": b["tod"],
                      "open": r2(b["o"]), "high": r2(b["h"]), "low": r2(b["l"]), "close": r2(b["c"]),
                      "volume": b["volume"]}).to_csv(dest / "intraday" / name, index=False)
    # 公开日历（非行情数据）原样附带
    for f in ("nyse_open_days.csv", "event_calendar.csv"):
        (dest / "data" / f).write_bytes((root / "data" / f).read_bytes())
    print(f"[OK] daily={len(d)} days {ds.iloc[0]}..{ds.iloc[-1]}  30m bars={len(out['bars30'])}  5m bars={len(out['bars5'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = synthesize(root, args.seed)
    write_examples(out, root, root / "examples" / "sample_data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
