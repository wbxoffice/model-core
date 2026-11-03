#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 examples/sample_data/ 里的示例数据（真实数据校准的合成数据，不是真实报价）安装到程序需要的各个位置，
让刚 clone 的新机器可以直接跑通 日跑 / 开盘后趋势 / Sigma 分位 / 训练 / 回顾。

- 日期平移：整段示例按交易日顺序平移，使最后一天 = 最近一个已收盘的 NYSE 交易日（按美东时间），
  这样日跑开跑前的导出体检不会因为"数据过旧"拦下。
- 不覆盖已有文件（已有真实数据时只补缺）；加 --force 才覆盖。
- 同时在 exported data/ 放一份"导出样例"，演示 TradingView 导出文件的命名和格式（日跑会把它导入并删除）。

用法：
    python tools/install_sample_data.py            # 安装（跳过已存在的文件）
    python tools/install_sample_data.py --force    # 覆盖安装
    python tools/install_sample_data.py --dry-run  # 只列出会写哪些文件
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "examples" / "sample_data"
NY = ZoneInfo("America/New_York")
LA = "America/Los_Angeles"
LATEST_DAYS = 300
EXPORT_BARS = 300


def latest_closed_day(open_days: pd.DatetimeIndex) -> pd.Timestamp:
    now = datetime.now(NY)
    today = pd.Timestamp(now.date())
    past = open_days[open_days < today]
    if today in open_days and now.hour >= 16:
        return today
    return past[-1]


def iso_pt(dates: pd.Series, tod: pd.Series) -> pd.Series:
    ts = pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + tod).dt.tz_localize(LA)
    return ts.map(lambda x: x.isoformat())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cal = pd.read_csv(SRC / "data" / "nyse_open_days.csv", encoding="utf-8-sig")
    open_days = pd.DatetimeIndex(pd.to_datetime(cal["date"])).sort_values()

    spx = pd.read_csv(SRC / "data" / "SP_SPX, 1D.csv")
    old_days = pd.DatetimeIndex(pd.to_datetime(spx["time"]))
    end = latest_closed_day(open_days)
    new_days = open_days[open_days <= end][-len(old_days):]
    if len(new_days) != len(old_days):
        raise SystemExit("[ERR] nyse_open_days.csv 交易日不够，先运行 generate_nyse_open_days.py 扩展日历")
    remap = dict(zip(old_days, new_days))
    shift = lambda s: pd.to_datetime(s).map(remap)  # noqa: E731
    print(f"[INFO] 示例数据平移：{old_days[0].date()}..{old_days[-1].date()} → {new_days[0].date()}..{new_days[-1].date()}")

    def daily(name: str) -> pd.DataFrame:
        d = pd.read_csv(SRC / "data" / name)
        d["time"] = shift(d["time"]).dt.strftime("%Y-%m-%d")
        return d

    spx, us500, vix = daily("SP_SPX, 1D.csv"), daily("US500_1D.csv"), daily("CBOE_DLY_VIX.csv")
    dix_csv = pd.read_csv(SRC / "data" / "DIX.csv")
    dix_csv["date"] = shift(dix_csv["date"]).dt.strftime("%Y-%m-%d")
    dix_xlsx = dix_csv.assign(date=pd.to_datetime(dix_csv["date"]))[["date", "dix", "gex", "price"]]

    def intraday(name: str) -> pd.DataFrame:
        b = pd.read_csv(SRC / "intraday" / name)
        b["date"] = shift(b["date"])
        return b

    b30, b5 = intraday("SPX_30min.csv"), intraday("SPX_5min.csv")
    ohlcv = ["open", "high", "low", "close"]
    spx30 = pd.DataFrame({"time": iso_pt(b30["date"], b30["tod"]), **{k: b30[k] for k in ohlcv}, "Volume": b30["volume"]})
    naive = lambda b: (b["date"].dt.strftime("%Y-%m-%d") + "T" + b["tod"] + ":00")  # noqa: E731
    merged5 = pd.DataFrame({"time": naive(b5), **{k: b5[k] for k in ohlcv}, "volume": b5["volume"], "source": "sample_5min"})
    merged30 = pd.DataFrame({"time": naive(b30), **{k: b30[k] for k in ohlcv}, "volume": b30["volume"], "source": "sample_30m"})
    exp5 = pd.DataFrame({"time": iso_pt(b5["date"], b5["tod"]), **{k: b5[k] for k in ohlcv}, "Volume": b5["volume"]})

    cal_bytes = (SRC / "data" / "nyse_open_days.csv").read_bytes()
    evt_bytes = (SRC / "data" / "event_calendar.csv").read_bytes()
    tail = lambda d: d.tail(LATEST_DAYS)  # noqa: E731

    plan: list[tuple[str, object]] = [
        ("data/nyse_open_days.csv", cal_bytes), ("data/event_calendar.csv", evt_bytes),
        ("data/SP_SPX, 1D.csv", spx), ("data/US500_1D.csv", us500), ("data/CBOE_DLY_VIX.csv", vix),
        ("data/DIX.xlsx", dix_xlsx), ("data/SP_SPX, 30.csv", spx30),
        ("data/latest data/SP_SPX, 1D.csv", tail(spx)), ("data/latest data/US500_1D.csv", tail(us500)),
        ("data/latest data/CBOE_DLY_VIX.csv", tail(vix)), ("data/latest data/DIX.xlsx", tail(dix_xlsx)),
        ("aio_v1/data/SP_SPX, 1D.csv", spx), ("aio_v1/data/CBOE_DLY_VIX.csv", vix), ("aio_v1/data/DIX.xlsx", dix_xlsx),
        ("spx_vdtr_v2/data/SP_SPX, 1D.csv", spx), ("spx_vdtr_v2/data/CBOE_DLY_VIX.csv", vix),
        ("spx_vdtr_v2/data/DIX.xlsx", dix_xlsx), ("spx_vdtr_v2/data/event_calendar.csv", evt_bytes),
        ("us500_vdtr_v2/data/US500_1D.csv", us500), ("us500_vdtr_v2/data/CBOE_DLY_VIX.csv", vix),
        ("us500_vdtr_v2/data/DIX.xlsx", dix_xlsx), ("us500_vdtr_v2/data/event_calendar.csv", evt_bytes),
        ("remaining_trend_intraday/data/SPX_5min_merged.csv", merged5),
        ("remaining_trend_intraday/data/SPX_5min_prediction_input.csv", merged5),
        ("remaining_trend_intraday/data/SP_SPX, 30_merged.csv", merged30),
        # 导出样例（TradingView 命名格式；日跑导入后会自动删除）
        ("exported data/SP_DLY_SPX, 1D_sample.csv", tail(spx)),
        ("exported data/CAPITALCOM_SPX500, 1D_sample.csv", tail(us500)),
        ("exported data/CBOE_DLY_VIX, 1D_sample.csv", tail(vix)),
        ("exported data/DIX.csv", tail(dix_csv)[["date", "price", "dix", "gex"]]),
        ("exported data/SP_DLY_SPX, 30_sample.csv", spx30.tail(EXPORT_BARS)),
        ("exported data/SP_DLY_SPX, 5_sample.csv", exp5.tail(EXPORT_BARS)),
    ]

    written = skipped = 0
    for rel, obj in plan:
        dst = ROOT / rel
        if dst.exists() and not args.force:
            print(f"[SKIP] 已存在: {rel}")
            skipped += 1
            continue
        print(f"[{'DRY' if args.dry_run else 'WRITE'}] {rel}")
        written += 1
        if args.dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, bytes):
            dst.write_bytes(obj)
        elif rel.endswith(".xlsx"):
            obj.to_excel(dst, index=False)
        else:
            obj.to_csv(dst, index=False, encoding="utf-8-sig" if "remaining_trend_intraday" in rel else "utf-8")
    print(f"\n[OK] 写入 {written} 个、跳过 {skipped} 个。示例数据只用于跑通流程，预测结果没有交易意义。")
    print("     接下来：双击 start.bat / start.command，选 1 每日日跑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
