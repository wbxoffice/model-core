#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把本机的真实历史预测记录快照到 history/（随 git 发布，用于展示模型过往表现）。

清理规则（公开仓库）：
- 去掉原始行情列（SPX 开高低收、收盘价、VIX 点位）：它们属于数据商的行情数据，与"预测对不对"无关；
  保留预测值、概率、真实标签（涨跌/趋势/反转/波动档）和收益率。
- 所有本机绝对路径改成相对项目根的路径。

用法：python tools/export_history.py   （每次想更新公开记录时运行，然后 git commit）
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "history"

SOURCES = {
    "aio_v1_prediction_log.csv": "aio_v1/log/prediction_log.csv",
    "spx_vdtr_v2_prediction_log_live.csv": "spx_vdtr_v2/log/prediction_log_live.csv",
    "spx_vdtr_v2_prediction_log_replay.csv": "spx_vdtr_v2/log/prediction_log_replay.csv",
    "us500_vdtr_v2_prediction_log_live.csv": "us500_vdtr_v2/log/prediction_log_live.csv",
    "us500_vdtr_v2_prediction_log_replay.csv": "us500_vdtr_v2/log/prediction_log_replay.csv",
    "trend_mode_daily_predictions.csv": "remaining_trend_intraday/results/trend_mode_daily_predictions.csv",
}
JSON_SOURCES = {
    "nextday_similarity_results.json": "log/nextday_similarity/nextday_similarity_results.json",
    "nextday_similarity_rolling_evaluation.json": "log/nextday_similarity/rolling_evaluation.json",
}
PRICE_COLS = {"spx_open", "spx_high", "spx_low", "spx_close", "spx_close_src", "vix_level"}
ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/)")


def _relpath(v):
    if not isinstance(v, str) or not ABS_PATH.match(v):
        return v
    p = v.replace("\\", "/")
    root = str(ROOT).replace("\\", "/")
    if p.lower().startswith(root.lower() + "/"):
        return p[len(root) + 1:]
    return p.rsplit("/", 1)[-1]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for name, rel in SOURCES.items():
        src = ROOT / rel
        if not src.exists():
            print(f"[SKIP] 不存在: {rel}")
            continue
        df = pd.read_csv(src)
        dropped = sorted(PRICE_COLS & set(df.columns))
        df = df.drop(columns=dropped)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(_relpath)
        df.to_csv(OUT / name, index=False)
        span = f"{df['date'].iloc[0]}..{df['date'].iloc[-1]}" if "date" in df.columns and len(df) else ""
        print(f"[OK] {name}: {len(df)} 行 {span}" + (f"（去掉 {dropped}）" if dropped else ""))
    for name, rel in JSON_SOURCES.items():
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, OUT / name)
            print(f"[OK] {name}")
    leftover = [f.name for f in OUT.glob("*.csv") if re.search(r"[A-Za-z]:\\\\|OneDrive|/Users/", f.read_text(encoding="utf-8"))]
    if leftover:
        raise SystemExit(f"[ERR] 仍含本机路径: {leftover}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
