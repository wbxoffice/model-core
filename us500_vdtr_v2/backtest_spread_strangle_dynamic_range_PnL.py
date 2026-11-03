#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
薄包装：唯一实现见 ../shared/backtest_spread_strangle_v1.py（2026-06-12 去重）。
本脚本只负责解析项目本地路径（config 优先，__file__ 回退，与原版语义一致）并传参调用。
原 786 行副本归档：History/pnl_twins_dedup_backup_2026-06-12/
用法不变：python backtest_spread_strangle_dynamic_range_PnL.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

from pathlib import Path
import argparse
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent

# 路径解析与原版完全一致：config 优先，缺失则 __file__ 回退
try:
    from config import (
        BASE_DIR as CFG_BASE_DIR,
        BACKTEST_DIR as CFG_BACKTEST_DIR,
        VIX_FILE as CFG_VIX_FILE,
    )  # type: ignore

    BASE_DIR = Path(CFG_BASE_DIR)
    BACKTEST_DIR = Path(CFG_BACKTEST_DIR)
    VIX_PATH = Path(CFG_VIX_FILE)
except Exception:
    BASE_DIR = _SCRIPT_DIR
    BACKTEST_DIR = BASE_DIR / "backtest"
    VIX_PATH = BASE_DIR / "data" / "CBOE_DLY_VIX.csv"

_PARENT_DIR = str(_SCRIPT_DIR.parent)
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

from shared.backtest_spread_strangle_v1 import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest spread+strangle strategy with dynamic range & IV*sqrt(T) based offsets (auto-discover log files)."
    )
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD (包含)")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD (包含)")
    args = parser.parse_args()

    main(args.start, args.end, base_dir=BASE_DIR, backtest_dir=BACKTEST_DIR, vix_path=VIX_PATH)
