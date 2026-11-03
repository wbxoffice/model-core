#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
薄包装：唯一实现见 ../shared/strategy_engine_v3.py（2026-06-12 去重）。
本脚本只负责解析项目本地路径（config 优先，__file__ 回退，与原版语义一致）并传参调用。
原 656 行副本归档：History/strategy_engine_dedup_backup_2026-06-12/
"""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent

# 路径解析与原版完全一致：config 优先，缺失则 __file__ 回退
try:
    from config import LOG_DIR, PREDICTION_LOG_FILE as LOG_PATH
except ImportError:
    LOG_DIR = BASE_DIR / "log"
    LOG_PATH = LOG_DIR / "prediction_log.csv"

try:
    from config import VIX_FILE
    VIX_PATH = Path(VIX_FILE)
except Exception:
    VIX_PATH = BASE_DIR.parent / "data" / "CBOE_DLY_VIX.csv"

QUANTILE_PATHS = [
    BASE_DIR / "range_quantiles_by_regime.json",
    BASE_DIR.parent / "range_quantiles_by_regime.json",
]

DATA_BANNER = "SPX"  # strategy.txt 第二行写 DATA BASED ON SPX（与原版一致）

_PARENT_DIR = str(BASE_DIR.parent)
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

from shared.strategy_engine_v3 import generate_today_plan, ACCOUNT_EQUITY


if __name__ == "__main__":
    plan = generate_today_plan(
        log_path=LOG_PATH,
        log_dir=LOG_DIR,
        vix_path=VIX_PATH,
        quantile_paths=QUANTILE_PATHS,
        data_banner=DATA_BANNER,
        account_equity=ACCOUNT_EQUITY,
    )
