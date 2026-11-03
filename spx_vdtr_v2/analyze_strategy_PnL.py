#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
薄包装：唯一实现见 ../shared/pnl_analysis_v1.py（2026-06-12 去重）。
本脚本只负责解析项目本地 BASE_DIR（config 优先，__file__ 回退，与原版语义一致）并传参调用。
原 762 行副本归档：History/pnl_twins_dedup_backup_2026-06-12/
用法不变：python analyze_strategy_PnL.py [--file xxx.csv]
"""

from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent

# 路径解析与原版完全一致：config.BASE_DIR 优先，缺失则脚本所在目录
try:
    from config import BASE_DIR as CFG_BASE_DIR  # type: ignore
    BASE_DIR = Path(CFG_BASE_DIR)
except Exception:
    BASE_DIR = _SCRIPT_DIR

_PARENT_DIR = str(_SCRIPT_DIR.parent)
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

from shared.pnl_analysis_v1 import main


if __name__ == "__main__":
    main(base_dir=BASE_DIR)
