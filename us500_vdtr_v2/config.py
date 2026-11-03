#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全局配置文件：路径 / 模型文件名 / 决策阈值 等。
只要移动整个文件夹，这些路径仍然有效（基于相对路径）。
"""

from pathlib import Path


# 项目根目录 = 当前文件所在目录
BASE_DIR = Path(__file__).resolve().parent

# 子目录
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
LOG_DIR = BASE_DIR / "log"
BACKTEST_DIR = BASE_DIR / "backtest"

# 确保默认存在
for d in [DATA_DIR, MODEL_DIR, LOG_DIR, BACKTEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 数据文件
SPX_FILE = DATA_DIR / "US500_1D.csv"
DIX_FILE = DATA_DIR / "DIX.xlsx"
VIX_FILE = DATA_DIR / "CBOE_DLY_VIX.csv"
EVENT_CAL_FILE = DATA_DIR / "event_calendar.csv"

# 模型文件名（你可以随时换：比如 latest / yearly / monthly）
VOL_MODEL_FILE  = BASE_DIR / "model" / "spx_vol_model_v2_latest.pkl"
DIR_MODEL_FILE  = BASE_DIR / "model" / "spx_dir_model_v2_latest.pkl"
TREND_MODEL_FILE= BASE_DIR / "model" / "spx_trend_model_v2_latest.pkl"
REV_MODEL_FILE  = BASE_DIR / "model" / "spx_rev_model_v2_latest.pkl"

VOL_MODEL_NAME   = "spx_vol_model_v2_latest"
DIR_MODEL_NAME   = "spx_dir_model_v2_latest"
TREND_MODEL_NAME = "spx_trend_model_v2_latest"
REV_MODEL_NAME   = "spx_rev_model_v2_latest"

# prediction_log 日志文件
PREDICTION_LOG_FILE = LOG_DIR / "prediction_log.csv"
# prediction 从哪天开始记录（只影响新跑出来的预测，不会自动删除旧 log）
PREDICTION_START_DATE = "2024-07-01"

# ===== 决策阈值（采用前面选出来的最优：0.45 / 0.55 / 0.55） =====
P_DIR_TH = 0.45   # 方向 LONG/SHORT 信号阈值
P_TRE_TH = 0.55   # 趋势 TREND_DAY 阈值
P_REV_TH = 0.55   # 反转 REVERSAL_RISK 阈值

# sklearn 版本信息（可写入模型 meta）
SKLEARN_VERSION = "1.7.2"
PYTHON_VERSION  = "3.11+"
