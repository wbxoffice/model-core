#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_all_models_v2.py

P4：SPX VDTR model v2.0

- Vol:    Rolling 3-Year (训练集 = 过去 3 年的 target_date)
- Dir/Rev:Expanding     (从 TRAIN_START_DATE 到 as_of_date)
- Trend:  Expanding

区别于 P2：
- 引入 event_calendar.csv（FOMC / VIX Exp / SPY OPEX）
- 使用 GEX/DIX v2 特征（包括分位数、变化率和交互）
"""

from pathlib import Path
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from feature_engineering_v2_1 import build_feature_frame_v21

from config import (
    SPX_FILE, DIX_FILE, EVENT_CAL_FILE, VIX_FILE,
    VOL_MODEL_FILE, DIR_MODEL_FILE, TREND_MODEL_FILE, REV_MODEL_FILE,
    VOL_MODEL_NAME, DIR_MODEL_NAME, TREND_MODEL_NAME, REV_MODEL_NAME,
    SKLEARN_VERSION, PYTHON_VERSION,
)

TRAIN_START_DATE = pd.Timestamp("2016-01-01")
ROLLING_YEARS_FOR_VOL = 8


# ================== 标签工具函数（与 P2 保持一致） ==================

def classify_range(r: float, low: float = 0.008, mid: float = 0.015) -> int:
    if pd.isna(r):
        return np.nan
    if r < low:
        return 0
    if r < mid:
        return 1
    return 2


def direction_label(ret: float, th: float = 0.002) -> int:
    if pd.isna(ret):
        return np.nan
    if ret > th:
        return 1
    if ret < -th:
        return -1
    return 0


def rolling_percentile(series: pd.Series, window: int = 252, min_periods: int = 60) -> np.ndarray:
    vals = series.values.astype(float)
    out = np.full_like(vals, np.nan, dtype=float)
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        w = vals[start:i+1]
        if len(w) < min_periods:
            continue
        out[i] = np.sum(w <= w[-1]) / len(w)
    return out


def rolling_pct(series, window=60, min_periods=20):
    vals = series.values.astype(float)
    out = np.full_like(vals, np.nan, dtype=float)
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        w = vals[start:i+1]
        if len(w) < min_periods:
            continue
        out[i] = np.sum(w <= w[-1]) / len(w)
    return out

# # ================== v2 特征工程 ==================
#
# def build_feature_frame_v2() -> pd.DataFrame:
#     """
#     读取 SPX + DIX + VIX + event_calendar，
#     构造标签 & v2 特征，并添加 target_date。
#     """
#     # -------- 读取基础数据 --------
#     spx = pd.read_csv(SPX_FILE)
#     dix = pd.read_excel(DIX_FILE)
#     events = pd.read_csv(EVENT_CAL_FILE, parse_dates=["date"])
#     vix = pd.read_csv(VIX_FILE)
#
#     # ---- 统一 date 列 ----
#     # SPX
#     if "time" in spx.columns:
#         spx["date"] = pd.to_datetime(spx["time"])
#     elif "date" in spx.columns:
#         spx["date"] = pd.to_datetime(spx["date"])
#     else:
#         raise ValueError("SPX 文件缺少 date/time 列")
#
#     # DIX
#     if "date" not in dix.columns:
#         raise ValueError("DIX 缺少 date 列")
#     dix["date"] = pd.to_datetime(dix["date"])
#
#     # VIX（与 SPX 一致：有 date / open / high / low / close）
#     if "time" in vix.columns:
#         vix["date"] = pd.to_datetime(vix["time"])
#     elif "date" in vix.columns:
#         vix["date"] = pd.to_datetime(vix["date"])
#     else:
#         raise ValueError("VIX 文件缺少 date/time 列")
#     if "close" not in vix.columns:
#         raise ValueError("VIX 文件缺少 close 列")
#     vix = vix[["date", "close"]].rename(columns={"close": "vix_close"})
#
#     # ---- 合并 ----
#     df = spx.merge(dix, on="date", how="inner")
#     df = df.merge(events, on="date", how="left")
#     df = df.merge(vix, on="date", how="left")  # 新增：VIX
#     df = df.sort_values("date").reset_index(drop=True)
#
#     # -------- target_date & 标签（与 P2 一致）--------
#     df["target_date"] = df["date"].shift(-1)
#
#     df["next_open"]  = df["open"].shift(-1)
#     df["next_high"]  = df["high"].shift(-1)
#     df["next_low"]   = df["low"].shift(-1)
#     df["next_close"] = df["close"].shift(-1)
#
#     df["next_range"] = (df["next_high"] - df["next_low"]) / df["close"]
#     df["real_regime"] = df["next_range"].apply(classify_range)
#
#     df["next_ret"]       = df["next_close"] / df["close"] - 1.0
#     df["real_direction"] = df["next_ret"].apply(direction_label)
#
#     df["real_trend"] = (
#         (df["next_close"] - df["next_open"]).abs() / df["next_open"] > 0.006
#     ).astype(float)
#     df.loc[df["next_open"].isna() | df["next_close"].isna(), "real_trend"] = np.nan
#
#     gap_up      = df["next_open"] > df["close"]
#     gap_down    = df["next_open"] < df["close"]
#     close_red   = df["next_close"] < df["next_open"]
#     close_green = df["next_close"] > df["next_open"]
#     df["real_reversal"] = ((gap_up & close_red) | (gap_down & close_green)).astype(float)
#     df.loc[df["next_open"].isna() | df["next_close"].isna(), "real_reversal"] = np.nan
#
#     # -------- 基础价格特征 --------
#     df["ret"] = df["close"].pct_change()
#     df["range"] = (df["high"] - df["low"]) / df["close"]
#
#     df["dow"] = df["date"].dt.weekday
#     df["moy"] = df["date"].dt.month
#
#     # ================== GEX v2 特征 ==================
#     df["gex_abs"] = df["gex"].abs()
#     df["gex_scaled"] = df["gex_abs"] / 1e9
#     df["gex_sign"] = np.sign(df["gex"])
#     df["log_gex_abs"] = np.log(df["gex_abs"].replace(0, np.nan))
#
#     df["gex_1d_chg"]  = df["gex"].diff(1)
#     df["gex_3d_chg"]  = df["gex"].diff(3)
#     df["gex_5d_chg"]  = df["gex"].diff(5)
#     df["gex_20d_chg"] = df["gex"].diff(20)
#
#     for col in ["gex_1d_chg", "gex_5d_chg", "gex_20d_chg"]:
#         df[f"{col}_z60"] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
#
#     # 绝对 GEX 252 天分位数（dealer 极端程度）
#     df["gex_abs_pct_252"] = (
#         df["gex_scaled"]
#         .rolling(252, min_periods=60)
#         .apply(lambda x: (x <= x.iloc[-1]).mean(), raw=False)
#     )
#
#     # 60 天分位数
#     df["gex_pct_60"] = (
#         df["gex_scaled"]
#         .rolling(60, min_periods=20)
#         .apply(lambda x: (x <= x.iloc[-1]).mean(), raw=False)
#     )
#
#     # 120 天 z-score
#     df["gex_scaled_z120"] = (
#         df["gex_scaled"] - df["gex_scaled"].rolling(120).mean()
#     ) / df["gex_scaled"].rolling(120).std()
#
#     # 极端位置 × 最近剧烈变化
#     df["gex_extreme_move_60"] = df["gex_pct_60"] * df["gex_5d_chg_z60"]
#
#     # ================== DIX v2 特征 ==================
#     for N in [20, 60, 120]:
#         roll = df["dix"].rolling(N)
#         df[f"dix_pct_{N}"] = roll.rank(pct=True)
#         df[f"dix_mean_{N}"] = roll.mean()
#
#     df["dix_1d_chg"] = df["dix"].diff(1)
#     df["dix_5d_chg"] = df["dix"].diff(5)
#
#     for col in ["dix_1d_chg", "dix_5d_chg"]:
#         df[f"{col}_z60"] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
#
#     # 简单的 trend_prev5：最近 5 日收益累积（你之后可以换成更严谨的定义）
#     df["trend_prev5"] = df["ret"].rolling(5).sum()
#     df["dix_trend_combo"] = df["dix_5d_chg_z60"] * df["trend_prev5"]
#
#     # ================== VIX 特征 ==================
#     df["vix_level"] = df["vix_close"]
#
#     df["vix_1d_chg"]  = df["vix_level"].diff(1)
#     df["vix_5d_chg"]  = df["vix_level"].diff(5)
#     df["vix_20d_chg"] = df["vix_level"].diff(20)
#
#     for col in ["vix_1d_chg", "vix_5d_chg", "vix_20d_chg"]:
#         df[f"{col}_z60"] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
#
#     # realized vol 用来比较隐含 vs 实现
#     df["realized_vol_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
#     df["vix_iv_annual"] = df["vix_level"] / 100.0
#     df["vix_vs_realized_20"] = df["vix_iv_annual"] - df["realized_vol_20"]
#
#     # ================== 事件特征 ==================
#     event_cols = [
#         "is_fomc", "is_fomc_window", "days_to_next_fomc", "days_from_prev_fomc",
#         "is_vix_expiry", "days_to_vix_expiry",
#         "is_spy_opex_m", "days_to_spy_opex_m",
#     ]
#     for c in event_cols:
#         if c not in df.columns:
#             df[c] = 0
#
#     # ================== 构造 lag 特征 ==================
#     base_cols_for_lag = [
#         "ret", "range",
#         "log_gex_abs", "gex_sign", "gex_abs_pct_252",
#         "gex_1d_chg_z60", "gex_5d_chg_z60", "gex_pct_60", "gex_scaled_z120",
#         "dix", "dix_pct_60", "dix_5d_chg_z60", "dix_trend_combo",
#         # VIX
#         "vix_level", "vix_5d_chg_z60", "vix_vs_realized_20",
#     ]
#
#     for col in base_cols_for_lag:
#         for lag in range(5):
#             df[f"{col}_lag{lag}"] = df[col].shift(lag)
#
#     # 清理 & 填补
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df = df.dropna(subset=["target_date"])  # 最后一行 target_date NaN 丢掉
#     df = df.fillna(0)
#
#     return df



# ================== 训练函数 ==================

def train_rf(X, y, model_name: str):
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X, y)
    print(f"[TRAIN] {model_name}: samples={len(X)}, classes={np.unique(y)}")
    return clf


def main(as_of_date: str | None):
    df = build_feature_frame_v21()

    df["target_date"] = pd.to_datetime(df["target_date"])

    # as_of_date = 训练截止 target_date
    if as_of_date:
        as_of = pd.to_datetime(as_of_date)
    else:
        as_of = df["target_date"].max()
    print(f"[v2 TRAIN] 截止 target_date = {as_of.date()}")

    # -------- 公共特征列：lag + dow/moy + 事件列 --------
    lag_cols = [c for c in df.columns if c.endswith(tuple([f"_lag{i}" for i in range(5)]))]
    event_cols = [
        "is_fomc", "is_fomc_window", "days_to_next_fomc", "days_from_prev_fomc",
        "is_vix_expiry", "days_to_vix_expiry",
        "is_spy_opex_m", "days_to_spy_opex_m",
    ]
    feature_cols = lag_cols + ["dow", "moy"] + event_cols

    # -------- Vol: Rolling 3-Year --------
    vol_start = as_of - pd.DateOffset(years=ROLLING_YEARS_FOR_VOL)
    mask_vol = (df["target_date"] >= vol_start) & (df["target_date"] <= as_of)
    vol_df = df.loc[mask_vol].dropna(subset=feature_cols + ["real_regime"])
    X_vol = vol_df[feature_cols]
    y_vol = vol_df["real_regime"].astype(int)

    if len(X_vol) == 0:
        raise ValueError("Vol 训练集为空，请检查数据或窗口设置。")

    vol_model = train_rf(X_vol, y_vol, VOL_MODEL_NAME)
    vol_bundle = {
        "model": vol_model,
        "feature_cols": feature_cols,
        "label_name": "real_regime",
        "model_name": VOL_MODEL_NAME,
        "train_strategy": "rolling_3y_v2",
        "train_start_date": vol_start,
        "train_end_date": as_of,
        "sklearn_version": SKLEARN_VERSION,
        "python_version": PYTHON_VERSION,
        "version": "v2.0",
    }
    with open(VOL_MODEL_FILE, "wb") as f:
        pickle.dump(vol_bundle, f)
    print(f"[SAVE] Vol v2 model -> {VOL_MODEL_FILE}")

    # -------- Dir/Trend/Rev: Expanding --------
    mask_expanding = (df["target_date"] >= TRAIN_START_DATE) & (df["target_date"] <= as_of)
    base_df = df.loc[mask_expanding].dropna(subset=feature_cols)

    # Direction
    dir_df = base_df.dropna(subset=["real_direction"])
    X_dir = dir_df[feature_cols]
    y_dir = dir_df["real_direction"].astype(int)

    dir_model = train_rf(X_dir, y_dir, DIR_MODEL_NAME)
    dir_bundle = {
        "model": dir_model,
        "feature_cols": feature_cols,
        "label_name": "real_direction",
        "model_name": DIR_MODEL_NAME,
        "train_strategy": "expanding_v2",
        "train_start_date": TRAIN_START_DATE,
        "train_end_date": as_of,
        "sklearn_version": SKLEARN_VERSION,
        "python_version": PYTHON_VERSION,
        "version": "v2.0",
    }
    with open(DIR_MODEL_FILE, "wb") as f:
        pickle.dump(dir_bundle, f)
    print(f"[SAVE] Direction v2 model -> {DIR_MODEL_FILE}")

    # Trend
    tr_df = base_df.dropna(subset=["real_trend"])
    X_tr = tr_df[feature_cols]
    y_tr = tr_df["real_trend"].astype(int)

    trend_model = train_rf(X_tr, y_tr, TREND_MODEL_NAME)
    trend_bundle = {
        "model": trend_model,
        "feature_cols": feature_cols,
        "label_name": "real_trend",
        "model_name": TREND_MODEL_NAME,
        "train_strategy": "expanding_v2",
        "train_start_date": TRAIN_START_DATE,
        "train_end_date": as_of,
        "sklearn_version": SKLEARN_VERSION,
        "python_version": PYTHON_VERSION,
        "version": "v2.0",
    }
    with open(TREND_MODEL_FILE, "wb") as f:
        pickle.dump(trend_bundle, f)
    print(f"[SAVE] Trend v2 model -> {TREND_MODEL_FILE}")

    # Reversal
    rev_df = base_df.dropna(subset=["real_reversal"])
    X_rev = rev_df[feature_cols]
    y_rev = rev_df["real_reversal"].astype(int)

    rev_model = train_rf(X_rev, y_rev, REV_MODEL_NAME)
    rev_bundle = {
        "model": rev_model,
        "feature_cols": feature_cols,
        "label_name": "real_reversal",
        "model_name": REV_MODEL_NAME,
        "train_strategy": "expanding_v2",
        "train_start_date": TRAIN_START_DATE,
        "train_end_date": as_of,
        "sklearn_version": SKLEARN_VERSION,
        "python_version": PYTHON_VERSION,
        "version": "v2.0",
    }
    with open(REV_MODEL_FILE, "wb") as f:
        pickle.dump(rev_bundle, f)
    print(f"[SAVE] Reversal v2 model -> {REV_MODEL_FILE}")

    print("=== [v2] All 4 models trained and saved ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as_of_date", type=str, default=None,
                        help="训练截止 target_date，例如 2025-12-05；默认用数据中最后一天")
    args = parser.parse_args()
    main(args.as_of_date)
