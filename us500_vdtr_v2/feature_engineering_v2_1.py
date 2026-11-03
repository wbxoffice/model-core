#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
feature_engineering_v2_1.py

VDTR v2.1 精简版特征工程：
- 读取 SPX + DIX + VIX + event_calendar
- 构造 4 个标签：real_regime / real_direction / real_trend / real_reversal
- 构造精简后的 v2.1 特征集（GEX / DIX / VIX / price / event）
- 打少量有用的 lag，减少噪声和维度

使用方式：
    from feature_engineering_v2_1 import build_feature_frame_v21
"""

from pathlib import Path
import numpy as np
import pandas as pd

from config import SPX_FILE, DIX_FILE, VIX_FILE, EVENT_CAL_FILE


# ====================== 标签工具函数 ======================

def classify_range(r: float, low: float = 0.008, mid: float = 0.015) -> int:
    """把 next day 的日内振幅映射到 3 个 regime: 0/1/2"""
    if pd.isna(r):
        return np.nan
    if r < low:
        return 0
    if r < mid:
        return 1
    return 2


def direction_label(ret: float, th: float = 0.002) -> int:
    """把 next day close-to-close 收益映射到 -1/0/1 三类"""
    if pd.isna(ret):
        return np.nan
    if ret > th:
        return 1
    if ret < -th:
        return -1
    return 0


# ====================== 主函数：构造特征表 ======================

def build_feature_frame_v21(for_training: bool = True) -> pd.DataFrame:
    """
    构造 VDTR v2.1 用的特征和标签 DataFrame

    for_training:
        - True  : 用于训练（丢掉最后一天 target_date 为空的行，并可以大胆 fillna）
        - False : 用于预测（保留最后一天，标签保持 NaN）
    """
    """
    构造 VDTR v2.1 用的特征和标签 DataFrame

    返回的 df 包含：
    - date / target_date
    - 4 个标签列：real_regime / real_direction / real_trend / real_reversal
    - 精简后的特征列 + 各自的 lag 列
    - 事件特征列（不打 lag）
    """

    # ---------- 读取原始数据 ----------
    spx = pd.read_csv(SPX_FILE)
    dix = pd.read_excel(DIX_FILE)
    vix = pd.read_csv(VIX_FILE)
    events = pd.read_csv(EVENT_CAL_FILE, parse_dates=["date"])

    # ---- 统一 date 列 ----
    # SPX
    if "time" in spx.columns:
        spx["date"] = pd.to_datetime(spx["time"], format="ISO8601")
    elif "date" in spx.columns:
        spx["date"] = pd.to_datetime(spx["date"])
    else:
        raise ValueError("SPX 文件缺少 date/time 列")

    # DIX
    if "date" not in dix.columns:
        raise ValueError("DIX 文件缺少 date 列")
    dix["date"] = pd.to_datetime(dix["date"])

    # VIX（与 SPX 相同格式：date + close）
    if "time" in vix.columns:
        vix["date"] = pd.to_datetime(vix["time"], format="ISO8601")
    elif "date" in vix.columns:
        vix["date"] = pd.to_datetime(vix["date"])
    else:
        raise ValueError("VIX 文件缺少 date/time 列")
    if "close" not in vix.columns:
        raise ValueError("VIX 文件缺少 close 列")
    vix = vix[["date", "close"]].rename(columns={"close": "vix_close"})

    # ---- 合并 ----
    df = spx.merge(dix, on="date", how="inner")
    df = df.merge(events, on="date", how="left")
    df = df.merge(vix, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # ====================== 构造标签 ======================

    # 下一交易日
    df["target_date"] = df["date"].shift(-1)

    df["next_open"] = df["open"].shift(-1)
    df["next_high"] = df["high"].shift(-1)
    df["next_low"] = df["low"].shift(-1)
    df["next_close"] = df["close"].shift(-1)

    # 日内振幅 & regime
    df["next_range"] = (df["next_high"] - df["next_low"]) / df["close"]
    df["real_regime"] = df["next_range"].apply(classify_range)

    # next day 收益 & 方向
    df["next_ret"] = df["next_close"] / df["close"] - 1.0
    df["real_direction"] = df["next_ret"].apply(direction_label)

    # 趋势日（body 占比较大）
    df["real_trend"] = (
        (df["next_close"] - df["next_open"]).abs() / df["next_open"] > 0.006
    ).astype(float)
    df.loc[df["next_open"].isna() | df["next_close"].isna(), "real_trend"] = np.nan

    # 反转日：gap one way, close the other way
    gap_up = df["next_open"] > df["close"]
    gap_down = df["next_open"] < df["close"]
    close_red = df["next_close"] < df["next_open"]
    close_green = df["next_close"] > df["next_open"]
    df["real_reversal"] = ((gap_up & close_red) | (gap_down & close_green)).astype(float)
    df.loc[df["next_open"].isna() | df["next_close"].isna(), "real_reversal"] = np.nan

    # ====================== 基础价格特征 ======================

    df["ret"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]

    df["dow"] = df["date"].dt.weekday
    df["moy"] = df["date"].dt.month

    # ====================== GEX 特征（精简版） ======================

    # 假设已有列 gex
    if "gex" not in df.columns:
        raise ValueError("合并后的 df 中找不到 'gex' 列，请检查 DIX/GEX 数据源")

    df["gex_abs"] = df["gex"].abs()
    df["gex_scaled"] = df["gex_abs"] / 1e9
    df["log_gex_abs"] = np.log(df["gex_abs"].replace(0, np.nan))

    # GEX 分位数（绝对值，252 天）
    df["gex_abs_pct_252"] = (
        df["gex_scaled"]
        .rolling(252, min_periods=60)
        .apply(lambda x: (x <= x.iloc[-1]).mean(), raw=False)
    )

    # 60 天分位数
    df["gex_pct_60"] = (
        df["gex_scaled"]
        .rolling(60, min_periods=20)
        .apply(lambda x: (x <= x.iloc[-1]).mean(), raw=False)
    )

    # 120 天 z-score
    gex_roll_120 = df["gex_scaled"].rolling(120)
    df["gex_scaled_z120"] = (df["gex_scaled"] - gex_roll_120.mean()) / gex_roll_120.std()

    # 5 日变化 z-score（捕捉 regime shift）
    df["gex_5d_chg"] = df["gex"].diff(5)
    gex5_roll_60 = df["gex_5d_chg"].rolling(60)
    df["gex_5d_chg_z60"] = (df["gex_5d_chg"] - gex5_roll_60.mean()) / gex5_roll_60.std()

    # ====================== DIX 特征（精简版） ======================

    # 60 天分位数
    dix_roll_60 = df["dix"].rolling(60)
    df["dix_pct_60"] = dix_roll_60.rank(pct=True)

    # 变化率
    df["dix_1d_chg"] = df["dix"].diff(1)
    df["dix_5d_chg"] = df["dix"].diff(5)

    dix5_roll_60 = df["dix_5d_chg"].rolling(60)
    df["dix_5d_chg_z60"] = (df["dix_5d_chg"] - dix5_roll_60.mean()) / dix5_roll_60.std()

    # ====================== VIX 特征（精简版） ======================

    df["vix_level"] = df["vix_close"]

    df["vix_5d_chg"] = df["vix_level"].diff(5)
    vix5_roll_60 = df["vix_5d_chg"].rolling(60)
    df["vix_5d_chg_z60"] = (df["vix_5d_chg"] - vix5_roll_60.mean()) / vix5_roll_60.std()

    # realized vol（20 日，年化）
    df["realized_vol_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["vix_iv_annual"] = df["vix_level"] / 100.0
    df["vix_vs_realized_20"] = df["vix_iv_annual"] - df["realized_vol_20"]

    # ====================== 事件特征（只保留布尔标记） ======================

    event_cols_keep = [
        "is_fomc",
        "is_fomc_window",
        "is_vix_expiry",
        "is_spy_opex_m",
    ]
    for c in event_cols_keep:
        if c not in df.columns:
            df[c] = 0

    # 其他 days_to_* 列可以忽略，不参与特征

    # ====================== 构造 lag 特征（v2.1 精简版） ======================

    # 为每一个 base 特征指定要保留的 lag 列
    lag_plan = {
        # price
        "ret": [0, 1],
        "range": [0, 1, 2],

        # GEX
        "log_gex_abs": [0, 1],
        "gex_abs_pct_252": [0, 1],
        "gex_scaled_z120": [0, 1],
        "gex_5d_chg_z60": [0],
        "gex_pct_60": [0, 1],

        # DIX
        "dix_5d_chg_z60": [0],
        "dix_pct_60": [0],
        "dix_1d_chg": [0],

        # VIX
        "vix_level": [0, 1],
        "vix_5d_chg_z60": [0],
        "vix_vs_realized_20": [0],
    }

    # 实际创建 lag 列
    for col, lags in lag_plan.items():
        if col not in df.columns:
            raise ValueError(f"预期中的基准特征列 {col} 不存在，请检查前面的构造步骤。")
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # -------- 公共：先把 inf 换成 NaN --------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 标签列（这些在预测时应该允许为 NaN）
    label_cols = [
        "real_regime",
        "real_direction",
        "real_trend",
        "real_reversal",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
        "next_ret",
        "next_range",
        "target_date",
    ]

    if for_training:
        # 训练：只用有 target_date（= 有 next day）的样本
        df = df.dropna(subset=["target_date"])
        # 所有剩余 NaN 统一填 0（对 RF OK）
        df = df.fillna(0)
    else:
        # 预测：保留最后一天（target_date 允许为 NaN）
        # 只对“特征列”填 0，标签保持 NaN
        feature_cols = [c for c in df.columns if c not in label_cols]
        df[feature_cols] = df[feature_cols].fillna(0)

    return df
