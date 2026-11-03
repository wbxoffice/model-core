#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest spread + strangle strategy (v3, dynamic range + IV*sqrt(T) based offsets, auto-discover log files)

关键点：
- VIX 数据从 config.VIX_FILE 指定的 CBOE_DLY_VIX.csv 读取
  (结构: time, open, high, low, close, Volume)
- 用 VIX_open -> IV*sqrt(T) 计算 1日σ价格波动，再按 σ 级别确定价外程度：
    * 高波 long strangle: offset ≈ 0.53σ · S_open
    * 趋势 long spread:   offset ≈ 0.13σ · S_open
- 如果某天 VIX 无法匹配，就不做高波 strangle (标记为 HIGHVOL_NO_TRADE_NO_VIX)
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

try:
    from shared.local_settings import account_equity
except ImportError:
    from local_settings import account_equity


# ===================== 全局路径 & 基本配置 =====================

# 共享实现（2026-06-12 去重）：原 spx_vdtr_v2 / us500_vdtr_v2 各持有一份逐字节相同的副本。
# 路径不再在模块级解析，由各项目薄包装通过 configure()/main(base_dir=...) 注入。
BASE_DIR = None
BACKTEST_DIR = None
VIX_PATH = None

# 初始资金
EQUITY0 = account_equity()  # 初始资金：取 local_settings.json（私有，已 gitignore）

# 区间风格: "aggressive" / "balanced" / "conservative"
RANGE_MODE = "balanced"

# 在预测上下限前 offset 多少点视为“提前埋伏触发”
ENTRY_OFFSET_PTS = 5.0

# 默认回测起止日期 (字符串 "YYYY-MM-DD" 或 None)
DEFAULT_START_DATE: str | None = "2024-07-01"   # 例如 "2024-07-01"
DEFAULT_END_DATE: str | None = "2025-12-05"     # 例如 "2025-12-05"

# 输出总目录（configure() 中初始化）
OUT_ROOT = None

# 本策略版本标签 (用于子目录与文件命名)
RUN_LABEL = "strategy_v1"

# ===================== BSM 风格参数（IV * sqrt(T)） =====================

# 假设 horizon 为 1 个交易日
T_DAYS = 1.0
T_YEAR = T_DAYS / 252.0

# 根据你的例子校准：
# S≈6875, VIX=17.3, 1日1σ≈75点, 想要 offset≈40 → SIGMA_MOVE_STRANGLE≈40/75≈0.53
SIGMA_MOVE_STRANGLE = 0.53  # 用于高波 long strangle 的“σ级别”价外程度

# 趋势 spread：以前固定 10 点，同一例子 10/75≈0.13
SIGMA_MOVE_TREND = 0.13     # 用于 trend spread 的“σ级别”价外程度

# 合理的点数上下限避免极端 VIX 时数值过大/过小
STRANGLE_OFFSET_MIN = 10.0
STRANGLE_OFFSET_MAX = 200.0
TREND_OFFSET_MIN = 3.0
TREND_OFFSET_MAX = 40.0


# ===================== 自动发现 prediction_log 文件 =====================

def discover_log_files() -> dict[str, Path]:
    patterns = ["backtest_walkforward_*.xlsx", "backtest_walkforward_*.xls", "backtest_walkforward_*.csv"]
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(BACKTEST_DIR.glob(pat))

    log_files: dict[str, Path] = {}
    for p in sorted(paths):
        stem = p.stem
        prefix = "backtest_walkforward_"
        if stem.startswith(prefix):
            scheme = stem[len(prefix):]
        else:
            scheme = stem
        log_files[scheme] = p

    if not log_files:
        print(f"[WARN] 在 {BACKTEST_DIR} 下未找到 backtest_walkforward_*.xlsx/csv 文件.")
    else:
        print(f"[INFO] 自动发现 {len(log_files)} 个 walk-forward prediction_log 文件:")
        for name, path in log_files.items():
            print(f"  - scheme={name}, file={path}")

    return log_files


EXPLICIT_LOG_FILES: dict[str, Path] = {
    # 如需手动指定, 在这里填:
    # "v21_rolling3y": BACKTEST_DIR / "backtest_walkforward_v21_rolling3y_2020-01-01_to_2025-12-05_Rolling3y.xlsx",
}


# ===================== 动态 range (与 meta v3 思路一致) =====================

# 分位数加载与动态 range 复用策略引擎的唯一实现（2026-06-12 收敛，原各持一份等价副本）
try:
    from shared.strategy_engine_v3 import (
        load_range_quantiles as _engine_load_range_quantiles,
        dynamic_daily_range as _engine_dynamic_daily_range,
    )
except ImportError:  # 兼容直接从 shared/ 目录以脚本方式运行
    from strategy_engine_v3 import (
        load_range_quantiles as _engine_load_range_quantiles,
        dynamic_daily_range as _engine_dynamic_daily_range,
    )


def load_range_quantiles() -> dict[int, dict[str, float]]:
    return _engine_load_range_quantiles([
        BASE_DIR / "range_quantiles_by_regime.json",
        BASE_DIR.parent / "range_quantiles_by_regime.json",
    ])


RANGE_QUANTILES = None  # configure() 中加载


def configure(base_dir, backtest_dir=None, vix_path=None):
    """绑定项目目录并初始化路径与分位数（替代原模块级解析）。"""
    global BASE_DIR, BACKTEST_DIR, VIX_PATH, OUT_ROOT, RANGE_QUANTILES
    BASE_DIR = Path(base_dir)
    BACKTEST_DIR = Path(backtest_dir) if backtest_dir is not None else BASE_DIR / "backtest"
    VIX_PATH = Path(vix_path) if vix_path is not None else BASE_DIR / "data" / "CBOE_DLY_VIX.csv"
    OUT_ROOT = BASE_DIR / "STRATEGY PnL backtest"
    OUT_ROOT.mkdir(exist_ok=True)
    RANGE_QUANTILES = load_range_quantiles()


def dynamic_daily_range(vol_regime: int, p0: float, p1: float, p2: float, spx_price: float) -> dict[str, float]:
    """委托策略引擎的同名实现（逻辑逐行一致，已 diff 验证）。"""
    return _engine_dynamic_daily_range(vol_regime, p0, p1, p2, spx_price, RANGE_QUANTILES)


# ===================== 基础工具 & PnL 函数 =====================

def nearest_strike(price: float, step: float = 5.0) -> float:
    return round(price / step) * step


def pnl_short_call_spread_10(S_close: float, K1: float, credit: float = 7.0) -> float:
    intrinsic1 = max(S_close - K1, 0.0)
    intrinsic2 = max(S_close - (K1 + 10.0), 0.0)
    payoff = -(intrinsic1 - intrinsic2)
    return payoff + credit  # 单份, 点数


def pnl_short_put_spread_10(S_close: float, K1: float, credit: float = 7.0) -> float:
    intrinsic1 = max(K1 - S_close, 0.0)
    intrinsic2 = max((K1 - 10.0) - S_close, 0.0)
    payoff = -(intrinsic1 - intrinsic2)
    return payoff + credit


def pnl_long_call_spread_10(S_close: float, K1: float, debit: float = 2.7) -> float:
    intrinsic1 = max(S_close - K1, 0.0)
    intrinsic2 = max(S_close - (K1 + 10.0), 0.0)
    payoff = intrinsic1 - intrinsic2
    return payoff - debit


def pnl_long_put_spread_10(S_close: float, K1: float, debit: float = 2.7) -> float:
    intrinsic1 = max(K1 - S_close, 0.0)
    intrinsic2 = max((K1 - 10.0) - S_close, 0.0)
    payoff = intrinsic1 - intrinsic2
    return payoff - debit


def pnl_long_strangle(S_close: float, K_call: float, K_put: float,
                      call_debit: float = 3.0, put_debit: float = 3.0) -> float:
    ic = max(S_close - K_call, 0.0)
    ip = max(K_put - S_close, 0.0)
    payoff = ic + ip
    return payoff - (call_debit + put_debit)


# ===================== 仓位管理 =====================

MAX_LOSS_POINTS = {
    "LVL_RANGE_SHORT_CALL_SPREAD": 3.0,
    "LVL_RANGE_SHORT_PUT_SPREAD": 3.0,
    "LOWVOL_TREND_LONG_CALL_SPREAD": 2.7,
    "LOWVOL_TREND_LONG_PUT_SPREAD": 2.7,
    "MIDVOL_TREND_LONG_CALL_SPREAD": 2.7,
    "MIDVOL_TREND_LONG_PUT_SPREAD": 2.7,
    "HIGHVOL_LONG_STRANGLE": 6.0,   # 3+3 点 = 600 美金最大亏损
}

RISK_PCT_CONFIG = {
    0: {
        "LVL_RANGE_SHORT_CALL_SPREAD": 0.02,
        "LVL_RANGE_SHORT_PUT_SPREAD": 0.02,
        "LOWVOL_TREND_LONG_CALL_SPREAD": 0.02,
        "LOWVOL_TREND_LONG_PUT_SPREAD": 0.02,
    },
    1: {
        "MIDVOL_TREND_LONG_CALL_SPREAD": 0.03,
        "MIDVOL_TREND_LONG_PUT_SPREAD": 0.03,
    },
    2: {
        "HIGHVOL_LONG_STRANGLE": 0.04,
    },
}
# 每种策略类型的最大合约张数上限（实盘可行性约束）
MAX_CONTRACTS = {
    # credit spread（低波区间卖出）
    "LVL_RANGE_SHORT_CALL_SPREAD": 5,
    "LVL_RANGE_SHORT_PUT_SPREAD": 5,

    # trend long spread（低/中波趋势方向单）
    "LOWVOL_TREND_LONG_CALL_SPREAD": 10,
    "LOWVOL_TREND_LONG_PUT_SPREAD": 10,
    "MIDVOL_TREND_LONG_CALL_SPREAD": 10,
    "MIDVOL_TREND_LONG_PUT_SPREAD": 10,

    # high vol strangle（高波 day-long strangle）
    "HIGHVOL_LONG_STRANGLE": 5,
}

def compute_n_contracts(vol_regime: int, trade_type: str, equity: float):
    """
    根据账户权益 + 风险百分比 + 单份最大亏损 -> 计算理论张数，
    然后再根据 MAX_CONTRACTS 做实盘上限截断。
    """
    risk_pct = RISK_PCT_CONFIG.get(vol_regime, {}).get(trade_type, 0.0)
    if risk_pct <= 0.0:
        return 0, 0.0, 0.0

    risk_cap = equity * risk_pct
    max_loss_pts = MAX_LOSS_POINTS.get(trade_type, 0.0)
    if max_loss_pts <= 0.0:
        return 0, 0.0, 0.0

    # 理论上按资金计算的张数
    max_loss_dollar = max_loss_pts * 100.0
    n_contracts = int(np.floor(risk_cap / max_loss_dollar))
    if n_contracts < 0:
        n_contracts = 0

    # 加上“实盘可行性上限”
    cap = MAX_CONTRACTS.get(trade_type, None)
    if cap is not None:
        n_contracts = min(n_contracts, cap)

    return n_contracts, risk_pct, risk_cap


# ===================== VIX 读取（从 config.VIX_FILE 指向的 CBOE_DLY_VIX.csv） =====================

def load_vix() -> pd.DataFrame:
    """
    从 CBOE_DLY_VIX.csv 读取 VIX，并返回标准格式:
    date (normalized), vix_open

    支持的列:
    - 日期列: time / date / trade_date 等 (你目前是 time)
    - 开盘列: open (你目前是 open)
    """
    if not VIX_PATH.exists():
        print("[ERROR] VIX 文件不存在:", VIX_PATH)
        return pd.DataFrame(columns=["date", "vix_open"])

    vix = pd.read_csv(VIX_PATH)

    # 日期列: 优先 time, 其次 date / trade_date
    date_col_candidates = [c for c in vix.columns if c.lower() in ("time", "date", "trade_date")]
    if not date_col_candidates:
        raise ValueError(f"VIX 文件中找不到日期列 (time/date/trade_date), 现有列: {list(vix.columns)}")

    date_col = date_col_candidates[0]
    vix["date"] = pd.to_datetime(vix[date_col]).dt.normalize()

    # open 列: open / vix_open 等
    open_col_candidates = [c for c in vix.columns if c.lower() in ("open", "vix_open")]
    if open_col_candidates:
        open_col = open_col_candidates[0]
    else:
        # 没有 open 就用 close 或最后一列兜底
        if "close" in vix.columns:
            open_col = "close"
        else:
            open_col = vix.columns[-1]

    vix = vix.rename(columns={open_col: "vix_open"})

    out = vix[["date", "vix_open"]].sort_values("date").reset_index(drop=True)
    print(f"[INFO] 读取 VIX 成功: {len(out)} 行, 日期范围 {out['date'].min().date()} ~ {out['date'].max().date()}")
    return out


# ===================== flag 决策函数 =====================

def decide_trend_flag(p_trend: float, threshold: float = 0.5) -> int:
    if pd.isna(p_trend):
        return 0
    return 1 if p_trend >= threshold else 0


def decide_dir_flag(p_up: float, p_down: float, threshold: float = 0.5) -> int:
    if pd.isna(p_up) or pd.isna(p_down):
        return 0
    if p_up > threshold and p_up > p_down:
        return 1
    if p_down > threshold and p_down > p_up:
        return -1
    return 0


# ===================== 单 scheme 回测 =====================

GLOBAL_START_DATE: pd.Timestamp | None = None
GLOBAL_END_DATE: pd.Timestamp | None = None


def backtest_one_scheme(
    scheme_name: str,
    log_path: Path,
    vix_df: pd.DataFrame,
) -> pd.DataFrame | None:
    print(f"\n========== Backtest scheme: {scheme_name} ==========")
    if not log_path.exists():
        print(f"[WARN] prediction_log not found: {log_path}, skip this scheme.")
        return None

    # 读取 prediction_log：支持 csv / xlsx(sheet='prediction_log')
    if log_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(log_path, sheet_name="prediction_log")
    else:
        df = pd.read_csv(log_path)

    if "date" not in df.columns:
        raise ValueError("prediction_log 中缺少 'date' 列")

    # 标准化日期为纯日 (避免时区/时刻导致 merge miss)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if GLOBAL_START_DATE is not None:
        df = df[df["date"] >= GLOBAL_START_DATE]
    if GLOBAL_END_DATE is not None:
        df = df[df["date"] <= GLOBAL_END_DATE]

    if df.empty:
        print(f"[WARN] scheme={scheme_name} 在指定时间范围内没有数据.")
        return None

    for col in [
        "spx_open",
        "spx_high",
        "spx_low",
        "spx_close",
        "spx_close_src",
        "dir_P_up",
        "dir_P_down",
        "trend_P_trend",
        "vol_P_regime0",
        "vol_P_regime1",
        "vol_P_regime2",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "vol_pred_regime" not in df.columns:
        raise ValueError("prediction_log 中缺少 vol_pred_regime 列")

    # 合并 VIX
    if not vix_df.empty:
        df = df.merge(vix_df, on="date", how="left")
    else:
        df["vix_open"] = np.nan

    # 检查 VIX 匹配覆盖率
    vix_coverage = df["vix_open"].notna().mean()
    print(f"[INFO] scheme={scheme_name} VIX 覆盖率: {vix_coverage*100:.1f}%")

    df = df.sort_values("date").reset_index(drop=True)

    equity = EQUITY0
    records: list[dict] = []

    for _, row in df.iterrows():
        date = row["date"]

        S_open = row.get("spx_open", np.nan)
        S_high = row.get("spx_high", np.nan)
        S_low = row.get("spx_low", np.nan)
        S_close = row.get("spx_close", np.nan)
        prev_close = row.get("spx_close_src", np.nan)

        # 预设输出字段
        pred_low = np.nan
        pred_high = np.nan
        range_low = np.nan
        range_base = np.nan
        range_high = np.nan
        K1 = np.nan
        K2 = np.nan
        K_call = np.nan
        K_put = np.nan

        if any(pd.isna(x) for x in [S_open, S_high, S_low, S_close, prev_close]):
            records.append({
                "scheme": scheme_name,
                "date": date,
                "vol_pred_regime": np.nan,
                "trend_flag": np.nan,
                "dir_flag": np.nan,
                "vix_open": row.get("vix_open", np.nan),
                "trade_type": "NO_TRADE_BADPRICE",
                "n_contracts": 0,
                "pnl": 0.0,
                "equity": equity,
                "risk_pct": 0.0,
                "risk_cap": 0.0,
                "S_open": S_open,
                "S_high": S_high,
                "S_low": S_low,
                "S_close": S_close,
                "prev_close": prev_close,
                "range_mode": RANGE_MODE,
                "entry_offset_pts": ENTRY_OFFSET_PTS,
                "range_low": range_low,
                "range_base": range_base,
                "range_high": range_high,
                "pred_low": pred_low,
                "pred_high": pred_high,
                "K1": K1,
                "K2": K2,
                "K_call": K_call,
                "K_put": K_put,
            })
            continue

        vol_reg = int(row.get("vol_pred_regime", 0))
        p0 = float(row.get("vol_P_regime0", 0.0))
        p1 = float(row.get("vol_P_regime1", 0.0))
        p2 = float(row.get("vol_P_regime2", 0.0))

        trend_p_trend = float(row.get("trend_P_trend", 0.0))
        dir_p_up = float(row.get("dir_P_up", 0.0))
        dir_p_down = float(row.get("dir_P_down", 0.0))
        trend_flag = decide_trend_flag(trend_p_trend, threshold=0.5)
        dir_flag = decide_dir_flag(dir_p_up, dir_p_down, threshold=0.5)

        vix_open = float(row.get("vix_open", np.nan))

        trade_type = "NO_TRADE"
        n_contracts = 0
        pnl_dollar = 0.0
        risk_pct_used = 0.0
        risk_cap_used = 0.0

        # 计算 IV*sqrt(T) 对应的一天 σ move（用于 trend & strangle）
        if not np.isnan(vix_open) and vix_open > 0:
            sigma_sqrtT = (vix_open / 100.0) * np.sqrt(T_YEAR)
            one_sigma_move = S_open * sigma_sqrtT
        else:
            sigma_sqrtT = np.nan
            one_sigma_move = np.nan

        # ---------- 低波 ----------
        if vol_reg == 0:
            range_info = dynamic_daily_range(vol_reg, p0, p1, p2, spx_price=prev_close)
            range_low = range_info["range_low"]
            range_base = range_info["range_base"]
            range_high = range_info["range_high"]

            if RANGE_MODE == "aggressive":
                D = range_low
            elif RANGE_MODE == "conservative":
                D = range_high
            else:
                D = range_base

            pred_low = prev_close - D
            pred_high = prev_close + D

            if trend_flag == 0:
                # 低波 + 无趋势: 提前埋伏卖 credit spread
                hit_call = S_high >= (pred_high - ENTRY_OFFSET_PTS)
                hit_put = S_low <= (pred_low + ENTRY_OFFSET_PTS)

                side: str | None = None
                if hit_call and not hit_put:
                    side = "CALL"
                elif hit_put and not hit_call:
                    side = "PUT"
                elif hit_call and hit_put:
                    if dir_flag == 1:
                        side = "PUT"
                    elif dir_flag == -1:
                        side = "CALL"
                    else:
                        side = None

                if side == "CALL":
                    trade_type = "LVL_RANGE_SHORT_CALL_SPREAD"
                    n_contracts, risk_pct_used, risk_cap_used = compute_n_contracts(vol_reg, trade_type, equity)
                    if n_contracts > 0:
                        K1 = nearest_strike(pred_high, step=5.0)
                        K2 = K1 + 10.0
                        pnl_single_pts = pnl_short_call_spread_10(S_close, K1, credit=7.0)
                        pnl_dollar = n_contracts * pnl_single_pts * 100.0
                elif side == "PUT":
                    trade_type = "LVL_RANGE_SHORT_PUT_SPREAD"
                    n_contracts, risk_pct_used, risk_cap_used = compute_n_contracts(vol_reg, trade_type, equity)
                    if n_contracts > 0:
                        K1 = nearest_strike(pred_low, step=5.0)
                        K2 = K1 - 10.0
                        pnl_single_pts = pnl_short_put_spread_10(S_close, K1, credit=7.0)
                        pnl_dollar = n_contracts * pnl_single_pts * 100.0

            else:
                # 低波 + 有趋势: long spread，价外程度按 σ move 调整
                if not np.isnan(one_sigma_move) and one_sigma_move > 0:
                    offset_trend = SIGMA_MOVE_TREND * one_sigma_move
                    offset_trend = float(np.clip(offset_trend, TREND_OFFSET_MIN, TREND_OFFSET_MAX))
                else:
                    offset_trend = TREND_OFFSET_MIN

                if dir_flag == 1:
                    trade_type = "LOWVOL_TREND_LONG_CALL_SPREAD"
                    n_contracts, risk_pct_used, risk_cap_used = compute_n_contracts(vol_reg, trade_type, equity)
                    if n_contracts > 0:
                        K1 = nearest_strike(S_open + offset_trend, step=5.0)
                        K2 = K1 + 10.0
                        pnl_single_pts = pnl_long_call_spread_10(S_close, K1, debit=2.7)
                        pnl_dollar = n_contracts * pnl_single_pts * 100.0
                elif dir_flag == -1:
                    trade_type = "LOWVOL_TREND_LONG_PUT_SPREAD"
                    n_contracts, risk_pct_used, risk_cap_used = compute_n_contracts(vol_reg, trade_type, equity)
                    if n_contracts > 0:
                        K1 = nearest_strike(S_open - offset_trend, step=5.0)
                        K2 = K1 - 10.0
                        pnl_single_pts = pnl_long_put_spread_10(S_close, K1, debit=2.7)
                        pnl_dollar = n_contracts * pnl_single_pts * 100.0

        # ---------- 中波 ----------
        elif vol_reg == 1:
            range_info = dynamic_daily_range(vol_reg, p0, p1, p2, spx_price=prev_close)
            range_low = range_info["range_low"]
            range_base = range_info["range_base"]
            range_high = range_info["range_high"]
            pred_low = prev_close - range_base
            pred_high = prev_close + range_base

            if trend_flag == 1:
                if not np.isnan(one_sigma_move) and one_sigma_move > 0:
                    offset_trend = SIGMA_MOVE_TREND * one_sigma_move
                    offset_trend = float(np.clip(offset_trend, TREND_OFFSET_MIN, TREND_OFFSET_MAX))
                else:
                    offset_trend = TREND_OFFSET_MIN

                if dir_flag == 1:
                    trade_type = "MIDVOL_TREND_LONG_CALL_SPREAD"
                    n_contracts, risk_pct_used, risk_cap_used = compute_n_contracts(vol_reg, trade_type, equity)
                    if n_contracts > 0:
                        K1 = nearest_strike(S_open + offset_trend, step=5.0)
                        K2 = K1 + 10.0
                        pnl_single_pts = pnl_long_call_spread_10(S_close, K1, debit=2.7)
                        pnl_dollar = n_contracts * pnl_single_pts * 100.0
                elif dir_flag == -1:
                    trade_type = "MIDVOL_TREND_LONG_PUT_SPREAD"
                    n_contracts, risk_pct_used, risk_cap_used = compute_n_contracts(vol_reg, trade_type, equity)
                    if n_contracts > 0:
                        K1 = nearest_strike(S_open - offset_trend, step=5.0)
                        K2 = K1 - 10.0
                        pnl_single_pts = pnl_long_put_spread_10(S_close, K1, debit=2.7)
                        pnl_dollar = n_contracts * pnl_single_pts * 100.0
            else:
                trade_type = "MIDVOL_NOTREND_NO_TRADE"
                pnl_dollar = 0.0

        # ---------- 高波 ----------
        elif vol_reg == 2:
            # 高波: 必须有 VIX; 没有 VIX 就不做交易
            if not np.isnan(one_sigma_move) and one_sigma_move > 0:
                offset_str = SIGMA_MOVE_STRANGLE * one_sigma_move
                offset_str = float(np.clip(offset_str, STRANGLE_OFFSET_MIN, STRANGLE_OFFSET_MAX))

                range_base = offset_str  # 作为参考
                pred_low = S_open - offset_str
                pred_high = S_open + offset_str

                trade_type = "HIGHVOL_LONG_STRANGLE"
                n_contracts, risk_pct_used, risk_cap_used = compute_n_contracts(vol_reg, trade_type, equity)
                if n_contracts > 0:
                    K_call = nearest_strike(S_open + offset_str, step=5.0)
                    K_put = nearest_strike(S_open - offset_str, step=5.0)
                    pnl_single_pts = pnl_long_strangle(S_close, K_call, K_put, call_debit=3.0, put_debit=3.0)
                    pnl_dollar = n_contracts * pnl_single_pts * 100.0
            else:
                trade_type = "HIGHVOL_NO_TRADE_NO_VIX"
                pnl_dollar = 0.0

        # 更新 equity
        equity += pnl_dollar

        records.append({
            "scheme": scheme_name,
            "date": date,
            "vol_pred_regime": vol_reg,
            "trend_flag": trend_flag,
            "dir_flag": dir_flag,
            "vix_open": vix_open,
            "trade_type": trade_type,
            "n_contracts": n_contracts,
            "pnl": pnl_dollar,
            "equity": equity,
            "risk_pct": risk_pct_used,
            "risk_cap": risk_cap_used,
            "S_open": S_open,
            "S_high": S_high,
            "S_low": S_low,
            "S_close": S_close,
            "prev_close": prev_close,
            "range_mode": RANGE_MODE,
            "entry_offset_pts": ENTRY_OFFSET_PTS,
            "range_low": range_low,
            "range_base": range_base,
            "range_high": range_high,
            "pred_low": pred_low,
            "pred_high": pred_high,
            "K1": K1,
            "K2": K2,
            "K_call": K_call,
            "K_put": K_put,
        })

    res = pd.DataFrame(records)
    if res.empty:
        print(f"[WARN] scheme {scheme_name} 结果为空.")
        return None

    print(f"[DONE] scheme={scheme_name}, equity0={EQUITY0:.2f}, final_equity={equity:.2f}, totalPnL={equity-EQUITY0:.2f}")
    print("交易日数:", len(res), "有交易天数:", (res['trade_type'] != 'NO_TRADE').sum())
    return res


# ===================== main =====================

def main(start: str | None = None, end: str | None = None,
         base_dir=None, backtest_dir=None, vix_path=None):
    global GLOBAL_START_DATE, GLOBAL_END_DATE

    if base_dir is not None:
        configure(base_dir, backtest_dir, vix_path)
    if BASE_DIR is None:
        raise RuntimeError("backtest_spread_strangle_v1: 需先 configure(...) 或 main(base_dir=...)")

    use_start = start or DEFAULT_START_DATE
    use_end = end or DEFAULT_END_DATE

    if use_start:
        GLOBAL_START_DATE = pd.Timestamp(use_start).normalize()
        print("[INFO] 回测起始日期:", GLOBAL_START_DATE.date())
    if use_end:
        GLOBAL_END_DATE = pd.Timestamp(use_end).normalize()
        print("[INFO] 回测结束日期:", GLOBAL_END_DATE.date())

    if EXPLICIT_LOG_FILES:
        log_files = EXPLICIT_LOG_FILES
        print(f"[INFO] 使用手动指定的 {len(log_files)} 个 LOG_FILES.")
    else:
        log_files = discover_log_files()

    if not log_files:
        print("[ERROR] 未找到任何 prediction_log 文件, 终止回测.")
        return

    vix_df = load_vix()

    all_results: list[pd.DataFrame] = []
    scheme_results: dict[str, pd.DataFrame] = {}

    for scheme, path in log_files.items():
        res = backtest_one_scheme(scheme, path, vix_df)
        if res is not None and not res.empty:
            all_results.append(res)
            scheme_results[scheme] = res

    if not all_results:
        print("[ERROR] 所有 scheme 回测结果均为空, 终止写文件.")
        return

    merged = pd.concat(all_results, ignore_index=True)
    d0 = pd.to_datetime(merged["date"]).min()
    d1 = pd.to_datetime(merged["date"]).max()
    range_suffix = f"{d0.strftime('%Y%m%d')}_{d1.strftime('%Y%m%d')}"

    run_dir = OUT_ROOT / f"{RUN_LABEL}_{range_suffix}"
    run_dir.mkdir(exist_ok=True)
    print("[INFO] 本次回测所有 csv 输出目录:", run_dir)

    for scheme, df_res in scheme_results.items():
        out_name = f"{RUN_LABEL}_{scheme}_{range_suffix}.csv"
        out_path = run_dir / out_name
        df_res.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("  已输出 scheme 结果:", out_path)

    all_name = f"{RUN_LABEL}_ALL_SCHEMES_{range_suffix}.csv"
    all_path = run_dir / all_name
    merged.to_csv(all_path, index=False, encoding="utf-8-sig")
    print("\n[SUMMARY] 所有 scheme 合并结果已输出:", all_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest spread+strangle strategy with dynamic range & IV*sqrt(T) based offsets (auto-discover log files)."
    )
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD (包含)")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD (包含)")
    args = parser.parse_args()

    main(args.start, args.end)
