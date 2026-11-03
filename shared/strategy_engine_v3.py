#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Daily Meta Strategy Engine for SPX 0DTE  (v3 — 动态日内波动范围 + strategy.txt 输出)

共享唯一实现（2026-06-12 去重）：
- 原先 aio_v1 / aio_v2 / spx_vdtr_v2 / us500_vdtr_v2 各持有一份副本，
  逻辑完全一致，唯一差异是 strategy.txt 中的数据源横幅（DATA BASED ON ...）。
- 现统一收敛到本文件；各子项目目录内保留同名薄包装脚本
  daily_meta_strategy_engine_v3.py，负责解析项目本地 config 路径并传参调用。
- 原副本归档：History/strategy_engine_dedup_backup_2026-06-12/
- vdtr_path_v1 的同名脚本是路径模型专用的不同实现，不属于本共享层。

核心功能：
- 从 prediction_log.csv 读取最新一行预测
- 从 range_quantiles_by_regime.json 读取按 real_regime 标定的 next_range 分位数（q25/q50/q75）
- 根据 vol_pred_regime + vol_P_regime0/1/2 动态估计当日日内 range（点数）
- 根据 vol_regime + trend_flag + dir_flag 决定 meta trade_type
- 根据回测统计 (p_win, payoff_b) + Kelly + 全局风险上限，给出推荐风险敞口与张数
- 自动在 log_dir 下写入当日策略文件 <date>_strategy.txt

接口约定（薄包装必须传入）：
- log_path        : prediction_log.csv 路径
- log_dir         : strategy.txt 输出目录
- vix_path        : VIX CSV 路径（不存在时 VIX 走 fallback）
- quantile_paths  : range_quantiles_by_regime.json 搜索路径列表（按序取第一个存在的）
- data_banner     : strategy.txt 第二行横幅（"SPX"/"US500"/None=不写，保持 aio 原行为）
"""

from pathlib import Path
import math
import json
import pandas as pd

try:
    from shared.local_settings import account_equity
except ImportError:  # 作为脚本直接从 shared/ 内运行
    from local_settings import account_equity

# ===================== 策略统计参数（来自回测 v3 汇总） =====================

STRATEGY_STATS = {
    # 低波区间 credit spreads
    "LVL_RANGE_SHORT_PUT_SPREAD": {
        "p_win": 0.5977,
        "payoff_b": 2.24,       # 约等于 3117 / 1389
        "max_loss_dollar": 300, # 对应你设想的 10 点宽、max loss ~300
        "width_points": 10,
    },
    "LVL_RANGE_SHORT_CALL_SPREAD": {
        "p_win": 0.5685,
        "payoff_b": 2.09,       # 约等于 2846 / 1362
        "max_loss_dollar": 300,
        "width_points": 10,
    },

    # 中波趋势 long spread
    "MIDVOL_TREND_LONG_CALL_SPREAD": {
        "p_win": 0.4170,
        "payoff_b": 2.48,        # 1965 / 793
        "max_loss_dollar": 800,  # 约等于你设想的 cost ~270~800 区间；可再调
        "width_points": 10,
    },
    "MIDVOL_TREND_LONG_PUT_SPREAD": {
        # 样本很少，用和 call 差不多的参数
        "p_win": 0.40,
        "payoff_b": 2.5,
        "max_loss_dollar": 800,
        "width_points": 10,
    },

    # 高波 long strangle
    "HIGHVOL_LONG_STRANGLE": {
        "p_win": 0.4511,
        "payoff_b": 5.41,        # 8938 / 1651
        # 一组 strangle 两边 cost ≈ 600；给一点缓冲
        "max_loss_dollar": 700,
        "width_points": 0,
    },

    # （可选）低波趋势 long spread：目前几乎没样本，用作占位
    "LOWVOL_TREND_LONG_CALL_SPREAD": {
        "p_win": 0.50,
        "payoff_b": 2.0,
        "max_loss_dollar": 800,
        "width_points": 10,
    },
}

SPX_MULTIPLIER = 100  # 每点 100 美金

# 账户与风险控制参数
# 账户总资金属于私人信息：实际值写在 local_settings.json（已 gitignore），缺省为占位值。
ACCOUNT_EQUITY = account_equity()
GLOBAL_RISK_CAP = 0.02          # 单日最多风险占比（2%）
KELLY_FRACTION_CAP = 0.5        # 使用 1/2 Kelly 更保守
MIN_NOTIONAL_PER_TRADE = 500.0  # 低于这个则至少 1 手

# BSM 偏移系数（之前回测打印的 K_BSM）
K_BSM_OFFSET = 0.033631
ONE_DAY_T = 1.0 / 252.0


# ===================== 加载 range quantiles JSON =====================

def load_range_quantiles(quantile_paths):
    """
    从 range_quantiles_by_regime.json 读取各 real_regime 的 q25/q50/q75。
    返回 dict:
    { 0: {"q25":..., "q50":..., "q75":..., "count":...}, 1: {...}, 2: {...} }

    注意：这里的 q25/q50/q75 是「next_range 比率」（例如 0.0058 = 0.58%），
    后续会乘以当日 SPX 收盘价转换成「点数」。
    """
    for p in quantile_paths:
        p = Path(p)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                q = {}
                for k, v in data.items():
                    rk = int(k)
                    q[rk] = {
                        "q25": float(v["q25"]),
                        "q50": float(v["q50"]),
                        "q75": float(v["q75"]),
                        "count": int(v.get("count", 0)),
                    }
                print(f"[INFO] 已从 {p} 载入 range quantiles.")
                return q
            except Exception as e:
                print(f"[WARN] 读取 {p} 失败: {e}")

    print("[WARN] 未找到 range_quantiles_by_regime.json，使用默认占位 quantile（比率）.")
    # 占位值：用大致的百分比（低/中/高波）做兜底
    return {
        0: {"q25": 0.0045, "q50": 0.0058, "q75": 0.0070, "count": 0},
        1: {"q25": 0.0093, "q50": 0.0107, "q75": 0.0126, "count": 0},
        2: {"q25": 0.0173, "q50": 0.0208, "q75": 0.0267, "count": 0},
    }


def dynamic_daily_range(vol_regime, p0, p1, p2, spx_price, range_quantiles):
    """
    根据 vol_pred_regime + 对应 regime 概率，动态给出 **点数** 日内 range。

    range_quantiles 中存的是「next_range 比率」（如 0.0058 = 0.58%），
    这里乘以 spx_price 转成「点数」。
    """
    q = range_quantiles.get(vol_regime, None)
    if q is None or spx_price is None or spx_price <= 0:
        return {
            "range_low": 20.0,
            "range_base": 30.0,
            "range_high": 40.0,
            "q_used": "fallback",
            "p_used": None,
            "raw_q25": None,
            "raw_q50": None,
            "raw_q75": None,
        }

    # 取该 regime 对应的概率
    if vol_regime == 0:
        p_regime = float(p0)
    elif vol_regime == 1:
        p_regime = float(p1)
    else:
        p_regime = float(p2)

    # 根据 p_regime 选择 base quantile
    if p_regime >= 0.7:
        base_q = q["q75"]
        which = "q75"
    elif p_regime <= 0.4:
        base_q = q["q25"]
        which = "q25"
    else:
        base_q = q["q50"]
        which = "q50"

    # 比率 × 当前价格 → 点数
    range_low_pts = q["q25"] * spx_price
    range_base_pts = base_q * spx_price
    range_high_pts = q["q75"] * spx_price

    return {
        "range_low": range_low_pts,
        "range_base": range_base_pts,
        "range_high": range_high_pts,
        "q_used": which,
        "p_used": p_regime,
        "raw_q25": q["q25"],
        "raw_q50": q["q50"],
        "raw_q75": q["q75"],
    }


# ===================== 一些基础工具函数 =====================

def nearest_strike(price, step=5):
    if pd.isna(price):
        raise ValueError("nearest_strike: price is NaN, please检查 spx_close 来源.")
    return round(price / step) * step


def kelly_fraction(p_win, b):
    q = 1.0 - p_win
    f = (b * p_win - q) / b
    return f


def per_contract_max_loss(strategy, price=None):
    cfg = STRATEGY_STATS.get(strategy, None)
    if cfg is None:
        return None
    if "max_loss_dollar" in cfg:
        return cfg["max_loss_dollar"]
    width = cfg.get("width_points", 0)
    return max(width, 1) * SPX_MULTIPLIER


def decide_trend_flag(trend_p_trend, threshold=0.5):
    if pd.isna(trend_p_trend):
        return 0
    return int(float(trend_p_trend) > threshold)


def decide_direction(dir_p_up, dir_p_down, margin=0.05):
    try:
        pu = float(dir_p_up)
        pdn = float(dir_p_down)
    except Exception:
        return 0
    diff = pu - pdn
    if diff > margin:
        return 1
    elif diff < -margin:
        return -1
    else:
        return 0


def load_latest_vix(vix_path, src_date=None):
    vix_path = Path(vix_path)
    if not vix_path.exists():
        return None

    df = pd.read_csv(vix_path)
    if "time" not in df.columns:
        return None

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    if src_date is not None:
        try:
            d = pd.to_datetime(src_date).normalize()
            df2 = df[df["time"] <= d]
            if not df2.empty:
                row = df2.iloc[-1]
            else:
                row = df.iloc[-1]
        except Exception:
            row = df.iloc[-1]
    else:
        row = df.iloc[-1]

    for col in ["close", "Close", "vix", "VIX", "vix_close", "VIX Close"]:
        if col in df.columns:
            try:
                return float(row[col])
            except Exception:
                continue

    try:
        return float(row.iloc[-1])
    except Exception:
        return None


# ===================== 策略决策逻辑 =====================

def decide_meta_trade(vol_regime, trend_flag, dir_flag):
    """
    根据 vol_regime + trend_flag + dir_flag 决定「预期策略类型」。
    """
    # 高波：一律 long strangle
    if vol_regime == 2:
        return "HIGHVOL_LONG_STRANGLE", None

    # 中波
    if vol_regime == 1:
        if trend_flag == 0:
            return "NO_TRADE", None
        if dir_flag >= 0:
            return "MIDVOL_TREND_LONG_CALL_SPREAD", "CALL"
        else:
            return "MIDVOL_TREND_LONG_PUT_SPREAD", "PUT"

    # 低波
    if vol_regime == 0:
        if trend_flag == 0:
            # 无趋势：看方向偏好，默认 PUT spread
            if dir_flag > 0:
                return "LVL_RANGE_SHORT_CALL_SPREAD", "CALL"
            elif dir_flag < 0:
                return "LVL_RANGE_SHORT_PUT_SPREAD", "PUT"
            else:
                return "LVL_RANGE_SHORT_PUT_SPREAD", "PUT"
        else:
            # 有趋势：偏多就 call spread；偏空就用中波 put spread 参数
            if dir_flag >= 0:
                return "LOWVOL_TREND_LONG_CALL_SPREAD", "CALL"
            else:
                return "MIDVOL_TREND_LONG_PUT_SPREAD", "PUT"

    return "NO_TRADE", None


def build_structure_strikes(
    trade_type,
    spx_price,
    vol_regime,
    dir_flag,
    range_quantiles,
    vix_value=None,
    p0=0.0,
    p1=0.0,
    p2=0.0,
):
    """
    根据 trade_type + 动态 range 构建预期结构行权价。
    """
    cfg = STRATEGY_STATS.get(trade_type, {})
    width = cfg.get("width_points", 10)

    atm = nearest_strike(spx_price, step=5)

    # 动态日内范围（点数）
    range_info = dynamic_daily_range(vol_regime, p0, p1, p2, spx_price, range_quantiles)
    # 结构用极端范围（q75）来放得远一些
    range_used = range_info["range_high"]

    pred_low_ext = spx_price - range_used
    pred_high_ext = spx_price + range_used

    pred_low_base = spx_price - range_info["range_base"]
    pred_high_base = spx_price + range_info["range_base"]

    result = {
        "type": None,
        "predicted_range_low": round(pred_low_ext, 2),
        "predicted_range_high": round(pred_high_ext, 2),
        "predicted_range_low_base": round(pred_low_base, 2),
        "predicted_range_high_base": round(pred_high_base, 2),
        "range_low_pts": round(range_info["range_low"], 2),
        "range_base_pts": round(range_info["range_base"], 2),
        "range_high_pts": round(range_info["range_high"], 2),
        "range_q_used": range_info["q_used"],
        "range_p_used": range_info["p_used"],
        "raw_q25": range_info["raw_q25"],
        "raw_q50": range_info["raw_q50"],
        "raw_q75": range_info["raw_q75"],
        "vix_used": vix_value,
    }

    # 低波区间：短 put/call spread
    if trade_type in ("LVL_RANGE_SHORT_PUT_SPREAD", "LVL_RANGE_SHORT_CALL_SPREAD"):
        offset_edge = 0  # 触及预测上下边界再往外 10 点
        if trade_type == "LVL_RANGE_SHORT_PUT_SPREAD":
            short_strike = nearest_strike(pred_low_ext - offset_edge, step=5)
            long_strike = short_strike - width
            result.update({
                "type": "short_put_spread",
                "short_put": short_strike,
                "long_put": long_strike,
            })
        else:
            short_strike = nearest_strike(pred_high_ext + offset_edge, step=5)
            long_strike = short_strike + width
            result.update({
                "type": "short_call_spread",
                "short_call": short_strike,
                "long_call": long_strike,
            })
        return result

    # 趋势日：long vertical spread
    if trade_type in (
        "MIDVOL_TREND_LONG_CALL_SPREAD",
        "MIDVOL_TREND_LONG_PUT_SPREAD",
        "LOWVOL_TREND_LONG_CALL_SPREAD",
    ):
        if "CALL" in trade_type or dir_flag >= 0:
            long_k = atm
            short_k = atm + width
            result.update({
                "type": "long_call_vertical",
                "long_call": long_k,
                "short_call": short_k,
            })
        else:
            long_k = atm
            short_k = atm - width
            result.update({
                "type": "long_put_vertical",
                "long_put": long_k,
                "short_put": short_k,
            })
        return result

    # 高波：long strangle
    if trade_type == "HIGHVOL_LONG_STRANGLE":
        if vix_value is None:
            # fallback：用高波 regime 的 q75 大致估一下
            q2 = range_quantiles.get(2, {"q75": 0.0208})
            approx_pts = q2["q75"] * spx_price / 2.0
            offset_points = approx_pts
        else:
            iv = float(vix_value) / 100.0
            sigma_sqrtT = iv * math.sqrt(ONE_DAY_T)
            offset_points = K_BSM_OFFSET * sigma_sqrtT * spx_price

        offset_points = max(offset_points, 20.0)
        call_strike = nearest_strike(spx_price + offset_points, step=5)
        put_strike = nearest_strike(spx_price - offset_points, step=5)
        result.update({
            "type": "long_strangle",
            "long_call": call_strike,
            "long_put": put_strike,
            "offset_points": round(offset_points, 2),
        })
        return result

    # NO_TRADE / 其它：仍然返回 ranges，但结构为 NO_STRUCTURE
    result["type"] = "NO_STRUCTURE"
    return result


def position_sizing(trade_type, spx_price, account_equity=ACCOUNT_EQUITY):
    cfg = STRATEGY_STATS.get(trade_type, None)
    if cfg is None:
        return {
            "kelly_full": 0.0,
            "kelly_used": 0.0,
            "max_risk_dollar": 0.0,
            "contracts": 0,
        }

    p = cfg["p_win"]
    b = cfg["payoff_b"]
    f_full = kelly_fraction(p, b)

    if f_full <= 0:
        return {
            "kelly_full": f_full,
            "kelly_used": 0.0,
            "max_risk_dollar": 0.0,
            "contracts": 0,
        }

    f_used = min(f_full * KELLY_FRACTION_CAP, GLOBAL_RISK_CAP)
    max_risk = account_equity * f_used

    per_loss = per_contract_max_loss(trade_type, spx_price)
    if per_loss is None or per_loss <= 0:
        return {
            "kelly_full": f_full,
            "kelly_used": f_used,
            "max_risk_dollar": max_risk,
            "contracts": 0,
        }

    contracts = max(int(max_risk // per_loss), 0)
    if contracts == 0 and max_risk >= MIN_NOTIONAL_PER_TRADE:
        contracts = 1

    return {
        "kelly_full": f_full,
        "kelly_used": f_used,
        "max_risk_dollar": max_risk,
        "contracts": contracts,
    }


# ===================== 输出到 strategy.txt =====================

def save_plan_to_txt(plan: dict, log_dir, data_banner=None):
    """把当日 plan 保存到 log_dir/<date>_strategy.txt"""
    if not plan:
        return

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    date_str = str(plan.get("date"))
    safe_date = date_str.replace("/", "-").replace(" ", "_")
    out_path = log_dir / f"{safe_date}_strategy.txt"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("======== Daily Meta Strategy Plan (v3) ========\n")
        if data_banner:
            f.write(f"======== DATA BASED ON {data_banner} ========\n")
        f.write(f"Date: {plan.get('date')}\n")
        f.write(f"SPX close used  : {plan.get('spx_price')}\n")
        f.write(f"Source price col: {plan.get('price_col')}\n")
        f.write(f"Vol regime pred : {plan.get('vol_regime')} ({plan.get('vol_regime_label')})\n")
        f.write(
            "Vol P (0/1/2)   : "
            f"{plan.get('vol_p0'):.3f} / {plan.get('vol_p1'):.3f} / {plan.get('vol_p2'):.3f}\n"
        )
        f.write(
            f"Trend flag      : {plan.get('trend_flag')} ({plan.get('trend_label')}) "
            f" | P_trend={plan.get('trend_p_trend'):.3f}\n"
        )
        f.write(
            f"Direction flag  : {plan.get('dir_flag')} ({plan.get('dir_label')}) "
            f" | P_up={plan.get('dir_p_up'):.3f}, P_down={plan.get('dir_p_down'):.3f}\n"
        )
        f.write(f"VIX (approx)    : {plan.get('vix')}\n")
        f.write("------------------------------------------------\n")
        f.write(f"Chosen trade_type : {plan.get('trade_type')}")
        side = plan.get("trade_side")
        if side:
            f.write(f" (side={side})")
        f.write("\n")

        f.write(
            "Predicted range (pts): "
            f"low={plan.get('range_low_pts'):.2f}, "
            f"base={plan.get('range_base_pts'):.2f}, "
            f"high={plan.get('range_high_pts'):.2f}  "
            f"(q_used={plan.get('range_q_used')}, p_used={plan.get('range_p_used'):.3f})\n"
        )
        f.write(
            "Predicted price range BASE: "
            f"[{plan.get('pred_low_base'):.2f} ~ {plan.get('pred_high_base'):.2f}]\n"
        )
        f.write(
            "Predicted price range EXTREME: "
            f"[{plan.get('pred_low_ext'):.2f} ~ {plan.get('pred_high_ext'):.2f}]\n"
        )
        f.write("------------------------------------------------\n")
        f.write("Structure:\n")
        structure = plan.get("structure", {})
        if not structure:
            f.write("  (NO_STRUCTURE / NO_TRADE)\n")
        else:
            for k, v in structure.items():
                f.write(f"  {k}: {v}\n")
        f.write("------------------------------------------------\n")
        f.write(f"Kelly full   : {plan.get('kelly_full'):.3f}\n")
        f.write(f"Kelly used   : {plan.get('kelly_used'):.3f} (cap={GLOBAL_RISK_CAP*100:.1f}%)\n")
        f.write(f"Risk capital : ${plan.get('risk_capital'):.2f}\n")
        f.write(f"Suggested size: {plan.get('contract_size')} SPX contracts\n")
        f.write("================================================\n")

    print(f"[SAVED] Strategy file written to: {out_path}")


# ===================== 主流程 =====================

def generate_today_plan(
    log_path,
    log_dir,
    vix_path,
    quantile_paths,
    data_banner=None,
    account_equity=ACCOUNT_EQUITY,
):
    log_path = Path(log_path)
    range_quantiles = load_range_quantiles(quantile_paths)

    if not log_path.exists():
        raise FileNotFoundError(f"prediction_log not found: {log_path}")

    df = pd.read_csv(log_path)
    if df.empty:
        raise RuntimeError("prediction_log is empty")

    last = df.iloc[-1]

    # 价格列优先顺序：spx_close -> spx_close_src
    price = None
    price_col_used = None
    for col in ["spx_close", "spx_close_src"]:
        if col in last.index:
            try:
                v = float(last[col])
                if not math.isnan(v):
                    price = v
                    price_col_used = col
                    break
            except Exception:
                continue

    if price is None:
        raise RuntimeError(
            "Both 'spx_close' and 'spx_close_src' are NaN in the last row; "
            "cannot decide strikes."
        )

    date = last["date"]
    spx_close = float(price)
    vol_regime = int(last["vol_pred_regime"])

    trend_p_trend = float(last.get("trend_P_trend", 0.0))
    dir_p_up = float(last.get("dir_P_up", 0.0))
    dir_p_down = float(last.get("dir_P_down", 0.0))

    p0 = float(last.get("vol_P_regime0", 0.0))
    p1 = float(last.get("vol_P_regime1", 0.0))
    p2 = float(last.get("vol_P_regime2", 0.0))

    trend_flag = decide_trend_flag(trend_p_trend, threshold=0.5)
    dir_flag = decide_direction(dir_p_up, dir_p_down, margin=0.05)

    vol_label_map = {0: "LOW VOL", 1: "MID VOL", 2: "HIGH VOL"}
    vol_regime_label = vol_label_map.get(vol_regime, "UNKNOWN")

    trend_label = "TREND" if trend_flag == 1 else "NO_TREND"
    if trend_flag not in (0, 1):
        trend_label = f"UNKNOWN({trend_flag})"

    if dir_flag > 0:
        dir_label = "UP_BIAS"
    elif dir_flag < 0:
        dir_label = "DOWN_BIAS"
    else:
        dir_label = "FLAT"

    src_date = last.get("src_date", None)
    vix_val = load_latest_vix(vix_path, src_date)

    trade_type, trade_side = decide_meta_trade(vol_regime, trend_flag, dir_flag)

    print("======== Daily Meta Strategy Plan (v3) ========")
    print(f"Row used        : date={date}, price_col={price_col_used}")
    print(f"Base price      : {spx_close:.2f}")
    print(f"vol_pred_regime : {vol_regime}  (0=low,1=mid,2=high)")
    print(f"vol_P (0/1/2)   : {p0:.3f} / {p1:.3f} / {p2:.3f}")
    print(f"trend_P_trend   : {trend_p_trend:.3f}  -> trend_flag={trend_flag}")
    print(f"dir_P_up/down   : {dir_p_up:.3f} / {dir_p_down:.3f}  -> dir_flag={dir_flag}")
    print(f"VIX (approx)    : {vix_val if vix_val is not None else 'N/A'}")
    print("-----------------------------------------------")

    print(f"Chosen trade_type : {trade_type}"
          + (f" (side={trade_side})" if trade_side else ""))

    strikes = build_structure_strikes(
        trade_type,
        spx_price=spx_close,
        vol_regime=vol_regime,
        dir_flag=dir_flag,
        range_quantiles=range_quantiles,
        vix_value=vix_val,
        p0=p0,
        p1=p1,
        p2=p2,
    )
    sizing = position_sizing(trade_type, spx_close, account_equity)

    print("Predicted range (pts): "
          f"low={strikes.get('range_low_pts')}, "
          f"base={strikes.get('range_base_pts')}, "
          f"high={strikes.get('range_high_pts')}  "
          f"(q_used={strikes.get('range_q_used')}, p_used={strikes.get('range_p_used')})")

    print("Predicted range (price): "
          f"BASE [{strikes.get('predicted_range_low_base')} ~ {strikes.get('predicted_range_high_base')}]  "
          f"EXTREME [{strikes.get('predicted_range_low')} ~ {strikes.get('predicted_range_high')}]")

    print("Structure       :")
    for k, v in strikes.items():
        if k in (
            "predicted_range_low",
            "predicted_range_high",
            "predicted_range_low_base",
            "predicted_range_high_base",
            "range_low_pts",
            "range_base_pts",
            "range_high_pts",
            "range_q_used",
            "range_p_used",
            "raw_q25",
            "raw_q50",
            "raw_q75",
            "vix_used",
        ):
            continue
        print(f"  - {k}: {v}")

    print("-----------------------------------------------")
    print(f"Kelly (full)    : {sizing['kelly_full']:.3f}")
    print(f"Kelly used      : {sizing['kelly_used']:.3f} (<= {GLOBAL_RISK_CAP*100:.1f}% cap)")
    print(f"Risk capital    : ${sizing['max_risk_dollar']:.2f}")
    print(f"Suggested size  : {sizing['contracts']} SPX contracts (per-leg size=1)")
    print("================================================")

    # 把 strikes 中和结构直接相关的字段抽出来（去掉 meta 信息）
    structure_fields_to_skip = {
        "predicted_range_low",
        "predicted_range_high",
        "predicted_range_low_base",
        "predicted_range_high_base",
        "range_low_pts",
        "range_base_pts",
        "range_high_pts",
        "range_q_used",
        "range_p_used",
        "raw_q25",
        "raw_q50",
        "raw_q75",
        "vix_used",
    }
    structure = {
        k: v for k, v in strikes.items()
        if k not in structure_fields_to_skip
    }

    plan = {
        "date": date,
        "spx_price": spx_close,
        "price_col": price_col_used,
        "vol_regime": vol_regime,
        "vol_regime_label": vol_regime_label,
        "vol_p0": p0,
        "vol_p1": p1,
        "vol_p2": p2,
        "trend_p_trend": trend_p_trend,
        "trend_flag": trend_flag,
        "trend_label": trend_label,
        "dir_p_up": dir_p_up,
        "dir_p_down": dir_p_down,
        "dir_flag": dir_flag,
        "dir_label": dir_label,
        "trade_type": trade_type,
        "trade_side": trade_side,
        "range_low_pts": strikes.get("range_low_pts"),
        "range_base_pts": strikes.get("range_base_pts"),
        "range_high_pts": strikes.get("range_high_pts"),
        "range_q_used": strikes.get("range_q_used"),
        "range_p_used": strikes.get("range_p_used"),
        "pred_low_base": strikes.get("predicted_range_low_base"),
        "pred_high_base": strikes.get("predicted_range_high_base"),
        "pred_low_ext": strikes.get("predicted_range_low"),
        "pred_high_ext": strikes.get("predicted_range_high"),
        "structure": structure,
        "kelly_full": sizing["kelly_full"],
        "kelly_used": sizing["kelly_used"],
        "risk_capital": sizing["max_risk_dollar"],
        "contract_size": sizing["contracts"],
        "vix": vix_val,
    }

    # 自动写 strategy.txt
    save_plan_to_txt(plan, log_dir, data_banner=data_banner)

    return plan
