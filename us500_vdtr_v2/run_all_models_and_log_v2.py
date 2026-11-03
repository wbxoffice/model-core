#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run SPX VDTR v2.1 (GEX+DIX+VIX+Events) models
and log daily predictions + realized outcomes into a CSV.

特点：
- 完全复用 P2 的 prediction_log / txt summary 输出格式
- 特征由 feature_engineering_v2_1.build_feature_frame_v21() 提供
- 模型路径、阈值等全部从 config.py 读取
"""

import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from config import (
    BASE_DIR, DATA_DIR, MODEL_DIR, LOG_DIR,
    # 下面这些路径虽然在本脚本中不直接用到，但保留从 config 导入
    SPX_FILE, DIX_FILE,  # noqa: F401
    PREDICTION_LOG_FILE,
    VOL_MODEL_FILE, DIR_MODEL_FILE, TREND_MODEL_FILE, REV_MODEL_FILE,
    VOL_MODEL_NAME, DIR_MODEL_NAME, TREND_MODEL_NAME, REV_MODEL_NAME,
    P_DIR_TH, P_TRE_TH, P_REV_TH, PREDICTION_START_DATE, EVENT_CAL_FILE,
)

PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nyse_calendar_utils import ensure_nyse_open_days_csv, load_nyse_open_days, align_series_to_next_open_day

# v2.1 特征工程（训练时用的就是这个）
from feature_engineering_v2_1 import build_feature_frame_v21

# 统一使用 config 中的路径
LOG_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_LOG_FILE = PREDICTION_LOG_FILE
REPLAY_LOG_FILE = LOG_DIR / "prediction_log_replay.csv"
LIVE_LOG_FILE = LOG_DIR / "prediction_log_live.csv"
SUMMARY_LOG_FILE = LIVE_LOG_FILE
LOG_FILE = SUMMARY_LOG_FILE


# ========== 工具函数（和 P2 保持一致） ==========

def load_model(path: Path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    return model, feature_cols


def proba_for_class(clf, proba: np.ndarray, cls_val):
    classes = list(clf.classes_)
    idx = classes.index(cls_val)
    return proba[:, idx]


def correctness(pred, real):
    if pd.isna(real):
        return np.nan, ""
    ok = (pred == real)
    if ok:
        return True, ""
    return False, f"pred={pred}, real={real}"


def _read_log_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _merge_overwrite_by_date(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new_df.sort_values("date").reset_index(drop=True).copy()

    old = existing.copy().set_index("date")
    new = new_df.copy().set_index("date")

    all_cols = sorted(set(old.columns) | set(new.columns))
    old = old.reindex(columns=all_cols).astype("object")
    new = new.reindex(columns=all_cols).astype("object")

    out = old.copy()
    out.update(new)
    missing_idx = new.index.difference(out.index)
    if len(missing_idx) > 0:
        out = pd.concat([out, new.loc[missing_idx]], axis=0)

    return out.reset_index().sort_values("date").reset_index(drop=True)


def _merge_live_settlement_only(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new_df.sort_values("date").reset_index(drop=True).copy()

    live_mutable_cols = {
        "real_regime", "real_direction", "real_trend", "real_reversal",
        "vol_correct", "vol_error_desc",
        "dir_correct", "dir_error_desc",
        "trend_correct", "trend_error_desc",
        "rev_correct", "rev_error_desc",
        "spx_open", "spx_high", "spx_low", "spx_close",
        "next_ret", "next_range",
    }

    old = existing.copy().set_index("date")
    new = new_df.copy().set_index("date")
    all_cols = sorted(set(old.columns) | set(new.columns))
    old = old.reindex(columns=all_cols).astype("object")
    new = new.reindex(columns=all_cols).astype("object")

    for dt in new.index:
        if dt not in old.index:
            old = pd.concat([old, new.loc[[dt]]], axis=0)
            continue
        for c in live_mutable_cols:
            if c in new.columns:
                old.at[dt, c] = new.at[dt, c]

    return old.reset_index().sort_values("date").reset_index(drop=True)


# ========== 主逻辑 ==========

def main():
    # ---------- 构造特征 + 标签 ----------
    # v2.1 的所有特征 + 标签 + target_date 都在这里完成
    df = build_feature_frame_v21(for_training=False)

    # ---------- 加载模型 ----------
    vol_model, vol_features = load_model(VOL_MODEL_FILE)
    dir_model, dir_features = load_model(DIR_MODEL_FILE)
    trend_model, trend_features = load_model(TREND_MODEL_FILE)
    rev_model, rev_features = load_model(REV_MODEL_FILE)

    # 取四个模型特征的并集，保证所有需要的列都完整
    all_features = sorted(
        set(vol_features) | set(dir_features) | set(trend_features) | set(rev_features)
    )
    df_feat = df.dropna(subset=all_features).copy()
    if df_feat.empty:
        raise ValueError("没有完整特征行，请检查数据或特征工程。")

    # ---------- 预测 ----------
    # Vol
    X_vol = df_feat[vol_features]
    vol_proba = vol_model.predict_proba(X_vol)
    vol_pred = vol_model.predict(X_vol)

    # Direction
    X_dir = df_feat[dir_features]
    dir_proba = dir_model.predict_proba(X_dir)
    dir_pred = dir_model.predict(X_dir)
    P_up = proba_for_class(dir_model, dir_proba, 1)
    P_neutral = proba_for_class(dir_model, dir_proba, 0)
    P_down = proba_for_class(dir_model, dir_proba, -1)

    # Trend
    X_trend = df_feat[trend_features]
    trend_proba = trend_model.predict_proba(X_trend)
    trend_pred = trend_model.predict(X_trend)
    P_trend = proba_for_class(trend_model, trend_proba, 1)
    P_notrend = proba_for_class(trend_model, trend_proba, 0)

    # Reversal
    X_rev = df_feat[rev_features]
    rev_proba = rev_model.predict_proba(X_rev)
    rev_pred = rev_model.predict(X_rev)
    P_rev = proba_for_class(rev_model, rev_proba, 1)
    P_norev = proba_for_class(rev_model, rev_proba, 0)

    # ---------- 构造“目标日 = 下一日”的输出（和 P2 一致） ----------
    out = pd.DataFrame(index=df_feat.index)

    # 特征日 t（有行情 / gex / dix / vix / event 的那天）
    out["src_date"] = df_feat["date"]
    out["spx_close_src"] = df_feat["close"]

    # 目标日 t+1：这里仍然沿用 “shift(-1)” 的写法，
    # 和 P2 的 prediction_log 完全对齐
    out["date"] = df_feat["date"].shift(-1)
    out["spx_open"] = df_feat["next_open"]
    out["spx_high"] = df_feat["next_high"]
    out["spx_low"] = df_feat["next_low"]
    out["spx_close"] = df_feat["next_close"]
    out["next_ret"] = df_feat["next_ret"]
    out["next_range"] = df_feat["next_range"]
    # ===== v2.1 特有：把 VIX / 事件信息也写进日志 =====
    # 注意：这里区分 t（日内特征日）和 t+1（预测日）
    if "vix_level" in df_feat.columns:
        out["vix_level"] = df_feat["vix_level"]

    # 特征日 t 的事件（用于回溯 / 调试）
    if "is_fomc" in df_feat.columns:
        out["is_fomc_t"] = df_feat["is_fomc"]
    if "is_fomc_window" in df_feat.columns:
        out["is_fomc_window_t"] = df_feat["is_fomc_window"]
    if "is_vix_expiry" in df_feat.columns:
        out["is_vix_expiry_t"] = df_feat["is_vix_expiry"]
    if "is_spy_opex_m" in df_feat.columns:
        out["is_spy_opex_m_t"] = df_feat["is_spy_opex_m"]


    # 真实标签（t+1 的 realized）
    out["real_regime"] = df_feat["real_regime"]
    out["real_direction"] = df_feat["real_direction"]
    out["real_trend"] = df_feat["real_trend"]
    out["real_reversal"] = df_feat["real_reversal"]

    # ----- 波动模型 -----
    out["vol_model_name"] = VOL_MODEL_NAME
    out["vol_pred_regime"] = vol_pred
    out["vol_P_regime0"] = proba_for_class(vol_model, vol_proba, 0)
    out["vol_P_regime1"] = proba_for_class(vol_model, vol_proba, 1)
    out["vol_P_regime2"] = proba_for_class(vol_model, vol_proba, 2)

    def vol_decision(r):
        if r == 0:
            return "SELL_GAMMA"
        if r == 1:
            return "NEUTRAL"
        if r == 2:
            return "BUY_GAMMA"
        return ""

    out["vol_decision"] = out["vol_pred_regime"].apply(vol_decision)

    # ----- 方向模型 -----
    out["dir_model_name"] = DIR_MODEL_NAME
    out["dir_pred"] = dir_pred
    out["dir_P_up"] = P_up
    out["dir_P_neutral"] = P_neutral
    out["dir_P_down"] = P_down

    def dir_decision(row, th=P_DIR_TH):
        p_up, p_down, p_neu = row["dir_P_up"], row["dir_P_down"], row["dir_P_neutral"]
        m = max(p_up, p_down, p_neu)
        if m < th:
            return "NO_SIGNAL"
        if m == p_neu:
            return "FLAT"
        if m == p_up:
            return "LONG"
        return "SHORT"

    out["dir_decision"] = out.apply(dir_decision, axis=1)

    # ----- 趋势模型 -----
    out["trend_model_name"] = TREND_MODEL_NAME
    out["trend_pred"] = trend_pred
    out["trend_P_trend"] = P_trend
    out["trend_P_notrend"] = P_notrend

    def trend_decision(row, th=P_TRE_TH):
        return "TREND_DAY" if row["trend_P_trend"] >= th else "NO_TREND_BET"

    out["trend_decision"] = out.apply(trend_decision, axis=1)

    # ----- 反转模型 -----
    out["rev_model_name"] = REV_MODEL_NAME
    out["rev_pred"] = rev_pred
    out["rev_P_rev"] = P_rev
    out["rev_P_norev"] = P_norev

    def rev_decision(row, th=P_REV_TH):
        return "REVERSAL_RISK" if row["rev_P_rev"] >= th else "NORMAL"

    out["rev_decision"] = out.apply(rev_decision, axis=1)

    # 对于最后一个有特征的交易日，没有实际 next 行，
    # 用“下一个工作日”作为目标日期（真实结果保持 NaN）
    mask_nan_date = out["date"].isna()
    if mask_nan_date.any():
        out.loc[mask_nan_date, "date"] = out.loc[mask_nan_date, "src_date"] + BDay(1)
    # 将预测日统一对齐到 NYSE 开市日（自动跳过周末 + 休市日）
    nyse_cal_file = PROJECT_ROOT / "data" / "nyse_open_days.csv"
    ensure_nyse_open_days_csv(nyse_cal_file)
    nyse_open_days = load_nyse_open_days(nyse_cal_file)
    out["date"] = align_series_to_next_open_day(out["date"], nyse_open_days)
    # 基于预测日 date，从 event_calendar 补充 t+1 的事件标记
    try:
        events = pd.read_csv(EVENT_CAL_FILE, parse_dates=["date"])
        cols = ["date", "is_fomc", "is_fomc_window", "is_vix_expiry", "is_spy_opex_m"]
        events = events[cols].rename(columns={
            "is_fomc": "is_fomc_t1",
            "is_fomc_window": "is_fomc_window_t1",
            "is_vix_expiry": "is_vix_expiry_t1",
            "is_spy_opex_m": "is_spy_opex_m_t1",
        })
        # 注意：这里按 out["date"] 合并，所以即使没有 t+1 的 SPX 行，也能拿到事件
        out = out.merge(events, on="date", how="left")
    except Exception as e:
        print("Warning: failed to merge event_calendar for t+1:", e)

    # 正确性标记
    vol_corr = out.apply(lambda r: correctness(r["vol_pred_regime"], r["real_regime"]), axis=1)
    dir_corr = out.apply(lambda r: correctness(r["dir_pred"], r["real_direction"]), axis=1)
    trend_corr = out.apply(lambda r: correctness(r["trend_pred"], r["real_trend"]), axis=1)
    rev_corr = out.apply(lambda r: correctness(r["rev_pred"], r["real_reversal"]), axis=1)

    out["vol_correct"] = [c[0] for c in vol_corr]
    out["vol_error_desc"] = [c[1] for c in vol_corr]
    out["dir_correct"] = [c[0] for c in dir_corr]
    out["dir_error_desc"] = [c[1] for c in dir_corr]
    out["trend_correct"] = [c[0] for c in trend_corr]
    out["trend_error_desc"] = [c[1] for c in trend_corr]
    out["rev_correct"] = [c[0] for c in rev_corr]
    out["rev_error_desc"] = [c[1] for c in rev_corr]

    # 过滤掉目标日期 NaT
    out = out.dropna(subset=["date"]).copy()
    out = out.sort_values("date").reset_index(drop=True)

    # 限定 prediction 起始日期（例如 config 里设成 2024-07-01）
    if PREDICTION_START_DATE:
        start_dt = pd.to_datetime(PREDICTION_START_DATE)
        out = out[out["date"] >= start_dt]

    # ---------- 双轨日志 ----------
    replay_old = _read_log_df(REPLAY_LOG_FILE)
    replay_new = _merge_overwrite_by_date(replay_old, out)
    replay_new.to_csv(REPLAY_LOG_FILE, index=False)

    live_seed = _read_log_df(LIVE_LOG_FILE)
    if live_seed.empty:
        live_seed = _read_log_df(LEGACY_LOG_FILE)
    live_new = _merge_live_settlement_only(live_seed, out)
    live_new.to_csv(LIVE_LOG_FILE, index=False)

    replay_new.to_csv(LEGACY_LOG_FILE, index=False)

    print(f"Replay log written to: {REPLAY_LOG_FILE}")
    print(f"Live log written to  : {LIVE_LOG_FILE}")
    print(f"Legacy log written to: {LEGACY_LOG_FILE}")

    # ========== 生成当日 txt 总结（最新预测 + 最近已实现表现） ==========
    try:
        source = SUMMARY_LOG_FILE if SUMMARY_LOG_FILE.exists() else REPLAY_LOG_FILE
        df_log = pd.read_csv(source)
        if df_log.empty:
            print("Log is empty, skip daily summary.")
            return

        df_log["date"] = pd.to_datetime(df_log["date"])
        if "src_date" in df_log.columns:
            df_log["src_date"] = pd.to_datetime(df_log["src_date"])
        df_log = df_log.sort_values("date")

        # 最新一条预测
        latest_row = df_log.iloc[-1]

        src_date = latest_row.get("src_date")
        if pd.notna(src_date):
            src_date = pd.to_datetime(src_date).date()

        target_date = latest_row.get("date")
        if pd.notna(target_date):
            target_date = pd.to_datetime(target_date).date()
        else:
            target_date = None

        close_src = latest_row.get("spx_close_src", np.nan)

        # ===== v2.1 特有：VIX + 事件信息（默认理解为预测日 t+1 的标记） =====
        vix_level = latest_row.get("vix_level", np.nan)
        # 这里全部用 *_t1，表示 “预测日 t+1 的事件”
        is_fomc = bool(latest_row.get("is_fomc_t1", 0) == 1)
        is_fomc_window = bool(latest_row.get("is_fomc_window_t1", 0) == 1)
        is_vix_expiry = bool(latest_row.get("is_vix_expiry_t1", 0) == 1)
        is_spy_opex_m = bool(latest_row.get("is_spy_opex_m_t1", 0) == 1)

        def classify_vix(x):
            if pd.isna(x):
                return "N/A"
            x = float(x)
            if x < 13:
                return "very low / complacent"
            elif x < 20:
                return "normal"
            elif x < 30:
                return "elevated"
            else:
                return "stressed"

        # 波动模型
        P_reg0 = latest_row.get("vol_P_regime0", np.nan)
        P_reg1 = latest_row.get("vol_P_regime1", np.nan)
        P_reg2 = latest_row.get("vol_P_regime2", np.nan)
        pred_regime = latest_row.get("vol_pred_regime", np.nan)
        vol_action = latest_row.get("vol_decision", "")

        # 方向模型
        P_up = latest_row.get("dir_P_up", np.nan)
        P_neutral = latest_row.get("dir_P_neutral", np.nan)
        P_down = latest_row.get("dir_P_down", np.nan)
        dir_pred = latest_row.get("dir_pred", np.nan)
        dir_dec = latest_row.get("dir_decision", "")

        # 趋势模型
        P_trend = latest_row.get("trend_P_trend", np.nan)
        trend_pred = latest_row.get("trend_pred", np.nan)
        trend_dec = latest_row.get("trend_decision", "")

        # 反转模型
        P_rev = latest_row.get("rev_P_rev", np.nan)
        rev_pred = latest_row.get("rev_pred", np.nan)
        rev_dec = latest_row.get("rev_decision", "")

        # 最新预测日本身的真实结果
        real_regime = latest_row.get("real_regime", np.nan)
        real_dir = latest_row.get("real_direction", np.nan)
        real_trend = latest_row.get("real_trend", np.nan)
        real_rev = latest_row.get("real_reversal", np.nan)

        REGIME_MAP = {
            0: "Low volatility",
            1: "Medium volatility",
            2: "High volatility",
        }

        RANGE_STATS = {
            0: (0.0045, 0.0059, 0.0075),
            1: (0.0091, 0.0108, 0.0128),
            2: (0.0168, 0.0203, 0.0258),
        }

        if (not pd.isna(pred_regime)
                and int(pred_regime) in RANGE_STATS
                and not pd.isna(close_src)):
            p25, med, p75 = RANGE_STATS[int(pred_regime)]
            low_pts = p25 * close_src
            mid_pts = med * close_src
            high_pts = p75 * close_src
        else:
            p25 = med = p75 = low_pts = mid_pts = high_pts = np.nan

        def fmt_real(x):
            return "N/A" if pd.isna(x) else str(int(x))

        DIR_MAP = {-1: "Down", 0: "Neutral", 1: "Up"}

        # ========= 正式开始拼 summary 文本 =========
        lines = []
        lines.append("========================================")
        lines.append("=DATA BASED ON US500=")
        lines.append("========================================")
        lines.append("[Latest prediction]")
        if src_date:
            lines.append(f"Feature date (t)          : {src_date}")
        if target_date:
            lines.append(f"Predicting next day (t+1) : {target_date}")
        lines.append("----------------------------------------")
        if not pd.isna(close_src):
            lines.append(f"SPX close on t            : {close_src:.2f}")
        lines.append("")

        # ===== VIX + 事件信息（针对预测日 t+1） =====
        lines.append("[VIX & Event on next day (t+1)]")
        # VIX 状态
        if not pd.isna(vix_level):
            lines.append(
                f"VIX(t)                    : {vix_level:.2f} ({classify_vix(vix_level)})"
            )
        else:
            lines.append("VIX(t)                    : N/A")

        # 事件具体描述
        event_flags = []
        if is_fomc:
            event_flags.append("FOMC decision day")
        if is_fomc_window and not is_fomc:
            event_flags.append("FOMC window (±1d)")
        if is_vix_expiry:
            event_flags.append("VIX futures/options expiry")
        if is_spy_opex_m:
            event_flags.append("SPY monthly OPEX")

        if event_flags:
            if is_fomc or is_vix_expiry or is_spy_opex_m:
                lines.append(">>> EVENT ALERT on predicted day (t+1) <<<")
            lines.append("Events on t+1             : " + ", ".join(event_flags))
        else:
            lines.append("Events on t+1             : None (normal session)")

        lines.append("")

        # 波动模型
        lines.append(f"[Volatility Model v2.1] ({VOL_MODEL_NAME})")
        lines.append(f"P_low  = {P_reg0:.3f}")
        lines.append(f"P_mid  = {P_reg1:.3f}")
        lines.append(f"P_high = {P_reg2:.3f}")
        reg_text = REGIME_MAP.get(int(pred_regime), "Unknown") if not pd.isna(pred_regime) else "Unknown"
        lines.append(f"Pred regime  : {pred_regime} ({reg_text})")
        lines.append(f"Decision     : {vol_action}")
        lines.append("")
        lines.append("Expected intraday HIGH-LOW range for next day:")
        if not pd.isna(p25):
            lines.append(f"- 25th pct : ~{p25 * 100:.2f}%  ≈ {low_pts:.1f} pts")
            lines.append(f"- Median   : ~{med * 100:.2f}%  ≈ {mid_pts:.1f} pts")
            lines.append(f"- 75th pct : ~{p75 * 100:.2f}%  ≈ {high_pts:.1f} pts")
        else:
            lines.append("- Not enough data for range stats.")
        lines.append("")

        # 方向模型
        lines.append(f"[Direction Model] ({DIR_MODEL_NAME})")
        lines.append(f"P_down    = {P_down:.3f}")
        lines.append(f"P_neutral = {P_neutral:.3f}")
        lines.append(f"P_up      = {P_up:.3f}")
        dir_txt = DIR_MAP.get(int(dir_pred), "Unknown") if not pd.isna(dir_pred) else "Unknown"
        lines.append(f"Pred direction : {dir_pred} ({dir_txt})")
        lines.append(f"Decision       : {dir_dec}")
        lines.append("")

        # 趋势模型
        lines.append(f"[Trend Model] ({TREND_MODEL_NAME})")
        lines.append(f"P_trend   = {P_trend:.3f}")
        lines.append(f"Pred trend    : {fmt_real(trend_pred)} (1=trend day, 0=non-trend)")
        lines.append(f"Decision      : {trend_dec}")
        lines.append("")

        # 反转模型
        lines.append(f"[Reversal Model] ({REV_MODEL_NAME})")
        lines.append(f"P_reversal    = {P_rev:.3f}")
        lines.append(f"Pred reversal : {fmt_real(rev_pred)} (1=reversal, 0=normal)")
        lines.append(f"Decision      : {rev_dec}")
        lines.append("")

        # ===== 基于 v2.1 的简单交易建议 =====
        lines.append("[Heuristic suggestion]")

        # 结构建议：根据 vol_decision
        if vol_action == "BUY_GAMMA":
            struct_sugg = "prefer long gamma: butterflies / straddles / narrow debit spreads"
        elif vol_action == "SELL_GAMMA":
            struct_sugg = "prefer short gamma: iron condors / credit spreads with defined risk"
        else:
            struct_sugg = "no strong vol edge: small verticals or wait for clearer setup"

        # 方向倾向：根据 dir_decision
        if dir_dec == "LONG":
            dir_sugg = "directional bias: mildly bullish (LONG)"
        elif dir_dec == "SHORT":
            dir_sugg = "directional bias: mildly bearish (SHORT)"
        elif dir_dec == "FLAT":
            dir_sugg = "directional bias: flat / hedged"
        else:
            dir_sugg = "directional bias: none (NO_SIGNAL)"

        # 风险备注：结合 trend / reversal / 事件
        risk_notes = []
        if trend_dec == "TREND_DAY":
            risk_notes.append("if breakout confirms, can trail winners instead of fading moves")
        if rev_dec == "REVERSAL_RISK":
            risk_notes.append("watch for fake breakouts / intraday mean reversion")

        if is_fomc or is_vix_expiry or is_spy_opex_m:
            risk_notes.append("event-driven flows today, position smaller and expect intraday spikes")

        lines.append(f"- Structure : {struct_sugg}")
        lines.append(f"- Direction : {dir_sugg}")
        if risk_notes:
            for rn in risk_notes:
                lines.append(f"- Note      : {rn}")
        else:
            lines.append("- Note      : no special risk flags from trend/reversal/events")

        lines.append("")

        # 最近一个已结算交易日的表现回顾
        realized_df = df_log[df_log["real_regime"].notna()].sort_values("date")
        if not realized_df.empty:
            last_real = realized_df.iloc[-1]
            last_real_date = pd.to_datetime(last_real["date"]).date()
            lines.append("----- Last realized day performance -----")
            lines.append(f"Real date (t+1): {last_real_date}")
            lines.append(
                f"Vol : pred={last_real.get('vol_pred_regime')}, "
                f"real={fmt_real(last_real.get('real_regime'))}, "
                f"correct={last_real.get('vol_correct')}"
            )
            rd = last_real.get("real_direction")
            lines.append(
                f"Dir : pred={last_real.get('dir_pred')}, "
                f"real={fmt_real(rd)}"
            )
            lines.append(
                f"Trend: pred={fmt_real(last_real.get('trend_pred'))}, "
                f"real={fmt_real(last_real.get('real_trend'))}"
            )
            lines.append(
                f"Rev : pred={fmt_real(last_real.get('rev_pred'))}, "
                f"real={fmt_real(last_real.get('real_reversal'))}"
            )
        else:
            lines.append("----- No realized day with labels yet -----")

        lines.append("========================================")
        lines.append("")

        # 文件名：优先用特征日 t，其次用目标日 t+1
        if src_date:
            base_for_name = src_date
        elif target_date:
            base_for_name = target_date
        else:
            base_for_name = df_log["date"].max().date()

        txt_name = f"{base_for_name.isoformat()}_summary.txt"
        txt_path = LOG_DIR / txt_name
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Daily summary written to: {txt_path}")
    except Exception as e:
        print("Warning: failed to write daily summary txt:", e)


if __name__ == "__main__":
    main()
