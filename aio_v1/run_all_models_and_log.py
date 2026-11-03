
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run SPX Fluctuation + Direction + Trend + Reversal models
and log daily predictions + realized outcomes into a CSV.

Python 3.11, for PyCharm.
"""

import os
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


# ====================== 路径配置 ======================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from nyse_calendar_utils import ensure_nyse_open_days_csv, load_nyse_open_days, align_series_to_next_open_day

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
LOG_DIR = BASE_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "prediction_log.csv"

SPX_FILE = DATA_DIR / "SP_SPX, 1D.csv"
DIX_FILE = DATA_DIR / "DIX.xlsx"

VOL_MODEL_FILE = MODEL_DIR / "spx_vol_regime_model_v3p0_fullhistory.pkl"
DIR_MODEL_FILE = MODEL_DIR / "spx_dir_model_v1.pkl"
TREND_MODEL_FILE = MODEL_DIR / "spx_trend_model_v1.pkl"
REV_MODEL_FILE = MODEL_DIR / "spx_reversal_model_v1.pkl"

VOL_MODEL_NAME = "vol_regime_v3p0_fullhistory"
DIR_MODEL_NAME = "direction_v1p0"
TREND_MODEL_NAME = "trend_v1p0"
REV_MODEL_NAME = "reversal_v1p0"


# ====================== 工具函数 ======================

def classify_range(r: float, low: float = 0.008, mid: float = 0.015) -> int:
    """0 = low, 1 = mid, 2 = high volatility regime."""
    if pd.isna(r):
        return np.nan
    if r < low:
        return 0
    if r < mid:
        return 1
    return 2


def direction_label(ret: float, th: float = 0.002) -> int:
    """
    三分类方向标签:
    > +0.2%  → +1 (up)
    < -0.2%  → -1 (down)
    其它      → 0 (neutral)
    """
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


def rolling_pct(series: pd.Series, window: int = 60, min_periods: int = 20) -> np.ndarray:
    vals = series.values.astype(float)
    out = np.full_like(vals, np.nan, dtype=float)
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        w = vals[start:i+1]
        if len(w) < min_periods:
            continue
        out[i] = np.sum(w <= w[-1]) / len(w)
    return out


def load_model(path: Path):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    return model, feature_cols


def proba_for_class(clf, proba: np.ndarray, cls_val):
    """给定 sklearn classifier 和 predict_proba 输出，取某个 class 的概率列。"""
    classes = list(clf.classes_)
    idx = classes.index(cls_val)
    return proba[:, idx]


# ====================== 主逻辑 ======================

def main():
    # ---------- 读取数据 ----------
    spx = pd.read_csv(SPX_FILE)
    dix = pd.read_excel(DIX_FILE)

    # 假定 SPX 有列 'time'，DIX 有 'date'
    spx["date"] = pd.to_datetime(spx["time"])
    dix["date"] = pd.to_datetime(dix["date"])

    df = spx.merge(dix, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # ---------- 构造标签（针对“目标日 = 下一日”） ----------
    df["next_open"] = df["open"].shift(-1)
    df["next_high"] = df["high"].shift(-1)
    df["next_low"] = df["low"].shift(-1)
    df["next_close"] = df["close"].shift(-1)

    df["next_range"] = (df["next_high"] - df["next_low"]) / df["close"]
    df["real_regime"] = df["next_range"].apply(classify_range)

    df["next_ret"] = df["next_close"] / df["close"] - 1.0
    df["real_direction"] = df["next_ret"].apply(direction_label)

    # 趋势标签: | (next_close - next_open) / next_open | > 0.6%
    df["real_trend"] = (
        (df["next_close"] - df["next_open"]).abs() / df["next_open"] > 0.006
    ).astype(float)
    df.loc[df["next_open"].isna() | df["next_close"].isna(), "real_trend"] = np.nan

    # 反转标签: gap_up+收跌 或 gap_down+收涨
    gap_up = df["next_open"] > df["close"]
    gap_down = df["next_open"] < df["close"]
    close_red = df["next_close"] < df["next_open"]
    close_green = df["next_close"] > df["next_open"]
    df["real_reversal"] = (
        (gap_up & close_red) | (gap_down & close_green)
    ).astype(float)
    df.loc[df["next_open"].isna() | df["next_close"].isna(), "real_reversal"] = np.nan

    # ---------- 构造与 v3.0 一致的特征 ----------
    df["ret"] = df["close"].pct_change()
    df["range"] = (df["high"] - df["low"]) / df["close"]

    df["gex_abs"] = df["gex"].abs()
    df["gex_scaled"] = df["gex_abs"] / 1e9
    df["gex_sign"] = np.sign(df["gex"])
    df["log_gex_abs"] = np.log(df["gex_abs"].replace(0, np.nan))

    df["gex_5d_chg"] = df["gex"].pct_change(5)
    df["gex_20d_chg"] = df["gex"].pct_change(20)
    df["dix_5d_roc"] = df["dix"].pct_change(5)
    df["dix_20d_roc"] = df["dix"].pct_change(20)

    df["dow"] = df["date"].dt.weekday
    df["moy"] = df["date"].dt.month

    df["gex_abs_pct_252"] = rolling_percentile(df["gex_scaled"], 252, 60)
    df["gex_1d_chg"] = df["gex"].pct_change(1)
    df["gex_3d_chg"] = df["gex"].pct_change(3)
    df["gex_pct_60"] = rolling_pct(df["gex_scaled"], 60, 20)
    df["gex_scaled_z120"] = (
        df["gex_scaled"] - df["gex_scaled"].rolling(120).mean()
    ) / df["gex_scaled"].rolling(120).std()

    # v3.0 base 特征列
    v30_cols = [
        "dix", "log_gex_abs", "gex_sign", "ret", "range",
        "gex_abs_pct_252", "gex_5d_chg", "gex_20d_chg",
        "dix_5d_roc", "dix_20d_roc",
        "gex_1d_chg", "gex_3d_chg", "gex_pct_60", "gex_scaled_z120"
    ]

    # 滞后特征 lag0~lag4
    for col in v30_cols:
        for lag in range(5):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # ---------- 加载四个模型 ----------
    vol_model, vol_features = load_model(VOL_MODEL_FILE)
    dir_model, dir_features = load_model(DIR_MODEL_FILE)
    trend_model, trend_features = load_model(TREND_MODEL_FILE)
    rev_model, rev_features = load_model(REV_MODEL_FILE)


    # 在 dropna 之前，统一把 inf / -inf 转成 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 需要满足所有模型的特征不缺失
    all_features = sorted(set(vol_features) | set(dir_features) |
                          set(trend_features) | set(rev_features))
    df_feat = df.dropna(subset=all_features).copy()

    # ---------- 预测 ----------
    # Volatility regime
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

    # ---------- 构造“目标日期 = 第二天”的日志 DataFrame ----------
    out = pd.DataFrame(index=df_feat.index)

    # 特征日 t
    out["src_date"] = df_feat["date"]

    # 目标日：优先用真实“下一条数据的日期”，对于最后一行没有下一条的，就用 t + 1 个工作日（BDay）
    out["date"] = df_feat["date"].shift(-1)

    # 对于最后一行（或任何 date 为 NaT 且 src_date 不为空的地方），用工作日+1 填上一个“预估的下一个交易日”
    mask_nat = out["date"].isna() & out["src_date"].notna()
    out.loc[mask_nat, "date"] = out.loc[mask_nat, "src_date"] + BDay(1)
    # 将预测日统一对齐到 NYSE 开市日（自动跳过周末 + 休市日）
    nyse_cal_file = PROJECT_ROOT / "data" / "nyse_open_days.csv"
    ensure_nyse_open_days_csv(nyse_cal_file)
    nyse_open_days = load_nyse_open_days(nyse_cal_file)
    out["date"] = align_series_to_next_open_day(out["date"], nyse_open_days)

    # 下一天 SPX 实际行情
    out["spx_open"] = df_feat["next_open"]
    out["spx_high"] = df_feat["next_high"]
    out["spx_low"] = df_feat["next_low"]
    out["spx_close"] = df_feat["next_close"]
    out["spx_close_src"] = df_feat["close"]
    out["next_ret"] = df_feat["next_ret"]
    out["next_range"] = df_feat["next_range"]

    # 实际标签
    out["real_regime"] = df_feat["real_regime"]
    out["real_direction"] = df_feat["real_direction"]
    out["real_trend"] = df_feat["real_trend"]
    out["real_reversal"] = df_feat["real_reversal"]

    # 波动模型预测 + 概率 + 决策
    out["vol_model_name"] = VOL_MODEL_NAME
    out["vol_pred_regime"] = vol_pred
    out["vol_P_regime0"] = proba_for_class(vol_model, vol_proba, 0)
    out["vol_P_regime1"] = proba_for_class(vol_model, vol_proba, 1)
    out["vol_P_regime2"] = proba_for_class(vol_model, vol_proba, 2)

    # 简单决策：0=SELL_GAMMA, 1=NEUTRAL, 2=BUY_GAMMA
    def vol_decision(r):
        if r == 0:
            return "SELL_GAMMA"
        if r == 1:
            return "NEUTRAL"
        if r == 2:
            return "BUY_GAMMA"
        return ""

    out["vol_decision"] = out["vol_pred_regime"].apply(vol_decision)

    # 方向模型预测 + 概率 + 决策
    out["dir_model_name"] = DIR_MODEL_NAME
    out["dir_pred"] = dir_pred
    out["dir_P_up"] = P_up
    out["dir_P_neutral"] = P_neutral
    out["dir_P_down"] = P_down

    # 决策规则（你可以后续自行调整阈值）
    def dir_decision(row, th=0.6):
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

    # 趋势模型预测 + 概率 + 决策
    out["trend_model_name"] = TREND_MODEL_NAME
    out["trend_pred"] = trend_pred
    out["trend_P_trend"] = P_trend
    out["trend_P_notrend"] = P_notrend

    def trend_decision(row, th=0.6):
        return "TREND_DAY" if row["trend_P_trend"] >= th else "NO_TREND_BET"

    out["trend_decision"] = out.apply(trend_decision, axis=1)

    # 反转模型预测 + 概率 + 决策
    out["rev_model_name"] = REV_MODEL_NAME
    out["rev_pred"] = rev_pred
    out["rev_P_rev"] = P_rev
    out["rev_P_norev"] = P_norev

    def rev_decision(row, th=0.55):
        return "REVERSAL_RISK" if row["rev_P_rev"] >= th else "NORMAL"

    out["rev_decision"] = out.apply(rev_decision, axis=1)

    # ---------- 预测正确性评估 ----------
    def correctness(pred, real):
        if pd.isna(real):
            return np.nan, ""
        ok = (pred == real)
        if ok:
            return True, ""
        return False, f"pred={pred}, real={real}"

    # 波动
    vol_corr = out.apply(lambda r: correctness(r["vol_pred_regime"], r["real_regime"]), axis=1)
    out["vol_correct"] = [c[0] for c in vol_corr]
    out["vol_error_desc"] = [c[1] for c in vol_corr]

    # 方向
    dir_corr = out.apply(lambda r: correctness(r["dir_pred"], r["real_direction"]), axis=1)
    out["dir_correct"] = [c[0] for c in dir_corr]
    out["dir_error_desc"] = [c[1] for c in dir_corr]

    # 趋势
    trend_corr = out.apply(lambda r: correctness(r["trend_pred"], r["real_trend"]), axis=1)
    out["trend_correct"] = [c[0] for c in trend_corr]
    out["trend_error_desc"] = [c[1] for c in trend_corr]

    # 反转
    rev_corr = out.apply(lambda r: correctness(r["rev_pred"], r["real_reversal"]), axis=1)
    out["rev_correct"] = [c[0] for c in rev_corr]
    out["rev_error_desc"] = [c[1] for c in rev_corr]

    # ---------- 只保留目标日期存在的行，并限制起始于 2024-07-01 ----------
    out = out.dropna(subset=["date"]).copy()
    out = out[out["date"] >= pd.to_datetime("2024-07-01")].copy()

    # 按“目标日期”排序
    out = out.sort_values("date").reset_index(drop=True)

    # ---------- 与已有 log 合并（按日期覆盖） ----------
    if LOG_FILE.exists():
        try:
            old = pd.read_csv(LOG_FILE)
            old["date"] = pd.to_datetime(old["date"])
            old = old.set_index("date")
            new = out.set_index("date")

            # 统一列集合
            all_cols = sorted(set(old.columns) | set(new.columns))
            old = old.reindex(columns=all_cols)
            new = new.reindex(columns=all_cols)

            # 用新结果覆盖老结果
            combined = old.copy()
            combined.update(new)  # new 覆盖 old 相同 index+col
            # 对于新出现的日期，直接 append
            missing_idx = new.index.difference(combined.index)
            if len(missing_idx) > 0:
                combined = pd.concat([combined, new.loc[missing_idx]], axis=0)

            combined = combined.sort_index()
            combined.reset_index().to_csv(LOG_FILE, index=False)
        except Exception as e:
            print("Warning: failed to merge with existing log, overwriting. Error:", e)
            out.to_csv(LOG_FILE, index=False)
    else:
        combined = out.set_index("date").sort_index()
        combined.reset_index().to_csv(LOG_FILE, index=False)

    print(f"Log written to: {LOG_FILE}")


    # ========== 生成当日 txt 总结 ==========
    try:
        # 重新从 log 文件读取，避免 index/column 混乱
        df_log = pd.read_csv(LOG_FILE)
        if df_log.empty:
            print("Log is empty, skip daily summary.")
            return

        df_log["date"] = pd.to_datetime(df_log["date"])
        if "src_date" in df_log.columns:
            df_log["src_date"] = pd.to_datetime(df_log["src_date"])
        df_log = df_log.sort_values("date")

        latest_row = df_log.iloc[-1]

        # t = 特征日，t+1 = 目标日
        src_date = latest_row.get("src_date")
        if pd.notna(src_date):
            src_date = pd.to_datetime(src_date).date()
        target_date = latest_row.get("date")
        if pd.notna(target_date):
            target_date = pd.to_datetime(target_date).date()
        else:
            target_date = None

        close_src = latest_row.get("spx_close_src", np.nan)

        # 波动模型概率与 regime
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

        # 真实结果（可能为空）
        real_regime = latest_row.get("real_regime", np.nan)
        real_dir = latest_row.get("real_direction", np.nan)
        real_trend = latest_row.get("real_trend", np.nan)
        real_rev = latest_row.get("real_reversal", np.nan)

        # Regime 文本映射
        REGIME_MAP = {
            0: "Low volatility",
            1: "Medium volatility",
            2: "High volatility",
        }

        # 粗略历史波动分位（range%）
        RANGE_STATS = {
            0: (0.0045, 0.0059, 0.0075),  # 低波动
            1: (0.0091, 0.0108, 0.0128),  # 中等
            2: (0.0168, 0.0203, 0.0258),  # 高波动
        }

        if not pd.isna(pred_regime) and int(pred_regime) in RANGE_STATS and not pd.isna(close_src):
            p25, med, p75 = RANGE_STATS[int(pred_regime)]
            low_pts = p25 * close_src
            mid_pts = med * close_src
            high_pts = p75 * close_src
        else:
            p25 = med = p75 = low_pts = mid_pts = high_pts = np.nan

        def fmt_real(x):
            return "N/A" if pd.isna(x) else str(int(x))

        DIR_MAP = {-1: "Down", 0: "Neutral", 1: "Up"}

        lines = []
        lines.append("========================================")
        lines.append(f"Feature date (t)          : {src_date}")
        if target_date:
            lines.append(f"Predicting next day (t+1) : {target_date}")
        lines.append("----------------------------------------")
        if not pd.isna(close_src):
            lines.append(f"SPX close on t            : {close_src:.2f}")
        lines.append("")

        # 波动模型
        lines.append(f"[Volatility Model v3.0] ({VOL_MODEL_NAME})")
        lines.append(f"P_low  = {P_reg0:.3f}")
        lines.append(f"P_mid  = {P_reg1:.3f}")
        lines.append(f"P_high = {P_reg2:.3f}")
        reg_text = REGIME_MAP.get(int(pred_regime), "Unknown") if not pd.isna(pred_regime) else "Unknown"
        lines.append(f"Pred regime   : {pred_regime} ({reg_text})")
        lines.append(f"Decision      : {vol_action}")
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

        # 真实结果
        lines.append("----- Realized (if available) -----")
        if target_date:
            lines.append(f"Real date (t+1): {target_date}")
        lines.append(f"Real regime    : {fmt_real(real_regime)}")
        if pd.isna(real_dir):
            lines.append("Real direction : N/A")
        else:
            lines.append(f"Real direction : {real_dir} ({DIR_MAP.get(int(real_dir), 'Unknown')})")
        lines.append(f"Real trend     : {fmt_real(real_trend)}")
        lines.append(f"Real reversal  : {fmt_real(real_rev)}")
        lines.append("========================================")
        lines.append("")

        # 以特征日 t 命名 txt
        if src_date:
            txt_name = f"{src_date.isoformat()}_summary.txt"
        else:
            # 兜底：用 target_date
            base_date = target_date if target_date else df_log["date"].max().date()
            txt_name = f"{base_date.isoformat()}_summary.txt"

        txt_path = LOG_DIR / txt_name
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Daily summary written to: {txt_path}")
    except Exception as e:
        print("Warning: failed to write daily summary txt:", e)


if __name__ == "__main__":
    main()
