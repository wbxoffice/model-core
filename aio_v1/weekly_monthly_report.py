#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
weekly_monthly_report.py

根据 prediction_log.csv 生成最近一段时间的模型表现回顾报告。

- 只使用“已经有真实结果的最后一天”作为窗口末尾
- 报告内容包括：
  * Vol 模型总体准确率 & 高波日召回率
  * Direction / Trend / Reversal 总体准确率 & 信号命中率
  * 本窗口内「高波动日列表」（real_regime == 2）
  * 本窗口内所有「TREND_DAY 信号日列表」（trend_decision == 'TREND_DAY'）

配置：
- MODE = "weekly" 或 "monthly"
- WINDOW_DAYS: 周报一般 7 天，月报可以 30 天
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_FILE = LOG_DIR / "prediction_log.csv"

# ===== 手动配置区域 =====
MODE = "weekly"      # "weekly" 或 "monthly"
WINDOW_DAYS = 7      # 周报可用 7，月报可用 30


def _decision_mask_direction(df: pd.DataFrame):
    """Direction: 有信号 = LONG / SHORT / FLAT（排除 NO_SIGNAL）"""
    return df["dir_decision"].isin(["LONG", "SHORT", "FLAT"])


def _decision_mask_trend(df: pd.DataFrame):
    """Trend: 有信号 = TREND_DAY"""
    return df["trend_decision"] == "TREND_DAY"


def _decision_mask_reversal(df: pd.DataFrame):
    """Reversal: 有信号 = REVERSAL_RISK"""
    return df["rev_decision"] == "REVERSAL_RISK"


def main():
    log_path = LOG_FILE
    if not log_path.exists():
        print(f"[report] prediction_log 不存在: {log_path}")
        return

    df = pd.read_csv(log_path)
    if df.empty:
        print("[report] prediction_log 为空，跳过。")
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # ---------- 只用“已经有真实结果的日期”作为窗口末尾 ----------
    # 只要 real_regime / real_direction / real_trend / real_reversal 任意一个有值，就认为这一天已经结算
    label_mask = (
        df.get("real_regime").notna()
        | df.get("real_direction").notna()
        | df.get("real_trend").notna()
        | df.get("real_reversal").notna()
    )

    if not label_mask.any():
        print("[report] 目前 prediction_log 中没有任何真实结果（real_*），无法生成报告。")
        return

    last_labeled_ts = df.loc[label_mask, "date"].max()
    last_date = last_labeled_ts.date()

    # 窗口：从 last_date 往前 WINDOW_DAYS-1 天
    start_ts = last_labeled_ts - pd.Timedelta(days=WINDOW_DAYS - 1)

    # 只统计 [start_ts, last_labeled_ts] 这个区间
    df_win = df[(df["date"] >= start_ts) & (df["date"] <= last_labeled_ts)].copy()
    if df_win.empty:
        print(f"[report] 窗口内没有数据 ({start_ts.date()} ~ {last_date})")
        return

    # ========== Vol 模型 ==========
    vol_used = df_win[df_win["real_regime"].notna()].copy()
    vol_total = len(vol_used)

    if vol_total > 0:
        vol_acc = (vol_used["vol_pred_regime"] == vol_used["real_regime"]).mean()
        hv_mask = vol_used["real_regime"] == 2
        hv_total = hv_mask.sum()
        if hv_total > 0:
            hv_hit = (vol_used.loc[hv_mask, "vol_pred_regime"] == 2).mean()
        else:
            hv_hit = np.nan
    else:
        vol_acc = np.nan
        hv_total = 0
        hv_hit = np.nan

    # ========== Direction ==========
    dir_used = df_win[df_win["real_direction"].notna()].copy()
    dir_total = len(dir_used)
    if dir_total > 0:
        dir_acc = (dir_used["dir_pred"] == dir_used["real_direction"]).mean()
    else:
        dir_acc = np.nan

    dir_dec_mask = _decision_mask_direction(dir_used)
    dir_signal_days = dir_dec_mask.sum()
    if dir_signal_days > 0:
        dir_signal_hit = (
            dir_used.loc[dir_dec_mask, "dir_pred"] == dir_used.loc[dir_dec_mask, "real_direction"]
        ).mean()
    else:
        dir_signal_hit = np.nan

    # ========== Trend ==========
    trend_used = df_win[df_win["real_trend"].notna()].copy()
    trend_total = len(trend_used)
    if trend_total > 0:
        trend_acc = (trend_used["trend_pred"] == trend_used["real_trend"]).mean()
    else:
        trend_acc = np.nan

    trend_dec_mask = _decision_mask_trend(trend_used)
    trend_signal_days = trend_dec_mask.sum()
    if trend_signal_days > 0:
        trend_signal_hit = (
            trend_used.loc[trend_dec_mask, "trend_pred"] == trend_used.loc[trend_dec_mask, "real_trend"]
        ).mean()
    else:
        trend_signal_hit = np.nan

    # ========== Reversal ==========
    rev_used = df_win[df_win["real_reversal"].notna()].copy()
    rev_total = len(rev_used)
    if rev_total > 0:
        rev_acc = (rev_used["rev_pred"] == rev_used["real_reversal"]).mean()
    else:
        rev_acc = np.nan

    rev_dec_mask = _decision_mask_reversal(rev_used)
    rev_signal_days = rev_dec_mask.sum()
    if rev_signal_days > 0:
        rev_signal_hit = (
            rev_used.loc[rev_dec_mask, "rev_pred"] == rev_used.loc[rev_dec_mask, "real_reversal"]
        ).mean()
    else:
        rev_signal_hit = np.nan
        # 追加：REVERSAL_RISK 信号日列表
    rev_signal_list = rev_used[rev_dec_mask].copy()
    rev_signal_list = rev_signal_list.sort_values("date")
    # ========== 小表格数据准备 ==========

    # 高波动日列表（real_regime == 2）
    hv_days = vol_used[vol_used["real_regime"] == 2].copy()
    hv_days = hv_days.sort_values("date")

    # TREND_DAY 信号日列表（trend_decision == 'TREND_DAY'）
    trend_signal_list = df_win[df_win["trend_decision"] == "TREND_DAY"].copy()
    trend_signal_list = trend_signal_list.sort_values("date")

    # ========== 生成文本报告 ==========
    reports_dir = LOG_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "weekly" if MODE.lower().startswith("w") else "monthly"
    report_name = f"{mode_str}_report_{last_date}.txt"
    out_path = reports_dir / report_name

    lines = []
    lines.append("========================================")
    lines.append(f"SPX Model Performance Report ({mode_str})")
    lines.append(f"Window: {start_ts.date()} ~ {last_date}")
    lines.append("========================================")
    lines.append("")

    # Vol
    lines.append("[Volatility Model v3.0]")
    lines.append(f"- Samples with label    : {vol_total}")
    lines.append(f"- Overall accuracy      : {vol_acc:.3f}" if not np.isnan(vol_acc) else "- Overall accuracy      : N/A")
    lines.append(f"- High-vol days count   : {hv_total}")
    if hv_total > 0 and not np.isnan(hv_hit):
        lines.append(f"- High-vol recall       : {hv_hit:.3f}")
    else:
        lines.append("- High-vol recall       : N/A")
    lines.append("")

    # Direction
    lines.append("[Direction Model]")
    lines.append(f"- Samples with label    : {dir_total}")
    lines.append(f"- Overall accuracy      : {dir_acc:.3f}" if not np.isnan(dir_acc) else "- Overall accuracy      : N/A")
    lines.append(f"- Signal days (≠ NO_SIGNAL): {dir_signal_days}")
    if not np.isnan(dir_signal_hit):
        lines.append(f"- Signal hit rate       : {dir_signal_hit:.3f}")
    else:
        lines.append("- Signal hit rate       : N/A")
    lines.append("")

    # Trend
    lines.append("[Trend Model]")
    lines.append(f"- Samples with label    : {trend_total}")
    lines.append(f"- Overall accuracy      : {trend_acc:.3f}" if not np.isnan(trend_acc) else "- Overall accuracy      : N/A")
    lines.append(f"- Signal days (TREND_DAY): {trend_signal_days}")
    if not np.isnan(trend_signal_hit):
        lines.append(f"- Signal hit rate       : {trend_signal_hit:.3f}")
    else:
        lines.append("- Signal hit rate       : N/A")
    lines.append("")

    # Reversal
    lines.append("[Reversal Model]")
    lines.append(f"- Samples with label    : {rev_total}")
    lines.append(f"- Overall accuracy      : {rev_acc:.3f}" if not np.isnan(rev_acc) else "- Overall accuracy      : N/A")
    lines.append(f"- Signal days (REVERSAL_RISK): {rev_signal_days}")
    if not np.isnan(rev_signal_hit):
        lines.append(f"- Signal hit rate       : {rev_signal_hit:.3f}")
    else:
        lines.append("- Signal hit rate       : N/A")
    lines.append("")

    # ========== 附录 1：高波动日列表 ==========
    lines.append("---------- High Volatility Days (real_regime = 2) ----------")
    if hv_days.empty:
        lines.append("(No high-vol days in this window.)")
    else:
        lines.append("date       | vol_pred | vol_decision | correct?")
        lines.append("-----------+----------+-------------+---------")
        for _, r in hv_days.iterrows():
            d = r["date"].date()
            pred = r.get("vol_pred_regime", np.nan)
            dec = r.get("vol_decision", "")
            corr = r.get("vol_correct", np.nan)
            corr_str = "Y" if (not pd.isna(corr) and bool(corr)) else ("N" if not pd.isna(corr) else "?")
            lines.append(f"{d} | {int(pred) if not pd.isna(pred) else 'NA':>8} | {dec:<11} | {corr_str:>3}")
    lines.append("")

    # ========== 附录 2：TREND_DAY 信号日列表 ==========
    lines.append("---------- TREND_DAY Signal Days (trend_decision = 'TREND_DAY') ----------")
    if trend_signal_list.empty:
        lines.append("(No TREND_DAY signals in this window.)")
    else:
        lines.append("date       | pred_trend | real_trend | correct?")
        lines.append("-----------+------------+-----------+---------")
        for _, r in trend_signal_list.iterrows():
            d = r["date"].date()
            pred = r.get("trend_pred", np.nan)
            real = r.get("real_trend", np.nan)
            corr = r.get("trend_correct", np.nan)
            corr_str = "Y" if (not pd.isna(corr) and bool(corr)) else ("N" if not pd.isna(corr) else "?")
            pred_str = str(int(pred)) if not pd.isna(pred) else "NA"
            real_str = str(int(real)) if not pd.isna(real) else "NA"
            lines.append(f"{d} | {pred_str:>10} | {real_str:>9} | {corr_str:>3}")
    lines.append("")

    # ========== 附录 3：REVERSAL_RISK 信号日列表 ==========
    lines.append("---------- REVERSAL_RISK Signal Days (rev_decision = 'REVERSAL_RISK') ----------")
    if rev_signal_list.empty:
        lines.append("(No REVERSAL_RISK signals in this window.)")
    else:
        lines.append("date       | pred_rev | real_rev | correct?")
        lines.append("-----------+----------+----------+---------")
        for _, r in rev_signal_list.iterrows():
            d = r["date"].date()
            pred = r.get("rev_pred", np.nan)
            real = r.get("real_reversal", np.nan)
            corr = r.get("rev_correct", np.nan)
            corr_str = "Y" if (not pd.isna(corr) and bool(corr)) else ("N" if not pd.isna(corr) else "?")
            pred_str = str(int(pred)) if not pd.isna(pred) else "NA"
            real_str = str(int(real)) if not pd.isna(real) else "NA"
            lines.append(f"{d} | {pred_str:>8} | {real_str:>8} | {corr_str:>3}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[report] Written {mode_str} report to: {out_path}")


if __name__ == "__main__":
    main()
