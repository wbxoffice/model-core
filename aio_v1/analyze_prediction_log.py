
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze rolling accuracy of 4 models based on prediction_log.csv

使用方法：
- 直接运行：统计最近 N_DAYS 的胜率（默认 60 天）
- 可根据需要修改 N_DAYS 数值
"""

from pathlib import Path
import pandas as pd
import numpy as np

# 路径配置（与 run_all_models_and_log.py 保持一致）
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_FILE = LOG_DIR / "prediction_log.csv"

# 最近 N 天
N_DAYS = 10  # 可改成 30 / 90 等


def rolling_accuracy():
    if not LOG_FILE.exists():
        print("Log file not found:", LOG_FILE)
        return

    df = pd.read_csv(LOG_FILE)
    df["date"] = pd.to_datetime(df["date"])

    # 只看有日期的行
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date")

    if df.empty:
        print("Log file is empty.")
        return

    # last_date = df["date"].max()
    # cutoff = last_date - pd.Timedelta(days=N_DAYS)
    # recent = df[df["date"] >= cutoff].copy()

    n = min(N_DAYS, len(df))
    recent = df.sort_values("date").tail(n+1).copy()

    print("========================================")
    print(f"Rolling accuracy analysis: last {N_DAYS} days")
    print(f"Date range: {recent['date'].min().date()} ~ {recent['date'].max().date()}")
    print("========================================")

    def acc_from_bool(col_name, label_col):
        sub = recent[recent[label_col].notna() & recent[col_name].notna()]
        if sub.empty:
            return 0, np.nan
        return len(sub), sub[col_name].mean()

    # 波动模型
    n_vol, acc_vol = acc_from_bool("vol_correct", "real_regime")
    # 方向模型
    n_dir, acc_dir = acc_from_bool("dir_correct", "real_direction")
    # 趋势模型
    n_trend, acc_trend = acc_from_bool("trend_correct", "real_trend")
    # 反转模型
    n_rev, acc_rev = acc_from_bool("rev_correct", "real_reversal")

    print("Samples & Accuracy (overall, including低置信度):")
    print(f"- Volatility model : N={n_vol:3d}, Acc={acc_vol*100:5.1f}%")
    print(f"- Direction model  : N={n_dir:3d}, Acc={acc_dir*100:5.1f}%")
    print(f"- Trend model      : N={n_trend:3d}, Acc={acc_trend*100:5.1f}%")
    print(f"- Reversal model   : N={n_rev:3d}, Acc={acc_rev*100:5.1f}%")
    print("")

    # 再加一组：高置信度区间的胜率
    print("High-confidence accuracy (方向: max(P_up,P_down)>0.5; 趋势: P_trend>0.6; 反转: P_rev>0.55)")

    # 方向高置信度
    if {"dir_P_up", "dir_P_down", "real_direction", "dir_pred"}.issubset(recent.columns):
        sub_dir = recent.copy()
        sub_dir = sub_dir[sub_dir["real_direction"].notna()]
        mask_dir = (sub_dir["dir_P_up"].notna()) & (sub_dir["dir_P_down"].notna())
        sub_dir = sub_dir[mask_dir]
        if not sub_dir.empty:
            max_p = sub_dir[["dir_P_up", "dir_P_down"]].max(axis=1)
            sub_dir = sub_dir[max_p > 0.5]
            if not sub_dir.empty:
                acc_hc_dir = (sub_dir["dir_pred"] == sub_dir["real_direction"]).mean()
                print(f"- Direction (HC) : N={len(sub_dir):3d}, Acc={acc_hc_dir*100:5.1f}%")
            else:
                print("- Direction (HC) : N=0")
        else:
            print("- Direction (HC) : N=0")

    # 趋势高置信度
    if {"trend_P_trend", "trend_pred", "real_trend"}.issubset(recent.columns):
        sub_tr = recent.copy()
        sub_tr = sub_tr[sub_tr["real_trend"].notna()]
        sub_tr = sub_tr[sub_tr["trend_P_trend"].notna()]
        sub_tr = sub_tr[sub_tr["trend_P_trend"] > 0.55]
        if not sub_tr.empty:
            acc_hc_tr = (sub_tr["trend_pred"] == sub_tr["real_trend"]).mean()
            print(f"- Trend (HC)     : N={len(sub_tr):3d}, Acc={acc_hc_tr*100:5.1f}%")
        else:
            print("- Trend (HC)     : N=0")

    # 反转高置信度
    if {"rev_P_rev", "rev_pred", "real_reversal"}.issubset(recent.columns):
        sub_rv = recent.copy()
        sub_rv = sub_rv[sub_rv["real_reversal"].notna()]
        sub_rv = sub_rv[sub_rv["rev_P_rev"].notna()]
        sub_rv = sub_rv[sub_rv["rev_P_rev"] > 0.55]
        if not sub_rv.empty:
            acc_hc_rv = (sub_rv["rev_pred"] == sub_rv["real_reversal"]).mean()
            print(f"- Reversal (HC)  : N={len(sub_rv):3d}, Acc={acc_hc_rv*100:5.1f}%")
        else:
            print("- Reversal (HC)  : N=0")

    print("========================================")


if __name__ == "__main__":
    rolling_accuracy()
