#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析 strategy_v1_ALL_SCHEMES_YYYYMMDD_YYYYMMDD.csv 的：
- 总体指标（合并全部 scheme）
- 按 scheme 指标对比
- 按 trade_type 表现（全局 + 分 scheme）
- 按 vol_pred_regime 表现（全局 + 分 scheme）
- 年度 / 月度分解 (Year × trade_type / regime / month)
- 单笔风险：理论最大亏损 & 风险占比分布

输出示例：
- STRATEGY PnL backtest/scheme_performance_summary_v1.csv
- STRATEGY PnL backtest/strategy_performance_by_type_v1.csv
- STRATEGY PnL backtest/strategy_performance_by_type_and_scheme_v1.csv
- STRATEGY PnL backtest/performance_by_vol_regime_v1.csv
- STRATEGY PnL backtest/performance_by_vol_regime_and_scheme_v1.csv
- STRATEGY PnL backtest/yearly_performance_v1.csv
- STRATEGY PnL backtest/yearly_trade_type_breakdown_v1.csv
- STRATEGY PnL backtest/yearly_regime_breakdown_v1.csv
- STRATEGY PnL backtest/monthly_scheme_breakdown_v1.csv
- STRATEGY PnL backtest/monthly_total_pnl_rank_v1.csv
- STRATEGY PnL backtest/risk_distribution_overall_v1.csv
- STRATEGY PnL backtest/risk_distribution_by_type_v1.csv

图表示例（放在 STRATEGY PnL backtest/figs_v1 下）：
- equity_curve_by_scheme.png
- total_pnl_by_scheme.png
- annual_return_by_scheme.png
- total_pnl_by_trade_type.png
- total_pnl_by_vol_regime.png
- pnl_distribution.png
- annual_return_by_year_and_scheme.png
- yearly_pnl_by_scheme.png
- yearly_max_dd_by_scheme.png
- heatmap_year_trade_type_<scheme>.png
- heatmap_year_regime_<scheme>.png
- heatmap_month_scheme_<scheme>.png
- monthly_total_pnl_bar.png
"""

from pathlib import Path
import argparse

import pandas as pd

try:
    from shared.local_settings import account_equity
except ImportError:
    from local_settings import account_equity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ================== 路径 & 全局配置 ==================

# 共享实现（2026-06-12 去重）：原 spx_vdtr_v2 / us500_vdtr_v2 各持有一份逐字节相同的副本。
# 路径不再在模块级解析，由各项目薄包装通过 configure()/main(base_dir=...) 注入。
BASE_DIR = None
RESULT_ROOT = None
FIG_DIR = None


def configure(base_dir):
    """绑定项目目录并初始化输出路径（替代原模块级路径解析）。"""
    global BASE_DIR, RESULT_ROOT, FIG_DIR
    BASE_DIR = Path(base_dir)
    RESULT_ROOT = BASE_DIR / "STRATEGY PnL backtest"
    FIG_DIR = RESULT_ROOT / "figs_v1"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

# 必须与 backtest 脚本中的配置保持一致
INITIAL_EQUITY = account_equity()  # 初始资金：取 local_settings.json（私有，已 gitignore）

# 和 backtest 中一致的最大亏损点数（用于重建理论最大亏损）
MAX_LOSS_POINTS = {
    "LVL_RANGE_SHORT_CALL_SPREAD": 3.0,
    "LVL_RANGE_SHORT_PUT_SPREAD": 3.0,
    "LOWVOL_TREND_LONG_CALL_SPREAD": 2.7,
    "LOWVOL_TREND_LONG_PUT_SPREAD": 2.7,
    "MIDVOL_TREND_LONG_CALL_SPREAD": 2.7,
    "MIDVOL_TREND_LONG_PUT_SPREAD": 2.7,
    "HIGHVOL_LONG_STRANGLE": 6.0,   # 3+3 点
}


# ================== 工具函数 ==================

def find_latest_all_schemes_csv(pattern_prefix: str = "strategy_v1_ALL_SCHEMES_") -> Path | None:
    """
    在 STRATEGY PnL backtest/** 中查找最新的 strategy_v1_ALL_SCHEMES_*.csv
    """
    pattern = pattern_prefix + "*.csv"
    paths = sorted(RESULT_ROOT.rglob(pattern))
    if not paths:
        print(f"[WARN] 未在 {RESULT_ROOT} 下找到 {pattern}")
        return None
    latest = paths[-1]
    print(f"[INFO] 自动使用最新的 ALL_SCHEMES 文件: {latest}")
    return latest


def max_drawdown(equity_curve: np.ndarray):
    """
    返回:
    - 最大回撤(绝对金额)
    - 回撤序列
    """
    running_max = np.maximum.accumulate(equity_curve)
    dd = running_max - equity_curve
    return dd.max(), dd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252):
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return np.nan
    if returns.std() == 0 or np.isnan(returns.std()):
        return np.nan
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252):
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return np.nan
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0 or np.isnan(downside.std()):
        return np.nan
    return (returns.mean() / downside.std()) * np.sqrt(periods_per_year)


def calmar_ratio(annual_return: float, max_dd_ratio: float):
    """
    Calmar = 年化收益 / 最大回撤比例
    这里 max_dd_ratio 是 max_drawdown / equity_peak，例如 0.2 表示 20% 最大回撤
    """
    if max_dd_ratio == 0 or np.isnan(max_dd_ratio):
        return np.nan
    return annual_return / abs(max_dd_ratio)


def compute_block_metrics(df_block: pd.DataFrame, label: str = "") -> dict:
    """
    对一个子集（可以是整体，也可以是某个 scheme）计算：
    - trades, win_rate, avg_win, avg_loss, reward/risk, expectation
    - max_dd(绝对值), max_dd_ratio(%)
    - annual_return (比例，例如 0.7 = 70%)
    - Sharpe, Sortino, Calmar
    注意：
    - 使用该 block 自身的 equity 曲线和日期范围
    """
    if df_block.empty:
        return {
            "label": label,
            "trades": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "reward_risk": np.nan,
            "expectation": np.nan,
            "max_dd": np.nan,
            "max_dd_ratio": np.nan,
            "annual_return": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "final_equity": np.nan,
        }

    df_block = df_block.sort_values("date").reset_index(drop=True)

    # 真实有下单的行: trade_type 不以 "NO_TRADE" 开头
    traded = df_block[~df_block["trade_type"].astype(str).str.startswith("NO_TRADE")].copy()

    if traded.empty:
        equity_curve = df_block["equity"].to_numpy()
        max_dd_val, _ = max_drawdown(equity_curve)
        peak = np.max(equity_curve) if len(equity_curve) > 0 else np.nan
        max_dd_ratio = max_dd_val / peak if peak and peak > 0 else np.nan
        return {
            "label": label,
            "trades": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "reward_risk": np.nan,
            "expectation": np.nan,
            "max_dd": max_dd_val,
            "max_dd_ratio": max_dd_ratio,
            "annual_return": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "final_equity": equity_curve[-1] if len(equity_curve) else np.nan,
        }

    wins = traded[traded["pnl"] > 0]
    losses = traded[traded["pnl"] < 0]

    win_rate = len(wins) / len(traded) if len(traded) > 0 else np.nan
    avg_win = wins["pnl"].mean() if not wins.empty else np.nan
    avg_loss = losses["pnl"].mean() if not losses.empty else np.nan
    reward_risk = avg_win / abs(avg_loss) if (avg_loss is not None and avg_loss < 0) else np.nan
    expectation = traded["pnl"].mean()

    # equity 曲线
    equity_curve = df_block["equity"].to_numpy()
    max_dd_val, dd_series = max_drawdown(equity_curve)
    peak = np.max(equity_curve) if len(equity_curve) > 0 else np.nan
    max_dd_ratio = max_dd_val / peak if peak and peak > 0 else np.nan

    initial_equity = equity_curve[0]
    final_equity = equity_curve[-1]

    # 用 unique 日期估算年数
    unique_days = pd.to_datetime(df_block["date"]).dt.normalize().nunique()
    years = unique_days / 252.0 if unique_days > 0 else np.nan
    if years and years > 0 and initial_equity > 0:
        annual_return = (final_equity / initial_equity) ** (1.0 / years) - 1.0
    else:
        annual_return = np.nan

    # 日收益率：用每日 equity 曲线来算（包含无交易日）
    daily_equity = (
        df_block.groupby("date")["equity"]
        .last()
        .sort_index()
    )
    daily_ret = daily_equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sr = sharpe_ratio(daily_ret) if not daily_ret.empty else np.nan
    sor = sortino_ratio(daily_ret) if not daily_ret.empty else np.nan
    cal = calmar_ratio(annual_return, max_dd_ratio)

    return {
        "label": label,
        "trades": len(traded),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "reward_risk": reward_risk,
        "expectation": expectation,
        "max_dd": max_dd_val,
        "max_dd_ratio": max_dd_ratio,
        "annual_return": annual_return,
        "sharpe": sr,
        "sortino": sor,
        "calmar": cal,
        "final_equity": final_equity,
    }


# ================== 主分析函数 ==================

def analyze_backtest(csv_file: Path):
    if not csv_file.exists():
        print("找不到文件：", csv_file)
        return

    print(f"[INFO] 读取回测文件: {csv_file}")
    df = pd.read_csv(csv_file)

    # 基础预处理
    df["date"] = pd.to_datetime(df["date"])
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    if "n_contracts" in df.columns:
        df["n_contracts"] = pd.to_numeric(df["n_contracts"], errors="coerce").fillna(0).astype(int)
    else:
        df["n_contracts"] = 0

    df["trade_type"] = df["trade_type"].astype(str)
    df = df.sort_values(["scheme", "date"]).reset_index(drop=True)

    # 标记有无交易
    df["is_trade"] = ~df["trade_type"].str.startswith("NO_TRADE")

    # ========== 新增：每笔交易的理论最大亏损 & 风险占比 ==========

    # 近似“开仓前权益” = 当日结束权益 - 当日PnL
    df["equity_before_trade"] = df["equity"] - df["pnl"]

    df["max_loss_pts"] = df["trade_type"].map(MAX_LOSS_POINTS).fillna(0.0)
    df["max_loss_theoretical"] = df["max_loss_pts"] * 100.0 * df["n_contracts"]

    # 风险占当日 equity_before 的比例
    df["risk_pct_equity_before"] = np.where(
        df["equity_before_trade"] > 0,
        df["max_loss_theoretical"] / df["equity_before_trade"],
        np.nan,
    )

    # ===== 1. 总体（合并所有 scheme） =====
    overall_metrics = compute_block_metrics(df, label="ALL_SCHEMES")

    print("\n========== 总体策略表现（合并所有 scheme） ==========")
    for k, v in overall_metrics.items():
        if k == "label":
            continue
        if isinstance(v, (float, np.floating)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # ===== 2. 按 scheme 分析 =====
    print("\n========== 按 scheme 分析（对比用） ==========")
    scheme_rows = []
    for scheme, sub in df.groupby("scheme"):
        met = compute_block_metrics(sub, label=scheme)
        scheme_rows.append(met)

    df_scheme = pd.DataFrame(scheme_rows).set_index("label")
    df_scheme = df_scheme.sort_values("final_equity", ascending=False)
    print(df_scheme.to_string())

    out_scheme = RESULT_ROOT / "scheme_performance_summary_v1.csv"
    df_scheme.to_csv(out_scheme, encoding="utf-8-sig")
    print("\n已输出 scheme-level 分析：", out_scheme)

    # ===== 3. 按 trade_type（全局 + 分 scheme） =====
    traded_all = df[df["is_trade"]].copy()
    print("\n========== 按 trade_type 分析（合并所有 scheme） ==========")
    rows_type = []
    for g, sub in traded_all.groupby("trade_type"):
        wins = sub[sub["pnl"] > 0]
        losses = sub[sub["pnl"] < 0]
        sub_sorted = sub.sort_values("date")
        mdd_val, _ = max_drawdown(sub_sorted["equity"].to_numpy())

        row = {
            "trade_type": g,
            "trades": len(sub),
            "win_rate": len(wins) / len(sub) if len(sub) > 0 else np.nan,
            "avg_win": wins["pnl"].mean() if not wins.empty else np.nan,
            "avg_loss": losses["pnl"].mean() if not losses.empty else np.nan,
            "reward/risk": (wins["pnl"].mean() / abs(losses["pnl"].mean())) if len(losses) else np.nan,
            "total_pnl": sub["pnl"].sum(),
            "max_dd": mdd_val,
        }
        rows_type.append(row)

    df_strategy = pd.DataFrame(rows_type).sort_values("total_pnl", ascending=False)
    print(df_strategy.to_string(index=False))

    out_type = RESULT_ROOT / "strategy_performance_by_type_v1.csv"
    df_strategy.to_csv(out_type, index=False, encoding="utf-8-sig")
    print("\n已输出 trade_type 分析（全局）：", out_type)

    # --- 新增：按 scheme + trade_type 细分 ---
    rows_type_scheme = []
    for (scheme, t), sub in traded_all.groupby(["scheme", "trade_type"]):
        wins = sub[sub["pnl"] > 0]
        losses = sub[sub["pnl"] < 0]
        sub_sorted = sub.sort_values("date")
        mdd_val, _ = max_drawdown(sub_sorted["equity"].to_numpy())
        rows_type_scheme.append(
            {
                "scheme": scheme,
                "trade_type": t,
                "trades": len(sub),
                "win_rate": len(wins) / len(sub) if len(sub) > 0 else np.nan,
                "avg_win": wins["pnl"].mean() if not wins.empty else np.nan,
                "avg_loss": losses["pnl"].mean() if not losses.empty else np.nan,
                "reward/risk": (wins["pnl"].mean() / abs(losses["pnl"].mean())) if len(losses) else np.nan,
                "total_pnl": sub["pnl"].sum(),
                "max_dd": mdd_val,
            }
        )
    df_type_scheme = pd.DataFrame(rows_type_scheme).sort_values(
        ["scheme", "total_pnl"], ascending=[True, False]
    )
    out_type_scheme = RESULT_ROOT / "strategy_performance_by_type_and_scheme_v1.csv"
    df_type_scheme.to_csv(out_type_scheme, index=False, encoding="utf-8-sig")
    print("已输出 trade_type × scheme 分析：", out_type_scheme)

    # ===== 4. 按 vol_pred_regime（全局 + 分 scheme） =====
    if "vol_pred_regime" in df.columns:
        print("\n========== 按 vol_pred_regime 分析（合并所有 scheme） ==========")
        rows_vol = []
        for rg, sub in traded_all.groupby("vol_pred_regime"):
            wins = sub[sub["pnl"] > 0]
            losses = sub[sub["pnl"] < 0]
            sub_sorted = sub.sort_values("date")
            mdd_val, _ = max_drawdown(sub_sorted["equity"].to_numpy())

            row = {
                "vol_regime": rg,
                "trades": len(sub),
                "win_rate": len(wins) / len(sub) if len(sub) > 0 else np.nan,
                "avg_win": wins["pnl"].mean() if not wins.empty else np.nan,
                "avg_loss": losses["pnl"].mean() if not losses.empty else np.nan,
                "reward/risk": (wins["pnl"].mean() / abs(losses["pnl"].mean())) if len(losses) else np.nan,
                "total_pnl": sub["pnl"].sum(),
                "max_dd": mdd_val,
            }
            rows_vol.append(row)

        df_vol = pd.DataFrame(rows_vol).sort_values("vol_regime")
        print(df_vol.to_string(index=False))

        out_vol = RESULT_ROOT / "performance_by_vol_regime_v1.csv"
        df_vol.to_csv(out_vol, index=False, encoding="utf-8-sig")
        print("已输出 vol_regime 分析（全局）：", out_vol)
    else:
        df_vol = pd.DataFrame()
        print("\n[WARN] CSV 中没有 vol_pred_regime 列，跳过按 vol 分析。")

    # --- 新增：按 scheme + vol_regime 细分 ---
    rows_vol_scheme = []
    if "vol_pred_regime" in df.columns:
        for (scheme, rg), sub in traded_all.groupby(["scheme", "vol_pred_regime"]):
            wins = sub[sub["pnl"] > 0]
            losses = sub[sub["pnl"] < 0]
            sub_sorted = sub.sort_values("date")
            mdd_val, _ = max_drawdown(sub_sorted["equity"].to_numpy())

            rows_vol_scheme.append(
                {
                    "scheme": scheme,
                    "vol_regime": rg,
                    "trades": len(sub),
                    "win_rate": len(wins) / len(sub) if len(sub) > 0 else np.nan,
                    "avg_win": wins["pnl"].mean() if not wins.empty else np.nan,
                    "avg_loss": losses["pnl"].mean() if not losses.empty else np.nan,
                    "reward/risk": (wins["pnl"].mean() / abs(losses["pnl"].mean())) if len(losses) else np.nan,
                    "total_pnl": sub["pnl"].sum(),
                    "max_dd": mdd_val,
                }
            )

        df_vol_scheme = pd.DataFrame(rows_vol_scheme).sort_values(
            ["scheme", "vol_regime"]
        )
        out_vol_scheme = RESULT_ROOT / "performance_by_vol_regime_and_scheme_v1.csv"
        df_vol_scheme.to_csv(out_vol_scheme, index=False, encoding="utf-8-sig")
        print("已输出 vol_regime × scheme 分析：", out_vol_scheme)

    # ===== 5. 单笔风险分布（overall + 按 trade_type） =====
    print("\n========== 单笔风险占比分布（risk_pct_equity_before） ==========")
    risk_trades = traded_all.copy()
    risk_trades = risk_trades[risk_trades["max_loss_theoretical"] > 0]

    def summarize_risk(series: pd.Series) -> dict:
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return {"count": 0, "median": np.nan, "p75": np.nan, "p90": np.nan, "max": np.nan}
        return {
            "count": len(s),
            "median": s.median(),
            "p75": s.quantile(0.75),
            "p90": s.quantile(0.90),
            "max": s.max(),
        }

    overall_risk = summarize_risk(risk_trades["risk_pct_equity_before"])
    df_risk_overall = pd.DataFrame([overall_risk])
    out_risk_overall = RESULT_ROOT / "risk_distribution_overall_v1.csv"
    df_risk_overall.to_csv(out_risk_overall, index=False, encoding="utf-8-sig")
    print("已输出总体风险分布：", out_risk_overall)
    print(df_risk_overall)

    # 按 trade_type 拆分风险分布
    rows_risk_type = []
    for t, sub in risk_trades.groupby("trade_type"):
        stats = summarize_risk(sub["risk_pct_equity_before"])
        stats["trade_type"] = t
        rows_risk_type.append(stats)
    df_risk_type = pd.DataFrame(rows_risk_type)[
        ["trade_type", "count", "median", "p75", "p90", "max"]
    ].sort_values("median", ascending=False)
    out_risk_type = RESULT_ROOT / "risk_distribution_by_type_v1.csv"
    df_risk_type.to_csv(out_risk_type, index=False, encoding="utf-8-sig")
    print("已输出按 trade_type 风险分布：", out_risk_type)
    print(df_risk_type)

    # ===== 6. 各类图表 =====
    print("\n========== 生成图表 ==========")

    # 6.1 各 scheme 的 equity 曲线
    plt.figure(figsize=(10, 6))
    for scheme, sub in df.groupby("scheme"):
        sub_sorted = sub.sort_values("date")
        plt.plot(sub_sorted["date"], sub_sorted["equity"], label=scheme)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Equity Curve by Scheme (strategy_v1)")
    plt.legend()
    plt.tight_layout()
    fig1_path = FIG_DIR / "equity_curve_by_scheme.png"
    plt.savefig(fig1_path, dpi=150)
    plt.close()
    print("已保存：", fig1_path)

    # 6.2 按 scheme 的 total_pnl 柱状图（用 final_equity - INITIAL_EQUITY）
    plt.figure(figsize=(8, 5))
    plt.bar(df_scheme.index, df_scheme["final_equity"] - INITIAL_EQUITY)
    plt.ylabel("Total PnL (Final equity - initial)")
    plt.title("Total PnL by Scheme (strategy_v1)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig2_path = FIG_DIR / "total_pnl_by_scheme.png"
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print("已保存：", fig2_path)

    # 6.3 按 scheme 的年化收益柱状图（这里 annual_return 已是比例，例如 0.7=70%）
    plt.figure(figsize=(8, 5))
    plt.bar(df_scheme.index, df_scheme["annual_return"])
    plt.ylabel("Annual Return")
    plt.title("Annualized Return by Scheme (strategy_v1)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig3_path = FIG_DIR / "annual_return_by_scheme.png"
    plt.savefig(fig3_path, dpi=150)
    plt.close()
    print("已保存：", fig3_path)

    # 6.4 按 trade_type 的 total_pnl 柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(df_strategy["trade_type"], df_strategy["total_pnl"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total PnL")
    plt.title("Total PnL by Trade Type (strategy_v1, all schemes)")
    plt.tight_layout()
    fig4_path = FIG_DIR / "total_pnl_by_trade_type.png"
    plt.savefig(fig4_path, dpi=150)
    plt.close()
    print("已保存：", fig4_path)

    # 6.5 按 vol_regime 的 total_pnl 柱状图（全局）
    if not df_vol.empty:
        plt.figure(figsize=(6, 4))
        plt.bar(df_vol["vol_regime"].astype(str), df_vol["total_pnl"])
        plt.xlabel("vol_pred_regime (0=low,1=mid,2=high)")
        plt.ylabel("Total PnL")
        plt.title("Total PnL by Vol Regime (strategy_v1, all schemes)")
        plt.tight_layout()
        fig5_path = FIG_DIR / "total_pnl_by_vol_regime.png"
        plt.savefig(fig5_path, dpi=150)
        plt.close()
        print("已保存：", fig5_path)

    # 6.6 单笔 PnL 分布直方图（所有 scheme 合并）
    plt.figure(figsize=(8, 5))
    plt.hist(traded_all["pnl"], bins=50)
    plt.xlabel("PnL per Trade")
    plt.ylabel("Count")
    plt.title("Distribution of Trade PnL (strategy_v1, all schemes)")
    plt.tight_layout()
    fig6_path = FIG_DIR / "pnl_distribution.png"
    plt.savefig(fig6_path, dpi=150)
    plt.close()
    print("已保存：", fig6_path)

    # ===== 7. 年度分解 (Yearly Breakdown) =====
    print("\n========== 分年份收益 (Yearly Breakdown) ==========")

    df["year"] = df["date"].dt.year

    yearly_rows = []
    for scheme, sub_scheme in df.groupby("scheme"):
        for year, sub_year in sub_scheme.groupby("year"):
            sub_year = sub_year.sort_values("date")
            if len(sub_year) == 0:
                continue

            eq_start = sub_year["equity"].iloc[0]
            eq_end = sub_year["equity"].iloc[-1]
            pnl_year = eq_end - eq_start
            ret_year = pnl_year / eq_start if eq_start > 0 else np.nan
            mdd_year, _ = max_drawdown(sub_year["equity"].to_numpy())

            yearly_rows.append({
                "scheme": scheme,
                "year": year,
                "trades": sub_year["is_trade"].sum(),
                "pnl_year": pnl_year,
                "return_year": ret_year,
                "max_dd_year": mdd_year,
                "eq_start": eq_start,
                "eq_end": eq_end,
            })

    df_yearly = pd.DataFrame(yearly_rows).sort_values(["scheme", "year"])
    out_yearly = RESULT_ROOT / "yearly_performance_v1.csv"
    df_yearly.to_csv(out_yearly, index=False, encoding="utf-8-sig")
    print("已输出 yearly breakdown:", out_yearly)
    print(df_yearly)

    # 年度图：收益、PnL、最大回撤
    # 7.1 Annual Return by Year & Scheme
    plt.figure(figsize=(12, 6))
    for scheme in df_yearly["scheme"].unique():
        sub = df_yearly[df_yearly["scheme"] == scheme]
        plt.plot(sub["year"], sub["return_year"], marker="o", label=scheme)
    plt.xlabel("Year")
    plt.ylabel("Annual Return (per scheme per year)")
    plt.title("Annual Returns by Scheme (strategy_v1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_y1 = FIG_DIR / "annual_return_by_year_and_scheme.png"
    plt.savefig(fig_y1, dpi=150)
    plt.close()
    print("已保存图：", fig_y1)

    # 7.2 Yearly PnL by Scheme
    plt.figure(figsize=(12, 6))
    for scheme in df_yearly["scheme"].unique():
        sub = df_yearly[df_yearly["scheme"] == scheme]
        plt.plot(sub["year"], sub["pnl_year"], marker="o", label=scheme)
    plt.xlabel("Year")
    plt.ylabel("PnL (USD)")
    plt.title("Yearly PnL by Scheme (strategy_v1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_y2 = FIG_DIR / "yearly_pnl_by_scheme.png"
    plt.savefig(fig_y2, dpi=150)
    plt.close()
    print("已保存图：", fig_y2)

    # 7.3 Yearly Max Drawdown by Scheme
    plt.figure(figsize=(12, 6))
    for scheme in df_yearly["scheme"].unique():
        sub = df_yearly[df_yearly["scheme"] == scheme]
        plt.plot(sub["year"], sub["max_dd_year"], marker="o", label=scheme)
    plt.xlabel("Year")
    plt.ylabel("Max Drawdown (USD)")
    plt.title("Yearly Max Drawdown by Scheme (strategy_v1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_y3 = FIG_DIR / "yearly_max_dd_by_scheme.png"
    plt.savefig(fig_y3, dpi=150)
    plt.close()
    print("已保存图：", fig_y3)

    # ===== 8. 年度分解：trade_type / regime / month =====
    print("\n========== 年度分解：trade_type / regime / month ==========")

    df_traded = df[df["is_trade"]].copy()
    df_traded["month"] = df_traded["date"].dt.month

    # 8.1 Year × trade_type 贡献表
    yearly_trade = (
        df_traded.groupby(["scheme", "year", "trade_type"])["pnl"]
        .sum()
        .reset_index()
        .sort_values(["scheme", "year", "pnl"], ascending=[True, True, False])
    )
    out_ytrade = RESULT_ROOT / "yearly_trade_type_breakdown_v1.csv"
    yearly_trade.to_csv(out_ytrade, index=False, encoding="utf-8-sig")
    print("已输出:", out_ytrade)

    for scheme in df["scheme"].unique():
        pivot_tt = yearly_trade[yearly_trade["scheme"] == scheme].pivot(
            index="trade_type", columns="year", values="pnl"
        )
        plt.figure(figsize=(14, 6))
        sns.heatmap(pivot_tt, annot=True, fmt=".0f", cmap="RdYlGn")
        plt.title(f"Year × Trade Type PnL — {scheme} (strategy_v1)")
        plt.tight_layout()
        fig_path = FIG_DIR / f"heatmap_year_trade_type_{scheme}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print("保存图：", fig_path)

    # 8.2 Year × vol_regime 贡献表
    if "vol_pred_regime" in df_traded.columns:
        yearly_regime = (
            df_traded.groupby(["scheme", "year", "vol_pred_regime"])["pnl"]
            .sum()
            .reset_index()
            .rename(columns={"vol_pred_regime": "regime"})
            .sort_values(["scheme", "year", "pnl"], ascending=[True, True, False])
        )
        out_yreg = RESULT_ROOT / "yearly_regime_breakdown_v1.csv"
        yearly_regime.to_csv(out_yreg, index=False, encoding="utf-8-sig")
        print("已输出:", out_yreg)

        for scheme in df["scheme"].unique():
            pivot_rg = yearly_regime[yearly_regime["scheme"] == scheme].pivot(
                index="regime", columns="year", values="pnl"
            )
            plt.figure(figsize=(10, 5))
            sns.heatmap(pivot_rg, annot=True, fmt=".0f", cmap="RdYlGn")
            plt.title(f"Year × Vol Regime PnL — {scheme} (strategy_v1)")
            plt.tight_layout()
            fig_path = FIG_DIR / f"heatmap_year_regime_{scheme}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print("保存图：", fig_path)

    # 8.3 Month × scheme（每月收益）
    monthly_scheme = (
        df_traded.groupby(["scheme", "year", "month"])["pnl"]
        .sum()
        .reset_index()
        .sort_values(["scheme", "year", "month"])
    )
    out_msch = RESULT_ROOT / "monthly_scheme_breakdown_v1.csv"
    monthly_scheme.to_csv(out_msch, index=False, encoding="utf-8-sig")
    print("已输出:", out_msch)

    for scheme in df["scheme"].unique():
        sub = monthly_scheme[monthly_scheme["scheme"] == scheme]
        pivot_m = sub.pivot(index="month", columns="year", values="pnl")
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_m, annot=True, fmt=".0f", cmap="RdBu_r")
        plt.title(f"Monthly PnL Heatmap — {scheme} (strategy_v1)")
        plt.tight_layout()
        fig_path = FIG_DIR / f"heatmap_month_scheme_{scheme}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print("保存图：", fig_path)

    # 8.4 哪些月份最赚钱 / 最危险（全 scheme 合并）
    monthly_total = (
        df_traded.groupby("month")["pnl"]
        .sum()
        .reset_index()
        .sort_values("pnl", ascending=False)
    )
    out_best_month = RESULT_ROOT / "monthly_total_pnl_rank_v1.csv"
    monthly_total.to_csv(out_best_month, index=False, encoding="utf-8-sig")
    print("已输出全年最强/最弱月份：", out_best_month)

    plt.figure(figsize=(10, 5))
    plt.bar(monthly_total["month"], monthly_total["pnl"])
    plt.title("Monthly Total PnL (All Schemes, strategy_v1)")
    plt.xlabel("Month")
    plt.ylabel("PnL (USD)")
    plt.grid(True)
    plt.tight_layout()
    fig_path = FIG_DIR / "monthly_total_pnl_bar.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("保存图：", fig_path)

    print("\n[DONE] 全量分析完成，可打开各 CSV 和 PNG 进一步查看细节。")


# ================== CLI ==================

def main(base_dir=None):
    if base_dir is not None:
        configure(base_dir)
    if RESULT_ROOT is None:
        raise RuntimeError("pnl_analysis_v1: 需先 configure(base_dir) 或 main(base_dir=...)")
    parser = argparse.ArgumentParser(
        description="Analyze strategy_v1 ALL_SCHEMES backtest results (full statistics + risk metrics)."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="指定 strategy_v1_ALL_SCHEMES_*.csv 文件路径; 若不指定则自动搜索最新的。",
    )
    args = parser.parse_args()

    if args.file is not None:
        csv_file = Path(args.file)
    else:
        csv_file = find_latest_all_schemes_csv()
        if csv_file is None:
            return

    analyze_backtest(csv_file)


if __name__ == "__main__":
    main()
