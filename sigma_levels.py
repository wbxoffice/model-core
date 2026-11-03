#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sigma 分位（ATR Levels）：自行计算下一交易日的 中枢 / 突破点 / 目标位。

公式（与 TradingView "ATR Levels" 指标日线模式一致，SPX 已逐日核对误差为 0）：
    中枢 center = 基准日收盘价
    σ   unit   = 基准日 Wilder ATR(14)（RMA，首值用前 14 根 TR 的简单均值）
    分位        = center ± k·σ，k ∈ {0.236(突破点), 0.618, 1, 1.236, 1.618, 2, 2.236, 2.618, 3}
基准日 = 适用交易日之前最近一根【已收盘】日线。默认适用交易日 = 基准日之后的下一个 NYSE 开市日。

未收盘判定（全部按美东时间，与电脑所在时区无关）：最新一根日线日期 = 美东今天，且
SPX 早于 16:00 / US500（CFD，日线在 17:00 收）早于 17:00。此时这根线的"收盘"只是导出时刻的价格，
会被跳过：基准日退回上一根已收盘日线，输出的就是【今天】盘中适用的分位。

输出（log/sigma_levels/）：
    sigma_levels_<适用日>.xlsx            每个标的一个 sheet（字段/数值 表格 + 距中枢 + 历史触及率）
    sigma_ladder_<标的>_<适用日>.png       价格阶梯图
    sigma_levels_history.csv              每日分位留档（按 适用日+标的 覆盖），供回测
并把摘要追加到 strategy_all_projects.txt。

用法：
    python sigma_levels.py --root .
    python sigma_levels.py --root . --date 2026-09-17          # 指定适用交易日（用此前的数据）
    python sigma_levels.py --root . --symbols SPX --no-append
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NY = ZoneInfo("America/New_York")
ATR_LEN = 14
TOUCH_WINDOW = 250

SYMBOLS = {
    "SPX": {"file": "SP_SPX, 1D.csv", "close_hour_ny": 16},
    "US500": {"file": "US500_1D.csv", "close_hour_ny": 17},
}

# 表格（图一样式）：从高到低
TABLE_LEVELS = [
    (3.0, "3σ"), (2.618, "2.618σ"), (2.236, "2.236σ"), (2.0, "2σ"), (1.618, "1.618σ"),
    (1.236, "1.236σ"), (1.0, "1.0σ"), (0.618, "0.618σ"), (0.236, "上行突破"), (0.0, "中枢"),
    (-0.236, "下行突破"), (-0.618, "-0.618σ"), (-1.0, "-1.0σ"), (-1.236, "-1.236σ"),
    (-1.618, "-1.618σ"), (-2.0, "-2σ"), (-2.236, "-2.236σ"), (-2.618, "-2.618σ"), (-3.0, "-3σ"),
]

# 阶梯图（图二样式）：(k, 标签, 备注)
LADDER = [
    (1.618, "目标 4 (极值)", "极限阻力区"),
    (1.236, "目标 3", ""),
    (1.0, "目标 2", ""),
    (0.618, "目标 1", ""),
    (0.236, "核心关键点 A", "上行突破点"),
    (0.0, "日内中枢", "多空强弱分界线"),
    (-0.236, "核心关键点 B", "下行突破点"),
    (-0.618, "目标 1", ""),
    (-1.0, "目标 2", ""),
    (-1.236, "目标 3", ""),
    (-1.618, "目标 4 (极值)", "极限支撑区"),
]


# ----------------------------- data -----------------------------

def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["time"].astype(str).str[:10], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def wilder_atr(df: pd.DataFrame, n: int = ATR_LEN) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    out = np.full(len(tr), np.nan)
    if len(tr) >= n:
        out[n - 1] = tr.iloc[:n].mean()
        for i in range(n, len(tr)):
            out[i] = (out[i - 1] * (n - 1) + tr.iloc[i]) / n
    return pd.Series(out, index=df.index)


def load_open_days(root: Path) -> pd.DatetimeIndex:
    cal = pd.read_csv(root / "data" / "nyse_open_days.csv", encoding="utf-8-sig")
    return pd.DatetimeIndex(pd.to_datetime(cal["date"])).sort_values()


def next_open_day(open_days: pd.DatetimeIndex, after: pd.Timestamp) -> pd.Timestamp:
    later = open_days[open_days > after]
    return later[0] if len(later) else after + pd.offsets.BDay(1)


# ----------------------------- levels -----------------------------

def compute_levels(df: pd.DataFrame, target: pd.Timestamp | None, open_days: pd.DatetimeIndex) -> dict:
    df = df.copy()
    df["atr"] = wilder_atr(df)
    if target is None:
        base = df.iloc[-1]
        target = next_open_day(open_days, base["date"])
    else:
        prior = df[df["date"] < target]
        if prior.empty:
            raise ValueError(f"没有 {target.date()} 之前的日线")
        base = prior.iloc[-1]
    center, unit = float(base["close"]), float(base["atr"])
    return {
        "target_date": target, "base_date": base["date"], "center": center, "atr": unit,
        "levels": {k: center + k * unit for k, _ in TABLE_LEVELS},
    }


def touch_rates(df: pd.DataFrame, window: int | None) -> dict:
    """历史上 当日最高/最低 触及 由前一日收盘与 ATR 推出的各分位 的比例。"""
    d = df.copy()
    d["atr"] = wilder_atr(d)
    d["center"] = d["close"].shift()
    d["unit"] = d["atr"].shift()
    d = d.dropna(subset=["center", "unit"])
    if window:
        d = d.tail(window)
    rates = {}
    for k, _ in TABLE_LEVELS:
        if k > 0:
            hit = d["high"] >= d["center"] + k * d["unit"]
        elif k < 0:
            hit = d["low"] <= d["center"] + k * d["unit"]
        else:
            hit = (d["high"] >= d["center"]) & (d["low"] <= d["center"])
        rates[k] = float(hit.mean()) if len(d) else np.nan
    return {"rates": rates, "n": len(d), "start": d["date"].min()}


def tv_check(df: pd.DataFrame) -> str | None:
    """用数据里 TradingView 导出的 Sigma 列核对最近 60 天自算结果（列存在时）。"""
    need = {"Previous Close", "Upper Trigger", "100%"}
    if not need.issubset(df.columns):
        return None
    d = df.copy()
    d["atr"] = wilder_atr(d)
    d["my_center"] = d["close"].shift()
    d["my_up"] = d["my_center"] + 0.236 * d["atr"].shift()
    d = d.dropna(subset=["Upper Trigger", "my_up"]).tail(60)
    if d.empty:
        return None
    err = (d["my_up"] - d["Upper Trigger"]).abs()
    center_ok = ((d["my_center"] - d["Previous Close"]).abs() < 0.05).mean()
    return f"近{len(d)}日对 TV 上行突破位误差 中位 {err.median():.2f} / 最大 {err.max():.2f} 点，中枢一致率 {center_ok:.0%}"


def drop_unclosed(df: pd.DataFrame, sym_cfg: dict, now_ny: datetime) -> tuple[pd.DataFrame, str]:
    """最新一根日线若是美东今天且未到收盘时刻，跳过它（其"收盘"只是导出时刻价格）。"""
    last = df["date"].iloc[-1]
    if last.date() == now_ny.date() and now_ny.hour < sym_cfg["close_hour_ny"]:
        note = (f"{_fmt_date(last)} 日线未收盘（美东 {now_ny:%H:%M} < {sym_cfg['close_hour_ny']}:00），"
                f"已改用上一根已收盘日线")
        return df.iloc[:-1].copy(), note
    return df, ""


# ----------------------------- outputs -----------------------------

def _fmt_date(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def build_table(sym: str, lv: dict, tr250: dict, trall: dict, note: str) -> pd.DataFrame:
    rows = [
        {"字段": "时间", "数值": pd.Timestamp(lv["target_date"]).strftime("%Y/%m/%d")},
        {"字段": "标的", "数值": sym},
        {"字段": "基于收盘日", "数值": _fmt_date(lv["base_date"])},
        {"字段": "ATR(14)", "数值": round(lv["atr"], 2)},
    ]
    if note:
        rows.append({"字段": "说明", "数值": note})
    for k, name in TABLE_LEVELS:
        rows.append({
            "字段": name,
            "数值": round(lv["levels"][k], 2),
            "距中枢(点)": round(k * lv["atr"], 2),
            f"触及率(近{tr250['n']}日)": round(tr250["rates"][k], 3),
            "触及率(全部)": round(trall["rates"][k], 3),
        })
    return pd.DataFrame(rows)


def _setup_fonts():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "PingFang SC", "Heiti SC", "Hiragino Sans GB", "SimHei",
        "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def draw_ladder(sym: str, lv: dict, tr250: dict, note: str, out_png: Path) -> None:
    _setup_fonts()
    up_c, mid_c, dn_c, ink, sub = "#C0392B", "#D68910", "#1E8449", "#1F2328", "#6B7280"
    n = len(LADDER)
    fig_h = 1.6 + 0.62 * n
    fig, ax = plt.subplots(figsize=(8.6, fig_h), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, n + 1.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    title = f"{sym} {pd.Timestamp(lv['target_date']).strftime('%m/%d')} 价格阶梯（从高到低）"
    ax.text(0.15, n + 1.05, title, fontsize=17, fontweight="bold", color=ink, va="center")
    ax.text(0.15, n + 0.45,
            f"适用交易日 {_fmt_date(lv['target_date'])} ｜ 基于 {_fmt_date(lv['base_date'])} 收盘 "
            f"{lv['center']:.2f} ｜ σ=ATR(14) {lv['atr']:.2f}",
            fontsize=10, color=sub, va="center")
    if note:
        ax.text(0.15, n - 0.2, f"注：{note}", fontsize=9, color=mid_c, va="center")

    for i, (k, label, remark) in enumerate(LADDER):
        y = n - 1 - i
        price = lv["center"] + k * lv["atr"]
        color = up_c if k > 0 else (dn_c if k < 0 else mid_c)
        key = abs(k) in (0.236, 0.0)
        if key:
            ax.add_patch(plt.Rectangle((0.05, y - 0.36), 9.9, 0.72, color=color, alpha=0.09, lw=0))
        ax.scatter([0.4], [y], s=90 if key else 55, color=color, zorder=3,
                   marker="D" if k == 0 else ("^" if k > 0 else "v") if key else "o")
        ax.text(0.8, y, f"[{label}]  {price:.0f}", fontsize=13.5, fontweight="bold", color=ink, va="center")
        if remark:
            ax.text(4.35, y, f"—— {remark}", fontsize=11.5, color=color, va="center")
        sign = "+" if k > 0 else ("−" if k < 0 else "±")
        rate = tr250["rates"][k]
        right = f"{sign}{abs(k):g}σ  {sign}{abs(k) * lv['atr']:.0f}点"
        if k != 0:
            right += f"  触及 {rate:.0%}"
        ax.text(9.9, y, right, fontsize=10, color=sub, va="center", ha="right")

    ax.text(0.15, -0.85,
            f"触及率 = 近{tr250['n']}个交易日中，当日最高/最低触及由前一日收盘与 ATR 推出的该分位的比例",
            fontsize=8.5, color=sub, va="center")
    fig.tight_layout()
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)


def upsert_history(path: Path, rows: list[dict]) -> None:
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        key = set(zip(new["target_date"], new["symbol"]))
        old = old[[(d, s) not in key for d, s in zip(old["target_date"], old["symbol"])]]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["target_date", "symbol"]).to_csv(path, index=False, encoding="utf-8-sig")


def summary_text(results: list[dict]) -> str:
    lines = ["", "=" * 80, "[sigma_levels] Sigma 分位（中枢 = 基准日收盘，σ = Wilder ATR(14)）", "=" * 80]
    for r in results:
        lv = r["lv"]
        lines.append(
            f"{r['symbol']}  适用 {_fmt_date(lv['target_date'])} ｜ 基于 {_fmt_date(lv['base_date'])} 收盘 {lv['center']:.2f} ｜ ATR {lv['atr']:.2f}"
        )
        for k, label, _ in LADDER:
            lines.append(f"  {label:<10s} {k:+.3f}σ  {lv['center'] + k * lv['atr']:>9.2f}"
                         + (f"   触及率 {r['tr250']['rates'][k]:.0%}" if k else ""))
        if r["note"]:
            lines.append(f"  注：{r['note']}")
        if r["check"]:
            lines.append(f"  校验：{r['check']}")
    return "\n".join(lines) + "\n"


# ----------------------------- main -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--date", default=None, help="适用交易日 YYYY-MM-DD（缺省：最新已收盘日线后的下一个开市日）")
    ap.add_argument("--symbols", default="SPX,US500")
    ap.add_argument("--no-append", action="store_true", help="不追加到 strategy_all_projects.txt")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "log" / "sigma_levels"
    out_dir.mkdir(parents=True, exist_ok=True)
    open_days = load_open_days(root)
    target = pd.Timestamp(args.date) if args.date else None
    now_ny = datetime.now(NY)

    results = []
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        cfg = SYMBOLS[sym]
        path = root / "data" / cfg["file"]
        if not path.exists():
            print(f"[WARN] {sym}: 缺少数据 {path}")
            continue
        df = load_daily(path)
        note = ""
        if target is None:
            df, note = drop_unclosed(df, cfg, now_ny)
        lv = compute_levels(df, target, open_days)
        hist = df[df["date"] <= lv["base_date"]]
        tr250 = touch_rates(hist, TOUCH_WINDOW)
        trall = touch_rates(hist, None)
        results.append({"symbol": sym, "lv": lv, "tr250": tr250, "trall": trall,
                        "note": note, "check": tv_check(hist)})
        print(f"[OK] {sym}: 适用 {_fmt_date(lv['target_date'])} 基于 {_fmt_date(lv['base_date'])} "
              f"center={lv['center']:.2f} atr={lv['atr']:.2f}" + (f"  ({note})" if note else ""))

    if not results:
        print("[ERR] 没有可计算的标的")
        return 1

    targets = sorted({_fmt_date(r["lv"]["target_date"]) for r in results})
    if len(targets) > 1:
        print(f"[WARN] 各标的适用日不同 {targets}：数据更新进度不一致（检查 exported data 是否导出齐）")
    tag = targets[-1]
    xlsx = out_dir / f"sigma_levels_{tag}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        for r in results:
            build_table(r["symbol"], r["lv"], r["tr250"], r["trall"], r["note"]).to_excel(
                w, sheet_name=r["symbol"], index=False)
    print(f"[OK] saved: {xlsx}")
    for r in results:
        png = out_dir / f"sigma_ladder_{r['symbol']}_{_fmt_date(r['lv']['target_date'])}.png"
        draw_ladder(r["symbol"], r["lv"], r["tr250"], r["note"], png)
        print(f"[OK] saved: {png}")

    upsert_history(out_dir / "sigma_levels_history.csv", [
        {"target_date": _fmt_date(r["lv"]["target_date"]), "symbol": r["symbol"],
         "base_date": _fmt_date(r["lv"]["base_date"]), "skipped_unclosed_bar": bool(r["note"]),
         "center": round(r["lv"]["center"], 4), "atr14": round(r["lv"]["atr"], 4),
         **{f"L{k:+g}": round(v, 4) for k, v in r["lv"]["levels"].items()},
         "generated_at_ny": now_ny.isoformat(timespec="seconds")}
        for r in results
    ])

    text = summary_text(results)
    print(text)
    if not args.no_append and args.date is None:
        with open(root / "strategy_all_projects.txt", "a", encoding="utf-8") as f:
            f.write(text)
        print("[OK] appended to strategy_all_projects.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
