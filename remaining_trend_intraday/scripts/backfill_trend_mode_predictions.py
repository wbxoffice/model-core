#!/usr/bin/env python3
"""Backfill missing post-training trend predictions and evaluate them.

Backfilled rows are kept separate from the immutable live prediction log. A
combined audit file is also written with an explicit ``origin`` column.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import run_trend_modes_daily as daily


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
ROOT = PROJECT_ROOT.parent
RESULT_DIR = PROJECT_ROOT / "results"
EVAL_DIR = PROJECT_ROOT / "evaluation"
LIVE_FILE = RESULT_DIR / "trend_mode_daily_predictions.csv"
BACKFILL_FILE = RESULT_DIR / "trend_mode_backfill_predictions.csv"
COMPLETE_FILE = RESULT_DIR / "trend_mode_predictions_complete.csv"
REPLAY_FILE = RESULT_DIR / "trend_mode_corrected_replay_predictions.csv"
SUMMARY_FILE = EVAL_DIR / "trend_mode_backfill_summary.csv"
REPORT_FILE = EVAL_DIR / "trend_mode_backfill_report_2026-08-27.md"
LABELS = [daily.LABEL_REMAIN, daily.LABEL_CLOSE150]


def _load_nyse_open_dates() -> set[pd.Timestamp]:
    path = ROOT / "data" / "nyse_open_days.csv"
    cal = pd.read_csv(path)
    if cal.empty:
        raise ValueError(f"NYSE calendar is empty: {path}")
    dates = pd.to_datetime(cal.iloc[:, 0], errors="coerce").dropna().dt.normalize()
    return set(dates)


def _model_train_end(bundles: dict[str, dict]) -> pd.Timestamp:
    values = [pd.to_datetime(b.get("train_end_date"), errors="coerce") for b in bundles.values()]
    values = [pd.Timestamp(v).normalize() for v in values if pd.notna(v)]
    if not values:
        raise ValueError("Model bundles do not contain train_end_date.")
    return max(values)


def _candidate_dates(
    frame: pd.DataFrame,
    existing_dates: set[pd.Timestamp],
    open_dates: set[pd.Timestamp],
    train_end: pd.Timestamp,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    work = frame.copy()
    work["target_date"] = pd.to_datetime(work["target_date"], errors="coerce").dt.normalize()
    settled = work[work[LABELS].notna().all(axis=1)].copy()
    available = sorted(set(settled["target_date"].dropna()))
    out = []
    for d in available:
        d = pd.Timestamp(d).normalize()
        if d <= train_end or d in existing_dates or d not in open_dates:
            continue
        if start is not None and d < start:
            continue
        if end is not None and d > end:
            continue
        out.append(d)
    return out


def _evaluate(df: pd.DataFrame) -> dict:
    y_true = df["real_trend"].astype(int).to_numpy()
    y_pred = df["trend_pred"].astype(int).to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    pos_ratio = float(np.mean(y_true))
    return {
        "samples": int(len(df)),
        "date_start": df["date"].min().date().isoformat(),
        "date_end": df["date"].max().date().isoformat(),
        "real_pos_ratio": pos_ratio,
        "pred_pos_ratio": float(np.mean(y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "majority_baseline_accuracy": float(max(pos_ratio, 1.0 - pos_ratio)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _summary_for_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for (label, mode), group in df.groupby(["label", "mode"]):
        row = _evaluate(group)
        row.update({"scope": scope, "label": label, "mode": mode})
        rows.append(row)
    for mode, group in df.groupby("mode"):
        row = _evaluate(group)
        row.update({"scope": scope, "label": "ALL_LABELS", "mode": mode})
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_actuals(pred: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    actual = frame[["target_date", *LABELS]].copy()
    actual["target_date"] = pd.to_datetime(actual["target_date"]).dt.normalize()
    actual = actual.sort_values("target_date").drop_duplicates("target_date", keep="last")
    maps = {label: actual.set_index("target_date")[label] for label in LABELS}

    out = pred.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["real_trend"] = [maps[str(lb)].get(d, np.nan) for d, lb in zip(out["date"], out["label"])]
    out["is_correct"] = np.where(
        out["real_trend"].notna(),
        (pd.to_numeric(out["trend_pred"], errors="coerce") == out["real_trend"]).astype(int),
        np.nan,
    )
    return out


def _write_report(
    summary: pd.DataFrame,
    backfill: pd.DataFrame,
    replay: pd.DataFrame,
    train_end: pd.Timestamp,
    source: Path,
    overlap_max_prob_diff: float,
    overlap_pred_mismatches: int,
) -> None:
    def pct(v) -> str:
        return f"{100.0 * float(v):.2f}%"

    lines = [
        "# Remaining Trend 历史缺失预测补录与准确率",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（Asia/Shanghai）",
        "",
        "## 结论口径",
        "",
        f"- 模型训练截止日：`{train_end.date().isoformat()}`；仅补录其后的正式 NYSE 开市日。",
        f"- 5分钟输入：`{source}`。特征只使用每日 open90（06:30–08:00 PT）及此前可得日线因子；实际标签使用随后时段，仅用于结算。",
        "- 阈值冻结为当前生产配置：remaining balanced/precision = 0.27/0.64；close150 balanced/precision = 0.24/0.50。阈值文件时间为 2026-03-26，早于补录区间。",
        "- 补录结果与实时日志分开保存，`origin=backfill`；实时历史没有被覆盖。",
        "",
        "## 补录范围",
        "",
        f"- 缺失交易日：{backfill['date'].nunique()} 天，{backfill['date'].min().date().isoformat()} 至 {backfill['date'].max().date().isoformat()}。",
        f"- 补录预测行：{len(backfill)} 行（2个标签 × 2种模式 × 每个交易日）。",
        f"- 修复后统一重放：{replay['date'].nunique()} 天，{replay['date'].min().date().isoformat()} 至 {replay['date'].max().date().isoformat()}。",
        f"- 与原3个 live 日期交叉检查：概率最大绝对差 `{overlap_max_prob_diff:.6f}`，分类不一致 `{overlap_pred_mismatches}` 条；原因是旧 live 使用短 exported 输入，因此不把两种输入口径混算准确率。",
        "",
        "## 准确率摘要",
        "",
        "| 范围 | 标签 | 模式 | 样本 | Accuracy | Balanced Accuracy | Precision | Recall | 多数类基线 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['scope']} | {row['label']} | {row['mode']} | {int(row['samples'])} "
            f"| {pct(row['accuracy'])} | {pct(row['balanced_accuracy'])} "
            f"| {pct(row['precision'])} | {pct(row['recall'])} "
            f"| {pct(row['majority_baseline_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## 使用限制",
            "",
            "- 这是冻结模型的历史回放，不是当日实时下单记录；因此应与 live 命中率分开观察。",
            "- Accuracy 容易受非趋势日占比影响；balanced 模式应同时看 balanced accuracy、precision 和 recall。",
            "- 模型判断的是后续时段是否达到 ATR 定义的趋势阈值，不直接等同于可交易方向或扣除成本后的收益。",
            "",
            "## 支持文件",
            "",
            f"- `{BACKFILL_FILE}`",
            f"- `{REPLAY_FILE}`",
            f"- `{COMPLETE_FILE}`",
            f"- `{SUMMARY_FILE}`",
        ]
    )
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="Optional first target date, YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="Optional last target date, YYYY-MM-DD")
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if not LIVE_FILE.exists():
        raise FileNotFoundError(f"Live prediction file missing: {LIVE_FILE}")

    bundles = daily._load_model_bundles()
    thresholds = daily._load_thresholds()
    train_end = _model_train_end(bundles)
    source = daily._resolve_prediction_intraday_source()
    frame = daily._build_prediction_frame(source)
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.normalize()

    live = pd.read_csv(LIVE_FILE)
    live["date"] = pd.to_datetime(live["date"]).dt.normalize()
    existing_dates = set(live["date"].dropna())
    open_dates = _load_nyse_open_dates()
    start = pd.Timestamp(args.start).normalize() if args.start else None
    end = pd.Timestamp(args.end).normalize() if args.end else None
    dates = _candidate_dates(frame, existing_dates, open_dates, train_end, start, end)
    if not dates:
        raise ValueError("No eligible missing post-training dates found.")
    replay_dates = _candidate_dates(frame, set(), open_dates, train_end, start, end)

    replay_rows = []
    for date in replay_dates:
        day = frame[frame["target_date"] == date].copy()
        pred = daily._predict_for_latest_date(day, bundles, thresholds)
        actual_row = day.iloc[-1]
        pred["origin"] = "corrected_replay"
        pred["run_mode"] = "corrected_replay"
        pred["model_train_end"] = train_end.date().isoformat()
        pred["real_trend"] = pred["label"].map({label: int(actual_row[label]) for label in LABELS})
        pred["is_correct"] = (pred["trend_pred"].astype(int) == pred["real_trend"].astype(int)).astype(int)
        replay_rows.append(pred)

    replay = pd.concat(replay_rows, ignore_index=True)
    replay["date"] = pd.to_datetime(replay["date"]).dt.normalize()
    replay = replay.sort_values(["date", "label", "mode"]).reset_index(drop=True)
    replay.to_csv(REPLAY_FILE, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    backfill = replay[replay["date"].isin(dates)].copy()
    backfill["origin"] = "backfill"
    backfill["run_mode"] = "backfill"
    backfill["date"] = pd.to_datetime(backfill["date"]).dt.normalize()
    backfill = backfill.sort_values(["date", "label", "mode"]).reset_index(drop=True)
    backfill.to_csv(BACKFILL_FILE, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    live_complete = live.copy()
    live_complete["origin"] = "live"
    if "run_mode" not in live_complete.columns:
        live_complete["run_mode"] = "legacy"
    live_complete["run_mode"] = live_complete["run_mode"].fillna("legacy").astype(str)
    live_complete["model_train_end"] = np.nan
    base_cols = list(dict.fromkeys([*live_complete.columns, *backfill.columns]))
    for col in base_cols:
        if col not in live_complete:
            live_complete[col] = np.nan
        if col not in backfill:
            backfill[col] = np.nan
    combined = pd.concat([live_complete[base_cols], backfill[base_cols]], ignore_index=True)
    combined = combined.sort_values(["date", "run_mode", "label", "mode", "origin"])
    combined = combined.drop_duplicates(
        ["date", "run_mode", "label", "mode"], keep="last"
    ).reset_index(drop=True)
    combined = _attach_actuals(combined, frame)
    combined.to_csv(COMPLETE_FILE, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    settled_backfill = backfill[backfill["real_trend"].notna()].copy()
    settled_replay = replay[replay["real_trend"].notna()].copy()
    summaries = [
        _summary_for_scope(settled_backfill, "BACKFILL_MISSING"),
        _summary_for_scope(settled_replay, "CORRECTED_REPLAY_POSTTRAIN"),
    ]
    summary = pd.concat(summaries, ignore_index=True)
    summary = summary.sort_values(["scope", "label", "mode"]).reset_index(drop=True)
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    live_posttrain = live[live["date"] > train_end].copy()
    overlap = live_posttrain.merge(
        replay,
        on=["date", "label", "mode"],
        suffixes=("_live", "_replay"),
    )
    overlap_max_prob_diff = (
        float((overlap["p_trend_live"] - overlap["p_trend_replay"]).abs().max())
        if not overlap.empty
        else float("nan")
    )
    overlap_pred_mismatches = (
        int((overlap["trend_pred_live"] != overlap["trend_pred_replay"]).sum())
        if not overlap.empty
        else 0
    )
    _write_report(
        summary,
        backfill,
        replay,
        train_end,
        source,
        overlap_max_prob_diff,
        overlap_pred_mismatches,
    )

    print(f"[SAVE] {BACKFILL_FILE}")
    print(f"[SAVE] {REPLAY_FILE}")
    print(f"[SAVE] {COMPLETE_FILE}")
    print(f"[SAVE] {SUMMARY_FILE}")
    print(f"[SAVE] {REPORT_FILE}")
    print(f"[INFO] model_train_end={train_end.date().isoformat()}")
    print(f"[INFO] backfill_dates={backfill['date'].nunique()} rows={len(backfill)}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
