#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd


def _acc(df: pd.DataFrame, pred_col: str, real_col: str) -> tuple[float | None, int]:
    if pred_col not in df.columns or real_col not in df.columns:
        return None, 0
    t = df[[pred_col, real_col]].dropna()
    if t.empty:
        return None, 0
    return float((t[pred_col] == t[real_col]).mean()), int(len(t))


def main() -> int:
    root = Path(__file__).resolve().parent
    xlsx = root / "prediction_log_all_models.xlsx"
    outdir = root / "model evaluation"
    outdir.mkdir(parents=True, exist_ok=True)

    if not xlsx.exists():
        print(f"[ERR] missing file: {xlsx}")
        return 2

    xl = pd.ExcelFile(xlsx)

    rows_by_regime: list[dict] = []
    rows_overall: list[dict] = []

    for sheet in xl.sheet_names:
        df = pd.read_excel(xlsx, sheet_name=sheet)
        for c in [
            "real_regime",
            "real_direction",
            "real_trend",
            "real_reversal",
            "dir_pred",
            "trend_pred",
            "rev_pred",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        d_acc, d_n = _acc(df, "dir_pred", "real_direction")
        t_acc, t_n = _acc(df, "trend_pred", "real_trend")
        r_acc, r_n = _acc(df, "rev_pred", "real_reversal")
        rows_overall.append(
            {
                "sheet": sheet,
                "days_total": int(len(df)),
                "dir_acc": d_acc,
                "dir_n": d_n,
                "trend_acc": t_acc,
                "trend_n": t_n,
                "rev_acc": r_acc,
                "rev_n": r_n,
            }
        )

        if "real_regime" not in df.columns:
            continue
        reg_df = df.dropna(subset=["real_regime"]).copy()
        for regime in sorted(reg_df["real_regime"].unique()):
            sub = reg_df[reg_df["real_regime"] == regime]
            d_acc, d_n = _acc(sub, "dir_pred", "real_direction")
            t_acc, t_n = _acc(sub, "trend_pred", "real_trend")
            r_acc, r_n = _acc(sub, "rev_pred", "real_reversal")
            rows_by_regime.append(
                {
                    "sheet": sheet,
                    "real_regime": int(regime),
                    "days": int(len(sub)),
                    "dir_acc": d_acc,
                    "dir_n": d_n,
                    "trend_acc": t_acc,
                    "trend_n": t_n,
                    "rev_acc": r_acc,
                    "rev_n": r_n,
                }
            )

    df_overall = pd.DataFrame(rows_overall).sort_values("sheet")
    df_reg = pd.DataFrame(rows_by_regime).sort_values(["sheet", "real_regime"])

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_xlsx = outdir / f"regime_live_perf_{ts}.xlsx"
    out_csv = outdir / f"regime_live_perf_{ts}_by_regime.csv"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_overall.to_excel(w, sheet_name="Overall", index=False)
        df_reg.to_excel(w, sheet_name="ByRealRegime", index=False)

    df_reg.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("[DONE]")
    print(f"- overall+regime xlsx: {out_xlsx}")
    print(f"- by-regime csv: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

