#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 backtest 2.0 Excel 文件中按 real_regime 统计 next_range 的分位数，
输出到 range_quantiles_by_regime.json

使用场景：
- 目录下有类似：
    backtest 2.0_2018-01-01_to_2025-12-05_P_0.450.550.55.xlsx
  这种聚合回测结果文件（2.0 版本）

脚本逻辑：
1. 寻找当前目录下所有 "backtest 2.0*.xlsx"
2. 每个文件：
   - 遍历所有 sheet
   - 只要该 sheet 同时包含 real_regime 和 next_range 两列，就读取并收集
3. 合并所有数据后：
   - 按 real_regime 分组
   - 统计 next_range 的 q25/q50/q75
4. 输出 JSON:
   range_quantiles_by_regime.json
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


def collect_from_one_excel(path: Path):
    """
    从单个 backtest 2.0 xlsx 中抽取 (real_regime, next_range) 数据。
    返回 DataFrame 或 None。
    """
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"[WARN] 无法打开 {path.name}: {e}")
        return None

    collected = []
    print(f"  -> 检查文件 {path.name} 的 sheets: {xls.sheet_names}")

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
        except Exception as e:
            print(f"     [WARN] 读取 sheet '{sheet}' 失败: {e}")
            continue

        cols = set(df.columns.astype(str))
        if ("real_regime" in cols) and ("next_range" in cols):
            sub = df[["real_regime", "next_range"]].copy()
            sub["source_file"] = path.name
            sub["source_sheet"] = sheet
            collected.append(sub)
            print(f"     [INFO] 在 sheet '{sheet}' 找到 real_regime & next_range.")

    if not collected:
        print(f"  -> 文件 {path.name} 未找到合适的 sheet。")
        return None

    out = pd.concat(collected, ignore_index=True)
    print(f"  -> 文件 {path.name} 收集到 {len(out)} 行数据。")
    return out


def main():
    base_dir = Path(__file__).resolve().parent

    # 搜索 backtest 2.0*.xlsx 文件
    backtest_files = sorted(base_dir.glob("backtest 2.0*.xlsx"))
    if not backtest_files:
        print("[ERROR] 未找到 'backtest 2.0*.xlsx' 文件，请检查目录。")
        return

    print("发现以下 backtest 2.0 文件：")
    for f in backtest_files:
        print("  -", f.name)

    dfs = []
    for f in backtest_files:
        df_sub = collect_from_one_excel(f)
        if df_sub is not None and not df_sub.empty:
            dfs.append(df_sub)

    if not dfs:
        print("[ERROR] 所有文件都无法使用（没有 real_regime/next_range），退出。")
        return

    all_df = pd.concat(dfs, ignore_index=True)

    # 清洗数据
    all_df["next_range"] = pd.to_numeric(all_df["next_range"], errors="coerce")
    all_df["real_regime"] = pd.to_numeric(all_df["real_regime"], errors="coerce")

    all_df = all_df.dropna(subset=["real_regime", "next_range"])
    all_df = all_df[all_df["next_range"] > 0]

    all_df["real_regime"] = all_df["real_regime"].astype(int)

    print(f"\n合并后有效样本数：{len(all_df)}")

    result = {}
    for regime, sub in all_df.groupby("real_regime"):
        arr = sub["next_range"].values
        if len(arr) < 10:
            print(f"[WARN] real_regime={regime} 样本数过少 ({len(arr)}), 结果可能不稳定。")

        q25 = float(np.quantile(arr, 0.25))
        q50 = float(np.quantile(arr, 0.50))
        q75 = float(np.quantile(arr, 0.75))

        result[int(regime)] = {
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "count": int(len(arr)),
        }

        print(
            f"real_regime={regime}: "
            f"q25={q25:.4f}, q50={q50:.4f}, q75={q75:.4f}, n={len(arr)}"
        )

    out_path = base_dir / "range_quantiles_by_regime.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n已写出 quantile 配置到：", out_path)


if __name__ == "__main__":
    main()
