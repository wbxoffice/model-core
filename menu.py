#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model core 统一入口：列出功能菜单，选择后交给 shared/launch.py 执行对应入口。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MENU = [
    ("每日日跑（收盘后）", "run_prediction_all_models",
     "数据体检 → 更新数据 → aio_v1 / SPX / US500 预测与策略 → 汇总 → 次日相似路径 → Sigma 分位"),
    ("开盘后趋势预测", "trend_daily", "开盘 90 分钟 → 剩余时段 / 尾盘 150 分钟趋势"),
    ("Sigma 分位（单独）", "sigma_levels", "日跑已含此步；这里用于单独重新生成分位表与阶梯图"),
    ("更新训练：日线模型", "train_all_projects", "重训 spx_vdtr_v2 / us500_vdtr_v2（aio_v1 冻结不训）"),
    ("更新训练：开盘后趋势模型", "trend_retrain", "合并 30min 源 → 体检 → 重训"),
    ("回顾：日线模型表现", "models_evaluation", "按窗口统计各模型预测命中（交互）"),
    ("回顾：开盘后趋势表现", "trend_evaluate", "全部历史的趋势预测评估"),
    ("整理日志", "log_organizer", "预览后归档旧日志"),
]


def _print_menu() -> None:
    print("=" * 60)
    print(" Model core 功能菜单（可多选，如 1 2 或 1,2；按顺序执行，某项失败则停止后续）")
    print("=" * 60)
    for i, (name, _, desc) in enumerate(MENU, 1):
        print(f" {i}. {name}\n    {desc}")
    print(" 0. 退出")


def _parse(choice: str) -> list[int] | None:
    parts = choice.replace(",", " ").replace("，", " ").split()
    if not parts or not all(p.isdigit() and 1 <= int(p) <= len(MENU) for p in parts):
        return None
    return [int(p) for p in parts]


def main() -> int:
    last_rc = 0
    while True:
        _print_menu()
        try:
            choice = input("\n选择编号：").strip()
        except EOFError:
            return last_rc
        if choice in ("", "0", "q", "Q"):
            return last_rc
        picks = _parse(choice)
        if picks is None:
            print(f"[WARN] 无效输入：{choice}（请输入 1-{len(MENU)} 的编号）\n")
            continue
        for n in picks:
            name, entry, _ = MENU[n - 1]
            print(f"\n>>> [{n}] {name}")
            last_rc = subprocess.call([sys.executable, str(ROOT / "shared" / "launch.py"), entry])
            if last_rc != 0:
                print(f"[ERR] [{n}] {name} 失败（return_code={last_rc}），后续所选项已跳过。")
                break
        print()


if __name__ == "__main__":
    raise SystemExit(main())
