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
    ("Sigma 分位", "sigma_levels", "单独生成下一交易日分位表与阶梯图"),
    ("更新训练：日线模型", "train_all_projects", "重训 spx_vdtr_v2 / us500_vdtr_v2（aio_v1 冻结不训）"),
    ("更新训练：开盘后趋势模型", "trend_retrain", "合并 30min 源 → 体检 → 重训"),
    ("回顾：日线模型表现", "models_evaluation", "按窗口统计各模型预测命中（交互）"),
    ("回顾：开盘后趋势表现", "trend_evaluate", "全部历史的趋势预测评估"),
    ("整理日志", "log_organizer", "预览后归档旧日志"),
]


def main() -> int:
    print("=" * 60)
    print(" Model core 功能菜单")
    print("=" * 60)
    for i, (name, _, desc) in enumerate(MENU, 1):
        print(f" {i}. {name}\n    {desc}")
    print(" 0. 退出")
    try:
        choice = input("\n选择编号：").strip()
    except EOFError:
        return 0
    if not choice.isdigit() or not (1 <= int(choice) <= len(MENU)):
        return 0
    entry = MENU[int(choice) - 1][1]
    return subprocess.call([sys.executable, str(ROOT / "shared" / "launch.py"), entry])


if __name__ == "__main__":
    raise SystemExit(main())
