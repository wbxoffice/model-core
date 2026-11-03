#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
子项目 Python 解释器选择（2026-06-12，B5 去重）。

原先 sync_run_and_aggregate_projects.py 的 pick_runner_for_project() 与
train_all_projects.py 的 pick_runner() 各持有一份等价实现，现收敛到本文件。

选择优先级（与原版一致）：
A) <project>/PYTHON_ENV.txt（最高）
   - 绝对/相对 python 路径（相对路径基于项目目录）
   - conda 环境声明：conda:<env_name>（走 `conda run -n <env> python`）
   - 多行候选按序尝试，# 开头为注释
B) 项目 venv：.venv / venv（平台原生布局优先；非 Windows 跳过 .exe）
C) fallback_python
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _non_comment_lines(p: Path) -> list[str]:
    try:
        raw = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = p.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln)
    return out


def pick_runner(project_dir: Path, fallback_python: str) -> list[str]:
    """返回 subprocess 命令前缀列表（如 [python路径] 或 [conda, run, -n, env, python]）。"""
    # A) 显式声明
    env_file = project_dir / "PYTHON_ENV.txt"
    if env_file.exists():
        for line in _non_comment_lines(env_file):
            if line.lower().startswith("conda:"):
                env_name = line.split(":", 1)[1].strip().strip('"').strip("'")
                if env_name:
                    conda_exe = os.environ.get("CONDA_EXE", "").strip()
                    conda_cmd = conda_exe if conda_exe else "conda"
                    if conda_exe:
                        if Path(conda_exe).exists():
                            return [conda_cmd, "run", "-n", env_name, "python"]
                    elif shutil.which("conda"):
                        return [conda_cmd, "run", "-n", env_name, "python"]
                continue
            p = Path(line)
            if not p.is_absolute():
                p = project_dir / p
            if p.exists():
                return [str(p)]

    # B) 项目 venv（平台原生布局优先；Windows 可执行文件在 mac/Linux 上不可用）
    if os.name == "nt":
        candidates = [
            project_dir / ".venv" / "Scripts" / "python.exe",
            project_dir / "venv" / "Scripts" / "python.exe",
            project_dir / ".venv" / "bin" / "python",
            project_dir / "venv" / "bin" / "python",
        ]
    else:
        candidates = [
            project_dir / ".venv" / "bin" / "python",
            project_dir / "venv" / "bin" / "python",
            project_dir / ".venv" / "Scripts" / "python.exe",
            project_dir / "venv" / "Scripts" / "python.exe",
        ]
    for c in candidates:
        if c.exists():
            if os.name != "nt" and c.suffix.lower() == ".exe":
                continue
            return [str(c)]

    # C) 兜底
    return [fallback_python]
