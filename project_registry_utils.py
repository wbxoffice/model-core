#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path


REGISTRY_FILE = "project_registry.json"


def load_registry(root: Path) -> dict:
    p = root / REGISTRY_FILE
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def validate_registry(root: Path, reg_map: dict[str, dict]) -> list[str]:
    """轻量校验：registry 中引用的项目目录/脚本是否存在。

    只打印 [WARN] 并返回问题列表，不中断流程（保持原有容错行为）。
    """
    issues: list[str] = []
    for name, row in reg_map.items():
        proj = root / name
        if not proj.is_dir():
            issues.append(f"registry 项目目录不存在: {name}")
            continue
        for field in ("run_script", "strategy_script", "train_script"):
            script = row.get(field)
            if isinstance(script, str) and script.strip():
                if not (proj / script).exists():
                    issues.append(f"{name}: {field}={script} 文件不存在")
    for msg in issues:
        print(f"[WARN] {msg}")
    return issues


def registry_map_by_name(registry: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    projects = registry.get("projects", [])
    if not isinstance(projects, list):
        return out
    for row in projects:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if isinstance(name, str) and name.strip():
            out[name] = row
    return out

