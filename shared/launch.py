#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
跨平台启动引擎（2026-06-12，B4 启动器去重）。

原先每个入口要手写 .bat/.command/.sh 三份（步骤、日志、计时、失败提示逻辑
在三个平台各复制一份，约 20 组）。现在：
- 步骤/日志/计时/条件跳过/文件检查 → 本文件唯一实现
- 入口声明 → launcher_config.json
- 平台文件 → generate_platform_launchers.py 生成的几行薄垫片（禁止手改）

用法：
    python shared/launch.py <entry_name> [--config <json路径>] [--print]

--print 只打印将执行的命令不实际运行（冒烟用）。

config 结构（entries.<name>）：
- dir            : 入口工作目录（相对仓库根，"." 或子项目名）
- mode           : "console"（输出直达控制台，支持交互脚本）
                   "runlog"（输出写 log/run_<时间戳>.txt，带计时与失败尾部展示）
- python_windows / python_posix : 步骤解释器覆盖；缺省用运行 launch.py 的解释器
- banner         : 运行前打印的行列表
- env_flags      : {"环境变量名": "--追加参数"}，变量值为 "1" 时附加到带
                   append_env_flags 的步骤（保留 mac 版 INSTALL_REQUIREMENTS 习惯）
- steps          : 步骤列表，按序执行，失败即停（见下）
- outputs        : 成功后打印的产物路径列表（相对 dir）
- ok_message     : 成功后打印的提示

step 字段：
- id             : 供 skip_if 引用
- title          : 步骤标题
- script + args  : 运行 dir 下的脚本；args 中 {ROOT}/{PY} 会被替换
- python_code    : 代替 script，跑 python -c 片段
- check_file     : 代替 script，检查 dir 下文件存在，缺失则失败
- allow_exit     : 视为成功的退出码列表（默认 [0]）
- skip_if        : {"step": id, "exit": code, "message": ...}
                   引用步骤以该退出码结束时跳过本步骤
- append_env_flags : true 时附加 env_flags 命中的参数
- error_message  : 失败时打印
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "launcher_config.json"

# 被重定向/管道时保持与子进程输出的先后顺序一致
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def load_config(path: Path) -> dict:
    # utf-8-sig：兼容 Windows 编辑器保存时附加的 BOM
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


PYTHON_ENV_FILE = REPO_ROOT / "python_env.txt"


def _python_from_env_file() -> str | None:
    """python_env.txt：每行一个解释器路径（支持 ~ 和环境变量），取本机第一个存在的。"""
    if not PYTHON_ENV_FILE.exists():
        return None
    for raw in PYTHON_ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(os.path.expandvars(os.path.expanduser(line)))
        if p.exists():
            return str(p)
    return None


def resolve_python(entry: dict) -> str:
    """优先级：环境变量 MODEL_CORE_PYTHON > 入口 python_windows/python_posix > python_env.txt > 当前解释器。"""
    env_py = os.environ.get("MODEL_CORE_PYTHON", "").strip()
    if env_py:
        return env_py
    if os.name == "nt":
        py = entry.get("python_windows")
    else:
        py = entry.get("python_posix")
    if py and py != "auto":
        return py
    return _python_from_env_file() or sys.executable


def expand(value: str, root: Path, py: str) -> str:
    return value.replace("{ROOT}", str(root)).replace("{PY}", py)


def build_step_cmd(step: dict, entry: dict, root: Path, py: str) -> list[str] | None:
    """返回步骤命令行；check_file 步骤返回 None。"""
    if "check_file" in step:
        return None
    if "python_code" in step:
        return [py, "-c", step["python_code"]]
    args = [expand(a, root, py) for a in step.get("args", [])]
    if step.get("append_env_flags"):
        for env_name, flag in entry.get("env_flags", {}).items():
            if os.environ.get(env_name) == "1":
                args.append(flag)
    return [py, str(root / step["script"])] + args


def tail_lines(path: Path, n: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return "(无法读取日志)"


def fmt_duration(seconds: float) -> str:
    """Readable elapsed time without hiding sub-second precheck steps as 0s."""
    if seconds < 10:
        return f"{seconds:.1f}".rstrip("0").rstrip(".")
    return str(int(round(seconds)))


def run_entry(name: str, config: dict, print_only: bool = False) -> int:
    entries = config.get("entries", {})
    if name not in entries:
        print(f"[ERR] launcher_config.json 中没有入口: {name}")
        print(f"可用入口: {', '.join(sorted(entries))}")
        return 2

    entry = entries[name]
    root = (REPO_ROOT / entry.get("dir", ".")).resolve()
    py = resolve_python(entry)
    mode = entry.get("mode", "console")

    print(f"[INFO] ROOT={root}")
    print(f"[INFO] PY={py}")
    for line in entry.get("banner", []):
        print(line)
    print()

    runlog: Path | None = None
    log_handle = None
    if mode == "runlog" and not print_only:
        log_dir = root / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        runlog = log_dir / f"run_{ts}.txt"
        log_handle = runlog.open("w", encoding="utf-8")
        log_handle.write(f"[INFO] ROOT={root}\n[INFO] PY={py}\n\n")
        log_handle.flush()

    exit_codes: dict[str, int] = {}
    durations: list[tuple[int, str, str, float]] = []
    t0 = time.perf_counter()

    try:
        for i, step in enumerate(entry.get("steps", []), start=1):
            step_id = step.get("id", f"step{i}")
            title = step.get("title", step.get("script", step_id))

            skip_if = step.get("skip_if")
            if skip_if and exit_codes.get(skip_if["step"]) == skip_if["exit"]:
                msg = skip_if.get("message", f"[INFO] skip {step_id}")
                print(msg)
                if log_handle:
                    log_handle.write(msg + "\n")
                continue

            if "check_file" in step:
                target = root / step["check_file"]
                if print_only:
                    print(f"[STEP {i}] (检查文件) {target}")
                    continue
                if not target.exists():
                    print()
                    print(step.get("error_message", f"[ERROR] 文件缺失: {target}"))
                    return 1
                exit_codes[step_id] = 0
                continue

            cmd = build_step_cmd(step, entry, root, py)
            if print_only:
                print(f"[STEP {i}] {title}")
                print(f"         cwd={root}")
                print(f"         cmd={subprocess.list2cmdline(cmd)}")
                continue

            print(f"[STEP {i}] {title}")
            if log_handle:
                log_handle.write(f"\n[STEP {i}] {title}\n")
                log_handle.flush()

            s = time.perf_counter()
            if log_handle:
                rc = subprocess.run(cmd, cwd=root, stdout=log_handle, stderr=subprocess.STDOUT).returncode
            else:
                rc = subprocess.run(cmd, cwd=root).returncode
            d = time.perf_counter() - s
            exit_codes[step_id] = rc
            durations.append((i, step_id, title, d))

            if log_handle:
                log_handle.write(f"[STEP {i}] elapsed_sec={fmt_duration(d)}\n[STEP {i}] return_code={rc}\n")
                log_handle.flush()

            allow = step.get("allow_exit", [0])
            if rc not in allow:
                print()
                print(step.get("error_message", f"[FAIL] STEP {i} failed. return_code={rc}"))
                if runlog is not None:
                    print(f"Open log: {runlog}")
                    print('Tip: search "SyntaxError" / "Traceback" / "[ERR]"')
                    print("---- last 80 lines ----")
                    log_handle.flush()
                    print(tail_lines(runlog, 80))
                return rc if rc != 0 else 1
    finally:
        if log_handle:
            total = time.perf_counter() - t0
            timing = ", ".join(
                f"step{idx}_{sid}_sec={fmt_duration(d)}" for idx, sid, _title, d in durations
            )
            log_handle.write(f"[TIMING] {timing}, total_sec={fmt_duration(total)}\n")
            log_handle.close()

    if print_only:
        return 0

    print()
    print(entry.get("ok_message", "[OK] All steps finished successfully."))
    if runlog is not None:
        print(f"Log: {runlog}")
    outputs = entry.get("outputs", [])
    if outputs:
        print("Outputs:")
        for o in outputs:
            print(f"  {root / o}")
    if durations:
        print("Timing:")
        for idx, sid, title, d in durations:
            print(f"  STEP {idx} ({sid}) {title}: {fmt_duration(d)} sec")
        print(f"  TOTAL : {fmt_duration(time.perf_counter() - t0)} sec")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="跨平台启动引擎（读取 launcher_config.json）")
    ap.add_argument("entry", help="入口名（launcher_config.json entries 的键）")
    ap.add_argument("--config", type=str, default=None, help="自定义 config 路径（测试用）")
    ap.add_argument("--print", dest="print_only", action="store_true", help="只打印命令不执行")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else DEFAULT_CONFIG
    config = load_config(cfg_path)
    return run_entry(args.entry, config, print_only=args.print_only)


if __name__ == "__main__":
    sys.exit(main())
