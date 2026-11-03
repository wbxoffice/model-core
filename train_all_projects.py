#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-click trainer for multi-project models.

Default behavior:
- Auto discover projects under root.
- Auto detect train script in each project.
- Skip "aio_v1" by default.
- Continue training other projects if one project fails.

Usage:
  python3 train_all_projects.py
  python3 train_all_projects.py --dry_run
  python3 train_all_projects.py --python python3
  python3 train_all_projects.py --include "spx_vdtr_v2"
  python3 train_all_projects.py --exclude "dip_buy"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from experiment_ledger import append_ledger_row
from project_registry_utils import load_registry, registry_map_by_name, validate_registry
from shared.runner import pick_runner as shared_pick_runner

SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", ".idea", "data", "latest data"}

DEFAULT_EXCLUDED_PROJECTS = {
    "aio_v1",
}

TRAIN_SCRIPT_CANDIDATES = [
    "run_train.py",
    "train_all_models_latest.py",
    "train_all_models_v3.py",
    "train_all_models_v2.py",
    "train_all_models_v1.py",
    "train_all_models_V1.py",
    "train_all_models.py",
    "train.py",
]

REQUIREMENTS_CANDIDATES = [
    "requirements-train.txt",
    "requirements.txt",
]


def find_projects(root: Path) -> list[Path]:
    projects: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.lower() in SKIP_DIRS:
            continue
        if (p / "config.py").exists() or (p / "data").exists() or (p / "log").exists():
            projects.append(p)
    return projects


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
    """实现已收敛到 shared/runner.py（与 sync 编排器共用同一份）。"""
    return shared_pick_runner(project_dir, fallback_python)


def find_train_script(project_dir: Path) -> Path | None:
    for name in TRAIN_SCRIPT_CANDIDATES:
        p = project_dir / name
        if p.exists():
            return p

    patterns = [
        "run_train*.py",
        "train_all*.py",
        "retrain*.py",
        "train*.py",
    ]
    seen: set[Path] = set()
    for pat in patterns:
        for p in sorted(project_dir.glob(pat)):
            if p in seen:
                continue
            seen.add(p)
            return p
    return None


def find_train_script_with_registry(project_dir: Path, row: dict | None) -> Path | None:
    if row:
        ts = row.get("train_script")
        if isinstance(ts, str) and ts.strip():
            p = project_dir / ts
            if p.exists():
                return p
    return find_train_script(project_dir)


def find_requirements_file(project_dir: Path, row: dict | None) -> Path | None:
    if row:
        rf = row.get("requirements_file")
        if isinstance(rf, str) and rf.strip():
            p = project_dir / rf
            if p.exists():
                return p
    for name in REQUIREMENTS_CANDIDATES:
        p = project_dir / name
        if p.exists():
            return p
    return None


def get_env_versions(runner: list[str], cwd: Path) -> dict[str, str]:
    code = (
        "import json,sys\n"
        "out={'python_version':sys.version.split()[0]}\n"
        "for m,k in [('numpy','numpy_version'),('pandas','pandas_version'),('sklearn','sklearn_version')]:\n"
        "  try:\n"
        "    mod=__import__(m)\n"
        "    out[k]=getattr(mod,'__version__','')\n"
        "  except Exception:\n"
        "    out[k]=''\n"
        "print(json.dumps(out))\n"
    )
    cmd = [*runner, "-c", code]
    try:
        cp = subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)
        if cp.returncode != 0:
            return {}
        import json
        return json.loads((cp.stdout or "").strip() or "{}")
    except Exception:
        return {}


def install_requirements(
    runner: list[str],
    fallback_python: str,
    requirements_file: Path,
    cwd: Path,
    dry_run: bool = False,
) -> int:
    cmd = [*runner, "-m", "pip", "install", "-r", str(requirements_file)]
    print(f"[ENV] ({cwd.name}) {' '.join(cmd)}")
    if dry_run:
        return 0
    try:
        cp = subprocess.run(cmd, cwd=str(cwd), check=False)
        if cp.returncode == 0:
            return 0
    except Exception:
        pass

    fallback_cmd = [fallback_python, "-m", "pip", "install", "-r", str(requirements_file)]
    print(f"[ENV-FALLBACK] ({cwd.name}) {' '.join(fallback_cmd)}")
    if dry_run:
        return 0
    try:
        cp = subprocess.run(fallback_cmd, cwd=str(cwd), check=False)
        return int(cp.returncode)
    except Exception as e:
        print(f"[ERR] install requirements failed: {e}")
        return 2


def run_with_fallback(
    runner: list[str],
    fallback_python: str,
    script: Path,
    cwd: Path,
    dry_run: bool = False,
) -> int:
    cmd = [*runner, str(script)]
    print(f"[RUN] ({cwd.name}) {' '.join(cmd)}")
    if dry_run:
        return 0

    try:
        cp = subprocess.run(cmd, cwd=str(cwd), check=False)
        return int(cp.returncode)
    except (OSError, FileNotFoundError) as e:
        msg = str(e)
        if "Exec format error" in msg or "No such file or directory" in msg:
            fallback_cmd = [fallback_python, str(script)]
            print(f"[FALLBACK] ({cwd.name}) {' '.join(fallback_cmd)}")
            try:
                cp = subprocess.run(fallback_cmd, cwd=str(cwd), check=False)
                return int(cp.returncode)
            except Exception as e2:
                print(f"[ERR] fallback failed: {e2}")
                return 3
        print(f"[ERR] run failed: {e}")
        return 2
    except Exception as e:
        print(f"[ERR] run failed: {e}")
        return 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="Root folder; default current directory.")
    ap.add_argument("--python", type=str, default=None, help="Fallback python executable.")
    ap.add_argument("--dry_run", action="store_true", help="Print actions only.")
    ap.add_argument("--include", action="append", default=[], help="Only train these project names (repeatable).")
    ap.add_argument("--exclude", action="append", default=[], help="Additional project names to skip (repeatable).")
    ap.add_argument(
        "--install_requirements",
        action="store_true",
        help="Install project requirements before running train script.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    fallback_python = args.python or sys.executable

    print(f"[INFO] root={root}")
    print(f"[INFO] fallback_python={fallback_python}")
    print(f"[INFO] dry_run={args.dry_run}")

    registry = load_registry(root)
    reg_map = registry_map_by_name(registry)
    print(f"[INFO] registry_loaded={'yes' if reg_map else 'no'}")
    validate_registry(root, reg_map)

    projects = find_projects(root)
    include_set = set(args.include)
    exclude_set = set(DEFAULT_EXCLUDED_PROJECTS) | set(args.exclude)

    selected: list[Path] = []
    for p in projects:
        row = reg_map.get(p.name)
        if include_set and p.name not in include_set:
            continue
        if p.name in exclude_set:
            print(f"[SKIP] excluded: {p.name}")
            continue
        if row and row.get("include_in_training") is False:
            print(f"[SKIP] registry include_in_training=false: {p.name}")
            continue
        selected.append(p)

    print(f"[INFO] selected_projects={len(selected)}")
    for p in selected:
        print(f"  - {p.name}")

    failed: list[str] = []
    skipped_no_train: list[str] = []
    ts_run = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ledger_path: Path | None = None

    for proj in selected:
        print("\n" + "#" * 80)
        print(f"[PROJECT] {proj.name}")
        t0 = time.time()
        notes = ""
        req_file: Path | None = None

        row = reg_map.get(proj.name)
        train_script = find_train_script_with_registry(proj, row)
        if train_script is None:
            print("[SKIP] train script not found.")
            skipped_no_train.append(proj.name)
            try:
                ledger_path = append_ledger_row(
                    root,
                    {
                        "run_ts": ts_run,
                        "project": proj.name,
                        "status": "SKIPPED_NO_TRAIN_SCRIPT",
                        "return_code": "",
                        "duration_sec": round(time.time() - t0, 3),
                        "runner": "",
                        "train_script": "",
                        "requirements_file": "",
                        "python_version": "",
                        "numpy_version": "",
                        "pandas_version": "",
                        "sklearn_version": "",
                        "notes": "",
                    },
                )
            except Exception:
                pass
            continue

        runner = pick_runner(proj, fallback_python)
        print(f"[INFO] runner={' '.join(runner)}")
        print(f"[INFO] train_script={train_script.name}")
        req_file = find_requirements_file(proj, row)
        if req_file:
            print(f"[INFO] requirements_file={req_file.name}")
        elif args.install_requirements:
            print("[WARN] --install_requirements is set but no requirements file found.")

        req_rc = 0
        if args.install_requirements and req_file is not None:
            req_rc = install_requirements(runner, fallback_python, req_file, proj, dry_run=args.dry_run)
            if req_rc != 0:
                print(f"[WARN] requirements install failed rc={req_rc}")
                notes = "requirements_install_failed"

        if req_rc != 0:
            rc = req_rc
        else:
            rc = run_with_fallback(runner, fallback_python, train_script, proj, dry_run=args.dry_run)
        print(f"[INFO] return_code={rc}")
        if rc != 0:
            failed.append(f"{proj.name} ({train_script.name}) rc={rc}")

        versions = {}
        if not args.dry_run:
            versions = get_env_versions(runner, proj)
        try:
            if args.dry_run:
                status = "DRY_RUN"
            else:
                status = "SUCCESS" if rc == 0 else "FAILED"
            ledger_path = append_ledger_row(
                root,
                {
                    "run_ts": ts_run,
                    "project": proj.name,
                    "status": status,
                    "return_code": rc,
                    "duration_sec": round(time.time() - t0, 3),
                    "runner": " ".join(runner),
                    "train_script": train_script.name,
                    "requirements_file": req_file.name if req_file else "",
                    "python_version": versions.get("python_version", ""),
                    "numpy_version": versions.get("numpy_version", ""),
                    "pandas_version": versions.get("pandas_version", ""),
                    "sklearn_version": versions.get("sklearn_version", ""),
                    "notes": notes,
                },
            )
        except Exception as e:
            print(f"[WARN] failed to append experiment ledger: {e}")

    print("\n" + "=" * 80)
    print("[DONE] train_all_projects")
    if ledger_path is not None:
        print(f"[INFO] experiment_ledger={ledger_path}")
    if skipped_no_train:
        print("[INFO] no-train-script projects:")
        for x in skipped_no_train:
            print(f"  - {x}")

    if failed:
        print("[WARN] failed projects:")
        for x in failed:
            print(f"  - {x}")
        return 1

    print("[OK] all selected train scripts finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
