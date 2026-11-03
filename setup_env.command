#!/usr/bin/env bash
# 创建 Model core 统一环境（版本见 requirements.txt）。路径需与 python_env.txt 中的 ~/venvs/model_core 一致。
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${1:-$HOME/venvs/model_core}"
PY="${PY:-python3.11}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
"$PY" -m venv "$VENV" || exit 1
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$HERE/requirements.txt" || exit 1
echo
echo "[OK] 环境已就绪：$VENV"
read -r -p "Press Enter to close..."
