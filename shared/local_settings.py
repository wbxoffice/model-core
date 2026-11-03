"""本机私有设置：读取仓库根目录 local_settings.json（已 gitignore）。

例如 {"account_equity": 100000}。缺文件或缺字段时返回调用方给的占位默认值。
"""

from __future__ import annotations

import json
from pathlib import Path

SETTINGS_FILE = Path(__file__).resolve().parent.parent / "local_settings.json"


def get_setting(key: str, default):
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8-sig"))[key]
    except Exception:
        return default


def account_equity(default: float = 10000.0) -> float:
    return float(get_setting("account_equity", default))
