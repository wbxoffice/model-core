#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_KEEP_LATEST = 3


@dataclass
class LogStats:
    log_dir: Path
    txt_total: int = 0
    keep_count: int = 0
    archive_count: int = 0
    deleted_count: int = 0
    bytes_archived: int = 0
    png_total: int = 0
    png_keep_count: int = 0
    png_archive_count: int = 0
    png_moved_count: int = 0
    png_bytes_archived: int = 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_sort_key(path: Path, mtime: float) -> tuple[str, float]:
    """Prefer the date/timestamp embedded in daily artifact names, then mtime."""
    name = path.name
    match = re.search(r"(\d{4}-\d{2}-\d{2})(?:[_-](\d{6}))?", name)
    if match:
        return (match.group(1).replace("-", "") + (match.group(2) or "000000"), mtime)
    match = re.search(r"(\d{8})(?:[_-](\d{6}))?", name)
    if match:
        return (match.group(1) + (match.group(2) or "000000"), mtime)
    return ("", mtime)


def list_log_dirs(root: Path) -> list[Path]:
    dirs: set[Path] = set()
    excluded_parts = {
        "History",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "archive",
        "archive_v2",
        "archive_old_dth_dtl",
    }

    for d in root.rglob("log"):
        if not d.is_dir():
            continue
        rel_parts = set(d.relative_to(root).parts[:-1])
        if rel_parts & excluded_parts:
            continue
        dirs.add(d)

    # This is a log-producing feature folder even though its name is not "log".
    nextday_similarity = root / "log" / "nextday_similarity"
    if nextday_similarity.is_dir():
        dirs.add(nextday_similarity)

    return sorted(dirs)


def ensure_layout(log_dir: Path, execute: bool) -> None:
    for sub in ("current", "archive", "reports"):
        target = log_dir / sub
        if execute:
            target.mkdir(parents=True, exist_ok=True)


def archive_old_txt(log_dir: Path, keep_latest: int, execute: bool) -> LogStats:
    st = LogStats(log_dir=log_dir)
    candidates: list[tuple[Path, float]] = []
    for p in log_dir.glob("*.txt"):
        try:
            mtime = p.stat().st_mtime
            candidates.append((p, mtime))
        except Exception:
            # OneDrive/offline placeholder timeout: skip unreadable file
            continue
    txt_files = [p for p, mtime in sorted(candidates, key=lambda x: artifact_sort_key(x[0], x[1]), reverse=True)]
    st.txt_total = len(txt_files)

    keep = txt_files[:keep_latest]
    old = txt_files[keep_latest:]
    st.keep_count = len(keep)
    st.archive_count = len(old)

    archive_path = log_dir / "archive" / "log_archive.jsonl"
    if execute and old:
        archive_path.parent.mkdir(parents=True, exist_ok=True)

    for p in old:
        try:
            mtime = p.stat().st_mtime
            size = p.stat().st_size
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            # Skip unreadable file to keep script moving.
            continue

        st.bytes_archived += len(content.encode("utf-8", errors="replace"))
        record = {
            "archived_at_utc": utc_now_iso(),
            "source_file": p.name,
            "source_path": str(p),
            "mtime_epoch": mtime,
            "size_bytes": size,
            "content": content,
        }
        if execute:
            with archive_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            p.unlink()
            st.deleted_count += 1

    return st


def archive_old_png(log_dir: Path, keep_latest: int, execute: bool, st: LogStats) -> None:
    candidates: list[tuple[Path, float]] = []
    for p in log_dir.glob("*.png"):
        try:
            candidates.append((p, p.stat().st_mtime))
        except Exception:
            continue

    groups: dict[str, list[tuple[Path, float]]] = {}
    for p, mtime in candidates:
        # Keep latest independently per artifact family, so weighted_path and
        # scenarios_path do not evict each other.
        family = re.sub(r"\d{4}-\d{2}-\d{2}(?:[_-]\d{6})?", "{date}", p.stem)
        groups.setdefault(family, []).append((p, mtime))

    old: list[Path] = []
    keep_count = 0
    for group in groups.values():
        files = [p for p, mtime in sorted(group, key=lambda x: artifact_sort_key(x[0], x[1]), reverse=True)]
        keep_count += min(len(files), keep_latest)
        old.extend(files[keep_latest:])
    st.png_total = len(candidates)
    st.png_keep_count = keep_count
    st.png_archive_count = len(old)

    archive_dir = log_dir / "archive" / "files"
    if execute and old:
        archive_dir.mkdir(parents=True, exist_ok=True)

    for p in old:
        try:
            size = p.stat().st_size
            dst = archive_dir / p.name
            if execute:
                # A regenerated chart for the same dated artifact supersedes
                # the archived copy instead of creating duplicate filenames.
                if dst.exists():
                    dst.unlink()
                shutil.move(str(p), str(dst))
                st.png_moved_count += 1
            st.png_bytes_archived += size
        except Exception:
            continue


def prune_small_files(root: Path, execute: bool) -> tuple[int, int]:
    """Returns (count_removed, bytes_freed_estimate)."""
    removed = 0
    bytes_freed = 0
    targets: list[Path] = []

    # Safe cleanup targets
    targets.extend(root.rglob(".DS_Store"))
    for d in root.rglob("__pycache__"):
        if d.is_dir():
            targets.append(d)

    for p in targets:
        try:
            if p.is_file():
                size = p.stat().st_size
                if execute:
                    p.unlink()
                removed += 1
                bytes_freed += size
            elif p.is_dir():
                size = sum(x.stat().st_size for x in p.rglob("*") if x.is_file())
                if execute:
                    for x in sorted(p.rglob("*"), reverse=True):
                        if x.is_file():
                            x.unlink(missing_ok=True)
                        elif x.is_dir():
                            x.rmdir()
                    p.rmdir()
                removed += 1
                bytes_freed += size
        except Exception:
            # Ignore transient OneDrive/file lock issues
            continue
    return removed, bytes_freed


def main() -> int:
    ap = argparse.ArgumentParser(description="Organize logs: keep latest N txt, archive older to JSONL.")
    ap.add_argument("--root", type=str, default=".", help="Project root path.")
    ap.add_argument("--keep_latest", type=int, default=DEFAULT_KEEP_LATEST, help="Keep latest N *.txt per log dir.")
    ap.add_argument("--execute", action="store_true", help="Apply changes; default is dry-run.")
    ap.add_argument(
        "--prune_small_files",
        action="store_true",
        help="Also remove __pycache__ dirs and .DS_Store files.",
    )
    args = ap.parse_args()

    if args.keep_latest <= 0:
        raise SystemExit("keep_latest must be > 0")

    root = Path(args.root).resolve()
    log_dirs = list_log_dirs(root)

    print(f"[INFO] root={root}")
    print(f"[INFO] execute={args.execute}")
    print(f"[INFO] keep_latest={args.keep_latest}")
    print(f"[INFO] log_dirs={len(log_dirs)}")
    for d in log_dirs:
        print(f"  - {d}")

    all_stats: list[LogStats] = []
    for d in log_dirs:
        ensure_layout(d, execute=args.execute)
        st = archive_old_txt(d, keep_latest=args.keep_latest, execute=args.execute)
        archive_old_png(d, keep_latest=args.keep_latest, execute=args.execute, st=st)
        all_stats.append(st)

    print("\n[SUMMARY]")
    total_txt = sum(x.txt_total for x in all_stats)
    total_keep = sum(x.keep_count for x in all_stats)
    total_archive = sum(x.archive_count for x in all_stats)
    total_deleted = sum(x.deleted_count for x in all_stats)
    total_bytes = sum(x.bytes_archived for x in all_stats)
    total_png = sum(x.png_total for x in all_stats)
    total_png_keep = sum(x.png_keep_count for x in all_stats)
    total_png_archive = sum(x.png_archive_count for x in all_stats)
    total_png_moved = sum(x.png_moved_count for x in all_stats)
    total_png_bytes = sum(x.png_bytes_archived for x in all_stats)
    print(f"- txt_total={total_txt}")
    print(f"- keep_latest_total={total_keep}")
    print(f"- to_archive_total={total_archive}")
    print(f"- deleted_total={total_deleted}")
    print(f"- archived_payload_bytes={total_bytes}")
    print(f"- png_total={total_png}")
    print(f"- png_keep_latest_total={total_png_keep}")
    print(f"- png_to_archive_total={total_png_archive}")
    print(f"- png_moved_total={total_png_moved}")
    print(f"- png_archived_bytes={total_png_bytes}")

    for x in all_stats:
        print(
            f"  * {x.log_dir}: txt={x.txt_total}, keep={x.keep_count}, "
            f"archive={x.archive_count}, deleted={x.deleted_count}; "
            f"png={x.png_total}, png_keep={x.png_keep_count}, "
            f"png_archive={x.png_archive_count}, png_moved={x.png_moved_count}"
        )

    if args.prune_small_files:
        removed, bytes_freed = prune_small_files(root, execute=args.execute)
        print(f"\n[PRUNE] removed_targets={removed}, bytes_freed_estimate={bytes_freed}")

    print("\n[DONE]")
    if not args.execute:
        print("[NOTE] dry-run only. Re-run with --execute to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
