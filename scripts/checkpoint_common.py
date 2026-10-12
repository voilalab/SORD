"""Shared helpers for the checkpoint scripts: read checkpoint_manifest.json, hash and check files.

The manifest is the single source of truth, also used at runtime by sord/paths.py.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sord import paths  # noqa: E402


def load_manifest(path: Path | None = None, full: bool = False) -> dict:
    with open(path or paths.MANIFEST) as f:
        m = json.load(f)
    return m if full else m["checkpoints"]


def select(entries: dict, only=None, include_optional=False) -> dict:
    if only:
        unknown = [k for k in only if k not in entries]
        if unknown:
            raise SystemExit(f"unknown checkpoint id(s): {', '.join(unknown)}")
        return {k: entries[k] for k in only}
    return {k: v for k, v in entries.items() if include_optional or v["required"] != "optional"}


def files_of(entry: dict, root: Path):
    """(local path, expected size, expected sha256) for every file of an entry."""
    base = Path(root) / entry["path"]
    for f in entry["files"]:
        yield (base / f["path"] if f["path"] else base), f["size"], f["sha256"]


def sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check_file(path: Path, size: int, digest: str) -> str:
    """'ok', 'missing', 'size mismatch' or 'sha256 mismatch'."""
    if not path.exists():
        return "missing"
    if path.stat().st_size != size:
        return "size mismatch"
    return "ok" if sha256(path) == digest else "sha256 mismatch"


def check_entry(entry: dict, root: Path) -> list:
    return [(p, check_file(p, s, d)) for p, s, d in files_of(entry, root)]
