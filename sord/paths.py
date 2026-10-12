"""Where SORD looks for checkpoints, data and outputs.

Nothing machine-specific is hard-coded: every location is a directory under one of five roots, each
overridable by an environment variable (defaults in parentheses, relative to the repository):

    PROJECT_ROOT     the repository                      (this checkout)
    DATA_ROOT        benchmark files                     (PROJECT_ROOT: reproduction/fetch_assets.py
                                                          extracts them under experiments/... there)
    CHECKPOINT_ROOT  model weights                       (PROJECT_ROOT/checkpoints)
    OUTPUT_ROOT      run outputs                         (PROJECT_ROOT/runs)
    CACHE_ROOT       caches of third-party libraries     (PROJECT_ROOT/.cache)

Checkpoints are referred to by an identifier from checkpoint_manifest.json, which records each file's
relative path under CHECKPOINT_ROOT, size and SHA256; the download / verify / package scripts read the
same manifest.  Configs may also give an explicit path (absolute, or relative to the corresponding root).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser()


def _root(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


DATA_ROOT = _root("DATA_ROOT", PROJECT_ROOT)
CHECKPOINT_ROOT = _root("CHECKPOINT_ROOT", PROJECT_ROOT / "checkpoints")
OUTPUT_ROOT = _root("OUTPUT_ROOT", PROJECT_ROOT / "runs")
CACHE_ROOT = _root("CACHE_ROOT", PROJECT_ROOT / ".cache")
MANIFEST = PROJECT_ROOT / "checkpoint_manifest.json"


@lru_cache(maxsize=1)
def manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


def checkpoint(name: str, root: Path | None = None) -> Path:
    """Local path of a checkpoint given by its manifest identifier (or by a path)."""
    entries = manifest()["checkpoints"]
    if name in entries:
        path = Path(root or CHECKPOINT_ROOT) / entries[name]["path"]
        if not path.exists():
            raise FileNotFoundError(
                f"checkpoint '{name}' not found at {path}. Download it with "
                f"`python scripts/download_checkpoints.py --only {name}` or set CHECKPOINT_ROOT "
                f"(see CHECKPOINTS.md).")
        return path
    path = Path(name).expanduser()
    return path if path.is_absolute() else Path(root or CHECKPOINT_ROOT) / path


def data(path: str) -> Path:
    """A benchmark file, absolute or relative to DATA_ROOT."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else DATA_ROOT / p
