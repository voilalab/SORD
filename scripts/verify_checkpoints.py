"""Verify local checkpoints against checkpoint_manifest.json (existence, size, SHA256).

    python scripts/verify_checkpoints.py --checkpoint-dir ./checkpoints
    python scripts/verify_checkpoints.py --only celeba_score edm2_clwd64
    python scripts/verify_checkpoints.py --include-optional        # also the baselines' checkpoints

Exits with status 1 if any selected checkpoint is missing or does not match.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from checkpoint_common import check_entry, load_manifest, paths, select


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, default=paths.CHECKPOINT_ROOT)
    ap.add_argument("--only", nargs="+", help="checkpoint ids from checkpoint_manifest.json")
    ap.add_argument("--include-optional", action="store_true", help="also check optional (baseline) checkpoints")
    a = ap.parse_args()

    bad = 0
    for key, entry in select(load_manifest(), a.only, a.include_optional).items():
        results = check_entry(entry, a.checkpoint_dir)
        status = "ok" if all(s == "ok" for _, s in results) else ", ".join(sorted({s for _, s in results if s != "ok"}))
        bad += status != "ok"
        print(f"{key:30s} {status:18s} {entry['path']}")
        for p, s in results:
            if s != "ok":
                print(f"    {s}: {p}")
    print(f"\n{'all selected checkpoints verified' if not bad else f'{bad} checkpoint(s) missing or invalid'}"
          f" (checkpoint dir: {a.checkpoint_dir})")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
