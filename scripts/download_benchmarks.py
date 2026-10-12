"""Download the image-benchmark inputs read by configs/sord/*.yaml and verify their SHA256.

    python scripts/download_benchmarks.py                       # all image benchmarks (~74 MB)
    python scripts/download_benchmarks.py --only celeba_cifar   # files for one config

The inputs are one archive, sord_benchmarks.zip, in the same Google Drive folder as the authors' weights
(checkpoint_manifest.json -> "gdrive" -> "folder_url", or --gdrive-folder URL). File list and checksums:
reproduction/benchmarks.json. Files are written under DATA_ROOT (default: repository root), at the paths the
configs expect. Already verified files are skipped. Audio inputs are built from Slakh2100 (see README).
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from checkpoint_common import load_manifest, paths, sha256
from download_checkpoints import archive_path

MANIFEST = Path(__file__).resolve().parents[1] / "reproduction" / "benchmarks.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="config names, e.g. celeba_cifar clwd")
    ap.add_argument("--data-root", type=Path, default=paths.DATA_ROOT)
    ap.add_argument("--gdrive-folder", help="share link of the release folder (overrides the manifest's gdrive.folder_url)")
    ap.add_argument("--cache-dir", type=Path, default=paths.CACHE_ROOT / "downloads")
    a = ap.parse_args()

    bench = json.loads(MANIFEST.read_text())
    files = [f for f in bench["files"] if not a.only or f["config"].split(" ")[0] in a.only]
    if not files:
        raise SystemExit(f"no benchmark files for {a.only}; configs: {sorted({f['config'].split(' ')[0] for f in bench['files']})}")
    todo = [f for f in files if not ((a.data_root / f["path"]).is_file() and sha256(a.data_root / f["path"]) == f["sha256"])]
    for f in files:
        if f not in todo:
            print(f"[skip] {f['path']}: already verified")
    if todo:
        cfg = dict(load_manifest(full=True).get("gdrive", {}))
        if a.gdrive_folder:
            cfg["folder_url"] = a.gdrive_folder
        with zipfile.ZipFile(archive_path(bench["archive"]["name"], a.cache_dir, cfg)) as z:
            for f in todo:
                out = a.data_root / f["path"]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(f["path"]))
    bad = [f["path"] for f in todo if sha256(a.data_root / f["path"]) != f["sha256"]]
    for f in todo:
        print(f"[{'BAD' if f['path'] in bad else ' ok'}] {f['path']}")
    if bad:
        raise SystemExit(f"{len(bad)} file(s) failed the SHA256 check")
    print(f"\nall selected benchmark inputs verified (data root: {a.data_root})")


if __name__ == "__main__":
    main()
