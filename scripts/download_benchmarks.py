"""Download the benchmark inputs read by configs/sord/*.yaml and verify their SHA256.

    python scripts/download_benchmarks.py                       # all image benchmarks (~74 MB)
    python scripts/download_benchmarks.py --only celeba_cifar   # files for one config
    python scripts/download_benchmarks.py --audio               # also the audio benchmark (~1.3 GB)
    python scripts/download_benchmarks.py --only audio_piano_speech   # audio, speech interferers (one archive for all audio)

The inputs are two archives: sord_benchmarks.zip (image tasks) and sord_audio_benchmarks.zip (audio tasks; clips
derived from Slakh2100 and LibriSpeech, CC BY 4.0). An archive is fetched from its "url" in
reproduction/benchmarks.json (a Google Drive file link or a direct HTTP(S) link; --archive-url NAME=URL overrides),
otherwise from the Google Drive folder of the authors' weights (checkpoint_manifest.json -> "gdrive" ->
"folder_url", or --gdrive-folder URL). File list and checksums: reproduction/benchmarks.json. Files are written
under DATA_ROOT (default: repository root), at the paths the configs expect. Already verified files are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

from checkpoint_common import load_manifest, paths, sha256
from download_checkpoints import archive_path, gdrive_download, http_download

MANIFEST = Path(__file__).resolve().parents[1] / "reproduction" / "benchmarks.json"


def fetch_archive(ar: dict, url: str | None, cache: Path, cfg: dict) -> Path:
    """Local copy of a benchmark archive (cache, its own link, or the Drive folder), checked against its SHA256."""
    hits = sorted(cache.rglob(ar["name"])) if cache.exists() else []
    if hits:
        path = hits[0]
    elif url:
        path = cache / ar["name"]
        m = re.search(r"(?:/d/|[?&]id=)([A-Za-z0-9_-]+)", url) if "drive.google.com" in url else None
        gdrive_download(m.group(1), path) if m else http_download(url, path)
    else:
        path = archive_path(ar["name"], cache, cfg)
    if sha256(path) != ar["sha256"]:
        raise SystemExit(f"{path} does not match the SHA256 of {ar['name']} (incomplete or wrong file; delete it and retry)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="config names, e.g. celeba_cifar clwd audio_piano")
    ap.add_argument("--audio", action="store_true", help="also download the audio benchmark (~1.3 GB)")
    ap.add_argument("--data-root", type=Path, default=paths.DATA_ROOT)
    ap.add_argument("--gdrive-folder", help="share link of the release folder (overrides the manifest's gdrive.folder_url)")
    ap.add_argument("--archive-url", nargs="+", default=[], metavar="NAME=URL",
                    help="download link of an archive, e.g. sord_audio_benchmarks.zip=https://... (overrides benchmarks.json)")
    ap.add_argument("--cache-dir", type=Path, default=paths.CACHE_ROOT / "downloads")
    a = ap.parse_args()

    archives = json.loads(MANIFEST.read_text())["archives"]
    configs = sorted({c for ar in archives for f in ar["files"] for c in f["config"].split()})
    if a.only and set(a.only) - set(configs):
        raise SystemExit(f"unknown config(s) {sorted(set(a.only) - set(configs))}; configs: {configs}")
    cfg = dict(load_manifest(full=True).get("gdrive", {}))
    if a.gdrive_folder:
        cfg["folder_url"] = a.gdrive_folder
    urls = dict(u.split("=", 1) for u in a.archive_url)
    bad = []
    for ar in archives:
        if a.only:
            files = [f for f in ar["files"] if set(f["config"].split()) & set(a.only)]
        else:
            files = ar["files"] if ar["kind"] == "image" or a.audio else []
        if not files:
            continue
        todo = [f for f in files if not ((a.data_root / f["path"]).is_file() and sha256(a.data_root / f["path"]) == f["sha256"])]
        print(f"{ar['name']}: {len(files) - len(todo)} of {len(files)} files already verified")
        if not todo:
            continue
        with zipfile.ZipFile(fetch_archive(ar, urls.get(ar["name"]) or ar.get("url"), a.cache_dir, cfg)) as z:
            for f in todo:
                out = a.data_root / f["path"]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(f["path"]))
                ok = sha256(out) == f["sha256"]
                if not ok:
                    bad.append(f["path"])
                if not ok or len(todo) <= 20:
                    print(f"[{' ok' if ok else 'BAD'}] {f['path']}")
        if len(todo) > 20:
            print(f"{ar['name']}: extracted {len(todo)} files")
    if bad:
        raise SystemExit(f"{len(bad)} file(s) failed the SHA256 check")
    print(f"\nall selected benchmark inputs verified (data root: {a.data_root})")


if __name__ == "__main__":
    main()
