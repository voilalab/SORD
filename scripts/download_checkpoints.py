"""Download the checkpoints listed in checkpoint_manifest.json and verify their SHA256.

    python scripts/download_checkpoints.py --output-dir ./checkpoints                 # official sources
    python scripts/download_checkpoints.py --source gdrive --output-dir ./checkpoints  # authors' weights (Google Drive folder)
    python scripts/download_checkpoints.py --only celeba_score                         # one checkpoint
    python scripts/download_checkpoints.py --include-optional                          # also the baselines' checkpoints

--source official (default) uses each checkpoint's original provider: Hugging Face (needs
`huggingface_hub`), direct HTTP (resumable), or the provider's Google Drive file (needs `gdown`).
The authors' own weights (fine-tuned priors, baseline weights) come from the release package: ONE shared
Google Drive folder holding the sord_*.zip archives. Its share link is set in a single place,
checkpoint_manifest.json -> "gdrive" -> "folder_url" (or pass --gdrive-folder URL). --source gdrive
downloads that folder once (needs `gdown`) and extracts each checkpoint from the archive named in its
"gdrive_archive" field. --gdrive-map FILE.json {zip name: file share link} overrides single archives.

Already verified files are skipped; a failed checksum is reported and the file is kept as *.bad.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from checkpoint_common import check_entry, load_manifest, paths, select


def http_download(url: str, dest: Path):
    """Resumable HTTP download to dest (via dest.part)."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req) as r:
        mode = "ab" if start and r.status == 206 else "wb"
        with open(part, mode) as f:
            shutil.copyfileobj(r, f, length=1 << 22)
    part.rename(dest)


def gdrive_download(file_id: str, dest: Path):
    try:
        import gdown
    except ImportError:
        raise SystemExit("Google Drive downloads need gdown:  pip install gdown")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if gdown.download(id=file_id, output=str(dest), quiet=False, resume=True) is None:
        raise RuntimeError(f"Google Drive download failed for id {file_id} (is the file shared 'anyone with the link'?)")


def drive_id(source: dict) -> str | None:
    if source.get("id"):
        return source["id"]
    m = re.search(r"/d/([A-Za-z0-9_-]+)", source.get("url", ""))
    return m.group(1) if m else None


def fetch_official(key: str, entry: dict, root: Path, cache: Path):
    src, target = entry["source"], root / entry["path"]
    kind = src.get("type")
    if kind == "huggingface":
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("Hugging Face downloads need huggingface_hub:  pip install huggingface_hub")
        snapshot_download(repo_id=src["repo_id"], local_dir=str(target))
    elif kind == "http":
        http_download(src["url"], target)
    elif kind == "gdrive-file":
        fid = drive_id(src)
        if src.get("archive"):                       # provider ships a tar holding several checkpoints
            archive = cache / src["archive"]
            if not archive.exists():
                gdrive_download(fid, archive)
            with tarfile.open(archive) as tar:
                member = next((m for m in tar.getmembers() if m.name.lstrip("./").endswith(src["archive_member"])), None)
                if member is None:
                    raise RuntimeError(f"{src['archive_member']} not found in {archive}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as fin, open(target, "wb") as fout:
                    shutil.copyfileobj(fin, fout, length=1 << 22)
        else:
            gdrive_download(fid, target)
    elif kind == "gdrive-folder":
        raise RuntimeError(f"published only as a Google Drive folder: download '{src.get('file', entry['path'])}' "
                           f"from {src.get('url') or src.get('id')} and place it at {target}")
    else:
        raise RuntimeError("no public source; use --source gdrive (release package) or obtain it from the authors")


def archive_path(name: str, cache: Path, cfg: dict) -> Path:
    """Local copy of a release archive: cache hit, a per-archive link, or the shared folder (downloaded once)."""
    hits = sorted(cache.rglob(name)) if cache.exists() else []
    if hits:
        return hits[0]
    link = (cfg.get("archives") or {}).get(name)
    if link:
        gdrive_download(drive_id({"url": link}) or link, cache / name)
        return cache / name
    folder = cfg.get("folder_url", "")
    if not folder.startswith("http"):
        raise RuntimeError('no Google Drive folder configured: set "gdrive" -> "folder_url" in checkpoint_manifest.json '
                           "(or pass --gdrive-folder URL); see CHECKPOINT_UPLOAD_GUIDE.md")
    try:
        import gdown
    except ImportError:
        raise SystemExit("Google Drive downloads need gdown:  pip install gdown")
    listing = gdown.download_folder(url=folder, skip_download=True, quiet=True) or []   # names + file ids only
    match = [f for f in listing if Path(f.path).name == name]
    if not match:
        raise RuntimeError(f"{name} not found in the Google Drive folder {folder}")
    gdrive_download(match[0].id, cache / name)                                          # only the needed archive
    return cache / name


def fetch_gdrive(key: str, entry: dict, root: Path, cache: Path, cfg: dict):
    """Extract checkpoints/<path> of this checkpoint from its release archive."""
    name = entry.get("gdrive_archive")
    if not name:
        raise RuntimeError("not part of the release package; use --source official")
    prefix = "checkpoints/" + entry["path"]
    with zipfile.ZipFile(archive_path(name, cache, cfg)) as z:
        members = [m for m in z.namelist() if (m == prefix or m.startswith(prefix + "/")) and not m.endswith("/")]
        if not members:
            raise RuntimeError(f"{prefix} not found in {name}")
        for m in members:
            out = root / m[len("checkpoints/"):]
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as fin, open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1 << 22)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=paths.CHECKPOINT_ROOT)
    ap.add_argument("--source", choices=("official", "gdrive"), default="official")
    ap.add_argument("--only", nargs="+", help="checkpoint ids from checkpoint_manifest.json")
    ap.add_argument("--include-optional", action="store_true", help="also download optional (baseline) checkpoints")
    ap.add_argument("--gdrive-folder", help="share link of the release folder (overrides the manifest's gdrive.folder_url)")
    ap.add_argument("--gdrive-map", type=Path, help="JSON {zip name: file share link} for single archives")
    ap.add_argument("--cache-dir", type=Path, default=paths.CACHE_ROOT / "downloads")
    a = ap.parse_args()

    cfg = dict(load_manifest(full=True).get("gdrive", {}))
    if a.gdrive_folder:
        cfg["folder_url"] = a.gdrive_folder
    if a.gdrive_map:
        cfg["archives"] = {**(cfg.get("archives") or {}), **json.load(open(a.gdrive_map))}
    failed = []
    for key, entry in select(load_manifest(), a.only, a.include_optional).items():
        if all(s == "ok" for _, s in check_entry(entry, a.output_dir)):
            print(f"[skip] {key}: already verified")
            continue
        print(f"[get ] {key} -> {a.output_dir / entry['path']}")
        try:
            if a.source == "gdrive":
                fetch_gdrive(key, entry, a.output_dir, a.cache_dir, cfg)
            else:
                fetch_official(key, entry, a.output_dir, a.cache_dir)
        except Exception as e:                       # report and continue with the other checkpoints
            print(f"[FAIL] {key}: {e}")
            failed.append(key)
            continue
        bad = [(p, s) for p, s in check_entry(entry, a.output_dir) if s != "ok"]
        for p, s in bad:
            print(f"[FAIL] {key}: {s}: {p}")
            if p.exists() and s != "missing":
                p.rename(p.with_name(p.name + ".bad"))
        (failed.append(key) if bad else print(f"[ ok ] {key}: verified"))
    if failed:
        raise SystemExit(f"\n{len(failed)} checkpoint(s) not obtained: {', '.join(failed)} (see CHECKPOINTS.md)")
    print("\nall selected checkpoints downloaded and verified")


if __name__ == "__main__":
    main()
