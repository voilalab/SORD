"""Package the redistributable checkpoints into upload-ready zip archives (one per group).

    python scripts/package_checkpoints.py --checkpoint-dir ./checkpoints --output-dir ./release_package

Only entries whose manifest field "redistribution" starts with "allowed" are packaged (the authors' own
weights and their CC BY-NC-SA derivatives); third-party checkpoints with unclear or restrictive terms are
listed as excluded and must be downloaded from their providers.  Every file is verified against its
SHA256 first; archives store files under checkpoints/<manifest path> (uncompressed), so unzipping at the
repository root reproduces the layout scripts/download_checkpoints.py --source gdrive expects.
Writes <output-dir>/<group>.zip, SHA256SUMS.txt (archives and files) and package_manifest.json
(checkpoint id -> archive, matching the manifest's "gdrive_archive" fields). Upload the archives into one
Google Drive folder and put its share link in checkpoint_manifest.json -> "gdrive" -> "folder_url".

--third-party packages the excluded third-party checkpoints instead (sord_thirdparty_*.zip) for a private
backup or private sharing with collaborators.  Do not publish those archives unless the provider's
license permits redistribution; the public instructions point to the original providers.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from checkpoint_common import check_entry, files_of, load_manifest, paths, sha256


def group_of(key: str, entry: dict) -> str:
    if not entry["redistribution"].startswith("allowed"):
        if key.startswith("msdm_") and entry["required"] != "optional":
            return "sord_thirdparty_msdm_instrument_priors"
        return f"sord_thirdparty_{key}"
    if entry["category"] == "method":
        return "sord_method_edm2_priors"
    if key.startswith("sgm_"):
        return "sord_baseline_sgm_noise_priors"
    if key.startswith("slbr_"):
        return "sord_baseline_slbr_scratch"
    if key == "speech_dm":
        return "sord_baseline_msdm_speech_dm"
    return f"sord_{entry['category']}_{key}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", type=Path, default=paths.CHECKPOINT_ROOT)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--third-party", action="store_true",
                    help="package the third-party checkpoints SORD needs (private use; see above)")
    a = ap.parse_args()

    groups, excluded = defaultdict(list), []
    for key, entry in load_manifest().items():
        third = not entry["redistribution"].startswith("allowed")
        if third != a.third_party or (a.third_party and entry["required"] == "optional"):
            if not a.third_party:
                excluded.append((key, entry["redistribution"]))
            continue
        bad = [(p, s) for p, s in check_entry(entry, a.checkpoint_dir) if s != "ok"]
        if bad:
            raise SystemExit(f"{key}: {bad[0][1]}: {bad[0][0]} -- fix before packaging (scripts/verify_checkpoints.py)")
        groups[group_of(key, entry)].append((key, entry))

    a.output_dir.mkdir(parents=True, exist_ok=True)
    sums, package = [], {}
    for name, members in sorted(groups.items()):
        zpath = a.output_dir / f"{name}.zip"
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for key, entry in members:
                for p, _, digest in files_of(entry, a.checkpoint_dir):
                    arc = "checkpoints/" + str(p.relative_to(a.checkpoint_dir))
                    z.write(p.resolve(), arc)                          # follows symlinks: real bytes
                    sums.append(f"{digest}  {arc}")
                package[key] = zpath.name
        sums.append(f"{sha256(zpath)}  {zpath.name}")
        print(f"{zpath.name}: {', '.join(k for k, _ in members)} ({zpath.stat().st_size / 1e9:.2f} GB)")
    (a.output_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    (a.output_dir / "package_manifest.json").write_text(json.dumps(package, indent=1) + "\n")
    if excluded:
        print("\nexcluded (download from the original provider):")
        for key, why in excluded:
            print(f"  {key:30s} {why}")


if __name__ == "__main__":
    main()
