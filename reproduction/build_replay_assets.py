"""Build small replay archives from traced inputs; omit unused NPZ arrays."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "release-assets")
    ap.add_argument("--part-mb", type=int, default=450)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    keys = {}; inputs = set()
    for p in (ROOT / "reproduction/logs").glob("*-inputs.json"):
        inputs.update(json.loads(p.read_text()))
    for p in (ROOT / "reproduction/logs").glob("*-keys.json"):
        for name, ks in json.loads(p.read_text()).items():
            keys.setdefault(name, set()).update(ks)
    inputs = {Path(p) for p in inputs if Path(p).is_file() and Path(p).is_relative_to(ROOT)}
    # Generated figures are outputs. Source PNGs needed by loaders live in
    # baseline result directories, not paper/figs.
    inputs = {p for p in inputs if not p.is_relative_to(ROOT / "paper/figs") or p.suffix == ".npz"}
    manifest = {"format": 1, "description": "Saved inputs and reconstructions for CPU metric replay; NPZ files contain only arrays used by registered replay scripts", "files": [], "archives": []}
    archive = None; size = 0; index = 0
    def close():
        if archive is None: return
        archive.close()
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        manifest['archives'].append({"name":path.name,"bytes":path.stat().st_size,"sha256":digest})
        print(path.name, path.stat().st_size, flush=True)
    for p in sorted(inputs):
        if p.suffix == '.npz':
            used = keys.get(str(p))
            if not used: continue
            buffer = io.BytesIO()
            with zipfile.ZipFile(p) as source, zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as dest:
                for key in sorted(used):
                    dest.writestr(key + '.npy', source.read(key + '.npy'))
            data = buffer.getvalue()
        else:
            data = p.read_bytes(); used = None
        if archive is None or size + len(data) > args.part_mb * 1_000_000:
            close(); index += 1; path = args.output / f'replay-{index:02d}.tar'
            archive = tarfile.open(path, 'w'); size = 0
        name = str(p.relative_to(ROOT)); info = tarfile.TarInfo(name)
        info.size = len(data); info.mode = 0o644; info.mtime = 0
        archive.addfile(info, io.BytesIO(data)); size += len(data)
        record = {"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"archive":path.name}
        if used: record['arrays'] = sorted(used)
        manifest['files'].append(record)
    close()
    (ROOT / 'reproduction/assets.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('assets:',len(manifest['files']),'archives:',len(manifest['archives']),flush=True)

if __name__ == '__main__':main()
