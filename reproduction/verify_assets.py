"""Verify extracted replay inputs against the release manifest."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'reproduction/assets.json').read_text())
    errors=[]
    for record in manifest['files']:
        p=ROOT/record['path']
        if not p.is_file():errors.append(record['path']+': missing');continue
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=record['sha256']:errors.append(record['path']+': checksum mismatch')
    print(f"Verified {len(manifest['files'])-len(errors)} / {len(manifest['files'])} inputs")
    for error in errors:print(error)
    return bool(errors)
if __name__=='__main__':sys.exit(main())
