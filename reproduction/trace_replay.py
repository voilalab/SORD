"""Record binary inputs used by one CPU replay script (local preparation)."""
import atexit
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
seen = set()
npz_keys = {}
manifest = Path(os.environ.get("SORD_TRACE", "/tmp/sord-trace.json"))

def hook(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    try:
        p = Path(os.fsdecode(args[0])).absolute()
        if p.suffix.lower() in {".npz", ".npy", ".png", ".jpg", ".jpeg"}:
            seen.add(str(p))
    except (ValueError, TypeError):
        pass

def save():
    manifest.write_text(json.dumps(sorted(seen), indent=2))
    manifest.with_name(manifest.stem + "-keys.json").write_text(json.dumps(
        {k: sorted(v) for k, v in npz_keys.items()}, indent=2))

atexit.register(save)
sys.addaudithook(hook)
import numpy as np
_getitem = np.lib.npyio.NpzFile.__getitem__
def traced_getitem(self, key):
    filename = getattr(self.zip, "filename", None)
    if filename:
        npz_keys.setdefault(str(Path(filename).absolute()), set()).add(key)
    return _getitem(self, key)
np.lib.npyio.NpzFile.__getitem__ = traced_getitem
script = sys.argv.pop(1)
sys.argv[0] = script
sys.path.insert(0, str(ROOT))
runpy.run_path(script, run_name="__main__")
