"""Run SORD on the audio benchmark (localized interferer removal) described by a YAML config.

    python -m sord.run_audio configs/sord/audio_piano.yaml --out runs/audio_piano --batch 4

Each benchmark cell is one 10 s clip of the target source with an interferer added inside one contiguous
window (the support), at a given SIR.  Cells are loaded and scored with the paper's benchmark code
(experiments/tango_msdm_2026_08_04/src/e2_smoke.py): 22.05 kHz waveforms, window SI-SDR.  For each cell
this appends one JSON line to <out>/results.jsonl with the window SI-SDR improvement (SI-SDRi) of the
returned estimate and of every iterate, the returned iterate and the sigma* history.
Cell (clip c, SIR s) always uses seed  cfg.seed + 1000 * c + s.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from sord.algorithm import SORDConfig, sord_batch
from sord.paths import data
from sord.priors import build_prior


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--clips", default=None, help="range a:b of clip indices (default: the config's)")
    ap.add_argument("--pool", default=None, help="interferer pool (default: the config's; 'speech' for speech interferers)")
    ap.add_argument("--batch", type=int, default=1, help="cells per network call (GPU memory permitting)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from experiments.tango_msdm_2026_08_04.src import e2_smoke as bench_io
    cfg = yaml.safe_load(Path(a.config).read_text())
    b = cfg["bench"]
    bench_io.BENCH = data(b.get("root", "data/audio/slakh_bench_v1"))   # wav/ + manifest.json, relative to DATA_ROOT
    c0, c1 = map(int, (a.clips or b["clips"]).split(":"))
    manifest = json.load(open(bench_io.BENCH / b.get("manifest", "manifest.json")))
    pool = a.pool or b["pool"]
    alg = SORDConfig(**cfg["sord"])
    prior = build_prior(cfg["prior"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(dict(cfg, bench=dict(b, pool=pool))))
    cells = [(c, s) for c in range(c0, c1) for s in b["sirs"]]
    for g in range(0, len(cells), a.batch):
        group = cells[g:g + a.batch]
        loaded = [bench_io.load_cell(manifest, c, pool, rho=b.get("rho", 0.5), sir=float(s), dev="cpu")
                  for c, s in group]
        Y = np.stack([cell[2][0, 0].numpy()[:, None] for cell in loaded]).astype(np.float64)
        OMEGA = np.stack([cell[3][0, 0].numpy() for cell in loaded])
        t0 = time.time()
        results = sord_batch(Y, OMEGA, prior, alg, seeds=[cfg.get("seed", 0) + 1000 * c + int(s) for c, s in group])
        sec = round((time.time() - t0) / len(group), 1)
        for (c, s), cell, r in zip(group, loaded, results):
            x22, y22, m = cell[0], cell[2], cell[3]
            score = lambda est: bench_io.metrics(torch.from_numpy(est[:, 0].astype(np.float32))[None, None], x22, y22, m)
            ret = score(r.x)
            row = dict(clip=c, sir=s, pool=pool, instrument=cell[6], sisdri=ret["win_sisdr"] - ret["win_sisdr_in"],
                       win_sisdr=ret["win_sisdr"], win_sisdr_in=ret["win_sisdr_in"],
                       sisdri_iterates=[score(x)["win_sisdr"] - ret["win_sisdr_in"] for x in r.iterates],
                       returned=r.returned, stopped_early=r.stopped_early,
                       sigma_star=[float(v) for v in r.sigma_star], seconds_per_cell=sec)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[clip {c} SIR {s}] SI-SDRi {row['sisdri']:.2f} returned x_{r.returned} "
                  f"sigma* {[round(v, 4) for v in row['sigma_star']]} {sec}s/cell", flush=True)


if __name__ == "__main__":
    main()
