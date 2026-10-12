"""Run SORD on a benchmark described by a YAML config.

    python -m sord.run configs/sord/celeba_mnist.yaml --images 0:25 --batch 8 --out runs/celeba_mnist

The benchmark npz holds x_A (clean), y (observation) and M_true (support and scoring region), images in
[0, 1].  An optional `support: {file, key}` entry gives the algorithm a different support mask (CLWD:
the dataset's own watermark mask) while PSNR is still computed on M_true; an image whose support is
empty is returned unchanged (x = y).
For each image i this writes <out>/img_<i>.npz (reconstruction, every iterate, sigma* history) and
appends one JSON line to <out>/results.jsonl with the on-support PSNR of the returned estimate and of
every iterate, so the stopping threshold can be studied afterwards without rerunning (sord/stop_sweep.py).
Image i always uses seed  cfg.seed + i,  and sigma* / stopping are per image, so results do not depend on
how images are split across jobs or batches (--batch only sets how many images share each network call).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from sord.algorithm import SORDConfig, sord_batch
from sord.paths import data
from sord.priors import build_prior


def psnr_on(x, ref, omega):
    """PSNR (dB) over the support Omega, images in [0, 1]."""
    err = ((np.clip(x, 0, 1) - ref) ** 2)[np.asarray(omega, bool)]
    return float(10 * np.log10(1.0 / max(err.mean(), 1e-12)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--images", default=None, help="range a:b (default: all)")
    ap.add_argument("--batch", type=int, default=1, help="images per network call (GPU memory permitting)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    bench = np.load(data(cfg["bench"]))             # absolute, or relative to DATA_ROOT
    X, Y, SCORE = bench["x_A"], bench["y"], bench["M_true"] > 0.5
    sup = cfg.get("support")
    OMEGA = np.load(data(sup["file"]))[sup.get("key", "M")] > 0.5 if sup else SCORE
    i0, i1 = (map(int, a.images.split(":")) if a.images else (0, len(Y)))
    alg = SORDConfig(**cfg["sord"])
    prior = build_prior(cfg["prior"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    todo = [i for i in range(i0, i1) if SCORE[i].any()]          # skip images with nothing to score
    for i in [i for i in todo if not OMEGA[i].any()]:             # empty support: nothing to remove
        row = dict(image=i, psnr=psnr_on(Y[i], X[i], SCORE[i]), psnr_input=psnr_on(Y[i], X[i], SCORE[i]),
                   psnr_iterates=[], returned=0, stopped_early=False, sigma_star=[], empty_support=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    todo = [i for i in todo if OMEGA[i].any()]
    for g in range(0, len(todo), a.batch):
        idx = todo[g:g + a.batch]
        t0 = time.time()
        results = sord_batch(Y[idx], OMEGA[idx], prior, alg, seeds=[cfg.get("seed", 0) + i for i in idx])
        sec = round((time.time() - t0) / len(idx), 1)
        for i, r in zip(idx, results):
            np.savez_compressed(out / f"img_{i:03d}.npz", x=r.x.astype(np.float32),
                                iterates=np.stack(r.iterates).astype(np.float32),
                                sigma_star=np.array(r.sigma_star), returned=r.returned)
            row = dict(image=i, psnr=psnr_on(r.x, X[i], SCORE[i]), psnr_input=psnr_on(Y[i], X[i], SCORE[i]),
                       psnr_iterates=[psnr_on(x, X[i], SCORE[i]) for x in r.iterates],
                       returned=r.returned, stopped_early=r.stopped_early,
                       sigma_star=[float(s) for s in r.sigma_star], seconds_per_image=sec)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{i}] PSNR {row['psnr']:.2f} (input {row['psnr_input']:.2f}) returned x_{r.returned} "
                  f"sigma* {[round(s, 4) for s in row['sigma_star']]} {sec}s/img", flush=True)


if __name__ == "__main__":
    main()
