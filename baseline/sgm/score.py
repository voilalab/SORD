"""Score SGM baseline runs exactly like the paper's tables: on-support PSNR of clip(x_A, 0, 1) per image,
mean +- sample std (ddof = 1) over the 100 images.

    python baselines/sgm/score.py natural sgmM sgm
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sord.paths import OUTPUT_ROOT, data       # noqa: E402

BENCH = "experiments/ros/results/r5/bench_{}.npz"


def main(bench, arms):
    Z = np.load(data(BENCH.format(bench)))
    gt = Z["x_A"]
    M = np.repeat((np.asarray(Z["M_true"]) > 0.5)[..., None], 3, -1)
    for arm in arms:
        fs = sorted(glob.glob(str(OUTPUT_ROOT / "sgm" / bench / arm / "c*.npz")), key=lambda s: int(Path(s).stem[1:]))
        xa = np.concatenate([np.load(f)["x_A"] for f in fs])
        p = np.array([10 * np.log10(1 / max(((np.clip(xa[i], 0, 1) - gt[i]) ** 2)[M[i]].mean(), 1e-20))
                      for i in range(len(xa))])
        print(f"{bench} {arm:10s} n={len(p):3d}  PSNR_Omega {p.mean():.2f} +- {p.std(ddof=1):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
