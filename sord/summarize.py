"""Summarize SORD runs: mean +- std over images (or audio cells) of the reported metric.

    python -m sord.summarize runs/celeba_cifar runs/audio_piano ...

Image runs report on-support PSNR (dB); audio runs report window SI-SDR improvement (dB), split by SIR.
The standard deviation is the sample standard deviation (ddof = 1), as in the paper's tables.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def describe(rows, key):
    v = np.array([r[key] for r in rows], np.float64)
    returned = np.bincount([r["returned"] for r in rows]).tolist()
    s1 = np.array([r["sigma_star"][0] for r in rows])
    s2 = np.array([r["sigma_star"][1] for r in rows if len(r["sigma_star"]) > 1])
    out = f"n={len(v):3d}  {v.mean():6.2f} +- {v.std(ddof=1) if len(v) > 1 else 0:5.2f}  returned x_m counts {returned}"
    out += f"  sigma*_1 {s1.min():.4f}-{s1.max():.4f}"
    if len(s2):
        out += f"  sigma*_2 {s2.min():.4f}-{s2.max():.4f}"
    return out


def main(paths):
    for p in paths:
        f = Path(p) / "results.jsonl"
        if not f.exists():
            print(f"{p}: no results.jsonl")
            continue
        rows = [json.loads(line) for line in open(f)]
        if rows and "sisdri" in rows[0]:                       # audio
            by_sir = defaultdict(list)
            for r in rows:
                by_sir[r["sir"]].append(r)
            for sir, rs in sorted(by_sir.items()):
                print(f"{p}  SIR {sir:g}  SI-SDRi  {describe(rs, 'sisdri')}")
        else:
            print(f"{p}  PSNR_Omega  {describe(rows, 'psnr')}")


if __name__ == "__main__":
    main(sys.argv[1:])
