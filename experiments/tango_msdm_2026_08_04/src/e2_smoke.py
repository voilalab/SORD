"""E2 smoke — tango_wave oracle-window arm on a bench_v1 subset + gates.

Per cell: recompose y at 44.1k (SHA-verified), resample parts to 22050
(resampling is linear => additivity survives exactly), crop to 26*8192,
oracle mask = interferer window dilated 256 samples (resampler tails).
Gates on the first cell: O (oracle-n isolates the x-solve) and circ
(identity n-prior => output ~ x_inp). Metrics: SI-SDR global + in-window
vs the input floor, and max |x_alt − y| off-window (C1 exactness).

Run: python -m \
  experiments.tango_msdm_2026_08_04.src.e2_smoke --tag s1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.tango_msdm_2026_08_04.src.msdm_prior import (   # noqa: E402
    MSDMPrior, SR_MODEL, resample)
from experiments.tango_msdm_2026_08_04.src import tango_wave as TW  # noqa: E402

def _default_bench():
    from sord.paths import DATA_ROOT
    return DATA_ROOT / "data/audio/slakh_bench_v1"


BENCH = Path(os.environ.get("SORD_AUDIO_BENCH", _default_bench()))   # wav/ + manifest.json
CLIP_LEN = 26 * 8192          # 212992 @ 22050
DILATE = 256                  # oracle-mask pad for resampler tails


def sisdr(est, ref):
    a = float(np.dot(est, ref) / max(np.dot(ref, ref), 1e-20))
    e = est - a * ref
    return 10 * np.log10(max(np.dot(a * ref, a * ref), 1e-20)
                         / max(np.dot(e, e), 1e-20))


def load_cell(man, ci, pool, rho, sir, dev):
    c = man["clips"][ci]
    xA, sr = sf.read(BENCH / "wav" / f"c{c['idx']:03d}_xA.wav",
                     dtype="float32")
    sB, _ = sf.read(BENCH / "wav" / f"c{c['idx']:03d}_{pool}_sB.wav",
                    dtype="float32")
    cell = next(k for k in c["pools"][pool]["cells"]
                if k["rho"] == rho and k["sir"] == sir)
    mw = np.zeros(len(xA), np.float32)
    mw[cell["w0"]:cell["w0"] + cell["wlen"]] = 1.0
    n44 = (cell["gain"] * (mw * sB)).astype(np.float32)
    y44 = (xA + n44).astype(np.float32)
    assert hashlib.sha256(y44.tobytes()).hexdigest() == cell["y_sha"]

    def to22(w):
        t = torch.from_numpy(np.ascontiguousarray(w))[None, None, :]
        return resample(t, sr, SR_MODEL)[..., :CLIP_LEN].to(dev)
    x22, n22, y22 = to22(xA), to22(n44), to22(y44)
    w0 = max(cell["w0"] // 2 - DILATE, 0)
    w1 = min((cell["w0"] + cell["wlen"]) // 2 + DILATE, CLIP_LEN)
    m = torch.zeros_like(y22, dtype=torch.bool)
    m[..., w0:w1] = True
    return x22, n22, y22, m, w0, w1 - w0, c["pools"][pool]["inst"]


def metrics(x_alt, x22, y22, m):
    xa, xr, yn = (t[0, 0].cpu().numpy() for t in (x_alt, x22, y22))
    mm = m[0, 0].cpu().numpy()
    out = dict(
        sisdr=sisdr(xa, xr), sisdr_in=sisdr(yn, xr),
        win_sisdr=sisdr(xa[mm], xr[mm]), win_sisdr_in=sisdr(yn[mm], xr[mm]),
        offwin_maxabs=float(np.abs((xa - yn)[~mm]).max()) if (~mm).any()
        else 0.0)
    out["gain"] = out["sisdr"] - out["sisdr_in"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--gates-only", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = json.load(open(BENCH / "manifest.json"))
    prior = MSDMPrior(device=dev)
    kw = dict(M=a.M, K=a.K, n_solve_steps=a.steps)
    rep = dict(tag=a.tag, **kw, gates={}, cells=[])

    # ---- gates on cell (clip0, invocab, rho .5, SIR 0) -------------------
    x22, n22, y22, m, w0, wl, inst = load_cell(man, 0, "invocab", 0.5, 0.0,
                                               dev)
    r = TW.tango_wave(prior, y22, m, w0, wl, oracle_n=n22, K=a.K,
                      n_solve_steps=a.steps)
    gO = metrics(r["x_alt"], x22, y22, m)
    rep["gates"]["O"] = gO
    print(f"GATE O (oracle-n): SI-SDR {gO['sisdr']:.2f} (input "
          f"{gO['sisdr_in']:.2f}, win {gO['win_sisdr']:.2f}) -> "
          f"{'PASS' if gO['sisdr'] > 20 else 'FAIL'} (bar 20)")

    r = TW.tango_wave(prior, y22, m, w0, wl, identity_n=True, blind=False,
                      **kw, verbose=False)
    d_inp = float((r["x_alt"] - r["x_inp"])[m].pow(2).mean().sqrt())
    rms_w = float(y22[m].pow(2).mean().sqrt())
    rep["gates"]["circ"] = dict(rms_dx_vs_inp=d_inp, win_rms=rms_w)
    print(f"GATE circ (identity n): RMS(x_alt - x_inp) on window "
          f"{d_inp:.5f} vs window RMS {rms_w:.4f}")
    if a.gates_only:
        json.dump(rep, open(REPO / "experiments/tango_msdm_2026_08_04/"
                            f"results/e2_smoke_{a.tag}.json", "w"),
                  indent=1)
        return

    # ---- oracle-window cells --------------------------------------------
    for ci in range(a.clips):
        for pool in ("invocab", "oov"):
            for sir in (0.0, 5.0):
                x22, n22, y22, m, w0, wl, inst = load_cell(
                    man, ci, pool, 0.5, sir, dev)
                r = TW.tango_wave(prior, y22, m, w0, wl, **kw,
                                  verbose=False)
                mt = metrics(r["x_alt"], x22, y22, m)
                mt.update(clip=ci, pool=pool, inst=inst, rho=0.5, sir=sir,
                          curves=r["curves"])
                rep["cells"].append(mt)
                print(f"  c{ci:03d} {pool:8s} {inst:10s} SIR{sir:>4.0f}: "
                      f"SI-SDR {mt['sisdr_in']:6.2f} -> {mt['sisdr']:6.2f} "
                      f"({mt['gain']:+.2f}) | win {mt['win_sisdr_in']:6.2f}"
                      f" -> {mt['win_sisdr']:6.2f} | offwin "
                      f"{mt['offwin_maxabs']:.2e}")
    cells = rep["cells"]
    print(f"\nmean gain {np.mean([c['gain'] for c in cells]):+.2f} dB | "
          f"improved {np.mean([c['gain'] > 0 for c in cells]):.2f} | "
          f"max offwin err {max(c['offwin_maxabs'] for c in cells):.2e}")
    out = REPO / "experiments/tango_msdm_2026_08_04/results" / \
        f"e2_smoke_{a.tag}.json"
    json.dump(rep, open(out, "w"), indent=1, default=float)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
