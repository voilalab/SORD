"""E-SP — MSDM baseline on the SPEECH pool with a per-source prior pair.

The fair prior-per-source baseline for the speech-interferer cells:
WeaklyMSDM separation over stems [speech, piano] — piano = the
published ruby-dew-290 ckpt (identical to E1), speech = the DM trained
on LibriSpeech train-clean-100 in the SAME framework/recipe
(train_speech_dm.py). Dirac differential on source 0 (speech = y −
piano), the 2-source analog of the published eval config (Dirac on the
first stem). Sampler config verbatim E1: KarrasSchedule(1e-4, 1.0,
rho 7), s_churn 40, 150 steps, 2 resamples.

WeaklyMSDMSeparator.denoise_fn hardcodes 4 sources — a 2-source
variant is defined here; separate_mixture and differential_with_dirac
are reused unmodified.

Run (msdm env):
  python -m \
    experiments.tango_msdm_2026_08_04.src.e1_speech \
    --speech-ckpt $CHECKPOINT_ROOT/baselines/msdm_speech/speech_dm/best.pt
"""
from __future__ import annotations

import argparse
import os
import functools
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
MSDM = REPO / "experiments/tango_msdm_2026_08_04/baselines/msdm"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MSDM))

from audio_diffusion_pytorch import (                          # noqa: E402
    KarrasSchedule, LogNormalDistribution)
from main.module_base import Model                             # noqa: E402
from main.separation import (                                  # noqa: E402
    differential_with_dirac, separate_mixture)
from experiments.tango_msdm_2026_08_04.src import e2_smoke as E  # noqa: E402

PIANO_CKPT = MSDM / "ckpts/ruby-dew-290/epoch=236-valid_loss=0.010.ckpt"
CK = Path(os.environ.get("SORD_MSDM_CKPT_DIR", "checkpoints/diffusion/msdm"))   # MSDM checkpoint directory
TARGET_CKPT = {"piano": PIANO_CKPT,
               "guitar": CK / "honest-fog-332/epoch=407-valid_loss=0.007.ckpt",
               "bass": CK / "laced-dream-329/epoch=443-valid_loss=0.002.ckpt",
               "drums": CK / "ancient-voice-289/epoch=258-valid_loss=0.019.ckpt"}
NUM_STEPS, NUM_RESAMPLES, S_CHURN = 150, 2, 40.0
STEMS = ("speech", "piano")          # Dirac on source 0 = speech
NFE = NUM_STEPS * NUM_RESAMPLES * len(STEMS)


def load_speech_model(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location="cpu")
    hp = dict(ck["hyper_parameters"])
    m = Model(**hp, diffusion_sigma_distribution=LogNormalDistribution(
        mean=-3.0, std=1.0))
    sd = {k: v for k, v in ck["state_dict"].items()}
    m.load_state_dict(sd)
    return m.to(dev)


def separate_two(models, y, num_steps, seed, stems=STEMS):
    fns = [m.model.diffusion.denoise_fn for m in models]
    dev = y.device

    def denoise_fn(x, sigma):
        xs = [fn(x[:, i:i + 1], sigma=sigma) for i, fn in enumerate(fns)]
        return torch.cat(xs, dim=1)

    torch.manual_seed(seed)
    out = separate_mixture(
        mixture=y, denoise_fn=denoise_fn,
        sigmas=KarrasSchedule(sigma_min=1e-4, sigma_max=1.0,
                              rho=7.0)(num_steps, dev),
        noises=torch.randn(y.shape[0], len(fns), y.shape[-1]).to(dev),
        differential_fn=functools.partial(differential_with_dirac,
                                          source_id=0),
        s_churn=S_CHURN, num_resamples=NUM_RESAMPLES, use_tqdm=False)
    return {s: out[:, i:i + 1, :] for i, s in enumerate(stems)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-ckpt", required=True)
    ap.add_argument("--tag", default="e1_msdm_speech")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--clip0", type=int, default=0)
    ap.add_argument("--nclips", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pools", default="speech",
                    help="comma-separated pools: speech (default) or the "
                         "b50 instrument pools invocab,oov (2026-09-10: "
                         "MSDM+speechDM on instrument-interferer cells)")
    ap.add_argument("--target", default="piano", choices=sorted(TARGET_CKPT),
                    help="target stem / its own MSDM single-source ckpt (2026-09-22: bass/drums/guitar)")
    ap.add_argument("--no-npz", action="store_true")
    ap.add_argument("--manifest", default="manifest.json",
                    help="manifest filename under BENCH (2026-09-24: "
                         "manifest_rawgain_speech.json for the g=1 speech cells)")
    ap.add_argument("--sirs", default="0,5",
                    help="comma-separated cell sir tags (raw-gain manifest: 1)")
    a = ap.parse_args()
    pools = a.pools.split(",")
    stems = ("speech", a.target)
    dev = "cuda"
    man = json.load(open(E.BENCH / a.manifest))
    sirs = [float(x) for x in a.sirs.split(",")]

    models = [load_speech_model(a.speech_ckpt, dev),
              Model.load_from_checkpoint(str(TARGET_CKPT[a.target])).to(dev)]

    root = Path(a.out) if a.out else \
        REPO / "experiments/tango_msdm_2026_08_04/results/arms"
    out = root / a.tag
    out.mkdir(parents=True, exist_ok=True)
    cells = [(ci, pool, sir) for ci in range(a.clip0, a.clip0 + a.nclips)
             for pool in pools for sir in sirs]
    rows = []
    for idx, (ci, pool, sir) in enumerate(cells):
        x22, n22, y22, m, w0, wl, inst = E.load_cell(
            man, ci, pool, rho=0.5, sir=sir, dev=dev)
        seps = separate_two(models, y22, NUM_STEPS, a.seed + idx, stems)
        x_est = seps[a.target].to(dev)
        mt = E.metrics(x_est, x22, y22, m)
        mm = m[0, 0].cpu().numpy()
        nh = (y22 - x_est)[0, 0].cpu().numpy()
        nt = n22[0, 0].cpu().numpy()
        mt.update(cell=[ci, pool, sir], inst=inst, nfe=NFE,
                  nhat_sisdr=E.sisdr(nh[mm], nt[mm]))
        if not a.no_npz: np.savez_compressed(
            out / f"c{ci}_{pool}_s{sir:g}.npz",
            **{f"{a.target}_est": x_est[0, 0].cpu().numpy()},
            speech_est=seps["speech"][0, 0].cpu().numpy(),
            clean=x22[0, 0].cpu().numpy(), y=y22[0, 0].cpu().numpy(),
            n_true=nt, mask=mm, w0=w0, wlen=wl)
        rows.append(mt)
        print(f"  c{ci:03d} {pool:7s} SIR{sir:>4.0f}: "
              f"win {mt['win_sisdr_in']:6.2f} -> {mt['win_sisdr']:6.2f} | "
              f"global {mt['sisdr_in']:6.2f} -> {mt['sisdr']:6.2f} | "
              f"offwin {mt['offwin_maxabs']:.2e} | "
              f"n̂ {mt['nhat_sisdr']:5.2f}", flush=True)

    row = dict(
        tag=a.tag,
        mean_win=float(np.mean([r["win_sisdr"] for r in rows])),
        mean_gain=float(np.mean([r["gain"] for r in rows])),
        offwin_max=float(max(r["offwin_maxabs"] for r in rows)),
        below_floor=int(sum(r["win_sisdr"] < r["win_sisdr_in"]
                            for r in rows)),
        nfe=NFE)
    print(f"[{a.tag}] mean_win {row['mean_win']:.2f} | gain "
          f"{row['mean_gain']:+.2f} | below-floor {row['below_floor']}"
          f"/{len(rows)} | offwin {row['offwin_max']:.1e} | NFE {NFE}")
    json.dump(dict(row=row, config=dict(
        num_steps=NUM_STEPS, num_resamples=NUM_RESAMPLES,
        s_churn=S_CHURN, source_id=0, stems=list(stems), target=a.target,
        speech_ckpt=str(a.speech_ckpt), seed=a.seed, pools=pools,
        manifest=a.manifest, sirs=sirs),
        cells=rows),
        open(out / "summary.json", "w"), indent=1, default=float)
    print(f"-> {out}/summary.json")


if __name__ == "__main__":
    main()
