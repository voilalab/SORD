"""MSDM baseline (Mariani et al.) on the SORD audio benchmark: the paper's "MSDM" column.

    python -m baselines.msdm.run_msdm configs/sord/audio_piano.yaml --out runs/msdm_piano                 # 4-stem MSDM
    python -m baselines.msdm.run_msdm configs/sord/audio_piano_speech.yaml --out runs/msdm_piano_speech         # MSDM + speech

Reads the same SORD audio config as `python -m sord.run_audio` (benchmark root, pool, clips, SIRs, window) and
scores with the same benchmark code (window SI-SDR at 22.05 kHz), so its results.jsonl is directly comparable.

* default: weakly-supervised MSDM with the four published Slakh single-source models (bass, drums, guitar, piano)
  separated jointly, Dirac likelihood on source 0 (the authors' eval_weakly_msdm_dirac configuration); the
  stem of the config's target is scored.
* --speech (default when the pool is speech, e.g. configs/sord/audio_<target>_speech.yaml): "MSDM + speech prior",
  the baseline the paper uses when the interferer is speech: two sources,
  [speech, target], with the speech diffusion model trained by the authors on LibriSpeech in the MSDM
  framework (checkpoint `speech_dm`), Dirac likelihood on the speech source.

Sampler (both): Karras schedule sigma 1e-4..1 (rho 7), 150 steps, 2 resamples, s_churn 40.
Cell k of a run (clips in order, then SIRs) uses torch seed  --seed + k, as in the paper's runs (see README.md).
Needs the audio environment (environment-audio.yml) and the MSDM code in third_party/msdm.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from sord import paths

STEMS4 = ("bass", "drums", "guitar", "piano")        # order of the authors' eval config; Dirac on source 0
NUM_STEPS, NUM_RESAMPLES, S_CHURN = 150, 2, 40.0


def load_speech_model(Model, ckpt_path, dev):
    from audio_diffusion_pytorch import LogNormalDistribution
    ck = torch.load(ckpt_path, map_location="cpu")
    m = Model(**dict(ck["hyper_parameters"]), diffusion_sigma_distribution=LogNormalDistribution(mean=-3.0, std=1.0))
    m.load_state_dict(ck["state_dict"])
    return m.to(dev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="a SORD audio config, configs/sord/audio_<target>.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", default=None, help="range a:b of clip indices (default: the config's)")
    ap.add_argument("--pool", default=None, help="interferer pool (default: the config's; 'speech' for speech interferers)")
    ap.add_argument("--speech", action=argparse.BooleanOptionalAction, default=None,
                    help="MSDM + speech prior (two sources) instead of the 4-stem MSDM (default: on iff the pool is speech)")
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    b, prior = cfg["bench"], cfg["prior"]
    target = prior["checkpoint"].removeprefix("msdm_")
    msdm_repo = Path(prior.get("msdm_repo", "third_party/msdm"))
    sys.path.insert(0, str(msdm_repo if msdm_repo.is_absolute() else paths.PROJECT_ROOT / msdm_repo))
    from audio_diffusion_pytorch import KarrasSchedule
    from main.module_base import Model
    from main.separation import WeaklyMSDMSeparator, differential_with_dirac, separate_mixture
    from experiments.tango_msdm_2026_08_04.src import e2_smoke as bench_io

    bench_io.BENCH = paths.data(b.get("root", "data/audio/slakh_bench_v1"))
    manifest = json.load(open(bench_io.BENCH / b.get("manifest", "manifest.json")))
    pool = a.pool or b["pool"]
    if a.speech is None:
        a.speech = pool == "speech"
    c0, c1 = map(int, (a.clips or b["clips"]).split(":"))
    for c in range(c0, c1):
        tc = manifest["clips"][c]["target_class"].lower()
        assert tc == target, f"clip {c} has target {tc}, config target is {target}"
    dev = "cuda"
    sigmas = KarrasSchedule(sigma_min=1e-4, sigma_max=1.0, rho=7.0)
    if a.speech:
        stems = ("speech", target)
        models = [load_speech_model(Model, paths.checkpoint("speech_dm"), dev),
                  Model.load_from_checkpoint(str(paths.checkpoint(prior["checkpoint"]))).to(dev)]
        fns = [m.model.diffusion.denoise_fn for m in models]

        def denoise_fn(x, sigma):
            return torch.cat([fn(x[:, i:i + 1], sigma=sigma) for i, fn in enumerate(fns)], dim=1)

        def separate(y):
            out = separate_mixture(
                mixture=y, denoise_fn=denoise_fn, sigmas=sigmas(NUM_STEPS, dev),
                noises=torch.randn(y.shape[0], len(fns), y.shape[-1]).to(dev),
                differential_fn=functools.partial(differential_with_dirac, source_id=0),
                s_churn=S_CHURN, num_resamples=NUM_RESAMPLES, use_tqdm=False)
            return {s: out[:, i:i + 1, :] for i, s in enumerate(stems)}
    else:
        stems = STEMS4
        separator = WeaklyMSDMSeparator(
            stem_to_model={s: Model.load_from_checkpoint(str(paths.checkpoint(f"msdm_{s}"))).to(dev) for s in STEMS4},
            sigma_schedule=sigmas, differential_fn=functools.partial(differential_with_dirac, source_id=0),
            s_churn=S_CHURN, num_resamples=NUM_RESAMPLES, use_tqdm=False)

        def separate(y):
            return separator.separate(mixture=y, num_steps=NUM_STEPS)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(dict(cfg, baseline=dict(
        method="msdm+speech" if a.speech else "msdm-4stem", stems=list(stems), pool=pool, clips=f"{c0}:{c1}",
        num_steps=NUM_STEPS, num_resamples=NUM_RESAMPLES, s_churn=S_CHURN, seed=a.seed))))
    cells = [(c, s) for c in range(c0, c1) for s in b["sirs"]]
    for k, (c, s) in enumerate(cells):
        x22, _, y22, m, *_, inst = bench_io.load_cell(manifest, c, pool, rho=b.get("rho", 0.5), sir=float(s), dev=dev)
        t0 = time.time()
        torch.manual_seed(a.seed + k)
        est = separate(y22)[target].to(dev)
        r = bench_io.metrics(est, x22, y22, m)
        row = dict(clip=c, sir=s, pool=pool, instrument=inst, method="msdm+speech" if a.speech else "msdm-4stem",
                   sisdri=r["win_sisdr"] - r["win_sisdr_in"], win_sisdr=r["win_sisdr"], win_sisdr_in=r["win_sisdr_in"],
                   seed=a.seed + k, seconds_per_cell=round(time.time() - t0, 1))
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[clip {c} SIR {s} {pool}] SI-SDRi {row['sisdri']:.2f}  {row['seconds_per_cell']}s", flush=True)


if __name__ == "__main__":
    main()
