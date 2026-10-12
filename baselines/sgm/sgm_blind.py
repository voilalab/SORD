"""ROS R5 — SGM baseline (Stevens et al. 2025 joint-posterior sampler,
PiGDM guidance) on the SAME 100-image additive benches, chunked.

Blind (mask-free) joint recovery of (x_A, n) from y — identical y to
our runs. Noise prior: mnist -> the repo's joint-diffusion-mnist-rgb
(the paper's own); natural AND watermark -> models/natural-noise-rgb
(no watermark prior exists; domain mismatch on wm is FLAGGED in the
report).

Usage (TensorFlow environment, GPU), the paper's "SGM 2 priors" column:
  python baselines/sgm/sgm_blind.py --bench natural --i0 0 --i1 25
Writes OUTPUT_ROOT/sgm/<bench>/<arm>/c<i0>.npz (x_A, x_B).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse                                                # noqa: E402
import sys                                                     # noqa: E402
from pathlib import Path                                       # noqa: E402

import numpy as np                                             # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from baselines.sgm.sgm_config import build_config             # noqa: E402
from utils.inverse import get_denoiser                         # noqa: E402
from utils.runs import assert_run_exists, init_config          # noqa: E402

from sord.paths import OUTPUT_ROOT, checkpoint, data      # noqa: E402

BENCH = "experiments/ros/results/r5/bench_{}.npz"          # relative to DATA_ROOT (replay release layout)
CORRUPTOR = {"natural": "sgm_natural_patch_prior",      # noise priors used for the paper rows
             "cifar": "sgm_cifar_patch_prior"}         # (checkpoint_manifest.json identifiers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True,
                    choices=("natural", "cifar"))
    ap.add_argument("--i0", type=int, required=True)
    ap.add_argument("--i1", type=int, required=True)
    ap.add_argument("--num-scales", type=int, default=600)
    ap.add_argument("--guidance", default="pigdm")
    ap.add_argument("--corruptor", default=None,
                    help="override the noise-block score model run id "
                         "(e.g. ./models/joint-diffusion-celeba = the "
                         "signal prior in both blocks)")
    ap.add_argument("--arm", default="sgm",
                    help="output subdir under results/r5/<bench>/")
    a = ap.parse_args()

    Z = np.load(data(BENCH.format(a.bench)))
    sl = slice(a.i0, a.i1)
    x_A_gt = np.asarray(Z["x_A"], np.float32)[sl]
    # The paper's observation model is the MIXTURE y = (1-a)x + a*n with
    # a = noise_stddev = 0.5 (corruptors.py corrupt()). Our bench is
    # additive y = x + n, so y/2 = 0.5x + 0.5n is EXACTLY their model
    # with the same x and n — a lossless scalar reparameterization.
    y = np.asarray(Z["y"], np.float32)[sl] / 2.0
    k = y.shape[0]

    config, run_ids = build_config("local/celeba_mnist_pigdm", k,
                                   a.num_scales, a.guidance, run_id_sgm=checkpoint("celeba_score"))
    config.sgm.corruptor_run_id = str(checkpoint(a.corruptor or CORRUPTOR[a.bench]))
    run_id = str(checkpoint("celeba_score"))                 # clean-image prior (Stevens et al. CelebA)
    assert_run_exists(run_id, "sgm")
    model_config = init_config(run_id, config)
    Denoiser = get_denoiser("sgm")
    denoiser = Denoiser(config=model_config, dataset=None, verbose=True)

    import tensorflow as tf
    out = denoiser(noisy_samples=tf.convert_to_tensor(y, tf.float32),
                   target_samples=tf.convert_to_tensor(x_A_gt,
                                                       tf.float32),
                   plot=False)
    x_A_sgm, x_B_sgm = out
    d = OUTPUT_ROOT / "sgm" / a.bench / a.arm
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"c{a.i0}.npz",
                        x_A=np.asarray(x_A_sgm, np.float32),
                        x_B=np.asarray(x_B_sgm, np.float32))
    print(f"[{a.arm} {a.bench} c{a.i0}] saved")


if __name__ == "__main__":
    main()
