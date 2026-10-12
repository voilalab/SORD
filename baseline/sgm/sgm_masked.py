"""ROS R5 — MASKED SGM baseline (new file; r5_sgm.py untouched).

Same Stevens joint-posterior PiGDM sampler as r5_sgm.py, but the sampler
is given the exact support mask our exact-mask arms get: after every
joint guidance update, both blocks are re-projected RePaint-style —

  x <- M*x + (1-M)*(x_known + sigma_t*z)     x_known = y_add off-Omega
  n <- M*n + (1-M)*(sigma_t*z)               n = 0 off-Omega (exact)

(VE forward-diffused replacement; the mask information shapes the whole
trajectory — terminal-only projection would be invisible to the on-Omega
metric.) Everything else (priors, coeffs, 600 scales, y/2 mixture
reparameterization) is r5_sgm.py verbatim.

Usage (TensorFlow environment, GPU), the protocol of the paper's "SGM 2 priors + mask" column:
  python baselines/sgm/sgm_masked.py --bench natural --i0 0 --i1 25 --fix-init --final-composite
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


def mask_the_sampler(sampler, M3, x_known, project="both"):
    """Wrap joint_update_fn with the per-step masked re-projection.

    project="both": n -> diffused 0 AND x -> diffused y_add off-Omega.
    project="n":    n-support constraint only; x stays guidance-driven.
    """
    import tensorflow as tf

    M = tf.convert_to_tensor(M3, tf.float32)
    xk = tf.convert_to_tensor(x_known, tf.float32)
    orig = sampler.guidance.joint_update_fn
    sde = sampler.sde

    def masked_update(y, x, n, vec_t, x_mean, n_mean, gx, gn):
        x, n = orig(y, x, n, vec_t, x_mean, n_mean, gx, gn)
        _, std = sde.marginal_prob(tf.zeros_like(x), vec_t)
        s = tf.reshape(std, (-1, 1, 1, 1))
        if project == "both":
            x = M * x + (1.0 - M) * (xk + s * tf.random.normal(tf.shape(x)))
        n = M * n + (1.0 - M) * (s * tf.random.normal(tf.shape(n)))
        return x, n

    sampler.guidance.joint_update_fn = masked_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True,
                    choices=("natural", "cifar"))
    ap.add_argument("--i0", type=int, required=True)
    ap.add_argument("--i1", type=int, required=True)
    ap.add_argument("--num-scales", type=int, default=600)
    ap.add_argument("--guidance", default="pigdm")
    ap.add_argument("--corruptor", default=None)
    ap.add_argument("--mask-key", default="M_true")
    ap.add_argument("--project", default="both", choices=("both", "n"))
    ap.add_argument("--arm", default="sgmM")
    ap.add_argument("--fix-init", action="store_true",
                    help="start both chains at std_T (9.845) instead of the "
                         "shipped simple.prior_sampling N(0,1) (repo bug)")
    ap.add_argument("--final-composite", action="store_true",
                    help="output = sampler result ON Omega only; off-Omega "
                         "x = y_add, n = 0 exactly (the last-step projection "
                         "leaves sigma_eps~0.03 noise off-Omega otherwise)")
    a = ap.parse_args()

    Z = np.load(data(BENCH.format(a.bench)))
    sl = slice(a.i0, a.i1)
    x_A_gt = np.asarray(Z["x_A"], np.float32)[sl]
    y_add = np.asarray(Z["y"], np.float32)[sl]
    m2 = (np.asarray(Z[a.mask_key]) > 0.5)[sl]
    M3 = np.repeat(m2[..., None], 3, -1).astype(np.float32)
    # mixture reparameterization, exactly r5_sgm.py
    y = y_add / 2.0

    config, run_ids = build_config("local/celeba_mnist_pigdm", y.shape[0],
                                   a.num_scales, a.guidance, run_id_sgm=checkpoint("celeba_score"))
    config.sgm.corruptor_run_id = str(checkpoint(a.corruptor or CORRUPTOR[a.bench]))
    run_id = str(checkpoint("celeba_score"))                 # clean-image prior (Stevens et al. CelebA)
    assert_run_exists(run_id, "sgm")
    model_config = init_config(run_id, config)
    Denoiser = get_denoiser("sgm")
    denoiser = Denoiser(config=model_config, dataset=None, verbose=True)

    if a.fix_init:
        import tensorflow as tf
        sde = denoiser.sampler.sde
        stdT = float(sde.marginal_prob(tf.zeros([1]), tf.constant([sde.T]))[1][0])
        sde.prior_sampling = lambda shape: tf.random.normal(shape) * stdT
        print(f"[fix-init] chains start at std {stdT:.3f}")

    # off-Omega x is known exactly: y_add = x + n, n = 0 there
    mask_the_sampler(denoiser.sampler, M3, y_add, project=a.project)

    import tensorflow as tf
    out = denoiser(noisy_samples=tf.convert_to_tensor(y, tf.float32),
                   target_samples=tf.convert_to_tensor(x_A_gt, tf.float32),
                   plot=False)
    x_A_sgm, x_B_sgm = out
    if a.final_composite:
        x_A_sgm = M3 * np.asarray(x_A_sgm, np.float32) + (1.0 - M3) * y_add
        x_B_sgm = M3 * np.asarray(x_B_sgm, np.float32)
    d = OUTPUT_ROOT / "sgm" / a.bench / a.arm
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"c{a.i0}.npz",
                        x_A=np.asarray(x_A_sgm, np.float32),
                        x_B=np.asarray(x_B_sgm, np.float32))
    print(f"[{a.arm} {a.bench} c{a.i0}] saved")


if __name__ == "__main__":
    main()
