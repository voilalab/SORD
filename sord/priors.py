"""Clean-signal diffusion priors behind the two things SORD needs:

    prior.sigmas         noise ladder, descending: 120 levels of the variance-exploding SDE with sigma = 25,
                         std(t) = sqrt((25^(2t) - 1) / (2 ln 25)), t = linspace(1, 1e-3, 121)[:-1]
                         (9.845 at the top, 0.098 at the floor; data units, images in [0, 1])
    prior.denoise(x, i)  E[x0 | x] for x = x0 + sigmas[i] * z,  x of shape (B, H, W, C)

Every checkpoint used in the paper is served through the same adapter classes as the reported runs, so
the network calls (preconditioning, VE <-> VP mapping for ADM, clamping) are exactly those of the paper:

    celeba           CelebA score model of Stevens et al. (TensorFlow), 64^2 -- CelebA benches
    edm2             EDM2 network pickle (PyTorch), 64^2 -- CLWD (fine-tuned) and ISTD scene priors
    adm_ffhq256      guided-diffusion ffhq_10m (PyTorch), 256^2 -- FFHQ watermark / shadow
    adm_imagenet256  guided-diffusion 256x256_diffusion_uncond (PyTorch), 256^2 -- Imagenette benches
    msdm             single-source audio diffusion checkpoint of Mariani et al. (MSDM), 22.05 kHz waveforms;
                     EDM-parameterized, so it is queried at any sigma: AudioPrior.x0(x, sigma)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from . import paths

REPO = paths.PROJECT_ROOT
NUM_LEVELS = 120


class Prior:
    """Wraps one adapter: a fixed ladder plus a Tweedie denoiser at each ladder level."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.timesteps = list(adapter.timesteps)
        self.sigmas = np.array([float(adapter._std(t)) for t in self.timesteps])

    def denoise(self, x, i):
        import tensorflow as tf
        x0 = self.adapter.x0_hat(tf.constant(np.asarray(x, np.float32)), self.timesteps[i])
        return np.asarray(x0, np.float64)


class CelebAScorePrior:
    """Score model of Stevens et al. (TensorFlow): E[x0 | x] = x + std(t)^2 * score(x, t), on its own
    simple VE SDE, t = linspace(T, 1e-3, NUM_LEVELS + 1)[:-1]."""

    def __init__(self, run_dir):
        os.chdir(REPO)                                   # the TF loader resolves config paths relative to the repo
        sys.path.insert(0, str(REPO))
        import tensorflow as tf
        from generators.models import get_model
        from utils.checkpoints import ModelCheckpoint
        from utils.runs import init_config
        config = init_config(str(run_dir), verbose=False)
        self.net = get_model(config, training=False)
        ModelCheckpoint(self.net, config=config).restore()
        self.tf, self.sde = tf, self.net.sde
        self.timesteps = list(np.linspace(self.sde.T, 1e-3, NUM_LEVELS + 1)[:-1].astype(np.float64))
        self.sigmas = np.array([float(self._std([t])[0]) for t in self.timesteps])

    def _std(self, t):
        tf = self.tf
        _, std = self.sde.marginal_prob(tf.zeros([len(t), 1, 1, 1], tf.float32), tf.constant(t, tf.float32))
        return tf.cast(std, tf.float32)

    def denoise(self, x, i):
        tf = self.tf
        x = tf.constant(np.asarray(x, np.float32))
        t = tf.fill([tf.shape(x)[0]], float(self.timesteps[i]))
        score = self.net.get_score(x, t, training=False)
        _, std = self.sde.marginal_prob(x, t)
        std = tf.cast(std, tf.float32)[:, None, None, None]
        return np.asarray(x + std ** 2 * score, np.float64)


class AudioPrior:
    """Waveform prior: x0(x, sigma) = E[x0 | x] for x = x0 + sigma * z, x of shape (N, L, 1)."""

    def __init__(self, ckpt, msdm_repo):
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, str(msdm_repo))               # the MSDM authors' code (third party, not vendored)
        from experiments.tango_msdm_2026_08_04.src.msdm_prior import MSDMPrior
        import torch
        self.torch = torch
        self.net = MSDMPrior(ckpt_path=ckpt, device="cuda" if torch.cuda.is_available() else "cpu")

    def x0(self, x, sigma):
        t = self.torch
        xt = t.from_numpy(np.ascontiguousarray(np.asarray(x, np.float32)[..., 0])[:, None]).to(self.net.device)
        return self.net.x0_hat(xt, float(sigma))[:, 0].cpu().numpy()[..., None].astype(np.float64)


def build_prior(spec: dict):
    """Build a prior from a config entry {kind: ..., checkpoint: <manifest id or path>}."""
    sys.path.insert(0, str(REPO))
    kind = spec["kind"]
    ckpt = paths.checkpoint(spec.get("checkpoint") or spec["path"])
    if kind == "celeba":
        return CelebAScorePrior(ckpt)
    if kind == "edm2":
        from experiments.extbench.edm2_adapter import EDM2Adapter
        return Prior(EDM2Adapter(str(ckpt), num_scales=NUM_LEVELS))
    if kind == "adm_ffhq256":
        from experiments.extbench.gd256_adapter import GD256NativeAdapter
        return Prior(GD256NativeAdapter(str(ckpt), num_scales=NUM_LEVELS))
    if kind == "adm_imagenet256":
        from experiments.extbench.gd256_adapter import GD256NativeAdapterIN
        return Prior(GD256NativeAdapterIN(str(ckpt), num_scales=NUM_LEVELS))
    if kind == "msdm":
        repo = paths.PROJECT_ROOT / spec.get("msdm_repo", "third_party/msdm")
        return AudioPrior(str(ckpt), repo if not os.path.isabs(spec.get("msdm_repo", "")) else spec["msdm_repo"])
    raise ValueError(f"unknown prior kind {kind!r}")
