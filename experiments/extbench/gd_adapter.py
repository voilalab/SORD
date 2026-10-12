"""EXTBENCH — guided-diffusion (ADM/DDPM, torch) behind the TANGO adapter API.

The whole loop touches the prior through exactly six methods (verified by
grep over run_loop_test / DiffusionConditioner / map_solve / nstep):
timesteps, _std, x0_hat, ddim_step, init_state, forward_noise. This class
reproduces CelebAAdapter's VE ladder verbatim (simple-SDE std, sigma=25,
t in (eps, 1]) and answers x0_hat through a VP cosine DDPM via the
standard VE<->VP bridge:

  [0,1] VE state:  x_t = x0 + sigma z
  GD [-1,1] coords: x' = 2 x_t - 1, sigma' = 2 sigma
  alpha_bar = 1 / (1 + sigma'^2);  x_vp = sqrt(alpha_bar) x'
  eps_hat = UNet(x_vp, j(alpha_bar));  x0' = (x_vp - sqrt(1-a) eps)/sqrt(a)
  x0 = (x0' + 1) / 2      (learn_sigma: first 3 channels are eps)

Inputs/outputs are TF tensors (numpy accepted) so it is a drop-in for the
TF adapter; torch runs on the same visible GPU.
"""
from __future__ import annotations

import math

import numpy as np


def _cosine_alpha_bar(steps=1000, s=0.008):
    f = [math.cos((i / steps + s) / (1 + s) * math.pi / 2) ** 2
         for i in range(steps + 1)]
    return np.array([f[i + 1] / f[0] for i in range(steps)], np.float64)


class _FakeSimpleSDE:
    """simple-SDE facade for SigmaGrid (tango_upgrades/levels.py): only
    .T and .marginal_prob(x, t) -> (_, std) are consumed. Must match
    GDAdapter._std exactly."""
    T = 1.0

    def __init__(self, sigma):
        self._s = float(sigma)

    def marginal_prob(self, x, t):
        import tensorflow as tf
        t = tf.cast(t, tf.float32)
        s = self._s
        std = tf.sqrt((s ** (2.0 * t) - 1.0) / (2.0 * math.log(s)))
        return x, std


class GDAdapter:
    def __init__(self, ckpt_path, num_scales=100, sampling_eps=1e-3,
                 sigma=25.0, num_channels=128, num_res_blocks=3,
                 diffusion_steps=1000):
        import torch
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parent / "gd"))
        from guided_diffusion.script_util import create_model

        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else
                                "cpu")
        self.net = create_model(
            image_size=64, num_channels=num_channels,
            num_res_blocks=num_res_blocks, learn_sigma=True,
            channel_mult="", use_checkpoint=False,
            attention_resolutions="16,8", num_heads=4, num_head_channels=-1,
            num_heads_upsample=-1, use_scale_shift_norm=True, dropout=0.0,
            resblock_updown=False, use_fp16=False, use_new_attention_order=False,
            class_cond=False)
        sd = torch.load(ckpt_path, map_location="cpu")
        self.net.load_state_dict(sd)
        self.net.to(self.dev).eval()

        self._abar = _cosine_alpha_bar(diffusion_steps)
        self._sigma = float(sigma)
        self._sde = _FakeSimpleSDE(sigma)
        # CelebAAdapter ladder, verbatim
        self._ts = list(np.linspace(1.0, sampling_eps,
                                    num_scales + 1)[:-1].astype(np.float64))

    # ---- ladder ------------------------------------------------------
    @property
    def timesteps(self):
        return self._ts

    def _std_np(self, t):
        s = self._sigma
        return math.sqrt((s ** (2 * t) - 1.0) / (2.0 * math.log(s)))

    def _std(self, t):
        import tensorflow as tf
        return tf.constant(self._std_np(float(t)), tf.float32)

    # ---- the bridge ---------------------------------------------------
    def x0_hat(self, x_t, t):
        import tensorflow as tf
        torch = self.torch
        x = np.asarray(x_t, np.float32)
        sig = 2.0 * self._std_np(float(t))              # [-1,1] coords
        abar = 1.0 / (1.0 + sig * sig)
        j = int(np.abs(self._abar - abar).argmin())
        abar_j = float(self._abar[j])                    # use the model's own
        with torch.no_grad():
            xp = torch.from_numpy((2.0 * x - 1.0).transpose(0, 3, 1, 2)) \
                .to(self.dev)
            x_vp = math.sqrt(abar_j) * xp   # exact iff abar_j = 1/(1+sig^2);
            # nearest-j quantization error is <1e-3 at 1000 cosine steps
            tt = torch.full((x.shape[0],), j, dtype=torch.long,
                            device=self.dev)
            out = self.net(x_vp.float(), tt)
            eps = out[:, :3]
            x0p = (x_vp - math.sqrt(1.0 - abar_j) * eps) / math.sqrt(abar_j)
            x0 = ((x0p.clamp(-1.0, 1.0) + 1.0) / 2.0) \
                .cpu().numpy().transpose(0, 2, 3, 1)
        return tf.constant(x0, tf.float32)

    # ---- pure-sigma helpers (copied from CelebAAdapter) ---------------
    def ddim_step(self, x_t, x0_guided, t, t_prev):
        if t_prev is None:
            return x0_guided
        std_t, std_prev = self._std_np(float(t)), self._std_np(float(t_prev))
        eps = (np.asarray(x_t, np.float32)
               - np.asarray(x0_guided, np.float32)) / (std_t + 1e-8)
        import tensorflow as tf
        return tf.constant(np.asarray(x0_guided, np.float32)
                           + std_prev * eps, tf.float32)

    def init_state(self, shape, seed=None):
        import tensorflow as tf
        g = tf.random.Generator.from_seed(seed if seed is not None else 0)
        return g.normal(shape, dtype=tf.float32) \
            * self._std_np(self._ts[0])

    def forward_noise(self, x0, t, seed=None):
        import tensorflow as tf
        g = tf.random.Generator.from_seed(seed if seed is not None else 0)
        x0 = tf.cast(tf.convert_to_tensor(x0), tf.float32)
        return x0 + self._std(t) * g.normal(tf.shape(x0), dtype=tf.float32)
