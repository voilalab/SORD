"""EXTBENCH — EDM2 texture prior (torch pickle) behind the TANGO adapter API.

Same six-method contract as gd_adapter.GDAdapter (timesteps, _std, x0_hat,
ddim_step, init_state, forward_noise), same CelebAAdapter VE ladder verbatim
(simple-SDE std, sigma=25, t in (eps, 1]). Unlike GD there is NO VP bridge and
NO timestep quantization: EDM2's Precond IS a continuous-sigma denoiser
D(x, sigma) = E[x0 | x_sigma] in [-1,1] coords, so the bridge is just the
range/scale change:

  [0,1] VE state:   x_t = x0 + sigma z
  EDM2 [-1,1]:      x' = 2 x_t - 1,  sigma' = 2 sigma
  x0 = (clamp(D(x', sigma'), -1, 1) + 1) / 2

Class conditioning: the texture fine-tune trains with a fixed zero label
vector (unconditional path); we pass zeros(label_dim) at inference.
The model is fully convolutional (trained at 256^2, pretrained at 64^2);
the 64^2 bench resolution is its pretraining-native input size.

Checkpoint source: experiments/edm2_texture_prior_2026_08_04 campaign,
CHECKPOINT_ROOT/method/ (checkpoint_manifest.json). Pickles are
pure EDM2 persistence objects (fast_attn stripped) and load under torch 1.12
(verified 2026-08-04).
"""
from __future__ import annotations

import math
import pickle

import numpy as np

import os
from pathlib import Path

# dnnlib / torch_utils needed to unpickle EDM2 networks: the vendored copy (identical files)
EDM2_SRC = os.environ.get("SORD_EDM2_SRC", str(Path(__file__).resolve().parents[2] / "third_party" / "edm2"))


class _FakeSimpleSDE:
    """simple-SDE facade for SigmaGrid — identical to gd_adapter's."""
    T = 1.0

    def __init__(self, sigma):
        self._s = float(sigma)

    def marginal_prob(self, x, t):
        import tensorflow as tf
        t = tf.cast(t, tf.float32)
        s = self._s
        std = tf.sqrt((s ** (2.0 * t) - 1.0) / (2.0 * math.log(s)))
        return x, std


class EDM2Adapter:
    def __init__(self, pkl_path, num_scales=100, sampling_eps=1e-3,
                 sigma=25.0):
        import sys as _sys
        if EDM2_SRC not in _sys.path:
            _sys.path.insert(0, EDM2_SRC)
        import torch

        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else
                                "cpu")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.net = data["ema"].to(torch.float32).eval() \
            .requires_grad_(False).to(self.dev)
        self._lab = torch.zeros(1, self.net.label_dim, device=self.dev)

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
        sig = 2.0 * self._std_np(float(t))               # [-1,1] coords
        # EDM2_SB (2026-09-19): optional sub-batching of the forward pass on shared GPUs (K*B chains at once peaks
        # >13 GB at K=32,B=25).  Unset / 0 = whole batch in one call, bit-identical to before.
        import os
        SB = int(os.environ.get("EDM2_SB", "0")) or x.shape[0]
        outs = []
        with torch.no_grad():
            for j0 in range(0, x.shape[0], SB):
                xj = x[j0:j0 + SB]
                xp = torch.from_numpy((2.0 * xj - 1.0).transpose(0, 3, 1, 2)) \
                    .to(self.dev)
                st = torch.full((xj.shape[0],), sig, device=self.dev)
                lab = self._lab.expand(xj.shape[0], -1)
                D = self.net(xp, st, lab)
                outs.append(((D.clamp(-1.0, 1.0) + 1.0) / 2.0)
                            .cpu().numpy().transpose(0, 2, 3, 1))
        x0 = np.concatenate(outs, 0)
        return tf.constant(x0, tf.float32)

    # ---- pure-sigma helpers (copied from CelebAAdapter) ---------------
    def grad_x0(self, x_t, t, mode, y=None, known=None, zeta=1.0, target=None, w_in=1.0, chunk=16):
        """DPS ablation (2026-09-13): x0_hat AND the gradient of a data term through the EDM2 net (torch autograd).
        mode="inpaint": loss = zeta * sum_b ||known_b * (x0 - y_b)||_2   (un-squared per-image norm, DPS-inpaint convention)
        mode="gauss"  : loss = w_in * sum ||x0 - target||^2                  (DPS-Gauss on the read)
        Returns (x0 clamped, grad wrt x_t, residual) as numpy NHWC in [0,1] coords; the loss is taken on the unclamped x0 so
        saturated pixels keep a gradient. Chunked over the batch for memory."""
        torch = self.torch
        x = np.asarray(x_t, np.float32); N = x.shape[0]
        sig = 2.0 * self._std_np(float(t))
        x0_out = np.empty_like(x); g_out = np.empty_like(x); r_out = np.empty_like(x)
        ref = np.asarray(y if mode == "inpaint" else target, np.float32)
        for j0 in range(0, N, chunk):
            j1 = min(j0 + chunk, N)
            xp = torch.from_numpy((2.0 * x[j0:j1] - 1.0).transpose(0, 3, 1, 2)).to(self.dev).requires_grad_(True)
            st = torch.full((j1 - j0,), sig, device=self.dev); lab = self._lab.expand(j1 - j0, -1)
            with torch.enable_grad():
                D = self.net(xp, st, lab)
                x0u = (D + 1.0) / 2.0                                     # unclamped, [0,1] coords, NCHW
                rt = torch.from_numpy(ref[j0:j1].transpose(0, 3, 1, 2)).to(self.dev)
                r = x0u - rt
                if mode == "inpaint":
                    kt = torch.from_numpy(np.asarray(known[j0:j1], np.float32).transpose(0, 3, 1, 2)).to(self.dev)
                    r = kt * r
                    loss = zeta * torch.norm(r.reshape(j1 - j0, -1), dim=1).sum()
                else:
                    loss = float(w_in) * (r ** 2).sum()
                g, = torch.autograd.grad(loss, xp)
            # d/dx in [0,1] coords: x = (xp+1)/2 -> dx = dxp/2 -> grad_x = 2 * grad_xp
            g_out[j0:j1] = (2.0 * g).detach().cpu().numpy().transpose(0, 2, 3, 1)
            x0_out[j0:j1] = ((D.detach().clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().numpy().transpose(0, 2, 3, 1)
            r_out[j0:j1] = r.detach().cpu().numpy().transpose(0, 2, 3, 1)
        return x0_out, g_out, r_out

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
