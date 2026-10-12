"""EXTBENCH — DPS's pretrained FFHQ-256 guided-diffusion DM behind the
TANGO adapter API, serving the 64^2 loop through a resolution bridge.

GDAdapter (64^2 cosine ADM) subclass with:
  * model  = DPS ffhq_10m.pt config (256^2, num_res_blocks=1, attention 16,
    num_head_channels=64, resblock_updown, learn_sigma) built from the DPS
    repo's own guided_diffusion.unet.create_model;
  * schedule = LINEAR betas (1e-4 .. 0.02, 1000 steps), the schedule the
    checkpoint was trained with (diffusion_config.yaml: noise_schedule
    linear) — the cosine table would misindex t;
  * x0_hat bridge = nearest x4 upsample of the VE state (preserves the
    per-pixel noise marginal; noise becomes 4x4-block-correlated, an
    accepted approximation), VE<->VP bridge at 256, UNet eps, x0 at 256,
    bilinear (antialias) downsample back to 64.

FLAG: the model sees block-correlated noise off its training stats; the
bridge is an approximation, so this arm LOWER-BOUNDS what a native-256
matched prior could do inside the loop.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

from experiments.extbench.gd_adapter import GDAdapter, _FakeSimpleSDE

# parent directory of the guided_diffusion package (vendored copy of the DPS / OpenAI UNet files, identical)
DPS_REPO = Path(os.environ.get("SORD_GUIDED_DIFFUSION_SRC", Path(__file__).resolve().parents[2] / "third_party"))


def _linear_alpha_bar(steps=1000, b0=1e-4, b1=2e-2):
    betas = np.linspace(b0, b1, steps, dtype=np.float64)
    return np.cumprod(1.0 - betas)


class GD256Adapter(GDAdapter):
    # DPS model_config_ffhq.yaml (ffhq_10m.pt); the ImageNet subclass
    # swaps these for imagenet_model_config.yaml
    MODEL_ARCH = dict(num_channels=128, num_res_blocks=1,
                      attention_resolutions=16)

    def __init__(self, ckpt_path, num_scales=100, sampling_eps=1e-3,
                 sigma=25.0, diffusion_steps=1000):
        import torch
        sys.path.insert(0, str(DPS_REPO))
        from guided_diffusion.unet import create_model as dps_create_model

        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available()
                                else "cpu")
        self.net = dps_create_model(
            image_size=256, **self.MODEL_ARCH,
            channel_mult="", learn_sigma=True, class_cond=False,
            use_checkpoint=False, num_heads=4,
            num_head_channels=64, num_heads_upsample=-1,
            use_scale_shift_norm=True, dropout=0.0, resblock_updown=True,
            use_fp16=False, use_new_attention_order=False,
            model_path=str(ckpt_path)).to(self.dev).eval()

        self._abar = _linear_alpha_bar(diffusion_steps)
        self._sigma = float(sigma)
        self._sde = _FakeSimpleSDE(sigma)
        self._ts = list(np.linspace(1.0, sampling_eps,
                                    num_scales + 1)[:-1].astype(np.float64))
        # residual-noise factor of the bilinear x4 upsample (MC, once):
        # white noise at 64 keeps only r of its std after upsampling, so
        # the bridge tops it back up with sqrt(1-r^2) white noise at 256
        # to restore the model's training noise statistics.
        g = torch.Generator().manual_seed(0)
        z = torch.randn(16, 3, 64, 64, generator=g)
        zu = torch.nn.functional.interpolate(
            z, scale_factor=4, mode="bilinear", align_corners=False)
        self._up_r = float(zu.std())

    def x0_hat(self, x_t, t):
        import tensorflow as tf
        torch = self.torch
        F = torch.nn.functional
        x = np.asarray(x_t, np.float32)
        sig = 2.0 * self._std_np(float(t))               # [-1,1] coords
        abar = 1.0 / (1.0 + sig * sig)
        j = int(np.abs(self._abar - abar).argmin())
        abar_j = float(self._abar[j])
        with torch.no_grad():
            xp = torch.from_numpy((2.0 * x - 1.0).transpose(0, 3, 1, 2)) \
                .to(self.dev)
            xp = F.interpolate(xp, scale_factor=4, mode="bilinear",
                               align_corners=False)
            top_up = sig * math.sqrt(max(1.0 - self._up_r ** 2, 0.0))
            if top_up > 0:
                g = torch.Generator(device="cpu").manual_seed(
                    10007 * j + x.shape[0])
                xp = xp + top_up * torch.randn(
                    xp.shape, generator=g).to(self.dev)
            x_vp = math.sqrt(abar_j) * xp
            # sub-batch the forward: >=161*128ch*256^2 crosses the int32
            # element limit of upsample_nearest_nhwc inside the decoder
            SB = int(os.environ.get("GD256_SB", "128"))
            eps = torch.cat([
                self.net(x_vp[i:i + SB].float(),
                         torch.full((x_vp[i:i + SB].shape[0],), j,
                                    dtype=torch.long, device=self.dev))[:, :3]
                for i in range(0, x_vp.shape[0], SB)])
            x0p = (x_vp - math.sqrt(1.0 - abar_j) * eps) / math.sqrt(abar_j)
            x0p = F.interpolate(x0p.clamp(-1.0, 1.0), size=(64, 64),
                                mode="bilinear", antialias=True,
                                align_corners=False)
            x0 = ((x0p + 1.0) / 2.0).cpu().numpy().transpose(0, 2, 3, 1)
        return tf.constant(x0, tf.float32)


class GD256NativeAdapter(GD256Adapter):
    """ffhq_10m serving a NATIVE 256^2 loop state — no resolution bridge.

    Identical VE ladder and VE<->VP mapping as the parent, but the loop's
    x_t is already 256^2, so the upsample / white-noise top-up / downsample
    all drop out: the model sees exactly its training statistics (the
    64^2-arm FLAG does not apply). Forward sub-batched at SB (class attr:
    128 covers the int32 upsample_nearest_nhwc limit for ffhq_10m; the
    552M-param ImageNet subclass needs SB=8 — a K=32 batch through it in
    one forward allocates >47 GB and OOMs an A6000). GD256_SB env var
    overrides (b7-era shared-GPU runs: SB=8 fits beside DDP training)."""

    SB = int(os.environ.get("GD256_SB", "128"))

    def x0_hat(self, x_t, t):
        import tensorflow as tf
        torch = self.torch
        x = np.asarray(x_t, np.float32)
        sig = 2.0 * self._std_np(float(t))               # [-1,1] coords
        abar = 1.0 / (1.0 + sig * sig)
        j = int(np.abs(self._abar - abar).argmin())
        abar_j = float(self._abar[j])
        with torch.no_grad():
            xp = torch.from_numpy((2.0 * x - 1.0).transpose(0, 3, 1, 2)) \
                .to(self.dev)
            x_vp = math.sqrt(abar_j) * xp
            SB = self.SB
            eps = torch.cat([
                self.net(x_vp[i:i + SB].float(),
                         torch.full((x_vp[i:i + SB].shape[0],), j,
                                    dtype=torch.long, device=self.dev))[:, :3]
                for i in range(0, x_vp.shape[0], SB)])
            x0p = (x_vp - math.sqrt(1.0 - abar_j) * eps) / math.sqrt(abar_j)
            x0 = ((x0p.clamp(-1.0, 1.0) + 1.0) / 2.0) \
                .cpu().numpy().transpose(0, 2, 3, 1)
        return tf.constant(x0, tf.float32)


    def grad_x0(self, x_t, t, mode, y=None, known=None, zeta=1.0, target=None, w_in=1.0, chunk=None):
        """DPS ablation path for the NATIVE 256^2 loop (2026-09-20; mirrors EDM2Adapter.grad_x0 / CelebAAdapter):
        x0_hat AND the gradient of a data term through the guided-diffusion UNet via torch autograd.
        mode="inpaint": loss = zeta * sum_b ||known_b * (x0 - y_b)||_2  (un-squared per-image norm, DPS-inpaint convention)
        mode="gauss"  : loss = w_in * sum ||x0 - target||^2               (DPS-Gauss on the read; n-step)
        Returns (x0 clamped, grad wrt x_t, residual) as numpy NHWC in [0,1] coords; loss on the UNCLAMPED x0 so
        saturated pixels keep a gradient. Chunked at SB over the batch (K*B chains) for memory."""
        torch = self.torch
        x = np.asarray(x_t, np.float32); N = x.shape[0]
        sig = 2.0 * self._std_np(float(t))
        abar = 1.0 / (1.0 + sig * sig)
        j = int(np.abs(self._abar - abar).argmin()); abar_j = float(self._abar[j])
        SB = int(chunk) if chunk else self.SB
        ref = np.asarray(y if mode == "inpaint" else target, np.float32)
        x0_out = np.empty_like(x); g_out = np.empty_like(x); r_out = np.empty_like(x)
        for j0 in range(0, N, SB):
            j1 = min(j0 + SB, N)
            xp = torch.from_numpy((2.0 * x[j0:j1] - 1.0).transpose(0, 3, 1, 2)).to(self.dev).requires_grad_(True)
            with torch.enable_grad():
                x_vp = math.sqrt(abar_j) * xp
                eps = self.net(x_vp.float(), torch.full((j1 - j0,), j, dtype=torch.long, device=self.dev))[:, :3]
                x0p = (x_vp - math.sqrt(1.0 - abar_j) * eps) / math.sqrt(abar_j)
                x0u = (x0p + 1.0) / 2.0                                   # unclamped, [0,1] coords, NCHW
                rt = torch.from_numpy(ref[j0:j1].transpose(0, 3, 1, 2)).to(self.dev)
                r = x0u - rt
                if mode == "inpaint":
                    kt = torch.from_numpy(np.asarray(known[j0:j1], np.float32).transpose(0, 3, 1, 2)).to(self.dev)
                    r = kt * r
                    loss = zeta * torch.norm(r.reshape(j1 - j0, -1), dim=1).sum()
                else:
                    loss = float(w_in) * (r ** 2).sum()
                g, = torch.autograd.grad(loss, xp)
            g_out[j0:j1] = (2.0 * g).detach().cpu().numpy().transpose(0, 2, 3, 1)   # x=(xp+1)/2 -> grad_x = 2 grad_xp
            x0_out[j0:j1] = ((x0p.detach().clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().numpy().transpose(0, 2, 3, 1)
            r_out[j0:j1] = r.detach().cpu().numpy().transpose(0, 2, 3, 1)
            del xp, x_vp, eps, x0p, x0u, r, loss, g
        return x0_out, g_out, r_out


class GD256NativeAdapterIN(GD256NativeAdapter):
    """guided-diffusion 256x256_diffusion_uncond.pt (DPS's ImageNet
    prior; imagenet_model_config.yaml arch) serving the native-256 loop.
    Same linear-beta schedule (the ImageNet uncond ckpt is also
    linear-schedule ADM)."""

    MODEL_ARCH = dict(num_channels=256, num_res_blocks=2,
                      attention_resolutions="32,16,8")
    SB = 8
