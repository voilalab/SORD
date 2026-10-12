"""Adapter: baseline ISDM single-source checkpoint as THE one prior.

Wraps a weakly-MSDM (per-source) checkpoint from Mariani et al. —
ruby-dew-290 = piano — behind the small surface the tango loop needs:
Tweedie x0_hat and a deterministic multi-step denoise. The model is
EDM-parameterized (audio_diffusion_pytorch 0.0.43 Diffusion), so
denoise_fn(x, sigma) IS the x0 estimate; score = (x - x0)/sigma.
ONE-PRIOR RULE: only the target-source checkpoint may be loaded here;
interferers stay unmodeled corruption.

Model facts (train_weakly_msdm.yaml + ckpt hparams): 22050 Hz mono,
sigma_data 0.2, training sigmas ~ LogNormal(-3, 1) (~0.007..0.37 2sig),
UNet1d patch_factor 16 x factors 512 => length % 8192 == 0.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

MSDM_REPO = Path(__file__).resolve().parents[1] / "baselines" / "msdm"
SR_MODEL = 22050
LEN_MULTIPLE = 8192

def _default_piano_ckpt():
    from sord.paths import CHECKPOINT_ROOT
    return CHECKPOINT_ROOT / "diffusion/msdm/ruby-dew-290/epoch=236-valid_loss=0.010.ckpt"


PIANO_CKPT = Path(os.environ.get("SORD_MSDM_PIANO_CKPT", _default_piano_ckpt()))


def karras_sigmas(n, sigma_max, sigma_min=1e-4, rho=7.0, device="cpu"):
    """Karras et al. schedule from sigma_max down to sigma_min, n steps
    (n+1 values, last is 0 to land on x0), same form the baseline's
    KarrasSchedule uses."""
    i = torch.linspace(0, 1, n, device=device)
    sig = (sigma_max ** (1 / rho)
           + i * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([sig, torch.zeros(1, device=device)])


class MSDMPrior:
    def __init__(self, ckpt_path=PIANO_CKPT, device="cuda"):
        if str(MSDM_REPO) not in sys.path:
            sys.path.insert(0, str(MSDM_REPO))
        from main.module_base import Model
        try:
            self.model = Model.load_from_checkpoint(
                str(ckpt_path), map_location=device)
        except Exception:
            # plain-torch ckpt of train_speech_dm.py ({"state_dict",
            # "hyper_parameters"}): rebuild Model(**hp) and load weights
            from experiments.tango_msdm_2026_08_04.src.e1_speech import \
                load_speech_model
            self.model = load_speech_model(str(ckpt_path), device)
        self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.denoise_fn = self.model.model.diffusion.denoise_fn
        self.sigma_data = float(self.model.model.diffusion.sigma_data)
        self.nfe = 0                       # net evals, counted per batch row

    @torch.no_grad()
    def x0_hat(self, x, sigma):
        """One-step Tweedie/EDM denoise. x: (B,1,L) on device, L%8192==0."""
        self.nfe += int(x.shape[0])
        return self.denoise_fn(x, sigma=float(sigma))

    @torch.no_grad()
    def ddim(self, x_noisy, sigma_start, n_steps=8, sigma_min=1e-4, rho=7.0):
        """Deterministic Euler probability-flow descent from sigma_start,
        the baseline separator's update (separation.py separate_mixture)
        without churn and without conditioning: d = (x - x0)/sigma."""
        sigmas = karras_sigmas(n_steps, float(sigma_start), sigma_min, rho,
                               device=x_noisy.device)
        x = x_noisy
        for i in range(len(sigmas) - 1):
            sig, sig_next = float(sigmas[i]), float(sigmas[i + 1])
            d = (x - self.x0_hat(x, sig)) / sig
            x = x + d * (sig_next - sig)
        return x


def crop_to_multiple(w, mult=LEN_MULTIPLE):
    n = (w.shape[-1] // mult) * mult
    return w[..., :n]


def resample(w, sr_from, sr_to):
    import torchaudio.functional as AF
    return AF.resample(w, sr_from, sr_to)
