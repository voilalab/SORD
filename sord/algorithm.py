"""SORD: Support-guided OOD Removal with Diffusion -- Algorithm 1.

    y = x + n,   n = 0 outside the known corruption support Omega,

restored with ONE clean-signal diffusion prior (no model of the corruption).  Images may be processed in
batches to fill the GPU, but each image keeps its own samples, its own uncertainty sigma* and its own
stopping decision: its result is the same whichever images share its batch.

    K samples of diffusion inpainting  x_0^(k) ~ R_theta(y, Omega);  sigma*_1 = spread of x_0
    for m = 1..M:
        n_hat = D_theta(read of the corruption + sigma*_m eps, sigma*_m)          (sord.steps.n_step)
        x_m^(k) = D_theta(y - n_hat on Omega, y elsewhere; re-noised to sigma_m)  (sord.steps.x_step)
        sigma*_{m+1} = RMS over Omega of the spread of the K samples x_m^(k)
        if sigma*_m - sigma*_{m+1} <= tau_stop * sigma*_m:  return x_{m-1}
    return x_M

Every estimate is the mean of its K samples.  The stopping test discards a refinement that did not
reduce the uncertainty by a relative tau_stop, so the inpainting x_0 is an admissible output (a large
tau_stop returns it).  For the multiplicative shadow model y = x * h the corruption step reads
the log-gain instead (sord.steps.h_step); everything else is unchanged.  Audio (modality "audio") uses
the same loop with waveform versions of the steps (sord.steps_audio).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import steps, steps_audio


@dataclass
class SORDConfig:
    K: int = 32                          # diffusion samples
    M: int = 6                           # maximum refinement iterations
    sigma_start: float = 0.4             # re-noising schedule sigma_m: linear from sigma_start ...
    sigma_end: float = 0.1               # ... to sigma_end over the M iterations (data units, images in [0, 1])
    tau_stop: float = 0.01               # relative-decrease stopping threshold (configs/sord/tau_stop.yaml)
    model: str = "additive"              # "additive" (y = x + n) or "multiplicative" (y = x * h, shadows)
    corruption: str = "free"             # additive class: "pos" (n >= 0) or "free" (signed)
    zero_anchor: bool = False            # residual normalization for a non-negative contaminant on black
    modality: str = "image"              # "image" (sord.steps) or "audio" (sord.steps_audio)

    @property
    def sigmas(self):
        return np.linspace(self.sigma_start, self.sigma_end, self.M)


@dataclass
class SORDResult:
    x: np.ndarray                                    # returned reconstruction (H, W, C)
    returned: int                                    # index m of the returned iterate x_m (0 = inpainting)
    stopped_early: bool                              # True if line 6 fired
    sigma_star: list = field(default_factory=list)   # sigma*_1, sigma*_2, ... (spread of x_0, x_1, ...)
    iterates: list = field(default_factory=list)     # x_0, x_1, ... (each the mean of K samples)


def sord_batch(Y, OMEGA, prior, cfg: SORDConfig, seeds):
    """Run Algorithm 1 on B observations Y (B, H, W, C) in [0, 1] with supports OMEGA (B, H, W).

    seeds[b] seeds image b's own random generator.  Returns one SORDResult per image.
    For audio, Y is (B, L, 1) and OMEGA (B, L)."""
    S = steps_audio if cfg.modality == "audio" else steps
    Y = np.asarray(Y, np.float64)
    OMEGA = np.asarray(OMEGA, bool)
    B = len(Y)
    rngs = [np.random.default_rng(s) for s in seeds]
    results = [None] * B

    samples = S.inpaint(prior, Y, OMEGA, cfg.K, rngs)                      # inpainting x_0
    iterates = [[np.where(OMEGA[b][..., None], samples[b].mean(0), Y[b])] for b in range(B)]
    sigma = [[s] for s in S.sigma_star(samples, OMEGA)]                    # sigma*_1 per image
    active = list(range(B))                                                     # images still iterating

    for m in range(1, cfg.M + 1):
        a = np.array(active)
        y, omega, r = Y[a], OMEGA[a], [rngs[b] for b in active]
        x_hat = np.stack([iterates[b][m - 1] for b in active])
        s_m = np.array([sigma[b][m - 1] for b in active])                       # sigma*_m per image
        if cfg.model == "multiplicative":                                       # corruption estimate
            n_hat = S.h_step(prior, y, x_hat, omega, s_m, r)
        else:
            n_hat = S.n_step(prior, y, x_hat, omega, s_m, r, cfg.corruption, cfg.zero_anchor)
        samples = S.x_step(prior, y, n_hat, omega, cfg.sigmas[m - 1], cfg.K, r)   # signal estimate x_m
        for k, (b, s_next) in enumerate(zip(list(active), S.sigma_star(samples, omega))):
            iterates[b].append(samples[k].mean(0))
            sigma[b].append(float(s_next))                                      # sigma*_{m+1}
            if sigma[b][m - 1] - sigma[b][m] <= cfg.tau_stop * sigma[b][m - 1]:   # stop: keep x_{m-1}
                results[b] = SORDResult(iterates[b][m - 1], m - 1, True, sigma[b], iterates[b])
                active.remove(b)
        if not active:
            break

    for b in active:                                                            # no stop: x_M
        results[b] = SORDResult(iterates[b][cfg.M], cfg.M, False, sigma[b], iterates[b])
    return results


def sord(y, omega, prior, cfg: SORDConfig, seed: int = 0) -> SORDResult:
    """Run Algorithm 1 on one observation y (H, W, C) in [0, 1] with support omega (H, W)."""
    return sord_batch(np.asarray(y)[None], np.asarray(omega)[None], prior, cfg, [seed])[0]
