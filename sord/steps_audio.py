"""The building blocks of SORD (Algorithm 1) for audio: same functions and signatures as sord/steps.py.

Shapes: a batch of B waveforms y (B, L, 1) in [-1, 1] (22.05 kHz, mono), their supports omega (B, L) --
the interference window -- and K samples per waveform stacked as (B, K, L, 1).  The trailing axis of
length 1 plays the role of the image channel axis, so sord/algorithm.py runs unchanged.

The audio prior (sord/priors.AudioPrior) is EDM-parameterized: prior.x0(x, sigma) is the denoised estimate
at any noise level sigma, and sampling follows a Karras schedule from a chosen starting level down to 0
with deterministic DDIM steps (identical to the probability-flow Euler step for this parameterization):

    x_{t+1} = x0 + (sigma_{t+1} / sigma_t) * (x_t - x0),        x0 = prior.x0(x_t, sigma_t).

The checkpoints return silence from pure noise, so the inpainting samples start from the observation
re-noised to INIT_SIGMA rather than from the top of a noise ladder.
"""
from __future__ import annotations

import numpy as np

SIGMA_DATA = 0.2                  # the audio checkpoints' data scale
INIT_SIGMA = 1.6 * SIGMA_DATA     # inpainting starts from y re-noised to this level
SOLVE_STEPS = 16                  # Karras steps for inpainting and for the x-step
NSTEP_LEVELS = 8                  # Karras steps for the residual
NSTEP_FLOOR = 0.1 * SIGMA_DATA    # lowest refinement scale for the residual
SIGMA_MIN = 1e-4
LENGTH_MULTIPLE = 8192            # the 1-D UNet needs lengths that are a multiple of this


def noise(rngs, shape):
    return np.stack([r.standard_normal(shape) for r in rngs])


def karras(n, sigma_max, sigma_min=SIGMA_MIN, rho=7.0):
    """Karras et al. schedule: n levels from sigma_max to sigma_min, then 0."""
    i = np.linspace(0, 1, n)
    s = (sigma_max ** (1 / rho) + i * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return np.append(s, 0.0)


def descend(prior, x, sigma_top, steps, known=None, y=None):
    """DDIM from sigma_top to 0; with `known`, the estimate is replaced by y there at every level."""
    s = karras(steps, sigma_top)
    for t in range(len(s) - 1):
        x0 = prior.x0(x.reshape((-1,) + x.shape[-2:]), s[t]).reshape(x.shape)   # all samples in one call
        if known is not None:
            x0 = np.where(known, y, x0)
        x = x0 + (s[t + 1] / s[t]) * (x - x0) if s[t + 1] > 0 else x0
    return x


def window(omega):
    """First and last+1 index of the (contiguous) interference window."""
    idx = np.where(omega)[0]
    return idx.min(), idx.max() + 1


def sigma_star(samples, omega):
    """Per waveform: RMS over the window of the per-sample spread of the K samples (ddof=1)."""
    var = samples.var(axis=1, ddof=1).mean(axis=-1)
    return np.array([np.sqrt(v[o].mean()) for v, o in zip(var, omega)])


def inpaint(prior, y, omega, K, rngs):
    """R_theta(y, Omega): K samples per waveform, conditioned on y outside the window.

    Each sample starts from y re-noised to INIT_SIGMA and descends SOLVE_STEPS Karras levels, with the
    estimate replaced by y outside the window at every level; samples are clipped to [-1, 1]."""
    known = ~omega[:, None, :, None]
    yk = y[:, None]
    x = yk + INIT_SIGMA * noise(rngs, (K,) + y.shape[1:])
    x = descend(prior, x, INIT_SIGMA, SOLVE_STEPS, known, yk)
    return np.clip(np.where(known, yk, x), -1.0, 1.0)


def n_step(prior, y, x_hat, omega, sigma, rngs, kind="free", zero_anchor=False):
    """Lines 10-11: read the interferer off the current estimate on the window and denoise it with the
    clean prior.  Waveform corruptions are signed, so there is no class box (kind / zero_anchor unused).

    The window is cropped, zero-padded to a multiple of LENGTH_MULTIPLE and scaled to the prior's data
    scale (std -> SIGMA_DATA), re-noised to max(sigma_b, NSTEP_FLOOR) and denoised with NSTEP_LEVELS Karras
    steps; the result is scaled back and pasted on the window."""
    out = np.zeros_like(y)
    for b in range(len(y)):
        read = (y[b] - x_hat[b]) * omega[b][:, None]
        w0, w1 = window(omega[b])
        crop = read[w0:w1]
        padded = np.pad(crop, ((0, (-len(crop)) % LENGTH_MULTIPLE), (0, 0)))
        scale = max(float(padded.std()), 1e-8)
        s = max(float(sigma[b]), NSTEP_FLOOR)
        x = padded / scale * SIGMA_DATA + s * rngs[b].standard_normal(padded.shape)
        den = descend(prior, x[None], s, NSTEP_LEVELS)[0]
        out[b, w0:w1] = (den / SIGMA_DATA * scale)[:len(crop)]
    return out * omega[..., None]


def h_step(*args, **kwargs):
    raise NotImplementedError("the multiplicative model is defined for images only")


def x_step(prior, y, n_hat, omega, sigma_m, K, rngs):
    """Lines 12-13: subtract the interferer estimate on the window and pull the result back to the
    clean-source distribution: re-noise to sigma_m, SOLVE_STEPS Karras levels, y outside the window at
    every level, clip to [-1, 1].  Returns (B, K, L, 1)."""
    known = ~omega[:, None, :, None]
    yk = y[:, None]
    evidence = np.where(omega[..., None], y - n_hat, y)[:, None]
    x = evidence + sigma_m * noise(rngs, (K,) + y.shape[1:])
    x = descend(prior, x, sigma_m, SOLVE_STEPS, known, yk)
    return np.clip(np.where(known, yk, x), -1.0, 1.0)
