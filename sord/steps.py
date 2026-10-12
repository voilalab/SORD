"""The building blocks of SORD (Algorithm 1).

Shapes: a batch of B images y (B, H, W, C) in [0, 1], their supports omega (B, H, W) (boolean), and K
samples per image stacked as (B, K, H, W, C).  Every image has its own random generator (rngs[b]) and
every reduction (sigma*, normalization, stopping) is taken per image, so an image's result does not
depend on which other images share its batch; batching only fills the GPU.

All diffusion work goes through one clean-signal prior (sord/priors.py), which exposes

    prior.sigmas         the model's noise ladder, descending (120 levels, 9.85 ... 0.098)
    prior.denoise(x, i)  E[x0 | x] at ladder level i, i.e. the Tweedie estimate for x = x0 + sigmas[i] * z

and nothing else.  Sampling uses deterministic DDIM steps between ladder levels,

    x_{i+1} = x0 + (sigma_{i+1} / sigma_i) * (x_i - x0),        x0 = denoise(x_i, i).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

FRAME = 64              # the residual is denoised on a FRAME x FRAME crop of Omega's bounding box
NSTEP_FLOOR = 0.02      # lowest refinement scale for the residual (line 11)
NSTEP_LEVELS = 8        # DDIM levels used to denoise the residual
RESAMPLE_EVERY = 10     # inpainting: every 10th ladder level is re-noised and re-denoised ...
RESAMPLE_REPEATS = 2    # ... twice (RePaint-style resampling)
SHADOW_EPS = 2.0 / 255  # log-domain offset for the multiplicative (shadow) model
SHADOW_MIN_GAIN = 0.05  # a shadow can darken a pixel to at most 5% of its value


# ----------------------------------------------------------------------------------------------- utilities
def noise(rngs, shape):
    """Standard normal noise of the given per-image shape, drawn from each image's own generator."""
    return np.stack([r.standard_normal(shape) for r in rngs])


def denoise_samples(prior, x, i):
    """prior.denoise on (B, K, H, W, C): all samples of all images in one network call."""
    return prior.denoise(x.reshape((-1,) + x.shape[2:]), i).reshape(x.shape)


def levels_at_or_below(prior, sigma):
    """Indices of all ladder levels with noise <= sigma, from the highest down."""
    return [i for i, s in enumerate(prior.sigmas) if s <= sigma + 1e-9]


def nearest_level(prior, sigma):
    return int(np.argmin(np.abs(np.asarray(prior.sigmas) - sigma)))


def bounding_box(omega):
    rows, cols = np.where(omega)
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


def resize(a, hw):
    """Bilinear resize of an (h, w, C) array to (H, W)."""
    return zoom(a, (hw[0] / a.shape[0], hw[1] / a.shape[1], 1), order=1)


def sigma_star(samples, omega):
    """Line 5, per image: RMS over Omega of the per-pixel spread of the K samples.

    The variance is taken across samples (unbiased, ddof=1) and averaged over colour channels.
    samples (B, K, H, W, C), omega (B, H, W)  ->  (B,)"""
    var = samples.var(axis=1, ddof=1).mean(axis=-1)
    return np.array([np.sqrt(v[o].mean()) for v, o in zip(var, omega)])


# ------------------------------------------------------------------------------- line 2: inpainting init
def inpaint(prior, y, omega, K, rngs):
    """R_theta(y, Omega): K inpainting samples per image, conditioned on y outside Omega.

    Each sample descends the full ladder.  At every level the estimate x0 is replaced by y outside Omega
    before the DDIM step (RePaint's known-region replacement); every RESAMPLE_EVERY-th level the sample is
    re-noised to that level and re-denoised RESAMPLE_REPEATS more times so the known region can propagate
    into Omega.  The samples are returned without clipping."""
    known = ~omega[:, None, ..., None]
    yk = y[:, None]
    s = prior.sigmas
    x = np.where(known, yk, 0.0) + s[0] * noise(rngs, (K,) + y.shape[1:])
    last = len(s) - 1
    for i in range(len(s)):
        repeats = RESAMPLE_REPEATS if (i % RESAMPLE_EVERY == 0 and i < last) else 0
        for _ in range(repeats):
            x0 = np.where(known, yk, denoise_samples(prior, x, i))
            x = x0 + s[i] * noise(rngs, x.shape[1:])                    # back to level i
        x0 = np.where(known, yk, denoise_samples(prior, x, i))
        if i < last:
            x = x0 + (s[i + 1] / s[i]) * (x - x0)
    return x0


# ---------------------------------------------------------------------- lines 10-11: corruption estimate
def class_box(y, kind):
    """Bounds that any corruption n with y = x + n, x in [0, 1] must satisfy, per pixel.

    kind = "pos"  : non-negative contaminant (ink, patches drawn on top)  ->  0 <= n <= y
           "free" : signed contaminant (watermarks)                       ->  y - 1 <= n <= y"""
    lo, hi = y - 1.0, y.copy()
    if kind == "pos":
        lo = np.maximum(lo, 0.0)
    return lo, hi


def denoise_residual(prior, read, omega, sigma, box, rngs, zero_anchor=False, omega_weights=False):
    """D_theta(read + sigma * eps, sigma) for residuals that live on Omega (line 11), one per image.

    A residual is not an image from the prior's training distribution, so it is brought into the
    prior's input range first: crop Omega's bounding box, resize it to FRAME x FRAME and normalize it to
    mean 0.5 / std 0.25 (statistics over the crop, or over Omega only if omega_weights; for a
    non-negative contaminant on black, zero_anchor scales it so its 99.5th percentile is 1).  It is then
    re-noised to the ladder level nearest max(sigma_b, NSTEP_FLOOR) and denoised with NSTEP_LEVELS DDIM
    levels; the result is mapped back, clipped to the class box and pasted on Omega.  Images whose sigma
    selects the same levels share one network call."""
    B, C = read.shape[0], read.shape[-1]
    lo, hi = box
    boxes, normed, unnorm, plans = [], [], [], []
    for b in range(B):
        r0, r1, c0, c1 = bounding_box(omega[b])
        crop = resize(read[b, r0:r1, c0:c1], (FRAME, FRAME))
        if zero_anchor:
            scale = max(float(np.percentile(crop, 99.5)), 0.05)
            normed.append(crop / scale)
            unnorm.append(lambda z, scale=scale: z * scale)
        else:
            w = np.ones_like(crop)
            if omega_weights:
                w = np.clip(resize(np.repeat(omega[b, r0:r1, c0:c1, None].astype(float), C, -1),
                                   (FRAME, FRAME)), 0.0, 1.0)
            mean = float((crop * w).sum() / w.sum())
            std = float(np.sqrt(((crop - mean) ** 2 * w).sum() / w.sum())) + 1e-8
            normed.append((crop - mean) / std * 0.25 + 0.5)
            unnorm.append(lambda z, mean=mean, std=std: (z - 0.5) / 0.25 * std + mean)
        top = nearest_level(prior, max(float(sigma[b]), NSTEP_FLOOR))
        sub = levels_at_or_below(prior, prior.sigmas[top])
        plans.append(tuple(sub[k] for k in np.linspace(0, len(sub) - 1, min(NSTEP_LEVELS, len(sub))).astype(int)))
        boxes.append((r0, r1, c0, c1))

    den = np.zeros((B, FRAME, FRAME, C))
    for plan in set(plans):                                  # images that use the same DDIM levels
        idx = [b for b in range(B) if plans[b] == plan]
        x = np.stack([normed[b] for b in idx]) + prior.sigmas[plan[0]] * noise([rngs[b] for b in idx], (FRAME, FRAME, C))
        for i, j in zip(plan[:-1], plan[1:]):
            x0 = prior.denoise(x, i)
            x = x0 + (prior.sigmas[j] / prior.sigmas[i]) * (x - x0)
        den[idx] = prior.denoise(x, plan[-1])

    out = np.zeros_like(read)
    for b, (r0, r1, c0, c1) in enumerate(boxes):
        back = resize(unnorm[b](den[b]), (r1 - r0, c1 - c0))
        out[b, r0:r1, c0:c1] = np.clip(back, lo[b, r0:r1, c0:c1], hi[b, r0:r1, c0:c1])
    return out * omega[..., None]


def n_step(prior, y, x_hat, omega, sigma, rngs, kind="free", zero_anchor=False):
    """Lines 10-11, additive model y = x + n on Omega: read the corruption off the current estimate,
    clip it to what the class allows, and denoise it with the clean prior.  Returns n_hat (B, H, W, C)."""
    box = class_box(y, kind)
    read = np.clip(y - x_hat, *box) * omega[..., None]
    return denoise_residual(prior, read, omega, sigma, box, rngs, zero_anchor=zero_anchor)


def h_step(prior, y, x_hat, omega, sigma, rngs):
    """Lines 10-11 for the multiplicative (shadow) model y = x * h on Omega, 0.05 <= h <= 1.

    Works on the log-gain l = log(y) - log(x) <= 0, which turns the shadow into an additive corruption;
    the normalization statistics are taken over Omega only.  Returns the equivalent additive corruption
    n = y * (1 - 1/h), so that y - n = y / h is the de-shadowed evidence used by the x-step."""
    eps = SHADOW_EPS
    log_y = np.log(y + eps)
    read = log_y - np.log(np.clip(x_hat, 0.0, 1.0) + eps)
    lo = np.maximum(log_y - np.log(1.0 + eps), np.log(SHADOW_MIN_GAIN))
    hi = np.minimum(log_y - np.log(eps), 0.0)
    hi = np.maximum(hi, lo)
    read = np.clip(read, lo, hi) * omega[..., None]
    log_gain = denoise_residual(prior, read, omega, sigma, (lo, hi), rngs, omega_weights=True)
    return y * (1.0 - np.exp(-log_gain)) * omega[..., None]


# -------------------------------------------------------------------------------- lines 12-13: x-step
def x_step(prior, y, n_hat, omega, sigma_m, K, rngs):
    """Lines 12-13: subtract the corruption estimate on Omega (measurement consistency) and pull the result
    back to the clean-signal distribution.

    Each of the K samples re-noises the evidence to the highest ladder level <= sigma_m (the level the
    descent starts from) and descends every ladder level <= sigma_m;
    outside Omega the estimate is replaced by y at every level, so the output equals y there exactly.
    Samples are clipped to [0, 1].  Returns (B, K, H, W, C)."""
    known = ~omega[:, None, ..., None]
    yk = y[:, None]
    evidence = np.where(omega[..., None], y - n_hat, y)[:, None]
    steps = levels_at_or_below(prior, sigma_m)
    x = evidence + prior.sigmas[steps[0]] * noise(rngs, (K,) + y.shape[1:])
    for k, i in enumerate(steps):
        x0 = np.where(known, yk, denoise_samples(prior, x, i))
        x = x0 + (prior.sigmas[steps[k + 1]] / prior.sigmas[i]) * (x - x0) if k + 1 < len(steps) else x0
    return np.clip(np.where(known, yk, x), 0.0, 1.0)
