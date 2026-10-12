"""tango_wave — waveform port of the shipped TANGO-ALT loop.

Faithful 1D translation of tango_alt_sigma_test/run_loop_test.py
(probe=alt-decay, blind gamma-shrink, last iterate, off-Omega pinned)
with the MSDM piano checkpoint (msdm_prior.MSDMPrior) as THE one prior.
E4-E9 arms (findings.md E-series) are flag-gated via `arms`; every
default reproduces the shipped E2 path bit-identically.

  x-step  map_solve: reverse Karras ladder from an explicit altitude,
          per-step Eq.-1 tether  w = v_t/(v_t + sigma_e^2) toward
          y_tilde = y - n_hat, v_t = min(sigma_t^2, ensemble spread),
          off-Omega replaced by y every step. E9 adds a TF missing-bin
          reopen (declared-unobserved bins take the untethered x0).
  n-step  DUO content-agnostic denoise of the read; E4 projects the
          read and the post-DDIM estimate onto the feasible set C(y).
  tau^2   a*smooth((n-read)^2) + b*|n|^2 + floor^2.
  gamma   leak-control shrink by the 1D texture score.
  E5      pooled demix seed: Wiener shrink of the init read by the
          init ensemble's spread, then one seeded x-step before the
          alternation.
  E7      TF pareto exchange after each x-step (sum-preserving).
  E8      terminal harmonize solve + declared equal-power crossfade.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .msdm_prior import karras_sigmas
from . import arms as A

SIGMA_DATA = 0.2
SIGMA_MIN = 1e-4

ARMS_DEFAULT = dict(
    e4_box=None,          # None | "mag" | "magY"
    e4_perc=False, e4_onset_thresh=3.0,
    e5_seed=False,
    e7_exchange=False, e7_percussive=True,
    e8_harmonize=False, e8_xfade_ms=5.0,
    e9_missing=None,      # (F,T) bool of declared-missing TF bins
)


def _smooth1d(x, width):
    if width % 2 == 0:
        width += 1
    k = torch.ones(1, 1, width, device=x.device, dtype=x.dtype) / width
    xp = F.pad(x, (width // 2, width // 2), mode="replicate")
    return F.conv1d(xp, k)


def _hf_map(x, width=63):
    mu = _smooth1d(x, width)
    return (x - mu) ** 2


def texture_score(x, m, width_s=0.5, sr=22050):
    hf = _hf_map(x)
    ann = (_smooth1d(m.float(), int(2 * width_s * sr)) > 0) & ~m
    if not (m.any() and ann.any()):
        return 1.0
    R = float(hf[m].mean()) / max(float(hf[ann].mean()), 1e-12)
    return float(min(R, 1.0 / R)) if R > 0 else 0.0


def _reopen_missing(x0g, x0, miss):
    """E9: declared-missing TF bins take the untethered x0 (evidence
    weight 0 there). x0g, x0: (K,1,L); miss: (F,T) bool."""
    K, _, L = x0g.shape
    win = torch.hann_window(A.NFFT, device=x0g.device)
    Xg = torch.stft(x0g[:, 0], A.NFFT, A.HOP, window=win, center=True,
                    return_complex=True)
    X0 = torch.stft(x0[:, 0], A.NFFT, A.HOP, window=win, center=True,
                    return_complex=True)
    Xg = torch.where(miss[None], X0, Xg)
    out = torch.istft(Xg, A.NFFT, A.HOP, window=win, center=True,
                      length=L)
    return out[:, None]


DPS_CHUNK = 8       # chains per autograd tape (memory)


def _x0_and_grad(prior, x, s_t, loss_fn):
    """x0_hat(x, s_t) plus d loss_fn(x0)/dx, chunked over the chain dim.
    NFE counted as forward + backward per chain (ablation convention)."""
    x0 = torch.empty_like(x)
    g = torch.empty_like(x)
    for j0 in range(0, x.shape[0], DPS_CHUNK):
        j1 = min(j0 + DPS_CHUNK, x.shape[0])
        with torch.enable_grad():
            xt = x[j0:j1].detach().requires_grad_(True)
            x0_j = prior.denoise_fn(xt, sigma=float(s_t))
            loss = loss_fn(x0_j, j0, j1)
            g[j0:j1] = torch.autograd.grad(loss, xt)[0]
        x0[j0:j1] = x0_j.detach()
    prior.nfe += 2 * int(x.shape[0])
    return x0, g


def map_solve(prior, y_tilde, y, m, sigma_e2, sigma_start, n_steps=16,
              K=4, seed=0, pool_width=63, miss=None, w_zero=False,
              dps=False, dps_zeta=1.0):
    """One MAP column, K chains. Tensors (1,1,L); returns (x_map, v_map).
    dps=True (ABLATION 2026-09-14, audio port of the image arm): the
    off-window hard replacement is swapped for the DPS likelihood gradient
    zeta * ||(1-m)(x0_hat(x_t) - y)||_2 (repo DPS-inpaint convention),
    applied after the DDIM step; terminal projection unchanged."""
    dev = y.device
    sig = karras_sigmas(n_steps, float(sigma_start), SIGMA_MIN, device=dev)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    yt = y_tilde.expand(K, -1, -1)
    x = yt + float(sig[0]) * torch.randn(yt.shape, generator=g).to(dev)
    mK = m.expand(K, -1, -1)
    yK = y.expand(K, -1, -1)
    known = (~mK).float()
    for i in range(len(sig) - 1):
        s_t = float(sig[i])
        if dps:
            def _loss(x0_j, j0, j1):
                r = (known[j0:j1] * (x0_j - yK[j0:j1])).reshape(j1 - j0, -1)
                return dps_zeta * torch.linalg.norm(r, dim=1).sum()
            x0, g_dps = _x0_and_grad(prior, x, s_t, _loss)
        else:
            x0 = prior.x0_hat(x, s_t)
        if K > 1:
            v_ens = _smooth1d(x0.var(0, keepdim=True, unbiased=True),
                              pool_width)
            v_t = torch.clamp(v_ens, max=s_t ** 2).expand(K, -1, -1)
        else:
            v_t = torch.full_like(x, s_t ** 2)
        w = v_t / (v_t + sigma_e2)
        if w_zero:
            w = torch.zeros_like(w)
        x0g = x0 + w * (yt - x0)
        if miss is not None:
            x0g = _reopen_missing(x0g, x0, miss)
        if not dps:
            x0g = torch.where(mK, x0g, yK)
        s_n = float(sig[i + 1])
        if s_n > 0:
            x = x0g + (s_n / max(s_t, 1e-12)) * (x - x0g)
            if dps:
                x = x - g_dps
        else:
            x = x0g
    chains = torch.where(mK, x, y.expand(K, -1, -1)).clamp(-1, 1)
    x_map = chains.mean(0, keepdim=True)
    v_map = (chains.var(0, keepdim=True, unbiased=True) if K > 1
             else torch.zeros_like(x_map))
    return x_map, v_map


def ddim_dps_gauss(prior, cn, s_star, sigma_top, n_steps=16, seed=0,
                   zeta_in=1.0):
    """ABLATION 2026-09-14 (audio port of tango_alt.nstep.ddim_ve_dps): the
    n-step's renoise+DDIM denoise of the read is replaced by a DPS-Gauss
    reconstruction with the read `cn` as measurement: Karras ladder from
    sigma_top (n_steps), every step adds grad ||x0_hat - cn||^2/(2 s*^2)
    with the sigma-aware cap (pull <= dv/(dv+s*^2) of the residual).
    Init = cn + sigma_top*eps (y-init: this ckpt returns silence from pure
    noise, audit 2026-09-01). Output = x0_hat at the last level."""
    sig = karras_sigmas(n_steps, float(sigma_top), SIGMA_MIN, device=cn.device)
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 4242)
    x = cn + float(sig[0]) * torch.randn(cn.shape, generator=g).to(cn.device)
    w_in = zeta_in / (2.0 * s_star ** 2)
    for i in range(len(sig) - 1):
        s_t, s_n = float(sig[i]), float(sig[i + 1])
        x0, gr = _x0_and_grad(prior, x, s_t,
                              lambda x0_j, j0, j1: w_in * (x0_j - cn[j0:j1]).pow(2).sum())
        dv = s_t ** 2 - s_n ** 2
        alpha = dv / (dv + s_star ** 2)
        r_n = (x0 - cn).reshape(x.shape[0], -1).norm(dim=1)
        g_n = gr.reshape(x.shape[0], -1).norm(dim=1)
        cap = torch.clamp(alpha * r_n / (g_n + 1e-12), max=1.0)[:, None, None]
        x = x0 + (s_n / max(s_t, 1e-12)) * (x - x0) - cap * gr
    return prior.x0_hat(x, max(float(sig[-2]), SIGMA_MIN))


def n_step(prior, y, x_hat, m, sigma_star, w0, wlen, n_ddim=8, seed=0,
           identity=False, norm_floor=None, proj=None, dps=False,
           dps_sigma_top=1.6 * SIGMA_DATA):
    """DUO n-step on the contiguous window. proj = E4 projector applied
    to the read (point 8) and to the pasted estimate (point 11)."""
    read = (y - x_hat) * m
    if proj is not None:
        read = proj(read)
    if identity:
        return read.clone(), dict(read=read, dproj=0.0)
    crop = read[..., w0:w0 + wlen]
    pad = (-crop.shape[-1]) % 8192
    crop_p = F.pad(crop, (0, pad))
    s_ = float(crop_p.std())
    if norm_floor is not None:
        s_ = max(s_, float(norm_floor) * SIGMA_DATA)
    s_ = max(s_, 1e-8)
    cn = crop_p / s_ * SIGMA_DATA
    s_star = max(float(sigma_star), 0.1 * SIGMA_DATA)
    if dps:
        den = ddim_dps_gauss(prior, cn, s_star, dps_sigma_top, seed=seed)
    else:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        x_t = cn + s_star * torch.randn(cn.shape, generator=g).to(cn.device)
        den = prior.ddim(x_t, s_star, n_steps=n_ddim, sigma_min=SIGMA_MIN)
    d = den / SIGMA_DATA * s_
    n_hat = torch.zeros_like(y)
    n_hat[..., w0:w0 + wlen] = d[..., :crop.shape[-1]]
    n_hat = n_hat * m
    if proj is not None:
        n_hat = proj(n_hat)
    dproj = float((n_hat - read).norm() / max(float(read.norm()), 1e-12))
    return n_hat, dict(read=read, dproj=dproj)


def tau_from_projection(n_hat, read, floor, a_coef=1.0, b_coef=0.0,
                        width=221):
    return (a_coef * _smooth1d((n_hat - read) ** 2, width)
            + b_coef * n_hat ** 2 + floor ** 2)


def gamma_shrink(y, read, n_hat, m, grid=(0.5, 0.75, 1.0)):
    best, g_best = -1.0, 1.0
    for gam in grid:
        n_g = read + gam * (n_hat - read)
        s = texture_score(y - n_g, m)
        if s > best:
            best, g_best = s, gam
    return read + g_best * (n_hat - read), g_best


def _xfade(x_in, y, w0, w1, band):
    """E8: equal-power crossfade y->x over [w0, w0+band) and x->y over
    [w1-band, w1). Declared C1 relaxation inside the bands only."""
    out = y.clone()
    out[..., w0:w1] = x_in[..., w0:w1]
    t = torch.linspace(0, 1, band, device=y.device)
    up = torch.sin(t * torch.pi / 2) ** 2
    out[..., w0:w0 + band] = (1 - up) * y[..., w0:w0 + band] + \
        up * x_in[..., w0:w0 + band]
    out[..., w1 - band:w1] = up.flip(0) * x_in[..., w1 - band:w1] + \
        (1 - up.flip(0)) * y[..., w1 - band:w1]
    return out


def tango_wave(prior, y, m, w0, wlen, M=4, K=4, n_solve_steps=16,
               seed=12345, sig_start=1.6 * SIGMA_DATA,
               sig_end=0.4 * SIGMA_DATA, tau_floor=0.02 * SIGMA_DATA,
               tau_a=1.0, tau_b=0.0, blind=True, identity_n=False,
               oracle_n=None, verbose=True, arms=None,
               w_zero=False, no_shrink=False, probe_cmult=None,
               loop_sig_start=None, loop_sig_end=None, loop_sig_list=None,
               xstep_dps=False, nstep_dps=False, loop_n_steps=None,
               n_ddim=8, sig_fixed=None):
    """loop_sig_start/end (2026-09-10, user: 'use the smallest schedule for
    the DM in audio'): altitude schedule of the M loop x-steps ONLY; the
    init inpaint (and e5 seed) keep sig_start. None = sig_start/sig_end."""
    dev = y.device
    m = m.bool()
    cfg = dict(ARMS_DEFAULT)
    if arms:
        cfg.update(arms)
    miss = cfg["e9_missing"]

    proj = None
    if cfg["e4_box"]:
        onset = None
        if cfg["e4_perc"]:
            onset = A.onset_support(y, w0, wlen, cfg["e4_onset_thresh"])
        proj = (lambda n, _o=onset:
                A.e4_project(n, y, m, mode=cfg["e4_box"], onset_mask=_o))

    if oracle_n is not None:                      # gate O
        y_t = torch.where(m, y - oracle_n, y)
        se2 = torch.full_like(y, tau_floor ** 2)
        x_or, _ = map_solve(prior, y_t, y, m, se2, sig_start,
                            n_solve_steps, K, seed, miss=miss)
        return dict(x_alt=x_or)

    # INIT: inpaint backbone
    se2_inf = torch.full_like(y, 1e6)
    x_inp, v_post = map_solve(prior, y, y, m, se2_inf, sig_start,
                              n_solve_steps, K, seed)
    x_hat = x_inp.clone()
    sig_var = v_post

    if cfg["e5_seed"]:                            # pooled demix seed
        read0 = (y - x_inp) * m
        if proj is not None:
            read0 = proj(read0)
        n_seed = A.demix_seed(read0, v_post) * m
        tau2 = tau_from_projection(n_seed, read0, tau_floor, tau_a, tau_b)
        y_t = torch.where(m, y - n_seed, y)
        x_hat, v_map = map_solve(prior, y_t, y, m, tau2, sig_start,
                                 n_solve_steps, K, seed + 55, miss=miss,
                                 w_zero=w_zero)
        sig_var = v_map

    if loop_sig_list is not None:          # 2026-09-21: explicit per-iteration altitude list (waveform units)
        assert len(loop_sig_list) == M, (len(loop_sig_list), M)
        decay = torch.tensor([float(v) for v in loop_sig_list], device=y.device)
    else:
        decay = torch.linspace(sig_start if loop_sig_start is None else loop_sig_start,
                               sig_end if loop_sig_end is None else loop_sig_end, M)
    curves, prev_dx, n_hat = [], None, torch.zeros_like(y)
    x_iters, n_iters = [], []
    ns_loop = n_solve_steps if loop_n_steps is None else int(loop_n_steps)
    for it in range(1, M + 1):
        sigma_star = float(sig_var[m].mean().sqrt()) if m.any() else 0.0
        if sig_fixed is not None:      # ABLATION: constant n-step altitude
            sigma_star = float(sig_fixed)
        if probe_cmult is not None:
            # sigma_m = C * sigma* (image C-sweep port, 2026-08-25):
            # self-tuned altitude, capped at 4*sigma_data (the audio
            # analog of the image 0.8*(hi-lo) cap).
            probe_m = min(probe_cmult * sigma_star, 4.0 * SIGMA_DATA)
        else:
            probe_m = float(decay[it - 1])
        n_hat, nd = n_step(prior, y, x_hat, m, sigma_star, w0, wlen,
                           n_ddim=n_ddim, seed=seed + it,
                           identity=identity_n, proj=proj, dps=nstep_dps)
        gam = 1.0
        if blind and not identity_n and not no_shrink:
            n_hat, gam = gamma_shrink(y, nd["read"], n_hat, m)
        tau2 = tau_from_projection(n_hat, nd["read"], tau_floor,
                                   tau_a, tau_b)
        y_t = torch.where(m, y - n_hat, y)
        x_new, v_map = map_solve(prior, y_t, y, m, tau2, probe_m,
                                 ns_loop, K, seed + 100 * it,
                                 miss=miss, w_zero=w_zero, dps=xstep_dps)
        sig_var = v_map
        fired = 0
        if cfg["e7_exchange"]:
            x_new, n_hat, fired = A.tf_pareto_exchange(
                x_new, n_hat, y, m, w0, wlen,
                percussive=cfg["e7_percussive"])
        dx = float((x_new - x_hat)[m].pow(2).mean().sqrt()) if m.any() \
            else 0.0
        damped = False
        if prev_dx is not None and dx > prev_dx:
            x_new = 0.5 * (x_new + x_hat)
            damped = True
        prev_dx = dx
        curves.append(dict(m=it, sigma_star=sigma_star, probe=probe_m,
                           dproj=nd["dproj"], dx=dx, gamma=gam,
                           damped=damped, exchange_tiles=fired))
        if verbose:
            print(f"    it={it}: sigma*={sigma_star:.4f} "
                  f"probe={probe_m:.3f} dproj={nd['dproj']:.3f} "
                  f"dx={dx:.5f} gamma={gam}"
                  f"{' DAMPED' if damped else ''}"
                  f"{f' exch={fired}' if fired else ''}")
        x_hat = x_new
        x_iters.append(torch.where(m, x_hat, y))
        n_iters.append(n_hat.clone())

    x_alt = torch.where(m, x_hat, y)
    x_harm = None
    if cfg["e8_harmonize"]:
        tau_h = torch.where(m, sig_var.clamp_min(tau_floor ** 2),
                            torch.full_like(y, 1e-8))
        y_t = torch.where(m, y - n_hat, y)
        xh, _ = map_solve(prior, y_t, y, m, tau_h, sig_end,
                          n_solve_steps, K, seed + 777, miss=miss)
        band = int(cfg["e8_xfade_ms"] * 1e-3 * 22050)
        x_harm = _xfade(xh, y, w0, min(w0 + wlen, y.shape[-1]), band)
    return dict(x_alt=x_alt, x_inp=x_inp, n_hat=n_hat, curves=curves,
                x_iters=x_iters, n_iters=n_iters, x_harm=x_harm,
                last_read=nd["read"] if not curves == [] else None)
