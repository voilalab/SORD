"""tango_alt/nstep_alpha.py — ALPHA-BLEND READ-THROUGH n-step (2026-09-01).

Class declaration for semi-transparent watermarks: on Ω the observation is

    y = (1 − α) x + α w,        α ∈ [0, α_max) per pixel (scalar over RGB),
                                w = logo colour, K-colour palette per image.

The prior-based n-step (tango_alt/nstep.py) treats Ω as missing content and
regenerates it; on faint marks (CLWD: mean|n| = 0.074 of range vs content
std 0.249) the observation already carries 91% of the hidden variance, so
any fill loses to y itself. This step instead READS THROUGH the mark: given
the current x̂ it fits (α, w) to the residual r = y − x̂ = α (w − x̂) + (x − x̂)
by alternating least squares, then returns

    x_rt = (y − α w) / (1 − α)      (clipped to [lo, hi]),
    n̂    = y − x_rt   on Ω          (so the loop's ỹ = y − n̂ is x_rt).

The inpaint error (x − x̂) is fitted only along the single direction
(w_k − x̂) per pixel, so the K-colour palette + α smoothing are what keep
the estimator from absorbing it. Model order K is chosen by residual
reduction. Diagnostics carry α, w, palette size and the per-pixel misfit
e = r − α (w − x̂), from which a tether variance τ² = |e|² / (1−α)² + floor²
is derived (evidence trust: faint pixels keep y, opaque core is fitted).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _smooth_in_mask(a, m2, sig):
    """Mask-aware Gaussian smoothing (normalised convolution) of a 2-D field."""
    if sig <= 0:
        return a
    num = gaussian_filter(np.where(m2, a, 0.0), sig)
    den = gaussian_filter(m2.astype(np.float64), sig)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), a)
    return np.where(m2, out, a)


def _fit_palette(r, xh, K, lo, hi, iters=12, alpha_max=0.95, seed=0):
    """Alternating LS for (alpha_p, k_p, w_k) on the masked pixels.

    r, xh: (P, C). Returns alpha (P,), lab (P,), w (K, C), misfit (P, C)."""
    rng = np.random.RandomState(seed)
    P, C = r.shape
    # init: assume alpha0 = 0.5 => w0 = xh + 2 r, k-means on the clipped w0
    w0 = np.clip(xh + 2.0 * r, lo, hi)
    if K == 1:
        w = w0.mean(0, keepdims=True)
    else:
        idx = rng.choice(P, size=K, replace=False)
        w = w0[idx].copy()
        for _ in range(8):
            d = ((w0[:, None, :] - w[None]) ** 2).sum(-1)
            lab0 = d.argmin(1)
            for k in range(K):
                sel = lab0 == k
                if sel.any():
                    w[k] = w0[sel].mean(0)
    alpha = np.full(P, 0.5)
    lab = np.zeros(P, int)
    for _ in range(iters):
        # assignment + per-pixel alpha given w
        best = np.full(P, np.inf)
        for k in range(K):
            dk = w[k][None] - xh                       # (P, C)
            den = (dk ** 2).sum(-1) + 1e-12
            ak = np.clip((r * dk).sum(-1) / den, 0.0, alpha_max)
            res = ((r - ak[:, None] * dk) ** 2).sum(-1)
            upd = res < best
            best = np.where(upd, res, best)
            alpha = np.where(upd, ak, alpha)
            lab = np.where(upd, k, lab)
        # palette update given alpha, assignment
        for k in range(K):
            sel = lab == k
            if sel.any():
                a_ = alpha[sel][:, None]
                num = (a_ * (r[sel] + a_ * xh[sel])).sum(0)
                den = (a_ ** 2).sum() + 1e-12
                w[k] = np.clip(num / den, lo, hi)
    dk = w[lab] - xh
    misfit = r - alpha[:, None] * dk
    return alpha, lab, w, misfit


def _fit_smooth_w(r, xh, m2, lo, hi, w_sig=2.0, iters=10, alpha_max=0.95,
                  a_sig=0.0):
    """Alternating LS with a SPATIALLY SMOOTH per-pixel logo colour w
    (mask-aware Gaussian, alpha²-weighted) instead of a palette.
    r, xh: (P, C) masked pixels; m2: (H, W) mask. Returns alpha, w, misfit."""
    P, C = r.shape
    H, W = m2.shape
    w = np.clip(xh + 2.0 * r, lo, hi)             # alpha0 = 0.5 init
    alpha = np.full(P, 0.5)
    for _ in range(iters):
        dk = w - xh
        den = (dk ** 2).sum(-1) + 1e-12
        alpha = np.clip((r * dk).sum(-1) / den, 0.0, alpha_max)
        if a_sig > 0:
            a2 = np.zeros((H, W)); a2[m2] = alpha
            alpha = np.clip(_smooth_in_mask(a2, m2, a_sig)[m2], 0.0,
                            alpha_max)
        # per-pixel LS colour w_p = xh + r/alpha, then alpha^2-weighted
        # mask-aware smoothing (weights kill the unstable small-alpha pixels)
        a_safe = np.maximum(alpha, 0.05)
        w_p = np.clip(xh + r / a_safe[:, None], lo, hi)
        wt = np.zeros((H, W)); wt[m2] = alpha ** 2 + 1e-4
        for c in range(C):
            f = np.zeros((H, W)); f[m2] = w_p[:, c] * wt[m2]
            num = gaussian_filter(f, w_sig); den2 = gaussian_filter(wt, w_sig)
            w[:, c] = np.clip((num / np.maximum(den2, 1e-9))[m2], lo, hi)
    misfit = r - alpha[:, None] * (w - xh)
    return alpha, w, misfit


def _fit_global(r, xh, K, lo, hi, iters=15, alpha_max=0.95, seed=0,
                robust=False):
    """Most constrained model: ONE alpha per image + K-colour palette.
    The x̂ error averages out over the whole mask (4 + 3(K-1) params)."""
    rng = np.random.RandomState(seed)
    P, C = r.shape
    alpha = 0.3
    w0 = np.clip(xh + r / alpha, lo, hi)
    if K == 1:
        w = np.median(w0, 0, keepdims=True) if robust else w0.mean(0, keepdims=True)
    else:
        w = w0[rng.choice(P, size=K, replace=False)].copy()
    lab = np.zeros(P, int)
    for _ in range(iters):
        d = ((w0[:, None, :] - w[None]) ** 2).sum(-1) if K > 1 else None
        if K > 1:
            lab = d.argmin(1)
        dk = w[lab] - xh
        num = (r * dk).sum(-1); den = (dk ** 2).sum(-1) + 1e-12
        if robust:
            ok = den > np.percentile(den, 20)      # drop near-degenerate px
            alpha = float(np.clip(np.median(num[ok] / den[ok]), 0.02, alpha_max))
        else:
            alpha = float(np.clip(num.sum() / den.sum(), 0.02, alpha_max))
        w0 = np.clip(xh + r / alpha, lo, hi)
        for k in range(K):
            sel = lab == k
            if sel.any():
                w[k] = (np.median(w0[sel], 0) if robust else w0[sel].mean(0))
    dk = w[lab] - xh
    misfit = r - alpha * dk
    return np.full(P, alpha), lab, w, misfit


def _fit_w_known_alpha(r, xh, alpha, m2, lo, hi, K=None, w_sig=2.0,
                       fit_scale=False, iters=6, alpha_max=0.95):
    """DECLARED-alpha read-through: alpha given per pixel (soft mask /
    alpha matte). Unknown = logo colour w only: K-colour palette (K int)
    or spatially smooth w (K None, Gaussian w_sig). fit_scale: alpha_p =
    s * c_p with c_p the given coverage and s one scalar per image (the
    CLWD case: the 'Mask' is binary at 256, so the soft 64² mask is
    coverage, not opacity)."""
    H, W = m2.shape
    a = np.clip(alpha, 0.0, alpha_max).astype(np.float64)
    s_ = 1.0
    for it in range(iters):
        ap = np.clip(s_ * a, 0.0, alpha_max)
        a_safe = np.maximum(ap, 0.05)
        w_p = np.clip(xh + r / a_safe[:, None], lo, hi)
        if K is None:
            wt = np.zeros((H, W)); wt[m2] = ap ** 2 + 1e-4
            w = np.zeros_like(w_p)
            for c in range(xh.shape[1]):
                f = np.zeros((H, W)); f[m2] = w_p[:, c] * wt[m2]
                num = gaussian_filter(f, w_sig); den = gaussian_filter(wt, w_sig)
                w[:, c] = np.clip((num / np.maximum(den, 1e-9))[m2], lo, hi)
        else:
            # alpha^2-weighted k-means on w_p
            rng = np.random.RandomState(0)
            wk = w_p[rng.choice(len(w_p), size=K, replace=False)].copy()
            lab = np.zeros(len(w_p), int)
            for _ in range(10):
                d = ((w_p[:, None, :] - wk[None]) ** 2).sum(-1)
                lab = d.argmin(1)
                for k in range(K):
                    sel = lab == k
                    if sel.any():
                        wgt = (ap[sel] ** 2 + 1e-6)[:, None]
                        wk[k] = (w_p[sel] * wgt).sum(0) / wgt.sum()
            w = wk[lab]
        if not fit_scale:
            break
        dk = a[:, None] * (w - xh)                 # r ≈ s * a (w - xh)
        s_ = float(np.clip((r * dk).sum() / ((dk ** 2).sum() + 1e-12),
                           0.05, 1.0 / max(a.max(), 1e-3)))
    ap = np.clip(s_ * a, 0.0, alpha_max)
    misfit = r - ap[:, None] * (w - xh)
    return ap, w, misfit, s_


def _wmedian(v, w):
    """Weighted median per column. v (P,C), w (P,)."""
    out = np.zeros(v.shape[1])
    for c in range(v.shape[1]):
        o = np.argsort(v[:, c]); cw = np.cumsum(w[o]); k = int(np.searchsorted(cw, 0.5 * cw[-1]))
        out[c] = v[o[min(k, len(o) - 1)], c]
    return out


def _mode_filter(lab2, m2, K, it=1):
    """Majority (mode) filter of an integer label field inside the mask."""
    from scipy.ndimage import uniform_filter
    out = lab2.copy()
    for _ in range(it):
        votes = np.stack([uniform_filter(((out == k) & m2).astype(np.float64), 3)
                          for k in range(K)])
        new = votes.argmax(0)
        out = np.where(m2, new, out)
    return out


def _fit_w_known_alpha_pp(r, xh, alpha, m2, lo, hi, Kmax=4, k_gain=0.15,
                          w_sig=1.5, shrink=0.0, fit_scale=False, iters=4,
                          alpha_max=0.95):
    """PER-PIXEL COLOUR extension of the declared-alpha read-through.

    Per-pixel LS colour w_p = x̂ + r/α_p has noise ∝ σ_fill/α_p, so:
      1. α²-weighted, mask-aware Gaussian smoothing of w_p (w_sig px);
      2. weighted k-means on the SMOOTHED colours for K = 1..Kmax, labels
         mode-filtered (3×3) so single pixels cannot flip; palette colour =
         weighted median of the raw w_p in the cluster;
      3. K chosen by residual reduction (each extra colour must cut the
         weighted residual by > k_gain);
      4. optional shrinkage: w = w_pal + s_p (w_p − w_pal),
         s_p = α_p² / (α_p² + shrink) — lets strongly-marked pixels keep
         their own colour, faint ones fall back to the palette (0 = off).
    Opacity scale s (α = s·coverage) fitted jointly when fit_scale."""
    H, W = m2.shape
    P, C = r.shape
    a = np.clip(alpha, 0.0, alpha_max).astype(np.float64)
    s_ = 1.0
    rng = np.random.RandomState(0)
    for it in range(iters if fit_scale else 1):
        ap = np.clip(s_ * a, 0.0, alpha_max)
        a_safe = np.maximum(ap, 0.05)
        w_p = np.clip(xh + r / a_safe[:, None], lo, hi)
        wt = ap ** 2 + 1e-4
        # 1. smoothed colours
        wt2 = np.zeros((H, W)); wt2[m2] = wt
        w_s = np.zeros_like(w_p)
        den = gaussian_filter(wt2, w_sig)
        for c in range(C):
            f = np.zeros((H, W)); f[m2] = w_p[:, c] * wt
            w_s[:, c] = (gaussian_filter(f, w_sig) / np.maximum(den, 1e-9))[m2]
        # 2-3. palette by residual reduction
        best = None
        for K in range(1, Kmax + 1):
            if K > max(1, P // 12):
                break
            if K == 1:
                lab = np.zeros(P, int)
            else:
                cen = w_s[rng.choice(P, size=K, replace=False)].copy()
                lab = np.zeros(P, int)
                for _ in range(10):
                    d = ((w_s[:, None, :] - cen[None]) ** 2).sum(-1)
                    lab = d.argmin(1)
                    for k in range(K):
                        sel = lab == k
                        if sel.any():
                            cen[k] = (w_s[sel] * wt[sel, None]).sum(0) / wt[sel].sum()
                lab2 = np.zeros((H, W), int); lab2[m2] = lab
                lab = _mode_filter(lab2, m2, K, it=1)[m2]
            pal = np.zeros((K, C))
            for k in range(K):
                sel = lab == k
                pal[k] = _wmedian(w_p[sel], wt[sel]) if sel.any() else w_s.mean(0)
            w_k = pal[lab]
            res = float((wt[:, None] * (r - ap[:, None] * (w_k - xh)) ** 2).sum())
            if best is None or res < (1.0 - k_gain) * best[0]:
                best = (res, K, lab, pal, w_k)
        res, K_sel, lab, pal, w = best
        if shrink > 0:
            sp = (ap ** 2 / (ap ** 2 + shrink))[:, None]
            w = np.clip(w + sp * (w_p - w), lo, hi)
        if not fit_scale:
            break
        dk = a[:, None] * (w - xh)
        s_ = float(np.clip((r * dk).sum() / ((dk ** 2).sum() + 1e-12),
                           0.05, 1.0 / max(a.max(), 1e-3)))
    ap = np.clip(s_ * a, 0.0, alpha_max)
    misfit = r - ap[:, None] * (w - xh)
    return ap, w, misfit, s_, K_sel


def n_step_alpha(y, x_hat, M3, lo, hi, K_colors=(1, 2), smooth=1.0,
                 alpha_max=0.95, k_gain=0.2, floor=0.02, seed=0,
                 mode="palette", w_sig=2.0, robust=False,
                 alpha_known=None, known_K=None, fit_scale=False,
                 pp_Kmax=4, pp_shrink=0.0, pp_wsig=1.5, pp_kgain=0.15,
                 fuse_sigma=0.0):
    """Alpha-blend read-through n-step for a batch. Returns (n_hat, diag).

    K_colors: candidate palette sizes; the smallest K is kept unless a
      larger one cuts the fit residual by more than `k_gain` (fraction).
    smooth: Gaussian sigma (px) for mask-aware smoothing of alpha after
      the fit (0 = off). The palette is refit once after smoothing.
    diag: read (= y − x̂ on Ω), dproj, alpha (B,H,W), w (list of (K,C)),
      K (list), misfit_rms (list), tau (B,H,W) = |e|²/(1−α)² + floor²."""
    B, H, W, C = y.shape
    yy = np.asarray(y, np.float64)
    xh_all = np.asarray(x_hat, np.float64)
    n_hat = np.zeros_like(yy)
    reads = np.zeros_like(yy)
    alpha_out = np.zeros((B, H, W))
    tau = np.zeros((B, H, W))
    ws, Ks, mis, dproj = [], [], [], []
    for b in range(B):
        m2 = M3[b, ..., 0] > 0.5
        read = (yy[b] - xh_all[b]) * (M3[b] > 0.5)
        reads[b] = read
        if not m2.any():
            ws.append(np.zeros((1, C))); Ks.append(0); mis.append(0.0)
            dproj.append(0.0)
            continue
        r = read[m2]                                   # (P, C)
        xh = xh_all[b][m2]
        if mode == "known" and known_K == "pp":
            if alpha_known is None:
                raise ValueError("mode=known needs alpha_known (B,H,W)")
            alpha, w, misfit, s_, K_sel = _fit_w_known_alpha_pp(
                r, xh, np.asarray(alpha_known[b], np.float64)[m2], m2, lo,
                hi, Kmax=pp_Kmax, k_gain=pp_kgain, w_sig=pp_wsig,
                shrink=pp_shrink, fit_scale=fit_scale, alpha_max=alpha_max)
            lab = np.arange(r.shape[0])
        elif mode == "known":
            if alpha_known is None:
                raise ValueError("mode=known needs alpha_known (B,H,W)")
            alpha, w, misfit, s_ = _fit_w_known_alpha(
                r, xh, np.asarray(alpha_known[b], np.float64)[m2], m2, lo,
                hi, K=known_K, w_sig=w_sig, fit_scale=fit_scale,
                alpha_max=alpha_max)
            lab = np.arange(r.shape[0]); K_sel = -2
        elif mode == "smooth":
            alpha, w, misfit = _fit_smooth_w(r, xh, m2, lo, hi, w_sig=w_sig,
                                             alpha_max=alpha_max,
                                             a_sig=smooth)
            lab = np.arange(r.shape[0]); K_sel = -1
        fits = {}
        for K in ([] if mode in ("smooth", "known") else K_colors):
            K = int(K)
            if K > max(1, r.shape[0] // 8):
                continue
            if mode == "global":
                fits[K] = _fit_global(r, xh, K, lo, hi, alpha_max=alpha_max,
                                      seed=seed + b, robust=robust)
            else:
                fits[K] = _fit_palette(r, xh, K, lo, hi, alpha_max=alpha_max,
                                       seed=seed + b)
        Ks_sorted = sorted(fits)
        if fits:
            K_sel = Ks_sorted[0]
            res_sel = float((fits[K_sel][3] ** 2).sum())
            for K in Ks_sorted[1:]:
                res_K = float((fits[K][3] ** 2).sum())
                if res_K < (1.0 - k_gain) * res_sel:
                    K_sel, res_sel = K, res_K
            alpha, lab, w, misfit = fits[K_sel]
        if smooth > 0 and mode == "palette":
            a2 = np.zeros((H, W)); a2[m2] = alpha
            a2 = _smooth_in_mask(a2, m2, smooth)
            alpha = np.clip(a2[m2], 0.0, alpha_max)
            # refit palette once with the smoothed alpha
            for k in range(K_sel):
                sel = lab == k
                if sel.any():
                    a_ = alpha[sel][:, None]
                    num = (a_ * (r[sel] + a_ * xh[sel])).sum(0)
                    w[k] = np.clip(num / ((a_ ** 2).sum() + 1e-12), lo, hi)
            misfit = r - alpha[:, None] * (w[lab] - xh)
        # read-through
        wp = w[lab]
        x_rt = (yy[b][m2] - alpha[:, None] * wp) / (1.0 - alpha)[:, None]
        x_rt = np.clip(x_rt, lo, hi)
        if fuse_sigma > 0:
            # inverse-variance fusion with the fill x̂: var_rt = e²/(1−α)²
            # (e = smoothed misfit RMS, floored), var_fill = fuse_sigma²
            # ⇒ opaque pixels fall back to the fill, faint ones keep the
            # read-through (the evidence-trust rule, applied to ỹ itself)
            e2 = np.zeros((H, W)); e2[m2] = (misfit ** 2).mean(-1)
            e2 = _smooth_in_mask(e2, m2, 1.0)[m2]
            e2 = np.maximum(e2, (0.5 * floor * (hi - lo)) ** 2)
            var_rt = e2 / np.maximum(1.0 - alpha, 0.02) ** 2
            g = (var_rt / (var_rt + fuse_sigma ** 2))[:, None]   # weight on x̂
            x_rt = (1.0 - g) * x_rt + g * xh
            x_rt = np.clip(x_rt, lo, hi)
        nb = np.zeros((H, W, C)); nb[m2] = yy[b][m2] - x_rt
        n_hat[b] = nb
        alpha_out[b][m2] = alpha
        e2 = (misfit ** 2).mean(-1) / np.maximum(1.0 - alpha, 0.05) ** 2
        t2 = np.zeros((H, W)); t2[m2] = e2
        t2 = _smooth_in_mask(t2, m2, 1.0)
        tau[b] = t2 + (floor * (hi - lo)) ** 2
        ws.append(w); Ks.append(K_sel)
        mis.append(float(np.sqrt((misfit ** 2).mean())))
        num = float(np.linalg.norm(n_hat[b] - read))
        dproj.append(num / max(float(np.linalg.norm(read)), 1e-12))
    return n_hat, dict(read=reads, dproj=dproj, alpha=alpha_out, w=ws,
                       K=Ks, misfit_rms=mis, tau=tau)
