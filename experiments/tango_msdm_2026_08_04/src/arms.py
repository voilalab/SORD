"""E4-E9 arm machinery for tango_wave (findings.md E-series).

All STFT work uses one COLA configuration (hann 1024 / hop 256,
center=True) so analysis->synthesis is linear-consistent and
projections compose with the time-domain loop exactly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

NFFT, HOP = 1024, 256


def _win(dev):
    return torch.hann_window(NFFT, device=dev)


def stft(x):
    """(1,1,L) -> complex (F,T)."""
    return torch.stft(x[0], NFFT, HOP, window=_win(x.device),
                      center=True, return_complex=True)[0]


def istft(X, length):
    """complex (F,T) -> (1,1,L)."""
    return torch.istft(X[None], NFFT, HOP, window=_win(X.device),
                       center=True, length=length)[None]


# ---------------------------------------------------------------- E4 ----
def e4_project(n_hat, y, m, mode="mag", onset_mask=None):
    """Feasible-set projection Pi_C in the STFT domain.

    mode 'mag' : per-bin |n'| = min(|N|, |Y|), n-hat keeps its own phase.
    mode 'magY': same clip but clipped bins take y's phase.
    onset_mask : optional (T,) bool of onset-support frames (percussive
    pools); n-hat is zeroed outside supported frames INSIDE the window.
    Time support supp(n) is re-imposed after synthesis (mask m)."""
    L = n_hat.shape[-1]
    N, Y = stft(n_hat), stft(y)
    magN, magY = N.abs(), Y.abs()
    clip = magN > magY
    if mode == "mag":
        Np = torch.where(clip, N * (magY / magN.clamp_min(1e-12)), N)
    else:                                  # magY
        Np = torch.where(clip, magY * torch.exp(1j * Y.angle()), N)
    if onset_mask is not None:
        Np = Np * onset_mask[None, :].to(Np.dtype)
    out = istft(Np, L) * m
    return out


def onset_strength(y):
    """Spectral flux per frame, (T,): positive magnitude increase."""
    mag = stft(y).abs()
    flux = (mag[:, 1:] - mag[:, :-1]).clamp_min(0).sum(0)
    return torch.cat([flux[:1], flux])


def onset_support(y, w0, wlen, thresh, dilate=1):
    """(T,) bool: in-window frames whose onset flux exceeds thresh
    (units of the window's own median flux), dilated +-dilate frames."""
    fl = onset_strength(y)
    f0, f1 = w0 // HOP, (w0 + wlen) // HOP + 1
    ref = fl[f0:f1].median().clamp_min(1e-9)
    sup = torch.zeros_like(fl, dtype=torch.bool)
    sup[f0:f1] = fl[f0:f1] > thresh * ref
    if dilate:
        sup = F.max_pool1d(sup[None, None].float(), 2 * dilate + 1,
                           stride=1, padding=dilate)[0, 0] > 0
        sup[:f0] = False
        sup[f1:] = False
    return sup


# ---------------------------------------------------------------- E5 ----
def demix_seed(read, v_ens, width=221):
    """Wiener-style pooled seed: read = n + e, Var(e) ~ init-ensemble
    spread => per-sample shrink read * r2/(r2 + v). Port of the pooled
    demix seed's role (ensemble deviations -> cleaner initial split)."""
    r2 = _sm(read ** 2, width)
    v = _sm(v_ens, width)
    return read * (r2 / (r2 + v + 1e-12))


def _sm(x, width):
    if width % 2 == 0:
        width += 1
    k = torch.ones(1, 1, width, device=x.device, dtype=x.dtype) / width
    return F.conv1d(F.pad(x, (width // 2, width // 2), mode="replicate"), k)


# ---------------------------------------------------------------- E6 ----
def _flux_frames(x):
    mag = stft(x).abs()
    d = (mag[:, 1:] - mag[:, :-1]).clamp_min(0).sum(0)
    return torch.cat([d[:1], d])


def seam_of_audio(x, w0, wlen, band=3):
    """Excess boundary spectral flux: mean flux at the frames straddling
    w0 and w1 minus the matched mean over interior+exterior control
    frames at the same distances. Positive = boundary sticks out."""
    fl = _flux_frames(x)
    T = fl.shape[0]
    f_edges = [w0 // HOP, (w0 + wlen) // HOP]
    sel = []
    for fe in f_edges:
        sel += [i for i in range(fe - band, fe + band + 1) if 0 <= i < T]
    sel = sorted(set(sel))
    ctrl = [i for i in range(T) if all(abs(i - fe) > 3 * band
                                       for fe in f_edges)]
    if not sel or not ctrl:
        return 0.0
    return float(fl[sel].mean() - fl[ctrl].mean())


def seam_argmin_select(x_iters, w0, wlen):
    """Batch seam-argmin over loop iterates (no box arm in the pool).
    x_iters: list of (1,1,L). Returns (index, per-iter seams)."""
    seams = [seam_of_audio(x, w0, wlen) for x in x_iters]
    return int(min(range(len(seams)), key=lambda i: seams[i])), seams


# ---------------------------------------------------------------- E7 ----
def _tile_pool(a, tf_, tt):
    return F.avg_pool2d(a[None, None], (tf_, tt), ceil_mode=True)[0, 0]


def _harmonicity(mag, tf_, tt):
    """Piano-side score per tile: spectral smoothness along time
    (sustained partials) = -flux/energy."""
    flux = (mag[:, 1:] - mag[:, :-1]).abs()
    flux = F.pad(flux, (1, 0))
    e = _tile_pool(mag, tf_, tt) + 1e-9
    return -_tile_pool(flux, tf_, tt) / e


def _transientness(mag, tf_, tt):
    """Interferer-side score per tile (percussive): broadband
    verticality = frame flux share; for harmonic pools this becomes
    envelope-consistency handled by the caller's score choice."""
    fl = (mag[:, 1:] - mag[:, :-1]).clamp_min(0)
    fl = F.pad(fl, (1, 0))
    e = _tile_pool(mag, tf_, tt) + 1e-9
    return _tile_pool(fl, tf_, tt) / e


def tf_pareto_exchange(x_hat, n_hat, y, m, w0, wlen, frac=0.5,
                       tf_=64, tt=8, percussive=True):
    """Move a fraction of tile content x-hat -> n-hat where BOTH sides'
    scores improve. Proposal direction: the non-smooth part of x-hat
    (x - smoothed x in STFT magnitude), the guided-smoothing analog.
    Sum x+n is preserved exactly (transfer is antisymmetric).
    Returns (x', n', n_tiles_fired)."""
    L = x_hat.shape[-1]
    X, N = stft(x_hat), stft(n_hat)
    f0, f1 = w0 // HOP, (w0 + wlen) // HOP + 1
    magX, magN = X.abs(), N.abs()
    sx0 = _harmonicity(magX, tf_, tt)
    sn0 = _transientness(magN, tf_, tt) if percussive else \
        -_transientness(magN, tf_, tt)
    # proposal: transfer the temporally-rough part of X
    rough = (magX - _sm2(magX)) .clamp_min(0)
    T = rough / magX.clamp_min(1e-9) * frac          # per-bin fraction
    D = X * T                                        # moved content
    Xp, Np = X - D, N + D
    sx1 = _harmonicity(Xp.abs(), tf_, tt)
    sn1 = _transientness(Np.abs(), tf_, tt) if percussive else \
        -_transientness(Np.abs(), tf_, tt)
    acc = (sx1 > sx0) & (sn1 > sn0)
    # frames outside the window never exchange (supp(n) constraint)
    tmask = torch.zeros(acc.shape[1], dtype=torch.bool, device=acc.device)
    t0, t1 = f0 // tt, (f1 + tt - 1) // tt
    tmask[t0:t1] = True
    acc = acc & tmask[None, :]
    accB = acc.repeat_interleave(tf_, 0)[:X.shape[0]] \
              .repeat_interleave(tt, 1)[:, :X.shape[1]]
    Xf = torch.where(accB, Xp, X)
    Nf = torch.where(accB, Np, N)
    x_out = istft(Xf, L)
    n_out = istft(Nf, L) * m
    # exact sum preservation: give x the exact complement
    x_out = torch.where(m, y - n_out, x_out)
    return x_out, n_out, int(acc.sum())


def _sm2(mag, k=9):
    return F.avg_pool2d(F.pad(mag[None, None], (k // 2, k // 2, 0, 0),
                              mode="replicate"), (1, k), stride=1)[0, 0]


# ---------------------------------------------------------------- E9 ----
def sinusoid_tracks(y, w0, wlen, min_frames=20, prom=4.0):
    """Detect stable sinusoidal tracks inside the window: bins that are
    local spectral peaks with magnitude > prom * median-of-frame for at
    least min_frames consecutive frames. Returns (F,T) bool of tracked
    bins and per-track envelope flatness (used by the discriminant)."""
    mag = stft(y).abs()
    Fb, T = mag.shape
    f0, f1 = w0 // HOP, min((w0 + wlen) // HOP + 1, T)
    med = mag[:, f0:f1].median(0).values.clamp_min(1e-9)
    peaks = (mag[:, f0:f1] > prom * med[None])
    up = mag[:, f0:f1] >= F.pad(mag[:-1, f0:f1], (0, 0, 1, 0))
    dn = mag[:, f0:f1] >= F.pad(mag[1:, f0:f1], (0, 0, 0, 1))
    peaks &= up & dn
    # persistence: run-length along time per bin (+-1 bin tolerance)
    wide = peaks | F.pad(peaks[:-1], (0, 0, 1, 0)) | \
        F.pad(peaks[1:], (0, 0, 0, 1))
    run = torch.zeros_like(wide, dtype=torch.int32)
    acc = torch.zeros(wide.shape[0], dtype=torch.int32, device=y.device)
    for t in range(wide.shape[1]):
        acc = torch.where(wide[:, t], acc + 1, torch.zeros_like(acc))
        run[:, t] = acc
    keep = torch.zeros_like(wide)
    acc = torch.zeros_like(acc)
    for t in range(wide.shape[1] - 1, -1, -1):
        acc = torch.where(wide[:, t], torch.maximum(acc, run[:, t]),
                          torch.zeros_like(acc))
        keep[:, t] = acc >= min_frames
    tracked = torch.zeros_like(mag, dtype=torch.bool)
    tracked[:, f0:f1] = keep & peaks
    return tracked


def envelope_flatness(mag, tracked):
    """Per tracked bin-row: 1 - normalized decay. Piano partials decay
    (low flatness); organ partials sustain (high). Returns per-bin
    flatness for rows that have tracked content."""
    rows = tracked.any(1)
    out = {}
    for fb in torch.nonzero(rows)[:, 0].tolist():
        ts = torch.nonzero(tracked[fb])[:, 0]
        e = mag[fb, ts]
        if len(e) < 8:
            continue
        # flatness = late-energy / early-energy (sustain ~1, decay << 1)
        k = max(len(e) // 4, 1)
        out[fb] = float(e[-k:].mean() / e[:k].mean().clamp_min(1e-9))
    return out


def e9_missing_mask(y, w0, wlen, flat_thresh, min_frames=20, prom=4.0,
                    dilate_f=1):
    """(F,T) bool of TF bins declared MISSING: tracked sinusoids whose
    envelope flatness exceeds the train-calibrated piano threshold."""
    mag = stft(y).abs()
    tracked = sinusoid_tracks(y, w0, wlen, min_frames, prom)
    flat = envelope_flatness(mag, tracked)
    miss = torch.zeros_like(tracked)
    for fb, fl in flat.items():
        if fl > flat_thresh:
            lo, hi = max(fb - dilate_f, 0), min(fb + dilate_f + 1,
                                                miss.shape[0])
            miss[lo:hi] |= tracked[fb][None].expand(hi - lo, -1)
    return miss
