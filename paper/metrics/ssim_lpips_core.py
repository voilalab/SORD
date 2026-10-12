"""SSIM + LPIPS scoring core for the paper result tables (2026-09-15).

Registry contract (filled by table modules `arms_<table>.py`):
    ENTRIES: list of dict(table=str, bench=str, arm=str, loader=callable)
    loader() -> dict(gt=[N,H,W,3] float in [0,1], y=[N,H,W,3], M=[N,H,W,3] or [N,H,W,1] {0,1},
                     x=[N,H,W,3] reconstruction, project=bool (paste y back off-Omega before scoring),
                     psnr_ref=float or None (the paper table on-Omega mean, for a sanity check))
Metrics per image: PSNR on Omega (check), SSIM full-frame (skimage, data_range 1, channel_axis -1),
LPIPS-alex full-frame, plus SSIM/LPIPS on the tight mask bounding box (padded 4 px, min 32 px, resized to
>=64 for LPIPS) so that a local metric exists for the local corruption.
Run:  CUDA_VISIBLE_DEVICES=<g> ~/miniconda3/envs/joint-diffusion/bin/python paper/metrics/run_ssim_lpips.py [--table faces]
"""
from __future__ import annotations
import numpy as np, torch, lpips, json
from skimage.metrics import structural_similarity as ssim_fn

_LP = {}
def _lpips(net="alex"):
    if net not in _LP:
        _LP[net] = lpips.LPIPS(net=net, verbose=False).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return _LP[net]

def psnr_om(a, b, m):
    e = ((np.clip(a, 0, 1) - b) ** 2)[m > 0.5].mean(); return 10 * np.log10(1 / max(e, 1e-20))

def _bbox(m2, pad=4, minsz=32):
    ys, xs = np.where(m2 > 0.5); H, W = m2.shape
    if len(ys) == 0: return 0, H, 0, W
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H), max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
    for lo, hi, L in ((y0, y1, H), (x0, x1, W)):
        pass
    def grow(lo, hi, L):
        while hi - lo < min(minsz, L):
            if lo > 0: lo -= 1
            if hi < L: hi += 1
        return lo, hi
    y0, y1 = grow(y0, y1, H); x0, x1 = grow(x0, x1, W)
    return y0, y1, x0, x1

def _lp_batch(a, b, net="alex"):
    """a,b: [N,H,W,3] in [0,1] -> LPIPS per image (inputs scaled to [-1,1])."""
    net_ = _lpips(net); dev = next(net_.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(a), 32):
            ta = torch.from_numpy(np.ascontiguousarray(a[i:i+32])).permute(0, 3, 1, 2).float().to(dev) * 2 - 1
            tb = torch.from_numpy(np.ascontiguousarray(b[i:i+32])).permute(0, 3, 1, 2).float().to(dev) * 2 - 1
            if ta.shape[-1] < 64:
                ta = torch.nn.functional.interpolate(ta, size=(64, 64), mode="bilinear", align_corners=False)
                tb = torch.nn.functional.interpolate(tb, size=(64, 64), mode="bilinear", align_corners=False)
            out.append(net_(ta, tb).flatten().cpu().numpy())
    return np.concatenate(out)

def score(d):
    gt = np.clip(np.asarray(d["gt"], np.float32), 0, 1); x = np.clip(np.asarray(d["x"], np.float32), 0, 1)
    y = np.clip(np.asarray(d["y"], np.float32), 0, 1); M = np.asarray(d["M"], np.float32)
    if M.ndim == 3: M = M[..., None]
    if M.shape[-1] == 1: M = np.repeat(M, 3, -1)
    if gt.shape[-1] == 1: gt = np.repeat(gt, 3, -1); x = np.repeat(x, 3, -1); y = np.repeat(y, 3, -1)
    if d.get("project", False): x = np.where(M > 0.5, x, y).astype(np.float32)
    N = len(gt)
    ps = np.array([psnr_om(x[i], gt[i], M[i]) for i in range(N)])
    ss = np.array([ssim_fn(gt[i], x[i], data_range=1.0, channel_axis=-1) for i in range(N)])
    lp = _lpips_full = _lp_batch(x, gt)
    # local (bbox) variants
    cs, cl = [], []
    for i in range(N):
        y0, y1, x0, x1 = _bbox(M[i, ..., 0])
        a, b = gt[i, y0:y1, x0:x1], x[i, y0:y1, x0:x1]
        w = min(7, (min(a.shape[:2]) // 2) * 2 - 1)
        cs.append(ssim_fn(a, b, data_range=1.0, channel_axis=-1, win_size=max(w, 3)))
        cl.append(_lp_batch(a[None], b[None])[0])
    cs, cl = np.array(cs), np.array(cl)
    r = dict(n=int(N), psnr_om_mean=float(ps.mean()), psnr_om_worst=float(ps.min()),
             ssim_full_mean=float(ss.mean()), ssim_full_worst=float(ss.min()),
             lpips_full_mean=float(lp.mean()), lpips_full_worst=float(lp.max()),
             ssim_box_mean=float(cs.mean()), ssim_box_worst=float(cs.min()),
             lpips_box_mean=float(cl.mean()), lpips_box_worst=float(cl.max()))
    if d.get("psnr_ref") is not None:
        r["psnr_ref"] = float(d["psnr_ref"]); r["psnr_delta_vs_table"] = float(ps.mean() - d["psnr_ref"])
    return r
