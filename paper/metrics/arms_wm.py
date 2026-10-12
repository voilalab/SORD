"""Arm loaders for the watermark table (paper tab:wm): b2r, prwd (64^2) and ffhqwm0, ffhqwmU0, imagenetwm (native 256^2).
Contract: see paper/metrics/ssim_lpips_core.py. Every loader returns dict(gt, y, M, x, project, psnr_ref) on the SAME images
for a bench (b2r: the 99 images of the minDr run mapped to the bench by x_A match; 256 benches: chunk rows mapped to the
25-image bench by x_A match, overlapping chunk tails de-duplicated).  Verified 2026-09-15 with `python paper/metrics/arms_wm.py`.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np
from PIL import Image
REPO = next(p for p in Path(__file__).resolve().parents if (p / "sord").is_dir()); sys.path.insert(0, str(REPO))
X = REPO / "experiments/extbench/results"; DPSR = REPO / "external/dps/results"

def _png(p, size=None):
    im = Image.open(p).convert("RGB")
    if size: im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, np.float64) / 255.

def _down64(a):
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((64, 64), Image.BILINEAR), np.float64) / 255.

def _chunks(bench, name):
    return sorted(glob.glob(str(X / f"{bench}/{name}/c*/*.npz")), key=lambda s: int(Path(s).parent.name[1:]))

def _cat(fs, key):
    return np.concatenate([np.load(f)[key] for f in fs])

def _match(bench_xA, xA):
    """row index into bench for each row of xA (exact match), de-duplicated in order of first appearance."""
    n = bench_xA.shape[0]
    idx = np.array([int(np.argmin(np.abs(bench_xA - x).reshape(n, -1).mean(1))) for x in xA])
    assert np.abs(bench_xA[idx] - xA).max() < 1e-4, "x_A match failed"
    seen, keep_rows, keep_idx = set(), [], []
    for r, i in enumerate(idx):
        if i not in seen: seen.add(i); keep_rows.append(r); keep_idx.append(i)
    return np.array(keep_rows), np.array(keep_idx)

# ----------------------------------------------------------------------------------------------------------------------
# b2r (CLWD test logos on CLWD scenes, 64^2, n=99)
_B2R = {}
def _b2r():
    if _B2R: return _B2R
    B = np.load(X / "b2r/bench_b2r.npz")
    fs = _chunks("b2r", "minDr"); xA99 = _cat(fs, "x_A")
    rows, keep = _match(B["x_A"], xA99); assert len(keep) == 99 and (rows == np.arange(99)).all()
    M = B["M_true"][keep].astype(np.float32); M3 = np.repeat(M[..., None], 3, -1)
    _B2R.update(B=B, keep=keep, gt=xA99.astype(np.float32), y=B["y"][keep].astype(np.float32), M=M3, ids=B["ids"][keep])
    return _B2R

def _b2r_arm(name, key):
    """chunked arm on the same 99 images (re-matched by x_A in case chunk order differs)."""
    d = _b2r(); fs = _chunks("b2r", name); xA = _cat(fs, "x_A"); x = _cat(fs, key)
    rows, keep = _match(d["B"]["x_A"], xA)
    pos = {int(i): r for r, i in zip(rows, keep)}
    assert all(int(i) in pos for i in d["keep"]), f"{name}: not all 99 images present"
    return x[[pos[int(i)] for i in d["keep"]]]

def _b2r_base(x, ref, project=False, sub=None):
    d = _b2r(); s = slice(None) if sub is None else sub
    return dict(gt=d["gt"][s], y=d["y"][s], M=d["M"][s], x=np.asarray(x)[s] if sub is not None and len(x) == 99 else np.asarray(x),
                project=project, psnr_ref=ref)

def b2r_input():        return _b2r_base(_b2r()["y"], 22.55)
def b2r_inpaint():      return _b2r_base(_b2r_arm("minDr", "x_inp"), 18.93)
def b2r_lama():
    d = _b2r(); L = np.load(X / "b2r/cert_lama_recon256.npz"); lid = {int(i): k for k, i in enumerate(L["ids"])}
    return _b2r_base(np.stack([_down64(L["rec"][lid[int(i)]]) for i in d["ids"]]), 23.65)
def b2r_dps_inpaint():  return _b2r_base(np.load(X / "dpsg/b2r/dpsg_inf.npz")["x_hat"][_b2r()["keep"]], 17.14, project=True)
def b2r_dps_adaptive(): return _b2r_base(np.load(X / "dpsp/b2r/dpsp_auto.npz")["x_hat"][_b2r()["keep"]], 24.26, project=True)
def b2r_min_blind():    return _b2r_base(_b2r_arm("minDr", "x_alt"), 22.56)
def b2r_min_M12():      return _b2r_base(_b2r_arm("minDr_M12", "x_alt"), 23.28)
def b2r_min_blind_alpha_c0():
    """blind per-image alpha + 1 colour, chunk c0 only (n=25)."""
    d = _b2r(); f = _chunks("b2r", "minDr_aG"); z = np.load(f[0])
    rows, keep = _match(d["B"]["x_A"], z["x_A"])
    M3 = np.repeat(d["B"]["M_true"][keep].astype(np.float32)[..., None], 3, -1)
    return dict(gt=z["x_A"][rows].astype(np.float32), y=d["B"]["y"][keep].astype(np.float32), M=M3, x=z["x_alt"][rows], project=False, psnr_ref=23.77)
def b2r_min_kA():       return _b2r_base(_b2r_arm("minDr_kA", "x_alt"), 27.23)
def b2r_min_kA_pp():
    from tango_alt.nstep_alpha import n_step_alpha
    d = _b2r(); al = np.load(X / "b2r/alpha_b2r.npz")["alpha"][d["keep"]]
    nh, _ = n_step_alpha(d["y"], _b2r_arm("minDr_kA", "x_m6"), d["M"], 0., 1., mode="known", alpha_known=al, fit_scale=True,
                         known_K="pp", pp_shrink=0.02, fuse_sigma=0.05)
    return _b2r_base(np.clip(d["y"] - nh, 0, 1), 28.06)
def b2r_slbr():         return _b2r_base(np.stack([_png(REPO / f"experiments/extbench/slbr/results_b2/{int(i)}.png", 64) for i in _b2r()["ids"]]), 33.79)

# ----------------------------------------------------------------------------------------------------------------------
# prwd (real-world PRWD watermarks, 64^2, n=100, EDM2 prior fine-tuned on PRWD clean train)
_PRWD = {}
def _prwd():
    if _PRWD: return _PRWD
    B = np.load(X / "prwd/bench_prwd.npz"); fs = _chunks("prwd", "minE"); xA = _cat(fs, "x_A")
    rows, keep = _match(B["x_A"], xA); assert len(keep) == 100 and (keep == np.arange(100)).all() and (rows == np.arange(100)).all()
    M3 = np.repeat(B["M_true"].astype(np.float32)[..., None], 3, -1)
    _PRWD.update(B=B, gt=B["x_A"].astype(np.float32), y=B["y"].astype(np.float32), M=M3, ids=B["ids"])
    return _PRWD
def _prwd_arm(name, key):
    d = _prwd(); fs = _chunks("prwd", name); xA = _cat(fs, "x_A"); rows, keep = _match(d["B"]["x_A"], xA)
    assert len(keep) == 100; pos = {int(i): r for r, i in zip(rows, keep)}
    return _cat(fs, key)[[pos[i] for i in range(100)]]
def _prwd_base(x, ref, project=False):
    d = _prwd(); return dict(gt=d["gt"], y=d["y"], M=d["M"], x=np.asarray(x), project=project, psnr_ref=ref)

def prwd_input():        return _prwd_base(_prwd()["y"], 19.66)
def prwd_inpaint():      return _prwd_base(_prwd_arm("minE", "x_inp"), 21.64)
def prwd_lama():
    d = _prwd(); L = np.load(X / "prwd/cert_lama_recon256.npz"); lid = {int(i): k for k, i in enumerate(L["ids"])}
    return _prwd_base(np.stack([_down64(L["rec"][lid[int(i)]]) for i in d["ids"]]), 25.88)
def prwd_dps_zero_mean(): return _prwd_base(np.load(X / "dpsp/prwd/dpsp_mu0.npz")["x_hat"], 19.99, project=True)   # table "DPS, same prior"
def prwd_dps_adaptive():  return _prwd_base(np.load(X / "dpsp/prwd/dpsp_auto.npz")["x_hat"], 22.85, project=True)
def prwd_min_blind():     return _prwd_base(_prwd_arm("minE", "x_blind"), 23.73)   # blind-stop readout (x_alt = 23.81)
def prwd_min_M12_m8():    return _prwd_base(_prwd_arm("minE_M12", "x_m8"), 24.14)
def prwd_min_kA():        return _prwd_base(_prwd_arm("minE_kA", "x_alt"), 29.40)
def prwd_min_kA_pp():
    from tango_alt.nstep_alpha import n_step_alpha
    d = _prwd(); al = np.load(X / "prwd/alpha_prwd.npz")["alpha"]
    nh, _ = n_step_alpha(d["y"], _prwd_arm("minE_kA", "x_m6"), d["M"], 0., 1., mode="known", alpha_known=al, fit_scale=True,
                         known_K="pp", pp_shrink=0.02, fuse_sigma=0.0)
    return _prwd_base(np.clip(d["y"] - nh, 0, 1), 30.75)

# ----------------------------------------------------------------------------------------------------------------------
# native 256^2 benches: ffhqwm0 (held-out FFHQ, 12 CLWD logos), ffhqwmU0 (40 unseen logos), imagenetwm (Imagenette)
_N = {}
def _nat(bench, run):
    key = (bench, run)
    if key in _N: return _N[key]
    B = np.load(X / f"{bench}_native/bench_{bench}_native.npz")
    fs = sorted(glob.glob(str(X / f"{bench}_native/{run}/c*/*.npz")), key=lambda s: int(Path(s).parent.name[1:]))
    xA = _cat(fs, "x_A"); rows, keep = _match(B["x_A"], xA)
    assert len(keep) == 25 and sorted(keep.tolist()) == list(range(25)), f"{bench}: chunks cover {sorted(keep.tolist())}"
    order = np.argsort(keep); rows = rows[order]                         # bench order 0..24
    M3 = np.repeat(B["M_true"].astype(np.float32)[..., None], 3, -1)
    d = dict(B=B, fs=fs, rows=rows, gt=B["x_A"].astype(np.float32), y=B["y"].astype(np.float32), M=M3)
    _N[key] = d; return d
def _nat_arm(bench, run, key):
    d = _nat(bench, run); return _cat(d["fs"], key)[d["rows"]]
def _nat_base(bench, run, x, ref, project=False):
    d = _nat(bench, run); return dict(gt=d["gt"], y=d["y"], M=d["M"], x=np.asarray(x), project=project, psnr_ref=ref)
def _dps256(resdir):
    fs = sorted(glob.glob(str(DPSR / resdir / "recon_stack_c*.npz")), key=lambda s: int(Path(s).stem.split("_c")[-1]))
    out = []
    for f in fs:
        D = np.load(f); out += [D[k] for k in sorted(D.files, key=lambda k: int(k[1:]))]
    assert len(out) == 25; return np.stack(out)
def _pngs(dirp, n=25, size=None): return np.stack([_png(Path(dirp) / f"{i}.png", size) for i in range(n)])

RUN = dict(ffhqwm0="min256", ffhqwmU0="min256", imagenetwm="min256in")
DPSDIR = dict(ffhqwm0="ffhqwm0", ffhqwmU0="ffhqwmU0", imagenetwm="imagenetwm_imagenet")
REF = dict(ffhqwm0=dict(input=13.49, inpaint=28.52, dps=27.13, m1=27.80, slbr=24.84),
           ffhqwmU0=dict(input=14.15, inpaint=25.26, dps=23.51, m1=25.20, slbr=26.67, slbr_scratch=27.82),
           imagenetwm=dict(input=13.64, inpaint=26.58, dps=24.80, m1=26.43, slbr=24.47, slbr_scratch=21.66))
SLBR = dict(ffhqwm0=X / "ffhqwm0/slbr", ffhqwmU0=X / "ffhqwmU0/slbr_official", imagenetwm=X / "imagenetwm_native/slbr")
SLBR_SCRATCH = dict(ffhqwmU0=X / "ffhqwmU0/slbr_scratch", imagenetwm=X / "imagenetwm_native/slbr_inscratch")

def nat_input(b):   return _nat_base(b, RUN[b], _nat(b, RUN[b])["y"], REF[b]["input"])
def nat_inpaint(b): return _nat_base(b, RUN[b], _nat_arm(b, RUN[b], "x_inp"), REF[b]["inpaint"])
def nat_dps(b):     return _nat_base(b, RUN[b], _dps256(DPSDIR[b]), REF[b]["dps"], project=True)
def nat_m1(b):      return _nat_base(b, RUN[b], _nat_arm(b, RUN[b], "x_m1"), REF[b]["m1"])
def nat_slbr(b):    return _nat_base(b, RUN[b], _pngs(SLBR[b]), REF[b]["slbr"])
def nat_slbr_scratch(b): return _nat_base(b, RUN[b], _pngs(SLBR_SCRATCH[b]), REF[b]["slbr_scratch"])

# ----------------------------------------------------------------------------------------------------------------------
T = "wm"
# declared-alpha in-loop pp readout at native 256 (kApp runs: K16 M4, --nstep alpha known pp; 2026-09-15 U0 added)
KAPP_REF = dict(ffhqwm0=30.05, ffhqwmU0=26.76, imagenetwm=28.01)
def nat_kApp(b): return _nat_base(b, "kApp", _nat_arm(b, "kApp", "x_alt"), KAPP_REF[b])

ENTRIES = [
    dict(table=T, bench="b2r", arm="input y", loader=b2r_input),
    dict(table=T, bench="b2r", arm="exact-mask inpaint, same prior", loader=b2r_inpaint),
    dict(table=T, bench="b2r", arm="big-LaMa inpaint (supervised), rescored @64", loader=b2r_lama),
    dict(table=T, bench="b2r", arm="DPS, same prior, exact mask", loader=b2r_dps_inpaint),
    dict(table=T, bench="b2r", arm="adaptive DPS, oracle mean", loader=b2r_dps_adaptive),
    dict(table=T, bench="b2r", arm="TANGO-Min blind n", loader=b2r_min_blind),
    dict(table=T, bench="b2r", arm="TANGO-Min, M=12", loader=b2r_min_M12),
    dict(table=T, bench="b2r", arm="TANGO-Min, blind per-image alpha (c0, n=25)", loader=b2r_min_blind_alpha_c0),
    dict(table=T, bench="b2r", arm="TANGO-Min, declared alpha, K=1 colour", loader=b2r_min_kA),
    dict(table=T, bench="b2r", arm="+ per-pixel colour readout", loader=b2r_min_kA_pp),
    dict(table=T, bench="b2r", arm="SLBR official (CLWD train)", loader=b2r_slbr),
    dict(table=T, bench="prwd", arm="input y", loader=prwd_input),
    dict(table=T, bench="prwd", arm="exact-mask inpaint, same prior", loader=prwd_inpaint),
    dict(table=T, bench="prwd", arm="big-LaMa inpaint (supervised), rescored @64", loader=prwd_lama),
    dict(table=T, bench="prwd", arm="DPS, same prior, exact mask", loader=prwd_dps_zero_mean),
    dict(table=T, bench="prwd", arm="adaptive DPS, oracle mean", loader=prwd_dps_adaptive),
    dict(table=T, bench="prwd", arm="TANGO-Min blind n", loader=prwd_min_blind),
    dict(table=T, bench="prwd", arm="TANGO-Min, M=12 (best m8)", loader=prwd_min_M12_m8),
    dict(table=T, bench="prwd", arm="TANGO-Min, declared alpha, K=1 colour", loader=prwd_min_kA),
    dict(table=T, bench="prwd", arm="+ per-pixel colour readout", loader=prwd_min_kA_pp),
]
ENTRIES += [
    dict(table="wm", bench="ffhqwm0", arm="declared alpha, per-pixel readout (in-loop, 256)", loader=lambda: nat_kApp("ffhqwm0")),
    dict(table="wm", bench="ffhqwmU0", arm="declared alpha, per-pixel readout (in-loop, 256)", loader=lambda: nat_kApp("ffhqwmU0")),
    dict(table="wm", bench="imagenetwm", arm="declared alpha, per-pixel readout (in-loop, 256)", loader=lambda: nat_kApp("imagenetwm")),
]

for _b in ("ffhqwm0", "ffhqwmU0", "imagenetwm"):
    ENTRIES += [
        dict(table=T, bench=_b, arm="input y", loader=(lambda b=_b: nat_input(b))),
        dict(table=T, bench=_b, arm="exact-mask inpaint, same prior", loader=(lambda b=_b: nat_inpaint(b))),
        dict(table=T, bench=_b, arm="DPS, same prior, exact mask", loader=(lambda b=_b: nat_dps(b))),
        dict(table=T, bench=_b, arm="TANGO-Min blind n (m1)", loader=(lambda b=_b: nat_m1(b))),
        *([dict(table=T, bench=_b, arm="SLBR official (CLWD train)", loader=(lambda b=_b: nat_slbr(b)))] if _b == "ffhqwmU0" else []),  # 2026-09-18: ffhqwm0/imagenetwm logos are seen by SLBR -> not reported
    ]
    if _b in SLBR_SCRATCH:
        if _b == "ffhqwmU0":  # 2026-09-18: imagenetwm scratch model trained on 140 of the 160 CLWD train logos (seen)
            ENTRIES.append(dict(table=T, bench=_b, arm="SLBR from scratch, proper split", loader=(lambda b=_b: nat_slbr_scratch(b))))

if __name__ == "__main__":
    def psnr_om(a, b, m):
        e = ((np.clip(a, 0, 1) - b) ** 2)[m > 0.5].mean(); return 10 * np.log10(1 / max(e, 1e-20))
    print(f"{'bench':11s} {'arm':46s} {'n':>3s} {'psnr':>7s} {'table':>7s} {'delta':>6s}")
    for e in ENTRIES:
        try:
            d = e["loader"](); x = d["x"]; M = d["M"]
            if d.get("project"): x = np.where(M > 0.5, x, d["y"])
            p = np.array([psnr_om(x[i], d["gt"][i], M[i]) for i in range(len(x))])
            flag = "" if abs(p.mean() - d["psnr_ref"]) < 0.1 else ("  <-- check" if abs(p.mean() - d["psnr_ref"]) < 0.25 else "  <-- MISMATCH")
            print(f"{e['bench']:11s} {e['arm']:46s} {len(x):3d} {p.mean():7.2f} {d['psnr_ref']:7.2f} {p.mean()-d['psnr_ref']:+6.2f} (worst {p.min():.2f}){flag}")
        except Exception as ex:
            print(f"{e['bench']:11s} {e['arm']:46s} ERROR {type(ex).__name__}: {ex}")
