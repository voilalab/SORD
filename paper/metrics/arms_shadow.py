"""Arm loaders for tab:b5s (ISTD 64^2), tab:shadow-transfer (native 256^2, v1 + v2) and tab:cert
(exact-mask inpaint certification) -- SSIM/LPIPS registry, see ssim_lpips_core.py for the contract.
Every loader returns dict(gt, y, M, x, project, psnr_ref). Verified 2026-09-15 (python paper/metrics/arms_shadow.py).
"""
from __future__ import annotations
import glob, json, os
from pathlib import Path
import numpy as np
from PIL import Image

REPO = next(p for p in Path(__file__).resolve().parents if (p / "sord").is_dir()); R = REPO / "experiments/extbench/results"
SD = REPO / "experiments/extbench/shadowdiffusion"; SF = REPO / "experiments/extbench/shadowformer"
DPSR = REPO / "external/dps/results"

def _cat(pattern, key):
    fs = sorted(glob.glob(pattern), key=lambda s: int(Path(s).parent.name[1:]))
    assert fs, pattern
    return np.concatenate([np.load(f)[key] for f in fs]).astype(np.float32)

def _png(p, size=None):
    im = Image.open(p).convert("RGB")
    if size is not None and im.size != (size, size): im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.

# ------------------------------------------------------------------ b5s (ISTD, 64^2, n=100)
def _b5s():
    B = np.load(R / "b5s/bench_b5s.npz")
    return dict(gt=B["x_A"], y=B["y"], M=np.repeat(B["M_true"][..., None], 3, -1).astype(np.float32), stems=B["stems"])

def b5s_arm(run, key, ref, project=False):
    def f():
        b = _b5s(); x = _cat(str(R / f"b5s/{run}/c*/*.npz"), key)
        assert np.abs(_cat(str(R / f"b5s/{run}/c*/*.npz"), "x_A") - b["gt"]).max() < 1e-5
        return dict(gt=b["gt"], y=b["y"], M=b["M"], x=x, project=project, psnr_ref=ref)
    return f

def b5s_input():
    b = _b5s(); return dict(gt=b["gt"], y=b["y"], M=b["M"], x=b["y"], project=False, psnr_ref=14.48)

def b5s_dps_inpaint():
    b = _b5s(); return dict(gt=b["gt"], y=b["y"], M=b["M"], x=np.load(R / "b5s/dps_init.npz")["x_init"], project=True, psnr_ref=27.44)

def b5s_dps_adaptive():
    b = _b5s(); return dict(gt=b["gt"], y=b["y"], M=b["M"], x=np.load(R / "dpsp/b5s/dpsp_auto.npz")["x_hat"], project=True, psnr_ref=27.60)

def b5s_sd():
    b = _b5s(); # fair protocol (2026-09-18): recomposed input (== our bench y) + ckpt fine-tuned on recomposed train; raw-input published run = results_b5s (30.60 / 17.81)
    x = np.stack([_png(SD / "results_b5s/sr" / s, 64) for s in b["stems"]])   # untouched raw input (user rule 2026-09-18); recomposed-input run results_b5s_recompft = 30.78 / 17.31, not used
    return dict(gt=b["gt"], y=b["y"], M=b["M"], x=x, project=False, psnr_ref=30.60)

def b5s_sf():
    b = _b5s(); # fair protocol (2026-09-18): official ckpt on the recomposed input; raw-input run = results_b5s (30.49 / 17.33)
    x = np.stack([_png(SF / "results_b5s" / s, 64) for s in b["stems"]])   # untouched raw input; recomposed-input run results_b5s_recomp = 30.73 / 18.61, not used
    return dict(gt=b["gt"], y=b["y"], M=b["M"], x=x, project=False, psnr_ref=30.49)

# ------------------------------------------------------------------ shadow transfer (native 256^2, n=25)
_SUF = dict(ffhqsh0="g", ffhqsh="g", imagenetsh="in", lsunsh="lb")
_DPSDIR = dict(ffhqsh0="ffhqsh0_ffhq", ffhqsh="ffhqsh_ffhq", imagenetsh="imagenetsh_imagenet", lsunsh="lsunsh_lsun")
_SDNAME = dict(ffhqsh0="results_ffhqsh0", ffhqsh="results_ffhqsh", imagenetsh="results_imagenetsh", lsunsh="results_lsunsh")

def _sh(bench, v2=False):
    tag = bench + ("v2" if v2 else "")
    B = np.load(R / f"{tag}_native/bench_{tag}_native.npz")
    return tag, dict(gt=B["x_A"], y=B["y"], M=np.repeat(B["M_true"][..., None], 3, -1).astype(np.float32))

def sh_arm(bench, run, key, ref, v2=False):
    def f():
        tag, b = _sh(bench, v2); pat = str(R / f"{tag}_native/{run}/c*/*.npz")
        assert np.abs(_cat(pat, "x_A") - b["gt"]).max() < 1e-5
        return dict(**b, x=_cat(pat, key), project=False, psnr_ref=ref)
    return f

def sh_input(bench, ref, v2=False):
    def f():
        _, b = _sh(bench, v2); return dict(**b, x=b["y"], project=False, psnr_ref=ref)
    return f

def sh_dps(bench, ref, v2=False):
    def f():
        tag, b = _sh(bench, v2)
        if not v2:
            x = np.load(R / f"{tag}_native/dps_init.npz")["x_init"]
        else:
            z = np.load(DPSR / (_DPSDIR[bench].replace(bench, tag)) / "recon_stack_c0.npz")
            x = np.stack([z[f"r{i}"] for i in range(len(b["gt"]))]).astype(np.float32)
            if x.max() > 1.5: x = x / 255.
        return dict(**b, x=x, project=True, psnr_ref=ref)
    return f

def sh_sd(bench, ref, v2=False):
    def f():
        tag, b = _sh(bench, v2); d = SD / (_SDNAME[bench] + ("v2" if v2 else "")) / "sr"
        x = np.stack([_png(d / f"{i:03d}.png", 256) for i in range(len(b["gt"]))])
        return dict(**b, x=x, project=False, psnr_ref=ref)
    return f

# ------------------------------------------------------------------ certification (tab:cert)
def cert_x_inp(bench_npz, run_glob, ref, sub=None):
    """x_inp of an exact-mask run = inpaint of the clean image (y == x off Omega)."""
    def f():
        fs = sorted(glob.glob(run_glob), key=lambda s: int(Path(s).parent.name[1:]))
        assert fs, run_glob
        xA = np.concatenate([np.load(g)["x_A"] for g in fs]); y = np.concatenate([np.load(g)["y"] for g in fs])
        M = np.concatenate([np.load(g)["M3"] for g in fs]); x = np.concatenate([np.load(g)["x_inp"] for g in fs])
        return dict(gt=xA, y=y, M=M, x=x, project=False, psnr_ref=ref)
    return f

def cert_lama(bench_dir, lama_npz, ref):
    def f():
        B = np.load(R / bench_dir); L = np.load(R / lama_npz)
        idx = {int(i): k for k, i in enumerate(B["ids"])}; keep = np.array([idx[int(i)] for i in L["ids"]])
        M = np.repeat(B["M_true"][keep][..., None], 3, -1).astype(np.float32)
        return dict(gt=B["x_A"][keep], y=B["y"][keep], M=M, x=L["rec"], project=False, psnr_ref=ref)
    return f

def _E(table, bench, arm, loader): return dict(table=table, bench=bench, arm=arm, loader=loader)

ENTRIES = [
    # ---------------- tab:b5s
    _E("b5s", "b5s", "input y", b5s_input),
    _E("b5s", "b5s", "DPS-inpaint, same prior", b5s_dps_inpaint),
    _E("b5s", "b5s", "adaptive DPS, oracle mu", b5s_dps_adaptive),
    _E("b5s", "b5s", "K-chain inpaint (ours m0)", b5s_arm("multminEc", "x_inp", None)),
    _E("b5s", "b5s", "TANGO-Min additive", b5s_arm("minEc", "x_alt", 28.77)),
    _E("b5s", "b5s", "TANGO-Min multiplicative", b5s_arm("multminEc", "x_alt", 29.10)),
    _E("b5s", "b5s", "full loop multiplicative (m2)", b5s_arm("exactHmEc", "x_m2", 29.25)),
    _E("b5s", "b5s", "TANGO-Min mult + DPS init", b5s_arm("multminEcD", "x_alt", 28.81)),
    _E("b5s", "b5s", "ShadowDiffusion (supervised)", b5s_sd),
    _E("b5s", "b5s", "ShadowFormer (supervised)", b5s_sf),
]
_V1 = dict(ffhqsh0=(14.89, 17.39, 18.90, 20.34, 20.61, 19.15), ffhqsh=(13.98, 17.01, 17.14, 19.12, 20.06, 18.06),
           imagenetsh=(14.30, 15.97, 15.80, 19.05, 20.21, 18.57), lsunsh=(12.78, 18.15, 16.08, 18.12, 20.87, 19.51))
_V2 = dict(ffhqsh0=(19.44, 17.42, 19.14), ffhqsh=(19.43, 18.50, 17.82), imagenetsh=(17.67, 16.04, 18.22), lsunsh=(17.33, 18.28, 17.72))
for b, (i_, d_, a_, m_, t_, s_) in _V1.items():
    s = _SUF[b]
    ENTRIES += [_E("shadow-transfer-v1", b, "input y", sh_input(b, i_)),
                _E("shadow-transfer-v1", b, "DPS, same prior", sh_dps(b, d_)),
                _E("shadow-transfer-v1", b, "TANGO-Min additive m1", sh_arm(b, f"min256{s}", "x_m1", a_)),
                _E("shadow-transfer-v1", b, "TANGO-Min multiplicative gate", sh_arm(b, f"mult256{s}", "x_alt", m_)),
                _E("shadow-transfer-v1", b, "TANGO + DPS init", sh_arm(b, f"mult256{s}D", "x_alt", t_)),
                _E("shadow-transfer-v1", b, "ShadowDiffusion (supervised)", sh_sd(b, s_))]
for b, (m_, d_, s_) in _V2.items():
    s = _SUF[b]
    ENTRIES += [_E("shadow-transfer-v2", b, "input y", sh_input(b, None, v2=True)),
                _E("shadow-transfer-v2", b, "TANGO-Min multiplicative gate", sh_arm(b, f"mult256{s}", "x_alt", m_, v2=True)),
                _E("shadow-transfer-v2", b, "DPS, same prior", sh_dps(b, d_, v2=True)),
                _E("shadow-transfer-v2", b, "ShadowDiffusion (supervised)", sh_sd(b, s_, v2=True))]
ENTRIES += [
    _E("cert", "b2r (c0, n=25)", "EDM2-img64 pretrained, no fine-tune", cert_x_inp("b2r", str(R / "b2r/certPre/c*/*.npz"), 18.26)),
    _E("cert", "b2r (c0, n=25)", "EDM2 fine-tuned on CLWD", cert_x_inp("b2r", str(R / "b2r/minDr/c0/*.npz"), 18.48)),
    _E("cert", "b2r (n=99)", "EDM2 fine-tuned on CLWD (full bench)", cert_x_inp("b2r", str(R / "b2r/minDr/c*/*.npz"), 18.93)),
    _E("cert", "b2r256", "ADM ImageNet-256", cert_x_inp("b2r256", str(R / "b2r256/minG/c*/*.npz"), 18.04)),
    _E("cert", "b2r256", "big-LaMa (supervised inpainter)", cert_lama("b2r256/bench_b2r256.npz", "b2r/cert_lama_recon256.npz", 18.96)),
    _E("cert", "prwd (c0, n=25)", "EDM2-img64 pretrained, no fine-tune", cert_x_inp("prwd", str(R / "prwd/certPre/c*/*.npz"), 21.88)),
    _E("cert", "prwd (c0, n=25)", "EDM2 fine-tuned on PRWD", cert_x_inp("prwd", str(R / "prwd/minE/c0/*.npz"), 22.13)),
    _E("cert", "prwd (n=100)", "EDM2 fine-tuned on PRWD (full bench)", cert_x_inp("prwd", str(R / "prwd/minE/c*/*.npz"), 21.64)),
    _E("cert", "prwd256", "big-LaMa", cert_lama("prwd256/bench_prwd256.npz", "prwd/cert_lama_recon256.npz", 20.93)),
    _E("cert", "ffhqwm (native 256)", "ADM FFHQ on faces", cert_x_inp("ffhqwm", str(R / "ffhqwm_native/min256/c*/*.npz"), 28.60)),
    _E("cert", "ffhqwm (64^2)", "ADM FFHQ on faces, 64^2 pipeline", cert_x_inp("ffhqwm", str(R / "ffhqwm/min/c*/*.npz"), None)),
]

if __name__ == "__main__":
    import sys; sys.path.insert(0, str(REPO / "paper/metrics"))
    def psnr_om(a, b, m):
        e = ((np.clip(a, 0, 1) - b) ** 2)[m > 0.5].mean(); return 10 * np.log10(1 / max(e, 1e-20))
    print(f"{'table':20s} {'bench':16s} {'arm':40s} {'n':>4s} {'psnr':>7s} {'worst':>7s} {'ref':>7s} {'d':>6s}")
    for e in ENTRIES:
        try:
            d = e["loader"](); x = np.clip(d["x"].astype(np.float32), 0, 1); gt = np.clip(d["gt"], 0, 1); M = d["M"]
            if d.get("project"): x = np.where(M > 0.5, x, np.clip(d["y"], 0, 1))
            ps = np.array([psnr_om(x[i], gt[i], M[i]) for i in range(len(gt))])
            ref = d.get("psnr_ref"); dd = "" if ref is None else f"{ps.mean()-ref:+6.2f}"
            print(f"{e['table']:20s} {e['bench']:16s} {e['arm']:40s} {len(gt):4d} {ps.mean():7.2f} {ps.min():7.2f} {'' if ref is None else f'{ref:7.2f}':>7s} {dd:>6s}")
        except Exception as ex:
            print(f"{e['table']:20s} {e['bench']:16s} {e['arm']:40s} FAIL {type(ex).__name__}: {str(ex)[:80]}")
