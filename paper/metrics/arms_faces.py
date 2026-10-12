"""Faces table arms (natural / mnist / cifar; CelebA prior, 64^2, n=100) -> ENTRIES for ssim_lpips_core.

Every row of the combined faces table (paper tab:faces + cifar rows of paper_findings.md 2026-09-08/09
+ tango_last_attempt/tables/sgm_masked.md). Loader conventions:
  * ours arms: chunked r5 runs (c0/c25/c50/c75), file picked by a glob pattern; keys x_A, y, M3, x_inp, x_alt, x_m<k>.
  * DPS arms: extbench/results/dps{g,p}/<bench>/*.npz key x_hat, raw posterior samples -> project=True.
  * SGM arms: r5/<bench>/<tag>/c*.npz key x_A (unclipped; core clips).
  * blind-detected-mask rows (paper 22.60 / 27.87, tango1_detect phase3b): only masks were saved, no
    reconstructions -> no entry.
Run `python paper/metrics/arms_faces.py` to print the on-Omega PSNR verification table (no GPU).
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "sord").is_dir())
R5 = REPO / "experiments/ros/results/r5"
EXT = REPO / "experiments/extbench/results"
sys.path.insert(0, str(REPO / "paper/metrics"))

PAT = {  # the per-bench file pattern for the "standard" TANGO-Min config (gg1 w0 noex noharm, no sigma-floor tag)
    "natural": "natural_amp1_K32_M8_*_gg1_w0_noex_noharm_blind.npz",
    "mnist": "mnist_amp1_K32_M6_*_gg1_w0_noex_noharm_blind.npz",
    "cifar": "natural_amp1_K32_M8_*_gg1_w0_noex_noharm_blind.npz",
}
PAT_FULL = {"natural": "natural_amp1_K8_M8_*_blind.npz", "mnist": "mnist_amp1_K8_M6_*_blind.npz", "cifar": "natural_amp1_K8_M8_*_blind.npz"}
PAT_DMN = {"natural": "natural_amp1_K32_M8_*pb-natural-noise*_blind.npz", "mnist": "mnist_amp1_K32_M6_*pb-mnist-rgb*_blind.npz",
           "cifar": "natural_amp1_K32_M8_*pb-natural-noise*_blind.npz"}
XLAST = {"natural": "x_m8", "mnist": "x_m6", "cifar": "x_m8"}


def _chunk_files(d, pat):
    out = []
    for c in sorted(glob.glob(str(d / "c*")), key=lambda s: int(Path(s).name[1:])):
        fs = [f for f in glob.glob(f"{c}/{pat}") if "sign-free" not in f]
        if len(fs) != 1:
            raise FileNotFoundError(f"{c}/{pat}: {len(fs)} matches {fs}")
        out.append(fs[0])
    if len(out) != 4:
        raise FileNotFoundError(f"{d}/{pat}: {len(out)} chunks")
    return out


def _cat(files, key):
    return np.concatenate([np.load(f)[key] for f in files])


def base(bench):
    """Ground truth, y, mask of the 100-image bench (from the standard-config chunk run)."""
    d = R5 / ("audit/base" if bench in ("natural", "mnist") else "cifar/min_exact")
    fs = _chunk_files(d, PAT[bench])
    return dict(gt=_cat(fs, "x_A"), y=_cat(fs, "y"), M=_cat(fs, "M3")), fs


def ours(bench, run, key, pat=None, ref=None):
    def L():
        b, _ = base(bench)
        d = R5 / (f"{bench}/{run}" if bench == "cifar" or run != "vtangoDPS" else "vtangoDPS")
        fs = _chunk_files(d, pat or PAT[bench])
        assert np.abs(_cat(fs, "x_A") - b["gt"]).max() < 1e-5, f"{bench}/{run}: image order mismatch"
        return dict(**b, x=_cat(fs, key), project=False, psnr_ref=ref)
    return L


def ours_base(bench, key, ref):
    def L():
        b, fs = base(bench)
        return dict(**b, x=_cat(fs, key), project=False, psnr_ref=ref)
    return L


def inp(bench, ref):
    def L():
        b, _ = base(bench)
        return dict(**b, x=b["y"], project=False, psnr_ref=ref)
    return L


def dps(bench, fam, tag, ref):
    def L():
        b, _ = base(bench)
        return dict(**b, x=np.load(EXT / f"{fam}/{bench}/{tag}.npz")["x_hat"], project=True, psnr_ref=ref)
    return L


def sgm(bench, tag, ref):
    def L():
        b, _ = base(bench)
        fs = sorted(glob.glob(str(R5 / f"{bench}/{tag}/c*.npz")), key=lambda s: int(Path(s).stem[1:]))
        assert len(fs) == 4, (bench, tag, fs)
        return dict(**b, x=_cat(fs, "x_A"), project=False, psnr_ref=ref)
    return L


# (arm label, per-bench (loader-factory, ref)) ; refs = table on-Omega means
T = {}
def add(arm, natural=None, mnist=None, cifar=None):
    T[arm] = dict(natural=natural, mnist=mnist, cifar=cifar)

# input y: the core clips x to [0,1]; paper tab:faces prints the UNCLIPPED y (7.18 / 7.37), the clipped
# convention (sgm_masked.md / cifar row) is 9.37 / 12.09 / 7.92 -> refs below are the clipped ones.
add("input y", inp("natural", 9.37), inp("mnist", 12.09), inp("cifar", 7.92))
add("DPS-inpaint (sigma->inf)", dps("natural", "dpsg", "dpsg_inf", 16.84), dps("mnist", "dpsg", "dpsg_inf", 19.85), dps("cifar", "dpsg", "dpsg_inf", 17.28))
add("DPS-Gauss oracle sigma", dps("natural", "dpsg", "dpsg_auto", 9.26), dps("mnist", "dpsg", "dpsg_auto", 11.77), dps("cifar", "dpsg", "dpsg_auto", 7.65))
add("adaptive DPS oracle mu", dps("natural", "dpsp", "dpsp_auto", 15.74), dps("mnist", "dpsp", "dpsp_auto", 11.06), dps("cifar", "dpsp", "dpsp_auto", 14.87))
add("adaptive DPS zero mean", dps("natural", "dpsp", "dpsp_mu0", 9.32), dps("mnist", "dpsp", "dpsp_mu0", 11.91), dps("cifar", "dpsp", "dpsp_mu0", 7.83))
add("SGM two priors, no mask", sgm("natural", "sgm", 11.45), sgm("mnist", "sgm", 25.78), sgm("cifar", "sgm_free", 11.66))
add("SGM two priors + mask proj", sgm("natural", "sgmM", 16.96), sgm("mnist", "sgmM", 22.51), sgm("cifar", "sgmM", 11.25))
add("SGM two priors, mask on n only", sgm("natural", "sgmMn", 16.36), sgm("mnist", "sgmMn", 21.83), sgm("cifar", "sgmMn", None))
add("SGM one prior, no mask", sgm("natural", "sgm_celebaN", 13.18), sgm("mnist", "sgm_celebaN", 11.42), sgm("cifar", "sgm_celebaN", 12.33))
add("SGM one prior + mask proj", sgm("natural", "sgmM_celebaN", 14.77), sgm("mnist", "sgmM_celebaN", 15.60), sgm("cifar", "sgmM_celebaN", 13.98))
add("SGM one prior, noise score zeroed, no mask", sgm("natural", "sgm_x0n", 6.79), sgm("mnist", "sgm_x0n", 6.86), sgm("cifar", "sgm_x0n", 6.76))
add("SGM one prior, noise score zeroed + mask", sgm("natural", "sgmM_x0n", 6.69), sgm("mnist", "sgmM_x0n", 7.12), sgm("cifar", "sgmM_x0n", 6.75))
# paper tab:faces "no mask" row = the 2026-08-06 shipped FULL-loop (K8) run with an all-ones mask, r5/<b>/full, x_alt
# (= x_m{M}); cifar has only the TANGO-Min no-mask run (min_nomask, blind readout x_blind = 15.46 in paper_findings).
add("ours, no mask (all-ones Omega) [paper row: full loop K8]", ours("natural", "full", "x_alt", PAT_FULL["natural"], 15.53), ours("mnist", "full", "x_alt", PAT_FULL["mnist"], 19.54), ours("cifar", "min_nomask", "x_blind", ref=15.46))
add("TANGO-Min K32, no mask (blind readout)", ours("natural", "min_nomask", "x_blind", ref=15.72), ours("mnist", "min_nomask", "x_blind", ref=19.57), ours("cifar", "min_nomask", "x_blind", ref=15.46))
add("K-chain inpaint (ours m0)", ours_base("natural", "x_inp", None), ours_base("mnist", "x_inp", None), ours_base("cifar", "x_inp", None))
add("full loop K8", ours("natural", "exact", "x_m8", PAT_FULL["natural"], 21.15), ours("mnist", "exact", "x_m6", PAT_FULL["mnist"], 26.25), ours("cifar", "exact", "x_m8", PAT_FULL["cifar"], 21.19))
add("TANGO-Min K32", ours_base("natural", "x_alt", 21.96), ours_base("mnist", "x_alt", 27.16), ours_base("cifar", "x_alt", 22.05))
add("TANGO-Min + DPS-inpaint init", ours("natural", "vtangoDPS", "x_alt", ref=20.80), ours("mnist", "vtangoDPS", "x_alt", ref=26.26), ours("cifar", "vtangoDPS", "x_alt", ref=20.74))
# sgm_masked.md dmn rows are the BLIND readout (x_blind), not the last iterate.
add("TANGO-Min + matched contaminant DM n-step (blind readout)", ours("natural", "dmn_exact", "x_blind", PAT_DMN["natural"], 22.80), ours("mnist", "dmn_exact", "x_blind", PAT_DMN["mnist"], 21.07), ours("cifar", "dmn_exact", "x_blind", PAT_DMN["cifar"], 21.97))

ENTRIES = [dict(table="faces", bench=b, arm=arm, loader=L) for arm, d in T.items() for b, L in d.items() if L is not None]

if __name__ == "__main__":
    def psnr_om(a, b, m):
        e = ((np.clip(a, 0, 1) - b) ** 2)[m > 0.5].mean(); return 10 * np.log10(1 / max(e, 1e-20))
    print(f"{'arm':45s} {'bench':8s} {'n':>4s} {'psnr_om':>8s} {'table':>7s} {'delta':>6s}")
    bad = []
    for e in ENTRIES:
        try:
            d = e["loader"]()
        except Exception as ex:
            print(f"{e['arm']:45s} {e['bench']:8s} LOAD FAIL: {ex}"); bad.append((e['arm'], e['bench'], str(ex))); continue
        x, gt, M, y = np.asarray(d["x"], np.float32), np.asarray(d["gt"], np.float32), np.asarray(d["M"], np.float32), d["y"]
        if d.get("project"): x = np.where(M > 0.5, x, y)
        ps = np.array([psnr_om(x[i], gt[i], M[i]) for i in range(len(gt))])
        ref = d.get("psnr_ref"); dl = "" if ref is None else f"{ps.mean() - ref:+.2f}"
        flag = "" if ref is None or abs(ps.mean() - ref) <= 0.1 else "  <-- MISMATCH"
        print(f"{e['arm']:45s} {e['bench']:8s} {len(gt):4d} {ps.mean():8.2f} {'' if ref is None else f'{ref:7.2f}':>7s} {dl:>6s}{flag}")
        if flag: bad.append((e['arm'], e['bench'], f"{ps.mean():.2f} vs {ref}"))
    print("\nMISMATCHES / FAILURES:", bad if bad else "none")
