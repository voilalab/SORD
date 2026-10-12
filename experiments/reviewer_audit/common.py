"""Reviewer audit — shared loaders (2026-09-09)."""
from __future__ import annotations
import glob
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
R5 = REPO / "experiments/ros/results/r5"
EXT = REPO / "experiments/extbench/results"
BENCH_M = dict(natural=8, mnist=6, wm=6, b2r=6, b5s=8, cifar=8)
CLASS = dict(natural="iterate", mnist="iterate", b2r="iterate",
             b5s="stop", b5sEc="stop", wm="stop", cifar="iterate")
# ladder of the 64^2 VE prior (levels <= 0.5, from run_tango CelebAAdapter)
LADDER = [0.098, 0.137, 0.168, 0.196, 0.221, 0.246, 0.269, 0.291, 0.313,
          0.334, 0.356, 0.377, 0.398, 0.419, 0.44, 0.462, 0.483]
NSTEP_NFE = 8      # DDIM steps of the batched n-step


def root(bench):
    if bench == "b5sEc":
        return EXT / "b5s"
    return R5 if bench in ("natural", "mnist", "wm", "cifar") else EXT / bench


def tag_ok(bench, fname):
    """natural / mnist / wm share one results root: pick by tag."""
    if bench == "mnist":
        return fname.startswith("mnist_")
    if bench == "wm":
        return fname.startswith("natural_") and "sign-free" in fname
    if bench == "natural":
        return fname.startswith("natural_") and "sign-free" not in fname
    return True


def chunks(bench, name):
    """Sorted list of (i0, json_path, npz_path) for a run name."""
    out = []
    if bench == "b5sEc":
        name = name.replace("audit/", "auditEc/")
    for d in sorted(glob.glob(str(root(bench) / name / "c*"))):
        js = sorted(glob.glob(d + "/*.json"))
        js = [j for j in js if tag_ok(bench, Path(j).name)]
        if not js:
            continue
        # 2026-09-24: cifar/min_exact/c0 also holds sigma-threshold variants (_st0.012 / _st0.02); prefer the shipped file
        j = ([x for x in js if "_st0." not in Path(x).name] or js)[0]
        out.append((int(Path(d).name[1:]), j, j[:-5] + ".npz"))
    return sorted(out)


def load(bench, name, need_img=True):
    """Concatenate chunks. Returns dict with per-image arrays:
    inpaint (N,), traj (N, M) PSNR of x per iterate, sig (N, M) per-image
    spread read at the start of each iterate (sig[:,0] = init spread,
    sig[:,1] = spread after the first x-step = sigma*_2 ...), sig0 (N,),
    mask_frac, n_rms, nfe (M,) cumulative NFE per chain-batch,
    mean_curves (list of chunk curves), K, M."""
    rows = chunks(bench, name)
    if not rows:
        return None
    inp, alt, box, traj, sig, sig0, mf, nr, ytil, ntraj = ([] for _ in range(10))
    curves_all, K = [], None
    for i0, j, _ in rows:
        d = json.load(open(j))
        K = d["K"]
        inp += d["psnr"]["inpaint"]; alt += d["psnr"]["alt"]
        box += d["psnr"]["box"]
        cur = d["curves"]; curves_all.append(cur)
        if need_img and "psnr_x_img" in cur[0]:
            traj.append(np.array([c["psnr_x_img"] for c in cur]).T)
            sig.append(np.array([c["sigma_star_img"] for c in cur]).T)
            ntraj.append(np.array([c["psnr_n_img"] for c in cur]).T)
            if "psnr_ytilde_img" in cur[0]:
                ytil.append(np.array([c["psnr_ytilde_img"] for c in cur]).T)
            sig0 += d.get("sigma_star0_img", [])
            mf += d.get("mask_frac_img", [])
            nr += d.get("n_rms_img", [])
    out = dict(bench=bench, name=name, K=K, M=len(curves_all[0]),
               inpaint=np.array(inp), alt=np.array(alt), box=np.array(box),
               curves=curves_all, n=len(inp))
    if traj:
        out["traj"] = np.concatenate(traj)
        out["sig"] = np.concatenate(sig)
        out["ntraj"] = np.concatenate(ntraj)
        if ytil:
            out["ytil"] = np.concatenate(ytil)
        out["sig0"] = np.array(sig0); out["mask_frac"] = np.array(mf)
        out["n_rms"] = np.array(nr)
    # chunk-mean trajectories (available for every run, incl. old ones)
    out["mean_traj"] = np.array([[c["psnr_x"] for c in cur]
                                 for cur in curves_all])
    out["mean_sig"] = np.array([[c["sigma_star"] for c in cur]
                                for cur in curves_all])
    out["nfe_x"] = np.array([[c["nfe_extra"] for c in cur]
                             for cur in curves_all]).mean(0)
    if "nfe_n" in curves_all[0][0]:
        out["nfe_n"] = np.array([[c["nfe_n"] for c in cur] for cur in curves_all]).mean(0) / max(out["n"] / len(curves_all), 1)
    return out


def nfe_cum(run):
    """Cumulative NFE per image-batch after iterate m (m=0 = inpaint init).
    init ladder from 0.4: 13 levels x K chains; iterate m: n-step (8) +
    x-step (nfe_extra, already K-folded)."""
    K = run["K"]
    init = 13 * K
    nn = run.get("nfe_n")            # per-iterate n-step NFE logged since 2026-09-14 (batched over images -> per image)
    if nn is None and "mean_sig" in run:     # older runs: effective steps = ladder levels <= sigma*_m (>= 1), capped at 8
        ms = np.asarray(run["mean_sig"]).mean(0)
        nn = [min(NSTEP_NFE, max(1, int(sum(1 for L in LADDER if L <= ms[i] + 1e-9)))) for i in range(len(run["nfe_x"]))]
    per = [(float(nn[i]) if nn is not None and nn[i] > 0 else NSTEP_NFE) + float(v) for i, v in enumerate(run["nfe_x"])]
    return np.concatenate([[init], init + np.cumsum(per)])


def fmt(v):
    return f"{v:.2f}"
