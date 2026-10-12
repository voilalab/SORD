"""Consistency sweep table (2026-09-18): x-step ladder {all, 8 DDIM levels} x init {K-chain inpaint, DPS} on every bench,
paired per image against the shipped arm.  Tolerates missing runs (prints '--').  b2r/prwd cells = declared-alpha K1 loop
x_m6 + the shipped offline per-pixel readout (pp shrink .02; b2r + fuse .05 + fitted opacity scale), verified to
reproduce 28.06 / 30.75.  Writes results/consistency_table_20260918.{md,json} + paper/tables/consistency_sweep.tex."""
import glob, json, sys
from pathlib import Path
import numpy as np
REPO = next(p for p in Path(__file__).resolve().parents if (p / "sord").is_dir()); sys.path.insert(0, str(REPO))
R = REPO / "experiments/extbench/results"; R5 = REPO / "experiments/ros/results/r5"

def chunks(d, pref=None, excl=None, tagsel=None):
    out = []
    for c in sorted(glob.glob(str(d / "c*")), key=lambda p: int(Path(p).name[1:])):
        js = [j for j in sorted(glob.glob(c + "/*.json")) if (pref is None or Path(j).name.startswith(pref)) and (excl is None or excl not in Path(j).name) and (tagsel is None or tagsel(Path(j).name))]
        if js: out.append((int(Path(c).name[1:]), js[0]))
    return out

def alt(d, **kw):
    """per-image 'alt' PSNR placed at bench index (NaN where the chunk has not run), so partial runs pair correctly."""
    js = chunks(d, **kw)
    if not js: return None
    parts = {i0: np.asarray(json.load(open(j))["psnr"]["alt"], float) for i0, j in js}
    n = max(i0 + len(v) for i0, v in parts.items()); out = np.full(max(n, 100 if n > 25 else n), np.nan)
    for i0, v in parts.items(): out[i0:i0 + len(v)] = v
    return out

def psnr(x, gt, m):
    return np.array([10 * np.log10(1 / max(((np.clip(x[i], 0, 1) - gt[i]) ** 2)[m[i] > 0.5].mean(), 1e-20)) for i in range(len(gt))])

def readout(bench, run, fuse, fit_scale):
    """declared-alpha K1 loop x_m6 -> shipped per-pixel readout (paper/metrics/arms_wm.py b2r_min_kA_pp / prwd_min_kA_pp
    convention, incl. the empty-mask drop + chunk-order matching) -> on-Omega PSNR per image (None if run missing)."""
    if not chunks(R / bench / run): return None
    if len(chunks(R / bench / run)) < 4: return None   # readout needs the full bench (arms_wm matching)
    sys.path.insert(0, str(REPO / "paper/metrics")); import arms_wm as A
    from tango_alt.nstep_alpha import n_step_alpha
    if bench == "b2r":
        d = A._b2r(); al = np.load(R / "b2r/alpha_b2r.npz")["alpha"][d["keep"]]; x = A._b2r_arm(run, "x_m6")
    else:
        d = A._prwd(); al = np.load(R / "prwd/alpha_prwd.npz")["alpha"]; x = A._prwd_arm(run, "x_m6")
    nh, _ = n_step_alpha(d["y"], x, d["M"], 0., 1., mode="known", alpha_known=al, fit_scale=fit_scale, known_K="pp", pp_shrink=0.02, fuse_sigma=fuse)
    xr = np.clip(d["y"] - nh, 0, 1); M = np.asarray(d["M"]); M = M if M.ndim == 4 else np.repeat(M[..., None], 3, -1)
    return psnr(xr, d["gt"], M)

nat = lambda f: f.startswith("natural_") and "sign-free" not in f
CELLS = {  # bench -> config -> loader
 "Natural patch on CelebA": dict(ship=lambda: alt(R5 / "audit/base", tagsel=nat), xs8=lambda: alt(R5 / "xs8/natural"), alldps=lambda: alt(R5 / "vtangoDPS", tagsel=nat), xs8dps=lambda: alt(R5 / "xs8dps/natural")),
 "MNIST digit on CelebA": dict(ship=lambda: alt(R5 / "audit/base", pref="mnist_"), xs8=lambda: alt(R5 / "xs8/mnist"), alldps=lambda: alt(R5 / "vtangoDPS", pref="mnist_"), xs8dps=lambda: alt(R5 / "xs8dps/mnist")),
 "CIFAR-10 patch on CelebA": dict(ship=lambda: alt(R5 / "cifar/min_exact", excl="st0"), xs8=lambda: alt(R5 / "xs8/cifar"), alldps=lambda: alt(R5 / "alldps/cifar"), xs8dps=lambda: alt(R5 / "xs8dps/cifar")),
 "CLWD + watermark (b2r, declared alpha + pp readout)": dict(ship=lambda: readout("b2r", "minDr_kA", 0.05, True), xs8=lambda: readout("b2r", "xs8_kA", 0.05, True), alldps=lambda: readout("b2r", "alldps_kA", 0.05, True), xs8dps=lambda: readout("b2r", "xs8dps_kA", 0.05, True)),
 "PRWD + watermark (declared alpha + pp readout)": dict(ship=lambda: readout("prwd", "minE_kA", 0.0, False), xs8=lambda: readout("prwd", "xs8_kA", 0.0, False), alldps=lambda: readout("prwd", "alldps_kA", 0.0, False), xs8dps=lambda: readout("prwd", "xs8dps_kA", 0.0, False)),
 "FFHQ + watermark (ffhqwm0, kApp)": dict(ship=lambda: alt(R / "ffhqwm0_native/kApp"), xs8=lambda: alt(R / "ffhqwm0_native/xs8_kApp"), alldps=lambda: alt(R / "ffhqwm0_native/alldps_kApp"), xs8dps=lambda: alt(R / "ffhqwm0_native/xs8dps_kApp")),
 "FFHQ + watermark, unseen logos (ffhqwmU0, kApp)": dict(ship=lambda: alt(R / "ffhqwmU0_native/kApp"), xs8=lambda: alt(R / "ffhqwmU0_native/xs8_kApp"), alldps=lambda: alt(R / "ffhqwmU0_native/alldps_kApp"), xs8dps=lambda: alt(R / "ffhqwmU0_native/xs8dps_kApp")),
 "ImageNet + watermark (imagenetwm, kApp)": dict(ship=lambda: alt(R / "imagenetwm_native/kApp"), xs8=lambda: alt(R / "imagenetwm_native/xs8_kApp"), alldps=lambda: alt(R / "imagenetwm_native/alldps_kApp"), xs8dps=lambda: alt(R / "imagenetwm_native/xs8dps_kApp")),
 "ISTD shadow (b5s, mult. gated)": dict(ship=lambda: alt(R / "b5s/multminEc"), xs8=lambda: alt(R / "b5s/xs8_mult"), alldps=lambda: alt(R / "b5s/multminEcD"), xs8dps=lambda: alt(R / "b5s/xs8dps_mult")),
 "ISTD shadow + FFHQ (ffhqsh0, mult. gated)": dict(ship=lambda: alt(R / "ffhqsh0_native/mult256g"), xs8=lambda: alt(R / "ffhqsh0_native/xs8_mult256"), alldps=lambda: alt(R / "ffhqsh0_native/mult256gD"), xs8dps=lambda: alt(R / "ffhqsh0_native/xs8dps_mult256")),
 "ISTD shadow + ImageNet (imagenetsh, mult. gated)": dict(ship=lambda: alt(R / "imagenetsh_native/mult256in"), xs8=lambda: alt(R / "imagenetsh_native/xs8_mult256"), alldps=lambda: alt(R / "imagenetsh_native/mult256inD"), xs8dps=lambda: alt(R / "imagenetsh_native/xs8dps_mult256")),
}
CFG = [("ship", "all levels / inpaint init (shipped)"), ("xs8", "8 levels / inpaint init"), ("alldps", "all levels / DPS init"), ("xs8dps", "8 levels / DPS init")]
rows = {}
for bench, L in CELLS.items():
    v = {}
    for k, _ in CFG:
        try: v[k] = L[k]()
        except Exception as e: print(f"[{bench}/{k}] {type(e).__name__}: {e}"); v[k] = None
    rows[bench] = v
# audio (steps axis only; from the ablation-2 arms)
A = REPO / "experiments/tango_msdm_2026_08_04/results/arms/abl2"
for b, lab in (("piano", "Slakh piano (SI-SDR, oracle window)"), ("guitar", "Slakh guitar (SI-SDR, oracle window)")):
    v = {}
    for k, a in (("ship", "base"), ("xs8", "xs8")):
        p = A / b / a / "summary.json"
        v[k] = np.array([c["win_sisdr"] for c in json.load(open(p))["cells"]]) if p.exists() else None
    v["alldps"] = None; v["xs8dps"] = None; rows[lab] = v

def _pair(v, s):
    if v is None or s is None: return None, None
    n = min(len(v), len(s)); ok = ~np.isnan(v[:n]) & ~np.isnan(s[:n]); return v[:n][ok], s[:n][ok]
def cell(v, s):
    if v is None: return "--"
    vv = v[~np.isnan(v)]; txt = f"{vv.mean():.2f} / {vv.min():.2f}"
    if len(vv) < len(v): txt += f" [n={len(vv)}]"
    a, b = _pair(v, s)
    if a is not None and len(a) and v is not s: d = a - b; txt += f" ({d.mean():+.2f}, {(d > 0).sum()}/{len(d)})"
    return txt

L = ["# Consistency sweep: x-step ladder {all, 8 DDIM levels} x init {K-chain inpaint, DPS}; n-step 8 DDIM steps (2026-09-18)", "",
     "Cells: on-Omega PSNR mean / worst (audio: in-window SI-SDR), then (paired delta vs shipped, wins/n). '--' = not run (yet).", "",
     "| Bench | " + " | ".join(l for _, l in CFG) + " |", "|---|" + "---|" * len(CFG)]
J = {}
for bench, v in rows.items():
    s = v["ship"]; L.append(f"| {bench} | " + " | ".join(cell(v[k], s) for k, _ in CFG) + " |")
    J[bench] = {}
    for k, _ in CFG:
        if v[k] is None: J[bench][k] = None; continue
        vv = v[k][~np.isnan(v[k])]; a, b = _pair(v[k], s)
        J[bench][k] = dict(mean=float(vv.mean()), worst=float(vv.min()), n=int(len(vv)), delta=(None if (k == "ship" or a is None or not len(a)) else float((a - b).mean())))
# consistency summary: mean paired delta across benches per config (only benches where the config ran)
L += ["", "## Mean paired delta vs shipped across the benches where each configuration has run", ""]
for k, lab in CFG[1:]:
    ds = [J[b][k]["delta"] for b in J if J[b][k] and J[b][k]["delta"] is not None]
    L.append(f"- {lab}: {np.mean(ds):+.2f} dB over {len(ds)} benches" if ds else f"- {lab}: no runs yet")
txt = "\n".join(L); print(txt)
(REPO / "results/consistency_table_20260918.md").write_text(txt + "\n"); json.dump(J, open(REPO / "results/consistency_table_20260918.json", "w"), indent=1)
T = [r"% consistency sweep (2026-09-18), built by experiments/extbench/build_consistency_table_20260918.py", r"\begin{table*}[t]", r"    \centering",
     r"    \caption{\textbf{One sampler for every bench: solver steps and initialisation.} Shipped \methodmin{} arm of each bench (x-step over every ladder level at or below $\sigma_{\rm start}$, K-chain inpaint initialisation, $n$-step 8 DDIM steps) against the same arm with the x-step ladder capped at 8 DDIM levels and/or initialised from the DPS reconstruction with the same prior. PSNR (dB) over $\Omega$ (audio: in-window SI-SDR), mean / worst, and in brackets the paired difference to the shipped arm and the number of images on which the variant is better.}",
     r"    \label{tab:consistency}", r"    \small", r"    \setlength{\tabcolsep}{3pt}", r"    \begin{tabular}{@{}lcccc@{}}", r"        \toprule",
     r"        \textbf{Bench} & \shortstack{\textbf{all levels}\\\textbf{inpaint init (shipped)}} & \shortstack{\textbf{8 levels}\\\textbf{inpaint init}} & \shortstack{\textbf{all levels}\\\textbf{DPS init}} & \shortstack{\textbf{8 levels}\\\textbf{DPS init}} \\", r"        \midrule"]
for bench, v in rows.items():
    s = v["ship"]; T.append("        " + bench.replace("+", "$+$").replace("_", r"\_") + " & " + " & ".join(cell(v[k], s).replace("--", "--") for k, _ in CFG) + r" \\")
T += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
(REPO / "paper/tables/consistency_sweep.tex").write_text("\n".join(T) + "\n"); print("\nwrote paper/tables/consistency_sweep.tex")
