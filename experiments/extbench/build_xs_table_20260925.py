"""x-step DDIM level sweep table (user 2026-09-25): shipped (all levels) vs 8 vs 4 DDIM levels on every bench, paired per image,
with the per-image network-evaluation count of each arm (13K init + per iterate: logged x-step nfe_extra + n-step DDIM steps
<= 8 counted as ladder levels <= sigma*; declared-alpha arms: no DM n-step). Reuses the loaders of
build_consistency_table_20260918.py. Writes results/xs_table_20260925.{md,json} + paper/tables/xs_levels_sweep.tex."""
import glob, json, sys, re
from pathlib import Path
import numpy as np
REPO = next(p for p in Path(__file__).resolve().parents if (p / "sord").is_dir()); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/reviewer_audit"))
src = (REPO / "experiments/extbench/build_consistency_table_20260918.py").read_text().split("\nCFG = ")[0]
exec(compile(src, "build_consistency_table_20260918.py", "exec"))     # chunks, alt, psnr, readout, nat, CELLS, R, R5
from common import LADDER, NSTEP_NFE
XS = {
 "Natural patch on CelebA": (lambda: alt(R5 / "xs4/natural"), {"ship": (R5 / "audit/base", True, "last", nat), "xs8": (R5 / "xs8/natural", True, "last", None), "xs4": (R5 / "xs4/natural", True, "last", None)}),
 "MNIST digit on CelebA": (lambda: alt(R5 / "xs4/mnist"), {"ship": (R5 / "audit/base", True, "last", lambda f: f.startswith("mnist_")), "xs8": (R5 / "xs8/mnist", True, "last", None), "xs4": (R5 / "xs4/mnist", True, "last", None)}),
 "CIFAR-10 patch on CelebA": (lambda: alt(R5 / "xs4/cifar"), {"ship": (R5 / "cifar/min_exact", True, "last", lambda f: "st0" not in f), "xs8": (R5 / "xs8/cifar", True, "last", None), "xs4": (R5 / "xs4/cifar", True, "last", None)}),
 "CLWD + watermark (b2r, declared alpha + pp readout)": (lambda: readout("b2r", "xs4_kA", 0.05, True), {"ship": (R / "b2r/minDr_kA", False, "last", None), "xs8": (R / "b2r/xs8_kA", False, "last", None), "xs4": (R / "b2r/xs4_kA", False, "last", None)}),
 "PRWD + watermark (declared alpha + pp readout)": (lambda: readout("prwd", "xs4_kA", 0.0, False), {"ship": (R / "prwd/minE_kA", False, "last", None), "xs8": (R / "prwd/xs8_kA", False, "last", None), "xs4": (R / "prwd/xs4_kA", False, "last", None)}),
 "FFHQ + watermark (ffhqwm0, kApp)": (lambda: alt(R / "ffhqwm0_native/xs4_kApp"), {"ship": (R / "ffhqwm0_native/kApp", False, "last", None), "xs8": (R / "ffhqwm0_native/xs8_kApp", False, "last", None), "xs4": (R / "ffhqwm0_native/xs4_kApp", False, "last", None)}),
 "FFHQ + watermark, unseen logos (ffhqwmU0, kApp)": (lambda: alt(R / "ffhqwmU0_native/xs4_kApp"), {"ship": (R / "ffhqwmU0_native/kApp", False, "last", None), "xs8": (R / "ffhqwmU0_native/xs8_kApp", False, "last", None), "xs4": (R / "ffhqwmU0_native/xs4_kApp", False, "last", None)}),
 "ImageNet + watermark (imagenetwm, kApp)": (lambda: alt(R / "imagenetwm_native/xs4_kApp"), {"ship": (R / "imagenetwm_native/kApp", False, "last", None), "xs8": (R / "imagenetwm_native/xs8_kApp", False, "last", None), "xs4": (R / "imagenetwm_native/xs4_kApp", False, "last", None)}),
 "ISTD shadow (b5s, mult. gated)": (lambda: alt(R / "b5s/xs4_mult"), {"ship": (R / "b5s/multminEc", True, 1, None), "xs8": (R / "b5s/xs8_mult", True, 1, None), "xs4": (R / "b5s/xs4_mult", True, 1, None)}),
 "ISTD shadow + FFHQ (ffhqsh0, mult. gated)": (lambda: alt(R / "ffhqsh0_native/xs4_mult256"), {"ship": (R / "ffhqsh0_native/mult256g", True, 1, None), "xs8": (R / "ffhqsh0_native/xs8_mult256", True, 1, None), "xs4": (R / "ffhqsh0_native/xs4_mult256", True, 1, None)}),
 "ISTD shadow + ImageNet (imagenetsh, mult. gated)": (lambda: alt(R / "imagenetsh_native/xs4_mult256"), {"ship": (R / "imagenetsh_native/mult256in", True, "last", None), "xs8": (R / "imagenetsh_native/xs8_mult256", True, "last", None), "xs4": (R / "imagenetsh_native/xs4_mult256", True, "last", None)}),
}
def nfe(d, dm, ship, tagsel):
    js = chunks(d, tagsel=tagsel)
    if not js: return None
    tot = []
    for _, f in js:
        j = json.load(open(f)); K = j["K"]; cur = j["curves"]; M = len(cur); m = M if ship == "last" else min(ship, M); n = 13 * K
        for c in cur[:m]:
            ns = min(NSTEP_NFE, max(1, sum(1 for L in LADDER if L <= c["sigma_star"] + 1e-9)))
            n += float(c.get("nfe_extra", 0)) + (ns if dm else 0)
        tot.append(n)
    return float(np.mean(tot))
CFG = [("ship", "all levels (shipped)"), ("xs8", "8 DDIM levels"), ("xs4", "4 DDIM levels")]
rows, NF = {}, {}
for bench, L in CELLS.items():
    v = {}
    for k in ("ship", "xs8"):
        try: v[k] = L[k]()
        except Exception as e: print(f"[{bench}/{k}] {type(e).__name__}: {e}"); v[k] = None
    try: v["xs4"] = XS[bench][0]()
    except Exception as e: print(f"[{bench}/xs4] {type(e).__name__}: {e}"); v["xs4"] = None
    rows[bench] = v; NF[bench] = {k: nfe(*XS[bench][1][k]) for k in ("ship", "xs8", "xs4")}
A = REPO / "experiments/tango_msdm_2026_08_04/results/arms/abl2"
for b, lab in (("piano", "Slakh piano (SI-SDR, oracle window, K4 M4)"), ("guitar", "Slakh guitar (SI-SDR, oracle window, K4 M4)")):
    v, nf = {}, {}
    for k, a in (("ship", "base"), ("xs8", "xs8"), ("xs4", "xs4")):
        p = A / b / a / "summary.json"
        if p.exists():
            cells = json.load(open(p))["cells"]; v[k] = np.array([c["win_sisdr"] for c in cells]); nf[k] = float(np.mean([c["nfe"] for c in cells]))
        else: v[k] = None; nf[k] = None
    rows[lab] = v; NF[lab] = nf
def _pair(v, s):
    if v is None or s is None: return None, None
    n = min(len(v), len(s)); ok = ~np.isnan(v[:n]) & ~np.isnan(s[:n]); return v[:n][ok], s[:n][ok]
def cell(v, s, nfe_):
    if v is None: return "--"
    vv = v[~np.isnan(v)]; txt = f"{vv.mean():.2f} / {vv.min():.2f}"
    if len(vv) < len(v): txt += f" [n={len(vv)}]"
    a, b = _pair(v, s)
    if a is not None and len(a) and v is not s: d = a - b; txt += f" ({d.mean():+.2f}, {(d > 0).sum()}/{len(d)})"
    if nfe_ is not None: txt += f" [{nfe_:.0f}]"
    return txt
L = ["# x-step DDIM levels {all (shipped), 8, 4} on every bench; n-step 8 DDIM steps (2026-09-25)", "",
     "Cells: on-Omega PSNR mean / worst (audio: in-window SI-SDR), (paired delta vs shipped, wins/n), [network evaluations per image / cell at the shipped iterate]. '--' = not run (yet).", "",
     "| Bench | " + " | ".join(l for _, l in CFG) + " |", "|---|" + "---|" * len(CFG)]
J = {}
for bench, v in rows.items():
    s = v["ship"]; L.append(f"| {bench} | " + " | ".join(cell(v[k], s, NF[bench].get(k)) for k, _ in CFG) + " |"); J[bench] = {}
    for k, _ in CFG:
        if v[k] is None: J[bench][k] = None; continue
        vv = v[k][~np.isnan(v[k])]; a, b = _pair(v[k], s)
        J[bench][k] = dict(mean=float(vv.mean()), worst=float(vv.min()), n=int(len(vv)), nfe=NF[bench].get(k), delta=(None if (k == "ship" or a is None or not len(a)) else float((a - b).mean())))
L += ["", "## Mean paired delta vs shipped (benches where the configuration has run)", ""]
for k, lab in CFG[1:]:
    ds = [J[b][k]["delta"] for b in J if J[b][k] and J[b][k]["delta"] is not None]
    L.append(f"- {lab}: {np.mean(ds):+.2f} dB over {len(ds)} benches" if ds else f"- {lab}: no runs yet")
txt = "\n".join(L); print(txt)
(REPO / "results/xs_table_20260925.md").write_text(txt + "\n"); json.dump(J, open(REPO / "results/xs_table_20260925.json", "w"), indent=1)
T = [r"% x-step DDIM level sweep (2026-09-25), built by experiments/extbench/build_xs_table_20260925.py", r"\begin{table*}[t]", r"    \centering",
     r"    \caption{\textbf{Solver steps of the x-step.} Shipped \methodmin{} arm of each bench (x-step over every ladder level at or below $\sigma_m$, K-chain inpaint initialisation, $n$-step 8 DDIM steps) against the same arm with the x-step ladder capped at 8 or 4 DDIM levels. PSNR (dB) over $\Omega$ (audio: in-window SI-SDR), mean / worst, in parentheses the paired difference to the shipped arm and the number of images on which the variant is better, in brackets the network evaluations per image (audio: per cell) at the shipped output.}",
     r"    \label{tab:xs_levels}", r"    \small", r"    \setlength{\tabcolsep}{3pt}", r"    \begin{tabular}{@{}lccc@{}}", r"        \toprule",
     r"        \textbf{Bench} & \shortstack{\textbf{all levels}\\\textbf{(shipped)}} & \textbf{8 DDIM levels} & \textbf{4 DDIM levels} \\", r"        \midrule"]
for bench, v in rows.items():
    s = v["ship"]; T.append("        " + bench.replace("+", "$+$").replace("_", r"\_") + " & " + " & ".join(cell(v[k], s, NF[bench].get(k)) for k, _ in CFG) + r" \\")
T += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
(REPO / "paper/tables/xs_levels_sweep.tex").write_text("\n".join(T) + "\n"); print("\nwrote paper/tables/xs_levels_sweep.tex")
