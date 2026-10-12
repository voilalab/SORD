"""Ablation-2 analysis (2026-09-13): (1) Tweedie -> DPS (quality + runtime), (2) K sweep incl. K1/K64/cifar,
(4) sigma-schedule grid, (5) mask error incl. cifar.  Reads the audit chunk-runs (c0, n=25), writes
WARNING (2026-09-24): section 5 uses the run json psnr['alt'], which the loop measures on the GIVEN (perturbed) mask, NOT on the
true support; the true-Omega mask table is produced by make_mask_ablation_table.py / mask100/score.py from the npz.
results/ablation2.md + ablation2.json + paper/figs/ablation/fig_{k_sweep,sigma_schedule}.{png,pdf}.
python experiments/reviewer_audit/analyze_ablation2.py"""
import sys, json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "experiments/reviewer_audit"); from common import load, chunks, nfe_cum, HERE
OUT = HERE / "results"; FIG = Path("paper/figs/ablation"); FIG.mkdir(parents=True, exist_ok=True)
B_ALL = [("natural", "audit"), ("mnist", "audit"), ("cifar", "cifar/audit"), ("b2r", "audit"), ("b5sEc", "audit")]
LAB = dict(natural="natural", mnist="mnist", cifar="cifar", b2r="CLWD wm", b5sEc="ISTD shadow")
def c0(b, name):
    r = load(b, name)
    if r is None: return None
    n0 = 25 if b != "b2r" else min(25, r["n"])
    j = json.load(open(chunks(b, name)[0][1]))
    d = dict(K=r["K"], M=r["M"], last=float(r["alt"][:n0].mean()), worst=float(r["alt"][:n0].min()), m1=float(r["mean_traj"][0][0]),
             inpaint=float(r["inpaint"][:n0].mean()), nfe_last=float(nfe_cum(r)[-1]), wall=j.get("wall_s"),
             sig2=float(r["mean_sig"][0][1]) if r["M"] > 1 else None, traj=[float(v) for v in r["mean_traj"][0]])
    d["best"] = float(r["traj"][:n0].max(1).mean()) if "traj" in r else max(d["traj"])
    return d
res = {}; L = ["# Ablation-2 results (min config K=32 unless stated; c0 n=25; PSNR on the true Omega, mean / worst of the last iterate, best-iterate mean)\n"]
# ---------------- 1. Tweedie -> DPS
L += ["## 1. Tweedie denoising replaced by DPS (x-step: off-Omega replacement -> DPS likelihood gradient; n-step: renoise+DDIM -> DPS-Gauss on the read)\n",
      "| bench | arm | last | worst | best-iter | NFE/img (Tweedie count; DPS arms add one backward per forward) | wall [s] (25 imgs, shared GPU) | wall x base |", "|---|---|---|---|---|---|---|---|"]
res["dps"] = {}
for b, pre in [("natural", "audit"), ("mnist", "audit"), ("cifar", "cifar/audit"), ("b2r", "audit"), ("b5sEc", "audit")]:
    base = c0(b, f"{pre}/base_t") or c0(b, f"{pre}/base")
    for arm, lab in (("base_t" if b != "cifar" else "base", "Tweedie (shipped)"), ("dpsX", "DPS x-step"), ("dpsN", "DPS n-step"), ("dpsXN", "DPS both")):
        d = c0(b, f"{pre}/{arm}")
        if d is None: L.append(f"| {LAB[b]} | {lab} | (running) | | | | | |"); continue
        res["dps"][f"{b}/{arm}"] = d
        ratio = (d["wall"] / base["wall"]) if (d["wall"] and base and base["wall"]) else None
        L.append(f"| {LAB[b]} | {lab} | {d['last']:.2f} | {d['worst']:.2f} | {d['best']:.2f} | {d['nfe_last']:.0f} | {d['wall'] if d['wall'] else float('nan'):.0f} | {ratio if ratio else float('nan'):.2f} |")
# ---------------- 2. K sweep
L += ["\n## 2. K (number of chains)  [K=1 is undefined: sigma* is the K-chain spread; the loop divides by zero in the init demix]\n", "| bench | K | last | worst | best-iter | NFE/img@last |", "|---|---|---|---|---|---|"]
res["k"] = {}; fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for b, pre in B_ALL:
    ks, vals, ws = [], [], []
    for K in (1, 2, 4, 8, 16, 32, 64):
        name = f"{pre}/base" if K == 32 else f"{pre}/minK{K}"
        d = c0(b, name)
        if d is None: continue
        res["k"][f"{b}/K{K}"] = d; ks.append(K); vals.append(d["last"]); ws.append(d["worst"])
        L.append(f"| {LAB[b]} | {K} | {d['last']:.2f} | {d['worst']:.2f} | {d['best']:.2f} | {d['nfe_last']:.0f} |")
    if ks: ax[0].plot(ks, vals, "o-", label=LAB[b]); ax[1].plot(ks, ws, "o-", label=LAB[b])
for a_, t in zip(ax, ("mean PSNR on $\\Omega$ (last iterate)", "worst image")):
    a_.set_xscale("log", base=2); a_.set_xlabel("K"); a_.set_title(t, fontsize=9); a_.grid(alpha=0.3)
ax[0].legend(fontsize=7); fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(FIG / f"fig_k_sweep.{ext}", dpi=200)
plt.close(fig)
# ---------------- 3. solver steps (x-step ladder length, n-step DDIM steps)
L += ["\n## 3. Fewer solver steps: x-step ladder capped at N levels (shipped: all levels <= sigma_m, 13 at 0.4) and n-step DDIM steps (shipped 8; effective = levels <= sigma*, usually 1-2)\n",
      "| bench | arm | last | worst | best-iter | NFE/img@last | NFE x base | wall [s] |", "|---|---|---|---|---|---|---|---|"]
res["steps"] = {}
for b, pre in B_ALL:
    base = c0(b, f"{pre}/base")
    for arm, lab in (("base", "shipped (all levels, DDIM 8)"), ("xs4", "x-step 4 levels"), ("xs2", "x-step 2 levels"), ("xs1", "x-step 1 level (single Tweedie jump)"), ("nd2", "n-step DDIM 2"), ("nd1", "n-step DDIM 1 (single Tweedie)")):
        d = c0(b, f"{pre}/{arm}")
        if d is None: continue
        res["steps"][f"{b}/{arm}"] = d
        L.append(f"| {LAB[b]} | {lab} | {d['last']:.2f} | {d['worst']:.2f} | {d['best']:.2f} | {d['nfe_last']:.0f} | {d['nfe_last']/base['nfe_last']:.2f} | {d['wall']/25 if d['wall'] else float('nan'):.1f} |")
# ---------------- 4. sigma schedule
SS = [(0.2, 0.1), (0.3, 0.1), (0.4, 0.1), (0.5, 0.1), (0.6, 0.1), (0.8, 0.1), (0.4, 0.05), (0.4, 0.2), (0.4, 0.3)]
L += ["\n## 4. x-step renoise schedule sigma_start -> sigma_end (linear over M; shipped 0.4 -> 0.1) and constant-altitude controls\n",
      "| bench | schedule | last | worst | best-iter | sigma*_2 |", "|---|---|---|---|---|---|"]
res["sigma"] = {}; fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for b, pre in B_ALL:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for s0, s1 in SS:
        name = f"{pre}/base" if (s0, s1) == (0.4, 0.1) else f"{pre}/ss{s0:g}-{s1:g}"
        d = c0(b, name)
        if d is None: continue
        res["sigma"][f"{b}/ss{s0:g}-{s1:g}"] = d
        L.append(f"| {LAB[b]} | {s0:g} -> {s1:g} | {d['last']:.2f} | {d['worst']:.2f} | {d['best']:.2f} | {d['sig2'] if d['sig2'] else float('nan'):.3f} |")
        if s1 == 0.1: xs0.append(s0); ys0.append(d["last"])
        if s0 == 0.4: xs1.append(s1); ys1.append(d["last"])
    for s in (0.137, 0.2, 0.3):
        d = c0(b, f"{pre}/sigfix{s:g}")
        if d is None: continue
        res["sigma"][f"{b}/fix{s:g}"] = d; L.append(f"| {LAB[b]} | const sigma* = {s:g} (n-step altitude override) | {d['last']:.2f} | {d['worst']:.2f} | {d['best']:.2f} | (fixed) |")
    if xs0: o = np.argsort(xs0); ax[0].plot(np.array(xs0)[o], np.array(ys0)[o], "o-", label=LAB[b])
    if xs1: o = np.argsort(xs1); ax[1].plot(np.array(xs1)[o], np.array(ys1)[o], "o-", label=LAB[b])
ax[0].set_xlabel("$\\sigma_{start}$ (end 0.1)"); ax[1].set_xlabel("$\\sigma_{end}$ (start 0.4)")
for a_ in ax: a_.set_ylabel("mean PSNR on $\\Omega$"); a_.grid(alpha=0.3)
ax[0].legend(fontsize=7); fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(FIG / f"fig_sigma_schedule.{ext}", dpi=200)
plt.close(fig)
# ---------------- 5. mask error
L += ["\n## 5. Mask error (given mask != true support; PSNR on the TRUE Omega)\n", "| bench | mask | inpaint (m0) | last | worst | best-iter |", "|---|---|---|---|---|---|"]
res["mask"] = {}
for b, pre in [("natural", "audit"), ("mnist", "audit"), ("cifar", "cifar/audit"), ("b2r", "audit")]:
    for v in ("exact", "dil3", "dil5", "dil8", "erode1", "erode2", "shift2", "shift4", "drop30", "drop50", "blind"):
        d = c0(b, f"{pre}/base" if v == "exact" else f"{pre}/mask_{v}")
        if d is None: continue
        res["mask"][f"{b}/{v}"] = d; L.append(f"| {LAB[b]} | {v} | {d['inpaint']:.2f} | {d['last']:.2f} | {d['worst']:.2f} | {d['best']:.2f} |")
(OUT / "ablation2.md").write_text("\n".join(L) + "\n"); json.dump(res, open(OUT / "ablation2.json", "w"), indent=1)
print("\n".join(L))
