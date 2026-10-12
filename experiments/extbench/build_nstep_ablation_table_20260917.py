"""n-step identity ablation table (2026-09-17/18): drop the DM n-step (sigma* renoise+denoise of the residual,
--nfilter identity = clipped raw residual) on EVERY bench, paired vs the shipped DM-n-step arm on the same images.
Sources: 64^2 CelebA / b2r / b5s-additive rows = experiments/reviewer_audit/results/residual.json (audit/base vs
audit/filt_identity); the 27-job queue results/nstep_ablation_20260917 (cifar, wm, prwd, b5s-mult, ffhqwm0, imagenetwm,
ffhqsh0, imagenetsh, audio piano/guitar).  Writes results/nstep_ablation_table_20260917.md + paper/tables/ablation_nstep.tex."""
import glob, json
from pathlib import Path
import numpy as np
REPO = next(p for p in Path(__file__).resolve().parents if (p / "sord").is_dir())
E = REPO / "experiments/extbench/results"; R5 = REPO / "experiments/ros/results/r5"

def psnr(a, b, m):
    e = ((np.clip(a, 0, 1) - b) ** 2)[m > 0.5].mean(); return 10 * np.log10(1 / max(e, 1e-20))

def chunks(p, tagfilter=None):
    out = []
    for d in sorted(glob.glob(str(p / "c*"))):
        js = sorted(glob.glob(d + "/*.json"))
        if tagfilter: js = [j for j in js if tagfilter(Path(j).name)]
        if js: out.append((int(Path(d).name[1:]), js[0]))
    return sorted(out)

def load_json(p, tagfilter=None, key="alt"):
    v, m1, M = [], [], []
    for _, j in chunks(p, tagfilter):
        d = json.load(open(j)); v += d["psnr"][key]; M.append(len(d["curves"]))
        m1 += d["curves"][0].get("psnr_x_img", [])
    return np.array(v), (np.array(m1) if m1 else None), M

def load_npz_iter(p, key, tagfilter=None):
    v = []
    for _, j in chunks(p, tagfilter):
        z = np.load(j[:-5] + ".npz"); x = z[key]; gt = z["x_A"]; m = z["M3"]
        v += [psnr(x[i], gt[i], m[i]) for i in range(x.shape[0])]
    return np.array(v)

rows = []  # (bench label, class, n, ship mean, ship worst, ident mean, ident worst, delta, wins, note)
def add(label, ship, ident, note=""):
    n = min(len(ship), len(ident)); s, i = ship[:n], ident[:n]; d = i - s
    rows.append(dict(bench=label, n=n, ship=(float(s.mean()), float(s.min())), ident=(float(i.mean()), float(i.min())),
                     delta=float(d.mean()), wins=int((d > 0).sum()), note=note))

# --- 64^2 CelebA + b2r + b5s-additive: from the reviewer audit (already paired) ---
res = json.load(open(REPO / "experiments/reviewer_audit/results/residual.json"))
for b, lab in (("natural", "Natural patch on CelebA"), ("mnist", "MNIST digit on CelebA"), ("b2r", "CLWD + watermark (blind $n$)")):
    B, I = res[b]["base"], res[b]["filt_identity"]
    rows.append(dict(bench=lab, n=I["n"], ship=(B["last"], B["worst"]), ident=(I["last"], I["worst"]), delta=I["delta"], wins=I["wins"], note=""))
# cifar: shipped = r5/cifar/min_exact (plain tag, n=100), identity = audit_cifar/filt_identity (n=100)
plain = lambda f: "st0" not in f
s, _, _ = load_json(R5 / "cifar/min_exact", plain); i, _, _ = load_json(R5 / "audit_cifar/filt_identity")
add("CIFAR-10 patch on CelebA", s, i)
# wm on CelebA (stop-class, sign-free)
sf = lambda f: "sign-free" in f
s, _, _ = load_json(R5 / "audit/base", sf); i, _, _ = load_json(R5 / "audit/filt_identity", sf)
add("CLWD logos on CelebA (blind $n$)", s, i)
# prwd blind DM arm (minE, M6) vs minE_identity
s, _, _ = load_json(E / "prwd/minE"); i, _, _ = load_json(E / "prwd/minE_identity")
add("PRWD + watermark (blind $n$)", s, i)
# ffhqwm0 / imagenetwm: shipped blind DM arm ran M6 / M3; identity ran M2 -> pair at iterate 2 from the saved per-iterate images
for bench, ship, ident, lab, Mend in (("ffhqwm0_native", "min256", "min256_identity", "FFHQ + watermark (blind $n$, $m_2$)", 6),
                                ("imagenetwm_native", "min256in", "min256in_identity", "ImageNet + watermark (blind $n$, $m_2$)", 3)):
    s = load_npz_iter(E / bench / ship, "x_m2"); i = load_npz_iter(E / bench / ident, "x_m2")
    chk, _, _ = load_json(E / bench / ident); assert abs(chk.mean() - i.mean()) < 0.05, (bench, chk.mean(), i.mean())
    add(lab, s, i, f"paired at iterate 2 (shipped blind run continues to $m_{Mend}$ and declines)")
# b5s additive (audit) + multiplicative shipped arm
B, I = res["b5sEc"]["base25"], res["b5sEc"]["filt_identity"]
rows.append(dict(bench="ISTD shadow, additive loop ($m_8$, first 25 images)", n=I["n"], ship=(B["last"], B["worst"]), ident=(I["last"], I["worst"]), delta=I["delta"], wins=I["wins"], note="first 25 images only (shipped full-100: 29.14 / 13.00)"))
s, _, _ = load_json(E / "b5s/multminEc"); i, _, _ = load_json(E / "b5s/multminEc_identity")
add("ISTD shadow, multiplicative (shipped, gated $m_1$)", s, i)
for bench, ship, ident, lab in (("ffhqsh0_native", "mult256g", "mult256g_identity", "ISTD shadow + FFHQ (mult., gated)"),
                                ("imagenetsh_native", "mult256in", "mult256in_identity", "ISTD shadow + ImageNet (mult., gated)")):
    s, _, Ms = load_json(E / bench / ship); i, _, Mi = load_json(E / bench / ident)
    add(lab, s, i, f"gate stopped the shipped run at $m_{{{max(Ms)}}}$, the identity run at $m_{{{max(Mi)}}}$")
# audio (in-window SI-SDR, oracle window, K4 M4)
A = REPO / "experiments/tango_msdm_2026_08_04/results/arms/abl2"
audio = []
for b, lab in (("piano", "Slakh piano (IV+OOV, SIR 0/5)"), ("guitar", "Slakh guitar (balanced, SIR 0/5)")):
    cb = json.load(open(A / b / "base/summary.json"))["cells"]; ci = json.load(open(A / b / "identN/summary.json"))["cells"]
    assert [c["cell"] for c in cb] == [c["cell"] for c in ci]
    wb = np.array([c["win_sisdr"] for c in cb]); wi = np.array([c["win_sisdr"] for c in ci]); win_in = np.array([c["win_sisdr_in"] for c in cb])
    d = wi - wb
    audio.append(dict(bench=lab, n=len(wb), ship=(float(wb.mean()), float(wb.min())), ident=(float(wi.mean()), float(wi.min())),
                      delta=float(d.mean()), wins=int((d > 0).sum()), gain=(float((wb - win_in).mean()), float((wi - win_in).mean())),
                      nfe=(cb[0]["nfe"], ci[0]["nfe"])))

ORDER = ["Natural", "MNIST", "CIFAR", "CLWD logos on CelebA", "CLWD + watermark", "PRWD", "FFHQ + watermark", "ImageNet + watermark", "ISTD shadow, additive", "ISTD shadow, multiplicative", "ISTD shadow + FFHQ", "ISTD shadow + ImageNet"]
rows.sort(key=lambda r: next(k for k, o in enumerate(ORDER) if r["bench"].startswith(o)))
# ---- markdown ----
L = ["# n-step ablation: DM residual filter at sigma* (shipped) vs raw clipped residual (--nfilter identity), 2026-09-18",
     "", "Image rows: on-Omega PSNR mean / worst of the arm's readout; delta = identity - shipped, paired per image; wins = images where identity is better.",
     "Declared-alpha watermark arms (b2r kA+pp 28.06, prwd kA 30.75, ffhqwm0 kApp 30.05, imagenetwm kApp 28.01) have NO DM n-step by construction",
     "(closed-form alpha/colour fit), so their rows below are the blind-n arms of those benches.", "",
     "| Bench | n | shipped (DM n-step) | no DM n-step (identity) | delta (paired) | wins | note |", "|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['bench']} | {r['n']} | {r['ship'][0]:.2f} / {r['ship'][1]:.2f} | {r['ident'][0]:.2f} / {r['ident'][1]:.2f} | {r['delta']:+.2f} | {r['wins']}/{r['n']} | {r['note']} |")
L += ["", "Audio rows: in-window SI-SDR (dB) mean / worst of the last iterate, oracle window, tether-free K=4 M=4 loop (paper sampler config); SI-SDRi = mean improvement over the input.", "",
      "| Bench | cells | shipped (DM n-step) | no DM n-step | delta (paired) | wins | SI-SDRi shipped -> identity | NFE/cell |", "|---|---|---|---|---|---|---|---|"]
for r in audio:
    L.append(f"| {r['bench']} | {r['n']} | {r['ship'][0]:.2f} / {r['ship'][1]:.2f} | {r['ident'][0]:.2f} / {r['ident'][1]:.2f} | {r['delta']:+.2f} | {r['wins']}/{r['n']} | {r['gain'][0]:+.2f} -> {r['gain'][1]:+.2f} | {r['nfe'][0]:.0f} -> {r['nfe'][1]:.0f} |")
L += ["", "Sources: experiments/reviewer_audit/results/residual.json (natural, mnist, b2r, b5s additive); r5/cifar/min_exact vs r5/audit_cifar/filt_identity;",
      "r5/audit/{base,filt_identity} (wm); extbench prwd/minE{,_identity}, b5s/multminEc{,_identity}, ffhqwm0_native/min256{,_identity} (x_m2),",
      "imagenetwm_native/min256in{,_identity} (x_m2), ffhqsh0_native/mult256g{,_identity}, imagenetsh_native/mult256in{,_identity};",
      "tango_msdm_2026_08_04/results/arms/abl2/{piano,guitar}/{base,identN}. Queue: extbench/results/nstep_ablation_20260917/jobs.txt."]
md = "\n".join(L); print(md)
(REPO / "results/nstep_ablation_table_20260917.md").write_text(md + "\n")
json.dump(dict(image=rows, audio=audio), open(REPO / "results/nstep_ablation_table_20260917.json", "w"), indent=1)

# ---- LaTeX ----
def bold(a, b):  # bold the better mean
    return (f"\\textbf{{{a:.2f}}}", f"{b:.2f}") if a >= b else (f"{a:.2f}", f"\\textbf{{{b:.2f}}}")
T = [r"% n-step identity ablation (2026-09-18): built by experiments/extbench/build_nstep_ablation_table_20260917.py",
     r"% from results/nstep_ablation_table_20260917.md (per-row provenance there).",
     r"\begin{table*}[t]", r"    \centering",
     r"    \caption{\textbf{Dropping the diffusion $n$-step on every bench.} The shipped loop re-noises the residual to $\sigma^\star$ and denoises it with the same prior before the read-out; the ablation replaces that step by the clipped raw residual (identity filter), everything else unchanged. Image rows: PSNR (dB) over $\Omega$, mean / worst; $\Delta$ = identity $-$ shipped, paired per image; wins = images on which the identity arm is better. Watermark rows use the blind-$n$ arm of each bench: the declared-$\alpha$ arms fit $\alpha$ and colour in closed form and contain no diffusion $n$-step. The $256^2$ watermark rows are paired at iterate 2 (the shipped blind run continues and declines); the gated shadow arms are compared at the iterate their gate selects. Audio rows: in-window SI-SDR (dB) of the last iterate, oracle window, K$=$4, M$=$4, mean / worst over cells; SI-SDRi = mean improvement over the input, shipped $\to$ identity.}",
     r"    \label{tab:ablation_nstep}", r"    \small", r"    \setlength{\tabcolsep}{4pt}",
     r"    \begin{tabular}{@{}lrccrr@{}}", r"        \toprule",
     r"        \textbf{Bench} & $n$ & \textbf{shipped (DM $n$-step)} & \textbf{no DM $n$-step} & $\Delta$ & wins \\", r"        \midrule"]
groups = [(0, 4, None), (4, 8, None), (8, 12, None)]
for a, b, name in groups:
    for r in rows[a:b]:
        s, i = bold(r["ship"][0], r["ident"][0])
        T.append(f"        {r['bench']} & {r['n']} & {s} / {r['ship'][1]:.2f} & {i} / {r['ident'][1]:.2f} & ${r['delta']:+.2f}$ & {r['wins']}/{r['n']} \\\\")
    T.append(r"        \midrule")
for r in audio:
    s, i = bold(r["ship"][0], r["ident"][0])
    lab = f"{r['bench']}, SI-SDRi ${r['gain'][0]:+.2f} \\to {r['gain'][1]:+.2f}$"
    T.append(f"        {lab} & {r['n']} & {s} / ${r['ship'][1]:.2f}$ & {i} / ${r['ident'][1]:.2f}$ & ${r['delta']:+.2f}$ & {r['wins']}/{r['n']} \\\\")
T += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
(REPO / "paper/tables/ablation_nstep.tex").write_text("\n".join(T) + "\n")
print("\nwrote paper/tables/ablation_nstep.tex")
