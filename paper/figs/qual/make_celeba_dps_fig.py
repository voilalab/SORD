"""CelebA-prior benches (MNIST digit / natural patch / CIFAR-10 patch on CelebA faces, 64^2, exact mask), 2026-09-23.
User ask: "results of TANGO-Min over celeba + mnist / natural / cifar; columns input, DPS, DPS with mu = noise, DPS with mu = 0,
baseline 1, baseline 2, ours, GT".
Columns:
  input y | DPS-inpaint (sigma -> inf, dpsg_inf) | adaptive DPS, mu = mean of the noise on Omega (oracle mu-hat, dpsp_auto)
  | adaptive DPS, mu = 0 (dpsp_mu0) | baseline 1 = SGM two-prior (cifar: SGM with the CIFAR prior) | baseline 2 = SGM two-prior + mask proj.
  | TANGO-Min (ours, x_alt) | ground truth
DPS panels are pasted back to y off-Omega (raw DPS samples are not exact there); PSNR on Omega under each panel.
Outputs (paper/figs/qual/): fig_qual_celeba_dps_cols.{png,pdf} (2 rows per bench, the paper rows), fig_qual_celeba_dps_cols_extended_<bench>
(10 rows at the 5-95% quantiles of our on-mask PSNR), and the full-bench table (n=100) in celeba_dps_cols_table.md / paper/tables/celeba_dps_cols.tex.
python paper/figs/qual/make_celeba_dps_fig.py [--extended]
"""
import sys, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
sys.path.insert(0, "experiments/reviewer_audit"); from common import chunks
R = Path("experiments/extbench/results"); OUT = Path("paper/figs/qual")
PAPER_ROWS = dict(mnist=[3, 21], natural=[20, 31], cifar=[78, 7])      # same rows as fig_qual_all_image_benches_paper
COLS = [("y", "input $y$"), ("dps_inp", "DPS-inpaint"), ("dps_mu", "DPS, $\\mu=\\bar n_\\Omega$\n(adaptive, oracle $\\hat\\mu$)"),
        ("dps_mu0", "DPS, $\\mu=0$\n(adaptive)"), ("b1", "baseline 1"), ("b2", "baseline 2"), ("ours", "TANGO-Min\n(ours)"), ("gt", "ground truth")]
ROWLAB = [("mnist", "MNIST digit\non CelebA"), ("natural", "natural patch\non CelebA"), ("cifar", "CIFAR-10 patch\non CelebA")]

def psnr(a, b, m):
    e = ((np.clip(a, 0, 1) - b) ** 2)[m > 0.5].mean(); return 10 * np.log10(1 / max(e, 1e-20))

def load_bench(bench):
    if bench == "cifar":   # full-100 TANGO-Min run (K32 M8) = paper row 22.05/13.25
        fs = sorted(glob.glob("experiments/ros/results/r5/cifar/min_exact/c*/*gg1_w0_noex_noharm_blind.npz"), key=lambda s: int(Path(s).parts[-2][1:]))
        c = [(None, None, f) for f in fs]
    else:
        c = chunks(bench, "audit/base")
    cat = lambda k: np.concatenate([np.load(n)[k] for _, _, n in c])
    A = dict(y=cat("y"), gt=cat("x_A"), M=cat("M3"), ours=cat("x_alt"))
    A["dps_inp"] = np.load(R / f"dpsg/{bench}/dpsg_inf.npz")["x_hat"]
    A["dps_mu"] = np.load(R / f"dpsp/{bench}/dpsp_auto.npz")["x_hat"]
    A["dps_mu0"] = np.load(R / f"dpsp/{bench}/dpsp_mu0.npz")["x_hat"]
    t1, t2 = ("sgm_free", "sgmM") if bench == "cifar" else ("sgm", "sgmM")
    for k, tag in (("b1", t1), ("b2", t2)):
        fs = sorted(glob.glob(f"experiments/ros/results/r5/{bench}/{tag}/c*.npz"), key=lambda s: int(Path(s).stem[1:]))
        A[k] = np.concatenate([np.load(f)["x_A"] for f in fs])
    A["b1_lab"] = "SGM two-prior" + (" (CIFAR prior)" if bench == "cifar" else ""); A["b2_lab"] = "SGM two-prior + mask proj."
    n = len(A["gt"]); assert all(len(A[k]) == n for k, _ in COLS), {k: len(A[k]) for k, _ in COLS}
    for k, _ in COLS:
        if k.startswith("dps"): A[k] = np.where(A["M"] > 0.5, np.clip(A[k], 0, 1), np.clip(A["y"], 0, 1))   # paste y back off-Omega
    return A

def draw(rows, labels, fname, title):
    nr, nc = len(rows), len(COLS)
    fig, axes = plt.subplots(nr, nc, figsize=(1.4 * nc, 1.55 * nr + 0.45)); axes = np.atleast_2d(axes)
    for r, ((A, i), rl) in enumerate(zip(rows, labels)):
        m = A["M"][i, ..., 0] > 0.5; edge = m ^ ndi.binary_erosion(m)
        for c, (k, lab) in enumerate(COLS):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(lab, fontsize=7.2)
            im = np.clip(np.asarray(A[k][i], np.float64), 0, 1).copy()
            if k != "gt": ax.set_xlabel(f"{psnr(im, A['gt'][i], A['M'][i]):.1f} dB", fontsize=7, labelpad=1)
            if k == "y": im[edge] = [1, 0, 0]
            ax.imshow(im, interpolation="nearest")
            if k in ("b1", "b2"):
                ax.text(0.03, 0.97, A[k + "_lab"], ha="left", va="top", fontsize=4.6, color="w", transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.55))
        axes[r, 0].set_ylabel(rl, fontsize=7)
    fig.suptitle(title, fontsize=9, y=0.995); fig.tight_layout(pad=0.25, rect=(0, 0, 1, 1 - 0.32 / (1.55 * nr + 0.45)))
    for ext in ("png", "pdf"): fig.savefig(OUT / f"{fname}.{ext}", dpi=220)
    plt.close(fig)

def table(benches):
    md = ["# CelebA-prior benches under the DPS-variant columns (on-Omega PSNR, mean ± std / worst, n=100 each)", "",
          "DPS-inpaint = DPS with the exact mask, sigma -> inf (no in-mask data term). adaptive DPS = textbook zeta_t = zeta/||y' - x0_hat|| with y' = y - mu m;",
          "mu = mean of the true noise on Omega (oracle) vs mu = 0. baseline 1 = SGM two-prior (cifar: SGM with the CIFAR prior), baseline 2 = + mask projection.",
          "TANGO-Min = shipped blind x_alt (audit/base; cifar min_exact K32 M8). All 100 images per bench; DPS pasted back to y off-Omega.", "",
          "| bench | " + " | ".join(lab.replace("\n", " ").replace("$", "") for k, lab in COLS if k != "gt") + " |", "|---" * (len(COLS)) + "|"]
    tex = ["% auto-generated by paper/figs/qual/make_celeba_dps_fig.py (2026-09-23); on-Omega PSNR mean $\\pm$ std / worst, n=100",
           "\\begin{tabular}{l" + "c" * (len(COLS) - 1) + "}", "\\toprule",
           "bench & input $y$ & DPS-inpaint & DPS, $\\mu=\\bar n_\\Omega$ & DPS, $\\mu=0$ & baseline 1 & baseline 2 & TANGO-Min (ours) \\\\", "\\midrule"]
    for bench, A in benches.items():
        cells = []
        for k, _ in COLS:
            if k == "gt": continue
            p = np.array([psnr(A[k][i], A["gt"][i], A["M"][i]) for i in range(len(A["gt"]))])
            cells.append(f"{p.mean():.2f} ± {p.std(ddof=1):.2f} / {p.min():.2f}")   # 2026-09-24: sample std (ddof=1), the paper-wide convention
        md.append(f"| {bench} on CelebA | " + " | ".join(cells) + " |")
        tex.append(bench + " & " + " & ".join(c.replace("±", "$\\pm$") for c in cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "celeba_dps_cols_table.md").write_text("\n".join(md) + "\n"); Path("paper/tables/celeba_dps_cols.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))

if __name__ == "__main__":
    benches = {b: load_bench(b) for b, _ in ROWLAB}
    table(benches)
    rows, labels = [], []
    for bench, rl in ROWLAB:
        for i in PAPER_ROWS[bench]: rows.append((benches[bench], i)); labels.append(f"{rl}\n#{i}")
    draw(rows, labels, "fig_qual_celeba_dps_cols", "CelebA prior, exact mask: two images per bench; PSNR on the corrupted region")
    if "--extended" in sys.argv:
        for bench, rl in ROWLAB:
            A = benches[bench]; p = np.array([psnr(A["ours"][i], A["gt"][i], A["M"][i]) for i in range(len(A["gt"]))]); o = np.argsort(p)
            sel = [int(o[int(q * (len(o) - 1))]) for q in np.linspace(0.05, 0.95, 10)][::-1]
            draw([(A, i) for i in sel], [f"{rl}\n#{i}" for i in sel], f"fig_qual_celeba_dps_cols_extended_{bench}",
                 f"{rl.replace(chr(10), ' ')}: 10 images at the 5-95% quantiles of our on-mask PSNR")
            print(bench, "extended rows", sel)
