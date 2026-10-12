"""Companion to make_mask_fig.py (2026-09-25 user: "we do not have any example of erosion and shift"): same 8 images
(natural #6/#24, mnist #12/#9, b2r #3/#15, cifar #24/#16 — identical selection rule), columns = input with the TRUE mask |
input with the 1-px ERODED mask | input with the mask SHIFTED by 2 px | ours with true / eroded / shifted mask | ground truth.
PSNR on the TRUE Omega under each panel; mask area / coverage of the true support under the inputs.
Data: audit/base, audit/mask_erode1, audit/mask_shift2 c0 runs (min config K=32).   python experiments/reviewer_audit/make_mask_fig_erode_shift.py [--labels]"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
sys.path.insert(0, "experiments/reviewer_audit"); from common import chunks
LABELS = "--labels" in sys.argv      # default = bare like the shipped fig_mask_error (no row labels, no suptitle)
FIG = Path("paper/figs/ablation"); FIG.mkdir(parents=True, exist_ok=True)
def psnr(a, b, m): return -10 * np.log10(max(float(((np.clip(a, 0, 1) - b) ** 2)[m].mean()), 1e-12))
def c0(bench, name):
    rows = chunks(bench, name); assert rows and rows[0][0] == 0, (bench, name); return np.load(rows[0][2])
BENCHES = [("natural", "audit", "natural patch"), ("mnist", "audit", "MNIST digit"), ("b2r", "audit", "CLWD watermark"), ("cifar", "cifar/audit", "CIFAR patch")]
VARS = [("mask_erode1", "eroded 1px"), ("mask_shift2", "shifted 2px")]
panels, labels = [], []
for b, pre, lab in BENCHES:
    base = c0(b, f"{pre}/base"); var = {k: c0(b, f"{pre}/{k}") for k, _ in VARS}
    Mt = base["M3"][..., 0] > 0.5; M3t = base["M3"] > 0.5
    ok = np.arange(len(Mt))
    p_ex = np.array([psnr(base["x_alt"][i], base["x_A"][i], M3t[i]) for i in ok]); o = np.argsort(p_ex)[::-1]
    sel = [int(o[len(o) // 4]), int(o[len(o) // 2])]          # same rule as make_mask_fig.py
    if b == "b2r":
        p_in = np.array([psnr(base["y"][i], base["x_A"][i], M3t[i]) for i in ok]); vis = np.argsort(p_in)[:8]
        sel = [int(v) for v in vis[np.argsort(p_ex[vis])[::-1][:2]]]
    for i in sel:
        d = dict(y=base["y"][i], gt=base["x_A"][i], Mt=Mt[i], M3t=M3t[i], ex=base["x_alt"][i])
        for k, _ in VARS: d[k] = var[k]["x_alt"][i]; d[k + "_M"] = var[k]["M3"][i, ..., 0] > 0.5
        panels.append(d); labels.append(f"{lab}\n#{i}")
COLS = ["input, true mask"] + [f"input, mask {v}" for _, v in VARS] + ["ours, true mask"] + [f"ours, {v}" for _, v in VARS] + ["ground truth"]
fig, axes = plt.subplots(len(panels), len(COLS), figsize=(1.45 * len(COLS), 1.55 * len(panels) + 0.3)); axes = np.atleast_2d(axes)
def outline(im, m): im = np.clip(im, 0, 1).copy(); im[m ^ ndi.binary_erosion(m)] = [1, 0, 0]; return im
for r, d in enumerate(panels):
    masks = [d["Mt"]] + [d[k + "_M"] for k, _ in VARS]
    ims = [outline(d["y"], m) for m in masks] + [d["ex"]] + [d[k] for k, _ in VARS] + [d["gt"]]
    for c, im in enumerate(ims):
        ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([]); ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
        if r == 0: ax.set_title(COLS[c], fontsize=7.5)
        nv = len(VARS) + 1
        if c < nv:
            m = masks[c]; cov = (m & d["Mt"]).sum() / max(d["Mt"].sum(), 1)   # fraction of the true support covered by the given mask
            ax.set_xlabel(f"{psnr(d['y'], d['gt'], d['M3t']):.1f} dB, covers {100*cov:.0f}%", fontsize=6.5, labelpad=1)
        elif c < 2 * nv: ax.set_xlabel(f"{psnr(ims[c], d['gt'], d['M3t']):.1f} dB", fontsize=7, labelpad=1)
    if LABELS: axes[r, 0].set_ylabel(labels[r], fontsize=7)
if LABELS: fig.suptitle("Mask error: the given mask is an erosion or a shift of the true support; PSNR on the true region", fontsize=9, y=0.995)
fig.tight_layout(pad=0.25, rect=(0, 0, 1, 0.975) if LABELS else None)
for ext in ("png", "pdf"): fig.savefig(FIG / f"fig_mask_error_erode_shift{'_labels' if LABELS else ''}.{ext}", dpi=200)
print("rows:", [l.replace(chr(10), " ") for l in labels])
