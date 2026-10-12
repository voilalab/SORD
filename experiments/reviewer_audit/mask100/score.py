"""Mask-error ablation over ALL bench images (user 2026-09-24: make tab:mask_ablation numbers match tab:localized, whose
'ours' column is the same shipped run over 100 images). Same runs / names as the 25-image table; chunks c25/c50/c75 added by
mask100/jobs.txt. PSNR on the TRUE corrupted region, mean ± sample std over images; IoU = per-image mean over the same images.
Writes mask100/results.md and paper/tables/mask_ablation.tex (only when every cell has all its chunks; otherwise prints n).
python experiments/reviewer_audit/mask100/score.py"""
import sys, re
from pathlib import Path
import numpy as np
sys.path.insert(0, "experiments/reviewer_audit"); from common import load, chunks, R5, EXT, HERE, REPO
BENCH = {"natural": R5 / "bench_natural.npz", "mnist": R5 / "bench_mnist.npz", "cifar": R5 / "bench_cifar.npz", "b2r": EXT / "b2r/bench_b2r.npz"}
PRE = {"natural": "audit", "mnist": "audit", "cifar": "cifar/audit", "b2r": "audit"}
NFULL = {"natural": 100, "mnist": 100, "cifar": 100, "b2r": 99}
LAB = {"natural": "Natural patch", "mnist": "MNIST digit", "b2r": "CLWD watermark", "cifar": "CIFAR-10 patch"}
COLS = [("exact", "Exact"), ("dil3", "Dilate 3 px"), ("dil5", "Dilate 5 px"), ("erode1", "Erode 1 px"), ("shift2", "Shift 2 px"), ("blind", "Detected mask")]

def cell(b, v):
    name = (f"{PRE[b]}/base" if b != "cifar" else "cifar/min_exact") if v == "exact" else f"{PRE[b]}/mask_{v}"
    r = load(b, name)
    if r is None: return None
    # PSNR recomputed from the saved x_alt exactly as tab:localized / celeba_dps_cols does (clip to [0,1], on the TRUE Omega)
    Z = np.load(BENCH[b]); Mt = Z["M_true"] > 0.5; GT = Z["x_A"]
    keep = np.where(Mt.reshape(len(Mt), -1).sum(1) > 0)[0]          # b2r: bench image 63 has an empty support and is skipped by the runs (n=99)
    Mt, GT = Mt[keep], GT[keep]
    ious, ps = [], []
    off = 0
    for i0, j, npz in chunks(b, name):
        D = np.load(npz); Mu = D["M3"][..., 0] > 0.5; k = len(Mu); mt = Mt[off:off + k]
        assert np.allclose(D["x_A"], GT[off:off + k]), (b, name, i0, "positional pairing broken"); i0 = off; off += k
        inter = (Mu & mt).reshape(k, -1).sum(1); union = (Mu | mt).reshape(k, -1).sum(1); ious += list(inter / union)
        for q in range(k):
            e = ((np.clip(D["x_alt"][q], 0, 1) - GT[i0 + q]) ** 2)[mt[q]].mean(); ps.append(10 * np.log10(1 / max(e, 1e-20)))
    a = np.asarray(ps, float); n = len(a)
    return dict(n=n, mean=a.mean(), std=a.std(ddof=1), worst=a.min(), iou=float(np.mean(ious)), full=(n >= NFULL[b]))

HEAD = (REPO / "paper/tables/mask_ablation.tex").read_text().split("        \\midrule\n")[0]
def main():
    res = {b: {v: cell(b, v) for v, _ in COLS} for b in LAB}
    md = ["# Mask-error ablation over all images (mean ± std, per-image IoU); n per cell", "", "| bench | " + " | ".join(l for _, l in COLS) + " |", "|---|" + "---|" * len(COLS)]
    for b in LAB:
        md.append(f"| {LAB[b]} | " + " | ".join("--" if res[b][v] is None else f"{res[b][v]['mean']:.2f} ± {res[b][v]['std']:.2f}" + ("" if v == "exact" else f" ({res[b][v]['iou']:.2f})") + f" n={res[b][v]['n']}" for v, _ in COLS) + " |")
    (HERE / "mask100/results.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
    complete = all(res[b][v] is not None and res[b][v]["full"] for b in LAB for v, _ in COLS if not (b in ("cifar", "b2r") and v == "blind"))
    print("all non-detected cells at full n:", complete)
    if "--write" in sys.argv or complete:
        s = HEAD.replace("over 25 images", "over all benchmark images ($n{=}100$; $99$ for CLWD), using the final\n    refinement iterate and $K=32$ chains; the detected-mask cells for CLWD and\n    CIFAR-10 are over the first $25$ images ($^{n=25}$)") + "        \\midrule\n\n"
        for b in LAB:
            cells = [res[b][v] for v, _ in COLS]; best = max((i for i, c in enumerate(cells) if c), key=lambda i: cells[i]["mean"])
            s += f"        {LAB[b]}\n"
            for i, (c, (v, _)) in enumerate(zip(cells, COLS)):
                num = "--" if c is None else (f"$\\mathbf{{{c['mean']:.2f} \\pm {c['std']:.2f}}}$" if i == best else f"${c['mean']:.2f} \\pm {c['std']:.2f}$")
                if c is not None and v != "exact": num += f" $({c['iou']:.2f})$"
                if c is not None and not c["full"]: num += f"$^{{n={c['n']}}}$"
                s += f"        & {num}" + (" \\\\\n" if i == 5 else "\n")
            s += "\n"
        s += "        \\bottomrule\n    \\end{tabular}\n    }\n\\end{table*}\n"
        (REPO / "paper/tables/mask_ablation_n100.tex").write_text(s); print("wrote paper/tables/mask_ablation_n100.tex")

if __name__ == "__main__": main()
