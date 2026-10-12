"""ICLR teaser (2026-09-25): one column per task, input y (support Omega outlined) on top, SORD (ours) below,
optional ground-truth row (--gt).  Examples = the paper-figure picks (make_all_image_fig.PAPER_ROWS) plus the
strongest 256^2 and audio cells.  python paper/figs/qual/make_teaser.py [--gt]"""
import sys, glob, json
from pathlib import Path
import numpy as np
from scipy.signal import stft
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "paper/figs/qual"); sys.argv_backup = list(sys.argv); sys.argv = [sys.argv[0]]
from make_all_image_fig import celeba_row, b2s_row, istd_row, psnr, R
sys.argv = sys.argv_backup
GT = "--nogt" not in sys.argv
NOTITLES = "--notitles" in sys.argv     # 2026-09-25 user: also drop the column titles -> fig_teaser_notitles
BARE = "--labels" not in sys.argv     # 2026-09-25 user: no row labels, no PSNR / SI-SDR captions (--labels restores them)
PER = int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else 1   # 2026-09-25 user: N examples per task ("a lot of examples, vertically small")
NONORM = "--norm" not in sys.argv     # 2026-09-25 user: clipped inputs are the default; --norm scales additive inputs to unit range     # 2026-09-25 user: ground-truth row is the default
ARMS = Path("experiments/tango_msdm_2026_08_04/results/arms"); SR = 22050

def ext_row(bench, arm, i, key="x_alt"):
    fs = sorted(glob.glob(f"{R}/{bench}/{arm}/c*/*.npz"), key=lambda s: int(s.split("/c")[-1].split("/")[0]))
    Z = [np.load(f) for f in fs]; cat = lambda k: np.concatenate([z[k] for z in Z])
    return dict(y=cat("y")[i], gt=cat("x_A")[i], M=cat("M3")[i], ours=cat(key)[i])

def sisdr(est, ref):
    a = (est @ ref) / (ref @ ref + 1e-12) * ref; return 10 * np.log10((a @ a) / ((est - a) @ (est - a) + 1e-12))
def spec(x):
    f, t, Z = stft(x, fs=SR, nperseg=1024, noverlap=768); return f, t, 20 * np.log10(np.abs(Z) + 1e-6)
def best_speech_cells(n=1):
    """top-n figS speech cells by in-window SI-SDR gain, one per clip index."""
    out, seen = [], set()
    for d in sorted(ARMS.glob("figS_speech_c*_s*")):
        z = np.load(next(d.glob("*.npz"))); m = z["mask"] > 0.5
        g = sisdr(z["est"][m], z["clean"][m]) - sisdr(z["y"][m], z["clean"][m]); out.append((g, d.name, z))
    out.sort(key=lambda t: -t[0]); res = []
    for g, name, z in out:
        clip = name.split("_c")[1].split("_")[0]
        if clip in seen: continue
        seen.add(clip); res.append((g, name, z))
        if len(res) == n: break
    return res
def best_speech_cell(): return best_speech_cells(1)[0]
def ext_rank(bench, arm, key="x_alt"):
    """image indices of an extbench arm ordered by our on-Omega PSNR, best first."""
    fs = sorted(glob.glob(f"{R}/{bench}/{arm}/c*/*.npz"), key=lambda s: int(s.split("/c")[-1].split("/")[0]))
    Z = [np.load(f) for f in fs]; cat = lambda k: np.concatenate([z[k] for z in Z])
    gt, M, x = cat("x_A"), cat("M3"), cat(key)
    return list(np.argsort([-psnr(x[i], gt[i], M[i]) for i in range(len(gt))]))

# per-task picks, first = the original teaser example, the rest = the paper / extended-figure picks (make_all_image_fig)
PICKS = {"mnist": [40, 28, 21, 83], "natural": [20, 31, 39, 6], "cifar": [7, 78, 21, 24], "b2s100": [7, 20, 19, 23],
         "ffhqwmU0": [18] + [int(i) for i in ext_rank("ffhqwmU0_native", "kApp") if i != 18][:3],   # 2026-09-25 user: logo over the eyes first
         "b5s": [20, 44, 72, 6]}
# 2026-09-25 user: "--fresh" = examples NOT used in fig_qual_all_image_benches_paper (PAPER_ROWS mnist 28/21, natural 20/31,
# cifar 78/7, b2s100 20/19, b5s 20/44); ranked by our on-Omega PSNR with those excluded (see session log 2026-09-25)
if "--fresh" in sys.argv:
    PICKS.update({"mnist": [40, 93, 39, 79], "natural": [58, 10, 79, 16], "cifar": [36, 19, 82, 93], "b2s100": [7, 8, 13, 16], "b5s": [56, 43, 89, 11]})   # 2026-09-25 user: left/right shadow slots swapped 11/10 -> 56/89   # b5s: by gain with shadow area >= 8 % (corner-speck shadows excluded)
TASKS = [("MNIST digit\nCelebA $64^2$", "mnist", lambda i: celeba_row("mnist", i)),
         ("Natural patch\nCelebA $64^2$", "natural", lambda i: celeba_row("natural", i)),
         ("CIFAR-10 patch\nCelebA $64^2$", "cifar", lambda i: celeba_row("cifar", i)),
         ("Watermark\nCLWD $64^2$", "b2s100", lambda i: b2s_row(100, i)),
         ("Watermark, unseen logo\nFFHQ $256^2$", "ffhqwmU0", lambda i: ext_row("ffhqwmU0_native", "kApp", i)),
         ("Shadow\nISTD $64^2$", "b5s", istd_row)]
SKIP = set(sys.argv[sys.argv.index("--skip") + 1].split(",")) if "--skip" in sys.argv else set()
DROP = set(sys.argv[sys.argv.index("--drop") + 1].split(",")) if "--drop" in sys.argv else set()
NAUDIO = int(sys.argv[sys.argv.index("--naudio") + 1]) if "--naudio" in sys.argv else None      # audio cells (default PER)
AUDW = float(sys.argv[sys.argv.index("--audio-w") + 1]) if "--audio-w" in sys.argv else 1.7   # audio cell width in tile units   # 2026-09-25 user: drop task groups (e.g. --skip cifar)
cols = []   # (title or None, row dict, first-of-group flag)
for title, key, load in TASKS:
    if key in SKIP: continue
    picks = [i for i in PICKS[key][:PER] if f"{key}:{i}" not in DROP]   # 2026-09-25 user: --drop mnist:40 removes single tiles
    for j, i in enumerate(picks): cols.append((title if j == 0 else None, load(i), j == 0))
print("picks:", {k: v[:PER] for k, v in PICKS.items()})
AUD = best_speech_cells(NAUDIO or PER); g, cell, A = AUD[0]
nr = 3 if GT else 2
# grid: PER tiles per task, a thin spacer column between task groups (PER > 1), PER audio columns at the right
GAP = float(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else (0.05 if PER > 1 else 0.0)   # 2026-09-25 user: --gap 0 = no group seam
wr, colpos = [], []
for c, (_, _, first) in enumerate(cols):
    if first and c > 0 and GAP: wr.append(GAP)
    colpos.append(len(wr)); wr.append(1)
if GAP: wr.append(GAP)
audpos = [len(wr) + k for k in range(len(AUD))]; wr += [AUDW] * len(AUD)
U = 0.98   # inches per unit column; cell height follows the tile width so rows stay tight at any PER
fig = plt.figure(figsize=(U * sum(wr) + 0.4, (U if BARE else U * 1.18) * nr + (0.1 if NOTITLES else 0.55)))
SP = float(sys.argv[sys.argv.index("--space") + 1]) if "--space" in sys.argv else 0.012   # 2026-09-25 user: --space 0 = tiles touch
gs = fig.add_gridspec(nr, len(wr), width_ratios=wr, wspace=SP, hspace=SP if BARE else 0.05)
def show(ax, im, title=None, label=None):
    ax.imshow(np.clip(im, 0, 1), interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    if title and not NOTITLES: ax.set_title(title, fontsize=8.5, pad=3)
    if label and not BARE: ax.set_xlabel(label, fontsize=7.2, labelpad=1.5)
for c, (title, d, _) in enumerate(cols):
    ydisp = d["y"] / d["y"].max() if (d["y"].max() > 1 and not NONORM) else d["y"]     # 2026-09-25 user: normalise, do not clip
    gc = colpos[c]
    ax = fig.add_subplot(gs[0, gc]); show(ax, ydisp, title)
    m = d["M"][..., 0] > 0.5
    ax.contour(m.astype(float), levels=[0.5], colors="#ffe14d", linewidths=0.9, linestyles="--")
    if not BARE: ax.set_xlabel(f"{psnr(d['y'], d['gt'], d['M']):.1f} dB", fontsize=7.2, labelpad=1.5)
    ax = fig.add_subplot(gs[1, gc]); show(ax, d["ours"], label=f"{psnr(d['ours'], d['gt'], d['M']):.1f} dB")
    if GT: ax = fig.add_subplot(gs[2, gc]); show(ax, d["gt"])
    if c == 0 and not BARE:
        fig.axes[-nr].set_ylabel("input $y$\n(support $\\Omega$ dashed)", fontsize=8.5)
        fig.axes[-nr + 1].set_ylabel("SORD (ours)", fontsize=8.5, fontweight="bold")
        if GT: fig.axes[-1].set_ylabel("ground truth", fontsize=8.5)
# audio column: mixture / ours (/ clean) spectrograms with the window dashed
for k, (g, cell, A) in enumerate(AUD):
    mw = A["mask"] > 0.5; cl = A["clean"]; _, _, Sc = spec(cl); vmax = Sc.max(); vmin = vmax - 70
    w0, w1 = np.argmax(mw) / SR, (len(mw) - np.argmax(mw[::-1])) / SR
    sigs = [("Speech on piano\naudio (spectrogram)" if k == 0 else None, A["y"]), (None, A["est"])] + ([(None, cl)] if GT else [])
    for r, (title, x) in enumerate(sigs):
        ax = fig.add_subplot(gs[r, audpos[k]]); f, t, S = spec(x)
        ax.pcolormesh(t, f / 1000, S, vmin=vmin, vmax=vmax, cmap="magma", shading="auto", rasterized=True); ax.set_ylim(0, 8)
        for w in (w0, w1): ax.axvline(w, color="#ffe14d", ls="--", lw=0.9)
        ax.set_box_aspect(1 / AUDW); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if title and not NOTITLES: ax.set_title(title, fontsize=8.5, pad=3)
        if r < 2 and not BARE: ax.set_xlabel(f"SI-SDR {sisdr(x[mw], cl[mw]):.1f} dB", fontsize=7.2, labelpad=1.5)
    print("audio cell", cell, f"gain {g:+.1f} dB")
out = Path("paper/figs/qual") / (("fig_teaser" if GT else "fig_teaser_2row") + ("" if NONORM else "_norm") + ("_labels" if not BARE else "") + ("_notitles" if NOTITLES else "") + (f"_per{PER}" if PER > 1 else "") + ("_fresh" if "--fresh" in sys.argv else "") + ("_no" + "-".join(sorted(SKIP)) if SKIP else "") + ("_drop" + str(len(DROP)) if DROP else "") + (f"_a{len(AUD)}w{AUDW:g}".replace(".", "p") if (NAUDIO or AUDW != 1.7) else "") + ("_tight" if "--space" in sys.argv else ""))   # no '.' (with_suffix would eat it)
if "--out" in sys.argv: out = Path("paper/figs/qual") / sys.argv[sys.argv.index("--out") + 1]   # 2026-09-25 user: fixed output name
for ext in ("png", "pdf"): fig.savefig(out.with_suffix("." + ext), dpi=260, bbox_inches="tight", pad_inches=0.02)
print("wrote", out)
