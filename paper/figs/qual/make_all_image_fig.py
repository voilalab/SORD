"""One-row-per-bench qualitative figure over ALL image experiments (2026-09-13).

Rows (one image each, drawn at random with seed 20260913 from the rows shown in the
per-bench figures: fig_qual_{mnist,natural,cifar}_baselines, fig_b2s_truealpha_check_v2,
fig_paper_qual_istd):
  mnist #83 | natural #39 | cifar #21 | b2s100 (CLWD watermark, size sweep 1.0x) #23 | ISTD (b5s) #72
Columns (same for every row):
  input y | DPS-inpaint | DPS-Gauss | adaptive DPS | baseline 1 | baseline 2 | K-chain inpaint (ours m0) | TANGO-Min (ours) | ground truth
  baseline 1/2 = SGM two-prior / SGM two-prior + mask proj. (rows 1-3; cifar: SGM with the CIFAR prior / + mask);
                 SLBR / DENet trained on CLWD (interim: PatchWiper PRWD ckpt zero-shot) (watermark row);
                 ShadowDiffusion / ShadowFormer, official ISTD ckpt (ISTD row).   [2026-09-13 in-domain second baselines]
DPS columns are pasted back to y off-Omega (raw DPS samples are not exact there); on-mask PSNR under each panel.
The b2s100 DPS arms come from experiments/extbench/run_queue_b2s100_dps_20260913.sh (dpsg_inf / dpsg_auto / dpsp_auto);
a missing file draws a "running" placeholder so the draft can be re-rendered as results land.
python paper/figs/qual/make_all_image_fig.py
"""
import sys, glob, json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
sys.path.insert(0, "experiments/reviewer_audit"); from common import chunks
R = Path("experiments/extbench/results"); OUT = Path("paper/figs/qual")
ROWS = dict(mnist=83, natural=39, cifar=21, b2s100=23, b5s=72)      # seed 20260913 draw, see docstring

def psnr(a, b, m):
    e = ((np.clip(a, 0, 1) - b) ** 2)[m > 0.5].mean(); return 10 * np.log10(1 / max(e, 1e-20))

def opt(path, key="x_hat", i=None):
    p = Path(path)
    if not p.exists(): return None
    z = np.load(p)[key]; return z if i is None else z[i]

def to64(p): return np.asarray(Image.open(p).convert("RGB").resize((64, 64), Image.BILINEAR), np.float64) / 255.

def celeba_row(bench, i):
    if bench == "cifar":
        fs = sorted(glob.glob("experiments/ros/results/r5/cifar/min_exact/c*/*gg1_w0_noex_noharm_blind.npz"), key=lambda s: int(Path(s).parts[-2][1:]))
        c = [(None, None, f) for f in fs]
    else:
        c = chunks(bench, "audit/base")
    cat = lambda k: np.concatenate([np.load(n)[k] for _, _, n in c])
    d = dict(y=cat("y")[i], gt=cat("x_A")[i], M=cat("M3")[i], inp=cat("x_inp")[i], ours=cat("x_alt")[i],
             dps_inp=opt(R / f"dpsg/{bench}/dpsg_inf.npz", i=i), dps_g=opt(R / f"dpsp/{bench}/dpsp_mu0.npz", i=i),
             dps_a=opt(R / f"dpsp/{bench}/dpsp_auto.npz", i=i))
    t1, t2 = ("sgm_free", "sgmM") if bench == "cifar" else ("sgm", "sgmM")
    for k, tag in (("b1", t1), ("b2", t2)):
        fs = sorted(glob.glob(f"experiments/ros/results/r5/{bench}/{tag}/c*.npz"), key=lambda s: int(Path(s).stem[1:]))
        d[k] = np.concatenate([np.load(f)["x_A"] for f in fs])[i]
    d["b1_lab"] = "SGM two-prior" + ("\n(CIFAR prior)" if bench == "cifar" else ""); d["b2_lab"] = "SGM two-prior\n+ mask proj."
    d["ours_tag"] = None
    return d

def b2s_row(s, i):
    z = np.load(R / f"b2s{s}/bench_b2s{s}.npz"); w = np.load(glob.glob(str(R / f"b2s{s}/kA/c0/*_blind.npz"))[0])
    j = int(np.argmin([np.abs(w["x_A"][jj] - z["x_A"][i]).mean() for jj in range(w["x_A"].shape[0])]))
    assert np.abs(w["x_A"][j] - z["x_A"][i]).max() < 1e-5
    registry = Path(__file__).resolve().parents[3] / "reproduction/image_ids.json"
    if registry.exists():
        ids = json.loads(registry.read_text())[f"b2s{s}"]
    else:
        ids = sorted(int(Path(p).stem) for p in glob.glob(str(R / f"b2s{s}/y256/*.png")))
    M = np.repeat(z["M_true"][i][..., None], 3, -1).astype(np.float32)
    return dict(y=z["y"][i], gt=z["x_A"][i], M=M, inp=w["x_inp"][j], ours=w["x_m6"][j],
                dps_inp=opt(R / f"dpsg/b2s{s}/dpsg_inf.npz", i=i), dps_g=opt(R / f"dpsp/b2s{s}/dpsp_mu0.npz", i=i),
                dps_a=opt(R / f"dpsp/b2s{s}/dpsp_auto.npz", i=i),
                # 2026-09-18 logo-exposure rule: the 12 b2s logos are CLWD train-split renders seen by SLBR / DENet;
                # only the PRWD-trained PatchWiper is logo-disjoint.  2026-09-23 user: fill baseline 2 with the official
                # CLWD SLBR anyway (b2s{s}/slbr, Aug-06 run) rather than leaving a grey panel -- logos seen, caveat in caption.
                b1=to64(R / f"b2s100/patchwiper_prwd/{ids[i]}.png"), b1_lab="PatchWiper\n(PRWD ckpt, logo-disjoint)",
                b2=to64(R / f"b2s{s}/slbr/{ids[i]}.png"), b2_lab="SLBR\n(official CLWD ckpt, logos seen)", ours_tag="true $\\alpha$")

def istd_row(i):
    B = np.load(R / "b5s/bench_b5s.npz"); SD = Path("experiments/extbench/shadowdiffusion/results_b5s")   # untouched raw input (user rule 2026-09-18)
    fs = sorted(glob.glob(str(R / "b5s/multminEc/c*/*.npz")), key=lambda s: int(s.split("/c")[-1].split("/")[0]))
    cat = lambda k: np.concatenate([np.load(f)[k] for f in fs])
    assert np.abs(cat("x_A")[i] - B["x_A"][i]).max() < 1e-5
    M = np.repeat(B["M_true"][i][..., None], 3, -1).astype(np.float32)
    return dict(y=B["y"][i], gt=B["x_A"][i], M=M, inp=cat("x_inp")[i], ours=cat("x_alt")[i],
                dps_inp=opt(R / "b5s/dps_init.npz", "x_init", i), dps_g=opt(R / "dpsp/b5s/dpsp_mu0.npz", i=i),
                dps_a=opt(R / "dpsp/b5s/dpsp_auto.npz", i=i),
                b1=to64(SD / "sr" / B["stems"][i]), b1_lab="ShadowDiffusion\n(supervised)",
                b2=to64(R.parent / f"shadowformer/results_b5s/{B['stems'][i]}"), b2_lab="ShadowFormer\n(supervised)", ours_tag="mult.")

# 2026-09-25 user: column 3 = adaptive DPS with mu = 0 (dpsp_mu0, the table's "DPS mu=0" arm; was DPS-Gauss oracle sigma),
# column 4 = adaptive DPS with mu = mean contaminant on Omega (dpsp_auto, the table's "DPS mu=mu_n")
COLS = [("y", "input $y$"), ("dps_inp", "DPS-inpaint"), ("dps_g", "DPS, $\\mu=0$"), ("dps_a", "DPS, $\\mu=\\mu_n$"),
        ("b1", "baseline 1"), ("b2", "baseline 2"), ("inp", "K-chain inpaint\n(ours, $m_0$)"),
        ("ours", "SORD\n(ours)"), ("gt", "ground truth")]
ROWLAB = [("mnist", "MNIST digit\non CelebA"), ("natural", "natural patch\non CelebA"), ("cifar", "CIFAR-10 patch\non CelebA"),
          ("b2s100", "CLWD\nwatermark"), ("b5s", "ISTD\nshadow")]
# 2026-09-23 user: "--bare" = drop the K-chain inpaint (ours m0) column; no column titles, no row labels, no suptitle,
# no in-panel tags (PSNR captions kept).  Output gets a "_bare" suffix so the titled versions stay on disk.
BARE = "--bare" in sys.argv
if BARE: COLS = [c for c in COLS if c[0] != "inp"]
SUF = "_bare" if BARE else ""
# 2026-09-24 user: "--collabels" = bare figure but WITH the column titles on the first row (row labels / tags still off)
COLLABELS = "--collabels" in sys.argv
if COLLABELS: SUF += "_labels"

def main():
    rows = [celeba_row("mnist", ROWS["mnist"]), celeba_row("natural", ROWS["natural"]), celeba_row("cifar", ROWS["cifar"]),
            b2s_row(100, ROWS["b2s100"]), istd_row(ROWS["b5s"])]
    nr, nc = len(rows), len(COLS)
    fig, axes = plt.subplots(nr, nc, figsize=(1.4 * nc, 1.55 * nr + 0.35))
    for r, (d, (bench, rl)) in enumerate(zip(rows, ROWLAB)):
        m = d["M"][..., 0] > 0.5; edge = m ^ ndi.binary_erosion(m)
        for c, (k, lab) in enumerate(COLS):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and (not BARE or COLLABELS): ax.set_title(lab, fontsize=7.5)
            im = d.get(k)
            if im is None:
                ax.imshow(np.full((64, 64, 3), 0.92));
                if not BARE: ax.text(0.5, 0.5, d.get(k + "_lab", "DPS run\npending"), ha="center", va="center", fontsize=5.5, color="0.3", transform=ax.transAxes)
                ax.set_xlabel("—", fontsize=7, labelpad=1); continue
            im = np.clip(np.asarray(im, np.float64), 0, 1).copy()
            if k.startswith("dps"): im = np.where(d["M"] > 0.5, im, np.clip(d["y"], 0, 1))   # paste y back off-Omega
            if k != "gt": ax.set_xlabel(f"{psnr(im, d['gt'], d['M']):.1f} dB", fontsize=7, labelpad=1)   # before the outline
            if k == "y": im[edge] = [1, 0, 0]
            ax.imshow(im, interpolation="nearest")
            tag = None if BARE else (d.get(k + "_lab") if k in ("b1", "b2") else (d["ours_tag"] if k == "ours" else None))
            if tag: ax.text(0.03, 0.97, tag.replace("\n", " "), ha="left", va="top", fontsize=4.6, color="w", transform=ax.transAxes,
                            bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.55))
        if not BARE: axes[r, 0].set_ylabel(rl, fontsize=7)
    if BARE: fig.tight_layout(pad=0.25)
    else:
        fig.suptitle("One image per bench (random draw from the per-bench figures); PSNR on the corrupted region", fontsize=9, y=0.995)
        fig.tight_layout(pad=0.25, rect=(0, 0, 1, 0.975))
    for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_qual_all_image_benches{SUF}.{ext}", dpi=220)
    for d, (bench, _) in zip(rows, ROWLAB):
        print(f"{bench:8s} #{ROWS[bench]:3d} | " + " | ".join(f"{k}: {psnr(np.clip(d[k],0,1), d['gt'], d['M']):.2f}" if d.get(k) is not None else f"{k}: --" for k, _ in COLS if k != "gt"))

if __name__ == "__main__" and "--extended" not in sys.argv and "--paper" not in sys.argv: main()

# ---------------------------------------------------------------------------------------------------------------
# EXTENDED version (2026-09-13 user: "make this image for 10 images"): 10 images per bench, same 9 columns, one
# 10-row figure per bench (fig_qual_all_image_benches_extended_<bench>) + one stacked 50-row file.  Rows = the
# 5..95 % quantiles of our on-mask PSNR over the bench (spread over outcomes, not best cases); the bench index is
# printed on the left.  b2s100 has 25 images (size-sweep 1.0x bench), all others 100.
def bench_psnr_order(bench):
    if bench in ("mnist", "natural", "cifar"):
        fs = sorted(glob.glob("experiments/ros/results/r5/cifar/min_exact/c*/*gg1_w0_noex_noharm_blind.npz"), key=lambda s: int(Path(s).parts[-2][1:])) if bench == "cifar" else None
        c = [(None, None, f) for f in fs] if fs else chunks(bench, "audit/base")
        cat = lambda k: np.concatenate([np.load(n)[k] for _, _, n in c]); ours, gt, M = cat("x_alt"), cat("x_A"), cat("M3")
    elif bench == "b2s100":
        z = np.load(R / "b2s100/bench_b2s100.npz"); w = np.load(glob.glob(str(R / "b2s100/kA/c0/*_blind.npz"))[0])
        j_of = [int(np.argmin([np.abs(w["x_A"][jj] - z["x_A"][i]).mean() for jj in range(w["x_A"].shape[0])])) for i in range(z["x_A"].shape[0])]
        ours, gt, M = w["x_m6"][j_of], z["x_A"], np.repeat(z["M_true"][..., None], 3, -1)
    else:
        B = np.load(R / "b5s/bench_b5s.npz"); fs = sorted(glob.glob(str(R / "b5s/multminEc/c*/*.npz")), key=lambda s: int(s.split("/c")[-1].split("/")[0]))
        ours, gt, M = np.concatenate([np.load(f)["x_alt"] for f in fs]), B["x_A"], np.repeat(B["M_true"][..., None], 3, -1)
    p = np.array([psnr(ours[i], gt[i], M[i]) for i in range(len(gt))]); return np.argsort(p)

def draw_rows(rows, labels, fname, title):
    nr, nc = len(rows), len(COLS)
    fig, axes = plt.subplots(nr, nc, figsize=(1.4 * nc, 1.55 * nr + 0.35)); axes = np.atleast_2d(axes)
    for r, (d, rl) in enumerate(zip(rows, labels)):
        m = d["M"][..., 0] > 0.5; edge = m ^ ndi.binary_erosion(m)
        for c, (k, lab) in enumerate(COLS):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and (not BARE or COLLABELS): ax.set_title(lab, fontsize=7.5)
            im = d.get(k)
            if im is None:
                ax.imshow(np.full((64, 64, 3), 0.92))
                if not BARE: ax.text(0.5, 0.5, d.get(k + "_lab", "pending"), ha="center", va="center", fontsize=5.5, color="0.3", transform=ax.transAxes)
                ax.set_xlabel("—", fontsize=7, labelpad=1); continue
            im = np.clip(np.asarray(im, np.float64), 0, 1).copy()
            if k.startswith("dps"): im = np.where(d["M"] > 0.5, im, np.clip(d["y"], 0, 1))
            if k != "gt": ax.set_xlabel(f"{psnr(im, d['gt'], d['M']):.1f} dB", fontsize=7, labelpad=1)
            if k == "y": im[edge] = [1, 0, 0]
            ax.imshow(im, interpolation="nearest")
            tag = None if BARE else (d.get(k + "_lab") if k in ("b1", "b2") else (d["ours_tag"] if k == "ours" else None))
            if tag: ax.text(0.03, 0.97, tag.replace("\n", " "), ha="left", va="top", fontsize=4.6, color="w", transform=ax.transAxes,
                            bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.55))
        if not BARE: axes[r, 0].set_ylabel(rl, fontsize=7)
    if BARE: fig.tight_layout(pad=0.25)
    else: fig.suptitle(title, fontsize=9, y=0.995); fig.tight_layout(pad=0.25, rect=(0, 0, 1, 1 - 0.3 / (1.55 * nr + 0.35)))
    for ext in ("png", "pdf"): fig.savefig(OUT / f"{fname}{SUF}.{ext}", dpi=200)
    plt.close(fig)

def extended(n=10):
    loaders = dict(mnist=lambda i: celeba_row("mnist", i), natural=lambda i: celeba_row("natural", i), cifar=lambda i: celeba_row("cifar", i),
                   b2s100=lambda i: b2s_row(100, i), b5s=istd_row)
    all_rows, all_labels = [], []
    for bench, rl in ROWLAB:
        o = bench_psnr_order(bench); sel = [int(o[int(q * (len(o) - 1))]) for q in np.linspace(0.05, 0.95, n)][::-1]
        rows = [loaders[bench](i) for i in sel]; labels = [f"{rl}\n#{i}" for i in sel]
        draw_rows(rows, labels, f"fig_qual_all_image_benches_extended_{bench}", f"{rl.replace(chr(10), ' ')}: {n} images at the 5-95% quantiles of our on-mask PSNR")
        print(bench, "rows", sel); all_rows += rows; all_labels += labels
    draw_rows(all_rows, all_labels, "fig_qual_all_image_benches_extended", f"Extended: {n} images per bench (5-95% quantiles of our on-mask PSNR)")

if __name__ == "__main__" and "--extended" in sys.argv: extended()

# PAPER selection (user 2026-09-13: "choose some rows that are good, 2 for each task"): taken from the extended 10-row set,
# the two rows per bench where ours is strongest relative to the best competing column (see fig_all_paper_rows.log).
PAPER_ROWS = dict(mnist=[28, 21], natural=[20, 31], cifar=[78, 7], b2s100=[20, 19], b5s=[20, 44])   # user 2026-09-13: natural 20,31; cifar 78,7; 2026-09-24: mnist 3 -> 28 (digit fully visible)
def paper():
    loaders = dict(mnist=lambda i: celeba_row("mnist", i), natural=lambda i: celeba_row("natural", i), cifar=lambda i: celeba_row("cifar", i),
                   b2s100=lambda i: b2s_row(100, i), b5s=istd_row)
    rows, labels = [], []
    for bench, rl in ROWLAB:
        for i in PAPER_ROWS[bench]: rows.append(loaders[bench](i)); labels.append(f"{rl}\n#{i}")
    draw_rows(rows, labels, "fig_qual_all_image_benches_paper", "Two images per bench; PSNR on the corrupted region")
    for d, l in zip(rows, labels): print(l.replace(chr(10), " "), " | ".join(f"{k}: {psnr(np.clip(d[k],0,1), d['gt'], d['M']):.2f}" for k, _ in COLS if k != "gt" and d.get(k) is not None))
if __name__ == "__main__" and "--paper" in sys.argv: paper()
