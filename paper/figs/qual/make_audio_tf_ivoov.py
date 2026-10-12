"""10-row time-frequency figure, piano target (2026-09-13 user spec): top half = IN-VOCABULARY interferers
(Drums / Guitar / Bass class-median cells of our window gain on the Slakh b50 bench + two LibriSpeech speech cells at the
0.5 / 0.25 quantiles), bottom half = OUT-OF-VOCABULARY interferers (class-median cell of the 5 largest OOV classes:
Strings (continued), Organ, Reed, Brass + a seeded pick among the 6-cell classes -> Synth Pad).
Columns: mixture y | interferer (true) | MSDM | ours (piano prior only, window given, K=64) | clean piano.
MSDM column = the model that HAS the interferer for in-vocab instrument rows (published 4-stem bass/drums/guitar/piano)
and the published 4-stem model on the OOV rows (their paper's setting).  Speech rows come in two variants:
  speech2: MSDM = our 2-source [speech DM (best-val 108k), piano DM]         (results/arms/e1_msdm_speech_best108k)
  speech5: MSDM = 5-source [speech DM + the 4 published instrument DMs]      (/data/.../arms_speech/e1_msdm_5stem_speech)
Under each panel: window SI-SDR (gain over input; LSD dropped 2026-09-13 per user); recomputed from waveforms and
asserted equal (<0.05 dB) to the table summaries.  Ours waveforms: results/arms/figS_* (min_floor_eval --save-npz).
python paper/figs/qual/make_audio_tf_ivoov.py [v2] [--rows]   (v2 = second seeded draw, disjoint cells)
"""
import json, sys
from pathlib import Path
REPO = next(p for p in Path(__file__).resolve().parents if (p / "sord").is_dir())
import numpy as np
from scipy.signal import stft
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ARMS = Path("experiments/tango_msdm_2026_08_04/results/arms"); D = REPO / "experiments/tango_msdm_2026_08_04/external_sources"
OUT = Path("paper/figs/qual"); SR = 22050
def cells(p): return {tuple(c["cell"]): c for c in json.load(open(p / "summary.json"))["cells"]}
def sisdr(est, ref):
    a = np.dot(est, ref) / np.dot(ref, ref); t = a * ref
    return 10 * np.log10(np.dot(t, t) / np.dot(est - t, est - t))
def lsd(est, ref, m):
    _, _, E = stft(est * m, fs=SR, nperseg=1024, noverlap=768); _, _, R = stft(ref * m, fs=SR, nperseg=1024, noverlap=768)
    return float(np.sqrt(((20 * np.log10(np.abs(E) + 1e-4) - 20 * np.log10(np.abs(R) + 1e-4)) ** 2).mean()))
def spec(x):
    f, t, Z = stft(x, fs=SR, nperseg=1024, noverlap=768); return f, t, 20 * np.log10(np.abs(Z) + 1e-6)
gain = lambda c, k: c[k]["win_sisdr"] - c[k]["win_sisdr_in"]
def quant(sub, c, q): sub = sorted(sub, key=lambda k: gain(c, k)); return sub[int(q * (len(sub) - 1))]
# ---- instrument bench (b50): ours = floorO_K64_b50a+b, MSDM 4 stems = e1_msdm_dirac_b50
co = {}; [co.update(cells(ARMS / f"floorO_K64_b50{p}")) for p in "ab"]
cm4 = cells(D / "arms_b50/e1_msdm_dirac_b50"); ks = sorted(set(co) & set(cm4))
# ---- speech bench: ours = floorO_K64_speech_p0/p1; MSDM 2-source = e1_msdm_speech_best108k; 5-source = e1_msdm_5stem_speech
cs = {}; [cs.update(cells(ARMS / f"floorO_K64_speech_{p}")) for p in ("p0", "p1")]
cs2 = cells(ARMS / "e1_msdm_speech_best108k"); cs5 = cells(D / "arms_speech/e1_msdm_5stem_speech")
kss = sorted(set(cs) & set(cs2) & set(cs5))
rows = [("b50", quant([k for k in ks if k[1] == "invocab" and co[k]["inst"] == inst], co, 0.5)) for inst in ("Drums", "Guitar", "Bass")]
rows += [("speech", quant(kss, cs, q)) for q in (0.5, 0.25)]
from collections import Counter
cnt = Counter(co[k]["inst"] for k in ks if k[1] == "oov"); rng = np.random.default_rng(20260913)
oov_cls = [c for c, _ in cnt.most_common(4)]; ties = [c for c in cnt if cnt[c] == 6]; oov_cls.append(ties[rng.integers(len(ties))])
rows += [("b50", quant([k for k in ks if k[1] == "oov" and co[k]["inst"] == inst], co, 0.5)) for inst in oov_cls]
assert len(rows) == 10
VARIANT = next((a for a in sys.argv[1:] if not a.startswith("--")), "")   # "" = v1 (class medians); "v2" = seeded random draw
if VARIANT == "v2":
    # v2 (user: "another image with different samples"): same strata, one seeded random draw per stratum from the
    # 0.2-0.8 band of our gain (speech: two draws, upper/lower half), never a v1 cell
    rng2 = np.random.default_rng(11); used = {k for _, k in rows}
    def band(sub, c, lo, hi): sub = sorted(sub, key=lambda k: gain(c, k)); n = len(sub); return sub[int(lo * (n - 1)):int(hi * (n - 1)) + 1]
    def draw(sub): sub = [k for k in sub if k not in used]; k = sub[rng2.integers(len(sub))]; used.add(k); return k
    rows = [("b50", draw(band([k for k in ks if k[1] == "invocab" and co[k]["inst"] == inst], co, 0.2, 0.8))) for inst in ("Drums", "Guitar", "Bass")]
    rows += [("speech", draw(band(kss, cs, lo, hi))) for lo, hi in ((0.5, 0.9), (0.1, 0.5))]
    rows += [("b50", draw(band([k for k in ks if k[1] == "oov" and co[k]["inst"] == inst], co, 0.2, 0.8))) for inst in oov_cls]
SUFFIX = f"_{VARIANT}" if VARIANT else ""
# 2026-09-23 user: "--bare" = no column titles, no row labels, no in-panel MSDM tag, no axis-name labels;
# SI-SDR captions and the in-vocab/OOV divider are kept.  Output suffix "_bare".
BARE = "--bare" in sys.argv
if BARE: SUFFIX += "_bare"
# 2026-09-24 user: "--collabels" = bare figure but WITH the column titles on the first row
COLLABELS = "--collabels" in sys.argv
if COLLABELS: SUFFIX += "_labels"
if "--rows" in sys.argv: print("rows:", rows); sys.exit()
COLS = ["mixture $y$", "interferer (true)", "MSDM", "SORD\n(ours)", "clean piano"]
def render(variant):
    fig, axes = plt.subplots(len(rows), 5, figsize=(13.0, 2.1 * len(rows) + 0.8), sharex=True, sharey=True)
    for r, (bench, k) in enumerate(rows):
        ci, pool, sir = k
        if bench == "b50":
            Do = np.load(ARMS / f"figS_b50_c{ci}_{pool}_s{sir:g}" / f"c{ci}_{pool}_s{sir:g}.npz"); Dm = np.load(D / f"arms_b50/e1_msdm_dirac_b50/c{ci}_{pool}_s{sir:g}.npz")
            cm, csum, inst = cm4[k], co[k], co[k]["inst"]; mtag = "4 instrument DMs"
        else:
            Do = np.load(ARMS / f"figS_speech_c{ci}_s{sir:g}" / f"c{ci}_{pool}_s{sir:g}.npz"); csum, inst = cs[k], "Speech"
            if variant == "speech2": Dm = np.load(ARMS / f"e1_msdm_speech_best108k/c{ci}_{pool}_s{sir:g}.npz"); cm = cs2[k]; mtag = "piano DM + speech DM"
            else: Dm = np.load(D / f"arms_speech/e1_msdm_5stem_speech/c{ci}_{pool}_s{sir:g}.npz"); cm = cs5[k]; mtag = "speech DM + 4 instrument DMs"
        assert np.allclose(Do["y"], Dm["y"], atol=1e-4) and ((Do["mask"] > 0.5) == (Dm["mask"] > 0.5)).all(), ("arms not on the same cell", k)
        mw = Do["mask"] > 0.5; cl = Do["clean"]
        sigs = [Do["y"], Do["n_true"], Dm["piano_est"], Do["est"], cl]
        sc = [(sisdr(x[mw], cl[mw]), lsd(x, cl, mw)) for x in sigs[:4]]
        for ref_json, j in ((cm, 2), (csum, 3)): assert abs(ref_json["win_sisdr"] - sc[j][0]) < 0.05, (k, j, ref_json["win_sisdr"], sc[j][0])
        _, _, Sc = spec(cl); vmax = Sc.max(); vmin = vmax - 70
        w0, w1 = np.argmax(mw) / SR, (len(mw) - np.argmax(mw[::-1])) / SR
        for c, x in enumerate(sigs):
            ax = axes[r, c]; f, t, S = spec(x)
            ax.pcolormesh(t, f / 1000, S, vmin=vmin, vmax=vmax, cmap="magma", shading="auto", rasterized=True)
            for w in (w0, w1): ax.axvline(w, color="w", ls="--", lw=0.8)
            if r == 0 and (not BARE or COLLABELS): ax.set_title(COLS[c], fontsize=9)
        if not BARE: axes[r, 2].text(0.02, 0.97, mtag, ha="left", va="top", fontsize=7, color="w", transform=axes[r, 2].transAxes,
                        bbox=dict(boxstyle="round,pad=0.2", fc="k", ec="none", alpha=0.55))
        vocab = "in-vocab" if pool in ("invocab", "speech") else "OOV"
        if not BARE: axes[r, 0].set_ylabel(f"{inst} ({vocab})\nc{ci} SIR{sir:g}  kHz", fontsize=7)
        axes[r, 0].set_xlabel(f"input SI-SDR {sc[0][0]:.1f} dB", fontsize=7)
        for j in (2, 3): axes[r, j].set_xlabel(f"SI-SDR {sc[j][0]:.1f} dB ({sc[j][0]-sc[0][0]:+.1f})", fontsize=7)
        if not BARE: axes[r, 4].set_xlabel("time [s]", fontsize=7.5)
    axes[0, 0].set_ylim(0, 8)
    # divider between the in-vocab half and the OOV half
    fig.tight_layout(pad=0.3)
    b4, b5 = axes[4, 0].get_position(), axes[5, 0].get_position()
    y_div = b5.y1 + 0.22 * (b4.y0 - b5.y1)      # below row 5's xlabels, above row 6's frames
    fig.add_artist(plt.Line2D([0.01, 0.99], [y_div, y_div], color="0.4", lw=1.0, ls=(0, (4, 3)), transform=fig.transFigure))
    for ext in ("png", "pdf"): fig.savefig(OUT / f"fig_audio_tf_ivoov_{variant}{SUFFIX}.{ext}", dpi=170)
    plt.close(fig); print("wrote", variant)
for v in ("speech2", "speech5"): render(v)
print("rows:", rows)
