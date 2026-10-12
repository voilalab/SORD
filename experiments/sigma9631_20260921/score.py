"""Paired scoring of the 0.9 -> 0.6 -> 0.3 -> 0.1 (M=4) x-step schedule campaign (…s9631 runs) against the SHIPPED paper arms, per bench.
BASE (blind) arms only.
Every cell is computed on the images / audio cells whose s9631 chunk has finished (n printed), and the shipped arm is
restricted to the SAME items, so every delta is paired. Images: on-Omega PSNR @ bench resolution, mean ± std (sample) /
worst. Audio: win-SI-SDR abs and improvement, mean ± std.
Run: python -m experiments.sigma9631_20260921.score  -> experiments/sigma9631_20260921/results.md
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
REPO = next(p for p in Path(__file__).resolve().parents if (p / "sord").is_dir()); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "paper/metrics"))
import arms_faces as AF, arms_wm as AW, arms_shadow as AS                     # noqa: E402
X = REPO / "experiments/extbench/results"; R5 = REPO / "experiments/ros/results/r5"
HERE = Path(__file__).resolve().parent
MD = ["# x-step schedule 0.9 -> 0.6 -> 0.3 -> 0.1 (M=4) vs shipped schedule (0.4 -> 0.1), PAIRED on finished items", "",
      "on-Omega PSNR: mean ± std / worst (n). Delta = s9631 arm minus shipped headline on the same images (wins/n).", ""]


def psnr_om(a, b, m):
    e = ((np.clip(a, 0, 1).astype(np.float64) - b) ** 2)[m > 0.5].mean(); return 10 * np.log10(1 / max(e, 1e-20))


def fmt(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if v.size == 0: return "--"
    sd = f" ± {v.std(ddof=1):.2f}" if v.size > 1 else ""
    return f"{v.mean():.2f}{sd} / {v.min():.2f} (n={v.size})"


def dfmt(a, b):
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() == 0: return "--"
    d = a[ok] - b[ok]; return f"{d.mean():+.2f} ({int((d > 0).sum())}/{ok.sum()})"


def match_rows(bench_xA, xA):
    """bench index for each row of xA (exact content match)."""
    n = bench_xA.shape[0]; idx = []
    for x in xA:
        i = int(np.argmin(np.abs(bench_xA - x).reshape(n, -1).mean(1)))
        assert np.abs(bench_xA[i] - x).max() < 1e-4, "x_A match failed"; idx.append(i)
    return np.array(idx)


def per_image(bench_xA, M3, xA_rows, x_rows):
    """per-bench-index PSNR array (nan where missing) from arm rows matched by x_A."""
    out = np.full(bench_xA.shape[0], np.nan)
    if xA_rows is None or len(xA_rows) == 0: return out
    k = len(xA_rows)   # positional pairing when the rows are an in-order prefix of the bench (ISTD b5s has duplicate GTs: 46 unique / 100)
    if k <= bench_xA.shape[0] and xA_rows.shape[1:] == bench_xA.shape[1:] and np.abs(bench_xA[:k] - xA_rows).max() < 1e-4:
        for i in range(k): out[i] = psnr_om(x_rows[i], bench_xA[i], M3[i])
        return out
    idx = match_rows(bench_xA, xA_rows)
    for i, r in zip(idx, range(len(idx))): out[i] = psnr_om(x_rows[r], bench_xA[i], M3[i])
    return out


def load_chunks(pattern, keys):
    fs = sorted(glob.glob(pattern), key=lambda s: int(Path(s).parent.name[1:]))
    if not fs: return None, {}
    z = [np.load(f) for f in fs]
    xA = np.concatenate([q["x_A"] for q in z])
    d = {}
    for k in keys:
        if all(k in q.files for q in z): d[k] = np.concatenate([q[k] for q in z])
    fired = []
    for f in z: pass
    return xA, d


def section(title, bench_xA, M3, shipped, s9631_pattern, s9631_keys, note=""):
    """shipped: list of (label, xA_rows, x_rows). s9631_keys: list of (label, npz key)."""
    xA8, d8 = load_chunks(s9631_pattern, [k for _, k in s9631_keys])
    n8 = 0 if xA8 is None else len(xA8)
    MD.extend([f"### {title}  (s9631 finished {n8}/{bench_xA.shape[0]})" + (f" -- {note}" if note else ""), ""])
    if n8 == 0: MD.extend(["_no s9631 chunks finished yet_", ""]); return
    have = ~np.isnan(per_image(bench_xA, M3, xA8, xA8))
    rows = []; head = None
    for lab, xa, xr in shipped:
        p = per_image(bench_xA, M3, xa, xr); p = np.where(have, p, np.nan); rows.append((lab, p))
        if head is None: head = p
    c8 = []
    for lab, k in s9631_keys:
        if k in d8: c8.append((f"0.9/0.6/0.3/0.1: {lab}", per_image(bench_xA, M3, xA8, d8[k])))
    MD.extend(["| arm | PSNR_Omega mean ± std / worst (n) | delta vs shipped headline |", "|---|---|---|"])
    for lab, p in rows: MD.append(f"| shipped: {lab} | {fmt(p)} | {dfmt(p, head)} |")
    for lab, p in c8: MD.append(f"| {lab} | {fmt(p)} | {dfmt(p, head)} |")
    MD.append("")


# ======================================================================================= faces (64^2, CelebA prior)
MD.extend(["## tab:faces (natural / mnist / cifar, n=100)", ""])
for b, name in (("natural", "audit/s9631"), ("mnist", "audit/s9631"), ("cifar", "cifar/min_exact_s9631")):
    try:
        base = AF.ours_base(b, "x_alt", None)(); inp = AF.ours_base(b, "x_inp", None)()["x"]; last = AF.XLAST[b]
        section(f"faces {b}", base["gt"], base["M"], [("TANGO-Min K32 (x_alt)", base["gt"], base["x"]), ("inpaint m0", base["gt"], inp)],
                str(R5 / f"{name}/c*/{AF.OODARG[b] if hasattr(AF, 'OODARG') else ('mnist' if b == 'mnist' else 'natural')}_*_sl0.9-0.6-0.3-0.1_*blind.npz"),
                [("gate/ship (x_alt)", "x_alt"), ("m1", "x_m1"), ("last (x_m4)", "x_m4"), ("inpaint m0", "x_inp")])
    except Exception as ex: MD.extend([f"### faces {b}: FAIL {type(ex).__name__}: {str(ex)[:120]}", ""])

# ======================================================================================= watermark 64^2 (b2r / prwd)
MD.extend(["## tab:wm 64^2 (b2r n=99, prwd n=100)", ""])
try:
    d = AW._b2r(); B = d["B"]; M3 = np.repeat(B["M_true"].astype(np.float32)[..., None], 3, -1)
    bl = AW.b2r_min_blind(); inp = AW.b2r_inpaint()
    section("b2r blind (minDr)", B["x_A"], M3, [("blind n (x_alt)", bl["gt"], bl["x"]), ("inpaint m0", inp["gt"], inp["x"])],
            str(X / "b2r/minDr_s9631/c*/*.npz"), [("gate/ship (x_alt)", "x_alt"), ("m1", "x_m1"), ("last (x_m4)", "x_m4"), ("inpaint m0", "x_inp")])
except Exception as ex: MD.extend([f"### b2r: FAIL {type(ex).__name__}: {str(ex)[:120]}", ""])
try:
    d = AW._prwd(); B = d["B"]; M3 = d["M"]
    bl = AW.prwd_min_blind(); inp = AW.prwd_inpaint()
    section("prwd blind (minE)", B["x_A"], M3, [("blind n (x_blind)", bl["gt"], bl["x"]), ("inpaint m0", inp["gt"], inp["x"])],
            str(X / "prwd/minE_s9631/c*/*.npz"), [("blind readout (x_blind)", "x_blind"), ("gate/ship (x_alt)", "x_alt"), ("m1", "x_m1"), ("inpaint m0", "x_inp")])
except Exception as ex: MD.extend([f"### prwd: FAIL {type(ex).__name__}: {str(ex)[:120]}", ""])

# ======================================================================================= watermark native 256
MD.extend(["## tab:wm native 256 (ffhqwm0 / ffhqwmU0 / imagenetwm, n=25)", ""])
for b in ("ffhqwm0", "ffhqwmU0", "imagenetwm"):
    B = np.load(X / f"{b}_native/bench_{b}_native.npz"); M3 = np.repeat(B["M_true"].astype(np.float32)[..., None], 3, -1)
    try:
        m1 = AW.nat_m1(b); inp = AW.nat_inpaint(b); run8 = "m4s9631" if b != "imagenetwm" else "min256in_s9631"
        section(f"{b} blind ({AW.RUN[b]})", B["x_A"], M3, [("blind m1", m1["gt"], m1["x"]), ("inpaint m0", inp["gt"], inp["x"])],
                str(X / f"{b}_native/{run8}/c*/*.npz"), [("m1", "x_m1"), ("gate/ship (x_alt)", "x_alt"), ("inpaint m0", "x_inp")])
    except Exception as ex: MD.extend([f"### {b} blind: FAIL {type(ex).__name__}: {str(ex)[:120]}", ""])

# ======================================================================================= ISTD b5s
MD.extend(["## tab:b5s (ISTD 64^2, n=100)", ""])
try:
    bb = AS._b5s()
    mult = AS.b5s_arm("multminEc", "x_alt", None)(); add = AS.b5s_arm("minEc", "x_alt", None)(); inp = AS.b5s_arm("multminEc", "x_inp", None)()
    section("b5s multiplicative (multminEc)", bb["gt"], bb["M"], [("TANGO-Min multiplicative (x_alt)", mult["gt"], mult["x"]), ("inpaint m0", inp["gt"], inp["x"])],
            str(X / "b5s/multminEc_s9631/c*/*.npz"), [("gate/ship (x_alt)", "x_alt"), ("m1", "x_m1"), ("m2", "x_m2"), ("last (x_m4)", "x_m4"), ("inpaint m0", "x_inp")])
    section("b5s additive (minEc)", bb["gt"], bb["M"], [("TANGO-Min additive (x_alt)", add["gt"], add["x"])],
            str(X / "b5s/minEc_s9631/c*/*.npz"), [("gate/ship (x_alt)", "x_alt"), ("m1", "x_m1"), ("last (x_m4)", "x_m4"), ("inpaint m0", "x_inp")])
except Exception as ex: MD.extend([f"### b5s: FAIL {type(ex).__name__}: {str(ex)[:120]}", ""])

# ======================================================================================= shadow transfer v1 / v2
for v2 in (False, True):
    MD.extend([f"## tab:shadow-transfer-{'v2' if v2 else 'v1'} (native 256, n=25)", ""])
    for b in ("ffhqsh0", "ffhqsh", "imagenetsh", "lsunsh"):
        s = AS._SUF[b]; tag, bb = AS._sh(b, v2)
        try:
            m = AS.sh_arm(b, f"mult256{s}", "x_alt", None, v2=v2)()
            section(f"{tag} multiplicative gate (mult256{s})", bb["gt"], bb["M"], [("multiplicative gate (x_alt)", m["gt"], m["x"])],
                    str(X / f"{tag}_native/mult256{s}_s9631/c*/*.npz"), [("gate/ship (x_alt)", "x_alt"), ("m1", "x_m1"), ("last (x_m4)", "x_m4"), ("inpaint m0", "x_inp")])
        except Exception as ex: MD.extend([f"### {tag} mult: FAIL {type(ex).__name__}: {str(ex)[:120]}", ""])
        if not v2:
            try:
                a = AS.sh_arm(b, f"min256{s}", "x_m1", None)()
                section(f"{tag} additive m1 (min256{s})", bb["gt"], bb["M"], [("additive m1", a["gt"], a["x"])],
                        str(X / f"{tag}_native/min256{s}_s9631/c*/*.npz"), [("m1", "x_m1"), ("gate/ship (x_alt)", "x_alt"), ("inpaint m0", "x_inp")])
            except Exception as ex: MD.extend([f"### {tag} add: FAIL {type(ex).__name__}: {str(ex)[:120]}", ""])

# ======================================================================================= audio (e3_blind loop arms)
MD.extend(["## audio (e3_blind loop arms; s9631 == --loop-sig-list 3.6,2.4,1.2,0.4 --M 4 in SIGMA_DATA units)", "",
           "win-SI-SDR abs mean ± std | improvement mean ± std | >= input", ""])
A = REPO / "experiments/tango_msdm_2026_08_04/results/arms"; DA = REPO / "experiments/tango_msdm_2026_08_04/external_sources"
C8 = DA / "arms_s9631"


def acells(paths):
    out = {}
    for p in paths:
        p = Path(p)
        if not p.exists(): continue
        s = json.load(open(p)); cs = s["cells"] if isinstance(s, dict) else s
        for c in cs: out[(int(c["cell"][0]), c["cell"][1], float(c["cell"][2]))] = (c["win_sisdr"], c["win_sisdr_in"])
    return out


def ams(a): a = np.asarray(a, float); return f"{a.mean():+.2f} ± {a.std(ddof=1):.2f}" if a.size > 1 else f"{a.mean():+.2f}"


AUD = [("piano b50 (K4 M4 blind window)", [DA / "arms_b50/e3_blind_b50/summary.json"], [C8 / "e3_blind_b50_s9631_p0/summary.json", C8 / "e3_blind_b50_s9631_p1/summary.json"], 200),
       ("guitar", [A / "guitar_blind/summary.json"], [C8 / "guitar_blind_s9631/summary.json"], 100),
       ("guitar, fresh clips", [A / "guitar_blind_x/summary.json"], [C8 / "guitar_blind_x_s9631/summary.json"], 100),
       ("piano + solo strings", [A / "e3_blind_strsolo/summary.json"], [C8 / "e3_blind_strsolo_s9631/summary.json"], 46),
]
MD.extend(["| bench | n paired | shipped abs | shipped impr | s9631 abs | s9631 impr | paired delta (wins/n) | >= input shipped / s9631 |", "|---|---|---|---|---|---|---|---|"])
for name, sp, cp, ntot in AUD:
    s = acells(sp); c = acells(cp); kk = sorted(set(s) & set(c))
    if not kk: MD.append(f"| {name} | 0/{ntot} | {ams([v[0] for v in s.values()]) if s else '--'} | -- | -- | -- | -- | -- |"); continue
    sa = np.array([s[k][0] for k in kk]); ca = np.array([c[k][0] for k in kk]); inp = np.array([s[k][1] for k in kk])
    d = ca - sa
    MD.append(f"| {name} | {len(kk)}/{ntot} | {ams(sa)} | {ams(sa - inp)} | {ams(ca)} | {ams(ca - inp)} | {d.mean():+.2f} ({int((d > 0).sum())}/{len(kk)}) | {int((sa >= inp).sum())} / {int((ca >= inp).sum())} |")
MD.append("")
MD.extend(["Not applicable: tab:balanced SIR rows, tab:bss, tab:audio_iv100, tab:audio_piano_classes and the speech-target rows report the",
           "K-chain inpaint FLOOR (min_floor_eval / floor_win, zero loop iterations); there is no x-step schedule to set to 0.8.", ""])

md = "\n".join(MD); print(md); open(HERE / "results.md", "w").write(md + "\n")
