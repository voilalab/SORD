"""Extended audio table (2026-09-22): IV / OOV / Speech pools, mean +- std over cells, old + new clips pooled.
Old arms: balGate_K64_<t> (floor_win) + balsir_msdm_<t>[_h*] + rawgbal_{ours,msdm}_<t>; speech rows floorO_K64_speech_p0/p1,
e1_msdm_speech_best108k (+speech DM), arms_speech/e1_msdm_4stem_speech, guitar floorO_K64_guitar_speech / e1_msdm_speech_guitar_60k.
New arms (ext_20260922): ext_floor_<t>_h{1,2}, ext_msdm_<t>_h{1,2}, ext_floor_g1_<t>, ext_msdm_g1_<t> on pool bal2;
sp_floor_<t>_h*, sp_msdm4_<t>_h*, sp_msdm2_<t>_h* on the speech pool of the 440 balanced clips.
Missing arms are skipped (partial tables while the queue runs).
python -m experiments.tango_msdm_2026_08_04.src.audio_ext_tables [--new-only]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
E = REPO / "experiments/tango_msdm_2026_08_04"; ARMS = E / "results/arms"
D = REPO / "experiments/tango_msdm_2026_08_04/external_sources"; EXT = D / "arms_ext"
STEMS = {"Bass", "Drums", "Guitar", "Piano"}
TARGETS = ("Piano", "Guitar", "Bass", "Drums")


def cells(p):
    p = Path(p)
    if not p.exists(): return []
    s = json.load(open(p)); return s["cells"] if isinstance(s, dict) else s


def imp_map(paths, floor=False):
    out = {}
    for p in paths:
        for c in cells(p):
            w = c["floor_win"] if floor else c["win_sisdr"]
            out[(int(c["cell"][0]), c["cell"][1], float(c["cell"][2]))] = dict(
                imp=w - c["win_sisdr_in"], inst=c.get("inst", "?"))
    return out


def merged(stem):
    return [q for q in [stem / "summary.json"] + sorted(stem.parent.glob(stem.name + "_h*/summary.json")) if q.exists()]


def ms(a):
    a = np.asarray(a, float)
    return f"{a.mean():+.2f} ± {a.std(ddof=1):.2f}" if len(a) > 1 else (f"{a.mean():+.2f}" if len(a) else "--")


def pair(o, m, sir, pool_fn=None):
    kk = sorted(k for k in o if k[2] == sir and k in m and (pool_fn is None or pool_fn(o[k]["inst"])))
    a = np.array([o[k]["imp"] for k in kk]); b = np.array([m[k]["imp"] for k in kk])
    return a, b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new-only", action="store_true"); a = ap.parse_args()
    rows = {"IV": [], "OOV": [], "Speech": []}
    for T in TARGETS:
        t = T.lower()
        # ---- SIR 0/5 and g=1: old bal + new bal2
        old_sir = (imp_map(merged(ARMS / f"balGate_K64_{t}"), floor=True), imp_map(merged(D / f"arms_bal/balsir_msdm_{t}")))
        old_g1 = (imp_map(merged(ARMS / f"rawgbal_ours_{t}")), imp_map(merged(D / f"arms_bal/rawgbal_msdm_{t}")))
        new_sir = (imp_map(merged(ARMS / f"ext_floor_{t}")), imp_map(merged(EXT / f"ext_msdm_{t}")))
        new_g1 = (imp_map([ARMS / f"ext_floor_g1_{t}/summary.json"]), imp_map([EXT / f"ext_msdm_g1_{t}/summary.json"]))
        for cname, sir, srcs in (("SIR 0", 0.0, (old_sir, new_sir)), ("SIR 5", 5.0, (old_sir, new_sir)), ("g=1", 1.0, (old_g1, new_g1))):
            for pool in ("IV", "OOV"):
                fn = (lambda i: (i.split(" ")[0] in STEMS)) if pool == "IV" else (lambda i: (i.split(" ")[0] not in STEMS))
                A, B, n_old = [], [], 0
                for j, (o, m) in enumerate(srcs):
                    if a.new_only and j == 0: continue
                    x, y = pair(o, m, sir, fn); A += list(x); B += list(y); n_old += len(x) if j == 0 else 0
                A, B = np.array(A), np.array(B)
                if len(A) == 0: continue
                rows[pool].append([T, cname, pool, f"{len(A)} ({n_old}+{len(A) - n_old})", ms(A), ms(B), ms(A - B), f"{(A > B).sum()}/{len(A)}"])
        # ---- speech interferer: new 110 clips (+ old piano 50 / guitar 25 where the arm is identical)
        sp_o = imp_map(merged(ARMS / f"sp_floor_{t}")); sp4 = imp_map(merged(EXT / f"sp_msdm4_{t}")); sp2 = imp_map(merged(EXT / f"sp_msdm2_{t}"))
        old = {}
        if T == "Piano":
            po = imp_map([ARMS / "floorO_K64_speech_p0/summary.json", ARMS / "floorO_K64_speech_p1/summary.json"])
            old = {"4stem": (po, imp_map([D / "arms_speech/e1_msdm_4stem_speech/summary.json"])),
                   "2src": (po, imp_map([ARMS / "e1_msdm_speech_best108k/summary.json"]))}
        # guitar old +speech row used speech ckpt step060000 (not best.pt) -> not pooled
        for label, new, key in (("Speech", (sp_o, sp4), "4stem"), ("Speech†", (sp_o, sp2), "2src")):
            for cname, sir in (("SIR 0", 0.0), ("SIR 5", 5.0)):
                A, B, n_old = [], [], 0
                if not a.new_only and key in old:
                    x, y = pair(*old[key], sir); A += list(x); B += list(y); n_old = len(x)
                x, y = pair(*new, sir); A += list(x); B += list(y)
                A, B = np.array(A), np.array(B)
                if len(A) == 0: continue
                rows["Speech"].append([T, cname, label, f"{len(A)} ({n_old}+{len(A) - n_old})", ms(A), ms(B), ms(A - B), f"{(A > B).sum()}/{len(A)}"])
    hdr = ["target", "cond", "pool", "n (old+new)", "ours", "MSDM", "paired Δ", "ours up"]
    md = ["# Extended audio table (IV / OOV / Speech), mean ± std of win-SI-SDRi over cells", "",
          "Ours = one prior (target's own), floor K64. MSDM = 4 priors; Speech† = MSDM with an added LibriSpeech prior (5 priors).",
          "IV/OOV rows pool the paper's 'bal' clips (30 IV / 80 OOV per row) with the new 'bal2' clips (same quotas, new windows).",
          "Speech rows: new speech pool on the 440 balanced clips (110 per target) + the paper's 50 piano clips where the MSDM arm is identical.", ""]
    tex = ["% extended audio table 2026-09-22 (audio_ext_tables.py)", "\\begin{tabular}{" + "l" * len(hdr) + "}", "\\toprule",
           " & ".join(hdr).replace("±", "$\\pm$").replace("Δ", "$\\Delta$") + "\\\\", "\\midrule"]
    for pool in ("IV", "OOV", "Speech"):
        md += [f"### {pool}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        md += ["| " + " | ".join(r) + " |" for r in rows[pool]] + [""]
        tex += [f"\\multicolumn{{{len(hdr)}}}{{l}}{{\\emph{{{pool}}}}}\\\\"]
        tex += [" & ".join(r).replace("±", "$\\pm$").replace("Δ", "$\\Delta$").replace("†", "$^\\dagger$") + "\\\\" for r in rows[pool]] + ["\\midrule"]
    tex = tex[:-1] + ["\\bottomrule", "\\end{tabular}"]
    tag = "_newonly" if a.new_only else ""
    (E / f"ext_20260922/table_ext{tag}.md").write_text("\n".join(md) + "\n")
    (REPO / "paper/tables" / f"audio_mean_std_ivoov_speech_ext{tag}.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
