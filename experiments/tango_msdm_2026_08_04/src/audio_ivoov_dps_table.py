"""IV / OOV / speech audio table with two baselines (user 2026-09-23): rows piano+IV, guitar+IV, piano+speech (speech in MSDM's
vocabulary = MSDM given the LibriSpeech prior), and piano+OOV, guitar+OOV, piano+speech OOV (published 4-stem MSDM), SIR 0 / 5.
Columns: ours (one prior, floor K64), MSDM (4 priors; speech-IV rows: + speech prior), DPS with the same single target prior
(y-init, sigma_max 0.1, 250 steps = the paper's audio DPS row; prior-of-target only, so it applies to IV and OOV alike).
All entries mean ± std of win-SI-SDRi over cells; paired deltas on the same cells.
Pools: IV/OOV = balanced 'bal' + 'bal2' clips (piano / guitar / bass / drums targets); speech = the 110 balanced clips per target + the paper's 50 piano clips.
Bass/drums speech g=1 arms were never run -> those rows come out with n=0 ('--') and are skipped by localized_table.py.
DPS arms (queue dps_ivoov_20260923): dpsW = DPS-inpaint (sigma inf), dpsG = DPS-Gauss (sigma = mean|n| in window, oracle), dpsA = adaptive DPS mu = mean
of the interferer in the window (oracle), dpsA0 = adaptive DPS mu = 0; tags <v>_{bal,bal2,sp}_{piano,guitar}_h{1,2}, <v>_sp50_piano, <v>_g1_{bal,bal2}_<t>_h{1,2}.
python -m experiments.tango_msdm_2026_08_04.src.audio_ivoov_dps_table
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from experiments.tango_msdm_2026_08_04.src.audio_ext_tables import imp_map, merged, ARMS, D, EXT, E, REPO, STEMS

def ms(a):
    a = np.asarray(a, float)
    return "--" if len(a) == 0 else (f"{a.mean():+.2f} ± {a.std(ddof=1):.2f}" if len(a) > 1 else f"{a.mean():+.2f}")

VARS = ("dpsW", "dpsG", "dpsA", "dpsA0")
VLAB = {"dpsW": "DPS-inpaint", "dpsG": "DPS-Gauss (oracle sigma)", "dpsA": "adaptive DPS mu = n̄ (oracle)", "dpsA0": "adaptive DPS mu = 0"}

def dvar(stem):
    """stem like 'dpsW_bal_piano' -> {variant: imp_map} for all four DPS variants (h1/h2 merged)."""
    return {v: imp_map(merged(stem.parent / stem.name.replace("dpsW", v, 1))) for v in VARS}

def row(name, cond, pool, srcs, sir, fn=None):
    """srcs = [(ours, msdm, {variant: dps}) ...]; every arm paired with ours on its own finished cells."""
    A, B = np.array([]), np.array([]); C = {v: [np.array([]), np.array([])] for v in VARS}
    for o, m, dv in srcs:
        kk = sorted(k for k in o if k[2] == sir and k in m and (fn is None or fn(o[k]["inst"])))
        A = np.r_[A, [o[k]["imp"] for k in kk]]; B = np.r_[B, [m[k]["imp"] for k in kk]]
        for v in VARS:
            kd = [k for k in kk if k in dv[v]]; C[v][0] = np.r_[C[v][0], [dv[v][k]["imp"] for k in kd]]; C[v][1] = np.r_[C[v][1], [o[k]["imp"] for k in kd]]
    out = [name, cond, pool, str(len(A)), ms(A), ms(B), ms(A - B), f"{int((A > B).sum())}/{len(A)}"]
    for v in VARS:
        c, ad = C[v]
        out += [f"{ms(c)} (n={len(c)})" if len(c) else "--", f"{ms(ad - c)} ({int((ad > c).sum())}/{len(c)})" if len(c) else "--"]
    return out

def main():
    IV = lambda i: i.split(" ")[0] in STEMS; OOV = lambda i: i.split(" ")[0] not in STEMS
    rows = {"IV": [], "OOV": []}
    for T in ("Piano", "Guitar", "Bass", "Drums"):   # bass/drums added 2026-09-25 (user: tab:localized for all four targets)
        t = T.lower()
        srcs = [(imp_map(merged(ARMS / f"balGate_K64_{t}"), floor=True), imp_map(merged(D / f"arms_bal/balsir_msdm_{t}")), dvar(ARMS / f"dpsW_bal_{t}")),
                (imp_map(merged(ARMS / f"ext_floor_{t}")), imp_map(merged(EXT / f"ext_msdm_{t}")), dvar(ARMS / f"dpsW_bal2_{t}"))]
        for cname, sir in (("SIR 0", 0.0), ("SIR 5", 5.0)):
            rows["IV"].append(row(T, cname, "IV instrument", srcs, sir, IV)); rows["OOV"].append(row(T, cname, "OOV instrument", srcs, sir, OOV))
        # g = 1: raw mixture, no SIR adjustment (rawgain manifests), bal + bal2
        g1 = [(imp_map(merged(ARMS / f"rawgbal_ours_{t}")), imp_map(merged(D / f"arms_bal/rawgbal_msdm_{t}")), dvar(ARMS / f"dpsW_g1_bal_{t}")),
              (imp_map([ARMS / f"ext_floor_g1_{t}/summary.json"]), imp_map([EXT / f"ext_msdm_g1_{t}/summary.json"]), dvar(ARMS / f"dpsW_g1_bal2_{t}"))]
        rows["IV"].append(row(T, "g = 1", "IV instrument", g1, 1.0, IV)); rows["OOV"].append(row(T, "g = 1", "OOV instrument", g1, 1.0, OOV))
        # speech interferer: ours sp_floor (+ paper's 50 piano clips), MSDM 4-stem (OOV) / + speech prior (IV)
        sp_o = imp_map(merged(ARMS / f"sp_floor_{t}")); sp4 = imp_map(merged(EXT / f"sp_msdm4_{t}")); sp2 = imp_map(merged(EXT / f"sp_msdm2_{t}"))
        sp_d = dvar(ARMS / f"dpsW_sp_{t}")
        old4 = old2 = []
        if T == "Piano":
            po = imp_map([ARMS / "floorO_K64_speech_p0/summary.json", ARMS / "floorO_K64_speech_p1/summary.json"]); pd = {v: imp_map([ARMS / f"{v}_sp50_piano/summary.json"]) for v in VARS}
            old4 = [(po, imp_map([D / "arms_speech/e1_msdm_4stem_speech/summary.json"]), pd)]; old2 = [(po, imp_map([ARMS / "e1_msdm_speech_best108k/summary.json"]), pd)]
        for cname, sir in (("SIR 0", 0.0), ("SIR 5", 5.0)):
            rows["IV"].append(row(T, cname, "speech (IV: MSDM + speech prior)", old2 + [(sp_o, sp2, sp_d)], sir))
            rows["OOV"].append(row(T, cname, "speech (OOV: 4-stem MSDM)", old4 + [(sp_o, sp4, sp_d)], sir))
        # speech at g = 1 (raw mixture, manifest_rawgain_speech.json; queue speech_g1_20260924, 2026-09-24): same clips as the
        # SIR rows (110 balanced per target + the paper's 50 piano clips), tags spg1_{floor,msdm4,msdm2}_<t>_{h1,h2,sp50}, dps*_spg1_<t>_*
        g_o = imp_map(merged(ARMS / f"spg1_floor_{t}")); g4 = imp_map(merged(EXT / f"spg1_msdm4_{t}")); g2 = imp_map(merged(EXT / f"spg1_msdm2_{t}"))
        g_d = dvar(ARMS / f"dpsW_spg1_{t}")
        g1_2, g1_4 = [(g_o, g2, g_d)], [(g_o, g4, g_d)]
        if T == "Piano":
            so = imp_map([ARMS / "spg1_floor_piano_sp50/summary.json"]); sd = dvar(ARMS / "dpsW_spg1_piano_sp50")
            g1_2 = [(so, imp_map([EXT / "spg1_msdm2_piano_sp50/summary.json"]), sd)] + g1_2
            g1_4 = [(so, imp_map([EXT / "spg1_msdm4_piano_sp50/summary.json"]), sd)] + g1_4
        rows["IV"].append(row(T, "g = 1", "speech (IV: MSDM + speech prior)", g1_2, 1.0))
        rows["OOV"].append(row(T, "g = 1", "speech (OOV: 4-stem MSDM)", g1_4, 1.0))
    hdr = ["target", "cond", "interferer", "n", "ours", "MSDM", "Δ ours−MSDM", "ours up"] + sum([[VLAB[v], f"Δ ours−{VLAB[v].split(chr(40))[0].strip()} (up)"] for v in VARS], [])
    md = ["# Audio: IV / OOV / speech with two baselines (win-SI-SDRi, mean ± std over cells)", "",
          "ours = one prior (target's own), K-chain floor K64. MSDM = published 4-stem MSDM; speech-IV rows = MSDM given an additional LibriSpeech prior.",
          "DPS = posterior sampling with the SAME single target prior (y-init, sigma_max 0.1, 250 steps; the paper's audio DPS row), prior-of-target only,",
          "so it is the second baseline on IV and OOV rows alike. IV/OOV instrument rows pool the balanced 'bal' + 'bal2' clips; speech rows = 110 balanced",
          "clips per target (+ the paper's 50 piano clips). Paired deltas and win counts on the same cells; the DPS column is paired on its own finished cells.", ""]
    tex = ["% audio IV/OOV/speech, two baselines, mean $\\pm$ std (audio_ivoov_dps_table.py 2026-09-23)", "\\begin{tabular}{" + "l" * len(hdr) + "}", "\\toprule",
           " & ".join(hdr).replace("±", "$\\pm$").replace("Δ", "$\\Delta$").replace("−", "$-$") + "\\\\", "\\midrule"]
    for pool, title in (("IV", "Interferer in the baseline's vocabulary"), ("OOV", "Interferer out of the baseline's vocabulary")):
        md += [f"## {title}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(r) + " |" for r in rows[pool]] + [""]
        tex += [f"\\multicolumn{{{len(hdr)}}}{{l}}{{\\emph{{{title}}}}}\\\\"] + [" & ".join(r).replace("±", "$\\pm$").replace("−", "$-$") + "\\\\" for r in rows[pool]] + ["\\midrule"]
    tex = tex[:-1] + ["\\bottomrule", "\\end{tabular}"]
    (E / "dps_ivoov_20260923/table_ivoov_two_baselines.md").write_text("\n".join(md) + "\n")
    (REPO / "paper/tables/audio_ivoov_two_baselines.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))

if __name__ == "__main__": main()
