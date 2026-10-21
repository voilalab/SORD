"""Run the whole audio benchmark, SORD and the MSDM baseline, and write one results table.

    python scripts/run_audio_all.py --out runs/audio_all --gpus 0 1          # base benchmark: 8 configs
    python scripts/run_audio_all.py --out runs/audio_all --gpus 0 1 --extra  # also the extra configs
    python scripts/run_audio_all.py --out runs/audio_smoke --limit 2         # smoke test: 2 clips per run

Runs (configs/sord/), for targets piano, guitar, bass, drums:
  base     audio_<target>.yaml (instrument interferers, pool bal, SIR 0/5) and audio_<target>_speech.yaml (speech, SIR 0/5);
           110 clips each.  Inputs: scripts/download_benchmarks.py --audio
  --extra  audio_<target>_bal2.yaml (second balanced set, SIR 0/5), audio_<target>_g1.yaml and audio_<target>_bal2_g1.yaml
           (instrument interferers at unit gain), audio_{piano,guitar}_speech_g1.yaml (speech at unit gain),
           audio_piano_speech50.yaml and audio_piano_speech50_g1.yaml (50 further piano speech clips).
           Inputs: scripts/download_benchmarks.py --audio --audio-extra
Every config is run with SORD (python -m sord.run_audio) and the MSDM baseline (python -m baselines.msdm.run_msdm):
4-stem MSDM on every config, and MSDM + speech prior also on the speech configs.  MSDM runs use the clip ranges of
baselines/msdm/README.md, because an MSDM cell's seed depends on where its run starts.

Jobs run one per GPU (--gpus), each in <out>/<method>/<run>/ with its own log.txt. A finished run (all rows in
results.jsonl) is skipped, so the command can be repeated after an interruption; an unfinished one is restarted
from its first clip (every cell is seeded, so the results are the same). At the end, <out>/summary.md holds
mean +- std (sample std) of the window SI-SDR improvement (dB), also printed: instrument interferers split into
in-vocabulary (bass, drums, guitar, piano: MSDM has a model for them) and out-of-vocabulary, pooled over the bal and
bal2 sets; speech interferers compared with MSDM + speech prior (IV) and with 4-stem MSDM (OOV).
Needs the audio environment, the MSDM code and the checkpoints msdm_piano/guitar/bass/drums and speech_dm.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("piano", "guitar", "bass", "drums")
IV = {"Bass", "Drums", "Guitar", "Piano"}            # instrument classes with an MSDM model (interferer in vocabulary)
SPEECH_ROWS = ["speech (IV: MSDM + speech prior)", "speech (OOV: 4-stem MSDM)"]
BAL = {"bass": (198, 308), "drums": (308, 418), "piano": (418, 528), "guitar": (528, 638)}
BAL2 = {"piano": (858, 968), "guitar": (968, 1078), "bass": (1078, 1188), "drums": (1188, 1298)}


def halves(r):
    a, b = r
    return [(a, (a + b) // 2), ((a + b) // 2, b)]


def runs(extra):
    """(target, config, kind, sirs, clips, {baseline: MSDM clip ranges}); kind = instrument | speech."""
    out = []
    for t in TARGETS:
        out += [(t, f"audio_{t}", "instrument", (0, 5), BAL[t], {"msdm4": [BAL[t]] if t == "piano" else halves(BAL[t])}),
                (t, f"audio_{t}_speech", "speech", (0, 5), BAL[t], {"msdm_speech": halves(BAL[t]), "msdm4": halves(BAL[t])})]
        if extra:
            out += [(t, f"audio_{t}_bal2", "instrument", (0, 5), BAL2[t], {"msdm4": halves(BAL2[t])}),
                    (t, f"audio_{t}_g1", "instrument", (1,), BAL[t], {"msdm4": [BAL[t]]}),
                    (t, f"audio_{t}_bal2_g1", "instrument", (1,), BAL2[t], {"msdm4": [BAL2[t]]})]
            if t in ("piano", "guitar"):
                out.append((t, f"audio_{t}_speech_g1", "speech", (1,), BAL[t], {"msdm_speech": halves(BAL[t]), "msdm4": halves(BAL[t])}))
    if extra:
        for cfg, sirs in (("audio_piano_speech50", (0, 5)), ("audio_piano_speech50_g1", (1,))):
            out.append(("piano", cfg, "speech", sirs, (0, 50), {"msdm_speech": [(0, 50)], "msdm4": [(0, 50)]}))
    return out


def jobs(extra, limit, batch):
    out = []
    for t, cfg, kind, sirs, clips, msdm in runs(extra):
        cut = lambda a, b: (a, min(b, a + limit) if limit else b)
        common = dict(target=t, kind=kind, sirs=sirs)
        out.append(dict(common, method="sord", run=cfg, clips=cut(*clips),
                        cmd=["-m", "sord.run_audio", f"configs/sord/{cfg}.yaml", "--batch", str(batch)]))
        for base, ranges in msdm.items():
            flag = ["--speech"] if base == "msdm_speech" else ["--no-speech"]
            for a, b in ranges:
                suffix = "_4stem" if kind == "speech" and base == "msdm4" else ""
                out.append(dict(common, method=base, run=f"{cfg}{suffix}_{a}_{b}", clips=cut(a, b),
                                cmd=["-m", "baselines.msdm.run_msdm", f"configs/sord/{cfg}.yaml", *flag]))
    return out


def run_dir(out: Path, j):
    return out / ("sord" if j["method"] == "sord" else "msdm") / j["run"]


def rows_of(d: Path):
    f = d / "results.jsonl"
    return [json.loads(line) for line in open(f)] if f.exists() else []


def summarize(out: Path, js):
    acc = defaultdict(dict)                 # (target, cond, row) -> method -> {(clip, sir, pool): sisdri}
    for j in js:
        for r in rows_of(run_dir(out, j)):
            cond = "g = 1" if j["sirs"] == (1,) else f"SIR {r['sir']:g}"
            if j["kind"] == "instrument":
                rows = ["IV instrument" if r["instrument"].split(" ")[0] in IV else "OOV instrument"]
            else:                           # SORD on both speech rows; each MSDM variant on its own row
                rows = {"sord": SPEECH_ROWS, "msdm_speech": SPEECH_ROWS[:1], "msdm4": SPEECH_ROWS[1:]}[j["method"]]
            for row in rows:
                acc[(j["target"], cond, row)].setdefault("sord" if j["method"] == "sord" else "msdm", {})[
                    (r["clip"], r["sir"], r["pool"])] = r["sisdri"]
    fmt = lambda v: f"{np.mean(v):+6.2f} +- {np.std(v, ddof=1) if len(v) > 1 else 0:5.2f}" if len(v) else "-"
    lines = ["# Audio benchmark: window SI-SDR improvement (dB), mean +- std over cells", "",
             "Instrument rows pool every instrument config run (bal, and bal2 with --extra);",
             "MSDM = 4-stem MSDM, except speech IV rows: MSDM + speech prior. n = cells finished by both methods.", "",
             "| target | condition | interferer | n | SORD | MSDM | SORD - MSDM |", "|---|---|---|---|---|---|---|"]
    order = ["IV instrument", "OOV instrument", *SPEECH_ROWS]
    for key in sorted(acc, key=lambda k: (TARGETS.index(k[0]), order.index(k[2]), k[1].startswith("g"), k[1])):
        s, m = acc[key].get("sord", {}), acc[key].get("msdm", {})
        both = sorted(set(s) & set(m))
        a, b = np.array([s[k] for k in both]), np.array([m[k] for k in both])
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(both)} | {fmt(a)} | {fmt(b)} | {fmt(a - b)} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("runs/audio_all"))
    ap.add_argument("--gpus", nargs="+", default=["0"], help="GPU ids; one job per GPU at a time")
    ap.add_argument("--methods", nargs="+", default=["sord", "msdm4", "msdm_speech"], choices=["sord", "msdm4", "msdm_speech"])
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=TARGETS)
    ap.add_argument("--extra", action="store_true", help="also the extra configs (bal2, g = 1, 50 piano speech clips)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips of every run (smoke test)")
    ap.add_argument("--batch", type=int, default=1, help="SORD cells per network call (GPU memory permitting)")
    ap.add_argument("--summary-only", action="store_true", help="only rebuild summary.md from finished rows")
    a = ap.parse_args()

    out = a.out if a.out.is_absolute() else Path.cwd() / a.out
    js = [j for j in jobs(a.extra, a.limit, a.batch) if j["method"] in a.methods and j["target"] in a.targets]
    if not a.summary_only:
        todo = []
        for j in js:
            d = run_dir(out, j)
            need = (j["clips"][1] - j["clips"][0]) * len(j["sirs"])
            if len(rows_of(d)) >= need:
                continue
            (d / "results.jsonl").unlink(missing_ok=True)      # unfinished: restart (cells are seeded)
            todo.append(j)
        print(f"{len(js) - len(todo)} of {len(js)} runs already finished; running {len(todo)} on GPU(s) {a.gpus}")
        free, running = list(a.gpus), []
        while todo or running:
            while todo and free:
                j, g = todo.pop(0), free.pop(0)
                d = run_dir(out, j)
                d.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, *j["cmd"], "--clips", f"{j['clips'][0]}:{j['clips'][1]}", "--out", str(d)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONPATH=str(ROOT))
                log = open(d / "log.txt", "w")
                print(f"[GPU {g}] {j['method']} {j['run']}", flush=True)
                running.append((subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), g, j, log))
            time.sleep(5)
            for p in [r for r in running if r[0].poll() is not None]:
                running.remove(p)
                free.append(p[1])
                p[3].close()
                status = "done" if p[0].returncode == 0 else f"FAILED (exit {p[0].returncode}), see log.txt"
                print(f"[GPU {p[1]}] {p[2]['method']} {p[2]['run']}: {status}", flush=True)
    summarize(out, js)


if __name__ == "__main__":
    main()
