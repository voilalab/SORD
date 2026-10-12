"""CPU regeneration of stored experiment tables; never starts training."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "images": ["paper/metrics/run_mean_std.py", "--redo", "--no-ssim"],
    "audio": ["experiments/tango_msdm_2026_08_04/src/audio_ivoov_dps_table.py"],
    "celeba": ["paper/figs/qual/make_celeba_dps_fig.py", "--extended"],
    "mask": ["experiments/reviewer_audit/mask100/score.py"],
    "mask-figures": ["experiments/reviewer_audit/make_mask_fig_erode_shift.py"],
    "samples": ["experiments/reviewer_audit/analyze_ablation2.py"],
    "nstep": ["experiments/extbench/build_nstep_ablation_table_20260917.py"],
    "steps": ["experiments/extbench/build_xs_table_20260925.py"],
    "mask-prior": ["experiments/reviewer_audit/make_mask_noise_ablation_table.py"],
    "audio-figures": ["paper/figs/qual/make_audio_tf_ivoov.py", "--bare", "--collabels"],
    "image-figures": ["paper/figs/qual/make_all_image_fig.py"],
    "teaser": ["paper/figs/qual/make_teaser.py", "--notitles", "--per", "3"],
    "schedules": ["experiments/sigma9631_20260921/score.py"],
    "schedule-stop": ["experiments/sd8421_20260922/score.py"],
}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="*", choices=list(COMMANDS))
    ap.add_argument("--trace", action="store_true", help="record binary assets used")
    args = ap.parse_args()
    logs = ROOT / "reproduction/logs"; logs.mkdir(parents=True, exist_ok=True)
    failures = []
    for name in args.targets or list(COMMANDS):
        import os
        env = os.environ.copy(); env["PYTHONPATH"] = str(ROOT)
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("MPLCONFIGDIR", "/tmp/sord-matplotlib")
        env["SORD_TRACE"] = str(logs / f"{name}-inputs.json")
        command = [sys.executable] + (["reproduction/trace_replay.py"] if args.trace else []) + COMMANDS[name]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        output = result.stdout + result.stderr
        (logs / f"{name}.log").write_text(output)
        failed = result.returncode != 0 or any(s in output for s in ("[ERR]", "Traceback (most recent call last)", "FileNotFoundError"))
        print(f"{name}: {'FAIL' if failed else 'PASS'} ({logs / (name + '.log')})", flush=True)
        if failed: failures.append(name)
    return bool(failures)

if __name__ == "__main__":
    sys.exit(main())
