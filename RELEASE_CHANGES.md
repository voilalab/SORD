# Release preparation changes

This branch starts a fresh history with only the files needed to run SORD, the baselines in `baselines/`,
the checkpoint scripts and the CPU replay (222 files); the full research history is in branch `sord-repro`.
`baselines/sgm/sgm_config.py` holds the SGM configuration helper (verbatim from the authors' comparison script,
with the clean-image prior resolved through the manifest).

Changes made to prepare the repository for public release. The method's update rules, schedules,
hyperparameters and metrics are unchanged; every behavioural check is listed with its result.

## Added

| file | purpose |
|---|---|
| `sord/algorithm.py`, `sord/steps.py`, `sord/steps_audio.py`, `sord/priors.py`, `sord/run.py`, `sord/run_audio.py`, `sord/summarize.py`, `sord/paths.py` | readable implementation of the method (images and audio), path resolution, runners, summary |
| `configs/sord/*.yaml` (13) | one config per benchmark, checkpoints by manifest id, benchmark paths relative to `DATA_ROOT` |
| `checkpoint_manifest.json`, `CHECKPOINTS.md`, `CHECKPOINT_UPLOAD_GUIDE.md` | checkpoint inventory (sizes, SHA256, sources, licenses) and release/upload procedure |
| `scripts/download_checkpoints.py`, `scripts/verify_checkpoints.py`, `scripts/package_checkpoints.py`, `scripts/checkpoint_common.py` | checkpoint download (official / Google Drive, resumable, SHA256-verified), verification, packaging |
| `scripts/build_clwd_dataset_mask.py` | rebuilds the CLWD support mask from the dataset's `Mask.png` files |
| `baselines/sgm/` | SGM two-prior baseline: blind and masked scripts, scorer, README |
| `environment.yml` (rewritten), `environment-audio.yml` | pinned environments of the runs |
| `tests/test_reference.py` (rewritten) | 9 CPU tests with toy priors |

## Removed

| file | reason |
|---|---|
| `sord/core.py` | older parallel implementation, superseded by `sord/algorithm.py` + `sord/steps.py` |

## Modified (behaviour preserved)

| file | original | change | why behaviour is preserved | check |
|---|---|---|---|---|
| `sord/priors.py` (CelebA prior) | wrapped `claude_experiments/copied_base_code/run_tango.CelebAAdapter` | same Tweedie computation in `CelebAScorePrior` | identical operations on the same model | outputs bit-identical at 4 ladder levels (max diff 0.0) |
| `experiments/extbench/edm2_adapter.py` | `EDM2_SRC` = absolute path outside the repository | `third_party/edm2` (env `SORD_EDM2_SRC`) | `dnnlib`/`torch_utils` files identical (`diff -r`) | `verify_checkpoints`, verification runs |
| `experiments/extbench/gd256_adapter.py` | `DPS_REPO` = absolute path | `third_party` (env `SORD_GUIDED_DIFFUSION_SRC`) | `unet.py`, `nn.py`, `fp16_util.py` byte-identical | same |
| `experiments/tango_msdm_2026_08_04/src/msdm_prior.py`, `e2_smoke.py` | absolute default checkpoint / benchmark paths | `CHECKPOINT_ROOT` / `DATA_ROOT` defaults (env overrides) | defaults only | audio check: 16 cells, 1.88 vs 1.93 SI-SDRi |
| `generators/SGM/SGM.py` | no time-conditional backbone | adds `UNetT` / `NCSNv2T` (opt-in via config) | existing configs take the old code path | SGM masked chunk 0 reproduces the original run (19.35 = 19.35, max pixel diff 1e-3) |
| `sord/stop_sweep.py` | wrote to `SORD/reproduction` relative to the parent folder; CIFAR glob also matched two adaptive-step runs | writes to `reproduction/`; glob fixed; stopping test at the end of the iteration | replays the saved iterates | tables regenerated (CIFAR now n=100, 22.05) |
| `reproduction/fetch_assets.py`, `publish_release.py`, `assets.json` | release repository hard-coded | `--repository` / `SORD_RELEASE_REPOSITORY` (default KheirAli/SORD) | archive names and SHA256 unchanged | syntax check |
| `README.md`, `.gitignore`, `reproduction/alg1_results.md`, `reproduction/PROVENANCE.md`, `configs/sord/tau_stop.yaml` | | rewritten / updated | documentation | |

## Algorithm-level decisions (documented, user-approved)

- Stopping test at the end of iteration m (inpainting x_0 admissible); one tau_stop per benchmark:
  0.01 iterate-class, 0.2 stop-class images, 0.8 audio.
- sigma* computed per image (the historical runners pooled it over 25-image chunks).
- CLWD: SORD receives the dataset's own watermark mask; scored on the benchmark region.
