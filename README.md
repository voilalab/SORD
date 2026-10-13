# SORD: Support-Guided OOD Removal with Diffusion

Official implementation of **Support-Guided OOD Removal with Diffusion (SORD)**. The manuscript is currently under double-blind review; a copy is available at [`reproduction/paper.pdf`](reproduction/paper.pdf).

SORD removes **localized, structured out-of-distribution (OOD) corruptions** using a known corruption-support mask and a pretrained **clean-signal diffusion prior**. It requires neither paired clean/corrupted training data nor a learned model of the corruption distribution. We study image-patch, watermark, and shadow removal, as well as localized audio interference.

## Quick start

The example below runs SORD on the first 25 examples of the **CelebA + CIFAR-10 patch** benchmark. Image experiments require **Linux and an NVIDIA CUDA GPU**.

**Before running:** Each task needs an appropriate pretrained clean-signal prior and benchmark inputs. The [checkpoint and prior guide](CHECKPOINTS.md) documents the required models, download sources, and manual setup. The [checkpoint manifest](checkpoint_manifest.json) contains the machine-readable checkpoint entries, while [`configs/sord/`](configs/sord/) specifies the prior and inputs for each task. You only need to download the checkpoint(s) for the task you run.

```bash
# 1. Install the image environment
conda env create -f environment.yml
conda activate sord
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# 2. Fetch benchmark inputs and the required pretrained prior
python scripts/download_benchmarks.py --only celeba_cifar
python scripts/download_checkpoints.py --only celeba_score --output-dir ./checkpoints
python scripts/verify_checkpoints.py --only celeba_score --checkpoint-dir ./checkpoints

# 3. Run SORD
python -m sord.run configs/sord/celeba_cifar.yaml \
  --images 0:25 --batch 8 --out runs/celeba_cifar

# 4. Summarize the results
python -m sord.summarize runs/celeba_cifar
```

Reconstructions and iteration histories are saved in `runs/celeba_cifar/img_<i>.npz`, with run metadata in `runs/celeba_cifar/results.jsonl`.

> **Data download:** `scripts/download_benchmarks.py` fetches the benchmark inputs of the image tasks (`sord_benchmarks.zip`, ~74 MB, from the [Google Drive release folder](CHECKPOINTS.md)) and verifies them against [`reproduction/benchmarks.json`](reproduction/benchmarks.json); omit `--only` to get all of them. Files are placed at the paths the configs expect, under `DATA_ROOT` (default: the repository root). Audio inputs (~1.3 GB) are fetched separately with `--audio` or `--only audio_<target>` (see [Audio](#audio)).

## Pretrained priors and datasets

SORD uses **task-specific pretrained diffusion models of clean signals**. For complete prior descriptions, checkpoint sources, expected locations, and any manual download steps, start with **[CHECKPOINTS.md](CHECKPOINTS.md)**. For configuring paths and datasets, refer to [`configs/sord/`](configs/sord/) and [`reproduction/`](reproduction/). The implementation of prior loading is in [`sord/priors.py`](sord/priors.py).

| Task configuration (`configs/sord/`) | Required prior checkpoint |
| --- | --- |
| `celeba_mnist`, `celeba_natural`, `celeba_cifar` | `celeba_score` |
| `clwd` | `edm2_clwd64` |
| `istd` | `edm2_scene64` |
| `ffhq_watermark`, `ffhq_shadow` | `adm_ffhq256` (manual setup; see [CHECKPOINTS.md](CHECKPOINTS.md)) |
| `imagenette_watermark`, `imagenette_shadow` | `adm_imagenet256` |
| `audio_piano`, `audio_guitar`, `audio_bass`, `audio_drums` | `msdm_piano`, `msdm_guitar`, `msdm_bass`, `msdm_drums`, respectively |
| `audio_<target>_speech` (same targets, speech interferers) | `msdm_<target>` |

Download and verify only the model(s) you need:

```bash
python scripts/download_checkpoints.py --only celeba_score --output-dir ./checkpoints
python scripts/verify_checkpoints.py --only celeba_score --checkpoint-dir ./checkpoints
```

Some checkpoints require a manual download. The repository also supports the Google Drive source configured under `gdrive.folder_url` in [`checkpoint_manifest.json`](checkpoint_manifest.json). For example:

```bash
python scripts/download_checkpoints.py --source gdrive --only edm2_clwd64 --output-dir ./checkpoints
```

See [CHECKPOINT_UPLOAD_GUIDE.md](CHECKPOINT_UPLOAD_GUIDE.md) for Google Drive and manual-download instructions. By default, checkpoints are stored in `./checkpoints`, data in the repository root, and outputs in `./runs`; override these with `CHECKPOINT_ROOT`, `DATA_ROOT`, and `OUTPUT_ROOT` (see [`sord/paths.py`](sord/paths.py)). Check the licenses of all external datasets and pretrained weights.

## Method overview

Let `y` be an observed signal, `x` its unknown clean counterpart, and `Ω` the known support of a localized corruption `n`. In the additive setting,
$y = x + n$, where ${supp}(n) \subseteq \Omega.$

SORD alternates between estimating the localized corruption and refining clean-signal candidates using a diffusion model trained on clean signals:

1. **Initialize:** Generate candidate clean reconstructions by diffusion inpainting within `Ω`, keeping the observation fixed outside the support.
2. **Estimate the corruption:** Form a masked residual from the current reconstruction estimate and refine it subject to task-specific corruption constraints.
3. **Restore the signal:** Subtract the estimated corruption and refine the clean candidates with the diffusion prior while enforcing consistency outside `Ω`.
4. **Stop adaptively:** Track the variation among reconstructions inside `Ω`; reject an update if it does not sufficiently reduce that variation.

For multiplicative shadows, SORD uses a log-domain formulation. See [`sord/algorithm.py`](sord/algorithm.py) for the main loop, [`sord/steps.py`](sord/steps.py) and [`sord/steps_audio.py`](sord/steps_audio.py) for image and audio updates, and [implementation notes](docs/IMPLEMENTATION_NOTES.md) for implementation details and differences from the submitted manuscript.

## Setup and running other tasks

### Images

The image environment is defined in [`environment.yml`](environment.yml) (Python 3.8, TensorFlow 2.9.3, and PyTorch 1.12.1 with CUDA 11.3). After installation, optionally run the CPU tests:

```bash
python -m unittest discover -s tests -v
```

Once the relevant dataset and prior are available, run a task using its configuration:

```bash
# FFHQ watermark removal
CUDA_VISIBLE_DEVICES=0 python -m sord.run configs/sord/ffhq_watermark.yaml --batch 1 --out runs/ffhq_watermark

# CLWD watermark removal (the dataset support mask is included in sord_benchmarks.zip;
# scripts/build_clwd_dataset_mask.py --clwd-root /path/to/CLWD rebuilds it from CLWD)
CUDA_VISIBLE_DEVICES=0 python -m sord.run configs/sord/clwd.yaml --batch 8 --out runs/clwd

# ISTD shadow removal
CUDA_VISIBLE_DEVICES=0 python -m sord.run configs/sord/istd.yaml --out runs/istd
```

### Audio

Audio uses a separate Conda environment, [`environment-audio.yml`](environment-audio.yml), and MSDM code pinned to a specific revision:

```bash
conda env create -f environment-audio.yml
conda activate sord-audio  # Use the name declared in environment-audio.yml
mkdir -p third_party
git clone https://github.com/gladia-research-group/multi-source-diffusion-models third_party/msdm
git -C third_party/msdm checkout 76def820666bf2f31af032e8c5a32e4e9de939fc

python scripts/download_benchmarks.py --only audio_piano audio_piano_speech   # audio inputs (one ~1.3 GB archive)
python scripts/download_checkpoints.py --only msdm_piano                 # the target's MSDM prior
CUDA_VISIBLE_DEVICES=0 python -m sord.run_audio configs/sord/audio_piano.yaml --out runs/audio_piano
CUDA_VISIBLE_DEVICES=0 python -m sord.run_audio configs/sord/audio_piano_speech.yaml --out runs/audio_piano_speech        # speech interferers
```

The audio benchmark (`sord_audio_benchmarks.zip`, downloaded from the link in `reproduction/benchmarks.json`) holds each target's 110 balanced
10 s clips with instrument (`bal`) and speech (`speech`) interferers at SIR 0 and 5 dB, derived from **Slakh2100** and
**LibriSpeech** (both CC BY 4.0). The MSDM baseline (4-stem MSDM, and MSDM + speech prior for speech interferers) is in
[`baselines/msdm/`](baselines/msdm/README.md).

> **Scope:** These commands run the provided benchmarks. They are not a general-purpose single-image or single-audio-file command-line interface.

## Documentation and reproducibility

Rather than duplicating experiment-specific material here, we maintain the detailed instructions in the following locations:

| Resource | What it covers |
| --- | --- |
| [Pretrained priors and checkpoints](CHECKPOINTS.md) | Model catalog, origins, and downloads |
| [Checkpoint download/setup guide](CHECKPOINT_UPLOAD_GUIDE.md) | Shared Google Drive and manual steps |
| [Task configurations](configs/sord/) | Dataset paths, task settings, and hyperparameters |
| [Reproduction tools](reproduction/) | Benchmark assets, evaluation scripts, and archived replay |
| [Result provenance](reproduction/PROVENANCE.md) | Source and traceability of reported results |
| [Implementation notes](docs/IMPLEMENTATION_NOTES.md) | Implementation details and manuscript differences |
| [SGM baseline](baselines/sgm/README.md) | Baseline-specific setup and usage |
| [MSDM baseline](baselines/msdm/README.md) | Audio baseline: 4-stem MSDM and MSDM + speech prior |

**Repository map:** [`sord/`](sord/) contains the algorithm, model interfaces, and entry points; [`scripts/`](scripts/) contains download and data-preparation utilities; [`tests/`](tests/) contains lightweight tests; and [`experiments/`](experiments/) contains additional experiments and prior adapters. `checkpoints/` and `runs/` are local artifact directories rather than required tracked files.

## Citation and acknowledgments

The manuscript is under review. Please use the following provisional reference until an official citation is available:

```bibtex
@misc{sord2026,
  title = {Support-Guided OOD Removal with Diffusion},
  year = {2026},
  note = {Manuscript under review}
}
```

SORD builds on external score-model and model-definition code, including Stevens et al. (`joint-diffusion`), NVIDIA EDM2, OpenAI guided-diffusion, and MSDM (Mariani et al.). Consult the attribution and license notices in [`generators/`](generators/), [`utils/`](utils/), and [`third_party/`](third_party/) before redistribution of code or weights.
