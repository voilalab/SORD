# MSDM baseline (audio)

The "MSDM" column of the audio results. `run_msdm.py` reads the same SORD audio configs as
`python -m sord.run_audio` (`configs/sord/audio_<target>.yaml`) and scores with the same benchmark code
(window SI-SDR improvement at 22.05 kHz), so its `results.jsonl` is directly comparable with SORD's.

| mode | sources | used for |
|---|---|---|
| default (4-stem MSDM) | the four published weakly-supervised MSDM models (bass, drums, guitar, piano), Dirac likelihood on source 0 (the authors' `eval_weakly_msdm_dirac` configuration) | instrument interferers (pool `bal`) |
| `--speech` (MSDM + speech prior) | [speech, target]: the target's MSDM model plus a speech diffusion model trained by the authors on LibriSpeech in the MSDM framework, Dirac likelihood on speech | speech interferers (pool `speech`) |

Sampler: Karras schedule sigma 1e-4 to 1 (rho 7), 150 steps, 2 resamples, s_churn 40 (the authors' settings).

## Setup

Audio environment, MSDM code and data as for SORD audio (see the main README), plus the speech model:

```bash
conda env create -f environment-audio.yml && conda activate sord-audio
git clone https://github.com/gladia-research-group/multi-source-diffusion-models third_party/msdm
git -C third_party/msdm checkout 76def820666bf2f31af032e8c5a32e4e9de939fc
python scripts/download_benchmarks.py --only audio_piano            # audio benchmark (all four targets)
python scripts/download_checkpoints.py --only msdm_piano msdm_guitar msdm_bass msdm_drums
python scripts/download_checkpoints.py --source gdrive --only speech_dm
```

## Run

```bash
# 4-stem MSDM, instrument interferers (the config's pool, SIR 0 and 5 dB)
python -m baselines.msdm.run_msdm configs/sord/audio_piano.yaml --out runs/msdm_piano
# MSDM + speech prior, speech interferers
python -m baselines.msdm.run_msdm configs/sord/audio_piano.yaml --pool speech --speech --out runs/msdm_speech_piano
# SORD on the same speech cells
python -m sord.run_audio configs/sord/audio_piano.yaml --pool speech --out runs/sord_speech_piano
```

Each cell takes about a minute on one GPU (4-stem: 1200 network calls; with `--speech`: 600).

## Seeds and the paper's runs

Cell k of a run (clips in order, SIR 0 then 5) uses torch seed `--seed + k` (default 12345), as in the paper's
runs, so a cell's result depends on where its run starts. The paper's MSDM numbers come from these runs:

| target | 4-stem MSDM, pool `bal` (`--clips`) | MSDM + speech, pool `speech` (`--clips`) |
|---|---|---|
| piano | `418:528` (one run) | `418:473`, `473:528` |
| guitar | `528:583`, `583:638` | `528:583`, `583:638` |
| bass | `198:253`, `253:308` | `198:253`, `253:308` |
| drums | `308:363`, `363:418` | `308:363`, `363:418` |

The released audio benchmark covers each target's 110 balanced clips with instrument (`bal`) and speech
interferers at SIR 0 and 5 dB. The paper's audio table additionally pools a second balanced instrument set
into its in-vocabulary / out-of-vocabulary rows, uses 50 further piano clips in the piano speech rows, and
has unit-gain (g = 1) rows; those inputs are not part of the release package.
