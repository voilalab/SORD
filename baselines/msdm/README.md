# MSDM baseline (audio)

`run_msdm.py` runs MSDM on the SORD audio configs (`configs/sord/audio_*.yaml`) and scores it with the same
benchmark code as `python -m sord.run_audio` (window SI-SDR improvement at 22.05 kHz).

| mode | sources |
|---|---|
| 4-stem MSDM (default for instrument interferers, `--no-speech`) | the four weakly-supervised MSDM models (bass, drums, guitar, piano), Dirac likelihood on source 0 |
| MSDM + speech prior (default for speech interferers, `--speech`) | [speech, target]: a LibriSpeech diffusion model in the MSDM framework and the target's MSDM model, Dirac likelihood on speech |

Sampler: Karras schedule sigma 1e-4 to 1 (rho 7), 150 steps, 2 resamples, s_churn 40.

## Setup

```bash
conda env create -f environment-audio.yml && conda activate sord-audio
git clone https://github.com/gladia-research-group/multi-source-diffusion-models third_party/msdm
git -C third_party/msdm checkout 76def820666bf2f31af032e8c5a32e4e9de939fc
python scripts/download_benchmarks.py --audio                # add --audio-extra for the extra configs
python scripts/download_checkpoints.py --only msdm_piano msdm_guitar msdm_bass msdm_drums
python scripts/download_checkpoints.py --source gdrive --only speech_dm
```

## Run

```bash
python -m baselines.msdm.run_msdm configs/sord/audio_piano.yaml --out runs/msdm_piano                       # 4-stem MSDM
python -m baselines.msdm.run_msdm configs/sord/audio_piano_speech.yaml --out runs/msdm_piano_speech         # MSDM + speech prior
python -m baselines.msdm.run_msdm configs/sord/audio_piano_speech.yaml --no-speech --out runs/msdm4_piano_speech   # 4-stem MSDM, speech interferers
```

## Seeds and clip ranges

Cell k of a run (clips in order, then SIRs) uses torch seed `--seed + k` (default 12345), so a cell's result depends
on where its run starts. `scripts/run_audio_all.py` uses these `--clips` ranges ("halves": the config's range split
into two equal runs):

| config | 4-stem MSDM | MSDM + speech prior |
|---|---|---|
| `audio_piano` | one run | |
| `audio_{guitar,bass,drums}` | halves | |
| `audio_<target>_speech` | halves | halves |
| `audio_<target>_bal2` | halves | |
| `audio_<target>_g1`, `audio_<target>_bal2_g1` | one run | |
| `audio_{piano,guitar}_speech_g1` | halves | halves |
| `audio_piano_speech50`, `audio_piano_speech50_g1` | one run | one run |
