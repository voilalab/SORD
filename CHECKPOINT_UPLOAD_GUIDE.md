# Checkpoint upload guide (Google Drive)

## 1. What to upload

Only the authors' own weights, built with

```bash
python scripts/package_checkpoints.py --checkpoint-dir ./checkpoints --output-dir ./release_package
```

| archive | checkpoints | license |
|---|---|---|
| `sord_method_edm2_priors.zip` | `edm2_clwd64`, `edm2_scene64` (EDM2-S 64x64 fine-tuned priors for CLWD and ISTD) | CC BY-NC-SA 4.0 (derivative of NVIDIA EDM2) |
| `sord_baseline_sgm_noise_priors.zip` | `sgm_natural_patch_prior`, `sgm_cifar_patch_prior` | authors' |
| `sord_baseline_slbr_scratch.zip` | `slbr_clwd_scratch`, `slbr_ffhq_scratch`, `slbr_imagenette_scratch` | authors' |
| `sord_baseline_msdm_speech_dm.zip` | `speech_dm` | authors' |
| `sord_benchmarks.zip` | benchmark inputs of the image tasks (not checkpoints; file list and SHA256 in `reproduction/benchmarks.json`, read by `scripts/download_benchmarks.py`) | per source dataset |
| `sord_audio_benchmarks.zip` | benchmark inputs of the audio tasks (clips derived from Slakh2100 and LibriSpeech; listed in `reproduction/benchmarks.json`; hosted outside the Drive folder, see section 4) | CC BY 4.0 (Slakh2100, LibriSpeech) |

The script verifies every file's SHA256 before packaging and writes `SHA256SUMS.txt` (every archive
and every file inside) and `package_manifest.json`. Archives store files under `checkpoints/<path>`
uncompressed, so unzipping at the repository root reproduces the expected layout.

## 2. What must come from the original providers

`celeba_score` (Hugging Face), `adm_ffhq256` (DPS Google Drive folder), `adm_imagenet256` (OpenAI URL),
`msdm_piano|guitar|bass|drums` (MSDM authors' Google Drive tar), and the official baseline checkpoints
(SLBR, PatchWiper, ShadowFormer, ShadowDiffusion, MSDM 4-stem, Stevens et al. MNIST). Their licenses do
not clearly permit redistribution (ImageNet ADM and ShadowFormer are MIT), so the public instructions point
to the providers; `scripts/download_checkpoints.py` fetches them automatically where the provider allows
(Hugging Face, HTTP, Google Drive file), except `ffhq_10m.pt`, which is only in a Drive folder.
`package_checkpoints.py --third-party` builds `sord_thirdparty_*.zip` for a private backup; do not post
those publicly.

## 3. Google Drive layout

```
SORD-checkpoints/
  README.txt
  SHA256SUMS.txt
  sord_method_edm2_priors.zip
  sord_baseline_sgm_noise_priors.zip
  sord_baseline_slbr_scratch.zip
  sord_baseline_msdm_speech_dm.zip
  sord_benchmarks.zip
```

## 4. Shareable links

1. Upload the five zips (four weight archives + `sord_benchmarks.zip`) into one folder, then share the folder: Share -> General access ->
   "Anyone with the link" (Viewer). Nothing in this guide changes sharing permissions automatically.
2. Copy the folder's share link (`https://drive.google.com/drive/folders/<id>?usp=sharing`).
3. Paste it in ONE place: `checkpoint_manifest.json` -> `"gdrive"` -> `"folder_url"`. Every checkpoint of
   the package already names its archive (`"gdrive_archive"`), so nothing else changes. Without editing
   the manifest, the same link can be passed as `--gdrive-folder <link>`.
4. `sord_audio_benchmarks.zip` (1.3 GB) is hosted separately. Paste its download link (a Google Drive file link
   or any direct HTTP(S) link) in `reproduction/benchmarks.json` -> that archive's `"url"`; until then it can be
   passed as `--archive-url sord_audio_benchmarks.zip=<link>`. The script checks the archive's SHA256.

## 5. Verify a download

```bash
python scripts/download_checkpoints.py --source gdrive --output-dir ./checkpoints
python scripts/verify_checkpoints.py --checkpoint-dir ./checkpoints --include-optional
```

The download script skips files that already verify, resumes interrupted HTTP downloads, and renames a
file whose SHA256 does not match to `*.bad`.

## 6. Reproduce the example after download

```bash
python scripts/download_benchmarks.py --only celeba_cifar        # benchmark inputs (sord_benchmarks.zip)
python scripts/download_checkpoints.py --only celeba_score
python -m sord.run configs/sord/celeba_cifar.yaml --images 0:25 --batch 8 --out runs/celeba_cifar
python -m sord.summarize runs/celeba_cifar
```
