# Checkpoints

Every checkpoint SORD loads is listed in [`checkpoint_manifest.json`](checkpoint_manifest.json) with its
relative path under `CHECKPOINT_ROOT` (default `./checkpoints`), size, SHA256, source and license. The
runtime loaders (`sord/paths.py`, `sord/priors.py`) and the scripts below read the same manifest.

```bash
python scripts/download_checkpoints.py --output-dir ./checkpoints                 # official providers
python scripts/download_checkpoints.py --source gdrive --output-dir ./checkpoints  # authors' release package
python scripts/verify_checkpoints.py --checkpoint-dir ./checkpoints               # size + SHA256
```

Add `--include-optional` for the baselines' checkpoints and `--only <id> ...` for specific ones.

**Google Drive release folder:** [https://drive.google.com/drive/folders/1DRm0mEuemf5vBI2e4I5vaxmH6zgdqH-8?usp=sharing](https://drive.google.com/drive/folders/1DRm0mEuemf5vBI2e4I5vaxmH6zgdqH-8?usp=sharing) (set once in `checkpoint_manifest.json` -> `"gdrive"` ->
`"folder_url"`). It holds the authors' weights (`sord_method_edm2_priors.zip`, `sord_baseline_*.zip`) and
copies of the third-party priors (`sord_thirdparty_*.zip`, subject to their providers' terms);
`--source gdrive` downloads only the archive each requested checkpoint needs. The same folder holds the image benchmark
inputs (`sord_benchmarks.zip`), fetched by `python scripts/download_benchmarks.py`.
Total size: 10.3 GB for SORD's benchmarks (required + quick start), 15.7 GB with the optional baseline checkpoints.

## Categories

- **Third-party, official public release** (`celeba_score`, `adm_ffhq256`, `adm_imagenet256`, `msdm_*`, and
  the official baseline checkpoints): downloaded from the original provider. They are not redistributed here;
  their licenses are stated where known.
- **Fine-tuned by the authors** (`edm2_clwd64`, `edm2_scene64`): EDM2-S 64x64 (`edm2-img64-s-1073741-0.075.pkl`,
  NVIDIA) fine-tuned for 512 kimg on clean CLWD training images and on shadow-free scene images. Derivatives of
  CC BY-NC-SA 4.0 weights; distributed in the release package under the same license.
- **Trained by the authors** (`sgm_natural_patch_prior`, `sgm_cifar_patch_prior`, `slbr_*_scratch`, `speech_dm`):
  baseline weights, distributed in the release package.

`required`: **quick-start** = needed for the quick-start example (`configs/sord/celeba_cifar.yaml`);
**benchmark** = needed for that benchmark's config; **optional** = baselines only.

## Inventory

| id | model | architecture | used for | source | file | size | SHA256 (prefix) | license / redistribution | required |
|---|---|---|---|---|---|---|---|---|---|
| `celeba_score` | Joint-diffusion CelebA score model (Stevens et al.) | NCSN++-style score network, TensorFlow, 64x64, VE SDE (sigma=25) | clean-image prior for the CelebA benchmarks (MNIST / natural / CIFAR-10 corruptions) | Hugging Face [`tristan-deep/joint-diffusion-celeba`](https://huggingface.co/tristan-deep/joint-diffusion-celeba) | `joint-diffusion-celeba` | 0.11 GB | `see manifest (di…` | not stated in the downloaded files; unclear: download from Hugging Face, do not repackage | quick-start |
| `edm2_clwd64` | EDM2-S 64x64 fine-tuned on CLWD clean training images | EDM2 (Karras et al. 2024) img64-s, PyTorch pickle | clean-image prior for CLWD watermark removal | authors (release package) | `network-snapshot-0000512-0.100.pkl` | 0.56 GB | `9d1c0df8ca5b5306…` | CC BY-NC-SA 4.0 (derivative of NVIDIA EDM2 weights, third_party/edm2/LICENSE.txt); allowed under CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution) | benchmark |
| `edm2_scene64` | EDM2-S 64x64 fine-tuned on shadow-free scene images | EDM2 img64-s, PyTorch pickle | clean-image prior for ISTD shadow removal | authors (release package) | `network-snapshot-0000512-0.050.pkl` | 0.56 GB | `d42a5c4a4c0927cb…` | CC BY-NC-SA 4.0 (derivative of NVIDIA EDM2 weights); allowed under CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution) | benchmark |
| `adm_ffhq256` | ADM FFHQ 256x256 (ffhq_10m.pt, DPS release) | guided-diffusion UNet, 256x256, linear beta schedule | clean-image prior for FFHQ watermark and FFHQ shadow | [provider's Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) | `ffhq_10m.pt` | 0.37 GB | `81d535743156ec6b…` | not stated in the DPS repository; unclear: download from the DPS link, do not repackage | benchmark |
| `adm_imagenet256` | ADM ImageNet 256x256 unconditional (OpenAI guided-diffusion) | guided-diffusion UNet, 256x256, linear beta schedule | clean-image prior for Imagenette watermark and Imagenette shadow | [official URL](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) | `256x256_diffusion_uncond.pt` | 2.21 GB | `a37c32fffd316cd4…` | MIT (openai/guided-diffusion); permitted by MIT, but download from the official URL | benchmark |
| `msdm_piano` | Weakly-MSDM single-source model, piano (Mariani et al.) | audio-diffusion UNet1d, EDM parameterization, 22.05 kHz mono, PyTorch Lightning | clean-source prior for the audio benchmark with target piano | [provider's Google Drive](https://drive.google.com/file/d/1A33CjKPfmaqsgSXyDqr1Pbd39y_KHgU1/view?usp=share_link), archive `weakly-msdm.tar` | `epoch=236-valid_loss=0.010.ckpt` | 1.62 GB | `65cf4c3ba71361b5…` | not stated in the MSDM repository; unclear: download from the MSDM authors, do not repackage | benchmark |
| `msdm_guitar` | Weakly-MSDM single-source model, guitar (Mariani et al.) | audio-diffusion UNet1d, EDM parameterization, 22.05 kHz mono, PyTorch Lightning | clean-source prior for the audio benchmark with target guitar | [provider's Google Drive](https://drive.google.com/file/d/1A33CjKPfmaqsgSXyDqr1Pbd39y_KHgU1/view?usp=share_link), archive `weakly-msdm.tar` | `epoch=407-valid_loss=0.007.ckpt` | 1.62 GB | `c6131c77fd5bb5b0…` | not stated in the MSDM repository; unclear: download from the MSDM authors, do not repackage | benchmark |
| `msdm_bass` | Weakly-MSDM single-source model, bass (Mariani et al.) | audio-diffusion UNet1d, EDM parameterization, 22.05 kHz mono, PyTorch Lightning | clean-source prior for the audio benchmark with target bass | [provider's Google Drive](https://drive.google.com/file/d/1A33CjKPfmaqsgSXyDqr1Pbd39y_KHgU1/view?usp=share_link), archive `weakly-msdm.tar` | `epoch=443-valid_loss=0.002.ckpt` | 1.62 GB | `997a2dc5968168c1…` | not stated in the MSDM repository; unclear: download from the MSDM authors, do not repackage | benchmark |
| `msdm_drums` | Weakly-MSDM single-source model, drums (Mariani et al.) | audio-diffusion UNet1d, EDM parameterization, 22.05 kHz mono, PyTorch Lightning | clean-source prior for the audio benchmark with target drums | [provider's Google Drive](https://drive.google.com/file/d/1A33CjKPfmaqsgSXyDqr1Pbd39y_KHgU1/view?usp=share_link), archive `weakly-msdm.tar` | `epoch=258-valid_loss=0.019.ckpt` | 1.62 GB | `db9be514f9f72884…` | not stated in the MSDM repository; unclear: download from the MSDM authors, do not repackage | benchmark |
| `sgm_mnist` | Joint-diffusion MNIST-RGB score model (Stevens et al.) |  | SGM two-prior baseline, MNIST benchmark | Hugging Face [`tristan-deep/joint-diffusion-mnist-rgb`](https://huggingface.co/tristan-deep/joint-diffusion-mnist-rgb) | `ckpt-7.data-00000-of-00001` | 0.11 GB | `ad62569d9f9a8e3c…` | not stated; unclear: download from Hugging Face | optional |
| `msdm_4stem` | MSDM 4-stem model (Mariani et al.) |  | MSDM baseline (audio) | [provider's Google Drive](https://drive.google.com/file/d/1mfozibogvNrUaeS283OBz26MWNeZv-OO/view?usp=share_link), archive `msdm.tar` | `epoch=729-valid_loss=0.014.ckpt` | 1.62 GB | `1959f0a92f5dbf7c…` | not stated; unclear: download from the MSDM authors | optional |
| `speech_dm` | Speech diffusion model (UNet1d, MSDM hyperparameters) |  | MSDM + speech prior baseline, speech-target rows | authors (release package) | `best.pt` | 1.62 GB | `f3b024b9416cda03…` | own; allowed (authors' weights) | optional |
| `slbr_official` | SLBR official checkpoint |  | SLBR baseline (official) | [provider's Google Drive](https://drive.google.com/file/d/1uTCzubnWZtu3HIXaK8xsXX-7x302ss13) | `slbr_ckpt.pth.tar` | 0.09 GB | `be5c0a89513803c8…` | not stated; unclear | optional |
| `slbr_clwd_scratch` | SLBR trained from scratch on CLWD |  | SLBR column, CLWD | authors (release package) | `model_best.pth.tar` | 0.27 GB | `5d2af2a640df88ba…` | own; allowed (authors' weights) | optional |
| `slbr_ffhq_scratch` | SLBR trained from scratch on FFHQ watermarks |  | SLBR column, FFHQ | authors (release package) | `model_best.pth.tar` | 0.09 GB | `7fac166039d6171e…` | own; allowed (authors' weights) | optional |
| `slbr_imagenette_scratch` | SLBR trained from scratch on Imagenette watermarks |  | SLBR column, Imagenette | authors (release package) | `model_best.pth.tar` | 0.09 GB | `8c0c5954981d01d1…` | own; allowed (authors' weights) | optional |
| `patchwiper` | PatchWiper (PRWD) restoration network |  | PatchWiper column | [provider's Google Drive](https://drive.google.com/file/d/1fhbSgktTkp7DhfVkZ8yg3sR3AFAs-YiH) | `PatchWiper_PRWD.pth` | 0.13 GB | `a1e8b830104daee4…` | not stated; unclear | optional |
| `patchwiper_wln` | PatchWiper watermark localization network |  | PatchWiper column | [provider's Google Drive](https://drive.google.com/file/d/18S5qVF9G8wBZ_kXiszYFVvReP7knZ-Wi) | `WLN.pth` | 0.07 GB | `a56e28c78c5fe799…` | not stated; unclear | optional |
| `shadowformer_istd` | ShadowFormer ISTD model |  | ShadowFormer column | [provider's Google Drive](https://drive.google.com/file/d/1bHbkHxY5D5905BMw2jzvkzgXsFPKzSq4) | `ISTD_model_latest.pth` | 0.14 GB | `4700ae374b965253…` | MIT; permitted by MIT | optional |
| `shadowdiffusion_gen` | ShadowDiffusion ISTD (istd_gen.pth) |  | ShadowDiffusion column | [provider's Google Drive](https://drive.google.com/file/d/12VV5HzxlIg_kCjMHsw8R9n543wYfGfAB/view?usp=sharing) | `istd_gen.pth` | 0.22 GB | `bca03ee321634786…` | not stated; unclear | optional |
| `shadowdiffusion_opt` | ShadowDiffusion ISTD (istd_opt.pth) |  | ShadowDiffusion column | [provider's Google Drive](https://drive.google.com/file/d/12VV5HzxlIg_kCjMHsw8R9n543wYfGfAB/view?usp=sharing) | `istd_opt.pth` | 0.67 GB | `05e6f145faa01887…` | not stated; unclear | optional |
| `shadowdiffusion_degradation` | ShadowDiffusion ISTD (model_best.pth) |  | ShadowDiffusion column | [provider's Google Drive](https://drive.google.com/file/d/12VV5HzxlIg_kCjMHsw8R9n543wYfGfAB/view?usp=sharing) | `model_best.pth` | 0.23 GB | `612da0043f618b72…` | not stated; unclear | optional |
| `sgm_natural_patch_prior` | SGM noise prior: natural-image patches (26-38 px crops of the benchmark's skimage photos) on a black 64x64 canvas, step 150k | score network with UNetT backbone (time-conditional), TensorFlow, 64x64, VE SDE (sigma=25); needs generators/SGM/SGM.py with UNetT | SGM two-prior baseline on the natural-patch benchmark (masked SGM 19.38 dB, blind 18.92 dB, n=100) | authors (release package) | `natural_patch_prior_150k` | 0.04 GB | `see manifest (di…` | own; allowed (authors' weights) | optional |
| `sgm_cifar_patch_prior` | SGM noise prior: CIFAR-10 training images resized to 16-40 px on a black 64x64 canvas, step 110k | score network with UNetT backbone (time-conditional), TensorFlow, 64x64, VE SDE (sigma=25); needs generators/SGM/SGM.py with UNetT | SGM two-prior baseline on the CIFAR-patch benchmark (masked SGM 15.34 dB, blind 14.36 dB, n=100) | authors (release package) | `cifar_patch_prior_110k` | 0.04 GB | `see manifest (di…` | own; allowed (authors' weights) | optional |

Full SHA256 values (and every file of the directory entries) are in `checkpoint_manifest.json`.

## Notes and unresolved points

- `adm_imagenet256`: the authors' copy (stored under another name) is assumed to be OpenAI's
  `256x256_diffusion_uncond.pt` from its name and size; the SHA256 above is of the authors' copy. If the
  official file's hash differs, `verify_checkpoints.py` will report it.
- `adm_ffhq256` is published only inside the DPS authors' Google Drive folder; download `ffhq_10m.pt` from
  there by hand and place it at `checkpoints/diffusion/guided_diffusion/ffhq_10m.pt`.
- MSDM instrument models come inside the authors' `weakly-msdm.tar`; the download script extracts the four
  files (the member paths are inferred from the extracted layout and checked by SHA256).
- The authors' release package (`edm2_*`, SGM priors, SLBR from scratch, speech model) is one shared Google
  Drive folder whose link is set once in `checkpoint_manifest.json` -> `"gdrive"` -> `"folder_url"`; see
  [CHECKPOINT_UPLOAD_GUIDE.md](CHECKPOINT_UPLOAD_GUIDE.md). Until it is set, `--source gdrive` reports
  these checkpoints as unavailable.
- Implicit downloads: none in `sord/`. The LPIPS metric used by some paper tables (not by `sord/`) downloads
  torchvision's AlexNet weights on first use.
