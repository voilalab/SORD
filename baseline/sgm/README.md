# SGM baseline (two diffusion priors, Stevens et al.)

The "SGM 2 priors" and "SGM 2 priors + mask" columns for the natural-patch and CIFAR-patch
benchmarks. SGM samples the clean image and the corruption jointly, each with its own diffusion
prior: the CelebA score model for the image and a corruption prior trained on the benchmark's
corruption class (patches on a black 64x64 canvas). The sampler is the authors' PiGDM-guided joint
sampler (`generators/SGM/`, `utils/inverse.py`), 600 noise scales, with the observation written as
the authors' mixture model y/2 = 0.5 x + 0.5 n (an exact reparameterization of y = x + n).

| file | purpose |
|---|---|
| `sgm_blind.py` | joint sampling without the support (paper column "SGM 2 priors") |
| `sgm_masked.py` | the same sampler with the known support enforced at every step ("SGM 2 priors + mask"); `--fix-init` starts both chains at the SDE's terminal std (9.845) instead of N(0, 1), `--final-composite` keeps y exactly outside the support |
| `score.py` | on-support PSNR, mean +- std (ddof = 1) over the 100 images |

## Checkpoints (checkpoint_manifest.json)

| identifier | what | notes |
|---|---|---|
| `celeba_score` | CelebA score model (clean-image prior) | Hugging Face `tristan-deep/joint-diffusion-celeba` |
| `sgm_natural_patch_prior` | corruption prior, natural-image patches, step 150k | authors' weights, release package `sord_baseline_sgm_noise_priors.zip` |
| `sgm_cifar_patch_prior` | corruption prior, CIFAR-10 patches, step 110k | same package |

Both corruption priors use the time-conditional `UNetT` backbone in `generators/SGM/SGM.py`.

## Reproduce (100 images per benchmark, four chunks of 25)

```bash
for i in 0 25 50 75; do
  python baselines/sgm/sgm_masked.py --bench natural --i0 $i --i1 $((i+25)) --arm sgmM --fix-init --final-composite
  python baselines/sgm/sgm_blind.py  --bench natural --i0 $i --i1 $((i+25)) --arm sgm
done
python baselines/sgm/score.py natural sgmM sgm
```

Replace `natural` by `cifar` for the CIFAR-patch benchmark. Results (on-support PSNR, dB, n = 100):

| benchmark | SGM 2 priors | SGM 2 priors + mask |
|---|---|---|
| natural patch | 18.92 +- 7.15 | 19.38 +- 6.38 |
| CIFAR-10 patch | 14.36 +- 3.93 | 15.34 +- 4.16 |

These are the authors' runs with exactly these checkpoints and flags (run with the original
scripts of this folder before they were made path-portable). The SGM rows of the MNIST benchmark
use the authors' own MNIST prior (Hugging Face `tristan-deep/joint-diffusion-mnist-rgb`).
