# PDF, algorithm and experiment provenance

The requested PDF is the 19-page ICLR 2027 draft. The older LaTeX manuscript
in `paper/` is retained as historical material and is not necessarily the
source of this PDF. The following paths locate the data and code used for
its examples and tables.

| PDF result | Source and replay |
|---|---|
| Table 1: CelebA + MNIST/natural/CIFAR | `paper/figs/qual/make_celeba_dps_fig.py`, `experiments/ros/results/r5/audit/base`, `r5/cifar/min_exact`; `replay.py celeba` |
| Table 2: watermark/shadow | `paper/tables/watermark_shadow_results_20260924.tex` has per-cell provenance; loaders in `paper/metrics/arms_wm.py` and `arms_shadow.py`; `replay.py images` |
| Table 3: K ablation | `experiments/reviewer_audit/analyze_ablation2.py`, `audit/minK*` and `audit/base`; `replay.py samples` |
| Table 4: DDIM levels | `experiments/extbench/build_xs_table_20260925.py`; `replay.py steps` |
| Table 5: mask errors | `experiments/reviewer_audit/mask100/score.py`; true-support scoring, including CLWD's empty-mask exclusion; `replay.py mask` |
| Table 6: residual diffusion ablation | `experiments/extbench/build_nstep_ablation_table_20260917.py`; `replay.py nstep` |
| Table 7: schedules | `experiments/sigma9631_20260921/score.py`, `experiments/sd8421_20260922/score.py`, `experiments/reviewer_audit/analyze_ablation2.py`; historical results retained alongside launch queues |
| Table 8: mask and contaminant prior | `experiments/reviewer_audit/make_mask_noise_ablation_table.py`; `replay.py mask-prior` |
| Table 9: four audio targets, IV/OOV instruments and speech | `experiments/tango_msdm_2026_08_04/src/audio_ivoov_dps_table.py`; per-example summaries under `results/arms` and `external_sources`; `replay.py audio` |
| Figure 2: image comparisons | generators in `paper/figs/qual/`, including `make_celeba_dps_fig.py`; original sample indices remain in the scripts |
| Figure 3: audio examples | `paper/figs/qual/make_audio_tf_ivoov.py`; `replay.py audio-figures` recomputes waveform SI-SDR and checks it against saved summaries |
| Figure 4: mask errors | `experiments/reviewer_audit/make_mask_fig_erode_shift.py`; `replay.py mask-figures` |

## Differences that affect reproducibility

1. Algorithm 1 uses a **relative decrease** of ensemble uncertainty to stop.
   The historical reported image runs often use an **absolute level** gate
   of `0.04 * (hi-lo)` after the first iteration. The saved
   `experiments/stop_sigma/rel_gate_shadow_fullloop_20260924.md` quantifies
   the differences. `sord/algorithm.py` implements the relative rule (line 6 as revised 2026-10-11);
   it is not claimed to reproduce tables produced with the older rule.
2. The CLWD value 28.06 uses a **declared-alpha, K=1** loop and per-pixel
   color readout. This is not the same as the generic K=32 diffusion
   residual step. The source scripts explicitly state that declared-alpha
   arms do not contain a diffusion n-step. Watermark ablation rows use a
   different, blind-residual variant. Reproducing one should not silently
   substitute the other.
3. ISTD and transferred shadows use multiplicative variants in several
   reported rows. Their composed inputs match clean targets outside the
   shadow support. Supervised shadow models are scored on their original
   inputs. Preserve this protocol when comparing baselines.
4. Image statistics use per-image regional PSNR and sample standard
   deviation (`ddof=1`), with clipping and image pairing conventions in the
   original loaders. Audio tables group per-example window SI-SDR
   improvements; the `floor_win` readout appears in several historical runs.


Replaying saved reconstructions validates metric calculations. It does not
establish that every table follows the PDF's generic pseudocode, nor does
it establish successful retraining or fresh stochastic reproduction.

## Published replay data

The private GitHub release `reproduction-v1` contains eight checksum-verified
tar archives (3,274,352,640 bytes). `assets.json` lists all 1,220 relative
paths, their SHA-256 checksums, the NPZ arrays retained, and the corresponding
release asset IDs. It includes processed observations, clean targets,
supports, saved reconstructions and the audio waveforms used by the figure
generator. It does not redistribute the full raw training corpora or model
weights. Training/inference requires those additional assets.

## CLWD cell (added 2026-10-07)

`clwd_diagnosis.md` shows that blind Algorithm 1 on the CLWD support scores 22.56 dB, equal to the input floor (22.55), and that no
lever consistent with the paper (zero tether, clean prior only, α not given) reaches 28.06; the best, with the undilated dataset mask as
the support, is 25.09. The 28.06 in the PDF's Table 2 is the declared-α K=1 extension. The text should either report 22.56 for SORD
(and name the violated assumptions) or state the undilated-mask protocol, and describe 28.06 as a separate extension that uses α.
