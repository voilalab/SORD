# Results under Algorithm 1's stopping rule (tether 0, saved iterates, no rerun)

**Stopping test (2026-10-11 form).** At the end of iteration m, after σ*_{m+1} (the spread of the new
estimate x_m) is computed: if σ*_m − σ*_{m+1} ≤ τ_stop σ*_m, x_m is discarded and x_{m-1} is returned. The
inpainting initialization x_0 is an admissible output (τ_stop ≥ 1 always returns it). The PDF's form tested
at the start of iteration m and returned x_{m-1}, so it could never return x_0, which made the audio rows (all
x_0) unreachable. Implemented in `sord/algorithm.py`; replayed by `sord/stop_sweep.py` (full sweep in
`stop_sweep.md`; the saved runs do not log the spread of the final x_M, so the replay cannot apply the test
after the last iteration).

**τ_stop (three values, `configs/sord/tau_stop.yaml`):** 0.01 for the iterate-class cells (CelebA benchmarks and
CLWD), 0.2 for the stop-class cells (256² watermarks, all shadows), 0.8 for audio.

| cell | n | τ_stop | Algorithm 1 | reported | Δ | τ range that reproduces |
|---|---|---|---|---|---|---|
| CelebA+MNIST (T1) | 100 | 0.01 | 27.10 | 27.16 | −0.06 | 0–0.04 |
| CelebA+natural (T1) | 100 | 0.01 | 21.96 | 21.96 | +0.00 | 0–0.01 |
| CelebA+CIFAR (T1) | 100 | 0.01 | 22.05 | 22.05 | +0.00 | 0–0.05 |
| FFHQ wm (T2) | 25 | 0.2 | 27.80 | 27.80 | +0.00 | 0.07–0.25 |
| Imagenette wm (T2) | 25 | 0.2 | 26.38 | 26.43 | −0.05 | 0.2–0.3 |
| Imagenette shadow (T2) | 25 | 0.2 | 19.05 | 19.05 | +0.00 | 0–0.2 |
| ISTD shadow (T2) | 100 | 0.2 | 29.10 | 29.10 | +0.00 | 0.1–0.4 |
| FFHQ shadow (T2) | 25 | 0.2 | 20.34 | 20.34 | +0.00 | 0.05–0.5 |
| CLWD (T2), support = CLWD's own mask | 99 | 0.01 | ≈25.1 (25.05 best iterate, 25.09 at τ=0.03; per-image run pending) | 28.06 | — | reported row (user decision 2026-10-11); scored on M_true; support from `scripts/build_clwd_dataset_mask.py` |
| CLWD (T2), support = dilated M_true | 99 | 0.01 | 22.56 | 28.06 | — | 0–0.07 → 22.56; the earlier 28.06 is the alpha-blend arm (coverage from CLWD's 256² mask, opacity and colour fitted), not Algorithm 1 |
| Audio, 8 iterated rows (T1) | 110 each | 0.8 | = x_0 on all 8 | x_0 | 0.00 | ≥ 0.8 (0.5 differs on drums SIR0: 11.53 vs 11.24) |

No single τ fits all cells: the CelebA cells need τ ≤ 0.04 (larger τ stops before the iterations that carry their gain;
MNIST 24.95 at 0.1) and the watermark/shadow cells need τ ≥ 0.07–0.2 (smaller τ iterates past their reported iterate,
e.g. Imagenette wm 24.55 at 0.01). On the two 256² watermark cells the inpainting x_0 scores above the reported number
(FFHQ 28.52, Imagenette 26.58; reached at τ ≥ 0.6); 0.2 reproduces the reported numbers.

Sources: ISTD and FFHQ shadow were gated at m1 in the reported runs (only x_m1 saved); they are replayed on their
same-seed full-loop reruns (`b5s/multw0minK32_full`, M8; `ffhqsh0_native/mult256gF8`, M8 — the reported FFHQ-shadow run
was M3, so iterates past m1 are not that run's trajectory). The CIFAR row previously read 21.68 (n=150) because the
sweep's glob also picked up two adaptive-step runs in chunk c0 (`_st0.012`, `_st0.02`); fixed. CelebA+CIFAR and natural
saved runs used M=8 (the paper states M=6). See PROVENANCE.md.

## CLWD: diagnosis and aligned variants (see `clwd_diagnosis.md`)

| CLWD variant (blind, Algorithm 1, tether 0, n=99, PSNR on `M_true`) | PSNR_Ω |
|---|---|
| input | 22.55 ± 5.30 |
| **Algorithm 1, dilated benchmark support (headline)** | **22.56 ± 4.31** |
| sign n≥0 / n≤0 | 22.43 / 21.61 |
| M=12 / M=24 (run to M) | 23.28 / 23.52 |
| undilated dataset mask as the support, M=6, τ_stop=0.03 | 25.09 ± 4.87 (protocol variant: the support given to the algorithm is the dataset mask; scoring still on `M_true`) |
| declared-α K=1 + per-pixel readout (not Algorithm 1) | 28.06 ± 6.48 |

Violated assumptions, in order of cost: the support contains ~38% essentially clean pixels (mask geometry, 2.5-3.3 dB); n depends on x (corr −0.42, ~0.8 dB); n is two-sided (~40% of pixels negative, ≤1 dB). With the true n the x-step reaches 34.21, so the bottleneck is the corruption estimate (lines 10-11), not the prior: pretrained or fine-tuned EDM2-64 priors differ by ≤0.3 dB.

## Audio (added 2026-10-11)

Every SORD audio row in the PDF is the K=64 inpainting floor x_0 (`experiments/tango_msdm_2026_08_04/src/audio_ivoov_dps_table.py`,
`floor_win`). With τ_stop = 0.8 the rule returns x_0 on every clip of the iterated runs `results/arms/balGate_K64_<target>_M6`
(110 balanced clips per target × SIR, K=64, M=6, per-clip σ*), so the audio rows are reproduced exactly. Window SI-SDRi:

| target | SIR | x_0 (reported protocol) | τ=0.8 | last x_6 | τ=0.1 |
|---|---|---|---|---|---|
| piano | 0 | 1.93 | 1.93 | 2.15 | 2.34 |
| piano | 5 | 3.19 | 3.19 | 0.04 | 1.88 |
| guitar | 0 | 2.77 | 2.77 | 2.32 | 2.88 |
| guitar | 5 | 4.15 | 4.15 | −0.24 | 2.71 |
| bass | 0 | 12.90 | 12.90 | 12.27 | 14.03 |
| bass | 5 | 13.80 | 13.80 | 9.65 | 13.85 |
| drums | 0 | 11.24 | 11.24 | 12.41 | 13.44 |
| drums | 5 | 10.68 | 10.68 | 8.21 | 11.16 |

At τ_stop = 0.8 the first refinement never reduces the audio σ* by 80% (median first-step drop 1–12%), so audio returns
the inpainting initialization: the audio rows are SORD's x_0, and the text should say so. Iterating helps at SIR 0
(τ=0.1: +0.11..+2.20) and hurts piano/guitar at SIR 5. The g=1 and speech runs saved no iterates, so τ=0.8 → x_0 there
is verified only through the 8 rows above (any clip whose first refinement cut σ* by > 80% would iterate).
