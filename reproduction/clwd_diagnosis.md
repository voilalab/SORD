# Why blind Algorithm 1 gets 22.56 dB on CLWD (b2r, 64x64, n=99) and what closes the gap

All numbers: on-support PSNR on the benchmark support M_true (dilated CLWD mask, 14.2% of pixels), n=99 images (1 empty-mask image dropped),
mean +- std over images, deltas paired against the saved shipped run `b2r/minDr` (22.56). Tether 0, no gamma-shrink, K=32, EDM2-64 CLWD prior
(`trackd_clwd64/...-0.100.pkl`), sign free unless stated. tau_stop = 0.03 replayed on saved iterates (same trajectory as online stopping).
Scripts: `SORD/experiments/clwd_diagnosis/` (offline_diag.py, offline_geometry.py, compare.py, score_run.py, oracle_run.py, jobq*.py, run_*.sh).
Run dirs: `experiments/extbench/results/b2r/{diag_pos,diag_neg,diag_M24,tight_free,maskraw_free,maskraw_M24,pre_free,prwd_free,oracle_n,sch_*}`, `b2rctl/ctl_free`.
Note: the 28.06 in the paper is the declared-alpha K=1 arm, not Algorithm 1. The input itself (y on Omega) scores 22.55 +- 5.30, so shipped Algorithm 1 == input (+0.01 +- 4.24, 52/99 up).

## A. Sign declaration
- True n=y-x on the support is two-sided: negative (< -1e-3) at 45.1 / 37.6 / 37.3 % of support pixels (R/G/B); positive at 44 / 49 / 54 %. Per image the negative fraction is 0.41 +- 0.21. n>=0 is false for ~40% of the support.
- Algorithm 1, only --sign changed (one run each, n=99):

| sign | PSNR (last = tau-stop .03) | paired vs free 22.56 | vs input |
|---|---|---|---|
| free (shipped) | 22.56 +- 4.31 | 0 | +0.01 |
| pos (n>=0, what the user asked to check) | 22.43 +- 4.84 | -0.13 +- 2.91 (48/99 up) | -0.12 |
| neg (n<=0) | 21.61 +- 4.80 (stop 21.72) | -0.95 +- 3.18 | -0.94 |

  Per-iterate means: free m1..m6 20.20 -> 22.56; pos 21.47 -> 22.43 (starts higher, saturates lower); neg 21.13 -> 21.61 (flat).
  So n>=0 neither fixes nor (much) breaks the blind loop; the declaration is not the lever. It is the wrong class (a wrong declaration only costs 0.1 dB because the n-step output stays near the evidence anyway), but nothing is gained by switching.
- Bracket clip (box_bounds): with sign free the box is n in [y-1, y], i.e. x in [0,1], which every alpha-blend satisfies for some (a,w), so it discards nothing and the inpaint "box" arm equals plain inpaint (18.93). With sign pos the clip n>=0 is a false constraint on ~41% of pixels; it still raises the box arm to ~20.3 (chunk means 20.5/20.3/20.6/19.9, vs 18.5-20.0 for inpaint) because it pulls inpainted pixels toward y, but the loop ends at 22.43.

## B. Which assumption fails (oracles on saved/true data)
(i) Oracle n (n-step replaced by true y-x, lines 12-13 only, same K/schedule): **34.21 +- 1.97** (+11.65 +- 3.75 vs shipped, 99/99 up). The x-step / prior / schedule are NOT the bottleneck; the n-step (lines 10-11) is. Real n-step quality at m=6: RMSE(n_hat,n)=0.08 vs RMS(n)=0.09 (explains ~20% of n energy), sign agreement with true n 58%.
(ii) Geometry. 38 +- 13 % of mask pixels have true alpha < 0.05 (essentially clean; input is **40.5 dB** there, the loop turns them into 26.0, inpaint 21.2). On alpha>=0.05 pixels: input 20.4, inpaint 18.1, loop 21.5 (+1.1). Compositing input on alpha<0.05 pixels and loop elsewhere gives only 23.61 (+1.06): geometry alone is a modest part when the same loop runs on the large mask; but *running* the loop on a tighter support helps much more (below): tight support (alpha>=.05, uses true alpha, DIAGNOSTIC) 25.90 +- 5.09 (+3.34 +- 2.00, 97/99 up; on its own support 23.89 vs input 20.43). Per-pixel oracle pick between y and loop output: 27.09.
   Prior: pretrained EDM2-img64 (not fine-tuned) 22.30 +- 4.26 (-0.26 +- 1.11, 26/99 up); PRWD-fine-tuned EDM2 22.47 +- 4.33 (-0.10 +- 0.43); so the CLWD prior is not the bottleneck (and clwd64 has zero train/held-out gap per memory). Native-256 priors (existing runs, M=8, b2r256 bench, input floor 19.38): CLWD-256 DDPM 14.03 +- 2.87 (-5.35 vs input, 1/99 up); ImageNet-256 ADM n=74 only (chunk c0 never ran) 20.42 +- 4.50 vs input 19.30 (+1.12, tau-stop 20.80, +1.50). Not re-run here (not 64x64, different bench scale).
   Stopping: loop is monotone increasing at M=6 (stop == last, 22.56); sigma* only falls.
(iii) Per-image: Spearman of loop PSNR with mean alpha -0.73, mask area -0.59, RMS(n) -0.66, background std -0.53, logo luminance -0.41; gain over input vs input PSNR -0.64 (the loop only helps images that are badly corrupted). Tertiles by mean alpha (0-.1/.1-.2/>.2): input 27.1/22.8/17.9, loop 25.9/22.4/19.5 (loop loses on faint marks, gains +1.6 on opaque ones). By mask area (small/mid/large): input 26.2/21.8/19.6, loop 25.9/21.5/20.3.
(iv) Violated assumptions. (a) "n independent of x": on CLWD n=a(w-x) so corr(n,x) = -0.42, slope -0.17 (n carries a negative ghost of the scene). Control with the same masks/alpha/logo but x replaced by another scene inside n (corr -0.02; y'=x+n_true+a(x-x_perm), clipped to [0,1]): loop 23.48 +- 4.23 vs input 22.67 (+0.81 +- 3.82, 67/99 up) versus real +0.01, i.e. x-dependence costs ~0.8 dB. (b) "support-only corruption": 38% of the support is essentially clean and the 14% blob mask is large (inpaint floor 18.9, below input) - the dominant, mask-geometry cost (+2.5..+3.3 dB when removed). (c) sign (non-negative): false for 40% of pixels but worth <=1 dB. (d) "additive": inverting the true blend (true alpha + registered logo colour) gives 29.61 +- 5.04 (+7.06 over input) -- the blend structure is what the declared-alpha arm exploits. A blind additive n-step recovers little of a faint, signed, scene-correlated, blob-shaped n (the n-step treats n as a clean image through D_theta after affine normalisation, sign agreement 58%).

## C. Legitimate levers on blind Algorithm 1 (support known, alpha not given, clean prior only)
All with tether 0, no gamma-shrink. "stop" = tau_stop 0.03 replay, "last" = M-th iterate. Scored on M_true.

| lever / config | n | last | stop | paired (stop or last) vs 22.56 | keeps paper claims? |
|---|---|---|---|---|---|
| shipped minDr (M6, dilated M_true, sign free) | 99 | 22.56 +- 4.31 | 22.56 | 0 | yes |
| sign pos / neg | 99 | 22.43 / 21.61 | 22.43 / 21.72 | -0.13 / -0.95 | yes (pos is the user's declaration) |
| M=12 (existing minDr_M12) | 99 | 23.28 +- 4.46 | 23.28 | +0.72 +- 1.24 | yes |
| M=24, no stop | 99 | 23.52 +- 4.55 | 21.24 | +0.95 +- 1.93 (last); tau-stop .03 -1.32 (stops at m~3: the relative rule fires on the fine grid; tau=0 stops ~m11, 22.70) | yes if run to M; tau rule not usable at M24 |
| sigma_m 0.8->0.1, M12 | 99 | 23.08 +- 4.29 | - | +0.52 (below M12 baseline 23.28) | yes |
| sigma_m 0.4->0.05, M12 | 99 | 23.29 +- 4.47 | 22.89 | +0.73 (= M12) | yes |
| sigma schedules 0.8/0.4/0.2, 0.9/0.6/0.3/0.1, const 0.8 (earlier campaigns, memory, not rerun) | | | | about -0.7..-1 on b2r | yes |
| prior: EDM2-64 pretrained / PRWD-FT | 99 | 22.30 / 22.47 | | -0.26 / -0.10 | yes (clean prior) |
| prior: CLWD-256 DDPM, ImageNet-256 ADM (256 bench) | 99 / 74 | 14.03 / 20.42 | 14.03 / 20.80 | -5.35 / +1.12(+1.50) vs input | yes, different scale |
| **support = dataset Mask.png (soft>0.05, NO 1-px dilation, area 10.7%), M=6** | 99 | 25.05 +- 4.88 | **25.09 +- 4.87** | **+2.49 +- 1.48 / +2.52 +- 1.51 (98/99 up)** | yes if the undilated dataset mask is accepted as "the support" (halo ring then treated as clean; scoring still on M_true) |
| same support, M=24 | 99 | 25.17 +- 5.15 | 24.14 | +2.61 / +1.58 | yes; M24 adds only +0.1 |
| tight support alpha>=0.05 (derived from TRUE alpha) | 99 | 25.90 +- 5.09 | 25.93 | +3.34 | NO as headline (uses alpha to build the mask); diagnostic |
| K, tau_stop variation | not run (tau_stop inert at M6; K not tried) | | | | |

Best aligned blind Algorithm-1 number: **25.09 +- 4.87 dB** (mask = dataset Mask.png soft>0.05 undilated; K=32, M=6, sigma_m 0.4->0.1 linear, sign free, tau_stop 0.03, CLWD-fine-tuned EDM2-64 `trackd_clwd64 network-snapshot-0000512-0.100.pkl`, flags `--K 32 --gamma-grid 1 --w0 --no-exchange --no-harmonize`, mask file `experiments/extbench/results/b2r/mask_raw_dataset.npz`, run `b2r/maskraw_free`). With the paper's own mask (dilated M_true) the best is M=24 no-stop 23.52 (or M=12 23.28). Neither reaches 28.06; the remaining ~3 dB gap is the information in the blend model (declared alpha), not a tuning issue: even with a perfect n the loop gives 34.2, while a blind n-step gets ~58% sign agreement.

## D. Diagnostic subset analysis (NOT a result; no subset is selected to report)
Shipped Alg.1: 10/99 images reach >=28 dB (input alone: 13/99); remaining 89 average 21.62. Best aligned config (undilated mask): 26/99 reach >=28 (remaining 73 average 22.85). By mean alpha (<.1 / .1-.2 / >.2; n=24/46/29): 29.54 / 25.70 / 20.44 (shipped 26.51 / 22.65 / 19.15); by mask area (<.08 / .08-.18 / >.18; n=35/34/30): 28.60 / 24.16 / 22.04 (shipped 25.83 / 21.26 / 20.22). I.e. ~28 dB is reached only on small-area, low-alpha marks, where the input already scores 26-27 dB.

## Recommendation for the paper CLWD cell
Keep 22.56 as the blind Algorithm-1 CLWD number (equal to the input floor 22.55; honest "does not help on faint CLWD marks") and say the 28.06 is the declared-alpha extension (K=1 + per-pixel readout). If the undilated dataset mask is considered the legitimate support, the blind Algorithm-1 cell can be 25.1 +- 4.9 with the mask choice stated explicitly (it changes the benchmark protocol relative to the shipped M_true, and scoring on M_true treats the halo ring as clean). Violated assumptions to state: n depends on x (n=a(w-x), corr -0.42), n is two-sided (~40% negative), and the mask contains ~38% essentially clean pixels.

## Process notes
- First launch put 2 jobs per GPU; shared GPUs also host a 22 GB foreign process, two sign chunks hit CUDA OOM (diag_neg c0, diag_pos c25) and were re-run sequentially (1 job/GPU thereafter). Oracle runs failed once because my queue.py shadowed the stdlib `queue` (fixed by renaming to jobq*.py). ImageNet-256 b2r256 chunk c0 (25 images) was never produced by the earlier campaign; not re-run.
- Other agents wrote jobq7-9.py, edge_alpha.py, fitscale_readout.py, range_alpha.py, signal_alpha.py into the same directory; not mine.
