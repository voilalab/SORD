# SORD implementation notes

The following differences between the submitted manuscript's Algorithm 1 and the implementation were documented in the original README. They are retained here because they affect interpretation and reproducibility. These observations are based on the supplied repository description; recheck against the exact source revision before publication.

- **Residual estimation:** The manuscript describes one residual per sample with denoising at the estimated uncertainty level. The code uses one residual from the mean reconstruction, clips to the corruption class, crops and normalizes it, and uses a diffusion ladder level with a minimum residual noise floor.
- **Stopping:** The code evaluates the uncertainty reduction after the candidate refinement and returns the *previous* reconstruction when the update fails the stopping threshold. Thus the initial inpainting solution can be returned; the manuscript describes a differently positioned test.
- **Iteration counts:** The manuscript states a common iteration count, whereas benchmark configuration files specify task-dependent limits.
- **Shadows:** The implementation includes a multiplicative/log-domain variant.
- **Uncertainty pooling:** The manuscript experiments pooled uncertainty over chunks of 25 images; the documented code computes it per image. This can alter stopping decisions.

For the full hyperparameter values and benchmark-specific settings, consult [`../configs/sord/`](../configs/sord/). For evaluations and result provenance, consult [`../reproduction/`](../reproduction/) and [`../reproduction/PROVENANCE.md`](../reproduction/PROVENANCE.md).
