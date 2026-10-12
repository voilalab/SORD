"""Sampler configuration for the SGM baseline (verbatim from the authors' comparison script).

The joint sampler of Stevens et al. is set up through the repository's inference pipeline
(`inference.inference_setup`) with one batch of k images, the given number of noise scales and the
PiGDM guidance.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from inference import inference_setup
from sord.paths import CACHE_ROOT, PROJECT_ROOT


def build_config(experiment, k, num_scales, guidance, run_id_sgm=None):
    if run_id_sgm is not None:
        # the inference config names the clean-image prior by a repository-relative directory; point it at the
        # checkpoint resolved through checkpoint_manifest.json (same weights)
        cfg = yaml.safe_load(open(PROJECT_ROOT / "configs/inference" / f"{experiment}.yaml"))
        cfg["run_id"]["sgm"] = str(run_id_sgm)
        resolved = Path(CACHE_ROOT) / "sgm" / f"{Path(experiment).name}.yaml"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
        experiment = str(resolved)
    args = SimpleNamespace(
        experiment=experiment,
        task="run_metrics",          # -> dataset=None, no celeba loader, gpu set
        num_img=k,
        models=["sgm"],
        sweep=False,
        eval_folder=None,
        data_paths=None,
    )
    config, _dataset, run_ids = inference_setup(args)
    # single batch, no animation, compute metrics ourselves later
    config.batch_size = k
    config.keep_track = False
    config.num_img = k
    config.sgm.batch_size = k
    config.sgm.num_scales = num_scales
    if guidance:
        config.sgm.guidance = guidance
    return config, run_ids
