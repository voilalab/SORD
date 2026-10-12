"""SORD: Support-guided OOD Removal with Diffusion (Algorithm 1)."""
import os

# TensorFlow (imported by the shared model code) otherwise reserves the whole GPU before a PyTorch prior
# (EDM2 / ADM) loads; growth on demand changes no results.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from .algorithm import SORDConfig, SORDResult, sord, sord_batch

__all__ = ["SORDConfig", "SORDResult", "sord", "sord_batch"]
