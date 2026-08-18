"""Reproducibility helpers.

Responsibility: seed all RNGs (python, numpy, torch, cuda) from a single seed so every run is
reproducible, per the project's reproducibility-first convention (CLAUDE.md).

Used by: experiments/*, env reset.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed python/numpy/torch RNGs. Set ``deterministic_torch`` for fully deterministic CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
