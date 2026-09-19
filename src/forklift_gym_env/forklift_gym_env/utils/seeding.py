"""Seeding helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = False) -> np.random.Generator:
    """Seed python, numpy and torch, and return a dedicated ``Generator``.

    Returning a ``Generator`` matters: the previous code seeded the *global*
    numpy RNG and then used ``np.random.*`` everywhere, so the environment, the
    replay buffer and the exploration noise all drew from one shared stream.
    Any change to how often one of them sampled silently changed the others,
    which makes a "reproducible" run reproducible only by accident.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)
