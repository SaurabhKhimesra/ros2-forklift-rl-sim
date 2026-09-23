"""Seeding helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = False) -> np.random.Generator:
    """Seed python, numpy and torch, and return a dedicated ``Generator``.

    Returning a ``Generator`` matters. Seeding the global numpy RNG and calling
    ``np.random.*`` everywhere puts the environment, the replay buffer and the
    exploration noise on one shared stream, so changing how often any one of
    them samples silently shifts the others -- which makes a "reproducible" run
    reproducible only by accident.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)
