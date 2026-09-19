"""Shared fixtures. Nothing here needs ROS, Gazebo or a GPU."""

from __future__ import annotations

import numpy as np
import pytest

from forklift_gym_env.config import Config
from forklift_gym_env.envs.forklift_env import ForkliftEnv


@pytest.fixture
def config() -> Config:
    cfg = Config()
    cfg.run.seed = 0
    cfg.env.max_episode_steps = 40
    cfg.env.kinematic.velocity_noise_std = 0.0  # deterministic for assertions
    cfg.env.goal.curriculum.enabled = False
    cfg.train.total_steps = 200
    cfg.train.warmup_steps = 50
    return cfg.validate()


@pytest.fixture
def env(config: Config) -> ForkliftEnv:
    e = ForkliftEnv(config)
    yield e
    e.close()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
