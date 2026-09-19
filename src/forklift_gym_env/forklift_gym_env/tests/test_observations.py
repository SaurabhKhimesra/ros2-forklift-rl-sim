import math
from itertools import pairwise

import numpy as np
import pytest

from forklift_gym_env.envs.backends.base import WorldState
from forklift_gym_env.envs.observations import OBSERVATION_FEATURES, ObservationBuilder
from forklift_gym_env.geometry import Pose2D

ALL = sorted(OBSERVATION_FEATURES)


def state(x=0.0, y=0.0, yaw=0.0, gx=3.0, gy=4.0, **kw):
    return WorldState(robot=Pose2D(x, y, yaw), goal=Pose2D(gx, gy), **kw)


@pytest.mark.parametrize("name", ALL)
def test_each_feature_emits_its_declared_width(name):
    feature = OBSERVATION_FEATURES[name]
    out = feature(state(), np.zeros(2, np.float32), True)
    assert out.shape == (feature.size,) and out.dtype == np.float32


def test_builder_size_matches_the_vector_it_produces():
    """The regression this locks down: observation_space used to be hard-coded
    to (2,) while the config advertised three feature groups."""
    for subset in ([ALL[0]], ALL[:3], ALL):
        builder = ObservationBuilder(subset)
        vector = builder(state(), np.zeros(2, np.float32))
        assert vector.shape == (builder.size,)
        assert builder.low.shape == builder.high.shape == (builder.size,)


def test_builder_layout_slices_cover_the_vector_exactly():
    builder = ObservationBuilder(ALL)
    layout = builder.layout()
    covered = sorted((s.start, s.stop) for s in layout.values())
    assert covered[0][0] == 0 and covered[-1][1] == builder.size
    for (_, end), (start, _) in pairwise(covered):
        assert end == start


def test_heading_error_is_continuous_across_the_pi_boundary():
    feature = OBSERVATION_FEATURES["heading_error"]
    just_under = feature(state(yaw=math.pi - 1e-4, gx=1.0, gy=0.0), np.zeros(2), True)
    just_over = feature(state(yaw=-math.pi + 1e-4, gx=1.0, gy=0.0), np.zeros(2), True)
    np.testing.assert_allclose(just_under, just_over, atol=1e-3)


def test_heading_error_encodes_sin_cos():
    out = OBSERVATION_FEATURES["heading_error"](state(gx=0.0, gy=1.0), np.zeros(2), True)
    np.testing.assert_allclose(out, [math.sin(math.pi / 2), math.cos(math.pi / 2)], atol=1e-6)


def test_goal_vector_is_in_the_body_frame():
    feature = OBSERVATION_FEATURES["goal_vector_body"]
    facing_east = feature(state(yaw=0.0, gx=2.0, gy=0.0), np.zeros(2), False)
    facing_north = feature(state(yaw=math.pi / 2, gx=0.0, gy=2.0), np.zeros(2), False)
    # Same relative geometry -> same observation, which is what lets a policy
    # generalise instead of memorising one absolute goal.
    np.testing.assert_allclose(facing_east, facing_north, atol=1e-9)


def test_normalisation_keeps_values_small():
    builder = ObservationBuilder(ALL, normalize=True)
    vector = builder(state(x=9.0, y=-9.0, gx=-9.0, gy=9.0), np.ones(2, np.float32))
    assert np.abs(vector).max() <= 5.0


def test_goal_dependence_flags_are_correct():
    dependent = {n for n, f in OBSERVATION_FEATURES.items() if not f.goal_independent}
    assert dependent == {
        "goal_vector_body",
        "goal_distance",
        "heading_error",
        "alignment_error",
        "goal_pose_world",
    }
