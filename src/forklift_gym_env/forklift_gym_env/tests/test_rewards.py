from itertools import pairwise

import numpy as np
import pytest

from forklift_gym_env.envs.backends.base import WorldState
from forklift_gym_env.envs.rewards import REWARD_TERMS, RewardContext, RewardFunction
from forklift_gym_env.geometry import Pose2D

ZERO = np.zeros(2, np.float32)


def ctx(
    d_prev=5.0,
    d_now=4.0,
    yaw_prev=0.0,
    yaw_now=0.0,
    action=ZERO,
    prev_action=ZERO,
    success=False,
    collision=False,
    gamma=0.99,
):
    return RewardContext(
        prev_state=WorldState(robot=Pose2D(0, 0, yaw_prev), goal=Pose2D(d_prev, 0)),
        state=WorldState(robot=Pose2D(0, 0, yaw_now), goal=Pose2D(d_now, 0)),
        action=np.asarray(action, np.float32),
        prev_action=np.asarray(prev_action, np.float32),
        success=success,
        collision=collision,
        gamma=gamma,
        dt=0.1,
    )


def test_progress_is_positive_when_closing_distance():
    assert REWARD_TERMS["progress"](ctx(5.0, 4.0)) > 0


def test_progress_is_negative_when_retreating():
    assert REWARD_TERMS["progress"](ctx(4.0, 5.0)) < 0


def test_progress_telescopes_over_a_trajectory():
    """With gamma=1 the shaping sums to d_start - d_end, independent of the path."""
    distances = [8.0, 7.1, 7.4, 5.0, 3.2, 0.4]
    total = sum(REWARD_TERMS["progress"](ctx(a, b, gamma=1.0)) for a, b in pairwise(distances))
    assert total == pytest.approx(distances[0] - distances[-1])


def test_progress_pays_a_stationary_agent_when_gamma_below_one():
    """Documents the trap that RewardConfig.check_idling_is_unprofitable guards."""
    idle = REWARD_TERMS["progress"](ctx(8.0, 8.0, gamma=0.98))
    assert idle == pytest.approx((1 - 0.98) * 8.0)
    assert idle > 0


def test_heading_rewards_turning_towards_the_goal():
    turning_to_face = ctx(5.0, 5.0, yaw_prev=1.0, yaw_now=0.2)
    turning_away = ctx(5.0, 5.0, yaw_prev=0.2, yaw_now=1.0)
    assert REWARD_TERMS["heading"](turning_to_face) > 0
    assert REWARD_TERMS["heading"](turning_away) < 0


def test_heading_fades_out_close_to_the_goal():
    """Inside a metre the pallet's bearing is meaningless; the term must not dominate."""
    near = abs(REWARD_TERMS["heading"](ctx(0.3, 0.3, yaw_prev=1.0, yaw_now=0.0)))
    far = abs(REWARD_TERMS["heading"](ctx(6.0, 6.0, yaw_prev=1.0, yaw_now=0.0)))
    assert near < far


def test_one_off_terms_are_indicators():
    assert REWARD_TERMS["success"](ctx(success=True)) == 1.0
    assert REWARD_TERMS["success"](ctx(success=False)) == 0.0
    assert REWARD_TERMS["collision"](ctx(collision=True)) == 1.0


def test_action_terms():
    assert REWARD_TERMS["action_magnitude"](ctx(action=[1.0, 1.0])) == pytest.approx(1.0)
    assert REWARD_TERMS["action_magnitude"](ctx(action=[0.0, 0.0])) == pytest.approx(0.0)
    jerky = REWARD_TERMS["action_smoothness"](ctx(action=[1.0, 1.0], prev_action=[-1.0, -1.0]))
    smooth = REWARD_TERMS["action_smoothness"](ctx(action=[1.0, 1.0], prev_action=[0.9, 0.9]))
    assert jerky > smooth


def test_reverse_term_only_fires_on_negative_linear():
    assert REWARD_TERMS["reverse"](ctx(action=[-0.7, 0.0])) == pytest.approx(0.7)
    assert REWARD_TERMS["reverse"](ctx(action=[0.7, 0.0])) == 0.0


def test_reward_function_sums_weighted_terms_and_reports_the_breakdown():
    fn = RewardFunction({"progress": 2.0, "time": -0.5, "success": 100.0})
    total, breakdown = fn(ctx(5.0, 4.0, success=True))
    assert set(breakdown) == {"progress", "time", "success"}
    assert breakdown["success"] == 100.0
    assert breakdown["time"] == -0.5
    assert total == pytest.approx(sum(breakdown.values()))


def test_zero_weight_terms_are_dropped_entirely():
    fn = RewardFunction({"progress": 1.0, "collision": 0.0})
    _, breakdown = fn(ctx())
    assert "collision" not in breakdown


def test_unknown_term_is_rejected():
    with pytest.raises(ValueError, match="unknown reward term"):
        RewardFunction({"l2_distance": 1.0})
