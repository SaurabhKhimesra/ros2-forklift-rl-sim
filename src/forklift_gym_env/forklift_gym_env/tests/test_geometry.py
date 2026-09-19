import math

import numpy as np
import pytest

from forklift_gym_env.geometry import (
    Pose2D,
    quaternion_from_yaw,
    wrap_angle,
    yaw_from_quaternion,
)


@pytest.mark.parametrize(
    "raw, expected",
    [(0.0, 0.0), (math.pi, -math.pi), (-math.pi, -math.pi), (3 * math.pi, -math.pi), (0.5, 0.5)],
)
def test_wrap_angle_scalar(raw, expected):
    assert wrap_angle(raw) == pytest.approx(expected, abs=1e-9)


def test_wrap_angle_is_always_in_range():
    values = np.linspace(-50, 50, 2001)
    wrapped = wrap_angle(values)
    assert np.all(wrapped >= -math.pi) and np.all(wrapped < math.pi)


def test_wrap_angle_has_no_discontinuity_in_magnitude():
    """The bug this guards: a raw subtraction jumps by 2*pi across the boundary."""
    before, after = wrap_angle(math.pi - 1e-6), wrap_angle(math.pi + 1e-6)
    assert abs(abs(before) - abs(after)) < 1e-5


def test_quaternion_yaw_roundtrip():
    for yaw in np.linspace(-math.pi + 1e-6, math.pi - 1e-6, 37):
        assert yaw_from_quaternion(*quaternion_from_yaw(yaw)) == pytest.approx(yaw, abs=1e-9)


def test_bearing_sign_is_left_positive():
    origin = Pose2D(0, 0, 0)
    assert origin.bearing_to(Pose2D(1, 1)) > 0  # target to the left
    assert origin.bearing_to(Pose2D(1, -1)) < 0  # target to the right
    assert origin.bearing_to(Pose2D(1, 0)) == pytest.approx(0.0)


def test_bearing_wraps_behind_the_robot():
    """Facing +x with the goal directly behind: the error must be +/-pi, not +pi only."""
    assert abs(Pose2D(0, 0, 0).bearing_to(Pose2D(-1, 0))) == pytest.approx(math.pi)


def test_to_local_is_translation_and_rotation_invariant():
    goal_offset = np.array([2.0, 1.0])
    for yaw in np.linspace(-math.pi, math.pi, 13):
        for shift in ([0, 0], [5, -3], [-8, 8]):
            c, s = math.cos(yaw), math.sin(yaw)
            rotated = np.array(
                [c * goal_offset[0] - s * goal_offset[1], s * goal_offset[0] + c * goal_offset[1]]
            )
            robot = Pose2D(shift[0], shift[1], yaw)
            goal = Pose2D(shift[0] + rotated[0], shift[1] + rotated[1])
            np.testing.assert_allclose(robot.to_local(goal), goal_offset, atol=1e-9)
