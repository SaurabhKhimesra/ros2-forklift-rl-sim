"""Small, dependency-light geometry helpers.

Everything here is pure ``numpy``/``math`` so it can be unit-tested without ROS,
Gazebo or torch. Angles are radians unless the name says otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (or array of angles) to ``[-pi, pi)``.

    This matters more than it looks. A heading error computed as a raw
    subtraction jumps by ``2*pi`` as the robot crosses the +/-pi boundary, which
    puts a step discontinuity straight into the observation *and* the reward.
    Wrapping removes it.
    """
    return (
        (np.asarray(angle) + math.pi) % TWO_PI - math.pi
        if isinstance(angle, np.ndarray)
        else (angle + math.pi) % TWO_PI - math.pi
    )


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract the yaw (rotation about +Z) from a quaternion.

    Avoids a dependency on ``tf_transformations``, which is a ROS-only package
    and was previously imported inside a hot loop.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Return ``(x, y, z, w)`` for a pure yaw rotation."""
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


@dataclass(frozen=True, slots=True)
class Pose2D:
    """A planar pose. Immutable so it can be safely stashed in trajectories."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    def distance_to(self, other: Pose2D) -> float:
        return float(math.hypot(other.x - self.x, other.y - self.y))

    def bearing_to(self, other: Pose2D) -> float:
        """Heading error to ``other``, wrapped to ``[-pi, pi)``.

        Positive means the target is to the robot's left.
        """
        return wrap_angle(math.atan2(other.y - self.y, other.x - self.x) - self.yaw)

    def to_local(self, other: Pose2D) -> np.ndarray:
        """Express ``other``'s position in this pose's body frame.

        Body-frame coordinates are translation- and rotation-invariant, so a
        policy trained on them generalises to goals anywhere in the world
        instead of memorising one absolute location.
        """
        dx, dy = other.x - self.x, other.y - self.y
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy], dtype=np.float64)
