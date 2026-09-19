"""Action space and the mapping from normalised actions to drive commands.

The agent always acts in ``[-1, 1]^n`` -- that is what a ``tanh`` policy head
emits natively, and it keeps the exploration-noise scale meaningful across
robots. Converting to SI units and enforcing the robot's acceleration limits
happens here, in one place, instead of being scattered through the env.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forklift_gym_env.config import RobotConfig
from forklift_gym_env.envs.backends.base import DriveCommand


@dataclass
class DifferentialDriveActuator:
    """Maps ``[-1, 1]^2`` to ``(linear, angular)`` velocity with rate limiting.

    Rate limiting is what stops the policy from commanding a full
    forward-to-reverse flip in a single 100 ms control tick, which no real
    forklift can follow and which makes a Gazebo-trained policy untransferable.
    """

    robot: RobotConfig
    _last_linear: float = 0.0
    _last_angular: float = 0.0

    #: Number of scalars the policy must output.
    size: int = 2

    def reset(self) -> None:
        self._last_linear = 0.0
        self._last_angular = 0.0

    def __call__(self, action: np.ndarray, dt: float) -> DriveCommand:
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if action.size != self.size:
            raise ValueError(f"expected {self.size} action values, got {action.size}")

        target_linear = float(action[0]) * self.robot.max_linear_speed
        if not self.robot.allow_reverse:
            # Remap [-1, 1] -> [0, 1] rather than clipping, so half the action
            # range does not collapse onto "stopped".
            target_linear = (float(action[0]) + 1.0) * 0.5 * self.robot.max_linear_speed
        target_angular = float(action[1]) * self.robot.max_angular_speed

        linear = _rate_limit(self._last_linear, target_linear, self.robot.max_linear_accel, dt)
        angular = _rate_limit(self._last_angular, target_angular, self.robot.max_angular_accel, dt)
        self._last_linear, self._last_angular = linear, angular
        return DriveCommand(linear=linear, angular=angular)


def _rate_limit(current: float, target: float, max_rate: float, dt: float) -> float:
    # Plain float arithmetic: ``np.clip`` on scalars costs ~2 us of dispatch,
    # which is real money when the fast sim runs ~10k control ticks a second.
    max_delta = max_rate * dt
    if target > current + max_delta:
        return current + max_delta
    if target < current - max_delta:
        return current - max_delta
    return target
