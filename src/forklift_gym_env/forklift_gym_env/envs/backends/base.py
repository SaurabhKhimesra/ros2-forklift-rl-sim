"""The contract every simulator backend implements.

Splitting the simulator out from the Gym environment is the single biggest
structural change in this rewrite. The environment, the observation builder, the
reward terms and the RL code now depend only on :class:`WorldState`, so all of
them can be exercised at thousands of steps per second against the kinematic
backend and unit-tested in CI with no ROS installed at all.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from forklift_gym_env.geometry import Pose2D


@dataclass(frozen=True, slots=True)
class WorldState:
    """Everything the environment needs to know about the world at one instant."""

    robot: Pose2D
    goal: Pose2D
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    in_collision: bool = False
    sim_time: float = 0.0
    contacts: tuple[str, ...] = field(default=())

    @property
    def goal_distance(self) -> float:
        return self.robot.distance_to(self.goal)

    @property
    def heading_error(self) -> float:
        """Angle between the robot's forward axis and the bearing to the goal."""
        return self.robot.bearing_to(self.goal)

    @property
    def goal_vector_body(self) -> np.ndarray:
        return self.robot.to_local(self.goal)

    @property
    def alignment_error(self) -> float:
        """Difference between the robot's yaw and the pallet's yaw.

        Distinct from :attr:`heading_error`, which is the bearing *to* the
        pallet. Reaching a pallet and being square with it are different
        problems -- forks only go in from one direction.
        """
        from forklift_gym_env.geometry import wrap_angle

        return float(wrap_angle(self.goal.yaw - self.robot.yaw))


@dataclass(frozen=True, slots=True)
class DriveCommand:
    """A velocity command in SI units, already clipped to the robot's limits."""

    linear: float = 0.0
    angular: float = 0.0
    fork: float = 0.0


class SimulationBackend(abc.ABC):
    """Minimal interface a simulator must provide.

    Implementations must be safe to ``close()`` twice and must never block
    forever: every wait has to be bounded by a timeout.
    """

    #: Human-readable name used in logs and run metadata.
    name: str = "backend"

    @abc.abstractmethod
    def reset(self, robot: Pose2D, goal: Pose2D) -> WorldState:
        """Place the robot and the goal, then return the settled world state."""

    @abc.abstractmethod
    def step(self, command: DriveCommand, dt: float) -> WorldState:
        """Apply ``command`` for ``dt`` seconds of simulated time."""

    def close(self) -> None:  # noqa: B027 - intentionally optional, not abstract
        """Release simulator resources. Must be idempotent.

        Not abstract: a backend with nothing to release (the kinematic one) should
        not be forced to write an empty override.
        """

    def __enter__(self) -> SimulationBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
