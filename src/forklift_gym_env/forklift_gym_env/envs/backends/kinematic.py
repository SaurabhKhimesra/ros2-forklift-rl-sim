"""A fast, ROS-free forklift simulator.

Gazebo is the right tool for validating contact, sensors and controller
dynamics, but it is a terrible tool for iterating on a reward function: one
Gazebo step costs a service round-trip, a physics tick and a process
synchronisation, and every ``reset()`` in the original code paid a hard-coded
six seconds of ``sleep``. Tuning a reward that way takes days.

This backend integrates a unicycle model with the same interface, at roughly
half a million steps per second on one core. The workflow it enables:

1. iterate on rewards, observations and hyperparameters here, in seconds;
2. confirm the learning curve is healthy;
3. move the *same* config to ``backend: gazebo`` for the physics-accurate run.

It is deliberately not a physics engine. It models what matters for the
navigation task -- kinematics, actuation noise, circular obstacles and world
bounds -- and nothing else.
"""

from __future__ import annotations

import math

import numpy as np

from forklift_gym_env.config import KinematicConfig, RobotConfig
from forklift_gym_env.envs.backends.base import DriveCommand, SimulationBackend, WorldState
from forklift_gym_env.geometry import Pose2D, wrap_angle


class KinematicBackend(SimulationBackend):
    """Unicycle-model forklift on a bounded plane with circular obstacles."""

    name = "kinematic"

    def __init__(
        self,
        robot: RobotConfig,
        config: KinematicConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.robot_cfg = robot
        self.cfg = config or KinematicConfig()
        self.rng = rng or np.random.default_rng()
        self._obstacles = np.asarray(self.cfg.obstacles, dtype=np.float64).reshape(-1, 3)
        self._pose = Pose2D()
        self._goal = Pose2D()
        self._linear = 0.0
        self._angular = 0.0
        self._sim_time = 0.0

    # ------------------------------------------------------------------ #
    def reset(self, robot: Pose2D, goal: Pose2D) -> WorldState:
        self._pose = robot
        self._goal = goal
        self._linear = 0.0
        self._angular = 0.0
        self._sim_time = 0.0
        return self._state()

    def step(self, command: DriveCommand, dt: float) -> WorldState:
        linear, angular = self._apply_actuation_noise(command)

        # Exact integration of the unicycle model over the interval. Using the
        # closed form rather than an Euler step keeps the trajectory accurate
        # even at a coarse 10 Hz control rate, where Euler visibly cuts corners.
        yaw = self._pose.yaw
        if abs(angular) < 1e-6:
            dx = linear * dt * math.cos(yaw)
            dy = linear * dt * math.sin(yaw)
            new_yaw = yaw
        else:
            radius = linear / angular
            new_yaw = wrap_angle(yaw + angular * dt)
            dx = radius * (math.sin(new_yaw) - math.sin(yaw))
            dy = -radius * (math.cos(new_yaw) - math.cos(yaw))

        x, y = self._pose.x + dx, self._pose.y + dy

        half_w, half_h = self.cfg.world_bounds
        out_of_bounds = abs(x) > half_w or abs(y) > half_h
        x = float(np.clip(x, -half_w, half_w))
        y = float(np.clip(y, -half_h, half_h))

        self._pose = Pose2D(x, y, new_yaw)
        self._linear, self._angular = linear, angular
        self._sim_time += dt

        return self._state(extra_collision=out_of_bounds)

    def close(self) -> None:  # nothing to release
        return None

    # ------------------------------------------------------------------ #
    def _apply_actuation_noise(self, command: DriveCommand) -> tuple[float, float]:
        """Multiplicative noise standing in for wheel slip and controller lag.

        Small, but enough that a policy cannot learn to exploit a perfectly
        deterministic transition model -- which is the usual way a fast
        surrogate sim produces a policy that collapses in Gazebo.
        """
        sigma = self.cfg.velocity_noise_std
        if sigma <= 0.0:
            return command.linear, command.angular
        scale = 1.0 + self.rng.normal(0.0, sigma, size=2)
        return float(command.linear * scale[0]), float(command.angular * scale[1])

    def _collides(self, x: float, y: float) -> tuple[bool, tuple[str, ...]]:
        if self._obstacles.size == 0:
            return False, ()
        d = np.hypot(self._obstacles[:, 0] - x, self._obstacles[:, 1] - y)
        hit = d < (self._obstacles[:, 2] + self.robot_cfg.footprint_radius)
        if not hit.any():
            return False, ()
        return True, tuple(f"obstacle_{i}" for i in np.flatnonzero(hit))

    def _state(self, extra_collision: bool = False) -> WorldState:
        hit, contacts = self._collides(self._pose.x, self._pose.y)
        if extra_collision:
            hit, contacts = True, (*contacts, "world_bounds")
        return WorldState(
            robot=self._pose,
            goal=self._goal,
            linear_velocity=self._linear,
            angular_velocity=self._angular,
            in_collision=hit,
            sim_time=self._sim_time,
            contacts=contacts,
        )
