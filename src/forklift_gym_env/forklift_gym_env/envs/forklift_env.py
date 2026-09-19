"""The Gymnasium environment.

Backend-agnostic: it owns the task (where the goal goes, when the episode ends,
what the reward is) and delegates "what does the world do when I drive" to a
:class:`~forklift_gym_env.envs.backends.base.SimulationBackend`.

Follows the modern Gymnasium five-tuple API and, crucially, distinguishes
``terminated`` (the robot reached the pallet or crashed) from ``truncated`` (the
step budget ran out). The original returned a single ``done`` flag and then
tried to reconstruct the difference in the training loop by comparing counters,
which silently bootstrapped through real terminal states.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised implicitly by the test suite
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required. Install it with `pip install gymnasium`.") from exc

from forklift_gym_env.config import Config, EnvConfig
from forklift_gym_env.envs.actions import DifferentialDriveActuator
from forklift_gym_env.envs.backends.base import SimulationBackend, WorldState
from forklift_gym_env.envs.observations import ObservationBuilder, achieved_goal, desired_goal
from forklift_gym_env.envs.rewards import RewardContext, RewardFunction
from forklift_gym_env.geometry import Pose2D, wrap_angle


class GoalSampler:
    """Samples start/goal pairs, optionally widening the distribution over time.

    The original environment pinned the pallet at ``[6.0, 2.0]`` for every single
    episode (the randomisation code was commented out). A policy trained that way
    has no reason to learn "drive to the pallet" when "drive to (6, 2)" is
    simpler and scores identically -- and it will not transfer to any other
    pallet position.
    """

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg.goal
        self.curriculum = cfg.goal.curriculum
        self._progress = 0.0  # 0 = easiest, 1 = full configured range
        self._outcomes: deque[bool] = deque(maxlen=self.curriculum.window)

    # -- difficulty ---------------------------------------------------- #
    @property
    def progress(self) -> float:
        return self._progress

    def current_ranges(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the ``(distance_range, bearing_range_rad)`` in force right now."""
        lo, hi = self.cfg.distance_range
        b_lo, b_hi = np.deg2rad(self.cfg.bearing_range_deg)
        if not self.curriculum.enabled:
            return (lo, hi), (b_lo, b_hi)

        c = self.curriculum
        max_dist = c.initial_distance + self._progress * (c.final_distance - c.initial_distance)
        max_bearing = np.deg2rad(
            c.initial_bearing_deg + self._progress * (c.final_bearing_deg - c.initial_bearing_deg)
        )
        return (
            (lo, float(np.clip(max_dist, lo, hi))),
            (float(max(b_lo, -max_bearing)), float(min(b_hi, max_bearing))),
        )

    def record_outcome(self, succeeded: bool) -> bool:
        """Log an episode result; returns ``True`` if difficulty was raised."""
        if not self.curriculum.enabled:
            return False
        self._outcomes.append(succeeded)
        c = self.curriculum
        if len(self._outcomes) < c.window or self._progress >= 1.0:
            return False
        if np.mean(self._outcomes) >= c.success_rate_to_advance:
            self._progress = min(1.0, self._progress + c.step_fraction)
            self._outcomes.clear()  # re-measure at the new difficulty
            return True
        return False

    # -- sampling ------------------------------------------------------ #
    def sample(self, rng: np.random.Generator) -> tuple[Pose2D, Pose2D]:
        start = Pose2D(0.0, 0.0, float(rng.uniform(-np.pi, np.pi)))
        if not self.cfg.randomize:
            return start, Pose2D(*self.cfg.fixed_position, 0.0)

        (d_lo, d_hi), (b_lo, b_hi) = self.current_ranges()
        # sqrt-uniform in radius gives a spatially uniform annulus rather than
        # crowding samples near the inner edge.
        u = rng.uniform()
        distance = float(np.sqrt(u * (d_hi**2 - d_lo**2) + d_lo**2))
        bearing = float(rng.uniform(b_lo, b_hi))
        angle = wrap_angle(start.yaw + bearing)
        goal = Pose2D(
            start.x + distance * np.cos(angle),
            start.y + distance * np.sin(angle),
            float(rng.uniform(-np.pi, np.pi)),
        )
        return start, goal


class ForkliftEnv(gym.Env):
    """Drive a forklift to a pallet.

    Parameters
    ----------
    config:
        A validated :class:`~forklift_gym_env.config.Config`.
    backend:
        Optional pre-built backend. If omitted one is created from
        ``config.env.backend``; passing one explicitly is how the tests inject a
        deterministic stub.
    goal_conditioned:
        When ``True`` the observation becomes a ``Dict`` of
        ``observation``/``achieved_goal``/``desired_goal``, which is the layout
        hindsight replay needs. Defaults to ``config.algo.use_her``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: Config,
        backend: SimulationBackend | None = None,
        goal_conditioned: bool | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.config = config
        self.env_cfg = config.env
        self.render_mode = render_mode
        self.goal_conditioned = (
            config.algo.use_her if goal_conditioned is None else goal_conditioned
        )

        self.rng = np.random.default_rng(config.run.seed)
        self._backend = backend if backend is not None else self._make_backend()
        self._obs_builder = ObservationBuilder(
            self.env_cfg.observation, self.env_cfg.normalize_observations
        )
        self._reward_fn = RewardFunction(self.env_cfg.reward.active())
        self._actuator = DifferentialDriveActuator(self.env_cfg.robot)
        self._goal_sampler = GoalSampler(self.env_cfg)

        self.action_space = spaces.Box(-1.0, 1.0, (self._actuator.size,), np.float32)
        obs_box = spaces.Box(self._obs_builder.low, self._obs_builder.high, dtype=np.float32)
        if self.goal_conditioned:
            goal_box = spaces.Box(-np.inf, np.inf, (2,), np.float32)
            self.observation_space = spaces.Dict(
                {"observation": obs_box, "achieved_goal": goal_box, "desired_goal": goal_box}
            )
        else:
            self.observation_space = obs_box

        self._state: WorldState | None = None
        self._prev_state: WorldState | None = None
        self._last_action = np.zeros(self._actuator.size, np.float32)
        self._prev_action = np.zeros(self._actuator.size, np.float32)
        self._elapsed = 0
        self._episode_return = 0.0
        self._trajectory: list[tuple[float, float]] = []

    # ------------------------------------------------------------------ #
    def _make_backend(self) -> SimulationBackend:
        if self.env_cfg.backend == "kinematic":
            from forklift_gym_env.envs.backends.kinematic import KinematicBackend

            return KinematicBackend(self.env_cfg.robot, self.env_cfg.kinematic, self.rng)
        if self.env_cfg.backend == "gazebo":
            # Imported lazily: this pulls in rclpy, which must not be a hard
            # requirement for running the fast sim or the test suite.
            from forklift_gym_env.envs.backends.gazebo import GazeboBackend

            return GazeboBackend(self.env_cfg.robot, self.env_cfg.gazebo)
        raise ValueError(f"unknown backend {self.env_cfg.backend!r}")

    # -- Gymnasium API -------------------------------------------------- #
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            super().reset(seed=seed)
            self.rng = np.random.default_rng(seed)

        start, goal = self._goal_sampler.sample(self.rng)
        if options:
            start = options.get("start", start)
            goal = options.get("goal", goal)

        self._actuator.reset()
        self._last_action = np.zeros(self._actuator.size, np.float32)
        self._prev_action = np.zeros(self._actuator.size, np.float32)
        self._elapsed = 0
        self._episode_return = 0.0

        self._state = self._backend.reset(start, goal)
        self._prev_state = self._state
        self._trajectory = [(self._state.robot.x, self._state.robot.y)]
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("step() called before reset()")

        action = np.clip(np.asarray(action, np.float32).reshape(-1), -1.0, 1.0)
        command = self._actuator(action, self.env_cfg.dt)

        self._prev_state = self._state
        self._state = self._backend.step(command, self.env_cfg.dt)
        self._prev_action, self._last_action = self._last_action, action
        self._elapsed += 1
        self._trajectory.append((self._state.robot.x, self._state.robot.y))

        succeeded = self._is_success(self._state)
        collided = self._state.in_collision and self.env_cfg.terminate_on_collision

        ctx = RewardContext(
            prev_state=self._prev_state,
            state=self._state,
            action=action,
            prev_action=self._prev_action,
            success=succeeded,
            collision=self._state.in_collision,
            gamma=self.config.algo.gamma,
            dt=self.env_cfg.dt,
        )
        reward, breakdown = self._reward_fn(ctx)
        self._episode_return += reward

        terminated = bool(succeeded or collided)
        truncated = bool(not terminated and self._elapsed >= self.env_cfg.max_episode_steps)

        info = self._info()
        info["reward_terms"] = breakdown
        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_return,
                "l": self._elapsed,
                "success": succeeded,
                "collision": self._state.in_collision,
                "final_distance": self._state.goal_distance,
            }
            self._goal_sampler.record_outcome(succeeded)

        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self._backend.close()

    # -- goal-conditioned helpers --------------------------------------- #
    def compute_reward(
        self,
        achieved: np.ndarray,
        desired: np.ndarray,
        info: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Sparse reward used by hindsight relabelling.

        Vectorised over a batch. Deliberately sparse (success bonus only): the
        dense shaping terms depend on the *previous* state, which relabelling
        does not preserve, so mixing them in here would be incorrect.
        """
        del info
        achieved = np.atleast_2d(np.asarray(achieved, np.float32))
        desired = np.atleast_2d(np.asarray(desired, np.float32))
        distance = np.linalg.norm(achieved - desired, axis=-1)
        hit = distance <= self.env_cfg.goal.position_tolerance
        return (hit * self.env_cfg.reward.success).astype(np.float32)

    def is_success(self, achieved: np.ndarray, desired: np.ndarray) -> np.ndarray:
        achieved = np.atleast_2d(np.asarray(achieved, np.float32))
        desired = np.atleast_2d(np.asarray(desired, np.float32))
        return np.linalg.norm(achieved - desired, axis=-1) <= self.env_cfg.goal.position_tolerance

    # -- internals ------------------------------------------------------ #
    def _is_success(self, state: WorldState) -> bool:
        if state.goal_distance > self.env_cfg.goal.position_tolerance:
            return False
        if not self.env_cfg.goal.require_heading:
            return True
        yaw_error = abs(wrap_angle(state.goal.yaw - state.robot.yaw))
        return yaw_error <= np.deg2rad(self.env_cfg.goal.heading_tolerance_deg)

    def _observation(self) -> Any:
        assert self._state is not None
        vector = self._obs_builder(self._state, self._last_action)
        if not self.goal_conditioned:
            return vector
        return {
            "observation": vector,
            "achieved_goal": achieved_goal(self._state),
            "desired_goal": desired_goal(self._state),
        }

    def _info(self) -> dict[str, Any]:
        assert self._state is not None
        return {
            "distance": self._state.goal_distance,
            "heading_error": self._state.heading_error,
            "robot": (self._state.robot.x, self._state.robot.y, self._state.robot.yaw),
            "goal": (self._state.goal.x, self._state.goal.y),
            "collision": self._state.in_collision,
            "contacts": self._state.contacts,
            "curriculum_progress": self._goal_sampler.progress,
            "is_success": self._is_success(self._state),
        }

    # -- introspection used by the plotting helpers and by scripted probes -- #
    @property
    def state(self) -> WorldState:
        """The current world state.

        Public because scripted baselines and diagnostics legitimately need the
        ground truth, which the observation deliberately does not always carry.
        """
        if self._state is None:
            raise RuntimeError("state is not available before reset()")
        return self._state

    @property
    def trajectory(self) -> np.ndarray:
        return np.asarray(self._trajectory, dtype=np.float64)

    @property
    def goal_sampler(self) -> GoalSampler:
        return self._goal_sampler

    @property
    def observation_layout(self) -> dict[str, slice]:
        return self._obs_builder.layout()
