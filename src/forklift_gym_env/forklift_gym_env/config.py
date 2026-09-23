"""Typed, validated configuration.

Every section is a dataclass. Unknown keys are rejected at load time with the
valid names listed, and values are range-checked before a simulator is started,
so a typo fails immediately rather than doing nothing or surfacing hundreds of
lines later inside the training loop.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a config file is malformed, out of range or has typos."""


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    name: str = "forklift"
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    output_dir: str = "runs"

    def validate(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ConfigError(f"run.device must be auto|cpu|cuda, got {self.device!r}")


@dataclass
class RobotConfig:
    """Physical limits of the forklift, shared by both backends."""

    max_linear_speed: float = 1.0  # m/s
    max_angular_speed: float = 1.0  # rad/s
    max_linear_accel: float = 2.0  # m/s^2  (rate limit, keeps commands smooth)
    max_angular_accel: float = 4.0  # rad/s^2
    allow_reverse: bool = True
    footprint_radius: float = 0.6  # m, used for collision checks in the fast sim

    def validate(self) -> None:
        for name in ("max_linear_speed", "max_angular_speed", "footprint_radius"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"robot.{name} must be > 0")


@dataclass
class CurriculumConfig:
    """Grow the task difficulty as the agent gets good at it.

    Starting every episode with the goal 8 m away and behind the robot means the
    early replay buffer is almost all failure, which is exactly the regime where
    a deterministic-policy method like TD3 struggles. Starting easy and widening
    the goal distribution on success is cheap and dramatically speeds up the
    first useful policy.
    """

    enabled: bool = True
    initial_distance: float = 2.5
    final_distance: float = 8.0
    initial_bearing_deg: float = 45.0
    final_bearing_deg: float = 180.0
    success_rate_to_advance: float = 0.7
    window: int = 50  # episodes used to estimate the success rate
    step_fraction: float = 0.1  # how much of the remaining range to unlock per promotion

    def validate(self) -> None:
        if not 0.0 < self.success_rate_to_advance <= 1.0:
            raise ConfigError("curriculum.success_rate_to_advance must be in (0, 1]")
        if self.initial_distance > self.final_distance:
            raise ConfigError("curriculum.initial_distance must be <= final_distance")
        if self.window < 1:
            raise ConfigError("curriculum.window must be >= 1")


@dataclass
class GoalConfig:
    randomize: bool = True
    distance_range: tuple[float, float] = (2.0, 8.0)
    bearing_range_deg: tuple[float, float] = (-180.0, 180.0)
    fixed_position: tuple[float, float] = (6.0, 2.0)  # used when randomize=False
    position_tolerance: float = 0.6  # m
    heading_tolerance_deg: float = 25.0  # only checked if require_heading
    require_heading: bool = False
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    def validate(self) -> None:
        lo, hi = self.distance_range
        if lo <= 0 or hi < lo:
            raise ConfigError(
                f"goal.distance_range must be 0 < lo <= hi, got {self.distance_range}"
            )
        if self.position_tolerance <= 0:
            raise ConfigError("goal.position_tolerance must be > 0")
        self.curriculum.validate()


@dataclass
class RewardConfig:
    """Weights for the composable reward terms.

    A term with weight ``0.0`` is skipped entirely, so this doubles as the
    on/off switch. See :mod:`forklift_gym_env.envs.rewards` for what each does.
    """

    progress: float = 1.0  # potential-based shaping on distance-to-goal
    heading: float = 0.05  # potential-based shaping on bearing-to-goal
    alignment: float = 0.0  # potential-based shaping on yaw match with the pallet
    success: float = 100.0  # one-off bonus for reaching the goal
    collision: float = -50.0  # one-off penalty, ends the episode
    time: float = -0.12  # per-step cost; must out-cost shaping, see below
    action_magnitude: float = -0.01
    action_smoothness: float = -0.02  # penalise jerky command changes
    reverse: float = 0.0  # extra penalty on negative linear velocity

    def active(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != 0.0}

    def validate(self) -> None:
        if not self.active():
            raise ConfigError("reward: every term is zero, the agent has nothing to learn from")

    def check_idling_is_unprofitable(self, gamma: float, max_distance: float) -> None:
        """Reject reward weights that pay the agent to stand still.

        Potential-based shaping contributes ``gamma*Phi(s') - Phi(s)``. With
        ``Phi = -distance`` and a *stationary* robot that is not zero, it is
        ``(1 - gamma) * distance``, which is **positive**. So with ``gamma < 1``
        a far-away agent earns a small income every step for doing nothing, and
        if the per-step cost does not exceed it, parking is a better policy than
        driving. It is a subtle trap -- the shaping still looks correct, the
        returns still look positive, and the episodes quietly all run to the
        time limit.

        The fix is a per-step cost steeper than the maximum idle income.
        """
        if self.progress <= 0.0:
            return
        idle_income = (1.0 - gamma) * max_distance * self.progress
        per_step_cost = -(self.time + self.action_magnitude)
        if per_step_cost <= idle_income:
            raise ConfigError(
                f"reward.time ({self.time:+g}) does not out-cost potential-based shaping: a "
                f"stationary agent {max_distance:g} m from the goal still earns "
                f"{idle_income:+.4f}/step at gamma={gamma:g}, so idling beats driving. "
                f"Set reward.time steeper than {-idle_income:.4f}, or raise algo.gamma."
            )


@dataclass
class KinematicConfig:
    """Options unique to the fast, ROS-free simulator."""

    obstacles: list[tuple[float, float, float]] = field(default_factory=list)  # (x, y, radius)
    world_bounds: tuple[float, float] = (12.0, 12.0)  # half-extents in x and y
    velocity_noise_std: float = 0.02  # multiplicative actuation noise; keeps sim2sim honest
    observation_noise_std: float = 0.0


@dataclass
class GazeboConfig:
    """Options unique to the ROS 2 / Gazebo backend."""

    world_file: str = "collision_detection.world"
    robot_entity: str = "forklift_bot"
    pallet_entity: str = "pallet"
    pallet_model: str = "pallet/model.sdf"
    spawn_z: float = 0.30
    collision_links: list[str] = field(default_factory=list)
    launch_gui: bool = False
    gazebo_verbose: bool = False
    service_timeout_s: float = 10.0
    settle_steps: int = 2  # physics steps to let the world settle after a teleport


@dataclass
class EnvConfig:
    backend: str = "kinematic"  # "kinematic" | "gazebo"
    max_episode_steps: int = 250
    control_hz: float = 10.0
    observation: list[str] = field(
        default_factory=lambda: [
            "goal_vector_body",
            "goal_distance",
            "heading_error",
            "velocity",
            "last_action",
        ]
    )
    normalize_observations: bool = True
    robot: RobotConfig = field(default_factory=RobotConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    kinematic: KinematicConfig = field(default_factory=KinematicConfig)
    gazebo: GazeboConfig = field(default_factory=GazeboConfig)
    terminate_on_collision: bool = True

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    def validate(self) -> None:
        if self.backend not in ("kinematic", "gazebo"):
            raise ConfigError(f"env.backend must be kinematic|gazebo, got {self.backend!r}")
        if self.max_episode_steps < 1:
            raise ConfigError("env.max_episode_steps must be >= 1")
        if self.control_hz <= 0:
            raise ConfigError("env.control_hz must be > 0")
        if not self.observation:
            raise ConfigError("env.observation must list at least one feature")
        # Import here to avoid a circular import at module load.
        from forklift_gym_env.envs.observations import OBSERVATION_FEATURES

        unknown = [f for f in self.observation if f not in OBSERVATION_FEATURES]
        if unknown:
            raise ConfigError(
                f"env.observation has unknown feature(s) {unknown}. "
                f"Available: {sorted(OBSERVATION_FEATURES)}"
            )
        if len(set(self.observation)) != len(self.observation):
            raise ConfigError("env.observation contains duplicates")
        self.robot.validate()
        self.goal.validate()
        self.reward.validate()


@dataclass
class AlgoConfig:
    name: str = "td3"  # "td3" | "ddpg"
    actor_hidden: list[int] = field(default_factory=lambda: [128, 128])
    critic_hidden: list[int] = field(default_factory=lambda: [128, 128])
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # polyak factor applied to the *online* net
    batch_size: int = 256
    buffer_size: int = 400_000
    exploration_noise: float = 0.2  # stddev, in units of the normalised action
    exploration_noise_final: float = 0.05
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    policy_delay: int = 2  # TD3 delayed actor update; 1 == DDPG
    max_grad_norm: float = 10.0
    use_her: bool = False
    her_ratio: float = 0.8  # fraction of sampled transitions that get relabelled

    def validate(self) -> None:
        if self.name not in ("td3", "ddpg"):
            raise ConfigError(f"algo.name must be td3|ddpg, got {self.name!r}")
        if not 0.0 < self.gamma < 1.0:
            raise ConfigError("algo.gamma must be in (0, 1)")
        if not 0.0 < self.tau <= 1.0:
            raise ConfigError("algo.tau must be in (0, 1]")
        if self.policy_delay < 1:
            raise ConfigError("algo.policy_delay must be >= 1")
        if self.batch_size < 1 or self.buffer_size < self.batch_size:
            raise ConfigError("algo.buffer_size must be >= batch_size >= 1")
        if not 0.0 <= self.her_ratio < 1.0:
            raise ConfigError("algo.her_ratio must be in [0, 1)")


@dataclass
class TrainConfig:
    total_steps: int = 200_000
    warmup_steps: int = 5_000  # uniform-random actions to seed the buffer
    updates_per_step: float = 1.0  # gradient steps per environment step
    update_every: int = 50  # batch the updates to amortise overhead
    eval_every: int = 10_000
    eval_episodes: int = 10
    checkpoint_every: int = 20_000
    log_every: int = 1_000

    def validate(self) -> None:
        if self.total_steps < 1:
            raise ConfigError("train.total_steps must be >= 1")
        if self.warmup_steps < 0:
            raise ConfigError("train.warmup_steps must be >= 0")
        if self.update_every < 1:
            raise ConfigError("train.update_every must be >= 1")
        if self.updates_per_step < 0:
            raise ConfigError("train.updates_per_step must be >= 0")


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self, check_reward_shaping: bool = True) -> Config:
        """Validate every section.

        ``check_reward_shaping=False`` skips only the idling check. It exists so
        ``scripts/reward_shaping_demo.py`` can deliberately construct the broken
        weighting and measure what it costs; nothing else should pass ``False``.
        """
        self.run.validate()
        self.env.validate()
        self.algo.validate()
        self.train.validate()
        if check_reward_shaping:
            self.env.reward.check_idling_is_unprofitable(
                self.algo.gamma, max(self.env.goal.distance_range)
            )
        if self.algo.use_her:
            if not self.env.goal.randomize:
                raise ConfigError("algo.use_her with a fixed goal gives no relabelling signal")
            from forklift_gym_env.envs.observations import OBSERVATION_FEATURES

            dependent = [
                n for n in self.env.observation if not OBSERVATION_FEATURES[n].goal_independent
            ]
            if dependent:
                raise ConfigError(
                    f"algo.use_her requires goal-independent observation features, but "
                    f"{dependent} depend on the goal. Hindsight relabelling rewrites the goal, "
                    f"so those values would no longer match. Use e.g. "
                    f"['robot_pose_world', 'velocity', 'last_action'] instead."
                )
        return self

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _build(cls: type[T], data: Mapping[str, Any], path: str = "") -> T:
    """Recursively build a dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        where = f"{path}." if path else ""
        raise ConfigError(
            f"unknown config key(s) {[where + u for u in unknown]}. "
            f"Valid keys here: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        f = known[name]
        child_path = f"{path}.{name}" if path else name
        if is_dataclass(f.type) and isinstance(f.type, type):
            kwargs[name] = _build(f.type, value, child_path)
        elif isinstance(value, Mapping) and _default_is_dataclass(cls, name):
            kwargs[name] = _build(type(_default_of(cls, name)), value, child_path)
        elif isinstance(value, list) and _default_is_tuple(cls, name):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _default_of(cls: type, name: str) -> Any:
    f = next(f for f in fields(cls) if f.name == name)
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return None


def _default_is_dataclass(cls: type, name: str) -> bool:
    return is_dataclass(_default_of(cls, name))


def _default_is_tuple(cls: type, name: str) -> bool:
    return isinstance(_default_of(cls, name), tuple)


def load_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
    check_reward_shaping: bool = True,
) -> Config:
    """Load and validate a YAML config.

    ``overrides`` is a flat mapping of dotted keys, e.g. ``{"train.total_steps": 1000}``,
    which is what the CLI's ``--set`` flag produces.

    ``check_reward_shaping=False`` skips only the idling check; see
    :meth:`Config.validate`.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if overrides:
        raw = _apply_overrides(raw, overrides)
    cfg = _build(Config, raw)
    return cfg.validate(check_reward_shaping=check_reward_shaping)


def _apply_overrides(raw: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    for dotted, value in overrides.items():
        node = out
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {dotted!r} conflicts with a scalar value")
        node[leaf] = value
    return out


def save_config(cfg: Config, path: str | Path) -> None:
    """Snapshot the fully-resolved config next to a run's outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
