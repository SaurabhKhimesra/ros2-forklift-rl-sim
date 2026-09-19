"""Observation features, assembled from config.

Previously the observation space was hard-coded to ``Box(shape=(2,))`` while the
config advertised three feature groups, and most of the concatenation lines were
commented out -- so what the agent actually saw had nothing to do with the YAML.
Here each feature declares its own width and bounds, the vector is built by
concatenating the features the config asks for, and the ``observation_space`` is
derived from that same list. The two cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from forklift_gym_env.envs.backends.base import WorldState

#: Scale factors that bring each feature roughly into ``[-1, 1]``. Unnormalised
#: inputs spanning very different magnitudes are a classic reason an actor's
#: ``tanh`` head saturates and stops producing gradients.
_DISTANCE_SCALE = 10.0
_VELOCITY_SCALE = 1.5


@dataclass(frozen=True, slots=True)
class Feature:
    """One named block of the observation vector."""

    name: str
    size: int
    fn: Callable[[WorldState, np.ndarray, bool], np.ndarray]
    low: float = -np.inf
    high: float = np.inf
    #: ``False`` if the feature's value depends on where the goal is. Hindsight
    #: relabelling rewrites the goal after the fact, so a goal-dependent feature
    #: baked into the observation would become a lie. Validated at config time.
    goal_independent: bool = True

    def __call__(self, state: WorldState, last_action: np.ndarray, normalize: bool) -> np.ndarray:
        value = np.asarray(self.fn(state, last_action, normalize), dtype=np.float32).reshape(-1)
        if value.size != self.size:
            raise RuntimeError(
                f"observation feature {self.name!r} produced {value.size} values, "
                f"expected {self.size}"
            )
        return value


def _goal_vector_body(state: WorldState, _a: np.ndarray, normalize: bool) -> np.ndarray:
    vec = state.goal_vector_body
    return vec / _DISTANCE_SCALE if normalize else vec


def _goal_distance(state: WorldState, _a: np.ndarray, normalize: bool) -> np.ndarray:
    d = state.goal_distance
    return np.array([d / _DISTANCE_SCALE if normalize else d])


def _heading_error(state: WorldState, _a: np.ndarray, normalize: bool) -> np.ndarray:
    """Heading error as ``(sin, cos)``.

    Feeding the raw angle -- as the original did -- hands the network a
    discontinuity at +/-pi. ``(sin, cos)`` is continuous everywhere and is the
    standard fix.
    """
    err = state.heading_error
    del normalize  # already in [-1, 1]
    return np.array([np.sin(err), np.cos(err)])


def _alignment_error(state: WorldState, _a: np.ndarray, normalize: bool) -> np.ndarray:
    """Yaw offset from the pallet's own orientation, as ``(sin, cos)``."""
    del normalize
    err = state.alignment_error
    return np.array([np.sin(err), np.cos(err)])


def _velocity(state: WorldState, _a: np.ndarray, normalize: bool) -> np.ndarray:
    v = np.array([state.linear_velocity, state.angular_velocity])
    return v / _VELOCITY_SCALE if normalize else v


def _last_action(_s: WorldState, last_action: np.ndarray, _n: bool) -> np.ndarray:
    return np.asarray(last_action, dtype=np.float32)


def _goal_pose_world(state: WorldState, _a: np.ndarray, normalize: bool) -> np.ndarray:
    xy = state.goal.xy
    return xy / _DISTANCE_SCALE if normalize else xy


def _robot_pose_world(state: WorldState, _a: np.ndarray, normalize: bool) -> np.ndarray:
    p = state.robot
    xy = p.xy / _DISTANCE_SCALE if normalize else p.xy
    return np.concatenate([xy, [np.sin(p.yaw), np.cos(p.yaw)]])


def _collision_flag(state: WorldState, _a: np.ndarray, _n: bool) -> np.ndarray:
    return np.array([1.0 if state.in_collision else 0.0])


#: Registry of available features. ``env.observation`` in the YAML must name
#: entries from here; anything else is rejected during config validation, which
#: is how typos get caught before a simulator is launched.
OBSERVATION_FEATURES: Mapping[str, Feature] = {
    "goal_vector_body": Feature("goal_vector_body", 2, _goal_vector_body, -5.0, 5.0, False),
    "goal_distance": Feature("goal_distance", 1, _goal_distance, 0.0, 5.0, False),
    "heading_error": Feature("heading_error", 2, _heading_error, -1.0, 1.0, False),
    "alignment_error": Feature("alignment_error", 2, _alignment_error, -1.0, 1.0, False),
    "velocity": Feature("velocity", 2, _velocity, -5.0, 5.0),
    "last_action": Feature("last_action", 2, _last_action, -1.0, 1.0),
    "goal_pose_world": Feature("goal_pose_world", 2, _goal_pose_world, -5.0, 5.0, False),
    "robot_pose_world": Feature("robot_pose_world", 4, _robot_pose_world, -5.0, 5.0),
    "collision_flag": Feature("collision_flag", 1, _collision_flag, 0.0, 1.0),
}


class ObservationBuilder:
    """Turns a :class:`WorldState` into the flat vector the policy consumes."""

    def __init__(self, feature_names: list[str], normalize: bool = True) -> None:
        self.features = [OBSERVATION_FEATURES[name] for name in feature_names]
        self.normalize = normalize
        self.size = sum(f.size for f in self.features)

    @property
    def low(self) -> np.ndarray:
        return np.concatenate([np.full(f.size, f.low, np.float32) for f in self.features])

    @property
    def high(self) -> np.ndarray:
        return np.concatenate([np.full(f.size, f.high, np.float32) for f in self.features])

    def layout(self) -> dict[str, slice]:
        """Map each feature name to its slice, for debugging and plots."""
        out, start = {}, 0
        for f in self.features:
            out[f.name] = slice(start, start + f.size)
            start += f.size
        return out

    def __call__(self, state: WorldState, last_action: np.ndarray) -> np.ndarray:
        parts = [f(state, last_action, self.normalize) for f in self.features]
        return np.concatenate(parts).astype(np.float32)


def achieved_goal(state: WorldState) -> np.ndarray:
    """The goal-space projection of the current state, used by HER."""
    return state.robot.xy.astype(np.float32)


def desired_goal(state: WorldState) -> np.ndarray:
    return state.goal.xy.astype(np.float32)
