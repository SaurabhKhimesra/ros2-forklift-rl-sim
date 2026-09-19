"""Composable reward terms.

The original reward function computed an angular cost and an action penalty and
then discarded both, returning ``-0.1 * l2_distance`` from a dead-code path. A
pure negative-distance reward is also a poor learning signal: its magnitude
depends on where the goal happens to be, so the return of a *good* episode with
a far goal can be worse than a *bad* episode with a near one.

This module replaces it with weighted, individually testable terms. The two
shaping terms use **potential-based shaping** (Ng, Harada & Russell, 1999):

    F(s, s') = gamma * Phi(s') - Phi(s)

which is the only general form of shaping guaranteed not to change the optimal
policy. Every term returns a plain float and is a pure function of its context,
so each one has a unit test.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from forklift_gym_env.envs.backends.base import WorldState


@dataclass(frozen=True, slots=True)
class RewardContext:
    """Everything a reward term is allowed to look at."""

    prev_state: WorldState
    state: WorldState
    action: np.ndarray  # normalised, in [-1, 1]
    prev_action: np.ndarray
    success: bool
    collision: bool
    gamma: float
    dt: float


RewardTerm = Callable[[RewardContext], float]


def progress(ctx: RewardContext) -> float:
    """Potential-based shaping on distance to the goal.

    Positive when the robot closes distance, negative when it backs away, and
    scale-free: the total shaping contribution over an episode telescopes to
    roughly ``d_start - d_end`` regardless of how far the goal started.
    """
    phi_prev = -ctx.prev_state.goal_distance
    phi_now = -ctx.state.goal_distance
    return ctx.gamma * phi_now - phi_prev


def heading(ctx: RewardContext) -> float:
    """Potential-based shaping on absolute heading error.

    Normalised by pi so its magnitude is comparable to :func:`progress`, and
    faded out near the goal where heading stops being meaningful (standing on
    the pallet, any heading is fine unless ``require_heading`` is set).
    """

    def phi(state: WorldState) -> float:
        fade = min(1.0, state.goal_distance)  # no heading pressure inside 1 m
        return -abs(state.heading_error) / math.pi * fade

    return ctx.gamma * phi(ctx.state) - phi(ctx.prev_state)


def alignment(ctx: RewardContext) -> float:
    """Potential-based shaping on the yaw offset from the pallet.

    Gated by proximity: pointing the right way is irrelevant from ten metres
    out, and rewarding it there would teach the robot to spin on the spot
    instead of driving. The gate ramps in over the last two metres.
    """

    def phi(state: WorldState) -> float:
        gate = max(0.0, 1.0 - state.goal_distance / 2.0)
        return -abs(state.alignment_error) / math.pi * gate

    return ctx.gamma * phi(ctx.state) - phi(ctx.prev_state)


def success(ctx: RewardContext) -> float:
    """One-off bonus, paid on the transition that reaches the goal."""
    return 1.0 if ctx.success else 0.0


def collision(ctx: RewardContext) -> float:
    """One-off penalty (the weight itself is negative)."""
    return 1.0 if ctx.collision else 0.0


def time(ctx: RewardContext) -> float:
    """Constant per-step cost; the weight is negative so dawdling is punished."""
    del ctx
    return 1.0


def action_magnitude(ctx: RewardContext) -> float:
    """Mean squared command, to discourage flooring it everywhere."""
    a = ctx.action
    return float(a @ a) / a.size


def action_smoothness(ctx: RewardContext) -> float:
    """Mean squared *change* in command.

    Without this, a velocity-controlled policy happily oscillates between the
    action limits -- fine for the reward, terrible for a real drivetrain.
    """
    d = ctx.action - ctx.prev_action
    return float(d @ d) / d.size


def reverse(ctx: RewardContext) -> float:
    """How much of the command was reverse motion, in ``[0, 1]``."""
    return float(max(0.0, -ctx.action[0]))


#: Registry consulted by :class:`RewardFunction`. Keys must match the field
#: names of :class:`~forklift_gym_env.config.RewardConfig`.
REWARD_TERMS: Mapping[str, RewardTerm] = {
    "progress": progress,
    "heading": heading,
    "alignment": alignment,
    "success": success,
    "collision": collision,
    "time": time,
    "action_magnitude": action_magnitude,
    "action_smoothness": action_smoothness,
    "reverse": reverse,
}


class RewardFunction:
    """Weighted sum of the active terms, with per-term bookkeeping.

    The per-term breakdown is logged to TensorBoard, which turns "the reward is
    going down and I don't know why" into a two-second glance at a chart.
    """

    def __init__(self, weights: Mapping[str, float]) -> None:
        unknown = sorted(set(weights) - set(REWARD_TERMS))
        if unknown:
            raise ValueError(f"unknown reward term(s): {unknown}")
        self.weights = {k: float(v) for k, v in weights.items() if v != 0.0}
        self.terms = {k: REWARD_TERMS[k] for k in self.weights}

    def __call__(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        breakdown = {name: self.weights[name] * fn(ctx) for name, fn in self.terms.items()}
        return float(sum(breakdown.values())), breakdown

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        parts = ", ".join(f"{k}={v:g}" for k, v in self.weights.items())
        return f"RewardFunction({parts})"
