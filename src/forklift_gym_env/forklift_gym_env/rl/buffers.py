"""Replay buffers.

A plain uniform buffer, and a ``future``-strategy Hindsight Experience Replay
buffer that stores episodes whole, samples them length-weighted, and recomputes
rewards and terminals against the relabelled goal. The tests construct episodes
that never reach their goal and assert the HER buffer manufactures successes
where the uniform one yields none -- worth pinning down, because a buffer that
relabels nothing still trains, just badly.

Both store ``float32`` numpy arrays and convert to torch only at sample time,
which keeps a 400k-transition buffer at a few tens of megabytes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch


@dataclass(slots=True)
class Batch:
    """One minibatch, already on the training device."""

    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor  # 1.0 where the episode really ended (not truncated)

    def __len__(self) -> int:
        return int(self.obs.shape[0])


class SupportsRelabel(Protocol):
    """The slice of the environment API that HER needs."""

    def compute_reward(
        self, achieved: np.ndarray, desired: np.ndarray, info: object
    ) -> np.ndarray: ...
    def is_success(self, achieved: np.ndarray, desired: np.ndarray) -> np.ndarray: ...


class ReplayBuffer:
    """Fixed-capacity uniform replay buffer backed by preallocated arrays."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        self._obs = np.zeros((capacity, obs_dim), np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), np.float32)
        self._action = np.zeros((capacity, action_dim), np.float32)
        self._reward = np.zeros((capacity, 1), np.float32)
        self._terminated = np.zeros((capacity, 1), np.float32)

        self._index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        i = self._index
        self._obs[i] = obs
        self._action[i] = action
        self._reward[i] = reward
        self._next_obs[i] = next_obs
        # Only *termination* stops the bootstrap. A time-limit truncation must
        # still bootstrap through the value function, otherwise the agent learns
        # that the world ends after `max_episode_steps` and becomes myopic.
        self._terminated[i] = float(terminated)
        self._index = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def sample(self, batch_size: int) -> Batch:
        if not self.can_sample(batch_size):
            raise ValueError(f"buffer holds {self._size} transitions, need {batch_size}")
        idx = self.rng.integers(0, self._size, size=batch_size)
        to = self._to_torch
        return Batch(
            obs=to(self._obs[idx]),
            action=to(self._action[idx]),
            reward=to(self._reward[idx]),
            next_obs=to(self._next_obs[idx]),
            terminated=to(self._terminated[idx]),
        )

    def _to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)


class HERReplayBuffer:
    """Hindsight Experience Replay with the ``future`` goal-selection strategy.

    Storage is episode-major -- ``[n_episodes, max_episode_len, dim]`` -- because
    relabelling needs to look forward inside the episode a transition came from,
    which a flat ring buffer cannot do once it wraps.

    The idea (Andrychowicz et al., 2017): a failed episode still demonstrates how
    to reach *whatever the robot actually reached*. Replacing the goal with a
    state visited later in the same episode turns that failure into a success,
    which is what makes a sparse success-only reward learnable.
    """

    def __init__(
        self,
        capacity_transitions: int,
        max_episode_len: int,
        obs_dim: int,
        action_dim: int,
        goal_dim: int,
        env: SupportsRelabel,
        her_ratio: float = 0.8,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= her_ratio < 1.0:
            raise ValueError("her_ratio must be in [0, 1)")
        self.max_episode_len = int(max_episode_len)
        self.n_episodes = max(1, int(capacity_transitions) // self.max_episode_len)
        self.her_ratio = float(her_ratio)
        self.env = env
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        shape = (self.n_episodes, self.max_episode_len)
        self._obs = np.zeros((*shape, obs_dim), np.float32)
        self._next_obs = np.zeros((*shape, obs_dim), np.float32)
        self._action = np.zeros((*shape, action_dim), np.float32)
        self._desired = np.zeros((*shape, goal_dim), np.float32)
        self._achieved_next = np.zeros((*shape, goal_dim), np.float32)
        self._lengths = np.zeros(self.n_episodes, np.int64)

        self._ep_index = 0
        self._n_stored = 0
        self._current: list[tuple[np.ndarray, ...]] = []

    def __len__(self) -> int:
        return int(self._lengths[: self._n_stored].sum())

    # -- writing --------------------------------------------------------- #
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        desired: np.ndarray,
        achieved_next: np.ndarray,
    ) -> None:
        """Buffer one transition of the episode in progress."""
        if len(self._current) >= self.max_episode_len:
            raise RuntimeError(
                f"episode exceeded max_episode_len={self.max_episode_len}; "
                "HER storage is episode-major and cannot grow"
            )
        self._current.append((obs, action, next_obs, desired, achieved_next))

    def end_episode(self) -> None:
        """Commit the episode in progress. Safe to call on an empty episode."""
        if not self._current:
            return
        e, n = self._ep_index, len(self._current)
        for t, (obs, action, next_obs, desired, achieved_next) in enumerate(self._current):
            self._obs[e, t] = obs
            self._action[e, t] = action
            self._next_obs[e, t] = next_obs
            self._desired[e, t] = desired
            self._achieved_next[e, t] = achieved_next
        self._lengths[e] = n
        self._ep_index = (e + 1) % self.n_episodes
        self._n_stored = min(self._n_stored + 1, self.n_episodes)
        self._current = []

    def can_sample(self, batch_size: int) -> bool:
        return self._n_stored > 0 and len(self) >= batch_size

    # -- reading --------------------------------------------------------- #
    def sample(self, batch_size: int) -> Batch:
        if not self.can_sample(batch_size):
            raise ValueError(f"HER buffer holds {len(self)} transitions, need {batch_size}")

        lengths = self._lengths[: self._n_stored]
        # Sample episodes in proportion to their length, so every stored
        # transition is equally likely -- uniform over *episodes* would
        # over-sample short (usually successful) ones and bias the buffer.
        probs = lengths / lengths.sum()
        eps = self.rng.choice(self._n_stored, size=batch_size, p=probs)
        ts = (self.rng.random(batch_size) * lengths[eps]).astype(np.int64)

        desired = self._desired[eps, ts].copy()
        achieved_next = self._achieved_next[eps, ts]

        n_relabel = round(self.her_ratio * batch_size)
        if n_relabel:
            which = self.rng.permutation(batch_size)[:n_relabel]
            # "future": a timestep at or after t, within the same episode.
            offsets = self.rng.random(n_relabel) * (lengths[eps[which]] - ts[which])
            future_t = ts[which] + offsets.astype(np.int64)
            desired[which] = self._achieved_next[eps[which], future_t]

        reward = np.asarray(self.env.compute_reward(achieved_next, desired, None), np.float32)
        terminated = np.asarray(self.env.is_success(achieved_next, desired), np.float32)

        # The policy is goal-conditioned: the goal is part of the network input.
        obs = np.concatenate([self._obs[eps, ts], desired], axis=-1)
        next_obs = np.concatenate([self._next_obs[eps, ts], desired], axis=-1)

        to = self._to_torch
        return Batch(
            obs=to(obs),
            action=to(self._action[eps, ts]),
            reward=to(reward.reshape(-1, 1)),
            next_obs=to(next_obs),
            terminated=to(terminated.reshape(-1, 1)),
        )

    def _to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=self.device)


BufferLike = ReplayBuffer | HERReplayBuffer
RelabelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
