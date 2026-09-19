"""TD3 (and DDPG, as the ``policy_delay == 1`` single-critic special case).

Bugs in the previous DDPG implementation that this replaces
------------------------------------------------------------
* ``update()`` ended with ``return critic_loss, actor_loss`` while ``actor_loss``
  was only assigned inside ``if self._n_updates % self.policy_update_delay == 0``.
  Every configuration with a delay above 1 -- that is, TD3 as actually described
  in the README -- raised ``UnboundLocalError`` on its second update. The shipped
  config set the delay to 1, which hid it.
* Target networks were frozen with
  ``for a, c in zip(actor_target.parameters(), critic_target.parameters())``.
  ``zip`` stops at the shorter sequence, so whenever the actor and critic had
  different layer counts part of the critic target stayed trainable.
* Polyak averaging ran only inside the delayed branch, coupling the target
  update rate to the policy delay.
* ``save_model`` wrote four ``.pkl`` files into the current working directory
  with fixed names, saved no optimiser state and no step counter, and
  ``torch.load`` was called without ``map_location``, so a GPU checkpoint could
  not be loaded on a CPU machine.
* Everything ran on CPU: there was no device handling anywhere.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from forklift_gym_env.config import AlgoConfig
from forklift_gym_env.rl.buffers import Batch
from forklift_gym_env.rl.networks import Actor, TwinCritic


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@dataclass
class UpdateMetrics:
    """Always fully populated -- ``actor_loss`` is ``None`` on a skipped update."""

    critic_loss: float
    actor_loss: float | None
    q_mean: float
    target_q_mean: float
    n_updates: int


class TD3Agent:
    """Twin Delayed DDPG.

    Setting ``policy_delay = 1`` and ``target_policy_noise = 0`` recovers DDPG,
    so the two algorithms share one tested code path instead of two.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: AlgoConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)

        self.actor = Actor(obs_dim, action_dim, cfg.actor_hidden).to(self.device)
        self.critic = TwinCritic(obs_dim, action_dim, cfg.critic_hidden).to(self.device)

        # deepcopy guarantees the targets start identical to the online nets,
        # including buffers, and freezing every parameter explicitly avoids the
        # zip() truncation bug in the original.
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_target.eval()
        self.critic_target.eval()

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.n_updates = 0
        self._ddpg = cfg.name == "ddpg"

    # ------------------------------------------------------------------ #
    # acting
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """Return an action in ``[-1, 1]``.

        ``noise_scale`` is the stddev of Gaussian exploration noise in action
        units. Pass ``0.0`` for deterministic evaluation -- the original agent
        flipped a ``self.mode`` string and toggled ``train()``/``eval()`` on the
        modules to achieve the same thing, which quietly changed BatchNorm
        behaviour at the same time.
        """
        was_training = self.actor.training
        self.actor.eval()
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        single = tensor.ndim == 1
        if single:
            tensor = tensor.unsqueeze(0)
        action = self.actor(tensor)
        if noise_scale > 0.0:
            action = action + noise_scale * torch.randn_like(action)
        action = action.clamp(-1.0, 1.0)
        if was_training:
            self.actor.train()
        result = action.cpu().numpy()
        return result[0] if single else result

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #
    def update(self, batch: Batch) -> UpdateMetrics:
        self.actor.train()
        self.critic.train()

        with torch.no_grad():
            next_action = self.actor_target(batch.next_obs)
            if not self._ddpg and self.cfg.target_policy_noise > 0.0:
                # Target policy smoothing: noise on the *target* action makes the
                # critic regress a locally averaged value, which stops the actor
                # from exploiting sharp peaks in an imperfect Q function.
                noise = (torch.randn_like(next_action) * self.cfg.target_policy_noise).clamp(
                    -self.cfg.target_noise_clip, self.cfg.target_noise_clip
                )
                next_action = (next_action + noise).clamp(-1.0, 1.0)

            q1_next, q2_next = self.critic_target(batch.next_obs, next_action)
            # Clipped double-Q: the pessimistic of the two targets. This is the
            # single change that most reduces DDPG's overestimation bias.
            q_next = q1_next if self._ddpg else torch.min(q1_next, q2_next)
            target_q = batch.reward + (1.0 - batch.terminated) * self.cfg.gamma * q_next

        q1, q2 = self.critic(batch.obs, batch.action)
        critic_loss = nn.functional.mse_loss(q1, target_q)
        if not self._ddpg:
            critic_loss = critic_loss + nn.functional.mse_loss(q2, target_q)

        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_optim.step()

        actor_loss_value: float | None = None
        if self.n_updates % self.cfg.policy_delay == 0:
            # Freeze the critic for the policy step so no gradient is computed
            # for weights we are not about to update.
            self.critic.requires_grad_(False)
            actor_loss = -self.critic.q1_only(batch.obs, self.actor(batch.obs)).mean()
            self.actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.cfg.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.actor_optim.step()
            self.critic.requires_grad_(True)
            actor_loss_value = float(actor_loss.detach())

            # Targets track the online nets on the *policy* clock, which is the
            # convention TD3 uses.
            self._polyak(self.actor, self.actor_target)
            self._polyak(self.critic, self.critic_target)

        self.n_updates += 1
        return UpdateMetrics(
            critic_loss=float(critic_loss.detach()),
            actor_loss=actor_loss_value,
            q_mean=float(q1.mean().detach()),
            target_q_mean=float(target_q.mean()),
            n_updates=self.n_updates,
        )

    @torch.no_grad()
    def _polyak(self, online: nn.Module, target: nn.Module) -> None:
        tau = self.cfg.tau
        for p_online, p_target in zip(online.parameters(), target.parameters(), strict=True):
            p_target.mul_(1.0 - tau).add_(p_online, alpha=tau)
        # Copy non-parameter buffers (e.g. LayerNorm has none, but a future
        # normalisation layer might) so the target is a faithful lagged copy.
        for b_online, b_target in zip(online.buffers(), target.buffers(), strict=True):
            b_target.copy_(b_online)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "n_updates": self.n_updates,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("obs_dim") not in (None, self.obs_dim):
            raise ValueError(
                f"checkpoint was trained with obs_dim={state['obs_dim']}, "
                f"this agent expects {self.obs_dim}. The env.observation list has changed."
            )
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        self.n_updates = int(state.get("n_updates", 0))

    def save(self, path: str | Path, **extra: Any) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**self.state_dict(), **extra}, path)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        cfg: AlgoConfig,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str = "cpu",
    ) -> TD3Agent:
        # map_location is what makes a GPU-trained checkpoint loadable on a
        # laptop; its absence in the original made checkpoints machine-specific.
        state = torch.load(Path(path), map_location=torch.device(device), weights_only=False)
        agent = cls(obs_dim, action_dim, cfg, device)
        agent.load_state_dict(state)
        return agent
