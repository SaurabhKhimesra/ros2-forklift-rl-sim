"""Actor and critic networks.

Two deliberate departures from the previous implementation:

* **No BatchNorm.** The old actor and critic both ended with a ``BatchNorm1d``
  just before the output layer. In an off-policy actor-critic that is actively
  harmful: the actor is evaluated with batch size 1 when acting and batch size
  256 when training, so the normalisation statistics differ between the two, and
  the critic's running statistics drift as the replay distribution shifts, which
  makes the Q-target non-stationary on top of the usual bootstrapping. Where
  normalisation helps, ``LayerNorm`` is the right choice -- it is batch-size
  independent -- and it is available behind a flag.

* **Twin critics.** TD3's clipped double-Q is the cheapest available fix for
  DDPG's well-known overestimation bias, and it is a single extra MLP.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
    layer_norm: bool = False,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers.append(nn.Linear(prev, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic policy mapping observations to actions in ``[-1, 1]``."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (256, 256),
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.net = mlp(
            obs_dim, hidden, action_dim, layer_norm=layer_norm, output_activation=nn.Tanh
        )
        # Small final-layer weights keep the initial policy near zero, so the
        # tanh starts in its linear region instead of saturated at the limits.
        final = self.net[-2]
        assert isinstance(final, nn.Linear)
        nn.init.uniform_(final.weight, -3e-3, 3e-3)
        nn.init.uniform_(final.bias, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TwinCritic(nn.Module):
    """Two independent Q networks evaluated on the same ``(s, a)`` pair."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (256, 256),
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.q1 = mlp(obs_dim + action_dim, hidden, 1, layer_norm=layer_norm)
        self.q2 = mlp(obs_dim + action_dim, hidden, 1, layer_norm=layer_norm)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Used for the actor loss; evaluating both critics there is wasted work."""
        return self.q1(torch.cat([obs, action], dim=-1))
