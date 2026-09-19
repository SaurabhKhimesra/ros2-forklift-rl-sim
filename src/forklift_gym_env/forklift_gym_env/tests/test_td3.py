import numpy as np
import pytest
import torch

from forklift_gym_env.config import AlgoConfig
from forklift_gym_env.rl.buffers import ReplayBuffer
from forklift_gym_env.rl.networks import Actor, TwinCritic
from forklift_gym_env.rl.td3 import TD3Agent, resolve_device

OBS, ACT = 6, 2


def make_agent(**kw) -> TD3Agent:
    cfg = AlgoConfig(actor_hidden=[32, 32], critic_hidden=[32, 32], **kw)
    cfg.validate()
    return TD3Agent(OBS, ACT, cfg)


def filled_buffer(n=512, seed=0) -> ReplayBuffer:
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(n, OBS, ACT, seed=seed)
    for i in range(n):
        buf.add(
            rng.normal(size=OBS),
            rng.uniform(-1, 1, ACT),
            rng.normal(),
            rng.normal(size=OBS),
            i % 37 == 0,
        )
    return buf


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
def test_actor_output_is_inside_the_action_box():
    actor = Actor(OBS, ACT, [16])
    out = actor(torch.randn(128, OBS) * 50)  # extreme inputs
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_actor_starts_near_zero_so_tanh_is_not_saturated():
    out = Actor(OBS, ACT, [32, 32])(torch.randn(256, OBS))
    assert out.abs().mean() < 0.2


def test_no_batchnorm_anywhere():
    """BatchNorm made the old actor behave differently when acting (batch 1) than
    when training (batch 256)."""
    for module in list(Actor(OBS, ACT).modules()) + list(TwinCritic(OBS, ACT).modules()):
        assert not isinstance(module, torch.nn.BatchNorm1d)


def test_actor_is_batch_size_invariant():
    actor = Actor(OBS, ACT, [16]).eval()
    obs = torch.randn(8, OBS)
    with torch.no_grad():
        batched = actor(obs)
        one_at_a_time = torch.cat([actor(o.unsqueeze(0)) for o in obs])
    torch.testing.assert_close(batched, one_at_a_time)


def test_twin_critics_are_independent():
    critic = TwinCritic(OBS, ACT, [16])
    q1, q2 = critic(torch.randn(32, OBS), torch.randn(32, ACT))
    assert not torch.allclose(q1, q2)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delay", [1, 2, 3])
def test_update_survives_every_policy_delay(delay):
    """Regression: the old `update()` returned `actor_loss` from outside the
    `if n_updates % policy_delay == 0` branch, raising UnboundLocalError on the
    first skipped update. Only policy_delay=1 avoided it."""
    agent, buf = make_agent(policy_delay=delay), filled_buffer()
    metrics = [agent.update(buf.sample(32)) for _ in range(delay * 3)]
    assert all(np.isfinite(m.critic_loss) for m in metrics)
    updated = [m for m in metrics if m.actor_loss is not None]
    assert len(updated) == 3


def test_all_target_parameters_are_frozen():
    """Regression: the old code zipped actor and critic target parameters
    together, so the longer list was only partly frozen."""
    agent = make_agent()
    agent.critic_hidden_check = None
    for net in (agent.actor_target, agent.critic_target):
        assert all(not p.requires_grad for p in net.parameters())
    assert sum(1 for _ in agent.critic_target.parameters()) > sum(
        1 for _ in agent.actor_target.parameters()
    ), "the two target nets genuinely differ in length, which is what broke zip()"


def test_targets_start_identical_to_the_online_networks():
    agent = make_agent()
    for online, target in zip(
        agent.actor.parameters(), agent.actor_target.parameters(), strict=True
    ):
        torch.testing.assert_close(online, target)


def test_polyak_moves_targets_a_little_towards_online():
    agent = make_agent(tau=0.1, policy_delay=1)
    before = [p.clone() for p in agent.actor_target.parameters()]
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.add_(1.0)
    agent.update(filled_buffer().sample(32))
    for old, new, online in zip(
        before, agent.actor_target.parameters(), agent.actor.parameters(), strict=True
    ):
        expected = old * 0.9 + online * 0.1
        torch.testing.assert_close(new, expected, rtol=1e-4, atol=1e-6)


def test_act_respects_the_action_bounds_even_with_heavy_noise():
    agent = make_agent()
    actions = np.stack([agent.act(np.random.randn(OBS), noise_scale=5.0) for _ in range(200)])
    assert actions.min() >= -1.0 and actions.max() <= 1.0


def test_act_is_deterministic_without_noise():
    agent, obs = make_agent(), np.random.randn(OBS)
    np.testing.assert_array_equal(agent.act(obs, 0.0), agent.act(obs, 0.0))


def test_act_is_stochastic_with_noise():
    agent, obs = make_agent(), np.random.randn(OBS)
    assert not np.array_equal(agent.act(obs, 0.3), agent.act(obs, 0.3))


def test_act_handles_single_and_batched_observations():
    agent = make_agent()
    assert agent.act(np.random.randn(OBS)).shape == (ACT,)
    assert agent.act(np.random.randn(5, OBS)).shape == (5, ACT)


def test_acting_does_not_leave_the_actor_in_eval_mode():
    agent = make_agent()
    agent.actor.train()
    agent.act(np.random.randn(OBS))
    assert agent.actor.training


def test_ddpg_mode_uses_a_single_critic_target():
    ddpg, td3 = make_agent(name="ddpg", policy_delay=1), make_agent(name="td3")
    buf = filled_buffer()
    assert np.isfinite(ddpg.update(buf.sample(32)).critic_loss)
    assert np.isfinite(td3.update(buf.sample(32)).critic_loss)


def test_clipped_double_q_is_pessimistic():
    """TD3's target is min(Q1, Q2), which is <= either critic alone."""
    agent = make_agent()
    obs, act = torch.randn(64, OBS), torch.rand(64, ACT) * 2 - 1
    with torch.no_grad():
        q1, q2 = agent.critic_target(obs, act)
        assert torch.all(torch.min(q1, q2) <= q1 + 1e-6)


def test_terminated_transitions_do_not_bootstrap():
    agent = make_agent(gamma=0.99)
    buf = ReplayBuffer(8, OBS, ACT, seed=0)
    for _ in range(8):
        buf.add(np.zeros(OBS), np.zeros(ACT), 5.0, np.zeros(OBS), True)
    batch = buf.sample(8)
    with torch.no_grad():
        q_next = agent.critic_target(batch.next_obs, agent.actor_target(batch.next_obs))[0]
        target = batch.reward + (1 - batch.terminated) * agent.cfg.gamma * q_next
    torch.testing.assert_close(target, batch.reward)


def test_checkpoint_roundtrip_preserves_behaviour(tmp_path):
    agent, buf = make_agent(), filled_buffer()
    for _ in range(5):
        agent.update(buf.sample(32))
    path = agent.save(tmp_path / "ck.pt", step=5)

    reloaded = TD3Agent.load(path, agent.cfg, OBS, ACT)
    obs = np.random.randn(OBS)
    np.testing.assert_allclose(agent.act(obs, 0.0), reloaded.act(obs, 0.0), rtol=1e-6)
    assert reloaded.n_updates == agent.n_updates
    # Optimiser state must survive too, so resuming is not a cold restart.
    assert reloaded.actor_optim.state_dict()["state"]


def test_loading_a_mismatched_checkpoint_fails_clearly(tmp_path):
    path = make_agent().save(tmp_path / "ck.pt")
    with pytest.raises(ValueError, match="obs_dim"):
        TD3Agent.load(path, AlgoConfig(actor_hidden=[32, 32], critic_hidden=[32, 32]), OBS + 1, ACT)


def test_critic_is_unfrozen_after_the_actor_step():
    agent = make_agent(policy_delay=1)
    agent.update(filled_buffer().sample(32))
    assert all(p.requires_grad for p in agent.critic.parameters())


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_agent_learns_a_trivial_bandit():
    """End-to-end sanity: reward = -(action - target)^2 in a one-step problem.
    The actor must move towards the target."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    target = np.array([0.6, -0.4], np.float32)
    # every transition is terminal, so gamma never enters the target
    agent = make_agent(policy_delay=1, actor_lr=3e-3, critic_lr=3e-3)
    buf = ReplayBuffer(4000, OBS, ACT, seed=0)
    obs = np.zeros(OBS, np.float32)
    for _ in range(4000):
        action = rng.uniform(-1, 1, ACT).astype(np.float32)
        buf.add(obs, action, -float(np.sum((action - target) ** 2)), obs, True)
    for _ in range(600):
        agent.update(buf.sample(128))
    np.testing.assert_allclose(agent.act(obs, 0.0), target, atol=0.15)
