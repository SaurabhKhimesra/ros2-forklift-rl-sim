import numpy as np
import pytest
import torch

from forklift_gym_env.rl.buffers import HERReplayBuffer, ReplayBuffer


@pytest.fixture
def buffer():
    return ReplayBuffer(capacity=64, obs_dim=3, action_dim=2, seed=0)


def test_length_tracks_additions(buffer):
    assert len(buffer) == 0
    for i in range(10):
        buffer.add(np.full(3, i), np.zeros(2), 1.0, np.full(3, i + 1), False)
    assert len(buffer) == 10 and not buffer.is_full


def test_ring_overwrites_and_caps_at_capacity(buffer):
    for i in range(200):
        buffer.add(np.full(3, i), np.zeros(2), float(i), np.zeros(3), False)
    assert len(buffer) == 64 and buffer.is_full
    # The newest transition must still be present after wrapping.
    assert buffer._obs.max() == 199


def test_sampling_before_enough_data_raises(buffer):
    buffer.add(np.zeros(3), np.zeros(2), 0.0, np.zeros(3), False)
    assert not buffer.can_sample(8)
    with pytest.raises(ValueError, match="need 8"):
        buffer.sample(8)


def test_batch_shapes_and_dtypes(buffer):
    for i in range(40):
        buffer.add(np.random.randn(3), np.random.randn(2), 1.0, np.random.randn(3), i % 5 == 0)
    batch = buffer.sample(16)
    assert len(batch) == 16
    assert batch.obs.shape == (16, 3) and batch.action.shape == (16, 2)
    assert batch.reward.shape == batch.terminated.shape == (16, 1)
    for tensor in (batch.obs, batch.action, batch.reward, batch.next_obs, batch.terminated):
        assert tensor.dtype == torch.float32


def test_stored_transitions_are_retrievable_verbatim():
    buf = ReplayBuffer(4, 2, 1, seed=0)
    obs = np.array([1.5, -2.5], np.float32)
    buf.add(obs, np.array([0.25]), -3.0, obs * 2, True)
    assert buf.can_sample(1)
    batch = buf.sample(1)
    np.testing.assert_allclose(batch.obs.numpy()[0], obs)
    assert batch.reward.item() == pytest.approx(-3.0)
    assert batch.terminated.item() == 1.0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        ReplayBuffer(0, 1, 1)


# --------------------------------------------------------------------------- #
# HER
# --------------------------------------------------------------------------- #
class FakeGoalEnv:
    """Minimal stand-in exposing only what HER consumes."""

    tolerance = 0.5

    def compute_reward(self, achieved, desired, info):
        return (np.linalg.norm(achieved - desired, axis=-1) <= self.tolerance) * 100.0

    def is_success(self, achieved, desired):
        return np.linalg.norm(achieved - desired, axis=-1) <= self.tolerance


def fill(buf, episodes=80, length=10, rng=None):
    rng = rng or np.random.default_rng(0)
    for _ in range(episodes):
        for t in range(length):
            buf.add(
                obs=rng.normal(size=3).astype(np.float32),
                action=rng.normal(size=2).astype(np.float32),
                next_obs=rng.normal(size=3).astype(np.float32),
                desired=np.array([100.0, 100.0], np.float32),  # never reached
                # 0.1 m per step: consecutive states are within tolerance of
                # each other, as they are for a real robot at 10 Hz. With one
                # metre per step no relabelled goal would ever be "reached" and
                # the test would measure the fixture, not HER.
                achieved_next=np.array([t * 0.1, 0.0], np.float32),
            )
        buf.end_episode()
    return buf


def make(her_ratio=0.8, max_len=10):
    return HERReplayBuffer(
        capacity_transitions=4000,
        max_episode_len=max_len,
        obs_dim=3,
        action_dim=2,
        goal_dim=2,
        env=FakeGoalEnv(),
        her_ratio=her_ratio,
        seed=0,
    )


def test_network_input_is_observation_concatenated_with_goal():
    batch = fill(make()).sample(32)
    assert batch.obs.shape == (32, 5)  # 3 obs + 2 goal
    assert batch.next_obs.shape == (32, 5)


def test_without_relabelling_the_reward_is_always_zero():
    """The unreachable goal means a plain buffer has no learning signal at all."""
    batch = fill(make(her_ratio=0.0)).sample(256)
    assert float(batch.reward.abs().max()) == 0.0


def test_relabelling_manufactures_successes():
    """This is the whole point of HER, and it is what the previous
    `commit_append` -- which just copied the staged data across -- never did."""
    batch = fill(make(her_ratio=0.8)).sample(512)
    positive = float((batch.reward > 0).float().mean())
    assert 0.5 < positive < 0.95


def test_relabelled_transitions_are_marked_terminal():
    batch = fill(make(her_ratio=0.99)).sample(256)
    reached = batch.reward.squeeze(-1) > 0
    assert torch.equal(batch.terminated.squeeze(-1)[reached], torch.ones(int(reached.sum())))


def test_her_ratio_controls_how_many_are_relabelled():
    low = float((fill(make(her_ratio=0.2)).sample(512).reward > 0).float().mean())
    high = float((fill(make(her_ratio=0.9)).sample(512).reward > 0).float().mean())
    assert high > low


def test_empty_episode_commit_is_a_no_op():
    buf = make()
    buf.end_episode()
    assert len(buf) == 0


def test_overlong_episode_is_rejected_loudly():
    buf = make(max_len=3)
    for _ in range(3):
        buf.add(np.zeros(3), np.zeros(2), np.zeros(3), np.zeros(2), np.zeros(2))
    with pytest.raises(RuntimeError, match="max_episode_len"):
        buf.add(np.zeros(3), np.zeros(2), np.zeros(3), np.zeros(2), np.zeros(2))


def test_invalid_her_ratio_rejected():
    with pytest.raises(ValueError, match="her_ratio"):
        make(her_ratio=1.0)


def test_variable_length_episodes_are_sampled_in_proportion_to_length():
    buf = make(max_len=20)
    rng = np.random.default_rng(1)
    fill(buf, episodes=20, length=2, rng=rng)
    fill(buf, episodes=20, length=18, rng=rng)
    # Long episodes hold 90% of the transitions, so they must dominate the batch.
    # Their achieved goals run up to t=17, short ones only to t=1.
    buf.sample(256)  # must not raise: sampling is length-weighted
    assert len(buf) == 20 * 2 + 20 * 18
