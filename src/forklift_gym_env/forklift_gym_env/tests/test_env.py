import numpy as np
import pytest

from forklift_gym_env.config import Config
from forklift_gym_env.envs.backends.base import DriveCommand
from forklift_gym_env.envs.backends.kinematic import KinematicBackend
from forklift_gym_env.envs.forklift_env import ForkliftEnv, GoalSampler
from forklift_gym_env.geometry import Pose2D


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert {"distance", "goal", "robot", "is_success"} <= set(info)


def test_step_returns_the_five_tuple(env):
    env.reset(seed=1)
    out = env.step(env.action_space.sample())
    assert len(out) == 5
    obs, reward, terminated, truncated, _info = out
    assert env.observation_space.contains(obs)
    assert (
        isinstance(reward, float) and isinstance(terminated, bool) and isinstance(truncated, bool)
    )


def test_step_before_reset_raises(config):
    with pytest.raises(RuntimeError, match="before reset"):
        ForkliftEnv(config).step(np.zeros(2, np.float32))


def test_truncation_is_distinct_from_termination(config):
    """The bug this guards: the old env returned one `done` flag, so a time-limit
    cut-off was indistinguishable from reaching the pallet and the learner
    stopped bootstrapping through it."""
    config.env.max_episode_steps = 5
    env = ForkliftEnv(config)
    env.reset(seed=3, options={"start": Pose2D(0, 0, 0), "goal": Pose2D(50, 50)})
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, np.float32))
    assert truncated and not terminated
    env.close()


def test_reaching_the_goal_terminates_without_truncation(config):
    env = ForkliftEnv(config)
    env.reset(seed=3, options={"start": Pose2D(0, 0, 0), "goal": Pose2D(0.1, 0.0)})
    _, _, terminated, truncated, info = env.step(np.array([0.1, 0.0], np.float32))
    assert terminated and not truncated and info["is_success"]
    env.close()


def test_seeded_resets_are_reproducible(config):
    a, b = ForkliftEnv(config), ForkliftEnv(config)
    for seed in (0, 7, 99):
        oa, ia = a.reset(seed=seed)
        ob, ib = b.reset(seed=seed)
        np.testing.assert_array_equal(oa, ob)
        assert ia["goal"] == ib["goal"]
    a.close(), b.close()


def test_actions_outside_the_box_are_clipped_not_rejected(env):
    env.reset(seed=0)
    env.step(np.array([50.0, -50.0], np.float32))  # must not raise


def test_episode_info_is_emitted_once_at_the_end(config):
    config.env.max_episode_steps = 6
    env = ForkliftEnv(config)
    env.reset(seed=1, options={"start": Pose2D(0, 0, 0), "goal": Pose2D(40, 40)})
    infos = [env.step(np.zeros(2, np.float32))[4] for _ in range(6)]
    assert sum("episode" in i for i in infos) == 1
    assert set(infos[-1]["episode"]) == {"r", "l", "success", "collision", "final_distance"}
    env.close()


def test_reward_breakdown_sums_to_the_reward(env):
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.array([0.5, 0.2], np.float32))
    assert sum(info["reward_terms"].values()) == pytest.approx(reward)


def test_goal_conditioned_mode_exposes_the_her_keys(config):
    env = ForkliftEnv(config, goal_conditioned=True)
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    assert env.observation_space.contains(obs)
    env.close()


def test_compute_reward_is_vectorised_and_sparse(config):
    env = ForkliftEnv(config)
    tol = config.env.goal.position_tolerance
    achieved = np.array([[0.0, 0.0], [10.0, 0.0]], np.float32)
    desired = np.array([[tol * 0.5, 0.0], [0.0, 0.0]], np.float32)
    rewards = env.compute_reward(achieved, desired, None)
    assert rewards.shape == (2,)
    assert rewards[0] == pytest.approx(config.env.reward.success) and rewards[1] == 0.0
    env.close()


# --------------------------------------------------------------------------- #
# Goal sampling and curriculum
# --------------------------------------------------------------------------- #
def test_goals_are_randomised_by_default(config):
    env = ForkliftEnv(config)
    goals = {env.reset(seed=s)[1]["goal"] for s in range(20)}
    assert len(goals) == 20, "the old env pinned the pallet at a single fixed point"
    env.close()


def test_fixed_goal_mode_still_works(config):
    config.env.goal.randomize = False
    config.env.goal.fixed_position = (6.0, 2.0)
    env = ForkliftEnv(config)
    assert {env.reset(seed=s)[1]["goal"] for s in range(5)} == {(6.0, 2.0)}
    env.close()


def test_sampled_goals_respect_the_configured_distance_range(config):
    config.env.goal.distance_range = (3.0, 5.0)
    env = ForkliftEnv(config)
    for s in range(50):
        assert 3.0 - 1e-6 <= env.reset(seed=s)[1]["distance"] <= 5.0 + 1e-6
    env.close()


def test_curriculum_advances_only_on_sustained_success(config):
    config.env.goal.curriculum.enabled = True
    config.env.goal.curriculum.window = 10
    config.env.goal.curriculum.success_rate_to_advance = 0.8
    sampler = GoalSampler(config.env)

    for _ in range(9):
        assert not sampler.record_outcome(True)
    assert sampler.record_outcome(True)  # window complete, 100% >= 80%
    assert sampler.progress > 0.0

    before = sampler.progress
    for _ in range(10):
        sampler.record_outcome(False)
    assert sampler.progress == before  # failures must not promote


def test_curriculum_widens_the_sampling_range(config):
    config.env.goal.curriculum.enabled = True
    config.env.goal.curriculum.window = 5
    sampler = GoalSampler(config.env)
    easy, _ = sampler.current_ranges()
    for _ in range(20):
        sampler.record_outcome(True)
    hard, _ = sampler.current_ranges()
    assert hard[1] > easy[1]


# --------------------------------------------------------------------------- #
# Kinematic backend
# --------------------------------------------------------------------------- #
def test_straight_line_travel_matches_velocity_times_time():
    backend = KinematicBackend(Config().env.robot, rng=np.random.default_rng(0))
    backend.cfg.velocity_noise_std = 0.0
    backend.reset(Pose2D(0, 0, 0), Pose2D(10, 0))
    for _ in range(10):
        state = backend.step(DriveCommand(linear=1.0, angular=0.0), 0.1)
    assert state.robot.x == pytest.approx(1.0, abs=1e-9)
    assert state.robot.y == pytest.approx(0.0, abs=1e-9)


def test_pure_rotation_does_not_translate():
    backend = KinematicBackend(Config().env.robot, rng=np.random.default_rng(0))
    backend.cfg.velocity_noise_std = 0.0
    backend.reset(Pose2D(0, 0, 0), Pose2D(5, 0))
    state = backend.step(DriveCommand(linear=0.0, angular=1.0), 0.5)
    assert (state.robot.x, state.robot.y) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert state.robot.yaw == pytest.approx(0.5)


def test_constant_turn_traces_a_circle_of_the_right_radius():
    """Exact arc integration, not an Euler step - this is what keeps a 10 Hz
    control rate from visibly cutting corners."""
    cfg = Config().env
    cfg.kinematic.velocity_noise_std = 0.0
    backend = KinematicBackend(cfg.robot, cfg.kinematic, np.random.default_rng(0))
    backend.reset(Pose2D(0, 0, 0), Pose2D(5, 0))
    # Chosen so one revolution is exactly 100 control ticks; a partial final
    # step would leave a residual and make the tolerance meaningless.
    dt, steps = 0.1, 100
    v, w = 1.0, 2 * np.pi / (steps * dt)
    for _ in range(steps):
        state = backend.step(DriveCommand(v, w), dt)
    assert state.robot.x == pytest.approx(0.0, abs=1e-6)
    assert state.robot.y == pytest.approx(0.0, abs=1e-6)


def test_obstacle_contact_is_reported():
    cfg = Config().env
    cfg.kinematic.velocity_noise_std = 0.0
    cfg.kinematic.obstacles = [(1.0, 0.0, 0.3)]
    backend = KinematicBackend(cfg.robot, cfg.kinematic, np.random.default_rng(0))
    backend.reset(Pose2D(0, 0, 0), Pose2D(5, 0))
    states = [backend.step(DriveCommand(1.0, 0.0), 0.1) for _ in range(12)]
    assert any(s.in_collision for s in states)


def test_world_bounds_clamp_and_flag():
    cfg = Config().env
    cfg.kinematic.velocity_noise_std = 0.0
    cfg.kinematic.world_bounds = (1.0, 1.0)
    backend = KinematicBackend(cfg.robot, cfg.kinematic, np.random.default_rng(0))
    backend.reset(Pose2D(0, 0, 0), Pose2D(5, 0))
    for _ in range(40):
        state = backend.step(DriveCommand(1.0, 0.0), 0.1)
    assert state.robot.x == pytest.approx(1.0)
    assert state.in_collision and "world_bounds" in state.contacts


def test_acceleration_limit_prevents_instant_reversal(config):
    """A policy that flips full-forward to full-reverse in one 100 ms tick is not
    something a real drivetrain can follow."""
    env = ForkliftEnv(config)
    env.reset(seed=0)
    for _ in range(20):
        env.step(np.array([1.0, 0.0], np.float32))
    before = env.state.linear_velocity
    _, _, _, _, _ = env.step(np.array([-1.0, 0.0], np.float32))
    after = env.state.linear_velocity
    max_delta = config.env.robot.max_linear_accel * config.env.dt
    assert abs(after - before) <= max_delta * 1.2
    env.close()
