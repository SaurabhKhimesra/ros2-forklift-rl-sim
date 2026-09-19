import pytest
import yaml

from forklift_gym_env.config import Config, ConfigError, load_config, save_config


def write(tmp_path, payload):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_defaults_are_valid():
    Config().validate()


def test_unknown_top_level_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config(write(tmp_path, {"trian": {"total_steps": 10}}))


def test_unknown_nested_key_names_its_path(tmp_path):
    with pytest.raises(ConfigError, match=r"algo.gama"):
        load_config(write(tmp_path, {"algo": {"gama": 0.9}}))


def test_unknown_observation_feature_lists_the_valid_ones(tmp_path):
    with pytest.raises(ConfigError, match="unknown feature"):
        load_config(write(tmp_path, {"env": {"observation": ["fork_positon"]}}))


def test_duplicate_observation_features_rejected(tmp_path):
    with pytest.raises(ConfigError, match="duplicates"):
        load_config(write(tmp_path, {"env": {"observation": ["velocity", "velocity"]}}))


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"algo": {"gamma": 1.5}}, "gamma"),
        ({"algo": {"tau": 0.0}}, "tau"),
        ({"algo": {"policy_delay": 0}}, "policy_delay"),
        ({"algo": {"name": "sac"}}, "algo.name"),
        ({"env": {"backend": "mujoco"}}, "env.backend"),
        ({"env": {"max_episode_steps": 0}}, "max_episode_steps"),
        ({"run": {"device": "tpu"}}, "device"),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, payload, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, payload))


def test_her_rejects_goal_dependent_observations(tmp_path):
    with pytest.raises(ConfigError, match="goal-independent"):
        load_config(
            write(
                tmp_path,
                {
                    "algo": {"use_her": True},
                    "env": {"observation": ["goal_vector_body", "velocity"]},
                },
            )
        )


def test_her_accepts_goal_independent_observations(tmp_path):
    cfg = load_config(
        write(
            tmp_path,
            {
                "algo": {"use_her": True},
                "env": {
                    "observation": ["robot_pose_world", "velocity"],
                    "reward": {
                        "progress": 0.0,
                        "heading": 0.0,
                        "time": 0.0,
                        "action_magnitude": 0.0,
                        "action_smoothness": 0.0,
                    },
                },
            },
        )
    )
    assert cfg.algo.use_her


def test_idling_loophole_is_caught(tmp_path):
    """Potential shaping with gamma<1 pays a stationary agent; the cost must exceed it."""
    with pytest.raises(ConfigError, match="idling beats driving"):
        load_config(write(tmp_path, {"algo": {"gamma": 0.98}, "env": {"reward": {"time": -0.01}}}))


def test_idling_check_passes_with_a_steep_enough_cost(tmp_path):
    load_config(write(tmp_path, {"algo": {"gamma": 0.99}, "env": {"reward": {"time": -0.2}}}))


def test_overrides_apply_to_nested_keys(tmp_path):
    cfg = load_config(write(tmp_path, {}), {"train.total_steps": 7, "algo.batch_size": 8})
    assert cfg.train.total_steps == 7 and cfg.algo.batch_size == 8


def test_save_then_load_roundtrips(tmp_path):
    cfg = Config().validate()
    save_config(cfg, tmp_path / "out.yaml")
    assert load_config(tmp_path / "out.yaml").to_dict() == cfg.to_dict()


def test_missing_file_message_names_the_path(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
