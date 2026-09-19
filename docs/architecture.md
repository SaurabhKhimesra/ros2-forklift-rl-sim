# Architecture

## The one decision everything else follows from

The environment does not know what a simulator is.

```
                      ┌──────────────────────────────┐
                      │  ForkliftEnv (gymnasium.Env) │
                      │  - goal sampling, curriculum │
                      │  - termination vs truncation │
                      │  - reward assembly           │
                      └──────────────┬───────────────┘
                                     │  WorldState  /  DriveCommand
                      ┌──────────────┴───────────────┐
                      │      SimulationBackend       │   (abstract)
                      └───────┬──────────────┬───────┘
                              │              │
              ┌───────────────▼──┐        ┌──▼──────────────────┐
              │ KinematicBackend │        │   GazeboBackend     │
              │ pure numpy       │        │ rclpy + gazebo_ros  │
              │ ~13,000 steps/s  │        │ physics, sensors,   │
              │ no ROS at all    │        │ contacts, meshes    │
              └──────────────────┘        └─────────────────────┘
```

`WorldState` is a frozen dataclass: robot pose, goal pose, velocities, a
collision flag, the contact list and the simulated time. `DriveCommand` is a
velocity command in SI units. That is the entire contract.

Three things fall out of it:

1. **Rewards, observations, buffers and the agent never import `rclpy`.** The
   whole test suite runs on a stock GitHub runner in under 20 seconds.
2. **Reward iteration stops being expensive.** A reward-function change can be
   validated over 60,000 environment steps in about five minutes on a laptop CPU
   rather than several hours of Gazebo wall clock.
3. **The Gazebo run is a one-line config change**, not a different code path, so
   what you tuned is what you deploy.

> **Note on the Gazebo backend's status.** It was rewritten with the rest of the
> codebase but has not been re-run end to end since; the machine this was done on
> had no ROS installation. Everything above the backend interface is covered by
> the test suite against the kinematic backend, and `td3_gazebo.yaml` is
> validated in CI, but the first Gazebo run should be watched rather than
> trusted.

The fast simulator is a tool for *iterating*, not a replacement for Gazebo. It
models unicycle kinematics, acceleration limits, actuation noise, circular
obstacles and world bounds. It does not model contact dynamics, wheel slip,
sensor noise, mesh geometry or the ros2_control loop. Anything that depends on
those has to be validated in Gazebo.

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | Typed dataclasses, validation, YAML load/save, dotted overrides |
| `geometry.py` | `Pose2D`, angle wrapping, quaternion↔yaw. Pure functions |
| `paths.py` | Finds packaged configs/worlds/models via the ament index, with a source-tree fallback |
| `envs/backends/base.py` | `WorldState`, `DriveCommand`, `SimulationBackend` |
| `envs/backends/kinematic.py` | The fast simulator |
| `envs/backends/gazebo.py` | One persistent ROS 2 node, cached clients, teleport-based reset |
| `envs/observations.py` | Feature registry; the observation space is *derived* from it |
| `envs/rewards.py` | Weighted, individually-tested reward terms |
| `envs/actions.py` | Normalised action → rate-limited SI velocity command |
| `envs/forklift_env.py` | The Gymnasium env, goal sampling and the curriculum |
| `rl/networks.py` | Actor and twin critic MLPs |
| `rl/buffers.py` | Uniform replay + a real HER buffer |
| `rl/td3.py` | TD3 (DDPG is `policy_delay=1`, single critic) |
| `rl/train.py` | Training and evaluation loops |
| `utils/` | Seeding, run logging (TB + CSV), figures |
| `cli.py` | `train` / `eval` / `report` / `config` / `list-configs` |

## Data flow through one step

```
action ∈ [-1,1]²
   └─► DifferentialDriveActuator   scale to m/s, rad/s; apply accel limits
         └─► SimulationBackend.step(DriveCommand, dt)  ──►  WorldState
               ├─► ObservationBuilder  ──► float32 vector (space derived from it)
               ├─► RewardFunction      ──► scalar + per-term breakdown
               └─► termination check   ──► terminated / truncated, kept separate
```

## Why the observation space cannot drift from the observation

`OBSERVATION_FEATURES` maps a name to a `Feature` that declares its own width
and bounds. `ObservationBuilder` concatenates exactly the features the config
lists, and `observation_space` is built from the same list. A feature that
produces the wrong number of values raises immediately, naming itself. A config
that names a feature that does not exist is rejected before a simulator starts,
with the valid names listed in the error.

## Goal-conditioned mode and HER

With `algo.use_her: true` the observation becomes a `Dict` of
`observation` / `achieved_goal` / `desired_goal`, and the policy input is
`concat(observation, desired_goal)`.

Hindsight relabelling rewrites the goal after the episode is over, so any
observation feature computed *from* the goal would silently become wrong.
Features declare `goal_independent`, and the config loader refuses a HER run
whose observation includes a goal-dependent feature — with a message that says
which ones and what to use instead.

## Extending it

**A new reward term:** write a function of `RewardContext` in `envs/rewards.py`,
add it to `REWARD_TERMS`, add a field with the same name to `RewardConfig`.
Weight `0.0` disables it. Add a test.

**A new observation feature:** write a function in `envs/observations.py`, add a
`Feature` to `OBSERVATION_FEATURES` with its width, bounds and goal-dependence.
Nothing else changes — the space resizes itself.

**A new backend:** subclass `SimulationBackend`, implement `reset`, `step` and
`close`, and add a branch to `ForkliftEnv._make_backend`. Isaac Sim, MuJoCo or a
real forklift over ROS all fit the same three methods.

**A new algorithm:** the agent interface is `act(obs, noise) -> action` and
`update(Batch) -> UpdateMetrics`. SAC would be a new file next to `td3.py`.
