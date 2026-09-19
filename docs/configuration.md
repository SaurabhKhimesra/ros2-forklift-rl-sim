# Configuration

Every experiment is one YAML file. It is parsed into typed dataclasses
(`forklift_gym_env/config.py`), and **unknown keys are an error** — a
misspelled key fails at load time with the valid names listed, instead of
silently doing nothing.

```bash
forklift list-configs                      # what this install can find
forklift config -c td3_kinematic.yaml      # print the fully-resolved config
forklift train  -c td3_kinematic.yaml --set train.total_steps=5000
```

`--set` takes dotted paths and YAML-parses the value, so
`--set env.goal.distance_range="[3, 5]"` works.

## Shipped configs

| File | Task | Notes |
|---|---|---|
| `td3_kinematic.yaml` | drive to the pallet | start here; trains in ~5 min on a CPU |
| `ddpg_kinematic.yaml` | same | single critic, no smoothing, undelayed actor |
| `td3_align_kinematic.yaml` | drive to it **and square up with it** | harder; no curriculum |
| `ddpg_align_kinematic.yaml` | same | the DDPG baseline for that task |
| `td3_her_kinematic.yaml` | sparse reward + hindsight replay | no shaping at all |
| `td3_gazebo.yaml` | same task, real physics | needs a sourced ROS 2 workspace |

## Sections

### `run`
| Key | Default | |
|---|---|---|
| `name` | `forklift` | prefix for the run directory |
| `seed` | `42` | seeds python, numpy, torch and a dedicated env generator |
| `device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `output_dir` | `runs` | run directories are created under here |

### `env`
| Key | Default | |
|---|---|---|
| `backend` | `kinematic` | `kinematic` \| `gazebo` |
| `max_episode_steps` | `250` | truncation limit (not a termination) |
| `control_hz` | `10.0` | one action every `1/control_hz` seconds of sim time |
| `observation` | see below | ordered list of feature names |
| `normalize_observations` | `true` | divide by fixed scales so inputs sit near ±1 |
| `terminate_on_collision` | `true` | |

#### Observation features

| Name | Width | Goal-dependent | |
|---|---|---|---|
| `goal_vector_body` | 2 | yes | pallet position in the robot frame |
| `goal_distance` | 1 | yes | |
| `heading_error` | 2 | yes | bearing to the pallet as `(sin, cos)` |
| `alignment_error` | 2 | yes | yaw offset from the pallet's own heading, as `(sin, cos)` |
| `velocity` | 2 | no | current `(linear, angular)` |
| `last_action` | 2 | no | previous command |
| `robot_pose_world` | 4 | no | `x, y, sin(yaw), cos(yaw)` |
| `goal_pose_world` | 2 | yes | absolute pallet position |
| `collision_flag` | 1 | no | |

Angles are always `(sin, cos)`, never a raw angle: a raw heading error jumps by
2π as the robot crosses ±π, and that discontinuity lands in both the observation
and the reward.

Body-frame features (`goal_vector_body`, `heading_error`) are what let one
policy handle a pallet anywhere. World-frame features tie the policy to absolute
coordinates and generalise worse — they exist because HER needs goal-independent
inputs.

### `env.robot`
`max_linear_speed`, `max_angular_speed`, `max_linear_accel`, `max_angular_accel`,
`allow_reverse`, `footprint_radius`. The acceleration limits are enforced in the
actuator, so the policy cannot command a full forward-to-reverse flip in one
100 ms tick. Keep them consistent with `diff_cont` in
`src/forklift_robot/config/my_controllers.yaml`.

### `env.goal`
| Key | Default | |
|---|---|---|
| `randomize` | `true` | **leave this on** — see below |
| `distance_range` | `[2.0, 8.0]` | metres, sampled uniformly over the annulus |
| `bearing_range_deg` | `[-180, 180]` | relative to the robot's initial heading |
| `fixed_position` | `[6.0, 2.0]` | only used when `randomize: false` |
| `position_tolerance` | `0.6` | metres |
| `require_heading` | `false` | also require the yaw to match the pallet's |
| `heading_tolerance_deg` | `25.0` | only checked when `require_heading` |

With a fixed goal there is no reason for the policy to learn "drive to the
pallet" when "drive to (6, 2)" scores identically and is simpler — and the
result transfers to no other pallet position.

### `env.goal.curriculum`
Starts with near goals in a narrow bearing cone and widens the distribution each
time the success rate over the last `window` episodes clears
`success_rate_to_advance`. Set `enabled: false` for a clean baseline.

### `env.reward`
Weights; `0.0` switches a term off.

| Term | Default | |
|---|---|---|
| `progress` | `1.0` | potential-based shaping on distance |
| `heading` | `0.05` | potential-based shaping on bearing to the pallet |
| `alignment` | `0.0` | potential-based shaping on squaring up, gated near the pallet |
| `success` | `100.0` | one-off |
| `collision` | `-50.0` | one-off, ends the episode |
| `time` | `-0.12` | per step — **see the warning below** |
| `action_magnitude` | `-0.01` | mean squared command |
| `action_smoothness` | `-0.02` | mean squared change in command |
| `reverse` | `0.0` | extra cost on reverse motion |

> **The idling trap.** Potential-based shaping adds `γ·Φ(s′) − Φ(s)`. With
> `Φ = −distance` and a **stationary** robot that is not zero, it is
> `(1 − γ)·distance`, which is *positive*. So with `γ < 1` a distant agent earns
> a small income every step for doing nothing. If the per-step cost does not
> exceed it, parking beats driving — and the symptom is subtle: returns look
> positive and healthy while every episode quietly runs to the time limit.
>
> The loader rejects this. `reward.time` must be steeper than
> `(1 − γ) · max(distance_range) · progress`. At `γ = 0.99` and an 8 m goal that
> is `−0.08`/step; the shipped configs use `−0.12`.
>
> This is not hypothetical — it is what the first version of this reward did,
> and the fix took the success rate from 15% to 100%.

### `algo`
`name` (`td3` | `ddpg`), `actor_hidden`, `critic_hidden`, `actor_lr`,
`critic_lr`, `gamma`, `tau`, `batch_size`, `buffer_size`, `exploration_noise`
and `exploration_noise_final` (linearly annealed), `target_policy_noise`,
`target_noise_clip`, `policy_delay`, `max_grad_norm`, `use_her`, `her_ratio`.

`policy_delay: 1` plus `target_policy_noise: 0.0` and `name: ddpg` is DDPG.

### `train`
`total_steps`, `warmup_steps` (uniform-random actions to seed the buffer),
`updates_per_step`, `update_every` (updates are batched to amortise overhead),
`eval_every`, `eval_episodes`, `checkpoint_every`, `log_every`.

In Gazebo the simulator is the bottleneck, so raise `updates_per_step` and lower
`total_steps` — squeeze more learning out of each transition you paid for.

## Run outputs

```
runs/<name>_<algo>_<backend>_<timestamp>/
├── config.yaml        fully-resolved config, for exact reproduction
├── run_meta.json      observation layout, dims, device, active reward terms
├── metrics.csv        every logged scalar
├── tb/                tensorboard events
├── best.pt            highest evaluation success rate
├── latest.pt          most recent periodic checkpoint
├── final.pt           end of training
└── final_eval.json    closing evaluation
```

Checkpoints carry the optimiser state and the step count, so resuming is not a
cold restart, and are loaded with `map_location`, so a GPU-trained checkpoint
opens on a laptop.
