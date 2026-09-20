# Rewrite notes (v1.0)

What the v1.0 Python stack does differently from the version it grew out of,
and the reasoning behind each decision.

This is a technical record, not a scorecard. The original project got a forklift,
a full sensor stack and a contact-detection plugin working in Gazebo — the hard,
unglamorous part — and all of it is carried over here intact. What changed is the
Python side, and it is written up in this much detail because the failure modes
are genuinely instructive: most are the kind that produce a plausible-looking
training curve and a policy that never works, and most are mistakes anyone
writing RL code makes once.

---

## Correctness bugs

### `actor_loss` referenced outside the branch that defines it
`DDPG_Agent.update()` ended with `return critic_loss, actor_loss`, but
`actor_loss` was only assigned inside `if self._n_updates % self.policy_update_delay == 0`.
Any `policy_update_delay > 1` — that is, TD3 as the README described it — raised
`UnboundLocalError` on the second update. The shipped config used `1`, which hid it.

*Now:* `update()` returns an `UpdateMetrics` dataclass whose `actor_loss` is
`None` on a skipped update. Tested at delays 1, 2 and 3.

### `zip()` silently left half the target network trainable
```python
for actor_p, critic_p in zip(self.actor_target.parameters(),
                             self.critic_target.parameters()):
    actor_p.requires_grad = False
    critic_p.requires_grad = False
```
`zip` stops at the shorter sequence. The actor and critic have different numbers
of parameter tensors whenever their hidden-layer lists differ, so the tail of the
longer target network stayed trainable.

*Now:* `copy.deepcopy(net).requires_grad_(False)`, with a test asserting the two
target networks genuinely differ in length.

### `CollisionDetectionSubscriber` was never imported
`calc_reward_collision_penalty` referenced it in both branches. Enabling
`collision_penalty` in the config raised `NameError` at the first step.
Confirmed with `pyflakes`.

### The reward function threw away everything it computed
`calc_reward_L2_dist` computed `angular_cost` and `action_norm_penalty_reward`,
then fell through to `return -0.1 * l2_dist`. Both locals were dead.

### Heading error was never wrapped
`total_angle_difference_to_goal = angle_to_goal - theta`, unwrapped, fed straight
into the observation. It jumps by 2π as the robot crosses ±π — a step
discontinuity in the policy input.

*Now:* wrapped to `[-π, π)` and encoded as `(sin, cos)`, with tests at the boundary.

### A trailing comma made a float into a tuple
```python
x = obs[...]['position'].x,      # <- tuple of length 1
inc_x = goal.x - x               # np.float64 - tuple -> array of shape (1,)
```
It survived only because NumPy broadcast the operation and `math.atan2` accepted
a size-1 array. NumPy ≥ 1.25 raises on that conversion.

### `if __name__ == main:` in `run_pytest.py`
Comparing to the function object, not `"__main__"`. And `args = ["-m replay_buffer or forklift_env"]`
passed one argument where pytest needs two.

### Entry points pointing at modules that do not exist
`setup.py` declared console scripts for `forklift_gym_env.rl.DDPG_HER.train_HER_DDPG`
and `forklift_gym_env.rl.DDPG.DDPG_openai`. Neither file is in the repository, so
`make train_HER_DDPG` and `make DDPG_openAI_gym` failed with `ImportError`.

### `ActionType.FORK_JOINT_CONT = "jork_joint_control"`
Typo'd value. The README and configs said `fork_joint_control`, so selecting it
raised `ValueError`.

### `train_sb3.py` silently trained on Pendulum
The `DDPG` branch — the one the file selected by default — replaced the forklift
environment with `gym.make('Pendulum-v1')` — leftover debug code that was never
taken back out.

---

## The config described an environment that did not exist

`config_DDPG_forklift_env.yaml` declared
`observation_types: ["fork_position", "target_tf", "latest_action"]`.

In `flatten_and_concatenate_observation`, the concatenation lines for
`target_tf`, `pallet_position` and `latest_action` were all commented out. The
observation was two numbers — heading error and distance — and
`observation_space_factory` returned a hard-coded `Box(shape=(2,))` regardless of
what the config said.

That is what forced this, in the training script:

```python
agent.actor.model_layers[1].weight.data[:] = torch.tensor([[0.0, 0.3], [1.5, 0.0]])
```

A 2×2 matrix written directly into the actor to seed it as a proportional
controller — which also meant the code crashed for any other observation or
action size, and that the resulting curve reflected a partly hand-set controller
rather than a fully learned policy.

*Now:* features declare their own width; the space is derived from the same list
the builder uses; the config rejects unknown feature names. No hand-set weights
anywhere.

---

## Design choices that held learning back

### The pallet never moved
Every randomisation line in `reset()` was commented out, leaving
`self._target_transform = np.array([6.0, 2.0])` for all 200 episodes. A policy
trained against one fixed goal has no reason to learn the task rather than the
coordinate.

*Now:* goals are sampled over an annulus, with an optional curriculum that widens
the distribution as the success rate rises. `randomize: false` is still there for
debugging.

### `done` conflated termination with truncation
The env returned one flag; the training loop tried to reconstruct the difference
by comparing `cur_iteration < max_episode_length`. A time-limit cut-off was
bootstrapped as a real terminal state, which teaches the agent the world ends at
step 300.

*Now:* the Gymnasium five-tuple, and only `terminated` stops the bootstrap.

### BatchNorm in the actor and the critic
Both networks had a `BatchNorm1d` before the output layer. The actor is
evaluated at batch size 1 when acting and 256 when training, so the two disagree;
the critic's running statistics drift as the replay distribution shifts, making
the Q-target non-stationary on top of the usual bootstrapping.

*Now:* no BatchNorm (a test asserts it), small final-layer initialisation so
`tanh` starts unsaturated, and `LayerNorm` available behind a flag.

### The HER buffer did not relabel
The buffer was documented as Hindsight Experience Replay, but `commit_append()`,
the method meant to do the relabelling, copied the staged transitions across
unchanged.

*Now:* a real `future`-strategy HER buffer with episode-major storage,
length-weighted episode sampling and recomputed rewards and terminals. Tested by
constructing episodes that never reach their goal and asserting that relabelling
manufactures successes where a plain buffer yields none.

### The exploration schedule depended on episode length
`epsilon *= epsilon_decay` ran once per *action*, so the effective noise at a
given training step depended on how long episodes happened to be.

*Now:* a linear anneal in environment steps.

---

## Throughput

### A new ROS 2 node per service call
`SimulationController` called `rclpy.create_node(...)` — a new DDS participant,
with the discovery traffic that implies — and destroyed it again, for *every*
service call. Two per `step()` (pause + unpause) and eight or more per `reset()`.

*Now:* one node, created once, with every client and publisher cached on it and a
single executor spinning it in a background thread.

### Six seconds of `sleep` per reset
`change_entity_location` deleted the entity, slept 3 s "to make sure that
DeleteEntity request is finished", then respawned it and reloaded the ros2_control
controllers. Called twice per reset, plus a `time.sleep(3)` in the training loop.

*Now:* entities are spawned once at launch and **teleported** with
`/gazebo/set_entity_state` (zeroing the twist, or they keep drifting). Controllers
stay loaded. No fixed sleeps.

### Unbounded busy-waits
The observation loops were `while ...:` with no timeout, no sleep and no spin — a
stalled `gzserver` meant an infinite loop.

*Now:* every wait has a timeout and raises `GazeboTimeoutError` naming what it was
waiting for.

### Physics was capped at real time
The world set `real_time_update_rate: 1000` with `max_step_size: 0.001`, pinning
the simulation to 1× real time.

*Now:* `real_time_update_rate: 0`. The environment paces itself off `/clock`, so
uncapping it is free throughput.

### A logging call per tensorboard key, every step
`Logger.log_tabular` did a synchronous `logging.info` on each call, ~10 times per
environment step. And `logging.getLogger('MyCustomLogger')` ignored the `name`
argument, so every `Logger` shared one underlying logger and each construction
stacked another handler onto it.

*Now:* batched scalars to TensorBoard plus a `metrics.csv`, printed at a
configurable interval.

---

## Packaging and reproducibility

| Was | Now |
|---|---|
| `packages=[package_name]` — subpackages never installed; worked only under `--symlink-install` | `find_packages()` |
| `config/` commented out of `data_files`, so three scripts hard-coded `'build/forklift_gym_env/...'` | configs installed to `share/`, resolved through the ament index with a source fallback |
| `package.xml`: no dependencies declared, placeholder description and license | real `exec_depend` list, real metadata, Apache-2.0 |
| `requirements.txt`: 150-line `pip freeze` pinning ROS message packages, **both** `gym==0.21.0` and `gymnasium==0.27.1`, `numpy==1.23.3` against torch 1.13 | 4 runtime pins; ROS deps in `package.xml` where `rosdep` can act on them |
| Dockerfile copied the repo before installing, and had both dependency installs commented out | layered so deps cache; a code edit rebuilds in seconds; plus a ROS-free `lite` stage for CI |
| Six launch files differing only in what was commented out | one, with arguments |
| `Makefile` with a hard-coded absolute home-directory path | no absolute paths |
| No CI | GitHub Actions: lint, tests on 3.10–3.12, every config validated, a smoke train, and a separate colcon build |
| No tests that ran (the env test file was commented out, the rest renamed `no_test_*`) | 119 tests, no ROS required |
| Editor swap files not ignored | `.gitignore` covers them, along with `runs/` and checkpoints |
| Everything on CPU, no device handling | `device: auto`, `map_location` on load |
| `torch.save` of four `.pkl` files into the CWD, no optimiser state, no step count | one checkpoint with optimiser state, step, and the dims it was trained for |

---

## A bug introduced during this rewrite, and kept as a lesson

The first version of the new reward used potential-based shaping with
`γ = 0.98` and `time = -0.01`.

Potential-based shaping adds `γ·Φ(s′) − Φ(s)`. With `Φ = −distance` and a
stationary robot that is `(1 − γ)·distance` — **positive**. Measured with a
hand-written do-nothing policy on a 250-step episode with the pallet 7.6 m away:
standing still earned **+36**. A proportional controller that actually drives to
the pallet earned +112, so idling was not the global optimum — but it was a
positive-return local optimum sitting exactly where an agent starts, before it
has ever reached the goal and discovered the success bonus.

Does it break training? On this task, measurably no: run both weightings for
25,000 steps and both reach 100%, both dipping partway through. The guard below
is therefore a safeguard against a provably wrong reward function, not the
rescue of a failing run.

`RewardConfig.check_idling_is_unprofitable` now rejects any weighting where the
per-step cost does not exceed the maximum idle income, with a message that
states the threshold. `scripts/reward_shaping_demo.py` reproduces both
measurements.

**The meta-lesson is the more useful one.** The first write-up of this finding
claimed the fix took the success rate from 15% to 100%. It did not — that
comparison had changed four things at once (γ, the time penalty, the episode
limit and the network width). Re-running it with *only* the reward weights
changed produced the much smaller, real effect above. A result that flatters the
fix is exactly the one to re-run before publishing.
