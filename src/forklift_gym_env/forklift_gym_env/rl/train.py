"""Training and evaluation loops.

Differences from the previous ``train_DDPG.py`` worth calling out:

* ``terminated`` and ``truncated`` are kept apart, so a time-limit cut-off still
  bootstraps instead of teaching the agent that the world ends at step 300.
* Evaluation runs on a separate, noise-free environment with a fixed seed, so the
  reported success rate is comparable across checkpoints.
* Exploration noise is annealed on a linear schedule instead of the previous
  per-*action* multiplicative epsilon decay, which made the effective noise
  depend on how many steps an episode happened to last.
* No hand-set weights. The original wrote a hard-coded 2x2 matrix into the
  actor's first layer to seed it with a proportional controller -- which also
  meant the code crashed for any observation or action size but that one.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forklift_gym_env.config import Config, save_config
from forklift_gym_env.envs.forklift_env import ForkliftEnv
from forklift_gym_env.rl.buffers import HERReplayBuffer, ReplayBuffer
from forklift_gym_env.rl.td3 import TD3Agent, resolve_device
from forklift_gym_env.utils.logging import RunLogger
from forklift_gym_env.utils.seeding import seed_everything


@dataclass
class EvalResult:
    success_rate: float
    mean_return: float
    mean_length: float
    mean_final_distance: float
    collision_rate: float

    def as_metrics(self) -> dict[str, float]:
        return {
            "eval/success_rate": self.success_rate,
            "eval/return": self.mean_return,
            "eval/episode_length": self.mean_length,
            "eval/final_distance": self.mean_final_distance,
            "eval/collision_rate": self.collision_rate,
        }


def make_env(cfg: Config, seed: int | None = None, **kwargs: Any) -> ForkliftEnv:
    env = ForkliftEnv(cfg, **kwargs)
    env.reset(seed=cfg.run.seed if seed is None else seed)
    return env


def policy_input(obs: Any, goal_conditioned: bool) -> np.ndarray:
    """Flatten the env observation into the vector the networks consume."""
    if not goal_conditioned:
        return np.asarray(obs, np.float32)
    return np.concatenate([obs["observation"], obs["desired_goal"]]).astype(np.float32)


def evaluate(
    agent: TD3Agent,
    env: ForkliftEnv,
    episodes: int,
    seed: int = 12345,
    collect_trajectories: bool = False,
) -> tuple[EvalResult, list[np.ndarray], list[tuple[float, float]]]:
    """Deterministic rollout. Returns metrics and, optionally, trajectories."""
    goal_conditioned = env.goal_conditioned
    returns, lengths, successes, distances, collisions = [], [], [], [], []
    trajectories: list[np.ndarray] = []
    goals: list[tuple[float, float]] = []

    for i in range(episodes):
        obs, info = env.reset(seed=seed + i)
        goals.append(info["goal"])
        total, steps, done = 0.0, 0, False
        while not done:
            action = agent.act(policy_input(obs, goal_conditioned), noise_scale=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)
        successes.append(float(info.get("is_success", False)))
        distances.append(info["distance"])
        collisions.append(float(info["collision"]))
        if collect_trajectories:
            trajectories.append(env.trajectory.copy())

    result = EvalResult(
        success_rate=float(np.mean(successes)),
        mean_return=float(np.mean(returns)),
        mean_length=float(np.mean(lengths)),
        mean_final_distance=float(np.mean(distances)),
        collision_rate=float(np.mean(collisions)),
    )
    return result, trajectories, goals


def train(cfg: Config, run_dir: str | Path | None = None, quiet: bool = False) -> Path:
    """Run a full training job. Returns the run directory."""
    rng = seed_everything(cfg.run.seed)
    device = resolve_device(cfg.run.device)

    run_dir = Path(run_dir) if run_dir else Path(cfg.run.output_dir) / _run_name(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    env = make_env(cfg, seed=cfg.run.seed)
    eval_env = make_env(cfg, seed=cfg.run.seed + 1_000)
    goal_conditioned = env.goal_conditioned

    obs_dim = int(
        np.prod(
            env.observation_space["observation"].shape
            if goal_conditioned
            else env.observation_space.shape
        )
    )
    goal_dim = int(env.observation_space["desired_goal"].shape[0]) if goal_conditioned else 0
    action_dim = int(env.action_space.shape[0])

    agent = TD3Agent(obs_dim + goal_dim, action_dim, cfg.algo, device)

    if cfg.algo.use_her:
        buffer: Any = HERReplayBuffer(
            cfg.algo.buffer_size,
            cfg.env.max_episode_steps,
            obs_dim,
            action_dim,
            goal_dim,
            env,
            cfg.algo.her_ratio,
            device,
            seed=cfg.run.seed,
        )
    else:
        buffer = ReplayBuffer(cfg.algo.buffer_size, obs_dim, action_dim, device, seed=cfg.run.seed)

    logger = RunLogger(run_dir, quiet=quiet)
    logger.log_text(
        f"[{cfg.run.name}] {cfg.algo.name.upper()} on {env.config.env.backend} backend | "
        f"obs={obs_dim + goal_dim} act={action_dim} device={device} "
        f"her={cfg.algo.use_her} -> {run_dir}"
    )
    logger.save_json(
        "run_meta.json",
        {
            "obs_dim": obs_dim + goal_dim,
            "action_dim": action_dim,
            "device": str(device),
            "observation_layout": {k: [s.start, s.stop] for k, s in env.observation_layout.items()},
            "reward_terms": cfg.env.reward.active(),
        },
    )

    obs, _ = env.reset(seed=cfg.run.seed)
    ep_return, ep_length = 0.0, 0
    returns: deque[float] = deque(maxlen=50)
    successes: deque[float] = deque(maxlen=50)
    lengths: deque[float] = deque(maxlen=50)
    last_metrics: dict[str, float] = {}
    pending_updates = 0.0
    best_success = -1.0
    t_start = time.time()

    for step in range(1, cfg.train.total_steps + 1):
        warming_up = step <= cfg.train.warmup_steps
        if warming_up:
            action = rng.uniform(-1.0, 1.0, action_dim).astype(np.float32)
        else:
            action = agent.act(policy_input(obs, goal_conditioned), _noise(cfg, step))

        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        ep_length += 1

        if goal_conditioned:
            buffer.add(
                obs["observation"],
                action,
                next_obs["observation"],
                obs["desired_goal"],
                next_obs["achieved_goal"],
            )
        else:
            buffer.add(obs, action, reward, next_obs, terminated)

        obs = next_obs

        if terminated or truncated:
            if goal_conditioned:
                buffer.end_episode()
            episode = info.get("episode", {})
            returns.append(ep_return)
            successes.append(float(episode.get("success", False)))
            lengths.append(ep_length)
            obs, _ = env.reset()
            ep_return, ep_length = 0.0, 0

        # -- learning ---------------------------------------------------- #
        pending_updates += cfg.train.updates_per_step
        if (
            not warming_up
            and step % cfg.train.update_every == 0
            and buffer.can_sample(cfg.algo.batch_size)
        ):
            n = int(pending_updates)
            pending_updates -= n
            critic_losses, actor_losses, qs = [], [], []
            for _ in range(n):
                metrics = agent.update(buffer.sample(cfg.algo.batch_size))
                critic_losses.append(metrics.critic_loss)
                qs.append(metrics.q_mean)
                if metrics.actor_loss is not None:
                    actor_losses.append(metrics.actor_loss)
            last_metrics = {
                "train/critic_loss": float(np.mean(critic_losses)),
                "train/q_mean": float(np.mean(qs)),
                "train/n_updates": float(agent.n_updates),
            }
            if actor_losses:
                last_metrics["train/actor_loss"] = float(np.mean(actor_losses))

        # -- logging ----------------------------------------------------- #
        if step % cfg.train.log_every == 0:
            metrics = {
                "rollout/return": float(np.mean(returns)) if returns else 0.0,
                "rollout/success_rate": float(np.mean(successes)) if successes else 0.0,
                "rollout/episode_length": float(np.mean(lengths)) if lengths else 0.0,
                "rollout/exploration_noise": _noise(cfg, step),
                "rollout/curriculum": env.goal_sampler.progress,
                "time/steps_per_second": step / max(1e-9, time.time() - t_start),
                **last_metrics,
            }
            logger.log(step, metrics, print_it=True)

        if step % cfg.train.eval_every == 0 and not warming_up:
            result, _, _ = evaluate(agent, eval_env, cfg.train.eval_episodes)
            logger.log(step, result.as_metrics(), print_it=True)
            if result.success_rate > best_success:
                best_success = result.success_rate
                agent.save(run_dir / "best.pt", step=step, success_rate=result.success_rate)

        if step % cfg.train.checkpoint_every == 0:
            agent.save(run_dir / "latest.pt", step=step)

    agent.save(run_dir / "final.pt", step=cfg.train.total_steps)
    final, _, _ = evaluate(agent, eval_env, max(cfg.train.eval_episodes, 20))
    logger.log(cfg.train.total_steps, final.as_metrics(), print_it=True)
    logger.save_json("final_eval.json", final.__dict__)
    logger.log_text(
        f"done in {(time.time() - t_start) / 60:.1f} min | "
        f"final success {final.success_rate:.0%} | best eval success {max(best_success, 0):.0%}"
    )
    logger.close()
    env.close()
    eval_env.close()
    return run_dir


def _noise(cfg: Config, step: int) -> float:
    """Linear anneal of exploration noise over the training run."""
    start, end = cfg.algo.exploration_noise, cfg.algo.exploration_noise_final
    span = max(1, cfg.train.total_steps - cfg.train.warmup_steps)
    frac = min(1.0, max(0.0, (step - cfg.train.warmup_steps) / span))
    return float(start + frac * (end - start))


def _run_name(cfg: Config) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{cfg.run.name}_{cfg.algo.name}_{cfg.env.backend}_{stamp}"
