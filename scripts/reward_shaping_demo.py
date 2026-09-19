#!/usr/bin/env python3
"""Measure the idling trap in potential-based reward shaping.

Potential-based shaping adds ``gamma*Phi(s') - Phi(s)``. With ``Phi = -distance``
and a **stationary** robot that is not zero -- it is ``(1 - gamma) * distance``,
which is *positive*. A far-away agent therefore earns a small income every step
for doing nothing.

This script quantifies that, in two ways:

1. **Without training anything**, it rolls out two hand-written policies -- "do
   nothing" and a proportional controller that drives to the pallet -- under two
   reward weightings, and reports the return each earns. If "do nothing" scores
   positively, the reward function contains a local optimum that pays, and an
   agent that has not yet found the goal can settle into it.

2. **Optionally with training** (``--steps N``), it runs the same TD3 agent under
   both weightings. Be warned that at this task's scale the training comparison
   came out *inconclusive* -- both weightings converge to 100% and both dip
   partway through -- so the bar chart above, which is exact and needs no
   training, is the result worth quoting. The training half is here so anyone
   can check that for themselves rather than taking it on trust.

The broken weighting is what ``RewardConfig.check_idling_is_unprofitable`` now
rejects, so this script has to ask for it explicitly.

    python scripts/reward_shaping_demo.py --steps 25000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "forklift_gym_env"))

from forklift_gym_env.config import load_config  # noqa: E402
from forklift_gym_env.envs import ForkliftEnv  # noqa: E402
from forklift_gym_env.geometry import Pose2D  # noqa: E402
from forklift_gym_env.paths import find_config  # noqa: E402
from forklift_gym_env.rl.train import train  # noqa: E402

WEIGHTINGS = {
    "broken\ngamma 0.98, time -0.01": {"algo.gamma": 0.98, "env.reward.time": -0.01},
    "fixed\ngamma 0.99, time -0.12": {"algo.gamma": 0.99, "env.reward.time": -0.12},
}
SHORT = {k: k.split("\n")[0] for k in WEIGHTINGS}


def do_nothing(_env: ForkliftEnv) -> np.ndarray:
    return np.zeros(2, np.float32)


def drive_to_pallet(env: ForkliftEnv) -> np.ndarray:
    """A plain proportional controller: turn towards the goal and drive."""
    bearing = env.state.heading_error
    return np.array(
        [np.clip(0.9 * np.cos(bearing), -1, 1), np.clip(2.0 * bearing, -1, 1)], np.float32
    )


POLICIES = {"do nothing": do_nothing, "drive to the pallet": drive_to_pallet}


def build_config(overrides: dict, **extra):
    return load_config(
        find_config("td3_kinematic.yaml"),
        {
            **overrides,
            **extra,
            "env.goal.curriculum.enabled": False,
            "run.seed": 42,
        },
        check_reward_shaping=False,
    )


def measure_returns(episodes: int = 5) -> dict[str, dict[str, float]]:
    """Return {weighting: {policy: mean return}} with the simulator made deterministic."""
    out: dict[str, dict[str, float]] = {}
    for label, overrides in WEIGHTINGS.items():
        cfg = build_config(overrides, **{"env.kinematic.velocity_noise_std": 0.0})
        env = ForkliftEnv(cfg)
        scores: dict[str, float] = {}
        for name, policy in POLICIES.items():
            totals = []
            for seed in range(episodes):
                env.reset(seed=seed, options={"start": Pose2D(0, 0, 0), "goal": Pose2D(7.0, 3.0)})
                total, done = 0.0, False
                while not done:
                    _, reward, terminated, truncated, _ = env.step(policy(env))
                    total += reward
                    done = terminated or truncated
                totals.append(total)
            scores[name] = float(np.mean(totals))
        env.close()
        out[label] = scores
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="also train under both weightings for this many steps (0 = skip; "
        "see the module docstring on why this comparison is inconclusive)",
    )
    parser.add_argument("--figures", type=Path, default=REPO / "docs/figures")
    parser.add_argument("--run-root", type=Path, default=REPO / "runs/reward_shaping_demo")
    args = parser.parse_args()

    from forklift_gym_env.utils.viz import plot_learning_curves, plot_policy_return_comparison

    print("=" * 78)
    print("Returns earned by hand-written policies (no learning involved)")
    print("=" * 78)
    results = measure_returns()
    for weighting, scores in results.items():
        flat = weighting.replace("\n", "  ")
        idle, drive = scores["do nothing"], scores["drive to the pallet"]
        verdict = "idling PAYS" if idle > 0 else "idling costs"
        print(f"  {flat:34s} do nothing {idle:+8.2f}   drive {drive:+8.2f}   -> {verdict}")

    out = plot_policy_return_comparison(
        results,
        args.figures / "reward_shaping.png",
        title="Potential-based shaping can pay an agent to stand still",
        subtitle="Mean return over 5 fixed episodes, pallet at (7, 3). No learning involved.",
    )
    print(f"\nwrote {out}")

    if args.steps > 0:
        import pandas as pd

        frames = {}
        for i, (label, overrides) in enumerate(WEIGHTINGS.items()):
            cfg = build_config(
                overrides,
                **{
                    "train.total_steps": args.steps,
                    "train.eval_every": max(1000, args.steps // 20),
                    "train.eval_episodes": 20,
                    "train.log_every": args.steps,
                },
            )
            run_dir = args.run_root / ("broken" if i == 0 else "fixed")
            print(f"\n=== training: {label.replace(chr(10), '  ')} ===")
            train(cfg, run_dir)
            frames[SHORT[label]] = pd.read_csv(run_dir / "metrics.csv")

        out = plot_learning_curves(
            frames,
            args.figures / "reward_shaping_training.png",
            smooth_window=1,
            title="Same agent, same hyperparameters, two reward weightings",
        )
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
