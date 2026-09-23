"""Command-line interface.

Runnable three ways -- ``python -m forklift_gym_env``, the installed
``forklift`` script, or ``ros2 run forklift_gym_env forklift``. The first of
those is the one that matters: it means the kinematic backend can be debugged
and profiled with ordinary Python tooling, outside a sourced ROS shell.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from forklift_gym_env import __version__


def _parse_set(pairs: Sequence[str]) -> dict[str, Any]:
    """Parse ``--set a.b=1 --set c=hello`` into a dict of dotted overrides."""
    import yaml

    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = yaml.safe_load(value)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forklift",
        description="Deep RL forklift simulation (ROS 2 / Gazebo + a fast surrogate sim).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--config", default="td3_kinematic.yaml", help="config name or path")
        p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")

    train_p = sub.add_parser("train", help="train an agent")
    common(train_p)
    train_p.add_argument("--run-dir", default=None)
    train_p.add_argument("--quiet", action="store_true")

    eval_p = sub.add_parser("eval", help="evaluate a checkpoint")
    common(eval_p)
    eval_p.add_argument("checkpoint", type=Path)
    eval_p.add_argument("-n", "--episodes", type=int, default=20)
    eval_p.add_argument("--plot", type=Path, default=None, help="write a trajectory figure here")
    eval_p.add_argument("--gif", type=Path, default=None, help="write a rollout GIF here")

    report_p = sub.add_parser("report", help="plot learning curves from finished runs")
    report_p.add_argument(
        "runs", nargs="+", type=Path, help="run directories (or metrics.csv files)"
    )
    report_p.add_argument("-o", "--out", type=Path, default=Path("learning_curves.png"))
    report_p.add_argument("--labels", nargs="*", default=None)
    report_p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="moving-average window over eval points (1 = raw; eval is already "
        "averaged over eval_episodes, so smoothing here mostly hides variance)",
    )
    report_p.add_argument("--title", default="Training progress")

    conf_p = sub.add_parser("config", help="resolve and print a validated config")
    common(conf_p)

    sub.add_parser("list-configs", help="show the configs this install can find")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list-configs":
        from forklift_gym_env.paths import search_paths

        for root in search_paths():
            found = sorted((root / "config").glob("*.yaml"))
            if found:
                print(f"{root}/config")
                for f in found:
                    print(f"  {f.name}")
        return 0

    if args.command == "report":
        return _report(args)

    from forklift_gym_env.config import load_config
    from forklift_gym_env.paths import find_config

    cfg = load_config(find_config(args.config), _parse_set(args.overrides))

    if args.command == "config":
        import yaml

        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        return 0

    if args.command == "train":
        from forklift_gym_env.rl.train import train

        run_dir = train(cfg, args.run_dir, quiet=args.quiet)
        print(f"\nrun directory: {run_dir}")
        return 0

    if args.command == "eval":
        return _eval(args, cfg)

    raise SystemExit(f"unhandled command {args.command!r}")


def _eval(args: argparse.Namespace, cfg: Any) -> int:
    import numpy as np

    from forklift_gym_env.rl.td3 import TD3Agent, resolve_device
    from forklift_gym_env.rl.train import evaluate, make_env

    env = make_env(cfg)
    goal_dim = 2 if env.goal_conditioned else 0
    obs_dim = int(
        np.prod(
            env.observation_space["observation"].shape
            if env.goal_conditioned
            else env.observation_space.shape
        )
    )
    agent = TD3Agent.load(
        args.checkpoint,
        cfg.algo,
        obs_dim + goal_dim,
        int(env.action_space.shape[0]),
        resolve_device(cfg.run.device),
    )

    result, trajectories, goals = evaluate(
        agent, env, args.episodes, collect_trajectories=bool(args.plot or args.gif)
    )
    print(
        f"success {result.success_rate:.0%} | return {result.mean_return:+.1f} | "
        f"length {result.mean_length:.0f} | final distance {result.mean_final_distance:.2f} m | "
        f"collisions {result.collision_rate:.0%}"
    )

    if args.plot:
        from forklift_gym_env.utils.viz import plot_trajectories

        path = plot_trajectories(
            trajectories,
            goals,
            args.plot,
            cfg.env.goal.position_tolerance,
            subtitle=f"{args.episodes} episodes, randomised pallet positions",
        )
        print(f"wrote {path}")

    if args.gif:
        from forklift_gym_env.utils.viz import animate_rollout

        path = animate_rollout(trajectories[0], goals[0], args.gif, cfg.env.goal.position_tolerance)
        print(f"wrote {path}")

    env.close()
    return 0


def _report(args: argparse.Namespace) -> int:
    import pandas as pd

    from forklift_gym_env.utils.viz import plot_learning_curves

    frames = {}
    for i, path in enumerate(args.runs):
        csv = path if path.suffix == ".csv" else path / "metrics.csv"
        if not csv.is_file():
            raise SystemExit(f"no metrics.csv at {csv}")
        label = args.labels[i] if args.labels and i < len(args.labels) else csv.parent.name
        frames[label] = pd.read_csv(csv)
    out = plot_learning_curves(frames, args.out, smooth_window=args.smooth, title=args.title)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
