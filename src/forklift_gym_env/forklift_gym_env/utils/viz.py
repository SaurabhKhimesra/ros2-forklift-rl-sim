"""Figures: learning curves, trajectory maps and rollout animations.

One shared style so every figure the project emits looks like it came from the
same place. The categorical hues are the first three slots of a palette that has
been checked for colour-vision-deficiency separation (worst all-pairs deutan
Delta-E 9.2, normal-vision 24.0), and every series is directly labelled rather
than relying on colour alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2de"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua
GOAL = "#eb6834"
PATH = "#2a78d6"

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 9.5,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
}


def use_style() -> None:
    plt.rcParams.update(_RC)


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that keeps the array length."""
    if window <= 1 or y.size < 3:
        return y
    window = min(window, max(3, y.size // 3))
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_learning_curves(
    runs: Mapping[str, object],
    out_path: str | Path,
    smooth_window: int = 5,
    title: str = "Training progress",
) -> Path:
    """Small multiples of the three metrics that matter.

    Deliberately three separate axes rather than one chart with two y-scales:
    success rate, return and final distance have unrelated units, and a dual
    axis would let the reader infer a relationship the data does not support.
    """
    import pandas as pd

    use_style()
    panels = [
        ("eval/success_rate", "Success rate", "{:.0%}"),
        ("eval/return", "Episode return", "{:.0f}"),
        ("eval/final_distance", "Final distance to pallet (m)", "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))

    for ax, (column, label, fmt) in zip(axes, panels, strict=True):
        endpoints: list[tuple[float, str, str]] = []
        for i, (name, frame) in enumerate(runs.items()):
            frame = pd.DataFrame(frame)
            if column not in frame:
                continue
            sub = frame[["step", column]].dropna()
            if sub.empty:
                continue
            x = sub["step"].to_numpy()
            y = _smooth(sub[column].to_numpy(dtype=float), smooth_window)
            colour = SERIES[i % len(SERIES)]
            ax.plot(x, y, color=colour, label=name, zorder=3)
            endpoints.append((float(y[-1]), f"{name}  {fmt.format(y[-1])}", colour))
        # Direct labels at the end of each line, so identity never depends on
        # colour alone (the third hue also carries a contrast warning against a
        # light surface, which makes labels mandatory rather than optional).
        # Converged runs finish on top of each other, so nudge the labels apart
        # in display space rather than letting them overprint.
        _place_end_labels(ax, endpoints)
        ax.set_title(label)
        ax.set_xlabel("environment steps")
        ax.grid(axis="y", zorder=0)
        if column == "eval/success_rate":
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    if len(runs) > 1:
        axes[0].legend(loc="upper left")
    fig.suptitle(title, x=0.008, ha="left", fontsize=13, fontweight="700", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_path)


def plot_trajectories(
    trajectories: Sequence[np.ndarray],
    goals: Sequence[tuple[float, float]],
    out_path: str | Path,
    tolerance: float = 0.6,
    title: str = "Evaluation rollouts",
    subtitle: str = "",
) -> Path:
    """Top-down map of where the forklift actually drove."""
    use_style()
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    for path, goal in zip(trajectories, goals, strict=True):
        path = np.asarray(path)
        ax.plot(path[:, 0], path[:, 1], color=PATH, alpha=0.55, linewidth=1.6, zorder=2)
        ax.plot(path[0, 0], path[0, 1], "o", color=INK, markersize=4, zorder=4)
        ax.add_patch(
            Circle(
                goal, tolerance, facecolor=GOAL, alpha=0.16, edgecolor=GOAL, linewidth=1.2, zorder=1
            )
        )
        ax.plot(*goal, marker="*", color=GOAL, markersize=11, zorder=5)

    ax.set_aspect("equal")
    ax.grid(zorder=0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, pad=24 if subtitle else 10, loc="left")
    if subtitle:
        ax.text(
            0.0, 1.015, subtitle, transform=ax.transAxes, color=INK_MUTED, fontsize=9, va="bottom"
        )
    # Legend by proxy: three distinct marks, so identity is shape, not colour.
    handles = [
        plt.Line2D([], [], color=INK, marker="o", linestyle="", markersize=5, label="start"),
        plt.Line2D([], [], color=PATH, linewidth=2, label="path driven"),
        plt.Line2D(
            [], [], color=GOAL, marker="*", linestyle="", markersize=10, label="pallet (+tolerance)"
        ),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.09), ncol=3)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_policy_return_comparison(
    results: Mapping[str, Mapping[str, float]],
    out_path: str | Path,
    title: str = "What each reward weighting pays for",
    subtitle: str = "",
) -> Path:
    """Grouped bars: the return earned by each hand-written policy, per weighting.

    Used to show, without training anything, whether a reward function pays a
    do-nothing policy. A positive bar for "do nothing" is a local optimum the
    agent can settle into before it has ever reached the goal.
    """
    use_style()
    weightings = list(results)
    policies = list(next(iter(results.values())))
    n = len(policies)
    x = np.arange(len(weightings))
    width = 0.78 / n

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, policy in enumerate(policies):
        values = [results[w][policy] for w in weightings]
        offset = (i - (n - 1) / 2) * width
        ax.bar(
            x + offset, values, width * 0.92, color=SERIES[i % len(SERIES)], zorder=3, label=policy
        )
        for xi, v in zip(x + offset, values, strict=True):
            ax.annotate(
                f"{v:+.0f}",
                xy=(xi, v),
                xytext=(0, 4 if v >= 0 else -12),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="600",
                color=SERIES[i % len(SERIES)],
            )

    ax.axhline(0, color=INK_MUTED, linewidth=1.1, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(weightings)
    ax.set_ylabel("mean episode return")
    ax.grid(axis="y", zorder=0)
    ax.margins(y=0.18)
    ax.legend(loc="upper left", ncol=n)
    ax.set_title(title, pad=24 if subtitle else 10, loc="left")
    if subtitle:
        ax.text(
            0.0, 1.015, subtitle, transform=ax.transAxes, color=INK_MUTED, fontsize=9, va="bottom"
        )
    fig.tight_layout()
    return _save(fig, out_path)


def plot_reward_breakdown(
    breakdowns: Iterable[Mapping[str, float]],
    out_path: str | Path,
    title: str = "Where the reward comes from",
) -> Path:
    """Mean contribution of each reward term over an episode.

    Turns "the return went down" into "the smoothness penalty went up", which is
    the whole reason the terms are tracked separately.
    """
    use_style()
    rows = list(breakdowns)
    keys = sorted({k for row in rows for k in row})
    means = [float(np.mean([row.get(k, 0.0) for row in rows])) for k in keys]
    order = np.argsort(means)
    keys = [keys[i] for i in order]
    means = [means[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.8, 0.5 * len(keys) + 1.6))
    colours = [SERIES[0] if v >= 0 else SERIES[1] for v in means]
    ax.barh(keys, means, color=colours, height=0.6, zorder=3)
    for key, value in zip(keys, means, strict=True):
        offset = 4 if value >= 0 else -4
        ax.annotate(
            f"{value:+.3f}",
            xy=(value, key),
            xytext=(offset, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
            color=INK_MUTED,
        )
    ax.axvline(0, color=GRID, linewidth=1)
    ax.grid(axis="x", zorder=0)
    ax.set_xlabel("mean reward per step")
    ax.set_title(title)
    ax.margins(x=0.22)
    fig.tight_layout()
    return _save(fig, out_path)


def animate_rollout(
    trajectory: np.ndarray,
    goal: tuple[float, float],
    out_path: str | Path,
    tolerance: float = 0.6,
    fps: int = 15,
    trail: int = 40,
) -> Path:
    """Render a rollout to an animated GIF."""
    import imageio.v2 as imageio

    use_style()
    trajectory = np.asarray(trajectory)
    pad = 1.5
    xs = np.concatenate([trajectory[:, 0], [goal[0]]])
    ys = np.concatenate([trajectory[:, 1], [goal[1]]])
    bounds = (xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad)

    frames = []
    step = max(1, len(trajectory) // 120)  # cap the GIF at ~120 frames
    for end in range(1, len(trajectory) + 1, step):
        fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=90)
        ax.add_patch(Circle(goal, tolerance, facecolor=GOAL, alpha=0.16, edgecolor=GOAL, zorder=1))
        ax.plot(*goal, marker="*", color=GOAL, markersize=14, zorder=4)
        start = max(0, end - trail)
        ax.plot(
            trajectory[:end, 0],
            trajectory[:end, 1],
            color=PATH,
            alpha=0.25,
            linewidth=1.4,
            zorder=2,
        )
        ax.plot(
            trajectory[start:end, 0], trajectory[start:end, 1], color=PATH, linewidth=2.4, zorder=3
        )
        ax.plot(
            trajectory[end - 1, 0], trajectory[end - 1, 1], "o", color=PATH, markersize=8, zorder=5
        )
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.grid(zorder=0)
        ax.set_title(
            f"step {end}   distance {np.hypot(*(trajectory[end - 1] - np.array(goal))):.2f} m"
        )
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return out_path


def _place_end_labels(
    ax, endpoints: list[tuple[float, str, str]], min_gap_pt: float = 13.0
) -> None:
    """Annotate each series at its right-hand end, spaced so labels never overlap.

    Converged runs finish on top of each other, so the labels are nudged apart in
    display space rather than being allowed to overprint.
    """
    if not endpoints:
        return
    ax.margins(x=0.24)
    x_right = ax.get_xlim()[1]
    ordered = sorted(endpoints, key=lambda e: e[0])

    fig_height_pt = ax.get_figure().get_size_inches()[1] * 72
    axis_span = max(1e-9, ax.get_ylim()[1] - ax.get_ylim()[0])
    pt_per_data = (ax.get_position().height * fig_height_pt) / axis_span

    placed_pt: list[float] = []
    for value, _, _ in ordered:
        want = value * pt_per_data
        if placed_pt and want - placed_pt[-1] < min_gap_pt:
            want = placed_pt[-1] + min_gap_pt
        placed_pt.append(want)

    for (value, text, colour), target_pt in zip(ordered, placed_pt, strict=True):
        ax.annotate(
            text,
            xy=(x_right, value),
            xytext=(6, target_pt - value * pt_per_data),
            textcoords="offset points",
            color=colour,
            fontsize=8.5,
            fontweight="600",
            va="center",
            annotation_clip=False,
        )


def _save(fig: plt.Figure, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path
