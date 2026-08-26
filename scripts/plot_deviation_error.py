#!/usr/bin/env python3
"""Plot AURA, MPPI, and open-loop fixed-reference deviation statistics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.deviation import METHOD_LABELS, METHOD_ORDER


sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

METHOD_COLORS = {
    "aura": "#EB5B00",
    "mppi": "#9B59B6",
    "open_loop": "#A0C878",
}
METHOD_MARKERS = {"aura": "o", "mppi": "s", "open_loop": "^"}
SYSTEM_TITLES = {
    "double_integrator": "Double Integrator",
    "kinematic_car": "Kinematic Car",
    "pushing_object": "Learned Pushing Dynamics",
}
ENVIRONMENT_TITLES = {
    "gaussian": "Gaussian Noise",
    "mujoco": "MuJoCo Simulation",
}


def apply_condition_title(axis, system: str, environment: str) -> None:
    axis.text(
        0.5,
        1.075,
        SYSTEM_TITLES.get(system, system.replace("_", " ").title()),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        clip_on=False,
    )
    axis.text(
        0.5,
        1.025,
        ENVIRONMENT_TITLES.get(environment, environment.replace("_", " ").title()),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontstyle="italic",
        clip_on=False,
    )


def render_tracking_plot(
    statistics: list[dict],
    output_dir: Path,
    *,
    cumulative: bool,
    stem: str | None = None,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = sorted(
        {(str(row["system"]), str(row["environment"])) for row in statistics}
    )
    if not conditions:
        raise ValueError("cannot plot an empty tracking-statistics dataset")
    figure, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(3.25 * len(conditions), 3.7),
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    for axis, (system, environment) in zip(flat_axes, conditions):
        for method in METHOD_ORDER:
            rows = sorted(
                (
                    row
                    for row in statistics
                    if row["system"] == system
                    and row["environment"] == environment
                    and row["method"] == method
                ),
                key=lambda row: int(row["step"]),
            )
            if not rows:
                continue
            prefix = "cumulative_" if cumulative else ""
            x = np.asarray([int(row["step"]) for row in rows])
            mean = np.asarray([float(row[f"{prefix}mean"]) for row in rows])
            low = np.asarray([float(row[f"{prefix}ci_low"]) for row in rows])
            high = np.asarray([float(row[f"{prefix}ci_high"]) for row in rows])
            axis.plot(
                x,
                mean,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=4.0,
                linewidth=1.6,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(
                x, low, high, color=METHOD_COLORS[method], alpha=0.16, linewidth=0
            )
        apply_condition_title(axis, system, environment)
        steps = sorted(
            {
                int(row["step"])
                for row in statistics
                if row["system"] == system and row["environment"] == environment
            }
        )
        axis.set_xticks(steps)
        axis.set_xlabel("Number of Controls Applied", fontsize=11, labelpad=5)
        axis.grid(True, axis="y", alpha=0.4, linewidth=0.6)
        axis.grid(False, axis="x")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.supylabel(
        "Mean Cumulative Tracking Error" if cumulative else "Mean Tracking Error",
        fontsize=12,
        x=0.004,
    )
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=True,
    )
    figure.subplots_adjust(top=0.82, bottom=0.27, left=0.055, right=0.99, wspace=0.32)
    name = stem or (
        "cumulative_tracking_error" if cumulative else "tracking_error_by_step"
    )
    paths = tuple(output_dir / f"{name}.{suffix}" for suffix in ("pdf", "png", "svg"))
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=300, bbox_inches="tight")
    figure.savefig(paths[2], bbox_inches="tight")
    plt.close(figure)
    return paths


def plot_tracking_statistics(
    step_statistics: list[dict], output_dir: Path
) -> dict[str, Path]:
    instant = render_tracking_plot(step_statistics, output_dir, cumulative=False)
    cumulative = render_tracking_plot(step_statistics, output_dir, cumulative=True)
    paths = {
        f"tracking_error_by_step_{suffix}": path
        for suffix, path in zip(("pdf", "png", "svg"), instant)
    }
    paths.update(
        {
            f"cumulative_tracking_error_{suffix}": path
            for suffix, path in zip(("pdf", "png", "svg"), cumulative)
        }
    )
    conditions = sorted(
        {(str(row["system"]), str(row["environment"])) for row in step_statistics}
    )
    for system, environment in conditions:
        rows = [
            row
            for row in step_statistics
            if row["system"] == system and row["environment"] == environment
        ]
        name = f"tracking_error_by_step__{system}__{environment}"
        rendered = render_tracking_plot(
            rows, output_dir, cumulative=False, stem=name
        )
        paths.update(
            {
                f"{name}_{suffix}": path
                for suffix, path in zip(("pdf", "png", "svg"), rendered)
            }
        )
    return paths
