#!/usr/bin/env python3
"""Plot task time or cost against control duration and planning time."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    REPO_ROOT / "results" / "initial_time_sensitivity" / "kinematic_car_gaussian" / "raw"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "initial_time_sensitivity"
METRICS = {
    "task_time": ("aura_time", "Task Time", "initial_time"),
    "cost": ("cost", "Cost", "initial_cost"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open average and median task-time surfaces for the initial-time study."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metric",
        choices=tuple(METRICS),
        default="task_time",
        help="quantity shown on the z axis (default: task_time)",
    )
    parser.add_argument(
        "--task-time-cap",
        type=float,
        default=120.0,
        help="cap each trial's task time before aggregation (default: 120 seconds)",
    )
    parser.add_argument(
        "--statistic",
        choices=("average", "median", "both"),
        default="both",
        help="surface to plot (default: both)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save the figures without opening interactive windows",
    )
    return parser.parse_args()


def load_values(
    results_dir: Path, metric: str, task_time_cap: float
) -> dict[tuple[float, float], list[float]]:
    if metric not in METRICS:
        raise ValueError(f"unsupported metric: {metric}")
    if metric == "task_time" and (
        not math.isfinite(task_time_cap) or task_time_cap <= 0.0
    ):
        raise ValueError("task-time cap must be finite and positive")
    column, label, _ = METRICS[metric]
    csv_paths = sorted(results_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no result CSVs found in {results_dir}")

    values: dict[tuple[float, float], list[float]] = defaultdict(list)
    total_rows = 0
    loaded_rows = 0
    capped_rows = 0
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                total_rows += 1
                try:
                    control_duration = float(row["control_duration"])
                    planning_time = float(row["planning_time"])
                    value = float(row[column])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"invalid {label.lower()} row in {path}") from error
                if not all(
                    math.isfinite(item)
                    for item in (control_duration, planning_time, value)
                ):
                    continue
                if metric == "task_time" and value > task_time_cap:
                    value = task_time_cap
                    capped_rows += 1
                values[(control_duration, planning_time)].append(value)
                loaded_rows += 1

    if not values:
        raise RuntimeError(f"no finite {label.lower()} rows found in {results_dir}")
    message = (
        f"[initial-time] Loaded {loaded_rows}/{total_rows} {label.lower()} "
        f"observations from {len(csv_paths)} CSV files"
    )
    if metric == "task_time":
        message += f"; capped {capped_rows} at {task_time_cap:g}s"
    print(message + ".")
    return values


def aggregate(
    values: dict[tuple[float, float], list[float]], statistic: str
) -> tuple[list[float], list[float], np.ndarray]:
    durations = sorted({key[0] for key in values})
    planning_times = sorted({key[1] for key in values})
    surface = np.full((len(planning_times), len(durations)), np.nan)

    reducer = {"average": np.mean, "median": np.median}[statistic]
    for y_index, planning_time in enumerate(planning_times):
        for x_index, duration in enumerate(durations):
            samples = values.get((duration, planning_time), [])
            if samples:
                surface[y_index, x_index] = float(reducer(samples))

    missing = np.argwhere(~np.isfinite(surface))
    if missing.size:
        cells = ", ".join(
            f"({durations[x]:g}, {planning_times[y]:g})" for y, x in missing
        )
        print(f"[initial-time] No observations at {cells}; plotting those cells as missing.")
    return durations, planning_times, surface


def format_tick(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def make_surface(
    pyplot,
    durations: list[float],
    planning_times: list[float],
    task_times: np.ndarray,
    statistic: str,
    metric_label: str,
):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    blue = "#174A6E"
    green = "#9BC56F"
    orange = "#F36C0A"
    colormap = LinearSegmentedColormap.from_list("aura", [blue, green, orange])
    x_grid, y_grid = np.meshgrid(durations, planning_times)
    finite = np.isfinite(task_times)
    if not np.any(finite):
        raise RuntimeError("the selected surface has no finite observations")
    minimum = float(np.nanmin(task_times))
    maximum = float(np.nanmax(task_times))
    span = max(maximum - minimum, 1.0)
    floor = max(0.0, minimum - 0.10 * span)
    ceiling = maximum + 0.08 * span
    normalizer = Normalize(vmin=minimum, vmax=maximum)

    figure = pyplot.figure(figsize=(7.2, 4.8), facecolor="white")
    try:
        figure.canvas.manager.set_window_title(
            f"{statistic.title()} {metric_label}"
        )
    except AttributeError:
        pass
    axes = figure.add_subplot(111, projection="3d")
    masked_times = np.ma.masked_invalid(task_times)
    axes.plot_surface(
        x_grid,
        y_grid,
        masked_times,
        cmap=colormap,
        norm=normalizer,
        linewidth=0.55,
        edgecolor=(1.0, 1.0, 1.0, 0.72),
        antialiased=True,
        alpha=0.95,
    )
    axes.contourf(
        x_grid,
        y_grid,
        masked_times,
        zdir="z",
        offset=floor,
        cmap=colormap,
        norm=normalizer,
        levels=14,
        alpha=0.20,
    )
    axes.scatter(
        x_grid[finite],
        y_grid[finite],
        task_times[finite],
        c=task_times[finite],
        cmap=colormap,
        norm=normalizer,
        s=30,
        edgecolor="white",
        linewidth=0.7,
        depthshade=False,
    )

    missing = ~finite
    if np.any(missing):
        axes.scatter(
            x_grid[missing],
            y_grid[missing],
            np.full(int(np.count_nonzero(missing)), floor),
            marker="x",
            color="#8A96A3",
            s=28,
            depthshade=False,
        )

    min_index = np.unravel_index(int(np.nanargmin(task_times)), task_times.shape)
    max_index = np.unravel_index(int(np.nanargmax(task_times)), task_times.shape)
    guide_x = min(durations)
    guide_y = max(planning_times)
    axes.plot(
        [guide_x, guide_x],
        [guide_y, guide_y],
        [minimum, maximum],
        color="#263849",
        linewidth=1.0,
        alpha=0.45,
    )
    for index, color in ((min_index, blue), (max_index, orange)):
        y_index, x_index = index
        value = float(task_times[index])
        axes.plot(
            [durations[x_index], guide_x],
            [planning_times[y_index], guide_y],
            [value, value],
            color=color,
            linestyle=(0, (5, 4)),
            linewidth=1.5,
        )

    axes.set_title(f"{statistic.title()} {metric_label}", pad=8)
    axes.set_xlabel("Maximum Control Duration", labelpad=9)
    axes.set_ylabel("Offline Planning Time", labelpad=11)
    axes.set_zlabel(metric_label, labelpad=8)
    axes.set_xticks(durations)
    axes.set_xticklabels([format_tick(value) for value in durations])
    axes.set_yticks(planning_times)
    axes.set_yticklabels([format_tick(value) for value in planning_times])
    axes.zaxis.set_major_locator(
        MaxNLocator(nbins=5, integer=True, min_n_ticks=4)
    )
    axes.zaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    figure.text(
        0.015,
        0.97,
        f"max {maximum:.0f}",
        ha="left",
        va="top",
        fontsize=11,
    )
    figure.text(
        0.015,
        0.03,
        f"min {minimum:.0f}",
        ha="left",
        va="bottom",
        fontsize=11,
    )
    axes.set_zlim(math.floor(floor), math.ceil(ceiling))
    axes.view_init(elev=27, azim=-128)
    axes.set_box_aspect((1.10, 1.0, 0.72))
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.pane.set_facecolor("#F4F7FA")
        axis.pane.set_edgecolor("#C8D0DA")
        axis.pane.set_alpha(0.62)
        axis._axinfo["grid"]["color"] = "#D7DEE7"
        axis._axinfo["grid"]["linewidth"] = 0.7
    axes.tick_params(pad=2)
    figure.subplots_adjust(left=0.01, right=0.98, top=0.91, bottom=0.04)
    return figure, minimum, maximum, min_index, max_index


def main() -> None:
    args = parse_args()
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as pyplot

    pyplot.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )
    _, metric_label, output_prefix = METRICS[args.metric]
    values = load_values(
        args.results_dir.resolve(), args.metric, args.task_time_cap
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    statistics = ("average", "median") if args.statistic == "both" else (args.statistic,)
    for statistic in statistics:
        durations, planning_times, task_times = aggregate(values, statistic)
        figure, minimum, maximum, min_index, max_index = make_surface(
            pyplot,
            durations,
            planning_times,
            task_times,
            statistic,
            metric_label,
        )
        output = args.output_dir.resolve() / f"{output_prefix}_{statistic}.png"
        figure.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.4)
        figures.append(figure)
        print(
            f"[initial-time] {statistic}: min={minimum:.2f} at "
            f"(duration={durations[min_index[1]]:g}, planning={planning_times[min_index[0]]:g}); "
            f"max={maximum:.2f} at "
            f"(duration={durations[max_index[1]]:g}, planning={planning_times[max_index[0]]:g})"
        )
        print(f"[initial-time] Saved {output}")

    if args.no_show:
        for figure in figures:
            pyplot.close(figure)
    else:
        label = "Both interactive figures are" if len(figures) == 2 else "The figure is"
        print(f"[initial-time] {label} open; close the window(s) to exit.")
        pyplot.show(block=True)


if __name__ == "__main__":
    main()
