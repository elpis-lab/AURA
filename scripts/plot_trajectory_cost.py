#!/usr/bin/env python3
"""Plot the available noise-free trajectory-cost comparison results.

Four systems (double_integrator, kinematic_car, pushing_object,
dubins_airplane), each with three planners (AORRT/AOEST/SSTStar) swept
across five planning-time budgets, both before ("initial", a single bounded
solve, "Vanilla-X") and after ("final", AURA's repeated replan()
refinement, "AURA-X"). Single row, mean only, no CI shading, no MPPI
reference line -- MPPI's own numbers are still computed and written to the
summary CSV/JSON, just not drawn on this chart.

The figure uses the paper's established colors, typography, and broken-axis
layout: Times New Roman, compact paper-style font sizes, the same
sststar/aoest/aorrt hex colors, and a broken y-axis on the
double_integrator panel (AORRT's cost sits far below AOEST/SSTStar's on
that system, so both bands would be unreadable sharing one linear axis).

The data is one self-consistent dataset for all four systems; see
``experiment/trajectory_cost.py`` for collection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]

PLANNERS = ("aorrt", "aoest", "sststar")
SYSTEMS = ("double_integrator", "kinematic_car", "pushing_object", "dubins_airplane")
SYSTEM_TITLES = {
    "double_integrator": "6D Double Integrator",
    "kinematic_car": "Kinematic Car",
    "pushing_object": "Learned Pushing Dynamics",
    "dubins_airplane": "6D Dubins Airplane",
}

# ---------------------------------------------------------------------------
# Shared paper theme.
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

# Planner colors used throughout the paper figures.
PLANNER_COLORS = {
    "sststar": "#EB5B00",
    "aoest": "#143D60",
    "aorrt": "#A0C878",
}
PLANNER_DISPLAY_NAMES = {
    "sststar": "SST",
    "aoest": "AOEST",
    "aorrt": "AORRT",
}

LINE_WIDTH = 1.6
MARKER_SIZE = 3.5
INITIAL_LINE_ALPHA = 0.5
GRID_ALPHA = 0.3
SHOW_TOP_SPINE = False
SHOW_RIGHT_SPINE = False
Y_TICK_COUNT = 6  # same number of y-axis tick labels on every panel
Y_TICK_PRECISION = 0.2  # tick step is always a multiple of this

# double_integrator's AORRT cost sits far below AOEST's; a single linear
# axis makes one band unreadable, so that panel alone gets a broken y-axis
# Broken-axis bounds for the double-integrator panel.
# SSTStar was grouped with AOEST here back when its cost also sat around
# 7-8 (before the selection_radius fix dropped it to ~AORRT's range) --
# left in the high group, _padded_range's span would stretch from SST's now
# much-lower values up to AOEST's still-high ones, producing a "high" range
# wide enough to also contain AORRT's values, so AORRT and SST were each
# drawn (fully, per plot_system_axis) into BOTH the top and bottom panels
# and only visually clipped by ylim -- i.e. every curve is always plotted
# on both axes; only the y-limits below decide what's visible in each.
SPLIT_AXIS_SYSTEM = "double_integrator"
SPLIT_LOW_PLANNERS = ("aorrt", "sststar")
SPLIT_HIGH_PLANNERS = ("aoest",)

# AOEST's high band sits at 6.79-7.24 and AORRT/SST's low band at 2.46-3.90
# (checked directly against the 100-trial summary); pin both explicitly
# rather than auto-padding around whatever the current means happen to be.
SPLIT_HIGH_RANGE_OVERRIDE: tuple[float, float] | None = (6.5, 7.5)
SPLIT_LOW_RANGE_OVERRIDE: tuple[float, float] | None = (2.0, 4.0)

# Explicit (ymin, ymax) overrides for specific (non-split) panels, bypassing
# the auto-scaled nice-tick search below -- ticks are evenly spaced across
# exactly this range.
Y_AXIS_OVERRIDES: dict[str, tuple[float, float]] = {
    "kinematic_car": (4.8, 13.8),
    "pushing_object": (3.0, 6.0),
}


def _finite(values) -> np.ndarray:
    array = np.asarray([float(v) for v in values], dtype=float)
    return array[np.isfinite(array)]


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    finite = _finite(values)
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite))
    if len(finite) < 2:
        return mean, mean, mean
    half_width = float(1.96 * np.std(finite, ddof=1) / np.sqrt(len(finite)))
    return mean, mean - half_width, mean + half_width


def _read_planner_rows(cost_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for planner in PLANNERS:
        paths = sorted(cost_dir.glob(f"{planner}_*.csv"))
        for path in paths:
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    rows.append(row)
        print(f"[data] {cost_dir.name}/{planner}: {len(paths)} runs")
    return rows


def _read_mppi_rows(cost_dir: Path) -> list[dict]:
    path = cost_dir / "mppi_summary.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def aggregate_planner_curves(rows: list[dict]) -> dict:
    """(planner, planning_time) -> {initial: (mean,lo,hi), final: ..., initial_median, final_median, n_success, n_total}"""
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["planner"], float(row["planning_time"]))].append(row)
    curves: dict[tuple[str, float], dict] = {}
    for key, group in groups.items():
        successes = [r for r in group if r.get("status") == "success"]
        initial_values = _finite([r["initial_cost"] for r in successes])
        final_values = _finite([r["final_cost"] for r in successes])
        initial_mean, initial_lo, initial_hi = _mean_ci95(initial_values)
        final_mean, final_lo, final_hi = _mean_ci95(final_values)
        curves[key] = {
            "initial": (initial_mean, initial_lo, initial_hi),
            "final": (final_mean, final_lo, final_hi),
            "initial_median": float(np.median(initial_values)) if len(initial_values) else float("nan"),
            "final_median": float(np.median(final_values)) if len(final_values) else float("nan"),
            "n_success": len(successes),
            "n_total": len(group),
        }
    return curves


def aggregate_mppi(rows: list[dict]) -> dict:
    successes = [r for r in rows if r.get("status") == "success"]
    mean, lo, hi = _mean_ci95([r["cost"] for r in successes])
    return {
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "n_success": len(successes),
        "n_total": len(rows),
    }


def _apply_uniform_yticks(ax, count: int, precision: float = Y_TICK_PRECISION) -> None:
    """Force exactly `count` y-tick labels, spaced by a step that's a clean
    multiple of `precision` (so labels read like 6.2/6.4/6.6, never
    6.234/6.567) -- computed from this axis's own auto-scaled data range,
    with the range nudged outward to the nearest tick so the numbers land
    round instead of forcing count evenly-spaced points through the exact
    (arbitrary) data min/max. Guarantees the final [ticks[0], ticks[-1]]
    range still fully covers the original (auto-scaled, padded) range."""
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    raw_step = span / max(count - 1, 1)
    # Search increasing multiples of `precision` for a step where the
    # floor-aligned start still reaches y_max after (count-1) hops -- a
    # step just barely >= raw_step can leave too little slack for a
    # precision-aligned start to cover both ends at once.
    n_precision_units = max(math.ceil(raw_step / precision - 1e-9), 1)
    step = None
    start = None
    for _ in range(1000):
        candidate_step = n_precision_units * precision
        candidate_start = math.floor(y_min / candidate_step + 1e-9) * candidate_step
        if candidate_start + (count - 1) * candidate_step >= y_max - 1e-9:
            step, start = candidate_step, candidate_start
            break
        n_precision_units += 1
    ticks = [round(start + i * step, 10) for i in range(count)]
    ax.set_yticks(ticks)
    ax.set_ylim(ticks[0], ticks[-1])


def plot_system_axis(ax, planner_curves: dict, *, show_title: str | None, show_xlabel: bool) -> None:
    planning_times = sorted({pt for (_planner, pt) in planner_curves})
    for planner in PLANNERS:
        color = PLANNER_COLORS[planner]
        display_name = PLANNER_DISPLAY_NAMES[planner]
        for condition, style, marker, label_prefix, alpha in (
            ("initial", "--", "o", "Vanilla", INITIAL_LINE_ALPHA),
            ("final", "-", "s", "AURA", 1.0),
        ):
            xs, ys = [], []
            for pt in planning_times:
                entry = planner_curves.get((planner, pt))
                if entry is None:
                    continue
                value = entry[condition][0]
                if not math.isfinite(value):
                    continue
                xs.append(pt)
                ys.append(value)
            if not xs:
                continue
            # Straight segments through the actual data points -- no spline
            # smoothing (the original plot connects real measurements only).
            ax.plot(
                xs,
                ys,
                marker + style,
                color=color,
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                alpha=alpha,
                label=f"{label_prefix}-{display_name}",
            )

    ax.set_xticks(planning_times)
    if show_xlabel:
        ax.set_xlabel("Offline Planning Time", labelpad=2)
    if show_title:
        ax.set_title(show_title, fontsize=plt.rcParams["axes.titlesize"])
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=2)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(SHOW_TOP_SPINE)
    ax.spines["right"].set_visible(SHOW_RIGHT_SPINE)


def _collect_curve_values(planner_curves: dict, planners: tuple[str, ...]) -> list[float]:
    values = []
    for (planner, _pt), entry in planner_curves.items():
        if planner not in planners:
            continue
        for condition in ("initial", "final"):
            value = entry[condition][0]
            if math.isfinite(value):
                values.append(value)
    return values


def _padded_range(values: list[float], pad_frac: float = 0.15) -> tuple[float, float]:
    low, high = min(values), max(values)
    pad = (high - low) * pad_frac
    if pad <= 0:
        pad = max(abs(high), 1.0) * 0.1
    return low - pad, high + pad


def create_split_axes(fig, base_ax):
    """Replace base_ax with two stacked axes sharing an x-axis, with the
    classic diagonal break marks -- ported from
    the original paper layout."""
    pos = base_ax.get_position()
    base_ax.set_visible(False)

    gap = pos.height * 0.06
    top_h = pos.height * 0.5
    bottom_h = pos.height - top_h - gap

    ax_top = fig.add_axes((pos.x0, pos.y0 + bottom_h + gap, pos.width, top_h))
    ax_bottom = fig.add_axes((pos.x0, pos.y0, pos.width, bottom_h), sharex=ax_top)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)

    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    return ax_top, ax_bottom


def _align_split_axes_to_base(base_ax, ax_top, ax_bottom) -> None:
    """Keep split axes exactly inside the (possibly adjusted) base subplot
    box -- subplots_adjust() only repositions gridspec-managed axes, not
    the manually fig.add_axes()'d split ones, so this must run after it."""
    pos = base_ax.get_position()
    gap = pos.height * 0.06
    top_h = pos.height * 0.5
    bottom_h = pos.height - top_h - gap
    ax_top.set_position((pos.x0, pos.y0 + bottom_h + gap, pos.width, top_h))
    ax_bottom.set_position((pos.x0, pos.y0, pos.width, bottom_h))


def write_summary(
    all_curves: dict[str, dict], all_mppi: dict[str, dict], output_dir: Path
) -> tuple[Path, Path]:
    summary_rows = []
    for system_name in SYSTEMS:
        curves = all_curves.get(system_name, {})
        for (planner, planning_time), entry in sorted(curves.items()):
            summary_rows.append(
                {
                    "system": system_name,
                    "planner": planner,
                    "planning_time": planning_time,
                    "n_success": entry["n_success"],
                    "n_total": entry["n_total"],
                    "initial_cost_mean": entry["initial"][0],
                    "initial_cost_ci95_low": entry["initial"][1],
                    "initial_cost_ci95_high": entry["initial"][2],
                    "initial_cost_median": entry["initial_median"],
                    "final_cost_mean": entry["final"][0],
                    "final_cost_ci95_low": entry["final"][1],
                    "final_cost_ci95_high": entry["final"][2],
                    "final_cost_median": entry["final_median"],
                }
            )
        mppi = all_mppi.get(system_name)
        if mppi is not None:
            summary_rows.append(
                {
                    "system": system_name,
                    "planner": "mppi",
                    "planning_time": float("nan"),
                    "n_success": mppi["n_success"],
                    "n_total": mppi["n_total"],
                    "initial_cost_mean": float("nan"),
                    "initial_cost_ci95_low": float("nan"),
                    "initial_cost_ci95_high": float("nan"),
                    "initial_cost_median": float("nan"),
                    "final_cost_mean": mppi["mean"],
                    "final_cost_ci95_low": mppi["lo"],
                    "final_cost_ci95_high": mppi["hi"],
                    "final_cost_median": float("nan"),
                }
            )

    csv_path = output_dir / "trajectory_cost_summary.csv"
    json_path = output_dir / "trajectory_cost_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "label": "Trajectory-cost comparison: 4 systems x "
                "(3 planners x {initial, final} + noise-free MPPI)",
                "rows": summary_rows,
            },
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        stream.write("\n")
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results/trajectory_cost_comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_root
    output_dir.mkdir(parents=True, exist_ok=True)

    all_curves: dict[str, dict] = {}
    all_mppi: dict[str, dict] = {}
    for system_name in SYSTEMS:
        cost_dir = args.results_root / system_name
        planner_rows = _read_planner_rows(cost_dir)
        all_curves[system_name] = aggregate_planner_curves(planner_rows)
        mppi_rows = _read_mppi_rows(cost_dir)
        all_mppi[system_name] = aggregate_mppi(mppi_rows) if mppi_rows else None

    summary_csv, summary_json = write_summary(all_curves, all_mppi, output_dir)

    fig, axes = plt.subplots(1, len(SYSTEMS), figsize=(3.0 * len(SYSTEMS), 3.4), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")

    legend_axes = []
    split_axes_pairs = []
    for col_index, (ax, system_name) in enumerate(zip(axes, SYSTEMS)):
        curves = all_curves[system_name]
        title = SYSTEM_TITLES.get(system_name, system_name)
        if system_name == SPLIT_AXIS_SYSTEM:
            low_range = SPLIT_LOW_RANGE_OVERRIDE or _padded_range(
                _collect_curve_values(curves, SPLIT_LOW_PLANNERS)
            )
            high_range = SPLIT_HIGH_RANGE_OVERRIDE or _padded_range(
                _collect_curve_values(curves, SPLIT_HIGH_PLANNERS)
            )
            ax_top, ax_bottom = create_split_axes(fig, ax)
            plot_system_axis(ax_top, curves, show_title=title, show_xlabel=False)
            plot_system_axis(ax_bottom, curves, show_title=None, show_xlabel=True)
            ax_top.set_ylim(*high_range)
            ax_bottom.set_ylim(*low_range)
            # This panel is split into 2 segments, so halve the per-segment
            # tick count -- otherwise it shows 2x as many labels overall as
            # every other (single-axis) panel. _apply_uniform_yticks re-picks
            # its own "nice" range from the axis's current auto-scaled data
            # and calls set_ylim again, so it would silently override an
            # explicit *_RANGE_OVERRIDE -- give an overridden segment fixed,
            # evenly spaced ticks across the pinned range instead, same as
            # the non-split Y_AXIS_OVERRIDES branch below.
            if SPLIT_HIGH_RANGE_OVERRIDE is not None:
                ax_top.set_ylim(*high_range)
                ax_top.set_yticks(np.linspace(high_range[0], high_range[1], Y_TICK_COUNT // 2))
            else:
                _apply_uniform_yticks(ax_top, Y_TICK_COUNT // 2)
            if SPLIT_LOW_RANGE_OVERRIDE is not None:
                ax_bottom.set_ylim(*low_range)
                ax_bottom.set_yticks(np.linspace(low_range[0], low_range[1], Y_TICK_COUNT // 2))
            else:
                _apply_uniform_yticks(ax_bottom, Y_TICK_COUNT // 2)
            if col_index == 0:
                ax_top.set_ylabel("Solution cost", labelpad=2)
                ax_bottom.set_ylabel(" ", labelpad=2)
            legend_axes.append(ax_top)
            legend_axes.append(ax_bottom)
            split_axes_pairs.append((ax, ax_top, ax_bottom))
        else:
            plot_system_axis(ax, curves, show_title=title, show_xlabel=True)
            if system_name in Y_AXIS_OVERRIDES:
                y_min, y_max = Y_AXIS_OVERRIDES[system_name]
                ax.set_ylim(y_min, y_max)
                ax.set_yticks(np.linspace(y_min, y_max, Y_TICK_COUNT))
            else:
                _apply_uniform_yticks(ax, Y_TICK_COUNT)
            if col_index == 0:
                ax.set_ylabel("Solution cost (path length)", labelpad=2)
            legend_axes.append(ax)

    seen_labels: dict[str, object] = {}
    for ax in legend_axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            seen_labels.setdefault(label, handle)
    # Fixed, deterministic order (not draw order) so the legend reads the
    # same every time regardless of which systems/planners had missing data.
    legend_order = [
        f"{prefix}-{PLANNER_DISPLAY_NAMES[planner]}"
        for planner in PLANNERS
        for prefix in ("Vanilla", "AURA")
    ]
    labels = [label for label in legend_order if label in seen_labels]
    handles = [seen_labels[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=6,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=1.0,
        facecolor="#E8E8E8",
        fontsize=plt.rcParams["legend.fontsize"],
    )
    fig.subplots_adjust(bottom=0.26, left=0.05, right=0.98, wspace=0.20, top=0.92)
    for base_ax, ax_top, ax_bottom in split_axes_pairs:
        _align_split_axes_to_base(base_ax, ax_top, ax_bottom)

    png_path = output_dir / "trajectory_cost_comparison.png"
    pdf_path = output_dir / "trajectory_cost_comparison.pdf"
    svg_path = output_dir / "trajectory_cost_comparison.svg"
    fig.savefig(png_path, dpi=300, facecolor="white", edgecolor="none")
    fig.savefig(pdf_path, facecolor="white", edgecolor="none")
    fig.savefig(svg_path, facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"[save] {summary_csv}")
    print(f"[save] {summary_json}")
    print(f"[save] {png_path}")
    print(f"[save] {pdf_path}")
    print(f"[save] {svg_path}")


if __name__ == "__main__":
    main()
