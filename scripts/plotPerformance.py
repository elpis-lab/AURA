#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import argparse
import re
import builtins
import sys
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set up plotting style with Seaborn and Times New Roman font
sns.set_theme(style="whitegrid", context="paper")
sns.set_palette("husl")

# Configure matplotlib to use Times New Roman for all text
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 7
plt.rcParams["figure.titlesize"] = 8

# Console output control: default to quiet mode.
QUIET_OUTPUT = True


def print(*args, **kwargs):
    """Silence non-essential logs unless quiet mode is disabled."""
    if not QUIET_OUTPUT:
        builtins.print(*args, **kwargs)


# ============================================================================
# MULTI-DIRECTORY CONFIGURATION - Plot multiple experiments side by side
# ============================================================================
# To plot multiple experiments as subfigures side by side, set EXPERIMENT_DIRECTORIES
# to a list of directory paths. Each directory should contain CSV files from compareCost.py
#
# Example usage:
# EXPERIMENT_DIRECTORIES = [
#     "results/planning/5_30_5",
#     "results/planning/10_40_10",
#     "results/planning/15_45_15"
# ]
#
# SUBFIGURE_TITLES = ["Exp 1", "Exp 2", "Exp 3"]  # Optional custom titles
#
# Leave as None to use single directory mode (original behavior)
EXPERIMENT_DIRECTORIES = [
    "results/planning/5_30_5",
    "results/planning/2_10_2",
    "results/planning/4_20_4",
]

# Subfigure titles (optional, one per directory)
# If None, will use directory names as titles
SUBFIGURE_TITLES = [
    "Kinematic Car in SE2",
    "6D Double Integrator in R3",
    "Learned Pushing Dynamics in SE2",
]

# ============================================================================
# PLOT STYLE CONFIGURATION - Customize appearance here
# ============================================================================
line_width = 1.0  # Width of plot lines
marker_size = 2.5  # Size of markers (circles, squares)
initial_line_alpha = 0.5  # Transparency for initial (dashed) lines (0=transparent, 1=opaque)
grid_alpha = 0.3  # Grid transparency
std_fill_alpha = 0.2  # Standard deviation shading transparency
std_fill_alpha_combined = 0.15  # Standard deviation shading for combined plot
std_scale_factor = 0.1  # Scale factor for std dev shading (0.5 = half std dev, 1.0 = full std dev)

# Curve smoothing
curve_smoothing_points = 300  # Number of points for smooth curve interpolation

# Legend positioning
legend_ncol = 6  # Number of columns in legend
legend_bbox_y_separate = -0.2  # Legend vertical position for separate plots
legend_bbox_y_combined = 0.0  # Legend vertical position for combined/improvement plots
bottom_margin_separate = 0.18  # Bottom margin for separate plots
bottom_margin_combined = 0.25  # Bottom margin for combined plot
bottom_margin_improvement = 0.20  # Bottom margin for improvement plot
X_AXIS_LABEL = "Offline Planning Time"  # Shared x-axis label text
Y_TICK_COUNT = 5  # Number of y-axis tick labels for regular subplots
SPLIT_Y_TICK_COUNT = 5  # Number of y-axis tick labels per split segment

# Custom y-axis ticks/labels for each subfigure in multi-directory mode.
# Set each entry to None for automatic behavior.
SUBFIGURE_Y_TICKS = [
    [2, 3, 4, 5, 6, 7, 8],  # Subfigure 1 defaults
    None,  # Subfigure 2 is split (uses SPLIT_*_Y_TICKS)
    [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2],  # Subfigure 3 defaults
]
SUBFIGURE_Y_TICK_LABELS = [None, None, None]
# Custom y-axis label text per subfigure in multi-directory mode.
# Set entry to None to hide that subplot's y-axis label.
SUBFIGURE_Y_AXIS_LABELS = [
    "Cost",
    None,
    None,
]

# Custom y-axis ticks/labels for split middle subplot (idx == 1).
SPLIT_TOP_Y_TICKS = [6.5, 7.0, 7.5, 8.0]
SPLIT_TOP_Y_TICK_LABELS = None
SPLIT_BOTTOM_Y_TICKS = [1.5, 2.0, 2.5]
SPLIT_BOTTOM_Y_TICK_LABELS = None

# Border visibility
show_top_spine = False  # Show top border
show_right_spine = False  # Show right border
show_left_spine = True  # Show left border
show_bottom_spine = True  # Show bottom border

# Background color for slide mode
slide_background_color = "#F5F5F5"  # Light gray background for slides
default_background_color = "white"  # White background for default mode

# Figure size configuration
figure_width_multi = 7.1  # IEEE paper width (inches) for multi-directory plots
figure_height_multi = 2.5  # Height for multi-directory plots (inches)

# Y-axis range configuration (set to None to use auto-scaling)
# Multi-directory mode: one entry per subfigure, e.g. [(0, 5), (2, 14), (0.8, 2.2)]
SUBFIGURE_Y_RANGES = [
    None,  # Subfigure 1
    None,  # Subfigure 2
    None,  # Subfigure 3
]

# Single-directory mode y-ranges
INITIAL_PLOT_Y_RANGE = None  # For "Initial Cost" subplot in create_plots
FINAL_PLOT_Y_RANGE = None  # For "Final Cost" subplot in create_plots
COMBINED_PLOT_Y_RANGE = None  # For create_combined_plot
IMPROVEMENT_PLOT_Y_RANGE = None  # For create_improvement_plot
# ============================================================================

# ============================================================================
# COLOR CONFIGURATION - Customize planner colors here
# ============================================================================
# You can use hex color codes (e.g., "#E74C3C") or named colors (e.g., "red")
PLANNER_COLORS = {
    "sststar": "#EB5B00",  # Red
    "aoest": "#143D60",  # Blue
    "aorrt": "#A0C878",  # Green
    "rrt": "#F39C12",  # Orange
}

# Planner name display mapping (for legend labels)
# Maps internal planner name (lowercase) to display name
PLANNER_DISPLAY_NAMES = {
    "sststar": "SST",  # Show "SST" instead of "SSTSTAR"
    "aoest": "AOEST",
    "aorrt": "AORRT",
    "rrt": "RRT",
}
# ============================================================================


def create_smooth_curve(x_data, y_data, num_points=300):
    """Create a smooth curve through data points using spline interpolation."""
    if len(x_data) < 3:
        # Not enough points for spline, return original
        return x_data, y_data

    try:
        # Sort data by x values
        sorted_indices = np.argsort(x_data)
        x_sorted = x_data[sorted_indices]
        y_sorted = y_data[sorted_indices]

        # Create smooth curve using cubic spline
        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), num_points)
        spline = make_interp_spline(x_sorted, y_sorted, k=min(3, len(x_sorted) - 1))
        y_smooth = spline(x_smooth)

        return x_smooth, y_smooth
    except Exception as e:
        # If interpolation fails, return original data
        print(f"Warning: Curve smoothing failed: {e}")
        return x_data, y_data


def find_run_csv_files(time_range=None, csv_dir="results/planning/5_30_5"):
    """Find all results CSV files for individual runs."""
    run_files = []

    # Look for new naming convention: {planner}_{run_number}.csv (e.g., aorrt_01.csv, sststar_01.csv)
    planner_patterns = ["aorrt", "sststar", "aoest", "sst", "rrt"]
    for planner in planner_patterns:
        pattern = f"{csv_dir}/{planner}_*.csv"
        files = glob.glob(pattern)
        run_files.extend(files)

    # Fallback: Look for old pattern: results_run_XX_time_X.XtoX.Xs.csv
    if not run_files:
        run_files = glob.glob(f"{csv_dir}/results_run_*_time_*to*s.csv")

    # Fallback: Look for old format: results_all_runs_*.csv
    if not run_files:
        run_files = glob.glob(f"{csv_dir}/results_all_runs_*.csv")

    if not run_files:
        # Try without csv_dir prefix (current directory)
        for planner in planner_patterns:
            pattern = f"{planner}_*.csv"
            files = glob.glob(pattern)
            run_files.extend(files)

    print(f"Found {len(run_files)} CSV files:")
    for file in sorted(run_files):
        print(f"  - {file}")

    # Filter by time range if specified
    if time_range:
        filtered_files = []
        for file in run_files:
            # Extract time range from filename (old format only)
            match = re.search(r"time_(\d+\.?\d*)to(\d+\.?\d*)s", file)
            if match:
                file_min_time = float(match.group(1))
                file_max_time = float(match.group(2))

                # Check if file's time range overlaps with requested range
                if file_min_time <= time_range[1] and file_max_time >= time_range[0]:
                    filtered_files.append(file)
                    print(f"  ✓ {file} (time range: {file_min_time}-{file_max_time}s)")
                else:
                    print(
                        f"  ✗ {file} (time range: {file_min_time}-{file_max_time}s) - outside requested range"
                    )
            else:
                # New format or files without time range info - include them
                # (time range filtering will be done by load_and_prepare_data based on CSV content)
                filtered_files.append(file)
                print(f"  ✓ {file} (no time range in filename, will filter by CSV content)")

        run_files = filtered_files

    if not run_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    return run_files


def load_and_prepare_data(csv_files, time_range=None, run_range=None):
    """Load CSV data from multiple files and prepare for plotting."""
    all_dataframes = []

    for csv_file in csv_files:
        print(f"Loading data from: {csv_file}")
        df = pd.read_csv(csv_file)

        # Verify required columns exist
        required_columns = ["run_number", "planner", "planning_time", "initial_cost", "final_cost"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️  Warning: {csv_file} missing columns: {missing_columns}")
            continue

        # Filter by run range if specified
        if run_range:
            df = df[(df["run_number"] >= run_range[0]) & (df["run_number"] <= run_range[1])]
            if len(df) == 0:
                print(f"  No data in run range {run_range} for {csv_file}")
                continue

        # Filter by time range if specified
        if time_range:
            df = df[(df["planning_time"] >= time_range[0]) & (df["planning_time"] <= time_range[1])]
            if len(df) == 0:
                print(f"  No data in time range {time_range} for {csv_file}")
                continue

        all_dataframes.append(df)
        print(f"  Loaded {len(df)} rows")

    if not all_dataframes:
        raise ValueError("No valid data found in the specified files, time range, and run range")

    # Combine all dataframes
    df = pd.concat(all_dataframes, ignore_index=True)

    print(f"\nCombined data: {len(df)} total rows")
    print(f"Planners: {df['planner'].unique()}")
    print(f"Planning times: {sorted(df['planning_time'].unique())}")
    print(f"Runs: {sorted(df['run_number'].unique())}")

    return df


def calculate_statistics(df, use_median=False):
    """Calculate mean/median and standard deviation for each planner and planning time."""
    if use_median:
        stats = (
            df.groupby(["planner", "planning_time"])
            .agg(
                {
                    "initial_cost": ["median", "std", "count"],
                    "final_cost": ["median", "std", "count"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        stats.columns = [
            "planner",
            "planning_time",
            "initial_cost_median",
            "initial_cost_std",
            "initial_cost_count",
            "final_cost_median",
            "final_cost_std",
            "final_cost_count",
        ]
    else:
        stats = (
            df.groupby(["planner", "planning_time"])
            .agg({"initial_cost": ["mean", "std", "count"], "final_cost": ["mean", "std", "count"]})
            .reset_index()
        )

        # Flatten column names
        stats.columns = [
            "planner",
            "planning_time",
            "initial_cost_mean",
            "initial_cost_std",
            "initial_cost_count",
            "final_cost_mean",
            "final_cost_std",
            "final_cost_count",
        ]

    return stats


def calculate_improvement_statistics(df, use_median=False):
    """Calculate improvement percentage statistics for each planner and planning time."""
    # Calculate improvement percentage for each row
    df = df.copy()
    df["improvement_percentage"] = (
        (df["initial_cost"] - df["final_cost"]) / df["initial_cost"]
    ) * 100

    if use_median:
        stats = (
            df.groupby(["planner", "planning_time"])
            .agg({"improvement_percentage": ["median", "std", "count"]})
            .reset_index()
        )

        # Flatten column names
        stats.columns = [
            "planner",
            "planning_time",
            "improvement_percentage_median",
            "improvement_percentage_std",
            "improvement_percentage_count",
        ]
    else:
        stats = (
            df.groupby(["planner", "planning_time"])
            .agg({"improvement_percentage": ["mean", "std", "count"]})
            .reset_index()
        )

        # Flatten column names
        stats.columns = [
            "planner",
            "planning_time",
            "improvement_percentage_mean",
            "improvement_percentage_std",
            "improvement_percentage_count",
        ]

    return stats


def create_plots(
    df,
    stats,
    output_dir="plots",
    time_range=None,
    show_std=True,
    use_median=False,
    use_curve=False,
    background_color="white",
):
    """Create the cost vs time plots with or without shaded standard deviation regions."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=background_color)
    fig.patch.set_facecolor(background_color)
    for ax in axes:
        ax.set_facecolor(background_color)

    # Determine which column to use for central tendency
    initial_col = "initial_cost_median" if use_median else "initial_cost_mean"
    final_col = "final_cost_median" if use_median else "final_cost_mean"
    stat_name = "Median" if use_median else "Mean"

    # Plot 1: Initial costs
    ax1 = axes[0]
    for planner in df["planner"].unique():
        planner_data = stats[stats["planner"] == planner]
        planner_lower = planner.lower()
        color = PLANNER_COLORS.get(planner_lower, "gray")
        display_name = PLANNER_DISPLAY_NAMES.get(planner_lower, planner.upper())

        # Sort by planning time for proper line plotting
        planner_data = planner_data.sort_values("planning_time")

        # Get data for plotting
        x_data = planner_data["planning_time"].values
        y_data = planner_data[initial_col].values

        if use_curve:
            # Create smooth curve
            x_smooth, y_smooth = create_smooth_curve(x_data, y_data, curve_smoothing_points)
            # Plot smooth curve
            ax1.plot(
                x_smooth,
                y_smooth,
                "--",  # Dashed line for initial
                label=f"Vanilla-{display_name}",
                color=color,
                linewidth=line_width,
                alpha=initial_line_alpha,
            )
            # Plot original data points
            ax1.plot(
                x_data,
                y_data,
                "o",  # Just markers
                color=color,
                markersize=marker_size,
                alpha=initial_line_alpha,
            )
        else:
            # Plot mean/median line with DASHED style for initial costs
            ax1.plot(
                x_data,
                y_data,
                "o--",  # Dashed line for initial
                label=f"Vanilla-{display_name}",
                color=color,
                linewidth=line_width,
                markersize=marker_size,
                alpha=initial_line_alpha,  # Add transparency
            )

        # Plot shaded region for standard deviation (if requested)
        if show_std:
            ax1.fill_between(
                planner_data["planning_time"],
                planner_data[initial_col] - planner_data["initial_cost_std"] * std_scale_factor,
                planner_data[initial_col] + planner_data["initial_cost_std"] * std_scale_factor,
                color=color,
                alpha=std_fill_alpha,
            )

    ax1.set_xlabel(X_AXIS_LABEL, labelpad=2)
    ax1.set_ylabel("Initial Cost", labelpad=2)
    if INITIAL_PLOT_Y_RANGE is not None:
        ax1.set_ylim(INITIAL_PLOT_Y_RANGE[0], INITIAL_PLOT_Y_RANGE[1])
    ax1.tick_params(axis="x", pad=2)  # Bring x-axis tick labels closer
    ax1.tick_params(axis="y", pad=2)  # Bring y-axis tick labels closer
    ax1.grid(True, alpha=grid_alpha)
    # Configure spines (borders)
    ax1.spines["top"].set_visible(show_top_spine)
    ax1.spines["right"].set_visible(show_right_spine)
    ax1.spines["left"].set_visible(show_left_spine)
    ax1.spines["bottom"].set_visible(show_bottom_spine)

    # Plot 2: Final costs
    ax2 = axes[1]
    for planner in df["planner"].unique():
        planner_data = stats[stats["planner"] == planner]
        planner_lower = planner.lower()
        color = PLANNER_COLORS.get(planner_lower, "gray")
        display_name = PLANNER_DISPLAY_NAMES.get(planner_lower, planner.upper())

        # Sort by planning time for proper line plotting
        planner_data = planner_data.sort_values("planning_time")

        # Get data for plotting
        x_data = planner_data["planning_time"].values
        y_data = planner_data[final_col].values

        if use_curve:
            # Create smooth curve
            x_smooth, y_smooth = create_smooth_curve(x_data, y_data, curve_smoothing_points)
            # Plot smooth curve
            ax2.plot(
                x_smooth,
                y_smooth,
                "-",  # Solid line for final
                label=f"AURA-{display_name}",
                color=color,
                linewidth=line_width,
            )
            # Plot original data points
            ax2.plot(
                x_data,
                y_data,
                "s",  # Just markers
                color=color,
                markersize=marker_size,
            )
        else:
            # Plot mean/median line with SOLID style for final costs
            ax2.plot(
                x_data,
                y_data,
                "s-",  # Solid line for final
                label=f"AURA-{display_name}",
                color=color,
                linewidth=line_width,
                markersize=marker_size,
            )

        # Plot shaded region for standard deviation (if requested)
        if show_std:
            ax2.fill_between(
                planner_data["planning_time"],
                planner_data[final_col] - planner_data["final_cost_std"] * std_scale_factor,
                planner_data[final_col] + planner_data["final_cost_std"] * std_scale_factor,
                color=color,
                alpha=std_fill_alpha,
            )

    ax2.set_xlabel(X_AXIS_LABEL, labelpad=2)
    ax2.set_ylabel("Final Cost", labelpad=2)
    if FINAL_PLOT_Y_RANGE is not None:
        ax2.set_ylim(FINAL_PLOT_Y_RANGE[0], FINAL_PLOT_Y_RANGE[1])
    ax2.tick_params(axis="x", pad=2)  # Bring x-axis tick labels closer
    ax2.tick_params(axis="y", pad=2)  # Bring y-axis tick labels closer
    ax2.grid(True, alpha=grid_alpha)
    # Configure spines (borders)
    ax2.spines["top"].set_visible(show_top_spine)
    ax2.spines["right"].set_visible(show_right_spine)
    ax2.spines["left"].set_visible(show_left_spine)
    ax2.spines["bottom"].set_visible(show_bottom_spine)

    # Add a single legend below both subplots
    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_bbox_y_separate),
        ncol=legend_ncol,
        frameon=True,
        fancybox=True,  # Rounded corners
        shadow=False,  # No shadow
        framealpha=1.0,  # Fully opaque frame
        facecolor="#E8E8E8",  # Light grey background
    )

    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=bottom_margin_separate)

    # Show the plot first
    plt.show()

    # Save in both PDF and SVG formats
    base_filename = Path(output_dir) / "costComparison"
    if time_range:
        base_filename = Path(output_dir) / f"costComparison_{time_range[0]}-{time_range[1]}s"
    if not show_std:
        base_filename = Path(output_dir) / f"{base_filename.name}_only"

    # Determine if we want transparency
    use_transparent = background_color != default_background_color

    # Save as PDF
    pdf_file = f"{base_filename}.pdf"
    plt.savefig(
        pdf_file,
        dpi=300,
        bbox_inches="tight",
        facecolor=background_color,
        edgecolor="none",
        transparent=use_transparent,
    )
    print(f"✅ PDF plot saved to: {pdf_file}")

    # Save as SVG - always use transparent background
    svg_file = f"{base_filename}.svg"
    plt.savefig(
        svg_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="none",  # Always transparent for SVG
        edgecolor="none",
        transparent=True,  # Always transparent for SVG
    )
    print(f"✅ SVG plot saved to: {svg_file}")

    # Save as PNG
    png_file = f"{base_filename}.png"
    plt.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
        facecolor=background_color,
        edgecolor="none",
        transparent=use_transparent,
    )
    print(f"✅ PNG plot saved to: {png_file}")

    return [pdf_file, svg_file, png_file]


def plot_on_axis(
    ax,
    df,
    stats,
    use_median=False,
    use_curve=False,
    show_std=True,
    title=None,
    background_color="white",
    show_ylabel=True,
    y_axis_label="Cost",
):
    """Plot data on a given axis (for multi-directory support)."""
    # Determine which column to use for central tendency
    initial_col = "initial_cost_median" if use_median else "initial_cost_mean"
    final_col = "final_cost_median" if use_median else "final_cost_mean"

    for planner in df["planner"].unique():
        planner_data = stats[stats["planner"] == planner]
        planner_lower = planner.lower()
        color = PLANNER_COLORS.get(planner_lower, "gray")
        display_name = PLANNER_DISPLAY_NAMES.get(planner_lower, planner.upper())

        # Sort by planning time for proper line plotting
        planner_data = planner_data.sort_values("planning_time")

        # Get data for plotting
        x_data = planner_data["planning_time"].values
        y_initial = planner_data[initial_col].values
        y_final = planner_data[final_col].values

        if use_curve:
            # Plot initial costs with smooth curve
            x_smooth_init, y_smooth_init = create_smooth_curve(
                x_data, y_initial, curve_smoothing_points
            )
            ax.plot(
                x_smooth_init,
                y_smooth_init,
                "--",  # Dashed line for initial
                label=f"Vanilla-{display_name}",
                color=color,
                linewidth=line_width,
                alpha=initial_line_alpha,
            )
            ax.plot(
                x_data,
                y_initial,
                "o",  # Just markers
                color=color,
                markersize=marker_size,
                alpha=initial_line_alpha,
            )

            # Plot final costs with smooth curve
            x_smooth_final, y_smooth_final = create_smooth_curve(
                x_data, y_final, curve_smoothing_points
            )
            ax.plot(
                x_smooth_final,
                y_smooth_final,
                "-",  # Solid line for final
                label=f"AURA-{display_name}",
                color=color,
                linewidth=line_width,
            )
            ax.plot(
                x_data,
                y_final,
                "s",  # Just markers
                color=color,
                markersize=marker_size,
            )
        else:
            # Plot initial costs with DASHED line
            ax.plot(
                x_data,
                y_initial,
                "o--",  # Dashed line for initial
                label=f"Vanilla-{display_name}",
                color=color,
                linewidth=line_width,
                markersize=marker_size,
                alpha=initial_line_alpha,  # Add transparency
            )

            # Plot final costs with SOLID line
            ax.plot(
                x_data,
                y_final,
                "s-",  # Solid line for final
                label=f"AURA-{display_name}",
                color=color,
                linewidth=line_width,
                markersize=marker_size,
            )

        # Plot shaded region for standard deviation (if requested)
        if show_std:
            ax.fill_between(
                planner_data["planning_time"],
                planner_data[initial_col] - planner_data["initial_cost_std"] * std_scale_factor,
                planner_data[initial_col] + planner_data["initial_cost_std"] * std_scale_factor,
                color=color,
                alpha=std_fill_alpha_combined,
            )

            ax.fill_between(
                planner_data["planning_time"],
                planner_data[final_col] - planner_data["final_cost_std"] * std_scale_factor,
                planner_data[final_col] + planner_data["final_cost_std"] * std_scale_factor,
                color=color,
                alpha=std_fill_alpha_combined,
            )

    ax.set_xlabel(X_AXIS_LABEL, labelpad=2)
    if show_ylabel and y_axis_label is not None:
        ax.set_ylabel(y_axis_label, labelpad=2)
    ax.tick_params(axis="x", pad=2)  # Bring x-axis tick labels closer
    ax.tick_params(axis="y", pad=2)  # Bring y-axis tick labels closer
    ax.grid(True, alpha=grid_alpha)
    # Configure spines (borders)
    ax.spines["top"].set_visible(show_top_spine)
    ax.spines["right"].set_visible(show_right_spine)
    ax.spines["left"].set_visible(show_left_spine)
    ax.spines["bottom"].set_visible(show_bottom_spine)

    if title:
        ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"])

    # Set background color
    ax.set_facecolor(background_color)


def create_split_axes_for_second_plot(fig, base_ax):
    """Replace the second subplot with two stacked axes: 6-8 (top) and 0-3 (bottom)."""
    pos = base_ax.get_position()
    base_ax.set_visible(False)

    gap = pos.height * 0.06
    top_h = pos.height * 0.45
    bottom_h = pos.height - top_h - gap

    ax1 = fig.add_axes([pos.x0, pos.y0 + bottom_h + gap, pos.width, top_h])
    ax2 = fig.add_axes([pos.x0, pos.y0, pos.width, bottom_h], sharex=ax1)

    ax1.set_ylim(6.5, 8)  # upper part
    ax2.set_ylim(1.5, 2.5)  # lower part
    # Keep the same number of y tick labels on both split axes.
    ax1.set_yticks(np.linspace(6.5, 8, SPLIT_Y_TICK_COUNT))
    ax2.set_yticks(np.linspace(1.5, 2.5, SPLIT_Y_TICK_COUNT))

    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.tick_params(labelbottom=False, bottom=False)

    d = 0.012
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    return ax1, ax2


def align_split_axes_to_base(base_ax, ax1, ax2):
    """Keep split axes exactly inside the (possibly adjusted) base subplot box."""
    pos = base_ax.get_position()
    gap = pos.height * 0.06
    top_h = pos.height * 0.45
    bottom_h = pos.height - top_h - gap
    ax1.set_position([pos.x0, pos.y0 + bottom_h + gap, pos.width, top_h])
    ax2.set_position([pos.x0, pos.y0, pos.width, bottom_h])


def apply_y_ticks(ax, tick_count=None, ticks=None, labels=None):
    """Apply y ticks with optional custom positions and labels."""
    if ticks is not None and len(ticks) > 0:
        ax.set_yticks(ticks)
    elif tick_count is not None:
        y_min, y_max = ax.get_ylim()
        if tick_count >= 2 and y_max > y_min:
            ax.set_yticks(np.linspace(y_min, y_max, tick_count))

    if labels is not None:
        ax.set_yticklabels(labels)
    else:
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda y, _: f"{int(round(y))}" if abs(y - round(y)) < 1e-8 else f"{y:.1f}"
            )
        )


def create_multi_directory_plot(
    directories,
    titles=None,
    output_dir="plots",
    time_range=None,
    run_range=None,
    show_std=True,
    use_median=False,
    use_curve=False,
    background_color="white",
):
    """Create a plot with multiple subfigures, one for each directory."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    num_dirs = len(directories)
    # Create figure with subfigures side by side
    # IEEE paper width: 7.1 inches total
    fig, axes = plt.subplots(
        1, num_dirs, figsize=(figure_width_multi, figure_height_multi), facecolor=background_color
    )
    fig.patch.set_facecolor(background_color)
    if num_dirs == 1:
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(background_color)

    stat_name = "Median" if use_median else "Mean"
    has_split_subplot = False
    split_axes_pairs = []

    for idx, (csv_dir, ax) in enumerate(zip(directories, axes)):
        print(f"\nProcessing directory {idx + 1}/{num_dirs}: {csv_dir}")

        try:
            # Find and load CSV files for this directory
            csv_files = find_run_csv_files(time_range, csv_dir)
            df = load_and_prepare_data(csv_files, time_range, run_range)
            stats = calculate_statistics(df, use_median)

            # Get title for this subfigure
            title = titles[idx] if titles and idx < len(titles) else Path(csv_dir).name

            # Per-subfigure y-axis label configuration.
            y_axis_label = (
                SUBFIGURE_Y_AXIS_LABELS[idx] if idx < len(SUBFIGURE_Y_AXIS_LABELS) else "Cost"
            )
            show_ylabel = y_axis_label is not None

            # Split only the second subplot into 6-8 (top) and 0-3 (bottom).
            if idx == 1:
                has_split_subplot = True
                ax1, ax2 = create_split_axes_for_second_plot(fig, ax)
                split_axes_pairs.append((ax, ax1, ax2))
                plot_on_axis(
                    ax1,
                    df,
                    stats,
                    use_median,
                    use_curve,
                    show_std,
                    title,
                    background_color,
                    False,
                    y_axis_label,
                )
                plot_on_axis(
                    ax2,
                    df,
                    stats,
                    use_median,
                    use_curve,
                    show_std,
                    None,
                    background_color,
                    show_ylabel,
                    y_axis_label,
                )
                # Hide all x-axis decorations on the top split axis.
                ax1.set_xlabel("")
                ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
                apply_y_ticks(
                    ax1,
                    tick_count=SPLIT_Y_TICK_COUNT,
                    ticks=SPLIT_TOP_Y_TICKS,
                    labels=SPLIT_TOP_Y_TICK_LABELS,
                )
                apply_y_ticks(
                    ax2,
                    tick_count=SPLIT_Y_TICK_COUNT,
                    ticks=SPLIT_BOTTOM_Y_TICKS,
                    labels=SPLIT_BOTTOM_Y_TICK_LABELS,
                )
            else:
                plot_on_axis(
                    ax,
                    df,
                    stats,
                    use_median,
                    use_curve,
                    show_std,
                    title,
                    background_color,
                    show_ylabel,
                    y_axis_label,
                )

            # Apply optional custom y-axis range per subfigure from top-level config
            if idx != 1 and idx < len(SUBFIGURE_Y_RANGES) and SUBFIGURE_Y_RANGES[idx] is not None:
                y_min, y_max = SUBFIGURE_Y_RANGES[idx]
                ax.set_ylim(y_min, y_max)
            if idx != 1:
                custom_ticks = SUBFIGURE_Y_TICKS[idx] if idx < len(SUBFIGURE_Y_TICKS) else None
                custom_labels = (
                    SUBFIGURE_Y_TICK_LABELS[idx] if idx < len(SUBFIGURE_Y_TICK_LABELS) else None
                )
                apply_y_ticks(ax, tick_count=Y_TICK_COUNT, ticks=custom_ticks, labels=custom_labels)

        except Exception as e:
            print(f"⚠️  Warning: Failed to process {csv_dir}: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error loading\n{csv_dir}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel(X_AXIS_LABEL, labelpad=2)
            ax.set_ylabel("Cost", labelpad=2)
            ax.tick_params(axis="x", pad=2)  # Bring x-axis tick labels closer
            ax.tick_params(axis="y", pad=2)  # Bring y-axis tick labels closer

    # Add a single shared legend below all subplots
    if num_dirs > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_bbox_y_combined),
            ncol=legend_ncol,
            frameon=True,
            fancybox=True,  # Rounded corners
            shadow=False,  # No shadow
            framealpha=1.0,  # Fully opaque frame
            facecolor="#E8E8E8",  # Light grey background
        )

    # Adjust layout to ensure everything fits within 7.1 inches
    # Avoid tight_layout when split axes are used because custom axes are not managed by it.
    if not has_split_subplot:
        plt.tight_layout(pad=0.1)
    plt.subplots_adjust(
        bottom=bottom_margin_combined,
        left=0.05,  # Minimal left margin
        right=0.98,  # Minimal right margin
        wspace=0.15,  # Spacing between subplots
        top=0.92,  # Top margin (increased from 0.95 for more space)
    )

    # When subplots_adjust moves base axes, keep split axes aligned to that updated box.
    for base_ax, ax1, ax2 in split_axes_pairs:
        align_split_axes_to_base(base_ax, ax1, ax2)

    # Save in multiple formats BEFORE showing (to preserve colors and state)
    base_filename = Path(output_dir) / "costComparison"
    if time_range:
        base_filename = Path(output_dir) / f"costComparison_{time_range[0]}-{time_range[1]}s"

    saved_files = []
    use_transparent = background_color != default_background_color
    for ext in ["pdf", "svg", "png"]:
        # Use Path.with_suffix() or convert to string explicitly
        file_path = str(base_filename) + f".{ext}"
        # For SVG, always use transparent background
        if ext == "svg":
            save_facecolor = "none"
            save_transparent = True
        else:
            save_facecolor = background_color
            save_transparent = use_transparent
        plt.savefig(
            file_path,
            dpi=300,
            bbox_inches=None,  # Use exact figure size (7.1 inches) instead of tight
            pad_inches=0,  # No padding
            facecolor=save_facecolor,
            edgecolor="none",
            transparent=save_transparent,
            format=ext,
        )
        print(f"✅ {ext.upper()} plot saved to: {file_path}")
        saved_files.append(file_path)

    # Show the plot after saving
    plt.show()

    return saved_files


def create_combined_plot(
    df,
    stats,
    output_dir="plots",
    time_range=None,
    show_std=True,
    use_median=False,
    use_curve=False,
    background_color="white",
):
    """Create a combined plot with both initial and final costs, with or without standard deviation."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))

    # Determine which column to use for central tendency
    initial_col = "initial_cost_median" if use_median else "initial_cost_mean"
    final_col = "final_cost_median" if use_median else "final_cost_mean"
    stat_name = "Median" if use_median else "Mean"

    for planner in df["planner"].unique():
        planner_data = stats[stats["planner"] == planner]
        planner_lower = planner.lower()
        color = PLANNER_COLORS.get(planner_lower, "gray")
        display_name = PLANNER_DISPLAY_NAMES.get(planner_lower, planner.upper())

        # Sort by planning time for proper line plotting
        planner_data = planner_data.sort_values("planning_time")

        # Get data for plotting
        x_data = planner_data["planning_time"].values
        y_initial = planner_data[initial_col].values
        y_final = planner_data[final_col].values

        if use_curve:
            # Plot initial costs with smooth curve
            x_smooth_init, y_smooth_init = create_smooth_curve(
                x_data, y_initial, curve_smoothing_points
            )
            ax.plot(
                x_smooth_init,
                y_smooth_init,
                "--",  # Dashed line for initial
                label=f"Vanilla - {display_name}",
                color=color,
                linewidth=line_width,
                alpha=initial_line_alpha,
            )
            ax.plot(
                x_data,
                y_initial,
                "o",  # Just markers
                color=color,
                markersize=marker_size,
                alpha=initial_line_alpha,
            )

            # Plot final costs with smooth curve
            x_smooth_final, y_smooth_final = create_smooth_curve(
                x_data, y_final, curve_smoothing_points
            )
            ax.plot(
                x_smooth_final,
                y_smooth_final,
                "-",  # Solid line for final
                label=f"AURA-{display_name}",
                color=color,
                linewidth=line_width,
            )
            ax.plot(
                x_data,
                y_final,
                "s",  # Just markers
                color=color,
                markersize=marker_size,
            )
        else:
            # Plot initial costs with DASHED line
            ax.plot(
                x_data,
                y_initial,
                "o--",  # Dashed line for initial
                label=f"Vanilla-{display_name}",
                color=color,
                linewidth=line_width,
                markersize=marker_size,
                alpha=initial_line_alpha,  # Add transparency
            )

            # Plot final costs with SOLID line
            ax.plot(
                x_data,
                y_final,
                "s-",  # Solid line for final
                label=f"AURA-{display_name}",
                color=color,
                linewidth=line_width,
                markersize=marker_size,
            )

        # Plot shaded region for standard deviation (if requested)
        if show_std:
            ax.fill_between(
                planner_data["planning_time"],
                planner_data[initial_col] - planner_data["initial_cost_std"] * std_scale_factor,
                planner_data[initial_col] + planner_data["initial_cost_std"] * std_scale_factor,
                color=color,
                alpha=std_fill_alpha_combined,
            )

            ax.fill_between(
                planner_data["planning_time"],
                planner_data[final_col] - planner_data["final_cost_std"] * std_scale_factor,
                planner_data[final_col] + planner_data["final_cost_std"] * std_scale_factor,
                color=color,
                alpha=std_fill_alpha_combined,
            )

    ax.set_xlabel(X_AXIS_LABEL, labelpad=2)
    ax.set_ylabel("Cost", labelpad=2)
    if COMBINED_PLOT_Y_RANGE is not None:
        ax.set_ylim(COMBINED_PLOT_Y_RANGE[0], COMBINED_PLOT_Y_RANGE[1])
    ax.tick_params(axis="x", pad=2)  # Bring x-axis tick labels closer
    ax.tick_params(axis="y", pad=2)  # Bring y-axis tick labels closer
    ax.grid(True, alpha=grid_alpha)
    # Configure spines (borders)
    ax.spines["top"].set_visible(show_top_spine)
    ax.spines["right"].set_visible(show_right_spine)
    ax.spines["left"].set_visible(show_left_spine)
    ax.spines["bottom"].set_visible(show_bottom_spine)

    # Adjust layout to make room for legend
    plt.tight_layout()

    # Add legend below the plot
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, legend_bbox_y_combined),
        ncol=legend_ncol,
        frameon=True,
        fancybox=True,  # Rounded corners
        shadow=False,  # No shadow
        framealpha=1.0,  # Fully opaque frame
        facecolor="#E8E8E8",  # Light grey background
    )
    plt.subplots_adjust(bottom=bottom_margin_combined)

    # Show the plot first
    plt.show()

    # Save in both PDF and SVG formats
    base_filename = Path(output_dir) / "costComparison"
    if time_range:
        base_filename = Path(output_dir) / f"costComparison_{time_range[0]}-{time_range[1]}s"
    if not show_std:
        base_filename = Path(output_dir) / f"{base_filename.name}_only"

    # Determine if we want transparency
    use_transparent = background_color != default_background_color

    # Save as PDF
    pdf_file = f"{base_filename}.pdf"
    plt.savefig(
        pdf_file,
        dpi=300,
        bbox_inches="tight",
        facecolor=background_color,
        edgecolor="none",
        transparent=use_transparent,
    )
    print(f"✅ PDF plot saved to: {pdf_file}")

    # Save as SVG - always use transparent background
    svg_file = f"{base_filename}.svg"
    plt.savefig(
        svg_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="none",  # Always transparent for SVG
        edgecolor="none",
        transparent=True,  # Always transparent for SVG
    )
    print(f"✅ SVG plot saved to: {svg_file}")

    # Save as PNG
    png_file = f"{base_filename}.png"
    plt.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
        facecolor=background_color,
        edgecolor="none",
        transparent=use_transparent,
    )
    print(f"✅ PNG plot saved to: {png_file}")

    return [pdf_file, svg_file, png_file]


def create_improvement_plot(
    df,
    improvement_stats,
    output_dir="plots",
    time_range=None,
    show_std=True,
    use_median=False,
    use_curve=False,
    background_color="white",
):
    """Create improvement percentage plot."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4), facecolor=background_color)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Determine which column to use for central tendency
    improvement_col = (
        "improvement_percentage_median" if use_median else "improvement_percentage_mean"
    )
    stat_name = "Median" if use_median else "Mean"

    for planner in df["planner"].unique():
        planner_data = improvement_stats[improvement_stats["planner"] == planner]
        planner_lower = planner.lower()
        color = PLANNER_COLORS.get(planner_lower, "gray")
        display_name = PLANNER_DISPLAY_NAMES.get(planner_lower, planner.upper())

        # Sort by planning time for proper line plotting
        planner_data = planner_data.sort_values("planning_time")

        # Get data for plotting
        x_data = planner_data["planning_time"].values
        y_data = planner_data[improvement_col].values

        if use_curve:
            # Create smooth curve
            x_smooth, y_smooth = create_smooth_curve(x_data, y_data, curve_smoothing_points)
            # Plot smooth curve
            ax.plot(
                x_smooth,
                y_smooth,
                "-",
                label=f"{display_name}",
                color=color,
                linewidth=line_width,
            )
            # Plot original data points
            ax.plot(
                x_data,
                y_data,
                "o",
                color=color,
                markersize=marker_size,
            )
        else:
            # Plot improvement percentage line
            ax.plot(
                x_data,
                y_data,
                "o-",
                label=f"{display_name}",
                color=color,
                linewidth=line_width,
                markersize=marker_size,
            )

        # Plot shaded region for standard deviation (if requested)
        if show_std:
            ax.fill_between(
                planner_data["planning_time"],
                planner_data[improvement_col]
                - planner_data["improvement_percentage_std"] * std_scale_factor,
                planner_data[improvement_col]
                + planner_data["improvement_percentage_std"] * std_scale_factor,
                color=color,
                alpha=std_fill_alpha,
            )

    ax.set_xlabel(X_AXIS_LABEL, labelpad=2)
    ax.set_ylabel("Improvement (%)", labelpad=2)
    if IMPROVEMENT_PLOT_Y_RANGE is not None:
        ax.set_ylim(IMPROVEMENT_PLOT_Y_RANGE[0], IMPROVEMENT_PLOT_Y_RANGE[1])
    ax.tick_params(axis="x", pad=2)  # Bring x-axis tick labels closer
    ax.tick_params(axis="y", pad=2)  # Bring y-axis tick labels closer
    ax.grid(True, alpha=grid_alpha)
    # Configure spines (borders)
    ax.spines["top"].set_visible(show_top_spine)
    ax.spines["right"].set_visible(show_right_spine)
    ax.spines["left"].set_visible(show_left_spine)
    ax.spines["bottom"].set_visible(show_bottom_spine)

    # Adjust layout to make room for legend
    plt.tight_layout()

    # Add legend below the plot
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, legend_bbox_y_combined),
        ncol=legend_ncol,
        frameon=True,
        fancybox=True,  # Rounded corners
        shadow=False,  # No shadow
        framealpha=1.0,  # Fully opaque frame
        facecolor="#E8E8E8",  # Light grey background
    )
    plt.subplots_adjust(bottom=bottom_margin_improvement)

    # Show the plot first
    plt.show()

    # Save in both PDF and SVG formats
    base_filename = Path(output_dir) / "costComparison"
    if time_range:
        base_filename = Path(output_dir) / f"costComparison_{time_range[0]}-{time_range[1]}s"
    if not show_std:
        base_filename = Path(output_dir) / f"{base_filename.name}_only"

    # Determine if we want transparency
    use_transparent = background_color != default_background_color

    # Save as PDF
    pdf_file = f"{base_filename}.pdf"
    plt.savefig(
        pdf_file,
        dpi=300,
        bbox_inches="tight",
        facecolor=background_color,
        edgecolor="none",
        transparent=use_transparent,
    )
    print(f"✅ PDF plot saved to: {pdf_file}")

    # Save as SVG - always use transparent background
    svg_file = f"{base_filename}.svg"
    plt.savefig(
        svg_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="none",  # Always transparent for SVG
        edgecolor="none",
        transparent=True,  # Always transparent for SVG
    )
    print(f"✅ SVG plot saved to: {svg_file}")

    # Save as PNG
    png_file = f"{base_filename}.png"
    plt.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
        facecolor=background_color,
        edgecolor="none",
        transparent=use_transparent,
    )
    print(f"✅ PNG plot saved to: {png_file}")

    return [pdf_file, svg_file, png_file]


def print_summary_statistics(stats, use_median=False):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    stat_name = "Median" if use_median else "Mean"
    initial_col = "initial_cost_median" if use_median else "initial_cost_mean"
    final_col = "final_cost_median" if use_median else "final_cost_mean"

    for planner in stats["planner"].unique():
        print(f"\n{planner.upper()} PLANNER:")
        print("-" * 40)
        planner_stats = stats[stats["planner"] == planner]

        for _, row in planner_stats.iterrows():
            print(f"Time: {row['planning_time']:.1f}s (n={row['initial_cost_count']})")
            print(f"  Initial Cost: {row[initial_col]:.3f} ± {row['initial_cost_std']:.3f}")
            print(f"  Final Cost:   {row[final_col]:.3f} ± {row['final_cost_std']:.3f}")
            improvement = row[initial_col] - row[final_col]
            print(f"  Improvement:  {improvement:.3f}")
            print()


def print_improvement_statistics(improvement_stats, use_median=False):
    """Print improvement percentage statistics."""
    print("\n" + "=" * 80)
    print("IMPROVEMENT PERCENTAGE STATISTICS")
    print("=" * 80)

    stat_name = "Median" if use_median else "Mean"
    improvement_col = (
        "improvement_percentage_median" if use_median else "improvement_percentage_mean"
    )

    for planner in improvement_stats["planner"].unique():
        print(f"\n{planner.upper()} PLANNER:")
        print("-" * 40)
        planner_stats = improvement_stats[improvement_stats["planner"] == planner]

        for _, row in planner_stats.iterrows():
            print(f"Time: {row['planning_time']:.1f}s (n={row['improvement_percentage_count']})")
            print(
                f"  Improvement: {row[improvement_col]:.2f}% ± {row['improvement_percentage_std']:.2f}%"
            )
            print()


def print_average_statistics(df):
    """Print overall average metrics per planner across all planning times and runs."""
    builtins.print("\n" + "=" * 80)
    builtins.print("AVERAGE STATISTICS (OVERALL)")
    builtins.print("=" * 80)

    avg_df = df.copy()
    avg_df["improvement_percentage"] = np.where(
        avg_df["initial_cost"] != 0,
        (avg_df["initial_cost"] - avg_df["final_cost"]) / avg_df["initial_cost"] * 100,
        np.nan,
    )

    planner_averages = (
        avg_df.groupby("planner")
        .agg(
            {
                "initial_cost": ["mean", "std", "count"],
                "final_cost": ["mean", "std", "count"],
                "improvement_percentage": ["mean", "std"],
            }
        )
        .reset_index()
    )

    planner_averages.columns = [
        "planner",
        "initial_cost_avg",
        "initial_cost_std",
        "initial_cost_count",
        "final_cost_avg",
        "final_cost_std",
        "final_cost_count",
        "improvement_avg",
        "improvement_std",
    ]

    for _, row in planner_averages.iterrows():
        builtins.print(f"\n{row['planner'].upper()} PLANNER:")
        builtins.print("-" * 40)
        builtins.print(
            f"  Initial Cost Average: {row['initial_cost_avg']:.3f} ± {row['initial_cost_std']:.3f} "
            f"(n={int(row['initial_cost_count'])})"
        )
        builtins.print(
            f"  Final Cost Average:   {row['final_cost_avg']:.3f} ± {row['final_cost_std']:.3f} "
            f"(n={int(row['final_cost_count'])})"
        )
        builtins.print(
            f"  Improvement Average:  {row['improvement_avg']:.2f}% ± {row['improvement_std']:.2f}%"
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Plot experiment results from multiple CSV files")
    parser.add_argument(
        "--csv-files",
        nargs="+",
        help="Path to specific CSV files (if not provided, will find all run files)",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Filter data by planning time range (e.g., --time-range 2.0 10.0)",
    )
    parser.add_argument(
        "--run-range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="Filter data by run number range (e.g., --run-range 19 50)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots", help="Output directory for plots"
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plots (only save)")
    parser.add_argument(
        "--no-std", action="store_true", help="Create plots without standard deviation"
    )
    parser.add_argument(
        "--median", action="store_true", help="Use median instead of mean for central tendency"
    )
    parser.add_argument(
        "--percentage",
        action="store_true",
        help="Plot improvement percentage instead of absolute costs",
    )
    parser.add_argument(
        "--curve",
        action="store_true",
        help="Fit smooth curves through data points for smoother visualization",
    )
    parser.add_argument(
        "--slide",
        action="store_true",
        help="Use slide background color instead of white",
    )
    parser.add_argument(
        "--print-average",
        action="store_true",
        help="Print overall average statistics per planner across all runs and planning times",
    )

    args = parser.parse_args()

    # Set background color based on --slide flag
    background_color = slide_background_color if args.slide else default_background_color

    try:
        # Check if multi-directory mode is enabled
        if EXPERIMENT_DIRECTORIES and len(EXPERIMENT_DIRECTORIES) > 1:
            # Multi-directory mode
            print(f"Multi-directory mode: plotting {len(EXPERIMENT_DIRECTORIES)} experiments")

            # Print averages per directory when requested
            if args.print_average:
                for idx, csv_dir in enumerate(EXPERIMENT_DIRECTORIES):
                    try:
                        csv_files = find_run_csv_files(args.time_range, csv_dir)
                        df = load_and_prepare_data(csv_files, args.time_range, args.run_range)
                        title = (
                            SUBFIGURE_TITLES[idx]
                            if SUBFIGURE_TITLES and idx < len(SUBFIGURE_TITLES)
                            else Path(csv_dir).name
                        )
                        builtins.print("\n" + "=" * 80)
                        builtins.print(f"AVERAGES FOR: {title} ({csv_dir})")
                        builtins.print("=" * 80)
                        print_average_statistics(df)
                    except Exception as e:
                        builtins.print(f"❌ Error computing averages for {csv_dir}: {e}")

            # Determine if we should show standard deviation
            show_std = not args.no_std

            # Create multi-directory plot
            saved_files = create_multi_directory_plot(
                EXPERIMENT_DIRECTORIES,
                SUBFIGURE_TITLES,
                args.output_dir,
                args.time_range,
                args.run_range,
                show_std,
                args.median,
                args.curve,
                background_color,
            )

            print(f"\n✅ Multi-directory plot saved to: {args.output_dir}/")
            print("Files created:")
            for file in saved_files:
                print(f"  - {file}")

            return 0

        # Single directory mode (original behavior)
        # Find or use specified CSV files
        if args.csv_files:
            csv_files = args.csv_files
        else:
            csv_files = find_run_csv_files(args.time_range)

        # Load and prepare data
        df = load_and_prepare_data(csv_files, args.time_range, args.run_range)

        if args.print_average:
            print_average_statistics(df)

        # Create plots
        if not args.no_show:
            plt.ion()  # Interactive mode

        # Determine if we should show standard deviation
        show_std = not args.no_std
        stat_type = "median" if args.median else "mean"

        if args.percentage:
            # Calculate improvement statistics
            improvement_stats = calculate_improvement_statistics(df, use_median=args.median)

            # Print improvement summary
            print_improvement_statistics(improvement_stats, use_median=args.median)

            # Create improvement percentage plot
            print(
                f"\nCreating improvement percentage plot (stat={stat_type}, std_dev={'on' if show_std else 'off'}, curve={'on' if args.curve else 'off'})..."
            )
            improvement_files = create_improvement_plot(
                df,
                improvement_stats,
                args.output_dir,
                args.time_range,
                show_std,
                args.median,
                args.curve,
                background_color,
            )

            # Print summary of saved files
            print(f"\n✅ Improvement plots saved to: {args.output_dir}/")
            print("Files created:")
            for file in improvement_files:
                print(f"  - {file}")
        else:
            # Calculate regular statistics
            stats = calculate_statistics(df, use_median=args.median)

            # Print summary
            print_summary_statistics(stats, use_median=args.median)

            # Create separate plots
            print(
                f"\nCreating separate plots (stat={stat_type}, std_dev={'on' if show_std else 'off'}, curve={'on' if args.curve else 'off'})..."
            )
            separate_files = create_plots(
                df,
                stats,
                args.output_dir,
                args.time_range,
                show_std,
                args.median,
                args.curve,
                background_color,
            )

            # Create combined plot
            print(
                f"\nCreating combined plot (stat={stat_type}, std_dev={'on' if show_std else 'off'}, curve={'on' if args.curve else 'off'})..."
            )
            combined_files = create_combined_plot(
                df,
                stats,
                args.output_dir,
                args.time_range,
                show_std,
                args.median,
                args.curve,
                background_color,
            )

            # Print summary of saved files
            print(f"\n✅ All plots saved to: {args.output_dir}/")
            print("Files created:")
            for file in separate_files + combined_files:
                print(f"  - {file}")

    except Exception as e:
        builtins.print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
