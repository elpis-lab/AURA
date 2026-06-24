#!/usr/bin/env python3
"""Average workspace replay metrics over runs and plot 3D metric surfaces."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict
from typing import Iterable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aura_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np

from utils.configHandler import DEFAULT_CONFIG_PATH, load_experiment_config

plt = None

SYSTEM_NAME = "kinematic_car"
AURA_ORANGE = "#EB5B00"
AURA_GREEN = "#A0C878"
AURA_BLUE = "#143D60"
TEXT_DARK = "#17202A"
TEXT_MUTED = "#586575"
GRID_COLOR = "#D8DEE7"
PANE_COLOR = "#F7F9FC"
FONT_FAMILY = "Times New Roman"
BASE_FONT_SIZE = 20
TITLE_FONT_SIZE = 30
SUBTITLE_FONT_SIZE = 20
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 17
SUMMARY_FONT_SIZE = 17
REPLAY_RE = re.compile(
    r"(?P<planner>.+)_cd(?P<control_duration>-?\d+(?:\.\d+)?)_r(?P<run_number>\d+)_"
    r"pt(?P<planning_time>-?\d+(?:\.\d+)?)\.npz$"
)


def _configure_matplotlib(*, show: bool) -> str:
    import matplotlib

    if show:
        configured = False
        env_backend = (os.environ.get("MPLBACKEND") or "").strip()
        candidates = [env_backend] if env_backend else []
        for backend in ("TkAgg", "Qt5Agg", "QtAgg", "Gtk3Agg"):
            if backend not in candidates:
                candidates.append(backend)
        for backend in candidates:
            if not backend:
                continue
            try:
                matplotlib.use(backend, force=True)
                configured = True
                break
            except Exception:
                continue
        if not configured:
            matplotlib.use("Agg", force=True)
            print(
                "[surfaces] No interactive matplotlib backend worked; saving PNG only."
            )
    else:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as pyplot

    pyplot.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": [FONT_FAMILY, "Times", "DejaVu Serif"],
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        }
    )
    global plt
    plt = pyplot
    return matplotlib.get_backend()


def _safe_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _sum_floats(values: Iterable) -> float:
    total = 0.0
    for value in values or []:
        fvalue = _safe_float(value)
        if fvalue is not None:
            total += fvalue
    return total


def _sanitize_metric_name(metric: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", metric).strip("_")


def _metric_label(metric: str) -> str:
    aliases = {
        "cost": "Total controls",
        "total_cost": "Total controls",
        "success_rate": "Success rate",
        "aura_time": "AURA Wall Time",
        "planning_time": "Initial planning time",
        "tracking_error": "Tracking error",
        "goal_distance": "Goal distance",
        "planned_goal_distance": "Planned goal distance",
        "num_controls": "Executed controls",
        "replay_total_cost": "Replay total cost",
        "replay_frame_count": "Replay frames",
        "replay_goal_solution_count": "Goal solution count",
        "replay_candidate_count": "Candidate path count",
    }
    if metric in aliases:
        return aliases[metric]
    return metric.replace("_", " ").strip().title()


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    rounded = round(float(value))
    if abs(float(value) - rounded) <= 1e-9:
        return str(int(rounded))
    if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.2e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _resolve_cmap(cmap_name: str):
    if plt is None:
        raise RuntimeError("matplotlib has not been configured")
    if str(cmap_name).lower() == "aura":
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(
            "aura_surface",
            [
                (0.00, AURA_BLUE),
                (0.48, AURA_GREEN),
                (1.00, AURA_ORANGE),
            ],
        )
    return plt.get_cmap(cmap_name)


def _style_3d_axes(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANE_COLOR)
        axis.pane.set_edgecolor("#C8D0DA")
        axis.pane.set_alpha(0.62)
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linewidth"] = 0.75
        axis._axinfo["axisline"]["color"] = "#7B8794"
        axis._axinfo["tick"]["color"] = "#7B8794"
    ax.tick_params(colors=TEXT_MUTED, pad=2)


def _z_axis_extreme_ticks(z_min: float, z_max: float) -> tuple[list[float], list[str]]:
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        return [], []
    if abs(float(z_max) - float(z_min)) <= 1e-12:
        return [float(z_min)], [f"min=max {float(z_min):.2f}"]
    return (
        [float(z_min), float(z_max)],
        [f"min {float(z_min):.2f}", f"max {float(z_max):.2f}"],
    )


def _apply_surface_axis_text(ax, *, z_min: float, z_max: float) -> None:
    z_ticks, z_labels = _z_axis_extreme_ticks(z_min, z_max)
    if z_ticks:
        ax.set_zticks(z_ticks)
        ax.set_zticklabels(z_labels)
    ax.xaxis.label.set_size(BASE_FONT_SIZE)
    ax.yaxis.label.set_size(BASE_FONT_SIZE)
    ax.zaxis.label.set_size(BASE_FONT_SIZE)
    for axis_name in ("x", "y"):
        ax.tick_params(axis=axis_name, which="major", labelsize=TICK_FONT_SIZE, pad=6)
    ax.tick_params(axis="z", which="major", labelsize=TICK_FONT_SIZE, pad=24)


def _draw_extreme_guides(
    ax,
    z_arrays: list[np.ndarray],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    control_durations: list[float],
    planning_times: list[float],
) -> None:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    for z_values in z_arrays:
        mask = np.isfinite(z_values)
        if not np.any(mask):
            continue
        xs.append(np.asarray(x_grid[mask], dtype=float).reshape(-1))
        ys.append(np.asarray(y_grid[mask], dtype=float).reshape(-1))
        zs.append(np.asarray(z_values[mask], dtype=float).reshape(-1))
    if not zs:
        return

    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    all_z = np.concatenate(zs)
    min_idx = int(np.argmin(all_z))
    max_idx = int(np.argmax(all_z))
    cd_min = float(min(control_durations))
    cd_max = float(max(control_durations))
    pt_min = float(min(planning_times))
    pt_max = float(max(planning_times))
    axis_x = cd_min
    axis_y = pt_max

    if abs(float(all_z[max_idx]) - float(all_z[min_idx])) <= 1e-12:
        callouts = [("min=max", min_idx, AURA_GREEN)]
    else:
        callouts = [("min", min_idx, AURA_BLUE), ("max", max_idx, AURA_ORANGE)]

    ax.plot(
        [axis_x, axis_x],
        [axis_y, axis_y],
        [float(np.min(all_z)), float(np.max(all_z))],
        color=TEXT_DARK,
        linewidth=1.5,
        alpha=0.42,
        zorder=17,
    )
    for _, idx, color in callouts:
        x_val = float(all_x[idx])
        y_val = float(all_y[idx])
        z_val = float(all_z[idx])
        ax.plot(
            [x_val, axis_x],
            [y_val, axis_y],
            [z_val, z_val],
            color=color,
            linestyle=(0, (5, 4)),
            linewidth=2.2,
            alpha=0.92,
            zorder=18,
        )
        ax.scatter(
            [x_val],
            [y_val],
            [z_val],
            color=color,
            marker="D",
            s=95,
            edgecolor="white",
            linewidth=1.2,
            depthshade=False,
            zorder=19,
        )


def _apply_xy_grid_limits(
    ax,
    control_durations: list[float],
    planning_times: list[float],
) -> None:
    cd_min = float(min(control_durations))
    cd_max = float(max(control_durations))
    pt_min = float(min(planning_times))
    pt_max = float(max(planning_times))
    if abs(cd_max - cd_min) <= 1e-12:
        ax.set_xlim(cd_min - 0.5, cd_max + 0.5)
    else:
        ax.set_xlim(cd_min, cd_max)
    if abs(pt_max - pt_min) <= 1e-12:
        ax.set_ylim(pt_min - 0.5, pt_max + 0.5)
    else:
        ax.set_ylim(pt_min, pt_max)


def _z_limits(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    z_min = float(np.min(finite))
    z_max = float(np.max(finite))
    span = z_max - z_min
    if span <= 1e-12:
        pad = max(1.0, abs(z_max) * 0.10)
    else:
        pad = 0.12 * span
    z_floor = z_min - pad
    return z_floor, z_min, z_max + pad


def _load_csv_index(
    results_dir: str, planner_name: str
) -> dict[tuple[float, int, float], dict]:
    csv_index: dict[tuple[float, int, float], dict] = {}
    pattern = os.path.join(results_dir, f"{SYSTEM_NAME}_{planner_name}_cd*_*.csv")
    file_re = re.compile(
        rf"{SYSTEM_NAME}_{re.escape(planner_name)}_cd(?P<cd>-?\d+(?:\.\d+)?)_(?P<run>\d+)\.csv$"
    )
    for csv_path in glob.glob(pattern):
        match = file_re.search(os.path.basename(csv_path))
        if not match:
            continue
        cd = round(float(match.group("cd")), 2)
        run_number = int(match.group("run"))
        try:
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    pt = _safe_float(row.get("planning_time"))
                    if pt is None:
                        continue
                    row_run = _safe_float(row.get("run_number"))
                    key = (
                        cd,
                        int(row_run) if row_run is not None else run_number,
                        round(pt, 2),
                    )
                    csv_index[key] = row
        except OSError as exc:
            print(f"[warning] could not read CSV {csv_path}: {exc}")
    return csv_index


def _read_replay_record(replay_path: str, csv_row: dict | None = None) -> dict | None:
    match = REPLAY_RE.search(os.path.basename(replay_path))
    if not match:
        return None
    record: dict = {
        "planner": match.group("planner"),
        "control_duration": float(match.group("control_duration")),
        "run_number": int(match.group("run_number")),
        "planning_time": float(match.group("planning_time")),
    }
    try:
        with np.load(replay_path, allow_pickle=True) as data:
            metadata = data["metadata"].item() if "metadata" in data.files else {}
            frames = list(data["frames"]) if "frames" in data.files else []
    except Exception as exc:
        print(f"[warning] skipping unreadable replay {replay_path}: {exc}")
        return None

    if not frames:
        print(f"[warning] skipping empty replay {replay_path}")
        return None

    final_frame = frames[-1]
    pose = np.asarray(final_frame.get("pose", []), dtype=float).reshape(-1)
    states = final_frame.get("states") or []
    controls = final_frame.get("controls") or []
    durations = final_frame.get("time") or []
    actual_states = final_frame.get("actual_states") or []
    candidates = final_frame.get("candidate_paths") or []
    goal_solutions = final_frame.get("goal_solution_paths") or []

    replay_metrics = {
        "replay_frame_count": float(len(frames)),
        "replay_final_step": _safe_float(final_frame.get("step")),
        "replay_final_is_final": 1.0 if bool(final_frame.get("is_final")) else 0.0,
        "replay_cost": _safe_float(final_frame.get("cost")),
        "replay_total_cost": _safe_float(final_frame.get("total_cost")),
        "replay_executed_control_count": _safe_float(
            final_frame.get("executed_control_count")
        ),
        "replay_remaining_control_count": _safe_float(
            final_frame.get("remaining_control_count")
        ),
        "replay_executed_path_cost": _safe_float(final_frame.get("executed_path_cost")),
        "replay_remaining_path_cost": _safe_float(
            final_frame.get("remaining_path_cost")
        ),
        "replay_remaining_plan_time": _sum_floats(durations),
        "replay_final_plan_state_count": float(len(states)),
        "replay_final_plan_control_count": float(len(controls)),
        "replay_actual_sample_count": float(len(actual_states)),
        "replay_candidate_count": float(len(candidates)),
        "replay_goal_solution_count": float(len(goal_solutions)),
        "replay_curve_step_size": _safe_float(metadata.get("curve_step_size")),
    }
    for key, value in replay_metrics.items():
        if value is not None:
            record[key] = float(value)
    if pose.size >= 1:
        record["replay_final_x"] = float(pose[0])
    if pose.size >= 2:
        record["replay_final_y"] = float(pose[1])
    if pose.size >= 3:
        record["replay_final_theta"] = float(pose[2])

    if csv_row:
        for key, value in csv_row.items():
            if key in {"planner", "failure_reason"}:
                continue
            if key == "status":
                status_text = str(value).strip().lower()
                record["status"] = status_text
                record["success_rate"] = 1.0 if status_text == "success" else 0.0
                continue
            fvalue = _safe_float(value)
            if fvalue is not None:
                if key in record and key not in {
                    "control_duration",
                    "planning_time",
                    "run_number",
                }:
                    record[f"csv_{key}"] = fvalue
                    continue
                record[key] = fvalue

    # User-facing aliases. In these replay plots, "cost" should mean the
    # accumulated cost shown in the workspace video badge, not the older CSV
    # path-distance column.
    if "replay_total_cost" in record:
        record["cost"] = float(record["replay_total_cost"])
        record["total_cost"] = float(record["replay_total_cost"])
    elif "replay_cost" in record:
        record["cost"] = float(record["replay_cost"])

    return record


def _collect_records(args, *, include_failures: bool) -> list[dict]:
    replay_dir = args.replay_dir or os.path.join(args.results_dir, "workspace_replays")
    csv_index = _load_csv_index(args.results_dir, args.planner_name)
    records: list[dict] = []
    skipped_failures = 0
    skipped_unknown_status = 0
    target_cds = (
        {round(float(v), 2) for v in args.control_durations}
        if args.control_durations
        else None
    )
    target_pts = (
        {round(float(v), 2) for v in args.planning_times}
        if args.planning_times
        else None
    )
    target_runs = set(range(1, int(args.num_runs) + 1)) if args.num_runs else None

    for replay_path in sorted(glob.glob(os.path.join(replay_dir, "*.npz"))):
        match = REPLAY_RE.search(os.path.basename(replay_path))
        if not match:
            continue
        planner = match.group("planner")
        cd = round(float(match.group("control_duration")), 2)
        run_number = int(match.group("run_number"))
        pt = round(float(match.group("planning_time")), 2)
        if planner != args.planner_name:
            continue
        if target_cds is not None and cd not in target_cds:
            continue
        if target_pts is not None and pt not in target_pts:
            continue
        if target_runs is not None and run_number not in target_runs:
            continue
        record = _read_replay_record(replay_path, csv_index.get((cd, run_number, pt)))
        if record is not None:
            if not bool(include_failures):
                success_rate = _safe_float(record.get("success_rate"))
                status = str(record.get("status", "")).strip().lower()
                is_success = success_rate == 1.0 or status == "success"
                if not is_success:
                    if success_rate is None and not status:
                        skipped_unknown_status += 1
                    else:
                        skipped_failures += 1
                    continue
            records.append(record)
    if not bool(include_failures):
        print(
            "[surfaces] Success-only filter: "
            f"kept {len(records)} successful replay(s), "
            f"skipped {skipped_failures} failure(s), "
            f"skipped {skipped_unknown_status} unknown-status replay(s)."
        )
    return records


def _run_surface_batch(
    *,
    records: list[dict],
    success_rate_records: list[dict],
    args,
    control_durations: list[float],
    planning_times: list[float],
    requested_metrics: list[str] | None,
    outdir: str,
    show: bool,
    mode_label: str,
    include_failures: bool,
) -> tuple[list[str], str | None, str | None]:
    if not records:
        print(f"[surfaces] {mode_label}: no matching records; skipping.")
        return [], None, None

    metrics = _metric_names(records, requested_metrics)
    if not metrics:
        print(f"[surfaces] {mode_label}: no numeric metrics found; skipping.")
        return [], None, None
    if not include_failures and "success_rate" in metrics:
        print(
            "[surfaces] Note: success_rate is computed after filtering to successful "
            "runs only, so it will be 1.0 wherever data exists. Use --include-failures "
            "or --both to plot the actual success rate."
        )

    os.makedirs(outdir, exist_ok=True)
    print(f"[surfaces] Loaded {len(records)} replay record(s).")
    print(f"[surfaces] Averaging mode: {mode_label}")
    print(f"[surfaces] Control durations: {control_durations}")
    print(f"[surfaces] Planning times: {planning_times}")
    print(f"[surfaces] Metrics: {', '.join(metrics)}")

    success_rate_grid, success_count_grid, total_count_grid, unknown_count_grid = (
        _aggregate_success_rate(
            success_rate_records,
            control_durations,
            planning_times,
        )
    )
    aggregates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    written: list[str] = []
    for metric in metrics:
        aggregates[metric] = _aggregate_metric(
            records,
            metric,
            control_durations,
            planning_times,
        )
        _print_metric_coverage(
            metric,
            aggregates[metric][2],
            control_durations,
            planning_times,
        )
        _print_grid_values(
            metric,
            aggregates[metric][0],
            aggregates[metric][2],
            success_rate_grid,
            success_count_grid,
            total_count_grid,
            unknown_count_grid,
            control_durations,
            planning_times,
            mode_label=mode_label,
        )
        out_path = _plot_surface(
            metric,
            aggregates[metric][0],
            aggregates[metric][2],
            control_durations,
            planning_times,
            outdir,
            cmap=args.cmap,
            dpi=int(args.dpi),
            show=show,
        )
        if out_path:
            written.append(out_path)

    summary_csv = os.path.join(outdir, "metric_surface_summary.csv")
    records_csv = os.path.join(outdir, "metric_records.csv")
    _write_summary_csv(
        summary_csv, metrics, aggregates, control_durations, planning_times
    )
    _write_records_csv(records_csv, records)

    print(f"[surfaces] Wrote {len(written)} surface plot(s) to {outdir}")
    print(f"[surfaces] Summary CSV: {summary_csv}")
    print(f"[surfaces] Per-run records CSV: {records_csv}")
    return written, summary_csv, records_csv


def _metric_names(records: list[dict], requested: list[str] | None) -> list[str]:
    skip = {"control_duration", "planning_time", "run_number"}
    names = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if key not in skip and _safe_float(value) is not None
        }
    )
    if requested:
        wanted = set(requested)
        names = [name for name in names if name in wanted]
        missing = sorted(wanted.difference(names))
        if missing:
            print(f"[warning] requested metric(s) not found: {', '.join(missing)}")
    return names


def _aggregate_metric(
    records: list[dict],
    metric: str,
    control_durations: list[float],
    planning_times: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: dict[tuple[float, float], list[float]] = defaultdict(list)
    for record in records:
        value = _safe_float(record.get(metric))
        if value is None:
            continue
        key = (
            round(float(record["control_duration"]), 2),
            round(float(record["planning_time"]), 2),
        )
        values[key].append(value)

    z_mean = np.full((len(planning_times), len(control_durations)), np.nan, dtype=float)
    z_std = np.full_like(z_mean, np.nan)
    z_count = np.zeros_like(z_mean)
    for iy, pt in enumerate(planning_times):
        for ix, cd in enumerate(control_durations):
            vals = np.asarray(
                values.get((round(float(cd), 2), round(float(pt), 2)), []), dtype=float
            )
            vals = vals[np.isfinite(vals)]
            if vals.size:
                z_mean[iy, ix] = float(np.mean(vals))
                z_std[iy, ix] = float(np.std(vals))
                z_count[iy, ix] = float(vals.size)
    return z_mean, z_std, z_count


def _record_success_status(record: dict) -> tuple[bool, bool]:
    success_rate = _safe_float(record.get("success_rate"))
    if success_rate is not None:
        return success_rate >= 1.0, True

    status = str(record.get("status", "")).strip().lower()
    if status:
        return status == "success", True
    return False, False


def _aggregate_success_rate(
    records: list[dict],
    control_durations: list[float],
    planning_times: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    successes: dict[tuple[float, float], int] = defaultdict(int)
    totals: dict[tuple[float, float], int] = defaultdict(int)
    unknowns: dict[tuple[float, float], int] = defaultdict(int)
    for record in records:
        key = (
            round(float(record["control_duration"]), 2),
            round(float(record["planning_time"]), 2),
        )
        is_success, known = _record_success_status(record)
        totals[key] += 1
        if is_success:
            successes[key] += 1
        if not known:
            unknowns[key] += 1

    rate = np.full((len(planning_times), len(control_durations)), np.nan, dtype=float)
    success_count = np.zeros_like(rate)
    total_count = np.zeros_like(rate)
    unknown_count = np.zeros_like(rate)
    for iy, pt in enumerate(planning_times):
        for ix, cd in enumerate(control_durations):
            key = (round(float(cd), 2), round(float(pt), 2))
            total = int(totals.get(key, 0))
            success = int(successes.get(key, 0))
            unknown = int(unknowns.get(key, 0))
            success_count[iy, ix] = float(success)
            total_count[iy, ix] = float(total)
            unknown_count[iy, ix] = float(unknown)
            if total > 0:
                rate[iy, ix] = float(success) / float(total)
    return rate, success_count, total_count, unknown_count


def _format_grid_number(value: float) -> str:
    return _format_value(float(value)) if np.isfinite(value) else "missing"


def _format_success_rate(
    rate: float,
    success_count: float,
    total_count: float,
    unknown_count: float,
) -> str:
    total = int(total_count)
    success = int(success_count)
    unknown = int(unknown_count)
    if total <= 0 or not np.isfinite(rate):
        return "missing"
    suffix = f", {unknown} unknown" if unknown else ""
    return f"{success}/{total} ({100.0 * float(rate):.1f}%{suffix})"


def _print_grid_values(
    metric: str,
    z_mean: np.ndarray,
    z_count: np.ndarray,
    success_rate: np.ndarray,
    success_count: np.ndarray,
    total_count: np.ndarray,
    unknown_count: np.ndarray,
    control_durations: list[float],
    planning_times: list[float],
    *,
    mode_label: str,
) -> None:
    label = _metric_label(metric)
    print(f"\n[surfaces] Grid values for {metric} ({label}) [{mode_label}]")
    print(f"{'cd':>8}  {'pt':>8}  {'mean':>14}  {'plot_n':>7}  {'success_rate':>24}")
    for iy, pt in enumerate(planning_times):
        for ix, cd in enumerate(control_durations):
            print(
                f"{_format_value(float(cd)):>8}  "
                f"{_format_value(float(pt)):>8}  "
                f"{_format_grid_number(z_mean[iy, ix]):>14}  "
                f"{int(z_count[iy, ix]):>7}  "
                f"{_format_success_rate(success_rate[iy, ix], success_count[iy, ix], total_count[iy, ix], unknown_count[iy, ix]):>24}"
            )


def _print_overlay_grid_values(
    metric: str,
    success_mean: np.ndarray,
    success_count_for_metric: np.ndarray,
    all_mean: np.ndarray,
    all_count_for_metric: np.ndarray,
    success_rate: np.ndarray,
    success_count: np.ndarray,
    total_count: np.ndarray,
    unknown_count: np.ndarray,
    control_durations: list[float],
    planning_times: list[float],
) -> None:
    label = _metric_label(metric)
    print(f"\n[surfaces] Grid values for {metric} ({label}) [--both overlay]")
    print(
        f"{'cd':>8}  {'pt':>8}  {'success_mean':>14}  {'success_n':>9}  "
        f"{'all_mean':>14}  {'all_n':>6}  {'success_rate':>24}"
    )
    for iy, pt in enumerate(planning_times):
        for ix, cd in enumerate(control_durations):
            print(
                f"{_format_value(float(cd)):>8}  "
                f"{_format_value(float(pt)):>8}  "
                f"{_format_grid_number(success_mean[iy, ix]):>14}  "
                f"{int(success_count_for_metric[iy, ix]):>9}  "
                f"{_format_grid_number(all_mean[iy, ix]):>14}  "
                f"{int(all_count_for_metric[iy, ix]):>6}  "
                f"{_format_success_rate(success_rate[iy, ix], success_count[iy, ix], total_count[iy, ix], unknown_count[iy, ix]):>24}"
            )


def _plot_surface(
    metric: str,
    z_mean: np.ndarray,
    z_count: np.ndarray,
    control_durations: list[float],
    planning_times: list[float],
    outdir: str,
    *,
    cmap: str,
    dpi: int,
    show: bool,
) -> str | None:
    if plt is None:
        raise RuntimeError("matplotlib has not been configured")
    mask = np.isfinite(z_mean)
    if not np.any(mask):
        return None

    from matplotlib.colors import Normalize

    x_grid, y_grid = np.meshgrid(control_durations, planning_times)
    cmap_obj = _resolve_cmap(cmap)
    z_floor, z_min, z_top = _z_limits(z_mean)
    z_values = z_mean[mask]
    z_max = float(np.max(z_values))
    norm = Normalize(vmin=z_min, vmax=z_max if z_max > z_min else z_min + 1.0)

    fig = plt.figure(figsize=(13.5, 9.6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    z_plot = np.ma.masked_invalid(z_mean)
    surface = None
    if np.count_nonzero(mask) >= 3:
        surface = ax.plot_surface(
            x_grid,
            y_grid,
            z_plot,
            cmap=cmap_obj,
            norm=norm,
            linewidth=0.55,
            edgecolor=(1.0, 1.0, 1.0, 0.70),
            antialiased=True,
            alpha=0.94,
            rstride=1,
            cstride=1,
            shade=True,
        )
        if z_max > z_min:
            ax.contourf(
                x_grid,
                y_grid,
                z_plot,
                zdir="z",
                offset=z_floor,
                cmap=cmap_obj,
                norm=norm,
                levels=14,
                alpha=0.22,
            )
            ax.contour(
                x_grid,
                y_grid,
                z_plot,
                zdir="z",
                offset=z_floor + 1e-9,
                colors="#2E3A47",
                linewidths=0.45,
                alpha=0.35,
                levels=8,
            )

    point_sizes = 44.0 + 9.0 * np.sqrt(np.maximum(z_count[mask], 1.0))
    ax.scatter(
        x_grid[mask],
        y_grid[mask],
        z_mean[mask],
        c=z_mean[mask],
        cmap=cmap_obj,
        norm=norm,
        s=point_sizes,
        edgecolor="white",
        linewidth=1.05,
        depthshade=False,
        label="_nolegend_",
        zorder=10,
    )
    for x_val, y_val, z_val in zip(x_grid[mask], y_grid[mask], z_mean[mask]):
        ax.plot(
            [float(x_val), float(x_val)],
            [float(y_val), float(y_val)],
            [z_floor, float(z_val)],
            color="#394554",
            linewidth=0.8,
            alpha=0.28,
            zorder=2,
        )

    missing_mask = ~mask
    if np.any(missing_mask):
        ax.scatter(
            x_grid[missing_mask],
            y_grid[missing_mask],
            np.full(int(np.count_nonzero(missing_mask)), z_floor),
            marker="x",
            color="#8A96A3",
            s=26,
            alpha=0.72,
            depthshade=False,
            label="Missing grid cell",
        )

    ax.set_xlabel("Control duration", labelpad=18)
    ax.set_ylabel("Initial planning time", labelpad=20)
    ax.set_zlabel(_metric_label(metric), labelpad=18)
    ax.set_xticks(control_durations)
    ax.set_yticks(planning_times)
    ax.set_yticklabels([_format_value(float(value)) for value in planning_times])
    ax.set_zlim(z_floor, z_top)
    ax.view_init(elev=27, azim=-128)
    try:
        ax.set_box_aspect((1.10, 1.0, 0.72))
    except Exception:
        pass
    _style_3d_axes(ax)
    _apply_surface_axis_text(ax, z_min=z_min, z_max=z_max)
    _draw_extreme_guides(
        ax,
        [z_mean],
        x_grid,
        y_grid,
        control_durations,
        planning_times,
    )
    _apply_xy_grid_limits(ax, control_durations, planning_times)

    coverage = int(np.sum(z_count))
    grid_count = int(np.count_nonzero(mask))
    total_grid_count = int(z_mean.size)
    summary = (
        f"{grid_count}/{total_grid_count} grid cells  |  "
        f"{coverage} samples  |  "
        f"mean {_format_value(float(np.mean(z_values)))}"
    )
    fig.suptitle(
        _metric_label(metric),
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        color=TEXT_DARK,
    )
    fig.text(
        0.055,
        0.925,
        "Mean over runs by control duration and initial planning time",
        ha="left",
        va="top",
        fontsize=SUBTITLE_FONT_SIZE,
        color=TEXT_MUTED,
    )
    ax.text2D(
        0.025,
        0.015,
        summary,
        transform=ax.transAxes,
        fontsize=SUMMARY_FONT_SIZE,
        color=TEXT_MUTED,
        bbox={
            "boxstyle": "round,pad=0.38,rounding_size=0.12",
            "facecolor": "white",
            "edgecolor": "#D4DAE2",
            "linewidth": 0.8,
            "alpha": 0.94,
        },
    )
    handles, labels = ax.get_legend_handles_labels()
    visible_handles = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label and not label.startswith("_")
    ]
    if visible_handles:
        legend = ax.legend(
            [handle for handle, _ in visible_handles],
            [label for _, label in visible_handles],
            loc="upper left",
            bbox_to_anchor=(0.025, 0.895),
            framealpha=0.92,
            borderpad=0.55,
            handletextpad=0.6,
            prop={"size": LEGEND_FONT_SIZE},
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#D4DAE2")
        legend.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.04)
    out_path = os.path.join(outdir, f"{_sanitize_metric_name(metric)}_surface.png")
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    if show:
        print(f"[surfaces] Interactive plot ready for metric: {metric}")
        plt.show(block=True)
    plt.close(fig)
    return out_path


def _plot_surface_overlay(
    metric: str,
    success_mean: np.ndarray,
    success_count: np.ndarray,
    all_mean: np.ndarray,
    all_count: np.ndarray,
    control_durations: list[float],
    planning_times: list[float],
    outdir: str,
    *,
    dpi: int,
    show: bool,
) -> str | None:
    if plt is None:
        raise RuntimeError("matplotlib has not been configured")

    success_mask = np.isfinite(success_mean)
    all_mask = np.isfinite(all_mean)
    if not np.any(success_mask) and not np.any(all_mask):
        return None

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    x_grid, y_grid = np.meshgrid(control_durations, planning_times)
    combined = np.concatenate(
        [
            success_mean[success_mask].reshape(-1),
            all_mean[all_mask].reshape(-1),
        ]
    )
    z_floor, z_min, z_top = _z_limits(combined)
    z_max = float(np.max(combined[np.isfinite(combined)]))

    fig = plt.figure(figsize=(13.5, 9.6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    def _draw_layer(
        z_values: np.ndarray,
        z_count: np.ndarray,
        mask: np.ndarray,
        *,
        color: str,
        label: str,
        marker: str,
        alpha: float,
        zorder: int,
    ) -> None:
        if not np.any(mask):
            return
        z_plot = np.ma.masked_invalid(z_values)
        if np.count_nonzero(mask) >= 3:
            ax.plot_surface(
                x_grid,
                y_grid,
                z_plot,
                color=color,
                linewidth=0.55,
                edgecolor=(1.0, 1.0, 1.0, 0.55),
                antialiased=True,
                alpha=alpha,
                rstride=1,
                cstride=1,
                shade=True,
            )
        point_sizes = 46.0 + 9.0 * np.sqrt(np.maximum(z_count[mask], 1.0))
        ax.scatter(
            x_grid[mask],
            y_grid[mask],
            z_values[mask],
            color=color,
            marker=marker,
            s=point_sizes,
            edgecolor="white",
            linewidth=1.05,
            depthshade=False,
            label=label,
            zorder=zorder,
        )
        for x_val, y_val, z_val in zip(x_grid[mask], y_grid[mask], z_values[mask]):
            ax.plot(
                [float(x_val), float(x_val)],
                [float(y_val), float(y_val)],
                [z_floor, float(z_val)],
                color=color,
                linewidth=0.75,
                alpha=0.18,
                zorder=2,
            )

    _draw_layer(
        all_mean,
        all_count,
        all_mask,
        color=AURA_ORANGE,
        label="All runs (includes failures)",
        marker="^",
        alpha=0.34,
        zorder=9,
    )
    _draw_layer(
        success_mean,
        success_count,
        success_mask,
        color=AURA_BLUE,
        label="Successful runs only (failures excluded)",
        marker="o",
        alpha=0.50,
        zorder=10,
    )

    missing_mask = ~(success_mask | all_mask)
    if np.any(missing_mask):
        ax.scatter(
            x_grid[missing_mask],
            y_grid[missing_mask],
            np.full(int(np.count_nonzero(missing_mask)), z_floor),
            marker="x",
            color="#8A96A3",
            s=26,
            alpha=0.72,
            depthshade=False,
            label="Missing grid cell",
        )

    ax.set_xlabel("Control duration", labelpad=18)
    ax.set_ylabel("Initial planning time", labelpad=20)
    ax.set_zlabel(_metric_label(metric), labelpad=18)
    ax.set_xticks(control_durations)
    ax.set_yticks(planning_times)
    ax.set_yticklabels([_format_value(float(value)) for value in planning_times])
    ax.set_zlim(z_floor, z_top)
    ax.view_init(elev=27, azim=-128)
    try:
        ax.set_box_aspect((1.10, 1.0, 0.72))
    except Exception:
        pass
    _style_3d_axes(ax)
    _apply_surface_axis_text(ax, z_min=z_min, z_max=z_max)
    _draw_extreme_guides(
        ax,
        [success_mean, all_mean],
        x_grid,
        y_grid,
        control_durations,
        planning_times,
    )
    _apply_xy_grid_limits(ax, control_durations, planning_times)

    success_samples = int(np.sum(success_count))
    all_samples = int(np.sum(all_count))
    success_cells = int(np.count_nonzero(success_mask))
    all_cells = int(np.count_nonzero(all_mask))
    summary = (
        f"successful only: {success_cells}/{success_mean.size} cells, "
        f"{success_samples} samples  |  all runs: {all_cells}/{all_mean.size} cells, "
        f"{all_samples} samples"
    )
    fig.suptitle(
        f"{_metric_label(metric)}: Successful vs All Runs",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        color=TEXT_DARK,
    )
    fig.text(
        0.055,
        0.925,
        "Blue excludes failed/unknown-status runs; orange includes every matching replay",
        ha="left",
        va="top",
        fontsize=SUBTITLE_FONT_SIZE,
        color=TEXT_MUTED,
    )
    ax.text2D(
        0.025,
        0.015,
        summary,
        transform=ax.transAxes,
        fontsize=SUMMARY_FONT_SIZE,
        color=TEXT_MUTED,
        bbox={
            "boxstyle": "round,pad=0.38,rounding_size=0.12",
            "facecolor": "white",
            "edgecolor": "#D4DAE2",
            "linewidth": 0.8,
            "alpha": 0.94,
        },
    )

    handles = [
        Patch(
            facecolor=AURA_BLUE,
            edgecolor="white",
            alpha=0.50,
            label="Successful runs only (failures excluded)",
        ),
        Patch(
            facecolor=AURA_ORANGE,
            edgecolor="white",
            alpha=0.34,
            label="All runs (includes failures)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=AURA_BLUE,
            markeredgecolor="white",
            markersize=7,
            label="Success-only mean point",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=AURA_ORANGE,
            markeredgecolor="white",
            markersize=7,
            label="All-runs mean point",
        ),
    ]
    if np.any(missing_mask):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                color="#8A96A3",
                linestyle="None",
                markersize=6,
                label="Missing grid cell",
            )
        )
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.025, 0.895),
        framealpha=0.92,
        borderpad=0.55,
        handletextpad=0.6,
        prop={"size": LEGEND_FONT_SIZE},
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#D4DAE2")
    legend.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.04)
    out_path = os.path.join(outdir, f"{_sanitize_metric_name(metric)}_both_surface.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    if show:
        print(f"[surfaces] Interactive comparison plot ready for metric: {metric}")
        plt.show(block=True)
    plt.close(fig)
    return out_path


def _run_overlay_batch(
    *,
    success_records: list[dict],
    all_records: list[dict],
    args,
    control_durations: list[float],
    planning_times: list[float],
    requested_metrics: list[str] | None,
    outdir: str,
    show: bool,
) -> tuple[list[str], str | None, str | None]:
    if not all_records:
        print("[surfaces] both: no matching records; skipping.")
        return [], None, None

    metrics = _metric_names(all_records, requested_metrics)
    if not metrics:
        print("[surfaces] both: no numeric metrics found; skipping.")
        return [], None, None

    os.makedirs(outdir, exist_ok=True)
    non_success_records = max(0, len(all_records) - len(success_records))
    print(f"[surfaces] Loaded {len(success_records)} successful replay record(s).")
    print(f"[surfaces] Loaded {len(all_records)} total replay record(s).")
    print(
        "[surfaces] --both overlay: blue surface excludes failures; "
        f"orange surface includes all matching replays ({non_success_records} "
        "failed/unknown-status replay(s) in the all-runs layer)."
    )
    print(f"[surfaces] Control durations: {control_durations}")
    print(f"[surfaces] Planning times: {planning_times}")
    print(f"[surfaces] Metrics: {', '.join(metrics)}")

    success_rate_grid, status_success_count, total_count_grid, unknown_count_grid = (
        _aggregate_success_rate(
            all_records,
            control_durations,
            planning_times,
        )
    )
    success_aggregates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    all_aggregates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    written: list[str] = []
    for metric in metrics:
        success_aggregates[metric] = _aggregate_metric(
            success_records,
            metric,
            control_durations,
            planning_times,
        )
        all_aggregates[metric] = _aggregate_metric(
            all_records,
            metric,
            control_durations,
            planning_times,
        )
        _print_metric_coverage(
            f"{metric} [successful only]",
            success_aggregates[metric][2],
            control_durations,
            planning_times,
        )
        _print_metric_coverage(
            f"{metric} [all runs]",
            all_aggregates[metric][2],
            control_durations,
            planning_times,
        )
        _print_overlay_grid_values(
            metric,
            success_aggregates[metric][0],
            success_aggregates[metric][2],
            all_aggregates[metric][0],
            all_aggregates[metric][2],
            success_rate_grid,
            status_success_count,
            total_count_grid,
            unknown_count_grid,
            control_durations,
            planning_times,
        )
        out_path = _plot_surface_overlay(
            metric,
            success_aggregates[metric][0],
            success_aggregates[metric][2],
            all_aggregates[metric][0],
            all_aggregates[metric][2],
            control_durations,
            planning_times,
            outdir,
            dpi=int(args.dpi),
            show=show,
        )
        if out_path:
            written.append(out_path)

    success_summary_csv = os.path.join(
        outdir, "metric_surface_summary_success_only.csv"
    )
    all_summary_csv = os.path.join(outdir, "metric_surface_summary_all_runs.csv")
    success_records_csv = os.path.join(outdir, "metric_records_success_only.csv")
    all_records_csv = os.path.join(outdir, "metric_records_all_runs.csv")
    _write_summary_csv(
        success_summary_csv,
        metrics,
        success_aggregates,
        control_durations,
        planning_times,
    )
    _write_summary_csv(
        all_summary_csv,
        metrics,
        all_aggregates,
        control_durations,
        planning_times,
    )
    _write_records_csv(success_records_csv, success_records)
    _write_records_csv(all_records_csv, all_records)

    print(f"[surfaces] Wrote {len(written)} overlaid surface plot(s) to {outdir}")
    print(f"[surfaces] Success-only summary CSV: {success_summary_csv}")
    print(f"[surfaces] All-runs summary CSV: {all_summary_csv}")
    return written, success_summary_csv, all_summary_csv


def _print_metric_coverage(
    metric: str,
    z_count: np.ndarray,
    control_durations: list[float],
    planning_times: list[float],
) -> None:
    present: list[tuple[float, float, int]] = []
    missing: list[tuple[float, float]] = []
    for iy, planning_time in enumerate(planning_times):
        for ix, control_duration in enumerate(control_durations):
            count = int(z_count[iy, ix])
            if count > 0:
                present.append((float(control_duration), float(planning_time), count))
            else:
                missing.append((float(control_duration), float(planning_time)))

    print(
        f"[surfaces] {metric}: averaged grid points {len(present)}/{z_count.size}; "
        f"samples {sum(item[2] for item in present)}"
    )
    if missing:
        preview = ", ".join(f"cd{cd:.2f}/pt{pt:.2f}" for cd, pt in missing[:16])
        suffix = "" if len(missing) <= 16 else f", ... +{len(missing) - 16} more"
        print(f"[surfaces] {metric}: missing grid cells: {preview}{suffix}")


def _write_summary_csv(
    out_path: str,
    metrics: list[str],
    aggregates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    control_durations: list[float],
    planning_times: list[float],
) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "control_duration",
                "planning_time",
                "mean",
                "std",
                "count",
            ],
        )
        writer.writeheader()
        for metric in metrics:
            z_mean, z_std, z_count = aggregates[metric]
            for iy, pt in enumerate(planning_times):
                for ix, cd in enumerate(control_durations):
                    writer.writerow(
                        {
                            "metric": metric,
                            "control_duration": float(cd),
                            "planning_time": float(pt),
                            "mean": z_mean[iy, ix],
                            "std": z_std[iy, ix],
                            "count": int(z_count[iy, ix]),
                        }
                    )


def _write_records_csv(out_path: str, records: list[dict]) -> None:
    fieldnames = sorted({key for record in records for key in record.keys()})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Shared sweep YAML config.",
    )
    config_args, _ = config_parser.parse_known_args()
    sweep_config = load_experiment_config(config_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Read AURA workspace replay files, average metrics over runs, and generate "
            "3D surfaces with x=control duration, y=initial planning time, z=metric."
        ),
        parents=[config_parser],
    )
    parser.add_argument(
        "--results-dir",
        default=str(sweep_config["results_dir"]),
        help="Experiment results directory containing workspace_replays and CSV files.",
    )
    parser.add_argument(
        "--replay-dir",
        default=None,
        help="Override replay directory. Default: results-dir/workspace_replays.",
    )
    parser.add_argument(
        "--planner-name",
        default=str(sweep_config["planner_name"]),
        help="Planner prefix to read.",
    )
    parser.add_argument(
        "--control-durations",
        type=float,
        nargs="+",
        default=list(sweep_config["control_durations"]),
        help="Control durations for the x-axis.",
    )
    parser.add_argument(
        "--planning-times",
        type=float,
        nargs="+",
        default=list(sweep_config["planning_times"]),
        help="Initial planning times for the y-axis.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=int(sweep_config["num_runs"]),
        help="Use run numbers 1..N. Set 0 to include all run numbers found.",
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help=(
            "Include failed runs in metric averages. By default, surfaces average "
            "only runs whose CSV status is success."
        ),
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help=(
            "Overlay two surfaces in one plot: successful-runs-only averages "
            "(failures excluded) and all-runs averages (failures included)."
        ),
    )
    parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Single metric to plot interactively. Example: --metric replay_total_cost. "
            "The window pops up unless --no-show is set."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional metric names to plot. Default: every numeric replay/CSV metric.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: results-dir/workspace_metric_surfaces.",
    )
    parser.add_argument(
        "--cmap",
        default="aura",
        help="Matplotlib colormap, or 'aura' for the project palette.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output PNG DPI.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive matplotlib window(s). --metric enables this by default.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Never open a window; only save PNG/CSV outputs.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print available metric names and exit without plotting.",
    )
    args = parser.parse_args()
    if args.metric and args.metrics:
        raise SystemExit(
            "Use either --metric for one plot or --metrics for a batch, not both."
        )

    control_durations = sorted({round(float(v), 2) for v in args.control_durations})
    planning_times = sorted({round(float(v), 2) for v in args.planning_times})
    requested_metrics = [args.metric] if args.metric else args.metrics
    all_records_for_rates = _collect_records(args, include_failures=True)
    if bool(args.both):
        all_records_for_listing = all_records_for_rates
        if not all_records_for_listing:
            raise SystemExit("No matching workspace replay records found.")
        if args.list_metrics:
            metrics = _metric_names(all_records_for_listing, requested_metrics)
            if not metrics:
                raise SystemExit("No numeric metrics found to plot.")
            print(f"[surfaces] Metrics: {', '.join(metrics)}")
            return
    else:
        records = (
            all_records_for_rates
            if bool(args.include_failures)
            else _collect_records(args, include_failures=False)
        )
        if not records:
            raise SystemExit("No matching workspace replay records found.")
        if args.list_metrics:
            metrics = _metric_names(records, requested_metrics)
            if not metrics:
                raise SystemExit("No numeric metrics found to plot.")
            print(f"[surfaces] Metrics: {', '.join(metrics)}")
            return

    show = (bool(args.metric) or bool(args.show)) and not bool(args.no_show)
    backend = _configure_matplotlib(show=show)
    if show:
        print(f"[surfaces] Matplotlib backend: {backend}")

    outdir = args.outdir or os.path.join(args.results_dir, "workspace_metric_surfaces")
    if bool(args.both):
        success_records = _collect_records(args, include_failures=False)
        _run_overlay_batch(
            success_records=success_records,
            all_records=all_records_for_listing,
            args=args,
            control_durations=control_durations,
            planning_times=planning_times,
            requested_metrics=requested_metrics,
            outdir=outdir,
            show=show,
        )
    else:
        _run_surface_batch(
            records=records,
            success_rate_records=all_records_for_rates,
            args=args,
            control_durations=control_durations,
            planning_times=planning_times,
            requested_metrics=requested_metrics,
            outdir=outdir,
            show=show,
            mode_label=(
                "all runs" if bool(args.include_failures) else "successful runs only"
            ),
            include_failures=bool(args.include_failures),
        )


if __name__ == "__main__":
    main()
