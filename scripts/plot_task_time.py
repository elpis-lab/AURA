#!/usr/bin/env python3
"""Validate, summarize, and plot end-to-end task-time results."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "full_time_comparison"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_io import METHODS, PANEL_LAYOUT, PLANNERS, SIMULATION_PANEL_ORDER


# Match the publication styling used by the original plotExp2.py.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

PUBLICATION_FONT_SIZE = 11
plt.rcParams["font.size"] = PUBLICATION_FONT_SIZE

BOXPLOT_WIDTH = 0.34
BOX_CENTER_SPACING = 0.38
METHOD_GROUP_GAP = 0.15
SINGLE_METHOD_GAP = 0.42
METHOD_LABEL_SIZE = PUBLICATION_FONT_SIZE
PANEL_TITLE_SIZE = PUBLICATION_FONT_SIZE
TICK_LABEL_SIZE = PUBLICATION_FONT_SIZE
AXIS_LABEL_SIZE = PUBLICATION_FONT_SIZE
Y_TICK_LABEL_SIZE = PUBLICATION_FONT_SIZE + 1
Y_AXIS_LABEL_SIZE = PUBLICATION_FONT_SIZE + 1
METHOD_LABEL_ROTATION = 90
DEFAULT_METHOD_SEPARATORS = True
METHOD_SEPARATOR_DASH_PATTERN = (4.0, 6.0)
Y_AXIS_TICK_COUNT = 6
PANEL_WIDTH = 2.0
PANEL_HEIGHT = 4.2
PANEL_HORIZONTAL_PAD = 0.05
PANEL_WSPACE = 0.30
PANEL_TITLE_Y = 1.069
PANEL_SUBTITLE_Y = 1.025


# Show every simulated system followed by the real-world pushing panel.
FIGURE_7_PANEL_ORDER = (
    "double_integrator_gaussian",
    "kinematic_car_gaussian",
    "dubins_airplane_gaussian",
    "pushing_gaussian",
    "kinematic_car_mujoco",
    "pushing_mujoco",
    "pushing_real",
)


# These are visualization/penalty caps, not experiment deadlines.  Every run
# still retains its uncapped ``overall_time`` and the campaign's 300-second
# execution limit in the compact source CSV.
PANEL_PLOT_TIME_CAP_SECONDS = {
    "double_integrator_gaussian": 250.0,
    "kinematic_car_gaussian": 300.0,
    "pushing_gaussian": 100.0,
    "dubins_airplane_gaussian": 300.0,
    "kinematic_car_mujoco": 250.0,
    "pushing_mujoco": 200.0,
    "pushing_real": 300.0,
}

def plot_time_cap_for_panel(panel_id: str, fallback: float = 300.0) -> float:
    return float(PANEL_PLOT_TIME_CAP_SECONDS.get(str(panel_id), fallback))


NUMERIC_FIELDS = {
    "run_number": int,
    "seed": int,
    "time": float,
    "cost": float,
    "num_controls": int,
    "overall_time": float,
    "plot_time_seconds": float,
    "max_time_seconds": float,
    "wall_time": float,
    "raw_process_wall_seconds": float,
    "initial_planning_seconds": float,
    "nominal_execution_seconds": float,
    "model_execution_seconds": float,
    "physical_execution_seconds": float,
    "physical_execution_step_seconds": float,
    "actual_execution_seconds": float,
    "online_replanning_seconds": float,
    "optimizer_seconds": float,
    "blocking_replanning_seconds": float,
    "compute_overrun_seconds": float,
    "replans": int,
    "num_replanning": int,
    "tracking_error": float,
    "tracking_error_mean": float,
    "goal_distance": float,
    "propagation_step_size_seconds": float,
    "min_control_duration": int,
    "max_control_duration": int,
    "planning_success": int,
    "execution_success": int,
    "collision": int,
    "uncertainty_level": float,
    "num_particles": int,
    "padding_epsilon": float,
    "tree_expansions": int,
    "tree_size": int,
    "dynamics_propagations": int,
    "particle_propagations": int,
    "minimum_obstacle_clearance": float,
}


def _convert_row(row: dict[str, str], *, panel_id: str | None = None) -> dict:
    output: dict = dict(row)
    if not output.get("panel_id") and panel_id is not None:
        output["panel_id"] = panel_id
    for field, converter in NUMERIC_FIELDS.items():
        value = output.get(field, "")
        if value not in ("", None):
            output[field] = converter(float(value))
        else:
            output[field] = None
    if str(output.get("method", "")).lower() == "replanning":
        output["method"] = "restartReplanning"
    # The preserved physical AURA/Replanning CSVs predate the compact result
    # schema. Keep their measured values untouched while filling only fields
    # needed by the common reader and plotter.
    if output.get("overall_time") is None:
        output["overall_time"] = output.get("time")
    if output.get("plot_time_seconds") is None:
        output["plot_time_seconds"] = output.get("time")
    if not str(output.get("status", "")).strip():
        measured_time = output.get("time")
        output["status"] = (
            "timeout"
            if measured_time is not None and float(measured_time) >= 600.0
            else "success"
        )
    if output.get("tracking_error_mean") is None:
        try:
            output["tracking_error_mean"] = math.hypot(
                float(output["actual_final_x"]) - float(output["planned_final_x"]),
                float(output["actual_final_y"]) - float(output["planned_final_y"]),
            )
        except (KeyError, TypeError, ValueError):
            pass
    steps = str(output.get("control_duration_steps", "")).strip()
    output["control_duration_steps_list"] = (
        [int(float(value)) for value in steps.split(";") if value] if steps else []
    )
    tracking_errors = str(output.get("tracking_errors", "")).strip()
    output["tracking_errors_list"] = (
        [float(value) for value in tracking_errors.split(";") if value]
        if tracking_errors
        else []
    )
    return output


def collect_rows(results_root: Path, manifest: dict | None = None) -> list[dict]:
    del manifest
    rows: list[dict] = []
    keys: dict[tuple, Path] = {}
    paths = sorted(
        path
        for panel_id in PANEL_LAYOUT
        for path in (Path(results_root) / panel_id).glob("*.csv")
    )
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                row = _convert_row(raw, panel_id=path.parent.name)
                if not raw.get("panel_id"):
                    try:
                        row["run_number"] = int(path.stem.rsplit("_", 1)[1])
                    except (IndexError, ValueError):
                        pass
                key = (
                    str(row["panel_id"]),
                    str(row["planner"]),
                    int(row["run_number"]),
                    str(row["method"]),
                )
                if key in keys:
                    raise ValueError(f"duplicate result {key}: {keys[key]} and {path}")
                keys[key] = path
                cap = plot_time_cap_for_panel(
                    str(row["panel_id"]),
                    fallback=float(row.get("max_time_seconds") or 300.0),
                )
                row["plot_cap_seconds"] = cap
                row["display_time_seconds"] = min(
                    float(row["plot_time_seconds"]), cap
                )
                row["_path"] = str(path)
                rows.append(row)
    return rows


def expected_simulation_keys(manifest: dict) -> set[tuple[str, str, int, str]]:
    runs = manifest.get("selected_runs")
    if runs is None:
        runs = range(1, int(manifest["num_simulation_trials"]) + 1)
    return {
        (panel["config"]["panel_id"], planner, int(run), method)
        for panel in manifest["panels"]
        if panel.get("execute", False)
        and panel["config"].get("simulator_mode") != "real"
        for planner in manifest["planners"]
        for run in runs
        for method in manifest["methods"]
    }


def expected_mppi_keys(manifest: dict) -> set[tuple[str, str, int, str]]:
    """Return the one-controller-per-panel MPPI matrix used by Figure 7."""
    default_runs = manifest.get(
        "mppi_selected_runs",
        manifest.get("selected_runs"),
    )
    if default_runs is None:
        default_runs = range(
            1, int(manifest["num_simulation_trials"]) + 1
        )
    runs_by_panel = dict(manifest.get("mppi_selected_runs_by_panel") or {})
    keys = set()
    for panel in manifest["panels"]:
        panel_id = panel["config"]["panel_id"]
        if not panel.get("execute", False):
            continue
        if panel["config"].get("simulator_mode") == "real":
            continue
        runs = runs_by_panel.get(panel_id, default_runs)
        keys.update(
            (panel_id, "mppi", int(run), "mppi")
            for run in runs
        )
    return keys


def validate_real_world_matrix(
    rows: list[dict],
) -> dict:
    """Report observed physical trials without imposing equal sample counts."""

    observed_runs = {"mppi": set(), "randup": set()}
    for row in rows:
        if str(row.get("panel_id")) != "pushing_real":
            continue
        method = str(row.get("method", "")).lower()
        planner = str(row.get("planner", "")).lower()
        if method == "mppi" or planner == "mppi":
            observed_runs["mppi"].add(int(row["run_number"]))
        elif "randup" in method or "randup" in planner:
            observed_runs["randup"].add(int(row["run_number"]))

    observed = any(observed_runs.values())
    return {
        "real_world_observed": observed,
        "real_world_methods_present": {
            method: bool(runs) for method, runs in observed_runs.items()
        },
        "real_world_mppi_observed_trials": len(observed_runs["mppi"]),
        "real_world_randup_observed_trials": len(observed_runs["randup"]),
        "real_world_mppi_run_numbers": sorted(observed_runs["mppi"]),
        "real_world_randup_run_numbers": sorted(observed_runs["randup"]),
    }


def validate_matrix(
    rows: list[dict], manifest: dict, *, require_complete_simulation: bool
) -> dict:
    observed = {
        (
            str(row["panel_id"]),
            str(row["planner"]),
            int(row["run_number"]),
            str(row["method"]),
        )
        for row in rows
    }
    expected_planning = expected_simulation_keys(manifest)
    mppi_observed = {
        key
        for key in observed
        if key[1].lower() == "mppi" or key[3].lower() == "mppi"
    }
    expected_mppi = expected_mppi_keys(manifest) if mppi_observed else set()
    # RandUP-RRT conditions (particle count, padding, uncertainty level) are
    # managed by its companion manifest. Accept every observed, uniquely-keyed
    # RandUP row here so it appears in the ordinary Figure-7 summaries without
    # changing the frozen AURA/RR matrix contract.
    observed_randup = {
        key
        for key in observed
        if key[1].lower().startswith("randup_rrt")
        or "randup_rrt" in key[3].lower()
    }
    expected = expected_planning | expected_mppi | observed_randup
    missing = sorted(expected - observed)
    unexpected = sorted(key for key in observed - expected if key[0] in SIMULATION_PANEL_ORDER)
    if unexpected and require_complete_simulation:
        raise ValueError(f"unexpected simulated rows: {unexpected[:5]}")
    if missing and require_complete_simulation:
        raise RuntimeError(
            f"simulated matrix is incomplete: {len(missing)} of {len(expected)} "
            f"method rows missing; first={missing[:5]}"
        )
    invalid_pairs = []
    planning_rows = [
        row
        for row in rows
        if row["planner"] in PLANNERS and row["method"] in METHODS
    ]
    pair_counts = Counter(
        (row["panel_id"], row["planner"], row["run_number"])
        for row in planning_rows
    )
    for key, count in pair_counts.items():
        if count != 2:
            invalid_pairs.append((*key, count))
    if invalid_pairs and require_complete_simulation:
        raise RuntimeError(f"paired CSV contract violated: {invalid_pairs[:5]}")
    contract_errors = []
    for row in rows:
        if str(row["panel_id"]) not in SIMULATION_PANEL_ORDER:
            continue
        is_mppi = row["planner"] == "mppi" and row["method"] == "mppi"
        is_randup = str(row["planner"]).startswith("randup_rrt")
        expected_min, expected_max = (
            (1, 1)
            if is_mppi
            else (
                (
                    int(row["min_control_duration"]),
                    int(row["max_control_duration"]),
                )
                if is_randup
                else (1, 5)
            )
        )
        if (
            row["min_control_duration"] != expected_min
            or row["max_control_duration"] != expected_max
            or not math.isclose(
                float(row["propagation_step_size_seconds"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            contract_errors.append(
                (
                    row["panel_id"],
                    row["planner"],
                    row["run_number"],
                    row["method"],
                    "duration_config",
                )
            )
        if any(
            step < expected_min or step > expected_max
            for step in row["control_duration_steps_list"]
        ):
            contract_errors.append(
                (
                    row["panel_id"],
                    row["planner"],
                    row["run_number"],
                    row["method"],
                    "executed_duration",
                )
            )
    for pair_key in pair_counts:
        pair = [
            row
            for row in planning_rows
            if (
                row["panel_id"],
                row["planner"],
                row["run_number"],
            )
            == pair_key
        ]
        if len(pair) == 2 and (
            pair[0].get("initial_plan_hash") != pair[1].get("initial_plan_hash")
            or pair[0].get("disturbance_schedule_hash")
            != pair[1].get("disturbance_schedule_hash")
        ):
            contract_errors.append((*pair_key, "pairing_hash"))
    if contract_errors and require_complete_simulation:
        raise RuntimeError(
            f"duration/pairing contract violated: {contract_errors[:5]}"
        )
    simulation_complete = not missing and not invalid_pairs and not contract_errors
    real_validation = validate_real_world_matrix(rows)
    return {
        "expected_simulation_method_rows": len(expected),
        "observed_simulation_method_rows": len(expected & observed),
        "missing_simulation_method_rows": len(missing),
        "duration_pairing_contract_errors": len(contract_errors),
        "mppi_expected_rows": len(expected_mppi),
        "mppi_observed_rows": len(expected_mppi & observed),
        "complete": simulation_complete,
        **real_validation,
    }


def _distribution(values) -> dict:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=float,
    )
    if not len(finite):
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(statistics.stdev(finite)) if len(finite) > 1 else 0.0,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def aggregate(rows: list[dict], manifest: dict | None = None) -> dict:
    del manifest
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault(
            (row["panel_id"], row["planner"], row["method"]), []
        ).append(row)
    summaries = []
    for (panel_id, planner, method), group in sorted(groups.items()):
        successes = [row for row in group if row["status"] == "success"]
        durations = Counter(
            step
            for row in group
            for step in row["control_duration_steps_list"]
        )
        summaries.append(
            {
                "panel_id": panel_id,
                "planner": planner,
                "method": method,
                "n_total": len(group),
                "n_success": len(successes),
                "n_failure": len(group) - len(successes),
                "success_rate": len(successes) / len(group),
                "plot_cap_seconds": plot_time_cap_for_panel(panel_id),
                "penalized_time": _distribution(
                    row["display_time_seconds"] for row in group
                ),
                "successful_overall_time": _distribution(
                    row["overall_time"] for row in successes
                ),
                "tracking_error": _distribution(
                    row["tracking_error_mean"] for row in group
                ),
                "num_controls": _distribution(row["num_controls"] for row in group),
                "num_replanning": _distribution(
                    row["num_replanning"] for row in group
                ),
                "duration_histogram": dict(sorted(durations.items())),
            }
        )
    return {"groups": summaries}


def paired_comparisons(rows: list[dict]) -> list[dict]:
    pairs: dict[tuple[str, str, int], dict[str, dict]] = {}
    for row in rows:
        pairs.setdefault(
            (row["panel_id"], row["planner"], int(row["run_number"])), {}
        )[row["method"]] = row
    output = []
    for (panel, planner, run), pair in sorted(pairs.items()):
        if set(pair) != set(METHODS):
            continue
        aura = pair["aura"]
        replanning = pair["restartReplanning"]
        output.append(
            {
                "panel_id": panel,
                "planner": planner,
                "run_number": run,
                "aura_status": aura["status"],
                "replanning_status": replanning["status"],
                "plot_cap_seconds": plot_time_cap_for_panel(panel),
                "aura_penalized_time": aura["display_time_seconds"],
                "replanning_penalized_time": replanning["display_time_seconds"],
                "replanning_minus_aura_time": (
                    replanning["display_time_seconds"]
                    - aura["display_time_seconds"]
                ),
                "aura_tracking_error": aura["tracking_error_mean"],
                "replanning_tracking_error": replanning["tracking_error_mean"],
                "aura_num_replanning": aura["num_replanning"],
                "replanning_num_replanning": replanning["num_replanning"],
            }
        )
    return output


def paired_success_tracking_summary(rows: list[dict]) -> list[dict]:
    """Summarize tracking only where both paired methods reached the goal.

    A failed run can still contain a short, low-error prefix.  Mixing those
    partial prefixes with complete trajectories creates a survivorship bias
    against the method that completes more difficult trials.  Wall time still
    includes every failure at the panel cap; this filter applies only to the
    tracking comparison.
    """
    pairs: dict[tuple[str, str, int], dict[str, dict]] = {}
    for row in rows:
        pairs.setdefault(
            (row["panel_id"], row["planner"], int(row["run_number"])), {}
        )[row["method"]] = row
    successful_pairs = [
        (key, pair)
        for key, pair in sorted(pairs.items())
        if set(pair) == set(METHODS)
        and all(pair[method]["status"] == "success" for method in METHODS)
    ]
    if not successful_pairs:
        return []

    group_specs: list[tuple[str, str, str, list[tuple]]] = [
        ("overall", "ALL", "ALL", successful_pairs)
    ]
    for panel_id in sorted({key[0] for key, _ in successful_pairs}):
        panel_pairs = [
            item for item in successful_pairs if item[0][0] == panel_id
        ]
        group_specs.append(("panel", panel_id, "ALL", panel_pairs))
        for planner in PLANNERS:
            planner_pairs = [
                item for item in panel_pairs if item[0][1] == planner
            ]
            if planner_pairs:
                group_specs.append(
                    ("panel_planner", panel_id, planner, planner_pairs)
                )

    output = []
    for scope, panel_id, planner, group in group_specs:
        aura_run_errors = [
            float(pair["aura"]["tracking_error_mean"]) for _, pair in group
        ]
        replanning_run_errors = [
            float(pair["restartReplanning"]["tracking_error_mean"])
            for _, pair in group
        ]
        aura_step_errors = [
            float(error)
            for _, pair in group
            for error in pair["aura"]["tracking_errors_list"]
        ]
        replanning_step_errors = [
            float(error)
            for _, pair in group
            for error in pair["restartReplanning"]["tracking_errors_list"]
        ]
        aura_run_mean = float(np.mean(aura_run_errors))
        replanning_run_mean = float(np.mean(replanning_run_errors))
        aura_step_mean = (
            float(np.mean(aura_step_errors)) if aura_step_errors else None
        )
        replanning_step_mean = (
            float(np.mean(replanning_step_errors))
            if replanning_step_errors
            else None
        )
        run_deltas = [
            replanning - aura
            for aura, replanning in zip(
                aura_run_errors, replanning_run_errors
            )
        ]
        output.append(
            {
                "scope": scope,
                "panel_id": panel_id,
                "planner": planner,
                "n_paired_successes": len(group),
                "aura_run_mean_tracking_error": aura_run_mean,
                "replanning_run_mean_tracking_error": replanning_run_mean,
                "replanning_minus_aura_run_mean": (
                    replanning_run_mean - aura_run_mean
                ),
                "aura_step_count": len(aura_step_errors),
                "aura_step_weighted_tracking_error": aura_step_mean,
                "replanning_step_count": len(replanning_step_errors),
                "replanning_step_weighted_tracking_error": replanning_step_mean,
                "replanning_minus_aura_step_weighted": (
                    replanning_step_mean - aura_step_mean
                    if replanning_step_mean is not None
                    and aura_step_mean is not None
                    else None
                ),
                "aura_lower_error_pairs": sum(delta > 0.0 for delta in run_deltas),
                "replanning_lower_error_pairs": sum(
                    delta < 0.0 for delta in run_deltas
                ),
                "equal_error_pairs": sum(delta == 0.0 for delta in run_deltas),
            }
        )
    return output


def _flatten_group(row: dict) -> dict:
    output = {key: value for key, value in row.items() if not isinstance(value, dict)}
    for prefix in (
        "penalized_time",
        "successful_overall_time",
        "tracking_error",
        "num_controls",
        "num_replanning",
    ):
        for key, value in row[prefix].items():
            output[f"{prefix}_{key}"] = value
    output["duration_histogram"] = ";".join(
        f"{key}:{value}" for key, value in row["duration_histogram"].items()
    )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    validation: dict,
    summary: dict,
    paired: list[dict],
    paired_tracking: list[dict] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "summary.csv",
        [_flatten_group(row) for row in summary["groups"]],
    )
    _write_csv(output_dir / "paired_comparisons.csv", paired)
    if paired_tracking is not None:
        _write_csv(
            output_dir / "paired_success_tracking.csv",
            paired_tracking,
        )
    lines = [
        "# Fig. 7 campaign validation",
        "",
        f"- Complete simulated matrix: {validation['complete']}",
        f"- Observed method rows: {validation['observed_simulation_method_rows']}",
        f"- Expected method rows: {validation['expected_simulation_method_rows']}",
        "",
        "Failures are included at each panel's visualization cap in every "
        "`penalized_time` statistic and in the plot.",
    ]
    if paired_tracking:
        overall = paired_tracking[0]
        lines.extend(
            [
                "",
                "Tracking is compared only for paired trials where both methods "
                "completed; failed runs can otherwise contribute misleadingly "
                "short partial trajectories.",
                "",
                f"- Paired successful trials: {overall['n_paired_successes']}",
                "- Pooled step-wise tracking error: "
                f"AURA {overall['aura_step_weighted_tracking_error']:.6f}, "
                "restart replanning "
                f"{overall['replanning_step_weighted_tracking_error']:.6f}",
                "- Mean per-run tracking error: "
                f"AURA {overall['aura_run_mean_tracking_error']:.6f}, "
                "restart replanning "
                f"{overall['replanning_run_mean_tracking_error']:.6f}",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(
    results_root: Path,
    *,
    include_empty_panels: bool = False,
    dash: bool = DEFAULT_METHOD_SEPARATORS,
    black: bool = False,
) -> None:
    create_figure(
        Path(results_root).resolve(),
        include_empty_panels=include_empty_panels,
        dash=dash,
        black=black,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help="Task-time result directory (defaults to results/full_time_comparison).",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Also require every simulation row listed by the manifest.",
    )
    parser.add_argument("--include-empty-panels", action="store_true")
    parser.add_argument(
        "--dash",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_METHOD_SEPARATORS,
        help="Show method-group separators (enabled by default).",
    )
    parser.add_argument("--black", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = collect_rows(root, manifest)
    validation = validate_matrix(
        rows,
        manifest,
        require_complete_simulation=args.require_complete,
    )
    summary = aggregate(rows, manifest)
    paired = paired_comparisons(rows)
    paired_tracking = paired_success_tracking_summary(rows)
    write_outputs(
        root / "summary",
        validation,
        summary,
        paired,
        paired_tracking,
    )
    if not args.no_plot:
        make_figure(
            root,
            include_empty_panels=args.include_empty_panels,
            dash=args.dash,
            black=args.black,
        )
    print(
        f"validated {len(rows)} method rows in {len(paired)} paired trials; "
        f"complete={validation['complete']}"
    )


PLANNER_COLORS = {
    "aorrt": "#A0C878",
    "aoest": "#143D60",
    "sststar": "#EB5B00",
}
METHOD_COLORS = {"mppi": "#9B59B6", "randup": "#2A9D8F"}
PLANNER_LABELS = {"aorrt": "AORRT", "aoest": "AOEST", "sststar": "SST"}
METHOD_LABELS = {
    "aura": "AURA",
    "replanning": "RR",
    "mppi": "MPPI",
    "randup": "RobRRT",
}
PANEL_TITLES = {
    "double_integrator_gaussian": ("Double Integrator", "Gaussian Noise"),
    "kinematic_car_gaussian": ("Kinematic Car", "Gaussian Noise"),
    "pushing_gaussian": ("Learned Pushing Dynamics", "Gaussian Noise"),
    "dubins_airplane_gaussian": ("6D Dubins Airplane", "Gaussian Noise"),
    "kinematic_car_mujoco": ("Kinematic Car", "MuJoCo Simulation"),
    "pushing_mujoco": ("Learned Pushing Dynamics", "MuJoCo Simulation"),
    "pushing_real": ("Learned Pushing Dynamics", "Real-World Hardware"),
}


def task_time(row: dict, cap: float) -> float:
    """Return a finite plotted task time, charging failures at the panel cap."""
    if str(row.get("status", "")).lower() != "success":
        return float(cap)
    for field in ("task_time_seconds", "plot_time_seconds", "overall_time", "time"):
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            return min(float(value), float(cap))
    return float(cap)


def method_key(row: dict) -> tuple[str, str] | None:
    method = str(row.get("method", "")).lower()
    planner = str(row.get("planner", "")).lower()
    if method == "aura" and planner in PLANNERS:
        return "aura", planner
    if method in {"restartreplanning", "replanning"} and planner in PLANNERS:
        return "replanning", planner
    if method == "mppi":
        return "mppi", "mppi"
    if "randup" in method or "randup" in planner:
        return "randup", "randup"
    return None


def grouped_times(
    rows: list[dict], panel_id: str
) -> list[tuple[tuple[str, str], list[float]]]:
    cap = PANEL_PLOT_TIME_CAP_SECONDS.get(panel_id, 300.0)
    groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if str(row.get("panel_id")) != panel_id:
            continue
        key = method_key(row)
        if key is not None:
            groups.setdefault(key, []).append(task_time(row, cap))
    order = [
        *(("aura", planner) for planner in PLANNERS),
        *(("replanning", planner) for planner in PLANNERS),
        ("mppi", "mppi"),
        ("randup", "randup"),
    ]
    return [(key, groups[key]) for key in order if groups.get(key)]


def box_label(key: tuple[str, str]) -> str:
    method, planner = key
    if method in {"aura", "replanning"}:
        prefix = "AURA" if method == "aura" else "RR"
        return f"{prefix}-{PLANNER_LABELS[planner]}"
    return METHOD_LABELS[method]


def box_color(key: tuple[str, str]) -> str:
    method, planner = key
    if method in {"aura", "replanning"}:
        return PLANNER_COLORS[planner]
    return METHOD_COLORS[method]


def darken_color(color: str, factor: float) -> tuple[float, float, float]:
    return tuple(np.clip(np.asarray(mcolors.to_rgb(color)) * factor, 0.0, 1.0))


def group_positions(
    groups: list[tuple[tuple[str, str], list[float]]],
) -> tuple[list[float], dict[str, list[float]]]:
    """Use plotExp2's compact method-first spacing."""

    positions: list[float] = []
    method_positions: dict[str, list[float]] = {}
    position = 1.0
    previous_method: str | None = None
    for (method, _planner), _values in groups:
        if previous_method is not None and method != previous_method:
            # Give the two single-box methods the same surrounding space.
            # Besides keeping their labels readable, equal gaps place MPPI
            # exactly halfway between its two method separators.
            position += (
                SINGLE_METHOD_GAP
                if method in {"mppi", "randup"}
                else METHOD_GROUP_GAP
            )
        positions.append(position)
        method_positions.setdefault(method, []).append(position)
        position += BOX_CENTER_SPACING
        previous_method = method
    return positions, method_positions


def method_boundaries(method_positions: dict[str, list[float]]) -> list[float]:
    """Return separators halfway between adjacent methods' nearest boxes."""

    ordered_methods = list(method_positions)
    return [
        (max(method_positions[left]) + min(method_positions[right])) / 2.0
        for left, right in zip(ordered_methods, ordered_methods[1:])
    ]


def horizontal_limits(
    positions: list[float],
    method_positions: dict[str, list[float]],
    boundaries: list[float],
) -> tuple[float, float]:
    """Keep singleton methods centered between separators and panel edges."""

    x_margin = (
        (max(positions) - min(positions)) * 0.02 if len(positions) > 1 else 0.05
    )
    half_box_width = BOXPLOT_WIDTH / 2.0
    left = min(positions) - half_box_width - x_margin
    right = max(positions) + half_box_width + x_margin
    ordered_methods = list(method_positions)
    if boundaries and len(method_positions[ordered_methods[0]]) == 1:
        center = method_positions[ordered_methods[0]][0]
        left = 2.0 * center - boundaries[0]
    if boundaries and len(method_positions[ordered_methods[-1]]) == 1:
        center = method_positions[ordered_methods[-1]][0]
        right = 2.0 * center - boundaries[-1]
    return left, right


def add_panel_title(axis, title: str, subtitle: str, *, color: str) -> None:
    axis.text(
        0.5,
        PANEL_TITLE_Y,
        title,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=PANEL_TITLE_SIZE,
        color=color,
        clip_on=False,
    )
    axis.text(
        0.5,
        PANEL_SUBTITLE_Y,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=PANEL_TITLE_SIZE,
        fontstyle="italic",
        color=color,
        clip_on=False,
    )


def draw_panel(
    axis,
    rows: list[dict],
    panel_id: str,
    *,
    show_ylabel: bool,
    dash: bool,
    black: bool,
) -> None:
    groups = grouped_times(rows, panel_id)
    title, subtitle = PANEL_TITLES.get(
        panel_id, (panel_id.replace("_", " ").title(), "")
    )
    background = "#000000" if black else "white"
    text_color = "white" if black else "black"
    axis.set_facecolor(background)
    add_panel_title(axis, title, subtitle, color=text_color)
    if not groups:
        axis.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            color=text_color,
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return
    data = [values for _, values in groups]
    positions, method_positions = group_positions(groups)
    boundaries = method_boundaries(method_positions)
    boxes = axis.boxplot(
        data,
        positions=positions,
        widths=BOXPLOT_WIDTH,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        whis=(10, 95),
        boxprops={"linewidth": 0.75},
        medianprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 0.75},
        capprops={"linewidth": 0.75},
    )
    for index, (patch, (key, _)) in enumerate(zip(boxes["boxes"], groups)):
        color = box_color(key)
        edge = darken_color(color, 0.55 if key[0] == "replanning" else 0.72)
        patch.set_facecolor(color)
        patch.set_edgecolor(edge)
        patch.set_alpha(1.0)
        if key[0] == "replanning":
            patch.set_hatch("///")
            if hasattr(patch, "set_hatch_linewidth"):
                patch.set_hatch_linewidth(2.0)
        boxes["medians"][index].set_color(edge)
        boxes["whiskers"][2 * index].set_color(edge)
        boxes["whiskers"][2 * index + 1].set_color(edge)
        boxes["caps"][2 * index].set_color(edge)
        boxes["caps"][2 * index + 1].set_color(edge)
    axis.set_xticks(positions)
    axis.set_xticklabels([""] * len(positions))
    axis.tick_params(axis="x", which="both", length=0)
    for method, values in method_positions.items():
        axis.text(
            (min(values) + max(values)) / 2.0,
            -0.02,
            METHOD_LABELS[method],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=METHOD_LABEL_SIZE,
            rotation=METHOD_LABEL_ROTATION,
            color=text_color,
        )
    axis.set_ylabel(
        "Wall Time" if show_ylabel else "",
        fontsize=Y_AXIS_LABEL_SIZE,
    )
    y_max = PANEL_PLOT_TIME_CAP_SECONDS.get(panel_id, 300.0)
    axis.set_ylim(0.0, y_max)
    axis.set_yticks(np.linspace(0.0, y_max, Y_AXIS_TICK_COUNT))
    axis.tick_params(axis="y", labelsize=Y_TICK_LABEL_SIZE)
    axis.grid(axis="y", alpha=0.4, linestyle="-")
    axis.grid(False, axis="x")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xlim(*horizontal_limits(positions, method_positions, boundaries))
    if dash:
        for boundary in boundaries:
            axis.axvline(
                boundary,
                color="0.45",
                linestyle=(0.0, METHOD_SEPARATOR_DASH_PATTERN),
                linewidth=0.8,
            )
    if black:
        axis.tick_params(colors="white")
        axis.xaxis.label.set_color("white")
        axis.yaxis.label.set_color("white")
        for spine in axis.spines.values():
            spine.set_color("white")


def figure_panels(rows: list[dict], include_empty_panels: bool = False) -> list[str]:
    """Return all six simulation panels plus real-world pushing."""

    observed = {str(row.get("panel_id")) for row in rows}
    return [
        panel
        for panel in FIGURE_7_PANEL_ORDER
        if include_empty_panels or panel in observed
    ]


def create_figure(
    results_root: Path,
    *,
    include_empty_panels: bool = False,
    dash: bool = DEFAULT_METHOD_SEPARATORS,
    black: bool = False,
) -> list[Path]:
    manifest_path = results_root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    rows = collect_rows(results_root, manifest)
    panels = figure_panels(rows, include_empty_panels)
    if not panels:
        raise RuntimeError(f"no task-time rows found under {results_root}")

    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(PANEL_WIDTH * len(panels), PANEL_HEIGHT),
        squeeze=False,
    )
    if black:
        figure.patch.set_facecolor("black")
    for axis, panel_id in zip(axes.flat, panels):
        draw_panel(
            axis,
            rows,
            panel_id,
            show_ylabel=axis is axes.flat[0],
            dash=dash,
            black=black,
        )
    figure.tight_layout(w_pad=PANEL_HORIZONTAL_PAD)
    figure.subplots_adjust(wspace=PANEL_WSPACE)
    paths = [
        results_root / f"task_time_comparison.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ]
    for path in paths:
        figure.savefig(path, dpi=300 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)
    return paths


if __name__ == "__main__":
    main()
