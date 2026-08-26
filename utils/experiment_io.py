"""Shared persistence and reproducibility helpers for experiment runners."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np


PLANNERS = ("aorrt", "sststar", "aoest")
METHODS = ("aura", "restartReplanning")
RESULT_SCHEMA_VERSION = 3
WALL_TIME_DEFINITIONS = {
    "aura": (
        "initial_offline_planning + nominal_control_execution "
        "+ blocking_aura_restart_planning"
    ),
    "restartReplanning": (
        "initial_offline_planning + nominal_control_execution "
        "+ blocking_restart_replanning"
    ),
}
SIMULATION_PANEL_ORDER = (
    "double_integrator_gaussian",
    "kinematic_car_gaussian",
    "pushing_gaussian",
    "dubins_airplane_gaussian",
    "kinematic_car_mujoco",
    "pushing_mujoco",
)
PANEL_LAYOUT = {
    "double_integrator_gaussian": ("double_integrator_gaussian", "double_integrator", 300.0),
    "kinematic_car_gaussian": ("kinematic_car_gaussian", "kinematic_car", 300.0),
    "pushing_gaussian": ("pushing_gaussian", "pushing_object", 300.0),
    "dubins_airplane_gaussian": ("dubins_airplane_gaussian", "dubins_airplane", 300.0),
    "kinematic_car_mujoco": ("kinematic_car_mujoco", "kinematic_car", 300.0),
    "pushing_mujoco": ("pushing_mujoco", "pushing_object", 300.0),
    "pushing_real": ("pushing_real", "pushing_object", 300.0),
}

REQUIRED_RESULT_FIELDS = {
    "schema_version", "panel_id", "system", "environment", "planner", "method",
    "run_number", "seed", "rng_streams", "config_hash", "initial_planning_seconds",
    "initial_plan_hash", "status", "failure_reason", "nominal_execution_seconds",
    "actual_execution_seconds", "online_replanning_seconds", "optimizer_seconds",
    "blocking_replanning_seconds", "compute_overrun_seconds", "wall_time_definition",
    "task_time_seconds", "raw_process_wall_seconds", "setup_reset_operator_pause_seconds",
    "num_controls", "num_replanning", "controls", "control_duration_steps",
    "control_duration_seconds",
}
FINITE_TIME_FIELDS = (
    "initial_planning_seconds", "nominal_execution_seconds", "actual_execution_seconds",
    "online_replanning_seconds", "optimizer_seconds", "blocking_replanning_seconds",
    "compute_overrun_seconds", "task_time_seconds", "raw_process_wall_seconds",
    "setup_reset_operator_pause_seconds",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def data_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def stable_seed(base_seed: int, *labels: object) -> int:
    payload = ":".join([str(int(base_seed)), *(str(value) for value in labels)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Iterable[str] | None = None,
) -> None:
    materialized = list(rows)
    fields = list(fieldnames) if fieldnames is not None else (
        list(materialized[0]) if materialized else []
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if fields:
                writer.writeheader()
                writer.writerows(materialized)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(json_value(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json_lines(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(json_value(row), sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def task_time_seconds(
    method: str,
    *,
    initial_planning_seconds: float,
    nominal_execution_seconds: float,
    blocking_replanning_seconds: float = 0.0,
) -> float:
    if method not in METHODS:
        raise ValueError(f"unsupported wall-time method: {method!r}")
    values = (
        float(initial_planning_seconds),
        float(nominal_execution_seconds),
        float(blocking_replanning_seconds),
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("wall-time components must be finite and nonnegative")
    return float(sum(values))


def validate_result(row: dict, *, expected_config_hash: str | None = None) -> None:
    missing = sorted(REQUIRED_RESULT_FIELDS - set(row))
    if missing:
        raise ValueError(f"result row is missing fields: {missing}")
    if expected_config_hash is not None and row["config_hash"] != expected_config_hash:
        raise ValueError("result configuration hash does not match frozen manifest")
    if row["status"] not in {"success", "failure", "timeout"}:
        raise ValueError(f"invalid result status {row['status']!r}")
    if row["status"] != "success" and not str(row["failure_reason"]).strip():
        raise ValueError("a failed/timeout row requires an explicit failure_reason")
    if row["wall_time_definition"] != WALL_TIME_DEFINITIONS.get(str(row["method"])):
        raise ValueError("wall_time_definition does not match the method accounting contract")
    for field in FINITE_TIME_FIELDS:
        value = row[field]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")
        if float(value) < 0.0:
            raise ValueError(f"{field} must be nonnegative")
    expected = task_time_seconds(
        str(row["method"]),
        initial_planning_seconds=float(row["initial_planning_seconds"]),
        nominal_execution_seconds=float(row["nominal_execution_seconds"]),
        blocking_replanning_seconds=float(row["blocking_replanning_seconds"]),
    )
    if not math.isclose(float(row["task_time_seconds"]), expected, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            "task_time_seconds violates the wall-time accounting contract: "
            f"stored={row['task_time_seconds']}, expected={expected}"
        )
    controls = row["controls"]
    steps = row["control_duration_steps"]
    seconds = row["control_duration_seconds"]
    if not (len(controls) == len(steps) == len(seconds) == int(row["num_controls"])):
        raise ValueError("controls and duration arrays must all match num_controls")
    if any(int(value) <= 0 for value in steps):
        raise ValueError("control_duration_steps must be positive integers")
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in seconds
    ):
        raise ValueError("control_duration_seconds must be finite and positive")
    if row["status"] == "success":
        for field in ("cost", "tracking_error_mean", "goal_distance"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"successful row requires finite {field}")


def finalize_result(row: dict, *, expected_config_hash: str) -> dict:
    normalized = json_value(row)
    nonfinite = [
        field
        for field in ("cost", "tracking_error_mean", "goal_distance")
        if normalized.get(field) is None
    ]
    if normalized["status"] == "success" and nonfinite:
        normalized["status"] = "failure"
        normalized["failure_reason"] = "nonfinite_result:" + ",".join(nonfinite)
    validate_result(normalized, expected_config_hash=expected_config_hash)
    return normalized


def result_path(root: Path, panel_id: str, planner: str, run_number: int) -> Path:
    if panel_id not in PANEL_LAYOUT:
        raise ValueError(f"unknown task-time panel: {panel_id}")
    return Path(root) / panel_id / f"{planner}_{int(run_number):03d}.csv"


def compact_numbers(values: Iterable[Any], *, precision: int = 12) -> str:
    if values is None:
        return ""
    return ";".join(f"{float(value):.{precision}g}" for value in values)


def compact_controls(values: Iterable[Iterable[Any]]) -> str:
    if values is None:
        return ""
    return "|".join(compact_numbers(row) for row in values)


def duration_histogram(values: Iterable[Any]) -> str:
    counts: dict[int, int] = {}
    if values is None:
        return ""
    for value in values:
        key = int(value)
        counts[key] = counts.get(key, 0) + 1
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def legacy_time_correction(panel_id: str, method: str, num_controls: int) -> float:
    if panel_id == "pushing_gaussian" and method in {"aura", "restartReplanning"}:
        return 2.0 * float(num_controls)
    if panel_id in {"double_integrator_gaussian", "kinematic_car_gaussian"} and method == "restartReplanning":
        return float(num_controls)
    return 0.0


def serialize_result(row: dict) -> dict:
    panel_id = str(row["panel_id"])
    _, _, max_time = PANEL_LAYOUT[panel_id]
    overall_time = float(row["task_time_seconds"])
    plot_time = min(overall_time, max_time) if row["status"] == "success" else max_time
    final_value = row.get("final_state")
    planned_value = row.get("planned_final_state")
    final_state = [] if final_value is None else list(final_value)
    planned_state = [] if planned_value is None else list(planned_value)
    steps = [int(value) for value in row.get("control_duration_steps", [])]
    seconds = [float(value) for value in row.get("control_duration_seconds", [])]
    audit = row.get("duration_audit_initial_tree") or {}
    initial_plan = row.get("initial_plan") or {}
    tree_histogram = audit.get("duration_step_histogram") or {}
    output = {
        "run_number": int(row["run_number"]),
        "planner": str(row["planner"]),
        "method": "replanning" if row["method"] == "restartReplanning" else str(row["method"]),
        "time": float(plot_time - legacy_time_correction(panel_id, str(row["method"]), int(row["num_controls"]))),
        "cost": row.get("cost"),
        "num_controls": int(row["num_controls"]),
        "overall_time": overall_time,
        "plot_time_seconds": float(plot_time),
        "max_time_seconds": float(max_time),
        "wall_time": overall_time,
        "raw_process_wall_seconds": float(row["raw_process_wall_seconds"]),
        "initial_planning_seconds": float(row["initial_planning_seconds"]),
        "nominal_execution_seconds": float(row["nominal_execution_seconds"]),
        "actual_execution_seconds": float(row["actual_execution_seconds"]),
        "online_replanning_seconds": float(row["online_replanning_seconds"]),
        "optimizer_seconds": float(row["optimizer_seconds"]),
        "blocking_replanning_seconds": float(row["blocking_replanning_seconds"]),
        "compute_overrun_seconds": float(row["compute_overrun_seconds"]),
        "wall_time_definition": str(row["wall_time_definition"]),
        "replans": int(row["num_replanning"]),
        "num_replanning": int(row["num_replanning"]),
        "initial_planning_attempts": int(row.get("initial_planning_attempts", 1)),
        "aura_restart_count": int(row.get("aura_restart_count", 0)),
        "aura_restart_planning_attempts": int(row.get("aura_restart_planning_attempts", 0)),
        "tracking_error": row.get("tracking_error_mean"),
        "tracking_error_mean": row.get("tracking_error_mean"),
        "goal_distance": row.get("goal_distance"),
        "status": str(row["status"]),
        "failure_reason": str(row["failure_reason"]),
        "panel_id": panel_id,
        "system": str(row["system"]),
        "environment": str(row["environment"]),
        "seed": int(row["seed"]),
        "config_hash": str(row["config_hash"]),
        "initial_plan_hash": row.get("initial_plan_hash") or "",
        "initial_plan_duration_steps": compact_numbers(initial_plan.get("duration_steps", initial_plan.get("time_steps", [])), precision=0),
        "initial_plan_duration_seconds": compact_numbers(initial_plan.get("duration_seconds", initial_plan.get("time", []))),
        "disturbance_schedule_hash": row.get("disturbance_schedule_hash", ""),
        "propagation_step_size_seconds": float(audit.get("propagation_step_size_seconds", 1.0)),
        "min_control_duration": int((audit.get("range_steps") or [1, 5])[0]),
        "max_control_duration": int((audit.get("range_steps") or [1, 5])[1]),
        "control_duration_steps": compact_numbers(steps, precision=0),
        "control_duration_seconds": compact_numbers(seconds),
        "duration_histogram": duration_histogram(steps),
        "initial_tree_duration_histogram": ";".join(
            f"{key}:{tree_histogram[key]}" for key in sorted(tree_histogram, key=lambda value: int(value))
        ),
        "tracking_errors": compact_numbers(row.get("tracking_error_list", [])),
        "controls": compact_controls(row.get("controls", [])),
    }
    for index in range(6):
        output[f"final_{index}"] = float(final_state[index]) if index < len(final_state) else ""
        output[f"planned_final_{index}"] = float(planned_state[index]) if index < len(planned_state) else ""
    if len(final_state) >= 3:
        output.update(
            final_state_x=float(final_state[0]), final_state_y=float(final_state[1]),
            final_state_theta=float(final_state[2]), actual_final_x=float(final_state[0]),
            actual_final_y=float(final_state[1]), actual_final_theta=float(final_state[2]),
        )
    if len(planned_state) >= 3:
        output.update(
            planned_final_x=float(planned_state[0]), planned_final_y=float(planned_state[1]),
            planned_final_theta=float(planned_state[2]),
        )
    return output


def write_paired_results(path: Path, rows: Iterable[dict]) -> None:
    materialized = list(rows)
    if not materialized or len(materialized) > 2:
        raise ValueError(f"paired CSV requires one or two rows, got {len(materialized)}")
    csv_rows = [serialize_result(row) for row in materialized]
    fields: list[str] = []
    for row in csv_rows:
        fields.extend(field for field in row if field not in fields)
    write_csv(path, csv_rows, fields)


def upsert_serialized_result(path: Path, new_row: dict) -> None:
    rows = read_csv(path)
    method = str(new_row["method"]).lower()
    rows = [row for row in rows if str(row.get("method", "")).lower() != method]
    rows.append(new_row)
    order = {"aura": 0, "fusion": 0, "replanning": 1}
    rows.sort(key=lambda row: order.get(str(row.get("method", "")).lower(), 9))
    fields: list[str] = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    write_csv(path, rows, fields)


def upsert_result(path: Path, result_row: dict) -> None:
    upsert_serialized_result(path, serialize_result(result_row))


def ensure_json(path: Path, value: Any, *, mismatch_message: str) -> dict:
    payload = json_value(value)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(mismatch_message)
        return payload
    write_json(path, payload)
    return payload


def job_complete(
    root: Path,
    panel_id: str,
    planner: str,
    run_number: int,
    methods: Iterable[str],
    *,
    expected_config_hash: str | None = None,
    accepted_statuses: set[str] | None = None,
) -> bool:
    rows = read_csv(result_path(root, panel_id, planner, run_number))
    if not rows:
        return False
    requested = set(methods)
    selected = [
        row
        for row in rows
        if (
            "restartReplanning"
            if str(row.get("method", "")).lower() == "replanning"
            else row.get("method", "")
        )
        in requested
    ]
    observed = {
        "restartReplanning"
        if str(row.get("method", "")).lower() == "replanning"
        else str(row.get("method", ""))
        for row in selected
    }
    if requested != observed:
        return False
    if expected_config_hash is not None and any(
        row.get("config_hash") != expected_config_hash for row in selected
    ):
        return False
    allowed = accepted_statuses or {"success", "failure", "timeout"}
    return all(row.get("status") in allowed for row in selected)


def import_successful_results(
    source_root: Path,
    destination_root: Path,
    frozen_manifest: dict,
) -> int:
    panels = {
        panel["config"]["panel_id"]: panel for panel in frozen_manifest["panels"]
    }
    imported = 0
    source_paths = (
        path
        for panel_id in PANEL_LAYOUT
        for path in (Path(source_root) / panel_id).glob("*.csv")
    )
    for source_path in sorted(source_paths):
        for row in read_csv(source_path):
            if str(row.get("status", "")).lower() != "success":
                continue
            panel_id = str(row.get("panel_id", ""))
            panel = panels.get(panel_id)
            if panel is None:
                continue
            task_limit = float(panel["config"].get("task_time_limit_seconds", 300.0))
            overall_time = float(row["overall_time"])
            if overall_time > task_limit + 1e-9:
                continue
            planner = str(row["planner"]).lower()
            run_number = int(row["run_number"])
            method = (
                "restartReplanning"
                if str(row["method"]).lower() == "replanning"
                else "aura"
            )
            if job_complete(
                destination_root,
                panel_id,
                planner,
                run_number,
                [method],
                expected_config_hash=panel["config_hash"],
                accepted_statuses={"success", "timeout"},
            ):
                continue
            migrated = dict(row)
            migrated.update(
                schema_version=RESULT_SCHEMA_VERSION,
                config_hash=panel["config_hash"],
                wall_time_definition=WALL_TIME_DEFINITIONS[method],
                max_time_seconds=task_limit,
                plot_time_seconds=min(overall_time, task_limit),
                result_origin="imported_success",
            )
            migrated["time"] = float(migrated["plot_time_seconds"]) - legacy_time_correction(
                panel_id, method, int(float(migrated["num_controls"]))
            )
            upsert_serialized_result(
                result_path(destination_root, panel_id, planner, run_number), migrated
            )
            imported += 1
    return imported


def mppi_result_complete(
    root: Path,
    panel_id: str,
    run_number: int,
    *,
    expected_config_hash: str,
    expected_source_hash: str,
    expected_environment_contract_hash: str,
) -> bool:
    rows = read_csv(result_path(root, panel_id, "mppi", run_number))
    matching = [
        row
        for row in rows
        if str(row.get("planner", "")).lower() == "mppi"
        and str(row.get("method", "")).lower() == "mppi"
    ]
    if len(matching) != 1:
        return False
    row = matching[0]
    try:
        return (
            int(float(row["run_number"])) == int(run_number)
            and row.get("config_hash") == expected_config_hash
            and row.get("mppi_source_hash") == expected_source_hash
            and row.get("shared_environment_contract_hash")
            == expected_environment_contract_hash
            and row.get("status") in {"success", "timeout", "failure"}
            and math.isfinite(float(row["overall_time"]))
            and math.isfinite(float(row["plot_time_seconds"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


RANDUP_TRIAL_FIELDS = (
    "schema_version", "panel_id", "system", "environment", "planner", "method",
    "method_label", "run_number", "seed", "planning_seed", "execution_seed",
    "uncertainty_level", "planning_success", "execution_success", "collision",
    "status", "failure_reason", "last_expansion_rejection",
    "initial_planning_seconds", "blocking_replanning_seconds",
    "execution_time_seconds", "actual_execution_seconds", "task_time_seconds",
    "cost", "goal_distance", "minimum_obstacle_clearance", "tree_expansions",
    "tree_size", "dynamics_propagations", "particle_propagations",
    "particle_collision_rejections", "nominal_collision_rejections",
    "numerical_propagation_failures", "num_particles", "padding_epsilon",
    "uncertainty_mode", "goal_requires_all_particles", "num_controls",
    "num_replanning", "executed_primitive_steps", "route_signature", "config_hash",
    "condition_hash", "disturbance_schedule_hash",
)


def write_mppi_result(root: Path, row: dict) -> Path:
    path = result_path(root, row["panel_id"], "mppi", int(row["run_number"]))
    serialized = serialize_result(json_value(row))
    serialized["time"] = float(serialized["plot_time_seconds"])
    parameters = row["mppi_parameters"]
    serialized.update(
        {
            "mppi_config_hash": str(row["mppi_config_hash"]),
            "mppi_source_hash": str(row["mppi_source_hash"]),
            "shared_environment_contract_hash": str(row["shared_environment_contract_hash"]),
            "mppi_horizon_steps": int(parameters["horizon_steps"]),
            "mppi_num_samples": int(parameters["num_samples"]),
            "mppi_temperature": float(parameters["temperature"]),
            "mppi_goal_cost_mode": str(parameters["goal_cost_mode"]),
            "mppi_goal_cost_scale": float(parameters["goal_cost_scale"]),
            "mppi_goal_threshold": float(row["mppi_goal_threshold"]),
            "mppi_smoothness_cost_scale": float(parameters["smoothness_cost_scale"]),
            "mppi_position_cost_scale": float(parameters.get("position_cost_scale", 1.0)),
            "mppi_yaw_cost_scale": float(parameters.get("yaw_cost_scale", 1.0)),
            "mppi_execution_device": str(row["execution_device"]),
            "mppi_torch_intraop_threads": int(row["torch_intraop_threads"]),
            "mppi_torch_interop_threads": int(row["torch_interop_threads"]),
            "mppi_control_noise_std": compact_numbers(parameters["control_noise_std"]),
            "mppi_computation_seconds": float(row["optimizer_seconds"]),
            "model_execution_seconds": float(row["model_execution_seconds"]),
            "physical_execution_seconds": float(row["physical_execution_seconds"]),
            "physical_execution_step_seconds": float(row["physical_execution_step_seconds"]),
            "physical_control_duration_seconds": compact_numbers(row["physical_control_duration_seconds"]),
            "mppi_effective_sample_size": compact_numbers(
                item["effective_sample_size"] for item in row.get("command_diagnostics", [])
            ),
            "measured_states": compact_controls(row.get("primitive_states", [])),
        }
    )
    upsert_serialized_result(path, serialized)
    return path


def randup_result_path(root: Path, row: dict) -> Path:
    return (
        Path(root)
        / str(row["panel_id"])
        / "artifacts"
        / f"{row['planner']}_trial-{int(row['run_number']):03d}.json"
    )


def rewrite_randup_trials(root: Path) -> Path:
    paths = sorted(
        path
        for panel_id in PANEL_LAYOUT
        for path in (Path(root) / panel_id / "artifacts").glob(
            "randup_rrt*_trial-*.json"
        )
    )
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    output = Path(root) / "randup_trials.csv"
    write_csv(
        output,
        ({field: row.get(field) for field in RANDUP_TRIAL_FIELDS} for row in rows),
        RANDUP_TRIAL_FIELDS,
    )
    return output


def write_randup_result(root: Path, row: dict, *, rebuild_trials_csv: bool = True) -> Path:
    path = randup_result_path(root, row)
    write_json(path, row)
    if rebuild_trials_csv:
        rewrite_randup_trials(root)
    if str(row["panel_id"]) in PANEL_LAYOUT:
        compact_path = result_path(
            root, str(row["panel_id"]), str(row["planner"]), int(row["run_number"])
        )
        serialized = serialize_result(json_value(row))
        serialized.update(
            {
                "method_label": row["method_label"],
                "planning_success": int(bool(row["planning_success"])),
                "execution_success": int(bool(row["execution_success"])),
                "collision": int(bool(row["collision"])),
                "uncertainty_level": float(row["uncertainty_level"]),
                "num_particles": int(row["num_particles"]),
                "padding_epsilon": float(row["padding_epsilon"]),
                "tree_expansions": int(row["tree_expansions"]),
                "tree_size": int(row["tree_size"]),
                "dynamics_propagations": int(row["dynamics_propagations"]),
                "particle_propagations": int(row["particle_propagations"]),
                "minimum_obstacle_clearance": row["minimum_obstacle_clearance"],
                "route_signature": row["route_signature"],
                "randup_condition_hash": row["condition_hash"],
            }
        )
        upsert_serialized_result(compact_path, serialized)
    return path


def write_randup_summary(root: Path, summaries: list[dict]) -> tuple[Path, Path]:
    csv_path = Path(root) / "randup_summary.csv"
    markdown_path = Path(root) / "randup_summary.md"
    write_csv(csv_path, summaries, list(summaries[0]) if summaries else [])
    lines = [
        "# RandUP-RRT validation summary",
        "",
        "Rates include every trial. Means show only defined finite samples; valid sample counts are in the CSV.",
        "",
        "| Method | M | Uncertainty | Plan found | Task success | Collision | Planning time | Execution time | Total time | Cost | Final error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        def metric(name: str) -> str:
            value = row[f"{name}_mean"]
            count = row[f"{name}_valid_n"]
            return "—" if value is None else f"{float(value):.4g} [{count}]"

        lines.append(
            f"| {row['method']} | {row['num_particles']} | {row['uncertainty_level']:.3g} | "
            f"{100.0 * row['planning_success_rate']:.1f}% | "
            f"{100.0 * row['overall_task_success_rate']:.1f}% | "
            f"{100.0 * row['collision_rate']:.1f}% | {metric('planning_time')} | "
            f"{metric('execution_time')} | {metric('total_time')} | {metric('cost')} | "
            f"{metric('final_error')} |"
        )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path
