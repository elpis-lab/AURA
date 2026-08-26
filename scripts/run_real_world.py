#!/usr/bin/env python3
"""Run one operator-gated MPPI or RandUp trial on the physical pushing setup."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SYSTEM_CONFIG = REPO_ROOT / "configs" / "systems" / "pushing_object.yaml"
DEFAULT_EXPERIMENT_CONFIG = (
    REPO_ROOT / "configs" / "experiments" / "task_time_efficiency.yaml"
)

METHODS = ("mppi", "randup")
EXPECTED_TRIALS = 20
RUNTIME_READY_ENV = "AURA_REAL_WORLD_RUNTIME_READY"
CANONICAL_RESULTS_ROOT = REPO_ROOT / "results/full_time_comparison"


def prepend_paths(paths: list[Path], existing: str | None) -> str:
    values = [str(path) for path in paths if path.is_dir()]
    if existing:
        values.append(existing)
    return os.pathsep.join(values)


def ensure_runtime_environment() -> None:
    """Relaunch live trials once with the repository's Python 3.10 runtime."""

    if os.environ.get(RUNTIME_READY_ENV) == "1":
        return

    python_executable = sys.executable
    if sys.version_info[:2] != (3, 10):
        python_executable = shutil.which("python3.10") or ""
        if not python_executable:
            raise RuntimeError("live real-world trials require Python 3.10")

    torch_environment = Path(
        os.environ.get("AURA_TORCH_ENV", Path.home() / "pytorch-gpu")
    ).expanduser()
    torch_packages = torch_environment / "lib/python3.10/site-packages"
    cudnn_library = torch_packages / "nvidia/cudnn/lib"
    if not torch_packages.is_dir() or not (cudnn_library / "libcudnn.so.9").is_file():
        raise RuntimeError(
            "real-world Torch runtime not found; expected Python packages and "
            f"libcudnn.so.9 under {torch_environment}"
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = prepend_paths(
        [
            REPO_ROOT / ".deps/ompl/lib/python3.10/site-packages",
            REPO_ROOT / ".deps/python",
            torch_packages,
        ],
        environment.get("PYTHONPATH"),
    )
    environment["LD_LIBRARY_PATH"] = prepend_paths(
        [cudnn_library], environment.get("LD_LIBRARY_PATH")
    )
    environment[RUNTIME_READY_ENV] = "1"
    command = [python_executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    print("[RUNTIME] Loading the project Python 3.10/Torch environment.", flush=True)
    os.execvpe(python_executable, command, environment)
    raise RuntimeError("failed to relaunch the real-world runtime")


def stream_seed(master_seed: int, *labels: object) -> int:
    payload = ":".join([str(int(master_seed)), *(str(value) for value in labels)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"real-world config must be a mapping: {path}")
    if "environments" not in loaded:
        return loaded
    experiment = yaml.safe_load(
        DEFAULT_EXPERIMENT_CONFIG.read_text(encoding="utf-8")
    ) or {}
    config = {
        **{
            key: value
            for key, value in loaded.items()
            if key not in {"environments", "title"}
        },
        **loaded["environments"]["real"],
        **experiment["condition_hyperparameters"]["pushing_real"],
    }
    real_methods = experiment.get("real_world_methods", {})
    config["mppi"] = dict(real_methods.get("mppi", {}))
    config.update(real_methods.get("randup", {}))
    config["num_trials"] = int(experiment["num_real_trials"])
    return config


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(json_value(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_config(config: dict[str, Any], method: str) -> None:
    if str(config.get("system_name")) != "pushing_object":
        raise ValueError("real-world runner requires system_name: pushing_object")
    if str(config.get("simulator_mode")) != "real":
        raise ValueError("real-world runner requires simulator_mode: real")
    if int(config.get("num_trials", 0)) != EXPECTED_TRIALS:
        raise ValueError(f"real-world config must define {EXPECTED_TRIALS} trials")
    if str(config.get("model_name", "")).strip() == "":
        raise ValueError("real-world model_name is required")
    model_path = repo_path(str(config.get("model_path", "")))
    if not model_path.is_file():
        raise FileNotFoundError(f"real-world learned model not found: {model_path}")
    object_path = repo_path(
        f"simulation/assets/{config.get('object_name', 'cracker_box_flipped')}/textured.obj"
    )
    if not object_path.is_file():
        raise FileNotFoundError(f"physical object mesh not found: {object_path}")
    push_duration = float(config.get("physical_push_duration_seconds", 0.0))
    if not math.isfinite(push_duration) or push_duration < 4.0:
        raise ValueError(
            "real physical pushes must take at least 4.0 seconds for smooth execution"
        )
    if not math.isclose(float(config.get("execution_dt", 0.0)), 0.008):
        raise ValueError("physical pushes must retain the collected 0.008-second dt")
    bounds = np.asarray(config.get("control_bounds"), dtype=float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
        raise ValueError("real control_bounds must be a finite 3x2 array")
    workspace = np.asarray(config.get("object_workspace_bounds"), dtype=float)
    if workspace.shape != (4,) or not np.isfinite(workspace).all():
        raise ValueError("object_workspace_bounds must be [xmin, xmax, ymin, ymax]")
    if workspace[0] >= workspace[1] or workspace[2] >= workspace[3]:
        raise ValueError("object_workspace_bounds intervals must be increasing")
    for state_name in ("start_state", "goal_state"):
        state = np.asarray(config.get(state_name), dtype=float)
        if state.shape != (3,) or not np.isfinite(state).all():
            raise ValueError(f"{state_name} must contain three finite values")
        if not (
            workspace[0] <= state[0] <= workspace[1]
            and workspace[2] <= state[1] <= workspace[3]
        ):
            raise ValueError(f"{state_name} lies outside object_workspace_bounds")
    goal_threshold = float(config.get("goal_threshold", 0.0))
    planner_goal_threshold = float(config.get("planner_goal_threshold", 0.0))
    if not math.isfinite(goal_threshold) or goal_threshold <= 0.0:
        raise ValueError("goal_threshold must be positive and finite")
    if not math.isfinite(planner_goal_threshold) or planner_goal_threshold <= 0.0:
        raise ValueError("planner_goal_threshold must be positive and finite")
    if planner_goal_threshold > goal_threshold:
        raise ValueError("planner_goal_threshold cannot exceed goal_threshold")
    start_threshold = float(config.get("start_threshold", 0.0))
    if not math.isfinite(start_threshold) or start_threshold <= 0.0:
        raise ValueError("start_threshold must be positive")
    gripper_velocity = float(config.get("gripper_velocity_percent", 0.0))
    if not math.isfinite(gripper_velocity) or not 1.0 <= gripper_velocity <= 100.0:
        raise ValueError("gripper_velocity_percent must be within [1, 100]")
    gripper_force = float(config.get("gripper_force_percent", 0.0))
    if not math.isclose(gripper_force, 100.0):
        raise ValueError("the real rod grasp requires gripper_force_percent: 100.0")
    if method == "mppi":
        values = dict(config.get("mppi") or {})
        if int(values.get("horizon_steps", 0)) < 1:
            raise ValueError("mppi.horizon_steps must be positive")
        if int(values.get("num_samples", 0)) < 2:
            raise ValueError("mppi.num_samples must be at least two")
        if len(values.get("control_noise_std", [])) != 3:
            raise ValueError("mppi.control_noise_std must have three values")
    else:
        if int(config.get("randup_num_particles", 0)) < 1:
            raise ValueError("randup_num_particles must be positive")
        replanning_threshold = float(config.get("replanning_max_distance", 0.0))
        if not math.isfinite(replanning_threshold) or replanning_threshold <= 0.0:
            raise ValueError("replanning_max_distance must be positive and finite")


def state_from_pose(pose: np.ndarray) -> np.ndarray:
    from geometry.pose import matrix_to_flat, project_se3_to_se2, wrap_to_pi

    state = project_se3_to_se2(matrix_to_flat(np.asarray(pose, dtype=float)))
    state = np.asarray(state, dtype=float).reshape(-1)[:3]
    state[2] = wrap_to_pi(float(state[2]))
    return state


def push_frame_summary(
    object_state: np.ndarray,
    control: np.ndarray,
    path: np.ndarray,
) -> dict[str, float]:
    """Describe an object-relative face and its resulting global push heading."""

    state = np.asarray(object_state, dtype=float).reshape(3)
    push = np.asarray(control, dtype=float).reshape(3)
    waypoints = np.asarray(path, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] < 2:
        raise ValueError("physical push path must have at least two XY waypoints")

    face_index = int(np.round(float(push[0]) * 4.0)) % 4
    relative_face = face_index * np.pi / 2.0
    contact_bearing = math.atan2(
        math.sin(float(state[2]) + relative_face),
        math.cos(float(state[2]) + relative_face),
    )
    expected_motion = math.atan2(
        math.sin(contact_bearing + np.pi),
        math.cos(contact_bearing + np.pi),
    )
    path_delta = waypoints[-1, :2] - waypoints[0, :2]
    if float(np.linalg.norm(path_delta)) <= 1e-9:
        raise ValueError("physical push path has no XY motion")
    path_motion = math.atan2(float(path_delta[1]), float(path_delta[0]))
    heading_error = abs(
        math.atan2(
            math.sin(path_motion - expected_motion),
            math.cos(path_motion - expected_motion),
        )
    )
    if heading_error > 1e-6:
        raise RuntimeError(
            "generated physical push heading does not match the object-relative face: "
            f"error {math.degrees(heading_error):.6f} degrees"
        )
    return {
        "object_yaw_degrees": math.degrees(float(state[2])),
        "relative_face_degrees": math.degrees(relative_face),
        "global_contact_bearing_degrees": math.degrees(contact_bearing),
        "global_push_heading_degrees": math.degrees(path_motion),
    }


def make_push_path(
    object_pose: np.ndarray,
    object_shape: np.ndarray,
    control: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Use the physical data-collection path generator without a second wrapper."""

    from geometry.pose import matrix_to_flat
    from geometry.random_push import generate_path_from_params

    tool_offset = np.array(
        [0.0, 0.0, -float(config["push_offset"]), 1.0, 0.0, 0.0, 0.0]
    )
    _, paths = generate_path_from_params(
        matrix_to_flat(np.asarray(object_pose, dtype=float))[None, :],
        np.asarray(object_shape, dtype=float),
        np.asarray(control, dtype=float).reshape(1, 3),
        tool_offset=tool_offset,
        pre_push_offset=float(config["pre_push_offset"]),
        duration=float(config["physical_push_duration_seconds"]),
        dt=float(config["execution_dt"]),
        push_height=config.get("push_height"),
        relative_push_offset=True,
    )
    return paths[0]


def configure_system(config: dict[str, Any]):
    from geometry.object_model import get_obj_shape
    from propagators import get_system

    system = get_system("pushing_object")
    system.set_control_bounds(config["control_bounds"])
    xmin, xmax, ymin, ymax = (
        float(value) for value in config["object_workspace_bounds"]
    )
    state_bounds = [
        (xmin, xmax),
        (ymin, ymax),
    ]
    system.set_state_bounds(state_bounds)
    system.object_shape = np.asarray(
        get_obj_shape(
            str(
                repo_path(
                    f"simulation/assets/{config.get('object_name', 'cracker_box_flipped')}/textured.obj"
                )
            )
        ),
        dtype=float,
    )
    system.model_name = str(config["model_name"])
    system.model_path = str(repo_path(config["model_path"]))
    system.configure_duration_contract(
        float(config["propagation_step_size"]),
        int(config["min_control_duration"]),
        int(config["max_control_duration"]),
    )
    return system, state_bounds


def state_in_workspace(state: np.ndarray, config: dict[str, Any]) -> bool:
    xmin, xmax, ymin, ymax = (
        float(value) for value in config["object_workspace_bounds"]
    )
    values = np.asarray(state, dtype=float).reshape(-1)
    return bool(xmin <= values[0] <= xmax and ymin <= values[1] <= ymax)


def goal_distance(state: np.ndarray, config: dict[str, Any]) -> float:
    from utils.utils import arrayDistance

    return float(
        arrayDistance(
            np.asarray(state, dtype=float),
            np.asarray(config["goal_state"], dtype=float),
            system="pushing_object",
        )
    )


def start_distance(state: np.ndarray, config: dict[str, Any]) -> float:
    measured = np.asarray(state, dtype=float).reshape(3)
    configured = np.asarray(config["start_state"], dtype=float).reshape(3)
    position_distance = float(np.linalg.norm(measured[:2] - configured[:2]))
    angle_delta = float(measured[2] - configured[2])
    angle_distance = abs(math.atan2(math.sin(angle_delta), math.cos(angle_delta)))
    return position_distance + angle_distance


def result_locations(
    results_root: Path,
    method: str,
    trial: int,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    if method == "mppi":
        planner = "mppi"
    else:
        planner = f"randup_rrt_m{int(config['randup_num_particles'])}"
    csv_path = results_root / "pushing_real" / f"{planner}_{trial:03d}.csv"
    artifact = (
        results_root
        / "pushing_real"
        / "artifacts"
        / f"{planner}_trial-{trial:03d}.json"
    )
    return csv_path, artifact


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(json_value(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def summarize_results(results_root: Path) -> dict[str, Any]:
    rows = []
    planning_root = results_root / "pushing_real"
    for path in sorted(planning_root.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    summaries = []
    for method in METHODS:
        group = [
            row
            for row in rows
            if str(row.get("panel_id")) == "pushing_real"
            and (
                str(row.get("method", "")).lower() == "mppi"
                if method == "mppi"
                else "randup" in str(row.get("method", "")).lower()
                or "randup" in str(row.get("planner", "")).lower()
            )
        ]
        successes = [row for row in group if str(row.get("status")) == "success"]
        successful_times = [float(row["overall_time"]) for row in successes]
        penalized_times = [float(row["plot_time_seconds"]) for row in group]
        summaries.append(
            {
                "method": method,
                "completed_trials": len(group),
                "expected_trials": EXPECTED_TRIALS,
                "successful_trials": len(successes),
                "success_rate": len(successes) / len(group) if group else None,
                "successful_mean_seconds": (
                    float(np.mean(successful_times)) if successful_times else None
                ),
                "figure7_penalized_mean_seconds": (
                    float(np.mean(penalized_times)) if penalized_times else None
                ),
            }
        )
    return {"results_root": str(results_root), "real_world": summaries}


def dry_run_spec(
    method: str,
    trial: int,
    seed: int,
    config_path: Path,
    results_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    csv_path, artifact_path = result_locations(results_root, method, trial, config)
    return {
        "dry_run": True,
        "robot_connection_attempted": False,
        "one_trial_only": True,
        "method": method,
        "trial": trial,
        "seed": seed,
        "config": str(config_path),
        "config_hash": config_hash(config),
        "learned_model": str(repo_path(config["model_path"])),
        "start_state": list(config["start_state"]),
        "start_threshold": float(config["start_threshold"]),
        "goal_state": list(config["goal_state"]),
        "goal_threshold": float(config["goal_threshold"]),
        "task_time_limit_seconds": float(
            config.get("task_time_limit_seconds", 300.0)
        ),
        "existing_csv_will_be_replaced": csv_path.exists(),
        "existing_artifact_will_be_replaced": artifact_path.exists(),
        "replacement_occurs_after_rerun_returns_a_result": True,
        "gripper_action": "close",
        "gripper_policy": "closed_with_torque_enabled_for_entire_trial",
        "gripper_force_percent": float(config["gripper_force_percent"]),
        "object_workspace_bounds": list(config["object_workspace_bounds"]),
        "physical_push_contract": {
            "generator": "geometry.random_push.generate_path_from_params",
            "duration_seconds": float(config["physical_push_duration_seconds"]),
            "dt_seconds": float(config["execution_dt"]),
            "tool_offset_z": -float(config["push_offset"]),
            "pre_push_offset": float(config["pre_push_offset"]),
            "push_height": config.get("push_height"),
            "relative_side_offset": True,
        },
        "csv_output": str(csv_path),
        "artifact_output": str(artifact_path),
    }


def close_gripper_for_trial(robot, config: dict[str, Any]):
    """Grip the rod before arm motion and keep maximum configured force."""

    result = robot.control_gripper(
        "close",
        velocity_percent=float(config["gripper_velocity_percent"]),
        force_percent=float(config["gripper_force_percent"]),
        wait=True,
    )
    if result is not None:
        outcome = "loaded contact grasp" if result.contact_grasp else "closed endpoint"
        print(
            f"[REAL] RH-P12-RN {outcome}: raw position "
            f"{result.start_position} -> {result.final_position}, "
            f"force {result.force_percent:.1f}%, torque enabled."
        )
    return result


def make_randup_plan(
    system,
    current: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any] | None, float]:
    from methods.RandUpRRT import RandUpRRTConfig
    from methods.plan import OMPLPlanner

    mapping = dict(config)
    mapping["randup_random_seed"] = int(seed)
    randup = RandUpRRTConfig.from_mapping(mapping)
    randup.validate()
    planner = OMPLPlanner(
        system=system,
        start_state=current,
        goal_state=np.asarray(config["goal_state"], dtype=float),
        planner_method="randup_rrt",
        goal_threshold=float(config["goal_threshold"]),
        min_max_control_duration=(
            int(randup.control_duration_min),
            int(randup.control_duration_max),
        ),
        propagation_step_size=float(config["propagation_step_size"]),
        initial_planning_time=float(randup.planning_time),
        pruning_radius=float(config.get("pruning_radius", 0.08)),
        goal_bias=float(randup.goal_bias),
        optimization_objective=str(config.get("optimization_objective", "control_duration")),
        randup_config=randup,
        single_solve=True,
    )
    started = time.monotonic()
    planner.plan()
    elapsed = time.monotonic() - started
    return planner.best_solution(), elapsed


def expand_randup_plan(
    system,
    start_state: np.ndarray,
    solution: dict[str, Any],
    propagation_step_size: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Expand variable-duration RandUp edges into physical push references."""

    edge_controls = list(solution.get("controls", []))
    duration_steps = list(solution.get("time_steps", []))
    if len(edge_controls) != len(duration_steps):
        raise ValueError("RandUp controls and time_steps must have equal lengths")

    current = np.asarray(start_state, dtype=float).reshape(3).copy()
    expanded_controls: list[np.ndarray] = []
    expected_states = [current.copy()]
    for edge_control, edge_duration in zip(edge_controls, duration_steps):
        control = np.asarray(edge_control, dtype=float).reshape(3)
        steps = int(edge_duration)
        if steps < 1:
            raise ValueError("RandUp control durations must be positive")
        for _ in range(steps):
            current = np.asarray(
                system.propagate(current, control, float(propagation_step_size)),
                dtype=float,
            ).reshape(3)
            if not np.isfinite(current).all():
                raise ValueError("RandUp expected trajectory contains nonfinite states")
            expanded_controls.append(control.copy())
            expected_states.append(current.copy())
    return expanded_controls, expected_states


def print_randup_plan(
    plan_number: int,
    planner_control_count: int,
    expanded_controls: list[np.ndarray],
    expected_states: list[np.ndarray],
) -> None:
    """Print the dense expected trajectory for one fresh RandUp plan."""

    print(
        f"[REAL] RANDUP plan {int(plan_number)}: "
        f"{int(planner_control_count)} planner controls expanded to "
        f"{len(expanded_controls)} physical pushes"
    )
    print(f"[REAL] RANDUP plan {int(plan_number)} expected trajectory:")
    for step, state in enumerate(expected_states):
        pose = np.array2string(
            np.asarray(state, dtype=float), precision=5, suppress_small=True
        )
        print(f"  step {step}: {pose}")


def randup_requires_replanning(
    measured_state: np.ndarray,
    expected_state: np.ndarray,
    threshold: float,
) -> tuple[float, bool]:
    """Compare the measured pose with the original RandUp plan reference."""

    from utils.utils import arrayDistance

    tracking_error = float(
        arrayDistance(measured_state, expected_state, system="pushing_object")
    )
    return tracking_error, tracking_error > float(threshold)


def print_step_progress(
    method: str,
    step: int,
    state: np.ndarray,
    distance: float,
) -> None:
    pose = np.array2string(
        np.asarray(state, dtype=float), precision=5, suppress_small=True
    )
    print(
        f"[REAL] {method.upper()} step {int(step)}: "
        f"pose={pose}; goal distance={float(distance):.6f}"
    )


def recorded_task_time(status: str, elapsed: float, limit: float) -> float:
    """Store the exact experiment limit for a timed-out trial."""

    return float(limit) if status == "timeout" else float(elapsed)


def run_physical_trial(
    args: argparse.Namespace,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    from methods.MPPI import MPPIController, parameters_from_config
    from real_world.physical_robot import PhysicalUR10
    from real_world.real_execution import (
        ROBOT_HOME_ACCELERATION,
        ROBOT_HOME_SPEED,
        detect_after_push,
        execute_push,
        two_step_detection,
    )
    from simulation.pushing_model import get_pushing_model
    from utils.utils import arrayDistance

    setup_started = time.monotonic()
    system, state_bounds = configure_system(config)
    get_pushing_model(
        system.object_shape,
        model_name=str(config["model_name"]),
        model_path=str(repo_path(config["model_path"])),
    )
    controller = None
    if args.method == "mppi":
        parameters = parameters_from_config("pushing_object", config)
        controller = MPPIController(
            system,
            config["goal_state"],
            propagation_step_size=float(config["propagation_step_size"]),
            goal_threshold=float(config["goal_threshold"]),
            parameters=parameters,
            model_name=str(config["model_name"]),
            model_path=str(repo_path(config["model_path"])),
            seed=stream_seed(seed, "controller"),
            device=args.device,
        )
    else:
        from ompl import util as ou

        ou.RNG.setSeed(max(1, stream_seed(seed, "ompl_exploration")))

    robot = PhysicalUR10(
        robot_ip=str(config["robot_ip"]),
        camera_host=str(config["camera_host"]),
    )
    close_gripper_for_trial(robot, config)
    robot.move_joint(
        np.asarray(config["robot_home"], dtype=float),
        speed=ROBOT_HOME_SPEED,
        acceleration=ROBOT_HOME_ACCELERATION,
        label="MPPI/RandUp setup: robot home",
    )
    rough_pose = np.asarray(config["rough_detect_pose"], dtype=float)
    object_pose, _ = two_step_detection(
        robot, rough_pose, height=float(config["detect_height"])
    )
    current = state_from_pose(object_pose)
    if not state_in_workspace(current, config):
        raise RuntimeError(
            "detected object center is outside object_workspace_bounds; "
            "do not move the robot until the table bounds are checked"
        )
    reset_distance = start_distance(current, config)
    if reset_distance > float(config["start_threshold"]):
        raise RuntimeError(
            "detected object is not at the configured start_state: "
            f"distance {reset_distance:.6f} exceeds start_threshold "
            f"{float(config['start_threshold']):.6f}; reset the object and retry"
        )
    if goal_distance(current, config) <= float(config["goal_threshold"]):
        raise RuntimeError("object already satisfies the goal; reset it before this trial")
    setup_seconds = time.monotonic() - setup_started

    measured_states = [current.copy()]
    predicted_states: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    tracking_errors: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    randup_queue: list[tuple[np.ndarray, np.ndarray]] = []
    first_randup_solution = None
    initial_planning_seconds = 0.0
    blocking_replanning_seconds = 0.0
    optimizer_seconds = 0.0
    actual_execution_seconds = 0.0
    path_cost = 0.0
    plan_count = 0
    successful_plan_count = 0
    status = "failure"
    failure_reason = ""
    trial_started = time.monotonic()
    max_pushes = int(config.get("max_real_pushes", 100))
    task_limit = float(config.get("task_time_limit_seconds", 300.0))
    replanning_threshold = float(config.get("replanning_max_distance", 0.1))
    print_step_progress(args.method, 0, current, goal_distance(current, config))

    try:
        while len(controls) < max_pushes:
            charged_time = (
                initial_planning_seconds
                + blocking_replanning_seconds
                + optimizer_seconds
                + actual_execution_seconds
            )
            if goal_distance(current, config) <= float(config["goal_threshold"]):
                if charged_time <= task_limit:
                    status = "success"
                else:
                    status = "timeout"
                    failure_reason = "goal_reached_after_task_time_limit"
                break
            if charged_time >= task_limit:
                status = "timeout"
                failure_reason = "task_time_limit_reached"
                break

            if args.method == "mppi":
                command_started = time.monotonic()
                control, diagnostic = controller.command(current)
                command_seconds = time.monotonic() - command_started
                optimizer_seconds += command_seconds
                diagnostics.append({**diagnostic, "computation_seconds": command_seconds})
            else:
                if not randup_queue:
                    plan_seed = stream_seed(seed, "randup_plan", plan_count)
                    solution, planning_seconds = make_randup_plan(
                        system, current, config, plan_seed
                    )
                    if not controls:
                        initial_planning_seconds += planning_seconds
                    else:
                        blocking_replanning_seconds += planning_seconds
                    plan_count += 1
                    if solution is None:
                        planning_total = (
                            initial_planning_seconds + blocking_replanning_seconds
                        )
                        if planning_total >= task_limit:
                            status = "timeout"
                            failure_reason = "randup_planning_retries_reached_deadline"
                            break
                        continue
                    if first_randup_solution is None:
                        first_randup_solution = solution
                    successful_plan_count += 1
                    expanded_controls, expected_trajectory = expand_randup_plan(
                        system,
                        current,
                        solution,
                        float(config["propagation_step_size"]),
                    )
                    print_randup_plan(
                        successful_plan_count,
                        len(solution["controls"]),
                        expanded_controls,
                        expected_trajectory,
                    )
                    randup_queue.extend(
                        (control, expected_state)
                        for control, expected_state in zip(
                            expanded_controls, expected_trajectory[1:]
                        )
                    )
                    if not randup_queue:
                        failure_reason = "randup_returned_empty_plan"
                        break
                control, randup_expected_state = randup_queue.pop(0)

            charged_time = (
                initial_planning_seconds
                + blocking_replanning_seconds
                + optimizer_seconds
                + actual_execution_seconds
            )
            if charged_time >= task_limit:
                status = "timeout"
                failure_reason = "task_time_limit_reached"
                break

            control = np.asarray(control, dtype=float).reshape(3)
            predicted = np.asarray(
                system.propagate(
                    current, control, float(config["propagation_step_size"])
                ),
                dtype=float,
            ).reshape(3)
            if not np.isfinite(predicted).all():
                failure_reason = "learned_model_returned_nonfinite_prediction"
                break
            if any(
                predicted[index] < low or predicted[index] > high
                for index, (low, high) in enumerate(state_bounds)
            ):
                failure_reason = "predicted_object_state_outside_planner_bounds"
                break

            path = make_push_path(object_pose, system.object_shape, control, config)
            execution_started = time.monotonic()
            execute_push(robot, path, control)
            wait_seconds = float(config.get("detection_wait_seconds", 0.0))
            if wait_seconds > 0.0:
                time.sleep(wait_seconds)
            completed_pushes = len(controls) + 1
            object_pose, _ = detect_after_push(
                robot,
                rough_detect_pose=rough_pose,
                completed_pushes=completed_pushes,
                height=float(config["detect_height"]),
                debug_img_id=f"{args.method}_{args.trial}_{completed_pushes}",
            )
            actual_execution_seconds += time.monotonic() - execution_started
            measured = state_from_pose(object_pose)
            if args.method == "randup":
                tracking_error, needs_fresh_plan = randup_requires_replanning(
                    measured,
                    randup_expected_state,
                    replanning_threshold,
                )
                prediction_reference = randup_expected_state
            else:
                tracking_error = float(
                    arrayDistance(measured, predicted, system="pushing_object")
                )
                needs_fresh_plan = False
                prediction_reference = predicted
            tracking_errors.append(tracking_error)
            path_cost += float(arrayDistance(current, measured, system="pushing_object"))
            controls.append(control.copy())
            predicted_states.append(np.asarray(prediction_reference, dtype=float).copy())
            current = measured
            measured_states.append(current.copy())
            print_step_progress(
                args.method,
                completed_pushes,
                current,
                goal_distance(current, config),
            )
            if (
                args.method == "randup"
                and needs_fresh_plan
                and goal_distance(current, config) > float(config["goal_threshold"])
            ):
                remaining_pushes = len(randup_queue)
                randup_queue.clear()
                print(
                    f"[REAL] RANDUP deviation {tracking_error:.6f} exceeds "
                    f"replanning threshold {replanning_threshold:.6f}; "
                    f"discarding {remaining_pushes} queued physical pushes and "
                    "replanning from the measured pose"
                )
        else:
            charged_time = (
                initial_planning_seconds
                + blocking_replanning_seconds
                + optimizer_seconds
                + actual_execution_seconds
            )
            if charged_time >= task_limit:
                status = "timeout"
                failure_reason = "task_time_limit_reached"
            elif (
                goal_distance(current, config) <= float(config["goal_threshold"])
            ):
                status = "success"
            else:
                failure_reason = "maximum_real_pushes_reached"
    except KeyboardInterrupt:
        try:
            robot.rtde.rtde_c.servoStop()
        except Exception:
            pass
        raise
    except Exception as error:
        status = "failure"
        failure_reason = f"exception:{error!r}"
        try:
            robot.rtde.rtde_c.servoStop()
        except Exception:
            pass

    if status != "success" and not failure_reason:
        failure_reason = "goal_not_reached"
    elapsed_task_time = (
        initial_planning_seconds
        + blocking_replanning_seconds
        + optimizer_seconds
        + actual_execution_seconds
    )
    task_time_seconds = recorded_task_time(status, elapsed_task_time, task_limit)
    planner_name = (
        "mppi"
        if args.method == "mppi"
        else f"randup_rrt_m{int(config['randup_num_particles'])}"
    )
    method_name = "mppi" if args.method == "mppi" else "randup_rrt"
    final_prediction = predicted_states[-1] if predicted_states else current
    return {
        "schema_version": 1,
        "panel_id": "pushing_real",
        "system": "pushing_object",
        "environment": "real",
        "planner": planner_name,
        "method": method_name,
        "run_number": int(args.trial),
        "seed": int(seed),
        "config_hash": config_hash(config),
        "start_state": np.asarray(config["start_state"], dtype=float),
        "measured_initial_state": measured_states[0],
        "goal_state": np.asarray(config["goal_state"], dtype=float),
        "initial_planning_seconds": initial_planning_seconds,
        "initial_plan_hash": None,
        "initial_plan": first_randup_solution,
        "status": status,
        "failure_reason": failure_reason,
        "nominal_execution_seconds": len(controls)
        * float(config["physical_push_duration_seconds"]),
        "actual_execution_seconds": actual_execution_seconds,
        "online_replanning_seconds": 0.0,
        "optimizer_seconds": optimizer_seconds,
        "blocking_replanning_seconds": blocking_replanning_seconds,
        "compute_overrun_seconds": 0.0,
        "wall_time_definition": (
            "measured blocking method computation plus measured robot push and "
            "post-push detection time; setup/reset excluded"
        ),
        "task_time_seconds": task_time_seconds,
        "raw_process_wall_seconds": time.monotonic() - trial_started,
        "setup_reset_operator_pause_seconds": setup_seconds,
        "num_controls": len(controls),
        "num_replanning": (
            len(controls)
            if args.method == "mppi"
            else max(0, successful_plan_count - 1)
        ),
        "cost": path_cost,
        "tracking_error_mean": float(np.mean(tracking_errors)) if tracking_errors else 0.0,
        "tracking_error_list": tracking_errors,
        "goal_distance": goal_distance(current, config),
        "final_state": current,
        "planned_final_state": final_prediction,
        "controls": controls,
        "control_duration_steps": [1] * len(controls),
        "control_duration_seconds": [
            float(config["propagation_step_size"])
        ]
        * len(controls),
        "physical_control_duration_seconds": [
            float(config["physical_push_duration_seconds"])
        ]
        * len(controls),
        "primitive_states": measured_states,
        "command_diagnostics": diagnostics,
        "duration_audit_initial_tree": {
            "range_steps": (
                [1, 1]
                if args.method == "mppi"
                else [
                    int(config["randup_control_duration_min"]),
                    int(config["randup_control_duration_max"]),
                ]
            ),
            "propagation_step_size_seconds": float(config["propagation_step_size"]),
            "duration_step_histogram": {},
        },
    }


def save_trial(
    result: dict[str, Any],
    results_root: Path,
    method: str,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    from utils.experiment_io import upsert_result
    from scripts.plot_task_time import make_figure

    csv_path, artifact_path = result_locations(
        results_root, method, int(result["run_number"]), config
    )
    write_json(artifact_path, result)
    upsert_result(csv_path, result)
    try:
        make_figure(results_root)
    except Exception as exc:
        print(
            "[WARNING] Trial results were saved, but the task-time plot "
            f"could not be regenerated: {exc}",
            file=sys.stderr,
            flush=True,
        )
    return csv_path, artifact_path


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or execute exactly one physical MPPI/RandUp trial."
    )
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--trial", type=int)
    parser.add_argument(
        "--config", default=str(DEFAULT_SYSTEM_CONFIG.relative_to(REPO_ROOT))
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execute", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summary", action="store_true")
    mode.add_argument(
        "--test-gripper",
        choices=["open", "close"],
        help="Run only one low-speed RH-P12-RN command; no arm or camera motion.",
    )
    return parser.parse_args(arguments)


def run_command(args: argparse.Namespace) -> None:
    if args.test_gripper:
        config_path = repo_path(args.config)
        config = load_config(config_path)
        print(
            json.dumps(
                {
                    "gripper_only": True,
                    "action": args.test_gripper,
                    "robot_ip": config["robot_ip"],
                    "confirmation_required": False,
                    "velocity_percent": config["gripper_velocity_percent"],
                    "force_percent": config["gripper_force_percent"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        from real_world.rtde import RTDE

        rtde = RTDE(str(config["robot_ip"]))
        print(f"[REAL] Connected to UR controller at {config['robot_ip']}.")
        try:
            result = rtde.control_robotis_gripper(
                args.test_gripper,
                velocity_percent=float(config["gripper_velocity_percent"]),
                force_percent=float(config["gripper_force_percent"]),
            )
        finally:
            rtde.disconnect()
        print(
            f"[REAL] RH-P12-RN {args.test_gripper} verified: "
            f"raw position {result.start_position} -> {result.final_position}."
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return
    results_root = CANONICAL_RESULTS_ROOT
    if args.summary:
        print(json.dumps(summarize_results(results_root), indent=2, sort_keys=True))
        return
    if args.method is None or args.trial is None:
        raise ValueError("--method and --trial are required unless --summary is used")
    if not 1 <= int(args.trial) <= EXPECTED_TRIALS:
        raise ValueError(f"--trial must be within [1, {EXPECTED_TRIALS}]")
    config_path = repo_path(args.config)
    config = load_config(config_path)
    validate_config(config, args.method)
    seed = (
        int(args.seed)
        if args.seed is not None
        else stream_seed(260527699, "pushing_real", args.method, int(args.trial))
    )
    np.random.seed(seed)
    spec = dry_run_spec(
        args.method,
        int(args.trial),
        seed,
        config_path,
        results_root,
        config,
    )
    if not args.execute:
        print(json.dumps(spec, indent=2, sort_keys=True))
        print("\n[SAFE] Dry-run complete. No robot or camera connection was attempted.")
        return

    import torch

    torch.manual_seed(seed)
    result = run_physical_trial(args, config, seed)
    csv_path, artifact_path = save_trial(result, results_root, args.method, config)
    print(f"[REAL] status: {result['status']}")
    print(f"[REAL] task time: {float(result['task_time_seconds']):.3f}s")
    print(f"[REAL] result: {csv_path}")
    print(f"[REAL] full artifact: {artifact_path}")


def main(arguments: list[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.execute and not args.summary and not args.test_gripper:
        ensure_runtime_environment()
    run_command(args)


if __name__ == "__main__":
    main()
