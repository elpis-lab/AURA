#!/usr/bin/env python3
"""Wall-time comparison for AURA vs fresh replanning on the double integrator."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import yaml
from ompl import base as ob
from ompl import util as ou

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aura_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from AURA import AURA
from plan import OMPL_Planner
from simulators import create_simulator
from systems import get_system
from utils.utils import arrayDistance


DEFAULT_CONFIG = {
    "system_name": "double_integrator",
    "simulator_mode": "gaussian",
    "planner_names": ["aorrt", "aoest", "sststar"],
    "methods": ["aura", "replanning"],
    "num_runs": 10,
    "results_dir": "results/planning/performance_double_integrator_gaussian",
    "start_state": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
    "goal_state": [1.5, 1.5, 2.0, 0.0, 0.0, 0.0],
    "goal_threshold": 0.35,
    "state_bounds": [
        [-3.0, 3.0],
        [-3.0, 3.0],
        [0.0, 3.0],
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
    ],
    "planning_time": 4.0,
    "replanning_time_budget": 1.0,
    "max_steps": 200,
    "propagation_step_size": 0.5,
    "min_control_duration": 1,
    "max_control_duration": 2,
    "pruning_radius": 0.1,
    "replanning_max_distance": 0.2,
    "sampling_position_std": 0.003,
    "sampling_velocity_std": 0.003,
    "optimizer_position_std": 0.01,
    "optimizer_velocity_std": 0.01,
    "optimizer_num_states": 10000,
    "optimizer_num_steps": 500,
    "optimizer_learning_rate": 0.1,
    "recovery_replanning_time": 1.0,
    "visualize": False,
}

PLANNERS = ("aorrt", "aoest", "sststar")
METHODS = ("aura", "replanning")
_OMPL_SEEDED = False


@contextmanager
def _quiet_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


@dataclass
class MethodResult:
    method: str
    planner: str
    run_number: int
    status: str
    failure_reason: str
    wall_time: float
    planning_wall_time: float
    execution_wall_time: float
    replans: int
    num_controls: int
    cost: float
    tracking_error_mean: float
    goal_distance: float
    final_state: np.ndarray
    planned_final_state: np.ndarray


def _config_value(config: dict, *names: str, default=None):
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    return default


def load_config(path: str | None) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config file must contain a mapping: {path}")
        config.update(loaded)

    # Accept the older camelCase keys too.
    config["system_name"] = str(_config_value(config, "system_name", "system", default="double_integrator"))
    config["simulator_mode"] = str(_config_value(config, "simulator_mode", default="gaussian"))
    config["start_state"] = np.asarray(
        _config_value(config, "start_state", "startState", default=DEFAULT_CONFIG["start_state"]),
        dtype=float,
    )
    config["goal_state"] = np.asarray(
        _config_value(config, "goal_state", "goalState", default=DEFAULT_CONFIG["goal_state"]),
        dtype=float,
    )
    config["goal_threshold"] = float(_config_value(config, "goal_threshold", default=0.35))
    config["planning_time"] = float(
        _config_value(config, "planning_time", "planningTime", default=4.0)
    )
    config["replanning_time_budget"] = float(
        _config_value(
            config,
            "replanning_time_budget",
            "replanningTime",
            default=config["propagation_step_size"],
        )
    )
    config["propagation_step_size"] = float(
        _config_value(config, "propagation_step_size", default=0.5)
    )
    config["min_control_duration"] = int(
        _config_value(config, "min_control_duration", default=1)
    )
    config["max_control_duration"] = int(
        _config_value(config, "max_control_duration", default=2)
    )
    config["pruning_radius"] = float(_config_value(config, "pruning_radius", default=0.1))
    config["replanningMaxDistance"] = float(
        _config_value(
            config,
            "replanningMaxDistance",
            "replanning_max_distance",
            "sampling_max_distance",
            default=0.2,
        )
    )
    config["sampling_position_std"] = float(
        _config_value(config, "sampling_position_std", default=0.003)
    )
    config["sampling_rotation_std"] = 0.0
    config["sampling_velocity_std"] = float(
        _config_value(config, "sampling_velocity_std", default=0.003)
    )
    config["optimizer_num_states"] = int(
        _config_value(config, "optimizer_num_states", default=10000)
    )
    config["optimizer_num_steps"] = int(
        _config_value(config, "optimizer_num_steps", default=500)
    )
    config["optimizer_learning_rate"] = float(
        _config_value(config, "optimizer_learning_rate", default=0.1)
    )
    config["optimizer_position_std"] = float(
        _config_value(
            config,
            "optimizer_position_std",
            "optimizer_pos_std",
            default=max(0.01, 3.0 * config["sampling_position_std"]),
        )
    )
    config["optimizer_velocity_std"] = float(
        _config_value(
            config,
            "optimizer_velocity_std",
            "optimizer_vel_std",
            default=max(0.01, 3.0 * config["sampling_velocity_std"]),
        )
    )
    config["recovery_replanning_time"] = float(
        _config_value(config, "recovery_replanning_time", default=1.0)
    )
    config["max_steps"] = int(_config_value(config, "max_steps", default=200))
    config["num_runs"] = int(_config_value(config, "num_runs", default=10))
    config["visualize"] = bool(_config_value(config, "visualize", default=False))
    config["state_bounds"] = [
        (float(low), float(high)) for low, high in _config_value(
            config,
            "state_bounds",
            default=DEFAULT_CONFIG["state_bounds"],
        )
    ]
    return config


def _fmt_state(state: np.ndarray | Iterable[float]) -> str:
    arr = np.asarray(state, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{x:.4f}" for x in arr) + "]"


def _set_seed(seed: int | None) -> None:
    global _OMPL_SEEDED
    if seed is None:
        return
    np.random.seed(int(seed))
    if not _OMPL_SEEDED:
        try:
            ou.RNG.setSeed(int(seed))
            _OMPL_SEEDED = True
        except Exception:
            pass


def apply_state_bounds(system, state_bounds: list[tuple[float, float]]) -> None:
    if not state_bounds:
        return
    if len(state_bounds) != len(system.state_bounds):
        raise ValueError(
            f"{system.name} expects {len(system.state_bounds)} state bounds, "
            f"got {len(state_bounds)}"
        )
    system.state_bounds = [(float(low), float(high)) for low, high in state_bounds]
    bounds = ob.RealVectorBounds(len(system.state_bounds))
    for i, (low, high) in enumerate(system.state_bounds):
        bounds.setLow(i, float(low))
        bounds.setHigh(i, float(high))
    system.state_space.setBounds(bounds)


def build_planner(system, config: dict, planner_name: str, start_state: np.ndarray, planning_time: float):
    planner = OMPL_Planner(
        system=system,
        start_state=np.asarray(start_state, dtype=float),
        goal_state=np.asarray(config["goal_state"], dtype=float),
        planner_method=planner_name,
        goal_threshold=float(config["goal_threshold"]),
        min_max_control_duration=(
            int(config["min_control_duration"]),
            int(config["max_control_duration"]),
        ),
        propagation_step_size=float(config["propagation_step_size"]),
        initial_planning_time=float(planning_time),
        pruning_radius=float(config["pruning_radius"]),
        obstacle_config=config.get("obstacles"),
    )
    planner.replanning_time = float(config["replanningMaxDistance"])
    planner.opt_model = None
    planner.optimizer_num_states = int(config["optimizer_num_states"])
    planner.optimizer_num_steps = int(config["optimizer_num_steps"])
    planner.optimizer_learning_rate = float(config["optimizer_learning_rate"])
    planner.optimizer_pos_std = float(config["optimizer_position_std"])
    planner.optimizer_rot_std = 0.0
    planner.optimizer_vel_std = float(config["optimizer_velocity_std"])
    planner.motion_validation_step_size = float(config["propagation_step_size"])
    planner.recovery_replanning_time = float(config["recovery_replanning_time"])
    return planner


def _state_to_numpy(state_obj, state_dim: int) -> np.ndarray:
    arr = np.asarray(state_obj, dtype=float).reshape(-1)
    return arr[:state_dim].astype(float)


def _trajectory_cost(trajectory: list[np.ndarray], system_name: str) -> float:
    return float(
        sum(
            arrayDistance(trajectory[i], trajectory[i + 1], system=system_name)
            for i in range(max(0, len(trajectory) - 1))
        )
    )


def run_aura_once(planner_name: str, run_number: int, config: dict, seed: int | None) -> MethodResult:
    system = get_system(config["system_name"])
    apply_state_bounds(system, config["state_bounds"])
    simulator = create_simulator(config["system_name"], config["simulator_mode"], config=config)

    _set_seed(seed)
    wall_start = time.time()
    planner = build_planner(
        system,
        config,
        planner_name,
        np.asarray(config["start_state"], dtype=float),
        float(config["planning_time"]),
    )

    plan_start = time.time()
    solutions, _ = planner.plan()
    planning_wall_time = time.time() - plan_start
    if not solutions:
        wall_time = time.time() - wall_start
        final = np.asarray(config["start_state"], dtype=float)
        return MethodResult(
            method="aura",
            planner=planner_name,
            run_number=run_number,
            status="failure",
            failure_reason="initial_plan_failed",
            wall_time=wall_time,
            planning_wall_time=planning_wall_time,
            execution_wall_time=0.0,
            replans=0,
            num_controls=0,
            cost=float("nan"),
            tracking_error_mean=float("nan"),
            goal_distance=float(arrayDistance(final, config["goal_state"], system=config["system_name"])),
            final_state=final,
            planned_final_state=np.full_like(final, np.nan, dtype=float),
        )

    aura = AURA(system=system, planner=planner, simulator=simulator)
    run_start = time.time()
    aura_result = aura.run(pause_each_step=False)
    execution_wall_time = time.time() - run_start
    wall_time = time.time() - wall_start

    final_state = np.asarray(aura_result.final_state, dtype=float)
    planned_final = (
        np.asarray(aura_result.final_planned_state, dtype=float)
        if getattr(aura_result, "final_planned_state", None) is not None
        else np.full_like(final_state, np.nan, dtype=float)
    )
    goal_distance = float(arrayDistance(final_state, config["goal_state"], system=config["system_name"]))
    status = str(getattr(aura_result, "status", "success") or "success")
    failure_reason = str(getattr(aura_result, "failure_reason", "") or "")
    if goal_distance > float(config["goal_threshold"]):
        status = "failure"
        failure_reason = failure_reason or f"goal_not_reached_{goal_distance:.6f}"

    return MethodResult(
        method="aura",
        planner=planner_name,
        run_number=run_number,
        status=status,
        failure_reason=failure_reason,
        wall_time=wall_time,
        planning_wall_time=planning_wall_time,
        execution_wall_time=execution_wall_time,
        replans=-1,
        num_controls=int(aura_result.num_controls),
        cost=float(aura_result.cost),
        tracking_error_mean=float(aura_result.tracking_error_mean),
        goal_distance=goal_distance,
        final_state=final_state,
        planned_final_state=planned_final,
    )


def run_replanning_once(
    planner_name: str,
    run_number: int,
    config: dict,
    seed: int | None,
) -> MethodResult:
    system = get_system(config["system_name"])
    apply_state_bounds(system, config["state_bounds"])
    simulator = create_simulator(config["system_name"], config["simulator_mode"], config=config)

    _set_seed(seed)
    simulator.reset()
    state_dim = int(len(config["start_state"]))
    current_state = _state_to_numpy(simulator.get_state(), state_dim)
    goal_state = np.asarray(config["goal_state"], dtype=float)
    goal_threshold = float(config["goal_threshold"])
    tracking_errors: list[float] = []
    trajectory = [current_state.copy()]
    executed_controls = 0
    executed_duration = 0.0
    replans = 0
    planning_wall_time = 0.0
    failure_reason = ""
    last_planned_final = np.full_like(current_state, np.nan, dtype=float)

    def fresh_plan(start: np.ndarray, budget: float):
        nonlocal planning_wall_time, replans, last_planned_final
        planner = build_planner(system, config, planner_name, start, float(budget))
        plan_start = time.time()
        solutions, _ = planner.plan()
        planning_wall_time += time.time() - plan_start
        if not solutions:
            return None
        replans += 1
        solution = solutions[0]
        if solution.get("states"):
            last_planned_final = _state_to_numpy(solution["states"][-1], state_dim)
        return solution

    current_solution = fresh_plan(current_state, float(config["planning_time"]))
    if current_solution is None:
        failure_reason = "initial_plan_failed"

    current_plan_idx = 0
    step = 0
    while current_solution is not None and step < int(config["max_steps"]):
        controls = current_solution.get("controls") or []
        states = current_solution.get("states") or []
        times = current_solution.get("time") or []
        if not controls or current_plan_idx >= len(controls) or len(states) < current_plan_idx + 2:
            current_solution = fresh_plan(current_state, float(config["replanning_time_budget"]))
            current_plan_idx = 0
            if current_solution is None:
                failure_reason = "replan_after_plan_exhausted_failed"
                break
            continue

        control = np.asarray(controls[current_plan_idx], dtype=float)
        duration = float(times[current_plan_idx]) if current_plan_idx < len(times) else float(config["propagation_step_size"])
        next_planned = _state_to_numpy(states[current_plan_idx + 1], state_dim)

        executed = simulator.execute_segment(control, duration)
        current_state = _state_to_numpy(executed, state_dim)
        trajectory.append(current_state.copy())
        executed_controls += 1
        executed_duration += float(duration)

        step_error = float(arrayDistance(current_state, next_planned, system=config["system_name"]))
        tracking_errors.append(step_error)
        goal_distance = float(arrayDistance(current_state, goal_state, system=config["system_name"]))
        if goal_distance <= goal_threshold:
            break

        if step_error > float(config["replanningMaxDistance"]):
            current_solution = fresh_plan(current_state, float(config["replanning_time_budget"]))
            current_plan_idx = 0
            if current_solution is None:
                failure_reason = f"replan_failed_after_tracking_error_{step_error:.6f}"
                break
        elif current_plan_idx >= len(controls) - 1:
            current_solution = fresh_plan(current_state, float(config["replanning_time_budget"]))
            current_plan_idx = 0
            if current_solution is None:
                failure_reason = "replan_after_plan_end_failed"
                break
        else:
            current_plan_idx += 1
        step += 1

    # For the replanning baseline, Gaussian simulator execution is effectively
    # instantaneous in Python, but the experiment should report closed-loop
    # elapsed time: all planning time plus the physical duration of executed
    # controls.
    wall_time = float(planning_wall_time + executed_duration)
    goal_distance = float(arrayDistance(current_state, goal_state, system=config["system_name"]))
    if not failure_reason and goal_distance > goal_threshold:
        if step >= int(config["max_steps"]):
            failure_reason = f"max_steps_{int(config['max_steps'])}_reached"
        else:
            failure_reason = f"goal_not_reached_{goal_distance:.6f}"
    status = "success" if goal_distance <= goal_threshold and not failure_reason else "failure"

    return MethodResult(
        method="replanning",
        planner=planner_name,
        run_number=run_number,
        status=status,
        failure_reason=failure_reason,
        wall_time=wall_time,
        planning_wall_time=planning_wall_time,
        execution_wall_time=float(executed_duration),
        replans=max(0, replans - 1),
        num_controls=executed_controls,
        cost=_trajectory_cost(trajectory, config["system_name"]),
        tracking_error_mean=float(np.mean(tracking_errors)) if tracking_errors else float("nan"),
        goal_distance=goal_distance,
        final_state=np.asarray(current_state, dtype=float),
        planned_final_state=np.asarray(last_planned_final, dtype=float),
    )


def _result_to_row(result: MethodResult, state_dim: int) -> dict:
    row = {
        "run_number": result.run_number,
        "planner": result.planner,
        "method": result.method,
        "status": result.status,
        "failure_reason": result.failure_reason,
        "wall_time": result.wall_time,
        "planning_wall_time": result.planning_wall_time,
        "execution_wall_time": result.execution_wall_time,
        "replans": result.replans,
        "num_controls": result.num_controls,
        "cost": result.cost,
        "tracking_error_mean": result.tracking_error_mean,
        "goal_distance": result.goal_distance,
    }
    final = np.asarray(result.final_state, dtype=float).reshape(-1)
    planned = np.asarray(result.planned_final_state, dtype=float).reshape(-1)
    for i in range(state_dim):
        row[f"final_{i}"] = float(final[i]) if i < len(final) else float("nan")
        row[f"planned_final_{i}"] = float(planned[i]) if i < len(planned) else float("nan")
    return row


def csv_fieldnames(state_dim: int) -> list[str]:
    fields = [
        "run_number",
        "planner",
        "method",
        "status",
        "failure_reason",
        "wall_time",
        "planning_wall_time",
        "execution_wall_time",
        "replans",
        "num_controls",
        "cost",
        "tracking_error_mean",
        "goal_distance",
    ]
    fields.extend(f"final_{i}" for i in range(state_dim))
    fields.extend(f"planned_final_{i}" for i in range(state_dim))
    return fields


def load_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(path: str, rows: list[dict], state_dim: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = csv_fieldnames(state_dim)
    rows = sorted(rows, key=lambda r: (int(r.get("run_number", 0)), str(r.get("planner", "")), str(r.get("method", ""))))
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    os.replace(tmp, path)


def upsert_row(rows: list[dict], new_row: dict) -> list[dict]:
    out = [
        row
        for row in rows
        if not (
            int(row.get("run_number", -1)) == int(new_row["run_number"])
            and str(row.get("planner")) == str(new_row["planner"])
            and str(row.get("method")) == str(new_row["method"])
        )
    ]
    out.append(new_row)
    return out


def row_exists(rows: list[dict], run_number: int, planner_name: str, method: str) -> bool:
    for row in rows:
        if (
            int(row.get("run_number", -1)) == int(run_number)
            and str(row.get("planner")) == str(planner_name)
            and str(row.get("method")) == str(method)
        ):
            return True
    return False


def methods_from_arg(method: str) -> list[str]:
    method = method.lower()
    if method == "both":
        return ["aura", "replanning"]
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    return [method]


def output_csv_path(results_dir: str, system_name: str, planner_name: str, run_number: int) -> str:
    return os.path.join(results_dir, f"{system_name}_{planner_name}_{int(run_number):02d}.csv")


def run_method(method: str, planner_name: str, run_number: int, config: dict, seed: int | None) -> MethodResult:
    if method == "aura":
        return run_aura_once(planner_name, run_number, config, seed)
    if method == "replanning":
        return run_replanning_once(planner_name, run_number, config, seed)
    raise ValueError(f"unsupported method: {method}")


def print_result(result: MethodResult) -> None:
    print(
        f"[{result.planner}][run {result.run_number:02d}][{result.method}] "
        f"status={result.status} wall={result.wall_time:.3f}s "
        f"planning={result.planning_wall_time:.3f}s controls={result.num_controls} "
        f"goal_distance={result.goal_distance:.6f}"
    )
    print(f"  final={_fmt_state(result.final_state)}")
    print(f"  planned_final={_fmt_state(result.planned_final_state)}")
    if result.failure_reason:
        print(f"  reason={result.failure_reason}")


def configure_ompl_logging() -> None:
    try:
        ou.setLogLevel(ou.LOG_WARN)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run double-integrator wall-time performance experiments for AURA "
            "and fresh replanning."
        )
    )
    parser.add_argument("legacy_planner_name", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_system_name", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("legacy_run_number", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument(
        "--config",
        default="configs/performance_double_integrator.yaml",
        help="YAML config for start/goal/timing/output defaults.",
    )
    parser.add_argument(
        "--planner-name",
        choices=PLANNERS,
        default=None,
        help="Planner to run.",
    )
    parser.add_argument(
        "--method",
        choices=("aura", "replanning", "both"),
        default="both",
        help="Method to run. Default: both.",
    )
    parser.add_argument("--run-number", type=int, default=None, help="Run number to write.")
    parser.add_argument("--num-runs", type=int, default=None, help="Run numbers 1..N.")
    parser.add_argument("--results-dir", default=None, help="Output directory.")
    parser.add_argument(
        "--simulator-mode",
        choices=("gaussian", "mujoco"),
        default=None,
        help="Execution simulator backend.",
    )
    parser.add_argument("--planning-time", type=float, default=None, help="Initial plan budget.")
    parser.add_argument("--replanning-time", type=float, default=None, help="Fresh replanning budget.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum executed controls.")
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip method rows that already exist in their output CSV.",
    )
    parser.add_argument(
        "--verbose-internal",
        action="store_true",
        help="Show verbose planner/AURA step logs while timing methods.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print requested runs only.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.legacy_planner_name and args.planner_name is None:
        args.planner_name = args.legacy_planner_name.lower()
    if args.legacy_system_name:
        config["system_name"] = args.legacy_system_name.lower()
    if args.legacy_run_number and args.run_number is None:
        args.run_number = int(args.legacy_run_number)

    planner_name = (args.planner_name or str(config.get("planner_name", "aorrt"))).lower()
    if planner_name not in PLANNERS:
        raise ValueError(f"planner must be one of {PLANNERS}, got {planner_name!r}")
    if config["system_name"] != "double_integrator":
        raise RuntimeError("wall_time_experiment.py currently targets double_integrator.")
    if args.simulator_mode is not None:
        config["simulator_mode"] = args.simulator_mode
    if config["simulator_mode"] != "gaussian":
        raise RuntimeError("double_integrator currently supports the gaussian simulator only.")
    if args.results_dir is not None:
        config["results_dir"] = args.results_dir
    if args.planning_time is not None:
        config["planning_time"] = float(args.planning_time)
    if args.replanning_time is not None:
        config["replanning_time_budget"] = float(args.replanning_time)
    if args.max_steps is not None:
        config["max_steps"] = int(args.max_steps)

    if args.num_runs is not None and args.run_number is not None:
        raise ValueError("Use either --run-number or --num-runs, not both.")
    if args.num_runs is not None:
        run_numbers = list(range(1, int(args.num_runs) + 1))
    elif args.run_number is not None:
        run_numbers = [int(args.run_number)]
    else:
        run_numbers = [1]

    requested_methods = methods_from_arg(args.method)
    state_dim = int(len(config["start_state"]))
    configure_ompl_logging()

    print("Performance experiment")
    print(f"  system:       {config['system_name']}")
    print(f"  simulator:    {config['simulator_mode']}")
    print(f"  planner:      {planner_name}")
    print(f"  methods:      {' '.join(requested_methods)}")
    print(f"  runs:         {' '.join(str(r) for r in run_numbers)}")
    print(f"  start:        {_fmt_state(config['start_state'])}")
    print(f"  goal:         {_fmt_state(config['goal_state'])}")
    print(f"  results dir:  {config['results_dir']}")

    if args.dry_run:
        return

    failures = 0
    for run_number in run_numbers:
        csv_path = output_csv_path(
            str(config["results_dir"]),
            str(config["system_name"]),
            planner_name,
            int(run_number),
        )
        rows = load_rows(csv_path)
        for method_index, method in enumerate(requested_methods):
            if args.skip_existing and row_exists(rows, run_number, planner_name, method):
                print(f"[skip] {planner_name} run={run_number:02d} method={method} -> {csv_path}")
                continue
            seed = int(args.seed) + int(run_number) * 100 + method_index
            try:
                with _quiet_output(not bool(args.verbose_internal)):
                    result = run_method(method, planner_name, int(run_number), config, seed)
            except Exception as exc:
                failures += 1
                traceback.print_exc()
                final = np.asarray(config["start_state"], dtype=float)
                result = MethodResult(
                    method=method,
                    planner=planner_name,
                    run_number=int(run_number),
                    status="failure",
                    failure_reason=f"{type(exc).__name__}: {exc}",
                    wall_time=float("nan"),
                    planning_wall_time=float("nan"),
                    execution_wall_time=float("nan"),
                    replans=0,
                    num_controls=0,
                    cost=float("nan"),
                    tracking_error_mean=float("nan"),
                    goal_distance=float("nan"),
                    final_state=final,
                    planned_final_state=np.full_like(final, np.nan, dtype=float),
                )
            print_result(result)
            if result.status != "success":
                failures += 1
            rows = upsert_row(rows, _result_to_row(result, state_dim))
            save_rows(csv_path, rows, state_dim)
            print(f"[save] {csv_path}")

    if failures:
        print(f"[warning] Completed with {failures} failed method run(s).")


if __name__ == "__main__":
    main()
