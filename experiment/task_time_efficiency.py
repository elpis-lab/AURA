#!/usr/bin/env python3
"""Compare end-to-end task time for AURA, replanning, MPPI, and RandUp."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml
from ompl import base as ob
from ompl import control as oc
from ompl import util as ou

from aura.AURA import AURA
from methods.MPPI import MPPIController, parameters_from_config
from methods.Replanning import ReplanningRunner
from methods.plan import OMPLPlanner
from methods.RandUpRRT import RandUpRRTConfig
from simulation.pushing_model import get_pushing_model
from simulation.simulator import create_simulator
from systems import get_system
from train_model import load_opt_model_2
from utils.control_duration import (
    duration_seconds_to_steps,
    task_execution_step_seconds,
)
from utils.childrenHandler import getOutgoingEdgeIndices
from utils.experiment_io import (
    METHODS,
    PANEL_LAYOUT,
    PLANNERS,
    RESULT_SCHEMA_VERSION,
    SIMULATION_PANEL_ORDER,
    WALL_TIME_DEFINITIONS,
    data_hash,
    ensure_json,
    finalize_result,
    import_successful_results,
    job_complete,
    json_value,
    legacy_time_correction,
    mppi_result_complete,
    result_path,
    rewrite_randup_trials,
    serialize_result,
    stable_seed,
    task_time_seconds,
    upsert_result,
    upsert_serialized_result,
    validate_result,
    write_json,
    write_mppi_result,
    write_paired_results,
    write_randup_result,
    write_randup_summary,
)
from utils.utils import (
    arrayDistance,
    is_state_array_valid,
    normalize_obstacle_config,
)


RUNTIME_SOURCE_PATHS = (
    REPO_ROOT / "aura/AURA.py",
    REPO_ROOT / "methods" / "Replanning.py",
    REPO_ROOT / "aura/optimization.py",
    REPO_ROOT / "methods" / "plan.py",
    REPO_ROOT / "systems.py",
    REPO_ROOT / "utils/auraHandler.py",
    REPO_ROOT / "utils/childrenHandler.py",
    REPO_ROOT / "utils/control_duration.py",
    REPO_ROOT / "utils/utils.py",
    REPO_ROOT / "simulation/simulator.py",
    REPO_ROOT / "simulation/mujoco_car.py",
    REPO_ROOT / "simulation/mujoco_pushing.py",
    REPO_ROOT / "simulation/inverse_kinematics.py",
    REPO_ROOT / "simulation/pushing_model.py",
    REPO_ROOT / "experiment/task_time_efficiency.py",
    REPO_ROOT / "scripts/plot_task_time.py",
    REPO_ROOT / "requirements.txt",
)

OMPL_SOURCE_PATHS = (
    REPO_ROOT / "planners" / "aorrt" / "aorrt.cpp",
    REPO_ROOT / "planners" / "aorrt" / "aorrt.h",
    REPO_ROOT / "planners" / "aoest" / "AOEST.cpp",
    REPO_ROOT / "planners" / "aoest" / "AOEST.h",
    REPO_ROOT / "planners" / "sststar" / "SSTStar.cpp",
    REPO_ROOT / "planners" / "sststar" / "SSTStar.h",
)


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _resolve_repo_path(path_value: str, *, relative_to: Path = REPO_ROOT) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _source_inventory() -> dict[str, dict[str, Any]]:
    inventory = {}
    for path in (*RUNTIME_SOURCE_PATHS, *OMPL_SOURCE_PATHS):
        key = (
            str(path.relative_to(REPO_ROOT))
            if path.is_relative_to(REPO_ROOT)
            else str(path)
        )
        inventory[key] = {
            "exists": path.is_file(),
            "sha256": (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file()
                else None
            ),
        }
    return inventory


def git_output(*arguments: str) -> str:
    """Return Git metadata without making campaign startup depend on Git."""

    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def validate_environment(
    device: str,
    runtime_sources: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Validate only dependencies required to start the task-time campaign."""

    errors = []
    missing_sources = [
        path for path, metadata in runtime_sources.items() if not metadata["exists"]
    ]
    if missing_sources:
        errors.append("Missing runtime sources: " + ", ".join(missing_sources))

    planner_status = {
        planner: hasattr(oc, planner) for planner in ("SSTStar", "AORRT", "AOEST")
    }
    missing_planners = [
        planner for planner, available in planner_status.items() if not available
    ]
    if missing_planners:
        errors.append(
            "OMPL is missing custom planners: " + ", ".join(missing_planners)
        )

    torch_report: dict[str, Any]
    try:
        torch_device = torch.device(device)
        probe = torch.ones(1, device=torch_device)
        torch_report = {
            "version": torch.__version__,
            "device": str(torch_device),
            "cuda_available": torch.cuda.is_available(),
            "probe": float(probe.cpu().item()),
        }
    except Exception as exc:
        torch_report = {"device": device, "error": repr(exc)}
        errors.append(f"Torch device {device!r} is unavailable: {exc}")

    return (
        {
            "python": {"executable": sys.executable, "version": sys.version},
            "torch": torch_report,
            "ompl": {"custom_planners": planner_status},
            "git": {
                "sha": git_output("rev-parse", "HEAD"),
                "status_porcelain": git_output("status", "--porcelain"),
            },
        },
        errors,
    )


def freeze_manifest(path: Path, *, device: str, allow_incomplete: bool) -> dict:
    source = _load_yaml(path)
    panels = []
    for item in source.get("panels", []):
        config_path = _resolve_repo_path(str(item["config"]))
        config = _load_yaml(config_path)
        if int(config["min_control_duration"]) >= int(config["max_control_duration"]):
            raise ValueError(
                f"production panel {config.get('panel_id')} does not have min < max"
            )
        panels.append(
            {
                "config_path": str(config_path),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "config_hash": data_hash(config),
                "config": config,
                "execute": bool(item.get("execute", True)),
            }
        )
    runtime_sources = _source_inventory()
    preflight, errors = validate_environment(device, runtime_sources)
    runtime_source_hash = data_hash(runtime_sources)
    frozen = {
        **source,
        "source_manifest": str(path.resolve()),
        "source_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "device": device,
        "panels": panels,
        "preflight": preflight,
        "preflight_errors": errors,
        "git_sha": preflight["git"]["sha"],
        "git_status_porcelain": preflight["git"]["status_porcelain"],
        "runtime_sources": runtime_sources,
        "runtime_source_hash": runtime_source_hash,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "wall_time_definitions": dict(WALL_TIME_DEFINITIONS),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    frozen["configuration_hash"] = data_hash(
        {
            "schema_version": frozen["schema_version"],
            "result_schema_version": frozen["result_schema_version"],
            "wall_time_definitions": frozen["wall_time_definitions"],
            "device": device,
            "runtime_source_hash": runtime_source_hash,
            "methods": frozen["methods"],
            "planners": frozen["planners"],
            "panels": [
                {
                    "config_hash": panel["config_hash"],
                    "execute": panel["execute"],
                }
                for panel in panels
            ],
        }
    )
    frozen["manifest_id"] = (
        "fig7-vardur-"
        + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + frozen["configuration_hash"][:10]
    )
    if errors and not allow_incomplete:
        raise RuntimeError(
            "dependency preflight failed:\n- " + "\n- ".join(errors)
        )
    return frozen


def _apply_bounds(system, config: dict) -> None:
    state_bounds = config.get("state_bounds")
    if state_bounds:
        system.set_state_bounds(state_bounds)
    control_bounds = config.get("control_bounds")
    if control_bounds:
        system.set_control_bounds(control_bounds)


def _build_planner(
    system,
    config: dict,
    planner_name: str,
    *,
    start_state: np.ndarray | None = None,
    optimizer_device: str | None = None,
) -> OMPLPlanner:
    planner = OMPLPlanner(
        system=system,
        start_state=np.asarray(
            config["start_state"] if start_state is None else start_state, dtype=float
        ),
        goal_state=np.asarray(config["goal_state"], dtype=float),
        planner_method=planner_name,
        goal_threshold=float(config["goal_threshold"]),
        min_max_control_duration=(
            int(config["min_control_duration"]),
            int(config["max_control_duration"]),
        ),
        propagation_step_size=float(config["propagation_step_size"]),
        initial_planning_time=float(config["planning_time"]),
        pruning_radius=float(config.get("pruning_radius", 0.1)),
        goal_bias=float(config.get("goal_bias", 0.05)),
        obstacle_config=config.get("obstacles"),
        optimization_objective=str(
            config.get("optimization_objective", "control_duration")
        ),
    )
    planner.replanning_time = float(config["replanning_time_budget"])
    # Keep each individual solve bounded. The experiment runner explicitly
    # retries bounded attempts until it finds an exact solution or exhausts the
    # run's task-time budget.
    planner.single_solve = True
    planner.motion_validation_step_size = float(
        config.get("motion_validation_step_size", config["propagation_step_size"])
    )
    planner.optimizer_num_states = int(config.get("optimizer_num_states", 2000))
    planner.optimizer_num_steps = int(config.get("optimizer_num_steps", 300))
    planner.optimizer_learning_rate = float(
        config.get("optimizer_learning_rate", 0.01)
    )
    planner.optimizer_pos_std = float(
        config.get("optimizer_position_std", 0.003)
    )
    planner.optimizer_rot_std = float(
        config.get("optimizer_rotation_std", 0.05)
    )
    planner.optimizer_vel_std = float(
        config.get("optimizer_velocity_std", planner.optimizer_pos_std)
    )
    planner.optimizer_max_children = int(config.get("optimizer_max_children", 0))
    planner.optimizer_device = optimizer_device
    planner.recovery_replanning_time = float(
        config.get("recovery_replanning_time", config["replanning_time_budget"])
    )
    planner.solution_continuity_max_distance = float(
        config.get(
            "solution_continuity_max_distance",
            config.get("replanning_max_distance", 0.1),
        )
    )
    if system.name == "pushing_object":
        pushing_model = get_pushing_model(
            system.object_shape,
            model_name=str(config.get("model_name", system.model_name)),
            model_path=config.get("model_path"),
        )
        planner.opt_model = load_opt_model_2(
            pushing_model,
            lr=planner.optimizer_learning_rate,
            epochs=planner.optimizer_num_steps,
        )
    else:
        planner.opt_model = None
    return planner


def _initial_plan_payload(solution: dict) -> dict:
    return {
        "states": json_value(solution.get("states", [])),
        "controls": json_value(solution.get("controls", [])),
        "duration_seconds": json_value(solution.get("time", [])),
        "duration_steps": json_value(solution.get("time_steps", [])),
        "cost": json_value(float(solution["cost"])),
        "control_count": int(solution["control_count"]),
    }


def _duration_audit(planner: OMPLPlanner) -> dict:
    planner_data = oc.PlannerData(planner.setup.getSpaceInformation())
    planner.setup.getPlanner().getPlannerData(planner_data)
    si = planner.setup.getSpaceInformation()
    h = float(si.getPropagationStepSize())
    counts: dict[str, int] = {}
    malformed = []
    edge_count = 0
    for source in range(int(planner_data.numVertices())):
        try:
            targets = getOutgoingEdgeIndices(planner_data, source)
        except Exception as exc:
            targets = []
            malformed.append({"source": source, "error": repr(exc)})
        for target in targets:
            edge_count += 1
            try:
                seconds = float(planner_data.getEdge(source, target).getDuration())
                steps = duration_seconds_to_steps(
                    seconds,
                    h,
                    min_steps=planner.min_max_control_duration[0],
                    max_steps=planner.min_max_control_duration[1],
                )
                counts[str(steps)] = counts.get(str(steps), 0) + 1
            except Exception as exc:
                malformed.append(
                    {"source": source, "target": target, "error": repr(exc)}
                )
    return {
        "edge_count": edge_count,
        "duration_step_histogram": counts,
        "distinct_duration_steps": sorted(int(value) for value in counts),
        "malformed_edges": malformed,
        "range_steps": list(planner.min_max_control_duration),
        "propagation_step_size_seconds": h,
    }


def _base_result(
    *,
    panel: dict,
    planner_name: str,
    method: str,
    run_number: int,
    seed: int,
    streams: dict,
    method_order: list[str],
    initial_planning_seconds: float,
    initial_payload: dict | None,
    config_hash: str,
) -> dict:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "panel_id": panel["config"]["panel_id"],
        "system": panel["config"]["system_name"],
        "environment": panel["config"]["simulator_mode"],
        "planner": planner_name,
        "method": method,
        "run_number": run_number,
        "seed": seed,
        "rng_streams": streams,
        "method_order": method_order,
        "config_hash": config_hash,
        "initial_planning_seconds": float(initial_planning_seconds),
        "initial_plan_hash": data_hash(initial_payload) if initial_payload else None,
        "initial_plan": initial_payload,
        "status": "failure",
        "failure_reason": "",
        "nominal_execution_seconds": 0.0,
        "actual_execution_seconds": 0.0,
        "online_replanning_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "blocking_replanning_seconds": 0.0,
        "compute_overrun_seconds": 0.0,
        "wall_time_definition": WALL_TIME_DEFINITIONS[method],
        "task_time_seconds": float(initial_planning_seconds),
        "raw_process_wall_seconds": 0.0,
        "setup_reset_operator_pause_seconds": 0.0,
        "num_controls": 0,
        "num_replanning": 0,
        "cost": None,
        "tracking_error_mean": None,
        "tracking_error_list": [],
        "goal_distance": None,
        "final_state": None,
        "planned_final_state": None,
        "controls": [],
        "control_duration_steps": [],
        "control_duration_seconds": [],
        "primitive_states": [],
    }


def plan_until_solution(
    planner: OMPLPlanner,
    *,
    attempt_budget_seconds: float,
    total_budget_seconds: float,
) -> tuple[dict | None, float, int, str]:
    """Continue one OMPL tree until it yields a usable exact solution.

    A failed bounded solve is not a task failure. Another bounded OMPL solve is
    attempted until a solution is available or the caller's remaining
    task-time budget is exhausted. The custom planners cannot ``resolve`` an
    empty solution set, so retry attempts clear and restart that search.
    """
    attempt_budget = max(1e-3, float(attempt_budget_seconds))
    total_budget = max(0.0, float(total_budget_seconds))
    started = time.monotonic()
    attempts = 0
    errors: list[str] = []
    first_attempt = True
    consecutive_immediate_returns = 0
    while time.monotonic() - started < total_budget:
        remaining = total_budget - (time.monotonic() - started)
        budget = min(attempt_budget, max(1e-3, remaining))
        attempts += 1
        attempt_started = time.monotonic()
        try:
            if first_attempt:
                old_initial_time = float(planner.initial_planning_time)
                planner.initial_planning_time = float(budget)
                try:
                    solutions, _ = planner.plan()
                finally:
                    planner.initial_planning_time = old_initial_time
                first_attempt = False
            else:
                solutions, _ = planner.retry_initial_plan(
                    time_budget=float(budget)
                )
        except Exception as exc:
            errors.append(repr(exc))
            first_attempt = False
            solutions = []
        attempt_elapsed = time.monotonic() - attempt_started
        solution = next(
            (
                candidate
                for candidate in (solutions or [])
                if candidate.get("controls")
                and len(candidate.get("states", [])) >= 2
            ),
            None,
        )
        if solution is not None:
            return (
                solution,
                min(
                    float(total_budget),
                    float(time.monotonic() - started),
                ),
                attempts,
                "",
            )
        # INVALID_START/INVALID_GOAL and equivalent planner aborts return
        # immediately rather than consuming the requested termination budget.
        # Retrying the unchanged problem millions of times cannot change that
        # state. Allow a few fresh retries (which also accommodates cheap test
        # doubles and transient setup behavior), then classify the episode as
        # an exhausted planning attempt and let the caller apply the 300 s
        # timeout penalty.
        immediate_threshold = min(
            0.05,
            max(0.005, 0.05 * float(budget)),
        )
        if attempt_elapsed < immediate_threshold:
            consecutive_immediate_returns += 1
        else:
            consecutive_immediate_returns = 0
        if consecutive_immediate_returns >= 3:
            errors.append(
                "planner_returned_immediately_without_exact_solution"
            )
            break
    return (
        None,
        min(
            float(total_budget),
            float(time.monotonic() - started),
        ),
        attempts,
        errors[-1] if errors else "no_exact_solution_before_task_time_limit",
    )


def run_aura(
    planner: OMPLPlanner,
    config: dict,
    streams: dict,
    planner_name: str,
    initial_planning_seconds: float,
) -> tuple[dict, float]:
    config = deepcopy(config)
    config["disturbance_seed"] = int(streams["execution"])
    setup_started = time.monotonic()
    simulator = create_simulator(
        config["system_name"], config["simulator_mode"], config=config
    )
    simulator.reset()
    simulator.set_state(config["start_state"])
    setup_seconds = time.monotonic() - setup_started
    np.random.seed(int(streams["optimization"]))
    torch.manual_seed(int(streams["optimization"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(streams["optimization"]))
    task_limit = float(config.get("task_time_limit_seconds", 300.0))
    blocking_restart_seconds = 0.0
    restart_attempts = 0
    restart_count = 0
    leg_results = []
    current_planner = planner
    raw_started = time.monotonic()
    final_failure_reason = ""
    final_status = "failure"

    while True:
        nominal_so_far = sum(
            float(leg.nominal_execution_seconds) for leg in leg_results
        )
        remaining_execution_budget = (
            task_limit
            - float(initial_planning_seconds)
            - float(blocking_restart_seconds)
            - nominal_so_far
        )
        if remaining_execution_budget < float(config["propagation_step_size"]) - 1e-9:
            final_status = "timeout"
            final_failure_reason = "task_time_limit_reached"
            break

        result = AURA(current_planner.system, current_planner, simulator).run(
            reset_sim=False,
            pause_each_step=False,
            max_steps=None,
            max_nominal_execution_seconds=max(
                0.0, float(remaining_execution_budget)
            ),
        )
        leg_results.append(result)
        goal_distance = float(
            arrayDistance(
                result.final_state,
                config["goal_state"],
                system=config["system_name"],
            )
        )
        if (
            result.status == "success"
            and goal_distance <= float(config["goal_threshold"])
        ):
            final_status = "success"
            final_failure_reason = ""
            break

        final_failure_reason = (
            result.failure_reason
            or f"goal_not_reached:{goal_distance:.9g}"
        )
        if final_failure_reason.startswith(
            ("execution_failed:", "final_execution_failed:")
        ):
            final_status = "failure"
            break
        if final_failure_reason == "task_time_limit_reached":
            final_status = "timeout"
            break

        nominal_so_far = sum(
            float(leg.nominal_execution_seconds) for leg in leg_results
        )
        remaining_planning_budget = (
            task_limit
            - float(initial_planning_seconds)
            - float(blocking_restart_seconds)
            - nominal_so_far
        )
        if remaining_planning_budget <= 1e-9:
            final_status = "timeout"
            final_failure_reason = "task_time_limit_reached"
            break

        measured_state = np.asarray(result.final_state, dtype=float)
        current_planner = _build_planner(
            planner.system,
            config,
            planner_name,
            start_state=measured_state,
            optimizer_device=planner.optimizer_device,
        )
        restart_count += 1
        solution, elapsed, attempts, planning_error = plan_until_solution(
            current_planner,
            attempt_budget_seconds=float(config["planning_time"]),
            total_budget_seconds=float(remaining_planning_budget),
        )
        blocking_restart_seconds += float(elapsed)
        restart_attempts += int(attempts)
        if solution is None:
            final_status = "timeout"
            final_failure_reason = (
                "task_time_limit_reached_during_aura_restart:"
                + str(planning_error)
            )
            break
        current_planner.solutions = [solution]

    raw_wall = time.monotonic() - raw_started
    if not leg_results:
        final_state = np.asarray(simulator.get_state(), dtype=float)
        final_planned_state = None
        controls = []
        duration_steps = []
        duration_seconds = []
        tracking_errors = []
        dense_states = [final_state.copy()]
        total_cost = 0.0
        nominal_execution = 0.0
        actual_execution = 0.0
        online_replanning = 0.0
        optimizer_seconds = 0.0
        compute_overrun = 0.0
        num_replanning = restart_count
    else:
        last_result = leg_results[-1]
        final_state = np.asarray(last_result.final_state, dtype=float)
        final_planned_state = last_result.final_planned_state
        controls = [
            np.asarray(control, dtype=float)
            for leg in leg_results
            for control in leg.controls_trajectory
        ]
        duration_steps = [
            int(value)
            for leg in leg_results
            for value in leg.control_duration_steps_trajectory
        ]
        duration_seconds = [
            float(value)
            for leg in leg_results
            for value in leg.control_duration_seconds_trajectory
        ]
        tracking_errors = [
            float(value)
            for leg in leg_results
            for value in leg.tracking_error_list
        ]
        dense_states = []
        for leg in leg_results:
            states = [
                np.asarray(state, dtype=float)
                for state in leg.dense_states_trajectory
            ]
            dense_states.extend(states if not dense_states else states[1:])
        total_cost = sum(float(leg.cost) for leg in leg_results)
        nominal_execution = sum(
            float(leg.nominal_execution_seconds) for leg in leg_results
        )
        actual_execution = sum(
            float(leg.actual_execution_seconds) for leg in leg_results
        )
        online_replanning = sum(
            float(leg.online_replanning_seconds) for leg in leg_results
        )
        optimizer_seconds = sum(
            float(leg.optimizer_seconds) for leg in leg_results
        )
        compute_overrun = sum(
            float(leg.compute_overrun_seconds) for leg in leg_results
        )
        num_replanning = (
            restart_count
            + sum(int(leg.num_replanning) for leg in leg_results)
        )

    goal_distance = float(
        arrayDistance(
            final_state,
            config["goal_state"],
            system=config["system_name"],
        )
    )
    success = final_status == "success" and goal_distance <= float(
        config["goal_threshold"]
    )
    status = "success" if success else final_status
    payload = {
        "status": status,
        "failure_reason": (
            ""
            if success
            else final_failure_reason
            or f"goal_not_reached:{goal_distance:.9g}"
        ),
        "nominal_execution_seconds": float(nominal_execution),
        "actual_execution_seconds": float(actual_execution),
        "online_replanning_seconds": float(online_replanning),
        "optimizer_seconds": float(optimizer_seconds),
        "blocking_replanning_seconds": float(blocking_restart_seconds),
        "compute_overrun_seconds": float(compute_overrun),
        "wall_time_definition": WALL_TIME_DEFINITIONS["aura"],
        "task_time_seconds": task_time_seconds(
            "aura",
            initial_planning_seconds=0.0,
            nominal_execution_seconds=float(nominal_execution),
            blocking_replanning_seconds=float(blocking_restart_seconds),
        ),
        "raw_process_wall_seconds": float(raw_wall),
        "setup_reset_operator_pause_seconds": float(setup_seconds),
        "num_controls": len(controls),
        "num_replanning": int(num_replanning),
        "cost": float(total_cost),
        "tracking_error_mean": (
            float(np.mean(tracking_errors)) if tracking_errors else 0.0
        ),
        "tracking_error_list": json_value(tracking_errors),
        "goal_distance": goal_distance,
        "final_state": json_value(final_state),
        "planned_final_state": json_value(final_planned_state),
        "controls": json_value(controls),
        "control_duration_steps": json_value(duration_steps),
        "control_duration_seconds": json_value(duration_seconds),
        "primitive_states": json_value(dense_states),
        "cycle_timing": json_value(
            [
                row
                for leg in leg_results
                for row in leg.cycle_timing
            ]
        ),
        "aura_restart_count": int(restart_count),
        "aura_restart_planning_attempts": int(restart_attempts),
        "disturbance_steps_consumed": int(
            getattr(simulator, "_disturbance_index", 0)
        ),
    }
    if hasattr(simulator, "close"):
        simulator.close()
    return payload, setup_seconds


def _run_restart_replanning(
    config: dict,
    planner_name: str,
    initial_solution: dict,
    streams: dict,
    initial_planning_seconds: float,
) -> tuple[dict, float]:
    config = deepcopy(config)
    config["disturbance_seed"] = int(streams["execution"])
    np.random.seed(int(streams["optimization"]))
    setup_started = time.monotonic()
    runner = ReplanningRunner(
        config["system_name"],
        planner_name,
        config,
        config["simulator_mode"],
        max_steps=None,
        initial_solution=deepcopy(initial_solution),
        task_time_budget_seconds=max(
            0.0,
            float(config.get("task_time_limit_seconds", 300.0))
            - float(initial_planning_seconds),
        ),
    )
    runner.simulator.reset()
    runner.simulator.set_state(config["start_state"])
    setup_seconds = time.monotonic() - setup_started
    raw_started = time.monotonic()
    result = runner.run(reset_sim=False)
    raw_wall = time.monotonic() - raw_started
    goal_distance = float(
        arrayDistance(
            result.final_state,
            config["goal_state"],
            system=config["system_name"],
        )
    )
    success = (
        result.status == "success"
        and goal_distance <= float(config["goal_threshold"])
    )
    timed_out = str(result.failure_reason).startswith("task_time_limit_reached")
    payload = {
        "status": "success" if success else ("timeout" if timed_out else "failure"),
        "failure_reason": (
            ""
            if success
            else result.failure_reason
            or f"goal_not_reached:{goal_distance:.9g}"
        ),
        "nominal_execution_seconds": float(result.nominal_execution_seconds),
        "actual_execution_seconds": float(result.actual_execution_seconds),
        "online_replanning_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "blocking_replanning_seconds": float(
            result.blocking_replanning_seconds
        ),
        "compute_overrun_seconds": float(
            result.compute_overrun_seconds
        ),
        "wall_time_definition": WALL_TIME_DEFINITIONS["restartReplanning"],
        "task_time_seconds": task_time_seconds(
            "restartReplanning",
            initial_planning_seconds=0.0,
            nominal_execution_seconds=float(result.nominal_execution_seconds),
            blocking_replanning_seconds=float(
                result.blocking_replanning_seconds
            ),
        ),
        "raw_process_wall_seconds": float(raw_wall),
        "setup_reset_operator_pause_seconds": float(setup_seconds),
        "num_controls": int(result.num_controls),
        "num_replanning": int(result.num_replanning),
        "cost": float(result.cost),
        "tracking_error_mean": float(result.tracking_error_mean),
        "tracking_error_list": json_value(result.tracking_error_list),
        "goal_distance": goal_distance,
        "final_state": json_value(result.final_state),
        "planned_final_state": json_value(result.planned_final_state),
        "controls": json_value(result.controls_trajectory),
        "control_duration_steps": json_value(
            result.control_duration_steps_trajectory
        ),
        "control_duration_seconds": json_value(
            result.control_duration_seconds_trajectory
        ),
        "primitive_states": json_value(result.primitive_trajectory),
        "cycle_timing": [],
        "disturbance_steps_consumed": int(
            getattr(runner.simulator, "_disturbance_index", 0)
        ),
    }
    simulator = runner.simulator
    if hasattr(simulator, "close"):
        simulator.close()
    return payload, setup_seconds


def run_paired_worker(
    frozen: dict,
    panel_id: str,
    planner_name: str,
    run_number: int,
    methods: list[str],
    results_root: Path,
) -> None:
    process_started = time.monotonic()
    panel = next(
        panel for panel in frozen["panels"] if panel["config"]["panel_id"] == panel_id
    )
    config = deepcopy(panel["config"])
    master_seed = stable_seed(
        int(frozen["base_seed"]), panel_id, planner_name, run_number
    )
    streams = {
        label: stable_seed(master_seed, label)
        for label in (
            "task",
            "initial_ompl",
            "execution",
            "optimization",
        )
    }
    # AURA must continue the exact tree created by the shared initial solve.
    # Run it before RR so baseline replans cannot advance OMPL's process-global
    # RNG stream and perturb AURA's retained-tree continuation.
    method_order = ["aura", "restartReplanning"]
    method_order = [method for method in method_order if method in methods]
    disturbance_count = (
        int(config["max_steps"]) * int(config["max_control_duration"])
    )
    disturbance_schedule = np.random.default_rng(
        int(streams["execution"])
    ).standard_normal(
        (
            disturbance_count,
            6 if str(config["system_name"]) == "dubins_airplane" else 3,
        )
    )
    config["disturbance_schedule"] = disturbance_schedule.tolist()
    disturbance_payload = {
        "seed": int(streams["execution"]),
        "shape": list(disturbance_schedule.shape),
        "standard_normal_samples": disturbance_schedule.tolist(),
    }
    disturbance_hash = data_hash(disturbance_payload)
    np.random.seed(int(streams["task"]))
    torch.manual_seed(int(streams["task"]))
    # OMPL's global seed must be installed before constructing state/control
    # spaces or planners. Calling setSeed after any RNG is instantiated has no
    # deterministic effect and emits an OMPL warning.
    ou.RNG.setSeed(int(streams["initial_ompl"]))
    system = get_system(config["system_name"])
    _apply_bounds(system, config)
    planner = _build_planner(
        system,
        config,
        planner_name,
        optimizer_device=str(frozen["device"]),
    )
    try:
        (
            initial_solution,
            initial_planning_seconds,
            initial_planning_attempts,
            planning_error,
        ) = plan_until_solution(
            planner,
            attempt_budget_seconds=float(config["planning_time"]),
            total_budget_seconds=float(
                config.get("task_time_limit_seconds", 300.0)
            ),
        )
    except Exception:
        initial_planning_seconds = 0.0
        initial_planning_attempts = 0
        initial_solution = None
        planning_error = traceback.format_exc()

    initial_solution = (
        deepcopy(initial_solution) if initial_solution is not None else None
    )
    initial_payload = (
        _initial_plan_payload(initial_solution)
        if initial_solution is not None
        else None
    )
    duration_audit = None
    if initial_solution is not None:
        duration_audit = _duration_audit(planner)
    paired_rows = []
    for method in method_order:
        base = _base_result(
            panel=panel,
            planner_name=planner_name,
            method=method,
            run_number=run_number,
            seed=master_seed,
            streams=streams,
            method_order=method_order,
            initial_planning_seconds=initial_planning_seconds,
            initial_payload=initial_payload,
            config_hash=panel["config_hash"],
        )
        base["duration_audit_initial_tree"] = duration_audit
        base["disturbance_schedule_hash"] = disturbance_hash
        base["initial_planning_attempts"] = int(initial_planning_attempts)
        if initial_solution is None:
            base["status"] = "timeout"
            base["failure_reason"] = (
                "task_time_limit_reached_during_initial_planning:"
                + planning_error[-2000:]
            )
            base["raw_process_wall_seconds"] = time.monotonic() - process_started
            base = finalize_result(
                base, expected_config_hash=panel["config_hash"]
            )
            paired_rows.append(base)
            continue
        method_started = time.monotonic()
        try:
            if method == "aura":
                payload, _ = run_aura(
                    planner,
                    config,
                    streams,
                    planner_name,
                    initial_planning_seconds,
                )
            else:
                payload, _ = _run_restart_replanning(
                    config,
                    planner_name,
                    initial_solution,
                    streams,
                    initial_planning_seconds,
                )
            base.update(payload)
            base["task_time_seconds"] += initial_planning_seconds
        except Exception as exc:
            base["failure_reason"] = f"exception:{exc!r}"
            base["traceback"] = traceback.format_exc()
        base["raw_process_wall_seconds"] = time.monotonic() - method_started
        base = finalize_result(
            base, expected_config_hash=panel["config_hash"]
        )
        paired_rows.append(base)
    output_path = result_path(
        results_root, panel_id, planner_name, run_number
    )
    for row in paired_rows:
        upsert_result(output_path, row)


def _method_values(value: str) -> list[str]:
    normalized = value.lower()
    if normalized == "both":
        return list(METHODS)
    if normalized == "replanning":
        return ["restartReplanning"]
    for method in METHODS:
        if normalized == method.lower():
            return [method]
    raise ValueError(value)


def _write_orchestrator_failure_rows(
    *,
    frozen: dict,
    results_root: Path,
    panel_id: str,
    planner_name: str,
    run_number: int,
    methods: list[str],
    reason: str,
) -> None:
    """Make a killed/crashed isolated worker an explicit, resumable outcome."""
    panel = next(
        item
        for item in frozen["panels"]
        if item["config"]["panel_id"] == panel_id
    )
    master_seed = stable_seed(
        int(frozen["base_seed"]), panel_id, planner_name, run_number
    )
    streams = {
        label: stable_seed(master_seed, label)
        for label in (
            "task",
            "initial_ompl",
            "execution",
            "optimization",
        )
    }
    method_order = ["aura", "restartReplanning"]
    method_order = [method for method in method_order if method in methods]
    path = result_path(results_root, panel_id, planner_name, run_number)
    if job_complete(
        results_root,
        panel_id,
        planner_name,
        run_number,
        methods,
        expected_config_hash=panel["config_hash"],
    ):
        return
    rows = []
    for method in method_order:
        row = _base_result(
            panel=panel,
            planner_name=planner_name,
            method=method,
            run_number=run_number,
            seed=master_seed,
            streams=streams,
            method_order=method_order,
            initial_planning_seconds=0.0,
            initial_payload=None,
            config_hash=panel["config_hash"],
        )
        if str(reason).startswith("orchestrator_timeout"):
            row["status"] = "timeout"
        row["failure_reason"] = reason
        row = finalize_result(
            row, expected_config_hash=panel["config_hash"]
        )
        rows.append(row)
    for row in rows:
        upsert_result(path, row)


def _worker_command(
    manifest_path: Path,
    results_root: Path,
    panel: str,
    planner: str,
    run_number: int,
    methods: list[str],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--frozen-manifest",
        str(manifest_path),
        "--results-root",
        str(results_root),
        "--panel",
        panel,
        "--planner",
        planner,
        "--run",
        str(run_number),
        "--method",
        "both" if set(methods) == set(METHODS) else methods[0],
    ]


def run_task_time_campaign(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="configs/fig7/manifest.yaml", help="Source manifest YAML."
    )
    parser.add_argument("--frozen-manifest")
    parser.add_argument("--config", help="Optional single panel config filter/path.")
    parser.add_argument("--system")
    parser.add_argument("--environment", choices=["gaussian", "mujoco", "real"])
    parser.add_argument("--panel")
    parser.add_argument("--planner", choices=[*PLANNERS, "all"], default="all")
    parser.add_argument("--method", default="both")
    parser.add_argument("--run", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-end", type=int)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--results-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Treat only success/timeout rows as complete and rerun failures.",
    )
    parser.add_argument(
        "--import-successes-from",
        help=(
            "Seed a new campaign root with successful <= task-limit rows from "
            "an earlier result root."
        ),
    )
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument(
        "--include-mppi",
        action="store_true",
        help="After AURA/RR, run the existing standalone MPPI baseline on the same jobs.",
    )
    parser.add_argument(
        "--include-randup",
        action="store_true",
        help="After AURA/RR, run offline RandUP-RRT through the same frozen panels.",
    )
    parser.add_argument(
        "--all-baselines",
        action="store_true",
        help="Equivalent to --include-mppi --include-randup.",
    )
    parser.add_argument("--randup-particle-counts", default="50")
    parser.add_argument("--randup-uncertainty-levels", default="1.0")
    parser.add_argument("--randup-padding-epsilon", type=float)
    parser.add_argument("--randup-max-iterations", type=int)
    parser.add_argument("--randup-planning-time", type=float)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(arguments)
    methods = _method_values(args.method)
    if args.worker:
        if not args.frozen_manifest or not args.results_root:
            raise ValueError("worker requires --frozen-manifest and --results-root")
        frozen = json.loads(Path(args.frozen_manifest).read_text(encoding="utf-8"))
        run_paired_worker(
            frozen,
            str(args.panel),
            str(args.planner),
            int(args.run),
            methods,
            Path(args.results_root).resolve(),
        )
        return

    source_manifest = _resolve_repo_path(args.manifest)
    frozen = freeze_manifest(
        source_manifest, device=args.device, allow_incomplete=args.dry_run
    )
    if args.run is not None:
        requested_runs = [int(args.run)]
    else:
        requested_start = int(args.seed_start or 1)
        requested_count = int(
            args.num_trials or frozen["num_simulation_trials"]
        )
        requested_end = int(
            args.seed_end or (requested_start + requested_count - 1)
        )
        requested_runs = list(range(requested_start, requested_end + 1))
    frozen["selected_runs"] = requested_runs
    frozen["num_selected_trials"] = len(requested_runs)
    frozen["campaign_configuration_hash"] = data_hash(
        {
            "configuration_hash": frozen["configuration_hash"],
            "selected_runs": requested_runs,
        }
    )
    frozen["manifest_id"] = (
        "fig7-vardur-"
        + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + frozen["campaign_configuration_hash"][:10]
    )
    if args.results_root:
        results_root = Path(args.results_root).expanduser().resolve()
    else:
        results_root = REPO_ROOT / "results/full_time_comparison"
    frozen_path = results_root / "manifest.json"
    if not args.dry_run:
        if frozen_path.exists():
            existing = json.loads(frozen_path.read_text(encoding="utf-8"))
            if existing.get("campaign_configuration_hash") != frozen.get(
                "campaign_configuration_hash"
            ):
                if args.resume or args.skip_existing:
                    raise RuntimeError(
                        f"cannot resume a different campaign: {frozen_path}"
                    )
                for panel in frozen["panels"]:
                    if panel.get("execute", False):
                        shutil.rmtree(
                            results_root / panel["config"]["panel_id"],
                            ignore_errors=True,
                        )
                shutil.rmtree(results_root / "summary", ignore_errors=True)
                for name in (
                    "mppi_manifest.json",
                    "randup_manifest.json",
                    "randup_trials.csv",
                    "randup_summary.csv",
                    "randup_summary.md",
                    "task_time_comparison.png",
                    "task_time_comparison.pdf",
                    "task_time_comparison.svg",
                ):
                    path = results_root / name
                    if path.is_file():
                        path.unlink()
                write_json(frozen_path, frozen)
            else:
                frozen = existing
        else:
            write_json(frozen_path, frozen)

    if args.import_successes_from:
        imported = import_successful_results(
            Path(args.import_successes_from).expanduser().resolve(),
            results_root,
            frozen,
        )
        print(f"imported {imported} successful method rows", flush=True)

    panels = [panel for panel in frozen["panels"] if panel["execute"]]
    if args.config:
        requested = _resolve_repo_path(args.config)
        panels = [
            panel for panel in panels if Path(panel["config_path"]) == requested
        ]
    if args.panel:
        panels = [
            panel
            for panel in panels
            if panel["config"]["panel_id"] == args.panel
        ]
    if args.system:
        panels = [
            panel
            for panel in panels
            if panel["config"]["system_name"] == args.system
        ]
    if args.environment:
        panels = [
            panel
            for panel in panels
            if panel["config"]["simulator_mode"] == args.environment
        ]
    planners = list(PLANNERS) if args.planner == "all" else [args.planner]
    runs = list(requested_runs)
    panel_hashes = {
        panel["config"]["panel_id"]: panel["config_hash"]
        for panel in panels
    }
    jobs = []
    for panel in panels:
        panel_id = panel["config"]["panel_id"]
        for planner_name in planners:
            for run_number in runs:
                missing_methods = list(methods)
                if args.skip_existing or args.resume:
                    accepted_statuses = (
                        {"success", "timeout"}
                        if args.retry_failures
                        else None
                    )
                    missing_methods = [
                        method
                        for method in methods
                        if not job_complete(
                            results_root,
                            panel_id,
                            planner_name,
                            run_number,
                            [method],
                            expected_config_hash=panel_hashes[panel_id],
                            accepted_statuses=accepted_statuses,
                        )
                    ]
                if missing_methods:
                    jobs.append(
                        (
                            panel_id,
                            planner_name,
                            run_number,
                            missing_methods,
                        )
                    )
    summary = {
        "manifest_id": frozen["manifest_id"],
        "results_root": str(results_root),
        "paired_jobs": len(jobs),
        "method_rows": sum(len(job[3]) for job in jobs),
        "panels": [panel["config"]["panel_id"] for panel in panels],
        "planners": planners,
        "methods": methods,
        "include_mppi": bool(args.include_mppi or args.all_baselines),
        "include_randup": bool(args.include_randup or args.all_baselines),
        "randup_particle_counts": str(args.randup_particle_counts),
        "randup_uncertainty_levels": str(args.randup_uncertainty_levels),
        "runs": runs,
        "preflight_passed": not frozen["preflight_errors"],
        "preflight_errors": frozen["preflight_errors"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return

    def launch(job):
        panel_id, planner_name, run_number, job_methods = job
        method_suffix = (
            ""
            if set(job_methods) == set(METHODS)
            else "-" + job_methods[0].lower()
        )
        log_path = (
            results_root
            / panel_id
            / "logs"
            / f"{planner_name}_trial-{run_number:03d}{method_suffix}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = _worker_command(
            frozen_path,
            results_root,
            panel_id,
            planner_name,
            run_number,
            job_methods,
        )
        with log_path.open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=float(args.timeout),
                )
            except subprocess.TimeoutExpired:
                _write_orchestrator_failure_rows(
                    frozen=frozen,
                    results_root=results_root,
                    panel_id=panel_id,
                    planner_name=planner_name,
                    run_number=run_number,
                    methods=job_methods,
                    reason=f"orchestrator_timeout_after_{float(args.timeout):.6g}s",
                )
                raise
        if completed.returncode != 0:
            _write_orchestrator_failure_rows(
                frozen=frozen,
                results_root=results_root,
                panel_id=panel_id,
                planner_name=planner_name,
                run_number=run_number,
                methods=job_methods,
                reason=f"worker_exit_code_{completed.returncode}",
            )
            raise RuntimeError(
                f"worker failed ({completed.returncode}); see {log_path}"
            )
        return job

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_parallel))) as executor:
        futures = {executor.submit(launch, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                print(f"[complete] {job}", flush=True)
            except Exception as exc:
                failures.append((job, repr(exc)))
                print(f"[failure] {job}: {exc}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} workers failed: {failures[:5]}")

    include_mppi = bool(args.include_mppi or args.all_baselines)
    include_randup = bool(args.include_randup or args.all_baselines)
    baseline_panels = [
        panel["config"]["panel_id"]
        for panel in panels
        if panel["config"].get("simulator_mode") != "real"
    ]
    for panel_id in baseline_panels:
        if include_mppi:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mppi-campaign",
                "--results-root",
                str(results_root),
                "--frozen-manifest",
                str(frozen_path),
                "--panel",
                str(panel_id),
                "--seed-start",
                str(min(runs)),
                "--seed-end",
                str(max(runs)),
                "--device",
                str(args.device),
                "--max-parallel",
                str(max(1, int(args.max_parallel))),
            ]
            if args.resume or args.skip_existing:
                command.append("--resume")
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        if include_randup:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--randup-campaign",
                "--frozen-manifest",
                str(frozen_path),
                "--results-root",
                str(results_root),
                "--panel",
                str(panel_id),
                "--seed-start",
                str(min(runs)),
                "--num-trials",
                str(max(runs) - min(runs) + 1),
                "--particle-counts",
                str(args.randup_particle_counts),
                "--uncertainty-levels",
                str(args.randup_uncertainty_levels),
                "--max-parallel",
                str(max(1, int(args.max_parallel))),
            ]
            optional = (
                ("--padding-epsilon", args.randup_padding_epsilon),
                ("--max-iterations", args.randup_max_iterations),
                ("--planning-time", args.randup_planning_time),
            )
            for flag, value in optional:
                if value is not None:
                    command.extend([flag, str(value)])
            if args.resume or args.skip_existing:
                command.append("--resume")
            subprocess.run(command, cwd=REPO_ROOT, check=True)

    if include_mppi or include_randup:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "plot_task_time.py"),
                "--results-root",
                str(results_root),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
MPPI_WALL_TIME_DEFINITION = (
    "blocking_mppi_optimization + physical_control_execution"
)
MPPI_SOURCE_PATHS = (
    REPO_ROOT / "methods" / "MPPI.py",
    Path(__file__).resolve(),
    REPO_ROOT / "systems.py",
    REPO_ROOT / "simulation" / "simulator.py",
    REPO_ROOT / "simulation" / "pushing_model.py",
)

SHARED_ENVIRONMENT_CONFIG_FIELDS = (
    "panel_id",
    "system_name",
    "simulator_mode",
    "start_state",
    "goal_state",
    "goal_threshold",
    "propagation_step_size",
    "min_control_duration",
    "max_control_duration",
    "sampling_position_std",
    "sampling_rotation_std",
    "sampling_velocity_std",
    "task_time_limit_seconds",
    "obstacles",
    "model_name",
    "model_path",
    "mujoco_car_throttle_ctrl_scale",
    "mujoco_car_steering_ctrl_scale",
)


def _mppi_source_inventory() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in MPPI_SOURCE_PATHS
    }


def _effective_goal_threshold(config: dict, parameters) -> float:
    del parameters
    return float(config["goal_threshold"])


def shared_environment_contract(
    config: dict,
    system,
    parameters,
) -> dict:
    """Return and validate the task fields shared by MPPI, AURA, and RR.

    MPPI has no goal-sampling operation, so OMPL's goal-bias parameter is
    recorded for provenance but is correctly marked as not applicable to the
    receding-horizon controller.  MPPI uses a fixed one-tick discretization;
    that action duration must lie inside OMPL's admissible duration range.
    """

    step_size = float(config["propagation_step_size"])
    minimum = int(config["min_control_duration"])
    maximum = int(config["max_control_duration"])
    action_steps = int(parameters.action_duration_steps)
    mppi_goal_threshold = _effective_goal_threshold(config, parameters)
    execution_step_seconds = task_execution_step_seconds(
        str(config["system_name"]), step_size
    )
    if not minimum <= action_steps <= maximum:
        raise ValueError(
            "MPPI action duration is outside the shared OMPL duration range: "
            f"{action_steps} not in [{minimum}, {maximum}]"
        )
    if not math.isclose(
        float(system.propagation_step_size),
        step_size,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("MPPI and OMPL propagation step sizes differ")
    if (
        int(system.min_control_duration) != minimum
        or int(system.max_control_duration) != maximum
    ):
        raise ValueError("MPPI and OMPL duration contracts differ")

    configured_control_bounds = config.get("control_bounds")
    if configured_control_bounds is not None and not np.allclose(
        np.asarray(configured_control_bounds, dtype=float),
        np.asarray(system.control_bounds, dtype=float),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("MPPI did not receive the configured control bounds")
    configured_state_bounds = config.get("state_bounds")
    if configured_state_bounds is not None and not np.allclose(
        np.asarray(configured_state_bounds, dtype=float),
        np.asarray(system.state_bounds, dtype=float),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("MPPI did not receive the configured state bounds")

    shared = {
        field: deepcopy(config.get(field))
        for field in SHARED_ENVIRONMENT_CONFIG_FIELDS
        if field in config
    }
    shared.update(
        {
            "effective_state_bounds": [
                [float(low), float(high)] for low, high in system.state_bounds
            ],
            "effective_control_bounds": [
                [float(low), float(high)] for low, high in system.control_bounds
            ],
            "ompl_goal_bias": float(config.get("goal_bias", 0.05)),
            "mppi_goal_bias": "not_applicable_no_state_sampler",
            "mppi_goal_threshold": mppi_goal_threshold,
            "mppi_action_duration_steps": action_steps,
            "mppi_action_duration_seconds": step_size * action_steps,
            "physical_execution_step_seconds": execution_step_seconds,
            "mppi_physical_action_seconds": (
                execution_step_seconds * action_steps
            ),
            "mppi_timing_definition": MPPI_WALL_TIME_DEFINITION,
        }
    )
    return json_value(shared)


def _panel_mppi_metadata(panel: dict) -> dict:
    config = deepcopy(panel["config"])
    system = get_system(config["system_name"])
    _apply_bounds(system, config)
    system.configure_duration_contract(
        float(config["propagation_step_size"]),
        int(config["min_control_duration"]),
        int(config["max_control_duration"]),
    )
    parameters = parameters_from_config(system.name, config)
    contract = shared_environment_contract(config, system, parameters)
    return {
        "config_hash": panel["config_hash"],
        "parameters": parameters.to_dict(),
        "shared_environment_contract": contract,
        "shared_environment_contract_hash": data_hash(contract),
    }


def build_mppi_campaign_manifest(
    results_root: Path,
    frozen: dict,
    *,
    selected_runs: list[int] | None = None,
    device: str | None = None,
    max_parallel_workers: int = 1,
) -> dict:
    inventory = _mppi_source_inventory()
    payload = {
        "schema_version": 1,
        "method": "mppi",
        "wall_time_definition": MPPI_WALL_TIME_DEFINITION,
        "base_seed": int(frozen["base_seed"]),
        "device": str(device or frozen.get("device", "cuda:0")),
        "max_parallel_workers": int(max_parallel_workers),
        "cpu_threads_per_worker": (
            1 if str(device or frozen.get("device", "cuda:0")).startswith("cpu") else None
        ),
        "selected_runs": [
            int(value)
            for value in (
                selected_runs
                if selected_runs is not None
                else frozen.get("selected_runs")
                or range(
                    1,
                    int(frozen.get("num_simulation_trials", 100)) + 1,
                )
            )
        ],
        "source_inventory": inventory,
        "source_hash": data_hash(inventory),
        "panels": {
            panel["config"]["panel_id"]: _panel_mppi_metadata(panel)
            for panel in frozen["panels"]
            if panel.get("execute", False)
            and panel["config"].get("simulator_mode") != "real"
        },
    }
    path = Path(results_root) / "mppi_manifest.json"
    return ensure_json(
        path,
        payload,
        mismatch_message=(
            "MPPI campaign source/configuration changed; refusing to mix rows in "
            f"{results_root}"
        ),
    )


def _torch_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _goal_distance(state: np.ndarray, config: dict) -> float:
    return float(
        arrayDistance(
            state,
            np.asarray(config["goal_state"], dtype=float),
            system=str(config["system_name"]),
        )
    )


def run_mppi_trial(
    panel: dict,
    run_number: int,
    *,
    base_seed: int,
    device: str,
) -> dict:
    """Run MPPI until its audited goal region or 300-second task limit."""

    if torch.device(device).type == "cpu":
        # The strict CPU ablation must not silently use all host cores for
        # batched tensor operations while OMPL is single-threaded.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # A caller may already have initialized the inter-op pool.  The
            # actual value is recorded below, so this cannot be hidden.
            pass

    config = deepcopy(panel["config"])
    panel_id = str(config["panel_id"])
    task_seed = stable_seed(int(base_seed), panel_id, "mppi", int(run_number))
    streams = {
        label: stable_seed(task_seed, label)
        for label in ("task", "controller", "execution")
    }
    np.random.seed(int(streams["task"]))
    torch.manual_seed(int(streams["task"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(streams["task"]))

    step_size = float(config["propagation_step_size"])
    execution_step_seconds = task_execution_step_seconds(
        str(config["system_name"]), step_size
    )
    task_limit = float(config.get("task_time_limit_seconds", 300.0))

    system = get_system(config["system_name"])
    _apply_bounds(system, config)
    system.configure_duration_contract(
        step_size,
        int(config["min_control_duration"]),
        int(config["max_control_duration"]),
    )
    if system.name == "pushing_object":
        system.model_name = str(config.get("model_name", system.model_name))
        system.model_path = config.get("model_path")
    parameters = parameters_from_config(system.name, config)
    goal_threshold = _effective_goal_threshold(config, parameters)
    physical_action_seconds = float(
        execution_step_seconds * int(parameters.action_duration_steps)
    )
    max_actions = int(math.ceil(task_limit / physical_action_seconds))
    disturbance_schedule = np.random.default_rng(
        int(streams["execution"])
    ).standard_normal(
        (
            max_actions + 1,
            6 if str(config["system_name"]) == "dubins_airplane" else 3,
        )
    )
    config["disturbance_seed"] = int(streams["execution"])
    config["disturbance_schedule"] = disturbance_schedule.tolist()
    disturbance_hash = data_hash(
        {
            "seed": int(streams["execution"]),
            "shape": list(disturbance_schedule.shape),
            "standard_normal_samples": disturbance_schedule.tolist(),
        }
    )
    environment_contract = shared_environment_contract(
        config, system, parameters
    )
    source_inventory = _mppi_source_inventory()

    setup_started = time.monotonic()
    simulator = create_simulator(
        config["system_name"], config["simulator_mode"], config=config
    )
    simulator.reset()
    simulator.set_state(config["start_state"])
    controller = MPPIController(
        system,
        config["goal_state"],
        propagation_step_size=step_size,
        parameters=parameters,
        obstacle_config=config.get("obstacles"),
        model_name=config.get("model_name"),
        model_path=config.get("model_path"),
        seed=int(streams["controller"]),
        device=device,
    )
    setup_seconds = time.monotonic() - setup_started

    raw_started = time.monotonic()
    current = np.asarray(simulator.get_state(), dtype=float).reshape(-1)
    measured_states = [current.copy()]
    controls: list[np.ndarray] = []
    duration_steps: list[int] = []
    duration_seconds: list[float] = []
    physical_duration_seconds: list[float] = []
    predicted_states: list[np.ndarray] = []
    tracking_errors: list[float] = []
    command_diagnostics: list[dict] = []
    nominal_execution_seconds = 0.0
    actual_execution_seconds = 0.0
    optimizer_seconds = 0.0
    path_cost = 0.0
    status = "failure"
    failure_reason = ""

    try:
        for _ in range(max_actions + 1):
            distance = _goal_distance(current, config)
            if distance <= goal_threshold:
                status = "success"
                break
            # A complete action must fit within the task-time budget.  MPPI is
            # blocking, so its measured command computation is charged before
            # the one-tick physical execution interval.
            task_time = nominal_execution_seconds + optimizer_seconds
            if task_time + physical_action_seconds > task_limit + 1e-9:
                status = "timeout"
                failure_reason = "task_time_limit_reached"
                break

            _torch_sync(device)
            command_started = time.monotonic()
            control, diagnostic = controller.command(current)
            _torch_sync(device)
            command_elapsed = time.monotonic() - command_started
            optimizer_seconds += float(command_elapsed)
            command_diagnostics.append(
                {**diagnostic, "computation_seconds": float(command_elapsed)}
            )
            if (
                nominal_execution_seconds
                + optimizer_seconds
                + physical_action_seconds
                > task_limit + 1e-9
            ):
                status = "timeout"
                failure_reason = "task_time_limit_reached_after_mppi_update"
                break

            predicted = controller.predict_next(current, control)
            execute_started = time.monotonic()
            measured = np.asarray(
                simulator.execute_segment(control, step_size), dtype=float
            ).reshape(-1)
            actual_execution_seconds += time.monotonic() - execute_started
            if measured.shape != current.shape or not np.isfinite(measured).all():
                raise FloatingPointError(
                    f"simulator returned invalid state {measured!r}"
                )

            tracking_errors.append(
                float(
                    arrayDistance(
                        measured,
                        predicted,
                        system=str(config["system_name"]),
                    )
                )
            )
            path_cost += float(
                arrayDistance(
                    current,
                    measured,
                    system=str(config["system_name"]),
                )
            )
            controls.append(np.asarray(control, dtype=float).copy())
            duration_steps.append(int(parameters.action_duration_steps))
            duration_seconds.append(float(step_size * parameters.action_duration_steps))
            physical_duration_seconds.append(physical_action_seconds)
            predicted_states.append(np.asarray(predicted, dtype=float).copy())
            current = measured
            measured_states.append(current.copy())
            nominal_execution_seconds += physical_action_seconds
        else:
            status = "timeout"
            failure_reason = "task_time_limit_reached"
    except Exception as exc:
        status = "failure"
        failure_reason = f"exception:{exc!r}"
        exception_traceback = traceback.format_exc()
    else:
        exception_traceback = ""
    finally:
        raw_process_seconds = time.monotonic() - raw_started
        if hasattr(simulator, "close"):
            simulator.close()

    goal_distance = _goal_distance(current, config)
    if status == "success" and goal_distance > goal_threshold:
        status = "failure"
        failure_reason = f"goal_not_reached:{goal_distance:.9g}"
    if status != "success" and not failure_reason:
        failure_reason = f"goal_not_reached:{goal_distance:.9g}"
    task_time_seconds = float(nominal_execution_seconds + optimizer_seconds)
    return {
        "schema_version": 3,
        "panel_id": panel_id,
        "system": str(config["system_name"]),
        "environment": str(config["simulator_mode"]),
        "planner": "mppi",
        "method": "mppi",
        "run_number": int(run_number),
        "seed": int(task_seed),
        "rng_streams": streams,
        "config_hash": str(panel["config_hash"]),
        "mppi_config_hash": data_hash(parameters.to_dict()),
        "mppi_source_hash": data_hash(source_inventory),
        "shared_environment_contract": environment_contract,
        "shared_environment_contract_hash": data_hash(environment_contract),
        "mppi_parameters": parameters.to_dict(),
        "mppi_goal_threshold": goal_threshold,
        "execution_device": str(device),
        "torch_intraop_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "initial_planning_seconds": 0.0,
        "initial_plan_hash": None,
        "initial_plan": None,
        "status": status,
        "failure_reason": failure_reason,
        "nominal_execution_seconds": float(nominal_execution_seconds),
        "model_execution_seconds": float(sum(duration_seconds)),
        "physical_execution_seconds": float(nominal_execution_seconds),
        "physical_execution_step_seconds": execution_step_seconds,
        "actual_execution_seconds": float(actual_execution_seconds),
        "online_replanning_seconds": 0.0,
        "optimizer_seconds": float(optimizer_seconds),
        "blocking_replanning_seconds": 0.0,
        "compute_overrun_seconds": 0.0,
        "wall_time_definition": MPPI_WALL_TIME_DEFINITION,
        "task_time_seconds": task_time_seconds,
        "raw_process_wall_seconds": float(raw_process_seconds),
        "setup_reset_operator_pause_seconds": float(setup_seconds),
        "num_controls": len(controls),
        "num_replanning": len(controls),
        "cost": float(path_cost),
        "tracking_error_mean": (
            float(np.mean(tracking_errors)) if tracking_errors else 0.0
        ),
        "tracking_error_list": tracking_errors,
        "goal_distance": float(goal_distance),
        "final_state": current,
        "planned_final_state": predicted_states[-1] if predicted_states else current,
        "controls": controls,
        "control_duration_steps": duration_steps,
        "control_duration_seconds": duration_seconds,
        "physical_control_duration_seconds": physical_duration_seconds,
        "primitive_states": getattr(simulator, "primitive_states", measured_states),
        "command_diagnostics": command_diagnostics,
        "disturbance_schedule_hash": disturbance_hash,
        "disturbance_steps_consumed": int(
            getattr(simulator, "_disturbance_index", 0)
        ),
        "duration_audit_initial_tree": {
            "range_steps": [1, 1],
            "propagation_step_size_seconds": step_size,
            "physical_execution_step_seconds": execution_step_seconds,
            "duration_step_histogram": {},
        },
        "traceback": exception_traceback,
    }


def run_worker(
    frozen: dict,
    panel_id: str,
    run_number: int,
    *,
    results_root: Path,
    device: str,
) -> Path:
    panel = next(
        item
        for item in frozen["panels"]
        if item["config"]["panel_id"] == panel_id
    )
    row = run_mppi_trial(
        panel,
        run_number,
        base_seed=int(frozen["base_seed"]),
        device=device,
    )
    return write_mppi_result(results_root, row)


def _mppi_worker_command(
    manifest_path: Path,
    results_root: Path,
    panel_id: str,
    run_number: int,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mppi-campaign",
        "--worker",
        "--frozen-manifest",
        str(manifest_path),
        "--results-root",
        str(results_root),
        "--panel",
        panel_id,
        "--run",
        str(run_number),
        "--device",
        device,
    ]


def run_mppi_campaign(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--frozen-manifest")
    parser.add_argument("--panel")
    parser.add_argument("--run", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-end", type=int)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(arguments)

    results_root = Path(args.results_root).expanduser().resolve()
    manifest_path = (
        Path(args.frozen_manifest).expanduser().resolve()
        if args.frozen_manifest
        else results_root / "manifest.json"
    )
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.worker:
        if not args.panel or args.run is None:
            raise ValueError("MPPI worker requires --panel and --run")
        output = run_worker(
            frozen,
            str(args.panel),
            int(args.run),
            results_root=results_root,
            device=str(args.device),
        )
        print(output, flush=True)
        return

    selected_runs = list(frozen.get("selected_runs") or [])
    if args.run is not None:
        runs = [int(args.run)]
    else:
        start = int(args.seed_start or (min(selected_runs) if selected_runs else 1))
        count = int(
            args.num_trials
            or len(selected_runs)
            or frozen.get("num_simulation_trials", 100)
        )
        end = int(args.seed_end or (start + count - 1))
        runs = list(range(start, end + 1))
    campaign = build_mppi_campaign_manifest(
        results_root,
        frozen,
        selected_runs=runs,
        device=str(args.device),
        max_parallel_workers=max(1, int(args.max_parallel)),
    )
    panels = [
        panel
        for panel in frozen["panels"]
        if panel.get("execute", False)
        and panel["config"].get("simulator_mode") != "real"
    ]
    if args.panel:
        panels = [
            panel
            for panel in panels
            if panel["config"]["panel_id"] == args.panel
        ]
    if not args.resume and not args.dry_run:
        for panel in panels:
            panel_dir = results_root / panel["config"]["panel_id"]
            for path in panel_dir.glob("mppi_*.csv"):
                path.unlink()
            for path in (panel_dir / "logs").glob("mppi_*.log"):
                path.unlink()
    jobs = []
    for panel in panels:
        panel_id = str(panel["config"]["panel_id"])
        for run_number in runs:
            if args.resume and mppi_result_complete(
                results_root,
                panel_id,
                run_number,
                expected_config_hash=str(panel["config_hash"]),
                expected_source_hash=str(campaign["source_hash"]),
                expected_environment_contract_hash=str(
                    campaign["panels"][panel_id][
                        "shared_environment_contract_hash"
                    ]
                ),
            ):
                continue
            jobs.append((panel_id, run_number))
    print(
        json.dumps(
            {
                "method": "mppi",
                "jobs": len(jobs),
                "panels": [panel["config"]["panel_id"] for panel in panels],
                "runs": runs,
                "device": args.device,
                "results_root": str(results_root),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    def launch(job):
        panel_id, run_number = job
        log_path = (
            results_root
            / panel_id
            / "logs"
            / f"mppi_trial-{run_number:03d}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = _mppi_worker_command(
            manifest_path,
            results_root,
            panel_id,
            run_number,
            str(args.device),
        )
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=float(args.timeout),
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"MPPI worker failed ({completed.returncode}); see {log_path}"
            )
        return job

    failures = []
    with ThreadPoolExecutor(
        max_workers=max(1, int(args.max_parallel))
    ) as executor:
        futures = {executor.submit(launch, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                print(f"[complete] {job}", flush=True)
            except Exception as exc:
                failures.append((job, repr(exc)))
                print(f"[failure] {job}: {exc}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} MPPI workers failed: {failures[:5]}")


METHOD = "randup_rrt"
ORACLE_METHOD = "oracle_randup_rrt"
WALL_TIME_DEFINITION = (
    "initial_offline_planning + simulated_physical_control_execution "
    "+ blocking_fresh_randup_replanning"
)
SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPO_ROOT / "methods" / "RandUpRRT.py",
    REPO_ROOT / "methods" / "plan.py",
    REPO_ROOT / "systems.py",
    REPO_ROOT / "simulation" / "simulator.py",
    REPO_ROOT / "utils" / "utils.py",
)

def _randup_load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _parse_numbers(value: str | Iterable[Any], converter) -> list:
    raw = value.split(",") if isinstance(value, str) else list(value)
    result = [converter(item) for item in raw if str(item).strip()]
    if not result:
        raise ValueError("at least one value is required")
    return result


def _condition_token(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _method_identity(config: RandUpRRTConfig) -> tuple[str, str]:
    if config.is_oracle:
        return ORACLE_METHOD, "Oracle RandUP-RRT"
    return METHOD, "RandUP-RRT"


def _randup_source_inventory() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): data_hash(path.read_text(encoding="utf-8"))
        for path in SOURCE_PATHS
    }


def _signed_aabb_distance(point: np.ndarray, bounds: Iterable[float]) -> float:
    xmin, ymin, xmax, ymax = [float(value) for value in bounds]
    center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    half = np.array([(xmax - xmin) / 2.0, (ymax - ymin) / 2.0])
    delta = np.abs(point - center) - half
    outside = float(np.linalg.norm(np.maximum(delta, 0.0)))
    inside = float(min(max(delta[0], delta[1]), 0.0))
    return outside + inside


def state_obstacle_clearance(state: Iterable[float], obstacle_config: dict | None) -> float | None:
    """Signed point-footprint clearance to the closest configured obstacle."""

    obstacles = normalize_obstacle_config(deepcopy(obstacle_config))
    if not obstacles or not obstacles.get("enabled", False):
        return None
    point = np.asarray(state, dtype=float).reshape(-1)[:2]
    safety = float(obstacles.get("safety_radius", 0.0))
    values: list[float] = []
    for cx, cy, radius in obstacles.get("circles", []):
        values.append(
            float(np.linalg.norm(point - np.array([cx, cy], dtype=float)))
            - float(radius)
            - safety
        )
    for bounds in obstacles.get("aabbs", []):
        values.append(_signed_aabb_distance(point, bounds) - safety)
    for cx, cy, hx, hy, yaw in obstacles.get("boxes", []):
        c, s = math.cos(-float(yaw)), math.sin(-float(yaw))
        offset = point - np.array([cx, cy], dtype=float)
        local = np.array(
            [c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]]
        )
        values.append(
            _signed_aabb_distance(local, (-float(hx), -float(hy), float(hx), float(hy)))
            - safety
        )
    return min(values) if values else None


def minimum_obstacle_clearance(
    states: Iterable[Iterable[float]], obstacle_config: dict | None
) -> float | None:
    values = [
        value
        for value in (state_obstacle_clearance(state, obstacle_config) for state in states)
        if value is not None
    ]
    return min(values) if values else None


def route_signature(states: Iterable[Iterable[float]], obstacle_config: dict | None) -> str:
    """Return integer winding signatures after closing the path start-to-goal.

    This is a diagnostic for route/homotopy changes in the 2-D narrow-passage
    task, not a general-purpose homotopy classifier.
    """

    points = np.asarray([np.asarray(state, dtype=float)[:2] for state in states])
    obstacles = normalize_obstacle_config(deepcopy(obstacle_config))
    if len(points) < 2 or not obstacles or not obstacles.get("enabled", False):
        return "not_applicable"
    centers = [np.asarray(circle[:2], dtype=float) for circle in obstacles.get("circles", [])]
    centers.extend(
        np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
        for box in obstacles.get("aabbs", [])
    )
    centers.extend(np.asarray(box[:2], dtype=float) for box in obstacles.get("boxes", []))
    if not centers:
        return "not_applicable"
    closed = np.vstack([points, points[0]])
    windings = []
    for center in centers:
        angles = np.arctan2(closed[:, 1] - center[1], closed[:, 0] - center[0])
        increments = (np.diff(angles) + np.pi) % (2.0 * np.pi) - np.pi
        windings.append(int(round(float(np.sum(increments) / (2.0 * np.pi)))))
    return ",".join(str(value) for value in windings)


def _planner_from_config(
    system,
    config: dict,
    randup: RandUpRRTConfig,
    *,
    start_state: Iterable[float] | None = None,
) -> OMPLPlanner:
    return OMPLPlanner(
        system=system,
        start_state=np.asarray(
            config["start_state"] if start_state is None else start_state,
            dtype=float,
        ),
        goal_state=np.asarray(config["goal_state"], dtype=float),
        planner_method="randup_rrt",
        goal_threshold=float(config["goal_threshold"]),
        min_max_control_duration=(
            int(randup.control_duration_min),
            int(randup.control_duration_max),
        ),
        propagation_step_size=float(config["propagation_step_size"]),
        initial_planning_time=float(randup.planning_time),
        pruning_radius=float(config.get("pruning_radius", 0.1)),
        goal_bias=float(randup.goal_bias),
        obstacle_config=config.get("obstacles"),
        optimization_objective=str(config.get("optimization_objective", "control_duration")),
        randup_config=randup,
    )


def _finite_mean(values: Iterable[Any]) -> tuple[float | None, int]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return (float(np.mean(finite)), len(finite)) if finite else (None, 0)


def run_randup_trial(
    panel: dict,
    run_number: int,
    *,
    base_seed: int,
    num_particles: int,
    uncertainty_level: float,
    overrides: dict[str, Any] | None = None,
) -> dict:
    """Plan, execute, and freshly replan until goal, collision, or deadline."""

    config = deepcopy(panel["config"])
    overrides = dict(overrides or {})
    level = float(uncertainty_level)
    if not math.isfinite(level) or level < 0.0:
        raise ValueError("uncertainty_level must be finite and nonnegative")
    base_position_std = float(config.get("sampling_position_std", 0.0))
    base_rotation_std = float(config.get("sampling_rotation_std", 0.0))
    base_velocity_std = float(config.get("sampling_velocity_std", base_position_std))
    config["sampling_position_std"] = level * base_position_std
    config["sampling_rotation_std"] = level * base_rotation_std
    config["sampling_velocity_std"] = level * base_velocity_std
    config["randup_num_particles"] = int(num_particles)
    config.update({key: value for key, value in overrides.items() if value is not None})
    if "randup_uncertainty_mode" not in config:
        if str(config.get("simulator_mode", "")).lower() == "gaussian":
            config["randup_uncertainty_mode"] = "gaussian_process_oracle"
        else:
            config["randup_uncertainty_mode"] = "none"
    if str(config["randup_uncertainty_mode"]).lower() == "aura_optimizer_gaussian":
        config["randup_position_std"] = level * float(
            config.get("optimizer_position_std", 0.0)
        )
        config["randup_rotation_std"] = level * float(
            config.get("optimizer_rotation_std", 0.0)
        )
        config["randup_velocity_std"] = level * float(
            config.get(
                "optimizer_velocity_std",
                config.get("optimizer_position_std", 0.0),
            )
        )

    panel_id = str(config["panel_id"])
    task_seed = stable_seed(
        int(base_seed), panel_id, METHOD, int(num_particles), level, int(run_number)
    )
    planning_seed = stable_seed(task_seed, "planning_particles")
    ompl_seed = max(1, stable_seed(task_seed, "ompl_exploration"))
    # Pair held-out execution exactly with the frozen AORRT/AURA and restart-
    # replanning row for this panel/trial. Planning particles remain on a
    # separate RandUP-only stream and never receive these samples.
    evaluation_reference_planner = "aorrt"
    evaluation_master_seed = stable_seed(
        int(base_seed), panel_id, evaluation_reference_planner, int(run_number)
    )
    execution_seed = stable_seed(evaluation_master_seed, "execution")
    config["randup_random_seed"] = int(planning_seed)
    randup = RandUpRRTConfig.from_mapping(config)
    randup.validate()
    method, method_label = _method_identity(randup)
    planner_key = (
        f"randup_rrt_m{int(randup.num_particles)}_u{_condition_token(level)}"
    )
    condition = {
        "panel_config_hash": str(panel["config_hash"]),
        "shared_start_state": list(config["start_state"]),
        "shared_goal_state": list(config["goal_state"]),
        "shared_goal_threshold": float(config["goal_threshold"]),
        "shared_state_bounds": deepcopy(config.get("state_bounds")),
        "shared_control_bounds": deepcopy(config.get("control_bounds")),
        "shared_propagation_step_size": float(config["propagation_step_size"]),
        "shared_obstacles": deepcopy(config.get("obstacles")),
        "num_particles": int(randup.num_particles),
        "padding_epsilon": float(randup.padding_epsilon),
        "uncertainty_level": level,
        "uncertainty_mode": str(randup.uncertainty_mode),
        "planning_uncertainty_contract": (
            "configured_ground_truth_action_noise_oracle"
            if randup.is_oracle
            else "no_process_noise_model_no_oracle_simulator_queries"
            if str(randup.uncertainty_mode).lower() in {"none", "deterministic", "zero"}
            else "configured_approximate_process_noise"
        ),
        "execution_uncertainty_contract": (
            "configured_gaussian_action_noise"
            if str(config.get("simulator_mode", "")).lower() == "gaussian"
            else "held_out_mujoco_model_mismatch"
        ),
        "planning_position_std": float(randup.position_std),
        "planning_rotation_std": float(randup.rotation_std),
        "planning_velocity_std": float(randup.velocity_std),
        "execution_position_std": float(config.get("sampling_position_std", 0.0)),
        "execution_rotation_std": float(config.get("sampling_rotation_std", 0.0)),
        "execution_velocity_std": float(config.get("sampling_velocity_std", 0.0)),
        "planning_time": float(randup.planning_time),
        "max_iterations": int(randup.max_iterations),
        "goal_bias": float(randup.goal_bias),
        "control_sampling_rule": (
            "one_unbiased_ompl_control_and_one_uniform_integer_duration;"
            "no_steering_no_candidate_ranking_no_heuristic"
        ),
        "evaluation_seed_reference": "aorrt_aura_and_restart_replanning",
        "control_duration_range": [
            int(randup.control_duration_min),
            int(randup.control_duration_max),
        ],
        "replanning_policy": (
            "aura_matched_bounded_fresh_tree_retries_until_plan_or_task_deadline;"
            "fresh_randup_after_executed_plan_misses_goal"
        ),
        "task_time_limit_seconds": float(
            config.get("task_time_limit_seconds", 300.0)
        ),
    }

    system = get_system(config["system_name"])
    _apply_bounds(system, config)
    if system.name == "pushing_object":
        system.model_name = str(config.get("model_name", system.model_name))
        system.model_path = config.get("model_path")
    system.configure_duration_contract(
        float(config["propagation_step_size"]),
        int(randup.control_duration_min),
        int(randup.control_duration_max),
    )
    ou.RNG.setSeed(int(ompl_seed))
    task_limit = float(config.get("task_time_limit_seconds", 300.0))
    max_steps = int(config.get("max_steps", 200))
    execution_step_seconds = task_execution_step_seconds(
        config["system_name"], float(config["propagation_step_size"])
    )
    disturbance_count = max(
        1,
        max_steps * int(config["max_control_duration"]),
    )
    disturbance_schedule = np.random.default_rng(int(execution_seed)).standard_normal(
        (
            disturbance_count,
            6 if str(config["system_name"]) == "dubins_airplane" else 3,
        )
    )
    disturbance_hash = data_hash(
        {
            "seed": int(execution_seed),
            "shape": list(disturbance_schedule.shape),
            "standard_normal_samples": disturbance_schedule.tolist(),
        }
    )

    current = np.asarray(config["start_state"], dtype=float)
    primitive_states = [current.copy()]
    executed_controls: list[list[float]] = []
    executed_steps: list[int] = []
    executed_seconds: list[float] = []
    tracking_errors: list[float] = []
    all_planned_states: list[list[float]] = [current.tolist()]
    all_planned_controls: list[list[float]] = []
    all_planned_steps: list[int] = []
    all_planned_seconds: list[float] = []
    planning_episodes: list[dict] = []
    planning_episode_seeds: list[int] = []
    first_solution: dict | None = None
    first_planning_seconds = 0.0
    total_planning_seconds = 0.0
    replanning_seconds = 0.0
    num_replanning = 0
    collision = False
    actual_execution_seconds = 0.0
    status = "failure"
    failure_reason = ""
    last_expansion_rejection = ""
    exception_tracebacks: list[str] = []
    aggregate_stats: dict[str, Any] = {
        "planning_time": 0.0,
        "iterations": 0,
        "tree_expansions": 0,
        "accepted_edges": 0,
        "tree_size": 0,
        "dynamics_propagations": 0,
        "particle_propagations": 0,
        "particle_collision_rejections": 0,
        "nominal_collision_rejections": 0,
        "numerical_propagation_failures": 0,
        "best_nominal_goal_distance": float("inf"),
        "best_worst_particle_goal_distance": float("inf"),
    }
    simulator = None
    raw_started = time.monotonic()
    episode_index = 0
    try:
        while True:
            elapsed_task = total_planning_seconds + len(primitive_states[1:]) * execution_step_seconds
            remaining = task_limit - elapsed_task
            if remaining <= 1e-9:
                status = "timeout"
                failure_reason = "task_time_limit_reached_before_replanning"
                break
            if len(primitive_states) - 1 >= max_steps:
                failure_reason = "execution_step_limit_reached"
                break

            if episode_index > 0:
                num_replanning += 1
            attempt_index = 0
            consecutive_immediate_returns = 0
            solution = None
            abort_planning_episode = False
            while solution is None:
                elapsed_task = (
                    total_planning_seconds
                    + (len(primitive_states) - 1) * execution_step_seconds
                )
                remaining = task_limit - elapsed_task
                if remaining <= 1e-9:
                    status = "timeout"
                    failure_reason = "task_time_limit_reached_during_planning"
                    abort_planning_episode = True
                    break
                if episode_index == 0 and attempt_index == 0:
                    attempt_seed = int(planning_seed)
                else:
                    attempt_seed = stable_seed(
                        planning_seed,
                        "episode",
                        episode_index,
                        "attempt",
                        attempt_index,
                    )
                planning_episode_seeds.append(int(attempt_seed))
                episode_randup = replace(
                    randup,
                    random_seed=int(attempt_seed),
                    planning_time=min(float(randup.planning_time), float(remaining)),
                )
                planner = _planner_from_config(
                    system,
                    config,
                    episode_randup,
                    start_state=current,
                )
                planning_started = time.monotonic()
                planning_exception = ""
                try:
                    solutions, _ = planner.plan()
                except Exception as exc:
                    solutions = []
                    planning_exception = f"numerical_propagation_failure:{exc!r}"
                    exception_tracebacks.append(traceback.format_exc())
                internal = getattr(planner, "randup_planner", None)
                episode_stats = deepcopy(getattr(internal, "stats", {}))
                measured_attempt_seconds = time.monotonic() - planning_started
                episode_planning_seconds = float(
                    episode_stats.get("planning_time", measured_attempt_seconds)
                )
                total_planning_seconds += episode_planning_seconds
                if episode_index == 0:
                    first_planning_seconds += episode_planning_seconds
                else:
                    replanning_seconds += episode_planning_seconds
                for field in (
                    "iterations",
                    "tree_expansions",
                    "accepted_edges",
                    "tree_size",
                    "dynamics_propagations",
                    "particle_propagations",
                    "particle_collision_rejections",
                    "nominal_collision_rejections",
                    "numerical_propagation_failures",
                ):
                    aggregate_stats[field] += int(episode_stats.get(field, 0))
                for field in (
                    "best_nominal_goal_distance",
                    "best_worst_particle_goal_distance",
                ):
                    value = float(episode_stats.get(field, float("inf")))
                    aggregate_stats[field] = min(
                        float(aggregate_stats[field]), value
                    )
                last_expansion_rejection = str(
                    episode_stats.get(
                        "last_expansion_rejection",
                        getattr(internal, "last_expansion_rejection", ""),
                    )
                )
                planner_failure = str(getattr(internal, "failure_reason", ""))
                solution = next(
                    (
                        candidate
                        for candidate in (solutions or [])
                        if candidate.get("controls")
                        and len(candidate.get("states", [])) >= 2
                    ),
                    None,
                )
                planning_episodes.append(
                    {
                        "episode": episode_index,
                        "attempt": attempt_index,
                        "seed": int(attempt_seed),
                        "start_state": current.tolist(),
                        "planning_budget_seconds": float(episode_randup.planning_time),
                        "planning_seconds": episode_planning_seconds,
                        "success": solution is not None,
                        "failure_reason": planning_exception or planner_failure,
                        "stats": episode_stats,
                        "solution": solution,
                    }
                )
                if solution is not None:
                    break
                prefix = "initial" if episode_index == 0 else "replanning"
                failure_reason = (
                    planning_exception
                    or planner_failure
                    or f"{prefix}_no_robust_plan_found"
                )
                immediate_threshold = min(
                    0.05,
                    max(0.005, 0.05 * float(episode_randup.planning_time)),
                )
                if measured_attempt_seconds < immediate_threshold:
                    consecutive_immediate_returns += 1
                else:
                    consecutive_immediate_returns = 0
                if consecutive_immediate_returns >= 3:
                    failure_reason = (
                        f"{prefix}_planner_returned_immediately_without_exact_solution"
                    )
                    abort_planning_episode = True
                    break
                attempt_index += 1
            if abort_planning_episode or solution is None:
                break

            if first_solution is None:
                first_solution = deepcopy(solution)
            controls = [list(control) for control in solution.get("controls", [])]
            planned_steps = [int(value) for value in solution.get("time_steps", [])]
            planned_seconds = [float(value) for value in solution.get("time", [])]
            solution_states = [list(state) for state in solution.get("states", [])]
            all_planned_controls.extend(controls)
            all_planned_steps.extend(planned_steps)
            all_planned_seconds.extend(planned_seconds)
            all_planned_states.extend(solution_states[1:] if solution_states else [])

            if simulator is None:
                execution_config = deepcopy(config)
                execution_config["disturbance_seed"] = int(execution_seed)
                execution_config["disturbance_schedule"] = disturbance_schedule.tolist()
                simulator = create_simulator(
                    config["system_name"],
                    config["simulator_mode"],
                    config=execution_config,
                )
                simulator.reset()
                simulator.set_state(config["start_state"])

            predicted = current.copy()
            step_size = float(config["propagation_step_size"])
            deadline_reached = False
            for control, edge_steps in zip(controls, planned_steps):
                edge_executed = 0
                for _ in range(int(edge_steps)):
                    executed_primitive_count = len(primitive_states) - 1
                    if executed_primitive_count >= max_steps:
                        break
                    if (
                        total_planning_seconds
                        + (executed_primitive_count + 1) * execution_step_seconds
                        > task_limit + 1e-9
                    ):
                        deadline_reached = True
                        break
                    predicted = np.asarray(
                        system.propagate(
                            predicted,
                            np.asarray(control, dtype=float),
                            step_size,
                        ),
                        dtype=float,
                    ).reshape(-1)
                    execute_started = time.monotonic()
                    current = np.asarray(
                        simulator.execute_segment(control, step_size), dtype=float
                    ).reshape(-1)
                    actual_execution_seconds += time.monotonic() - execute_started
                    primitive_states.append(current.copy())
                    edge_executed += 1
                    tracking_errors.append(
                        float(
                            arrayDistance(
                                current,
                                predicted,
                                system=config["system_name"],
                            )
                        )
                    )
                    if not is_state_array_valid(
                        current,
                        system=config["system_name"],
                        config=config,
                        obstacle_config=config.get("obstacles"),
                    ):
                        collision = True
                        break
                if edge_executed:
                    executed_controls.append([float(value) for value in control])
                    executed_steps.append(edge_executed)
                    executed_seconds.append(edge_executed * step_size)
                if collision or deadline_reached or len(primitive_states) - 1 >= max_steps:
                    break

            goal_distance_now = float(
                arrayDistance(current, config["goal_state"], system=config["system_name"])
            )
            if collision:
                failure_reason = "execution_collision"
                break
            if deadline_reached:
                status = "timeout"
                failure_reason = "task_time_limit_reached_during_execution"
                break
            if goal_distance_now <= float(config["goal_threshold"]) + 1e-9:
                status = "success"
                failure_reason = ""
                break
            if len(primitive_states) - 1 >= max_steps:
                failure_reason = "execution_step_limit_reached"
                break
            episode_index += 1
    except Exception as exc:
        failure_reason = f"execution_failure:{exc!r}"
        exception_tracebacks.append(traceback.format_exc())
    finally:
        if simulator is not None and hasattr(simulator, "close"):
            simulator.close()

    goal_distance = float(
        arrayDistance(current, config["goal_state"], system=config["system_name"])
    )
    planning_success = first_solution is not None
    execution_success = bool(status == "success")
    if not execution_success and not failure_reason:
        failure_reason = f"goal_not_reached:{goal_distance:.9g}"
    execution_time = float(sum(executed_steps) * execution_step_seconds)
    measured_task_time = float(total_planning_seconds + execution_time)
    if (
        status != "success"
        and not collision
        and measured_task_time >= task_limit - 1e-3
    ):
        status = "timeout"
        failure_reason = "task_time_limit_reached"
    cost = (
        float(
            sum(
                arrayDistance(first, second, system=config["system_name"])
                for first, second in zip(primitive_states, primitive_states[1:])
            )
        )
        if len(primitive_states) > 1
        else None
    )
    clearance = (
        minimum_obstacle_clearance(primitive_states, config.get("obstacles"))
        if len(primitive_states) > 1
        else None
    )
    aggregate_stats.update(
        {
            "planning_success": planning_success,
            "failure_reason": failure_reason,
            "planning_time": float(total_planning_seconds),
            "last_expansion_rejection": last_expansion_rejection,
            "num_particles": int(randup.num_particles),
            "padding_epsilon": float(randup.padding_epsilon),
            "uncertainty_mode": str(randup.uncertainty_mode),
            "goal_requires_all_particles": bool(randup.goal_requires_all_particles),
            "planning_episode_count": len(planning_episodes),
            "num_replanning": int(num_replanning),
            "planning_episodes": planning_episodes,
        }
    )
    row = {
        "schema_version": 1,
        "panel_id": panel_id,
        "system": str(config["system_name"]),
        "environment": str(config["simulator_mode"]),
        "planner": planner_key,
        "method": method,
        "method_label": method_label,
        "run_number": int(run_number),
        "seed": int(task_seed),
        "planning_seed": int(planning_seed),
        "execution_seed": int(execution_seed),
        "rng_streams": {
            "task": int(task_seed),
            "ompl_exploration": int(ompl_seed),
            "planning_particles": int(planning_seed),
            "planning_episode_particles": planning_episode_seeds,
            "held_out_execution": int(execution_seed),
            "held_out_execution_reference_planner": evaluation_reference_planner,
        },
        "uncertainty_level": level,
        "planning_success": planning_success,
        "execution_success": execution_success,
        "collision": collision,
        "status": status,
        "failure_reason": failure_reason,
        "last_expansion_rejection": last_expansion_rejection,
        "initial_planning_seconds": first_planning_seconds,
        "execution_time_seconds": execution_time,
        "nominal_execution_seconds": execution_time,
        "actual_execution_seconds": float(actual_execution_seconds),
        "online_replanning_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "blocking_replanning_seconds": float(replanning_seconds),
        "compute_overrun_seconds": 0.0,
        "wall_time_definition": WALL_TIME_DEFINITION,
        "task_time_seconds": measured_task_time,
        "raw_process_wall_seconds": float(time.monotonic() - raw_started),
        "setup_reset_operator_pause_seconds": 0.0,
        "cost": cost,
        "goal_distance": goal_distance,
        "minimum_obstacle_clearance": clearance,
        "tree_expansions": int(aggregate_stats["tree_expansions"]),
        "tree_size": int(aggregate_stats["tree_size"]),
        "dynamics_propagations": int(aggregate_stats["dynamics_propagations"]),
        "particle_propagations": int(aggregate_stats["particle_propagations"]),
        "particle_collision_rejections": int(aggregate_stats["particle_collision_rejections"]),
        "nominal_collision_rejections": int(aggregate_stats["nominal_collision_rejections"]),
        "numerical_propagation_failures": int(aggregate_stats["numerical_propagation_failures"]),
        "num_particles": int(randup.num_particles),
        "padding_epsilon": float(randup.padding_epsilon),
        "uncertainty_mode": str(randup.uncertainty_mode),
        "goal_requires_all_particles": bool(randup.goal_requires_all_particles),
        "num_controls": len(executed_controls),
        "num_replanning": int(num_replanning),
        "executed_primitive_steps": int(sum(executed_steps)),
        "route_signature": route_signature(all_planned_states, config.get("obstacles")),
        "config_hash": str(panel["config_hash"]),
        "condition_hash": data_hash(condition),
        "disturbance_schedule_hash": disturbance_hash,
        "initial_plan_hash": (
            data_hash(
                {
                    "states": first_solution.get("states", []),
                    "controls": first_solution.get("controls", []),
                    "time_steps": first_solution.get("time_steps", []),
                    "time": first_solution.get("time", []),
                    "cost": first_solution.get("cost"),
                }
            )
            if first_solution
            else None
        ),
        "initial_plan": first_solution,
        "tracking_error_mean": float(np.mean(tracking_errors)) if tracking_errors else 0.0,
        "tracking_error_list": tracking_errors,
        "final_state": current.tolist(),
        "planned_final_state": all_planned_states[-1] if all_planned_states else None,
        "controls": executed_controls,
        "control_duration_steps": executed_steps,
        "control_duration_seconds": executed_seconds,
        "planned_controls": all_planned_controls,
        "planned_control_duration_steps": all_planned_steps,
        "planned_control_duration_seconds": all_planned_seconds,
        "primitive_states": [state.tolist() for state in primitive_states],
        "duration_audit_initial_tree": {
            "range_steps": [
                int(randup.control_duration_min),
                int(randup.control_duration_max),
            ],
            "propagation_step_size_seconds": float(config["propagation_step_size"]),
            "duration_step_histogram": {},
        },
        "planner_stats": aggregate_stats,
        "planning_episodes": planning_episodes,
        "condition": condition,
        "source_inventory": _randup_source_inventory(),
        "traceback": "\n".join(exception_tracebacks),
    }
    return json_value(row)


def aggregate_randup_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for source_row in rows:
        row = dict(source_row)
        planner = str(row.get("planner", ""))
        particle_token = planner.rpartition("_m")[2].split("_", 1)[0]
        status_success = str(row.get("status", "")).lower() == "success"
        row.setdefault("method_label", "RandUP-RRT")
        row.setdefault("num_particles", int(particle_token))
        row.setdefault("padding_epsilon", 0.0)
        row.setdefault("uncertainty_level", 1.0)
        row.setdefault("planning_success", row.get("initial_plan") is not None)
        row.setdefault("execution_success", status_success)
        row.setdefault("collision", False)
        key = (
            row["panel_id"],
            row["method_label"],
            int(row["num_particles"]),
            float(row["padding_epsilon"]),
            float(row["uncertainty_level"]),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        panel_id, label, particles, padding, level = key
        planned = [row for row in group if bool(row["planning_success"])]
        successful = [row for row in group if bool(row["execution_success"])]
        summary = {
            "panel_id": panel_id,
            "method": label,
            "num_particles": particles,
            "padding_epsilon": padding,
            "uncertainty_level": level,
            "n_trials": len(group),
            "n_plans_found": len(planned),
            "n_executed_plans": len(planned),
            "n_execution_success": len(successful),
            "n_collisions": sum(bool(row["collision"]) for row in group),
            "planning_success_rate": len(planned) / len(group),
            "execution_success_rate": len(successful) / len(planned) if planned else None,
            "overall_task_success_rate": len(successful) / len(group),
            "collision_rate": sum(bool(row["collision"]) for row in group) / len(group),
        }
        metrics = {
            "planning_time": (row.get("initial_planning_seconds") for row in group),
            "execution_time": (row.get("execution_time_seconds") for row in planned),
            "total_time": (row.get("task_time_seconds") for row in group),
            "cost": (row.get("cost") for row in group),
            "final_error": (row.get("goal_distance") for row in group),
            "minimum_clearance": (row.get("minimum_obstacle_clearance") for row in planned),
            "tree_expansions": (row.get("tree_expansions") for row in group),
            "dynamics_propagations": (row.get("dynamics_propagations") for row in group),
            "particle_propagations": (row.get("particle_propagations") for row in group),
        }
        for name, values in metrics.items():
            mean, count = _finite_mean(values)
            summary[f"{name}_mean"] = mean
            summary[f"{name}_valid_n"] = count
        failure_counts: dict[str, int] = {}
        for row in group:
            reason = str(row.get("failure_reason") or "success")
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
        summary["failure_reasons"] = json.dumps(failure_counts, sort_keys=True)
        summary["route_signatures"] = json.dumps(
            sorted({str(row.get("route_signature", "")) for row in planned})
        )
        output.append(summary)
    return output


def _standalone_panel(path: Path) -> dict:
    config = _randup_load_yaml(path)
    return {
        "config_path": str(path.resolve()),
        "config": config,
        "config_hash": data_hash(config),
        "execute": True,
    }


def run_randup_campaign(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="One ordinary experiment panel YAML.")
    source.add_argument("--frozen-manifest", help="Frozen fig7 manifest.json.")
    parser.add_argument("--results-root")
    parser.add_argument("--panel")
    parser.add_argument("--run", type=int)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--particle-counts", default="50")
    parser.add_argument("--uncertainty-levels", default="1.0")
    parser.add_argument("--num-particles", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--padding-epsilon", type=float)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--planning-time", type=float)
    parser.add_argument("--goal-bias", type=float)
    parser.add_argument("--random-seed", type=int, help="Override campaign base seed.")
    parser.add_argument("--duration-min", type=int)
    parser.add_argument("--duration-max", type=int)
    parser.add_argument("--uncertainty-mode")
    parser.add_argument("--nominal-goal-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum isolated trial subprocesses to run concurrently.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(arguments)

    if args.frozen_manifest:
        manifest_path = Path(args.frozen_manifest).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        panels = [panel for panel in manifest["panels"] if panel.get("execute", False)]
        base_seed = int(args.base_seed or args.random_seed or manifest["base_seed"])
        results_root = (
            Path(args.results_root).expanduser().resolve()
            if args.results_root
            else manifest_path.parent
        )
    else:
        config_path = Path(args.config).expanduser().resolve()
        panels = [_standalone_panel(config_path)]
        base_seed = int(args.base_seed or args.random_seed or panels[0]["config"].get("randup_random_seed", 0))
        results_root = (
            Path(args.results_root).expanduser().resolve()
            if args.results_root
            else REPO_ROOT / "results" / "full_time_comparison"
        )
    if args.panel:
        panels = [panel for panel in panels if panel["config"]["panel_id"] == args.panel]
    if not panels:
        raise ValueError("no executable panel matched the requested filter")
    particle_counts = _parse_numbers(args.particle_counts, int)
    if args.num_particles is not None:
        particle_counts = [int(args.num_particles)]
    uncertainty_levels = _parse_numbers(args.uncertainty_levels, float)
    runs = [int(args.run)] if args.run is not None else list(
        range(int(args.seed_start), int(args.seed_start) + int(args.num_trials))
    )
    overrides = {
        "randup_padding_epsilon": args.padding_epsilon,
        "randup_max_iterations": args.max_iterations,
        "randup_planning_time": args.planning_time,
        "goal_bias": args.goal_bias,
        "randup_control_duration_min": args.duration_min,
        "randup_control_duration_max": args.duration_max,
        "randup_uncertainty_mode": args.uncertainty_mode,
        "randup_goal_requires_all_particles": not args.nominal_goal_only,
    }
    jobs = [
        (panel, run, particles, level)
        for panel in panels
        for particles in particle_counts
        for level in uncertainty_levels
        for run in runs
    ]
    print(
        json.dumps(
            {
                "method": METHOD,
                "jobs": len(jobs),
                "panels": [panel["config"]["panel_id"] for panel in panels],
                "runs": runs,
                "particle_counts": particle_counts,
                "uncertainty_levels": uncertainty_levels,
                "results_root": str(results_root),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if not args.worker:
        randup_manifest_path = results_root / "randup_manifest.json"
        if randup_manifest_path.is_file():
            randup_manifest = json.loads(
                randup_manifest_path.read_text(encoding="utf-8")
            )
            if int(randup_manifest.get("base_seed", base_seed)) != int(base_seed):
                if args.resume:
                    raise ValueError(
                        "existing RandUP manifest uses a different base seed"
                    )
                randup_manifest = {}
        else:
            randup_manifest = {}
        if not args.resume:
            for panel in panels:
                panel_dir = results_root / panel["config"]["panel_id"]
                for path in (panel_dir / "artifacts").glob(
                    "randup_rrt*_trial-*.json"
                ):
                    path.unlink()
                for path in panel_dir.glob("randup_rrt_*.csv"):
                    path.unlink()
                for path in (panel_dir / "logs").glob("randup_rrt_*.log"):
                    path.unlink()
        panel_conditions = dict(randup_manifest.get("panel_conditions", {}))
        for panel in panels:
            panel_conditions[str(panel["config"]["panel_id"])] = {
                "config_hash": str(panel["config_hash"]),
                "runs": runs,
                "particle_counts": particle_counts,
                "uncertainty_levels": uncertainty_levels,
                "overrides": {
                    key: value for key, value in overrides.items() if value is not None
                },
            }
        write_json(
            randup_manifest_path,
            {
                "schema_version": 2,
                "base_seed": base_seed,
                "source_inventory": _randup_source_inventory(),
                "panel_conditions": panel_conditions,
            },
        )
    if args.worker and len(jobs) != 1:
        raise ValueError("RandUP-RRT worker requires exactly one job")
    pending_commands: list[tuple[str, list[str], Path]] = []
    for panel, run, particles, level in jobs:
        probe_config = deepcopy(panel["config"])
        probe_config.update({key: value for key, value in overrides.items() if value is not None})
        if "randup_uncertainty_mode" not in probe_config:
            probe_config["randup_uncertainty_mode"] = (
                "gaussian_process_oracle"
                if str(probe_config.get("simulator_mode", "")).lower() == "gaussian"
                else "none"
            )
        probe_config["randup_num_particles"] = particles
        method, _ = _method_identity(RandUpRRTConfig.from_mapping(probe_config))
        planner_key = f"randup_rrt_m{particles}_u{_condition_token(level)}"
        expected_path = (
            results_root
            / panel["config"]["panel_id"]
            / "artifacts"
            / f"{planner_key}_trial-{run:03d}.json"
        )
        if args.resume and expected_path.is_file():
            print(f"[skip] {panel['config']['panel_id']} {planner_key} trial {run}")
            continue
        if not args.worker:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--randup-campaign",
                "--worker",
                "--results-root",
                str(results_root),
                "--panel",
                str(panel["config"]["panel_id"]),
                "--run",
                str(run),
                "--base-seed",
                str(base_seed),
                "--particle-counts",
                str(particles),
                "--uncertainty-levels",
                str(level),
            ]
            if args.frozen_manifest:
                command.extend(["--frozen-manifest", str(Path(args.frozen_manifest).expanduser().resolve())])
            else:
                command.extend(["--config", str(Path(args.config).expanduser().resolve())])
            option_values = (
                ("--padding-epsilon", args.padding_epsilon),
                ("--max-iterations", args.max_iterations),
                ("--planning-time", args.planning_time),
                ("--goal-bias", args.goal_bias),
                ("--duration-min", args.duration_min),
                ("--duration-max", args.duration_max),
                ("--uncertainty-mode", args.uncertainty_mode),
            )
            for flag, value in option_values:
                if value is not None:
                    command.extend([flag, str(value)])
            if args.nominal_goal_only:
                command.append("--nominal-goal-only")
            log_path = (
                results_root
                / str(panel["config"]["panel_id"])
                / "logs"
                / f"{planner_key}_trial-{run:03d}.log"
            )
            pending_commands.append(
                (
                    f"{panel['config']['panel_id']} {planner_key} trial {run}",
                    command,
                    log_path,
                )
            )
            continue
        row = run_randup_trial(
            panel,
            run,
            base_seed=base_seed,
            num_particles=particles,
            uncertainty_level=level,
            overrides=overrides,
        )
        write_randup_result(results_root, row, rebuild_trials_csv=False)
        print(
            f"[complete] {panel['config']['panel_id']} {method} M={particles} "
            f"u={level:g} trial={run} status={row['status']} "
            f"reason={row['failure_reason'] or 'success'}",
            flush=True,
        )
        return

    if not args.worker:
        max_parallel = max(1, int(args.max_parallel))

        def run_command(job: tuple[str, list[str], Path]) -> tuple[str, Path]:
            label, command, log_path = job
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                )
                raise RuntimeError(
                    f"RandUP-RRT worker failed ({label}); log={log_path}\n{tail}"
                )
            return label, log_path

        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(run_command, job): job[0] for job in pending_commands}
            for future in as_completed(futures):
                label, log_path = future.result()
                completed_count += 1
                print(
                    f"[campaign {completed_count}/{len(pending_commands)}] {label} "
                    f"log={log_path}",
                    flush=True,
                )
        rewrite_randup_trials(results_root)
    trial_paths = sorted(
        path
        for panel_id in PANEL_LAYOUT
        for path in (results_root / panel_id / "artifacts").glob(
            "randup_rrt*_trial-*.json"
        )
    )
    summaries = aggregate_randup_rows(
        [json.loads(path.read_text(encoding="utf-8")) for path in trial_paths]
    )
    csv_path, markdown_path = write_randup_summary(results_root, summaries)
    print(f"wrote {csv_path} and {markdown_path}", flush=True)


def main(arguments: list[str] | None = None) -> None:
    """Dispatch the public task-time campaign or one of its worker modes."""
    selected = list(sys.argv[1:] if arguments is None else arguments)
    if selected and selected[0] == "--mppi-campaign":
        run_mppi_campaign(selected[1:])
        return
    if selected and selected[0] == "--randup-campaign":
        run_randup_campaign(selected[1:])
        return
    if not any(
        flag in selected
        for flag in ("--include-mppi", "--include-randup", "--all-baselines")
    ):
        selected.insert(0, "--all-baselines")
    run_task_time_campaign(selected)


if __name__ == "__main__":
    main()
