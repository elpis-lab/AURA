#!/usr/bin/env python3
"""Shared-seed trajectory-cost experiment for AURA and vanilla planners.

The experiment compares AORRT, AOEST, and SSTStar before and after AURA's
noise-free tree refinement across four systems and 100 shared-seed trials.

Seeding contract
-----------------
For a given (system, run_number), every planner x planning_time combination
shares one master seed, derived deterministically from (base_seed,
system_name, run_number) -- deliberately NOT including planner_name or
planning_time. This lets the same random problem instance be compared across
methods.

OMPL's global RNG can only be seeded once per process, before any RNG-backed
object is created. Each (system, planner, run_number, planning_time)
combination therefore runs in its own fresh OS process. The default mode is
the orchestrator that computes the job list and dispatches those workers with
bounded parallelism.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from methods.plan import OMPLPlanner
from systems import get_system
from utils.experiment_io import stable_seed, write_csv, write_json


PLANNERS = ("aorrt", "aoest", "sststar")
DEFAULT_CONFIG = {
    "system_name": "kinematic_car",
    "start_state": [0.0, 0.0, 0.0],
    "goal_state": [4.75, 4.75, 0.0],
    "state_bounds": [(0.0, 5.0), (0.0, 5.0)],
    "goal_threshold": 0.1,
    "min_control_duration": 1,
    "max_control_duration": 1,
    "propagation_step_size": 1.0,
    "pruning_radius": 0.1,
    "optimization_objective": "path_length",
    "planning_times": [2.0, 4.0, 6.0, 8.0, 10.0],
}


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config file must contain a mapping: {path}")
    config = {**DEFAULT_CONFIG, **loaded}
    config["system_name"] = str(config["system_name"])
    config["start_state"] = np.asarray(config["start_state"], dtype=float)
    config["goal_state"] = np.asarray(config["goal_state"], dtype=float)
    config["state_bounds"] = [
        (float(low), float(high))
        for low, high in config["state_bounds"]
    ]
    control_bounds = config.get("control_bounds")
    config["control_bounds"] = (
        [(float(low), float(high)) for low, high in control_bounds]
        if control_bounds is not None
        else None
    )
    sampling_bounds = config.get("control_sampling_bounds")
    config["control_sampling_bounds"] = (
        [(float(low), float(high)) for low, high in sampling_bounds]
        if sampling_bounds is not None
        else None
    )
    for key in ("goal_threshold", "propagation_step_size", "pruning_radius"):
        config[key] = float(config[key])
    for key in ("min_control_duration", "max_control_duration"):
        config[key] = int(config[key])
    config["planning_times"] = [float(value) for value in config["planning_times"]]
    config["optimization_objective"] = str(config["optimization_objective"])
    return config


def build_planner(
    system, config: dict, planner_name: str, planning_time: float
) -> OMPLPlanner:
    if config.get("control_sampling_bounds") is not None:
        system.set_control_sampling_bounds(config["control_sampling_bounds"])
    overrides = (config.get("planner_overrides") or {}).get(planner_name, {})
    selection_radius = overrides.get(
        "selection_radius", config.get("selection_radius")
    )
    planner = OMPLPlanner(
        system=system,
        start_state=config["start_state"],
        goal_state=config["goal_state"],
        planner_method=planner_name,
        goal_threshold=config["goal_threshold"],
        min_max_control_duration=(
            config["min_control_duration"],
            config["max_control_duration"],
        ),
        propagation_step_size=config["propagation_step_size"],
        initial_planning_time=float(planning_time),
        pruning_radius=float(
            overrides.get("pruning_radius", config["pruning_radius"])
        ),
        obstacle_config=config.get("obstacles"),
        optimization_objective=config["optimization_objective"],
        goal_bias=float(config.get("goal_bias", 0.05)),
        selection_radius=(
            float(selection_radius) if selection_radius is not None else None
        ),
        single_solve=True,
    )
    return planner


def recomputed_path_lengths(solution: dict, system) -> tuple[float, float]:
    states = [
        np.asarray(state, dtype=float).reshape(-1)
        for state in solution.get("states", [])
    ]
    if not states:
        return float("nan"), float("nan")
    native = sum(
        system.state_distance(first, second)
        for first, second in zip(states, states[1:])
    )
    position_dim = 2 if system.name in ("kinematic_car", "pushing_object") else 3
    position = sum(
        float(np.linalg.norm(second[:position_dim] - first[:position_dim]))
        for first, second in zip(states, states[1:])
    )
    return float(native), float(position)


def best_solution_metrics(solutions: list[dict], system) -> dict[str, Any]:
    audited = []
    for solution in solutions or []:
        native, position = recomputed_path_lengths(solution, system)
        if np.isfinite(native):
            audited.append((native, position, solution))
    if not audited:
        return {
            "status": "failure",
            "failure_reason": "no exact solution reached the goal region",
            "cost": float("nan"),
            "ompl_cost": float("nan"),
            "recomputed_path_length": float("nan"),
            "position_path_length": float("nan"),
            "cost_consistency_error": float("nan"),
            "cost_consistent": False,
            "control_count": 0,
            "state_count": 0,
            "goal_distance": float("nan"),
            "control_duration_steps": [],
            "control_duration_seconds": [],
        }
    native, position, best = min(audited, key=lambda item: item[0])
    ompl_cost = float(best.get("cost", float("nan")))
    error = abs(ompl_cost - native)
    tolerance = 1e-6 * max(1.0, abs(ompl_cost), abs(native))
    return {
        "status": "success",
        "failure_reason": "",
        "cost": native,
        "ompl_cost": ompl_cost,
        "recomputed_path_length": native,
        "position_path_length": position,
        "cost_consistency_error": error,
        "cost_consistent": bool(np.isfinite(ompl_cost) and error <= tolerance),
        "control_count": int(best["control_count"]),
        "state_count": int(best["state_count"]),
        "goal_distance": float(best["goal_distance"]),
        "control_duration_steps": [int(value) for value in best.get("time_steps", [])],
        "control_duration_seconds": [float(value) for value in best.get("time", [])],
    }


CSV_FIELDS = (
    "run_number", "planner", "planning_time", "initial_cost", "final_cost",
    "initial_recomputed_path_length", "final_recomputed_path_length",
    "initial_position_path_length", "final_position_path_length",
    "initial_cost_consistency_error", "final_cost_consistency_error",
    "initial_cost_consistent", "final_cost_consistent", "improvement",
    "planning_wall_time", "refinement_wall_time", "refinement_iterations",
    "status", "failure_reason", "cost", "ompl_cost", "control_count",
    "state_count", "goal_distance", "initial_control_count",
    "final_control_count", "initial_control_duration_steps",
    "final_control_duration_steps", "initial_control_duration_seconds",
    "final_control_duration_seconds", "initial_goal_distance",
    "final_goal_distance", "seed", "ompl_seed", "system", "start_state",
    "goal_state", "cost_definition", "direct_start_goal_distance",
    "direct_position_distance", "goal_threshold",
    "goal_threshold_fraction_of_direct", "actuator_control_bounds",
    "ompl_control_sampling_bounds",
)


SYSTEMS = ("double_integrator", "kinematic_car", "pushing_object", "dubins_airplane")

CONFIG_PATHS = {
    name: REPO_ROOT / "configs" / "exp1" / f"{name}.yaml" for name in SYSTEMS
}
# dubins_airplane.yaml is the Stage-1 10-trial preview config (num_runs=10,
# a different results_dir); Stage-2's 100-trial sweep uses its own file with
# a wider planning_times range so the original preview stays reproducible.
CONFIG_PATHS["dubins_airplane"] = REPO_ROOT / "configs" / "exp1" / "dubins_airplane_stage2.yaml"

# --------------------------------------------------------------------------
# Execution-coupled AURA replanning: replan for the current plan's first
# control's own duration, commit to (propagate/execute) that control, then
# continue from the refined tree. Mirrors AURA.py's real online loop
# (`while control_count > 1: ...`, budget = duration of controls[0]) without
# a simulator/noise/local optimizer -- with no execution uncertainty, the
# tree's own noise-free propagation IS the execution, so there is nothing
# for the optimizer to correct. See AURA.py lines ~244-320 for the reference
# (state advance via simulator, `execution_control_duration` as both the
# execute and replan budget, loop condition `control_count > 1`).
# --------------------------------------------------------------------------


def _best_raw_solution(solutions: list[dict], system) -> dict | None:
    audited = []
    for solution in solutions or []:
        recomputed, _position = recomputed_path_lengths(solution, system)
        if math.isfinite(recomputed):
            audited.append((recomputed, solution))
    if not audited:
        return None
    _, best = min(audited, key=lambda item: item[0])
    return best


def _find_state_index(states: list[np.ndarray], target_state: np.ndarray, tol: float = 1e-6) -> int | None:
    target = np.asarray(target_state, dtype=float).reshape(-1)
    for index, state in enumerate(states):
        if np.allclose(np.asarray(state, dtype=float).reshape(-1), target, atol=tol, rtol=0.0):
            return index
    return None


def run_aura_replan_trajectory(planner, system, initial_solution: dict) -> dict:
    """Execute-and-replan loop; returns the trajectory cost of all visited states."""
    states = [np.asarray(s, dtype=float) for s in initial_solution["states"]]
    controls = [np.asarray(c, dtype=float) for c in initial_solution["controls"]]
    durations = [float(t) for t in initial_solution["time"]]

    current_state = states[0].copy()
    visited_states = [current_state.copy()]
    remaining_controls = list(controls)
    remaining_durations = list(durations)
    replanning_iterations = 0
    replanning_wall_time = 0.0
    continuations_from_refined_tree = 0
    continuations_from_fallback = 0

    while len(remaining_controls) > 1:
        budget = remaining_durations[0]
        started = time.monotonic()
        refined_solutions, _ = planner.replan(time_budget=float(budget))
        replanning_wall_time += time.monotonic() - started
        replanning_iterations += 1

        next_state = np.asarray(
            system.propagate(current_state, remaining_controls[0], remaining_durations[0]),
            dtype=float,
        )
        visited_states.append(next_state)
        current_state = next_state

        # Look up the refined tree's current best continuation from the
        # state just committed to, by finding where it falls in the
        # (globally re-derived) current best solution.
        best = _best_raw_solution(refined_solutions, system)
        continuation = None
        if best is not None:
            best_states = [np.asarray(s, dtype=float) for s in best["states"]]
            index = _find_state_index(best_states, current_state)
            if index is not None and index + 1 < len(best_states):
                continuation = (
                    [np.asarray(c, dtype=float) for c in best["controls"][index:]],
                    [float(t) for t in best["time"][index:]],
                )
        if continuation is not None:
            continuations_from_refined_tree += 1
        else:
            # The current globally-best path no longer passes through the
            # committed state (a different route scored better elsewhere in
            # the tree). Since a real online system cannot un-execute a
            # committed control, fall back to the pre-replan tail: these
            # custom incremental planners only ever add/improve solutions
            # across resolve() calls, they do not discard prior ones, so the
            # old continuation from this exact state remains valid.
            continuations_from_fallback += 1
            continuation = (remaining_controls[1:], remaining_durations[1:])
        remaining_controls, remaining_durations = continuation

    if remaining_controls:
        next_state = np.asarray(
            system.propagate(current_state, remaining_controls[0], remaining_durations[0]),
            dtype=float,
        )
        visited_states.append(next_state)

    cost = sum(
        float(system.state_distance(visited_states[i], visited_states[i + 1]))
        for i in range(len(visited_states) - 1)
    )
    return {
        "cost": cost,
        "control_count": max(0, len(visited_states) - 1),
        "state_count": len(visited_states),
        "visited_states": [s.tolist() for s in visited_states],
        "replanning_iterations": replanning_iterations,
        "replanning_wall_time": replanning_wall_time,
        "continuations_from_refined_tree": continuations_from_refined_tree,
        "continuations_from_fallback": continuations_from_fallback,
    }


def run_once_execution_coupled(
    config: dict,
    planner_name: str,
    planning_time: float,
    run_number: int,
    seed: int,
    *,
    initialize_ompl_rng: bool = True,
) -> dict:
    """Compare an initial solution with execution-coupled AURA refinement."""
    from ompl import util as ou

    np.random.seed(int(seed))
    ompl_seed = int(seed) + 1
    if initialize_ompl_rng:
        ou.RNG.setSeed(ompl_seed)

    system = get_system(config["system_name"])
    system.set_state_bounds(config["state_bounds"])
    if config["control_bounds"] is not None:
        system.set_control_bounds(config["control_bounds"])
    planner = build_planner(system, config, planner_name, planning_time)
    if str(config["optimization_objective"]).lower() != "path_length":
        raise ValueError("Experiment 1 requires optimization_objective: path_length")

    direct_start_goal_distance = system.state_distance(config["start_state"], config["goal_state"])
    position_dim = 2 if system.name in ("kinematic_car", "pushing_object") else 3
    direct_position_distance = float(
        np.linalg.norm(
            np.asarray(config["goal_state"], dtype=float)[:position_dim]
            - np.asarray(config["start_state"], dtype=float)[:position_dim]
        )
    )

    started_at = time.time()
    try:
        solutions, _ = planner.plan()
        initial = best_solution_metrics(solutions, system)
        aura_result = None
        if initial["status"] == "success":
            initial_best = _best_raw_solution(solutions, system)
            aura_started = time.time()
            aura_result = run_aura_replan_trajectory(planner, system, initial_best)
            aura_wall_time = time.time() - aura_started
        final_status = "success" if aura_result is not None else "failure"
        final_cost = aura_result["cost"] if aura_result is not None else float("nan")
        final_control_count = aura_result["control_count"] if aura_result is not None else 0
        final_goal_distance = (
            float(system.state_distance(aura_result["visited_states"][-1], config["goal_state"]))
            if aura_result is not None
            else float("nan")
        )
        row = {
            "status": "success" if initial["status"] == "success" and final_status == "success" else "failure",
            "failure_reason": "" if initial["status"] == "success" else initial.get("failure_reason", ""),
            "initial_cost": initial["cost"],
            "final_cost": final_cost,
            "initial_recomputed_path_length": initial["recomputed_path_length"],
            "final_recomputed_path_length": final_cost,
            "initial_position_path_length": initial["position_path_length"],
            "final_position_path_length": float("nan"),
            "initial_cost_consistency_error": initial["cost_consistency_error"],
            "final_cost_consistency_error": float("nan"),
            "initial_cost_consistent": initial["cost_consistent"],
            "final_cost_consistent": True if aura_result is not None else False,
            "improvement": (initial["cost"] - final_cost) if aura_result is not None else float("nan"),
            "initial_control_count": initial["control_count"],
            "final_control_count": final_control_count,
            "initial_control_duration_steps": initial["control_duration_steps"],
            "final_control_duration_steps": [],
            "initial_control_duration_seconds": initial["control_duration_seconds"],
            "final_control_duration_seconds": [],
            "initial_goal_distance": initial["goal_distance"],
            "final_goal_distance": final_goal_distance,
            "refinement_wall_time": aura_result["replanning_wall_time"] if aura_result is not None else 0.0,
            "refinement_iterations": aura_result["replanning_iterations"] if aura_result is not None else 0,
            "cost": final_cost,
            "ompl_cost": float("nan"),
            "control_count": final_control_count,
            "state_count": aura_result["state_count"] if aura_result is not None else 0,
            "goal_distance": final_goal_distance,
            "continuations_from_refined_tree": (
                aura_result["continuations_from_refined_tree"] if aura_result is not None else 0
            ),
            "continuations_from_fallback": (
                aura_result["continuations_from_fallback"] if aura_result is not None else 0
            ),
        }
    except Exception as exc:
        row = {
            "status": "failure",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "cost": float("nan"),
            "ompl_cost": float("nan"),
            "control_count": 0,
            "state_count": 0,
            "goal_distance": float("nan"),
            "initial_cost": float("nan"),
            "final_cost": float("nan"),
            "initial_recomputed_path_length": float("nan"),
            "final_recomputed_path_length": float("nan"),
            "initial_position_path_length": float("nan"),
            "final_position_path_length": float("nan"),
            "initial_cost_consistency_error": float("nan"),
            "final_cost_consistency_error": float("nan"),
            "initial_cost_consistent": False,
            "final_cost_consistent": False,
            "improvement": float("nan"),
            "initial_control_count": 0,
            "final_control_count": 0,
            "initial_control_duration_steps": [],
            "final_control_duration_steps": [],
            "initial_control_duration_seconds": [],
            "final_control_duration_seconds": [],
            "initial_goal_distance": float("nan"),
            "final_goal_distance": float("nan"),
            "refinement_wall_time": 0.0,
            "refinement_iterations": 0,
            "continuations_from_refined_tree": 0,
            "continuations_from_fallback": 0,
        }
    wall_time = time.time() - started_at

    row.setdefault("failure_reason", "")
    row.update(
        {
            "run_number": int(run_number),
            "planner": planner_name,
            "planning_time": float(planning_time),
            "planning_wall_time": wall_time,
            "seed": int(seed),
            "ompl_seed": ompl_seed,
            "system": str(config["system_name"]),
            "start_state": np.asarray(config["start_state"], dtype=float).tolist(),
            "goal_state": np.asarray(config["goal_state"], dtype=float).tolist(),
            "cost_definition": "sum_native_state_space_distance_over_executed_trajectory",
            "direct_start_goal_distance": float(direct_start_goal_distance),
            "direct_position_distance": float(direct_position_distance),
            "goal_threshold": float(config["goal_threshold"]),
            "goal_threshold_fraction_of_direct": float(
                config["goal_threshold"] / direct_start_goal_distance
            ),
            "actuator_control_bounds": [list(bound) for bound in system.control_bounds],
            "ompl_control_sampling_bounds": [
                list(bound) for bound in (config.get("control_sampling_bounds") or system.control_bounds)
            ],
        }
    )
    return row


# --------------------------------------------------------------------------
# Worker entry points (one fresh OS process each)
# --------------------------------------------------------------------------


def _worker_planner(args: argparse.Namespace) -> None:
    config = load_config(str(args.config))
    row = run_once_execution_coupled(
        config,
        args.planner,
        float(args.planning_time),
        int(args.run_number),
        int(args.seed),
        initialize_ompl_rng=True,
    )
    write_json(Path(args.output), row)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def _worker_command(kind: str, **kwargs: object) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        f"--worker-{kind}",
    ]
    for key, value in kwargs.items():
        command.append(f"--{key.replace('_', '-')}")
        command.append(str(value))
    return command


def _run_subprocess(command: list[str], *, timeout: float) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )
        return completed.returncode == 0, completed.stdout
    except subprocess.TimeoutExpired as exc:
        return False, f"timeout after {timeout}s: {exc}"


def orchestrate(
    systems: list[str],
    base_seed: int,
    num_runs: int,
    max_parallel: int,
    results_root: Path,
    scratch_root: Path,
    *,
    job_timeout: float = 600.0,
    resume: bool = False,
) -> None:
    if not resume:
        shutil.rmtree(scratch_root, ignore_errors=True)
        for system_name in systems:
            system_dir = results_root / system_name
            for planner_name in PLANNERS:
                for path in system_dir.glob(f"{planner_name}_*.csv"):
                    path.unlink()
    jobs: list[dict] = []
    for system_name in systems:
        config = load_config(str(CONFIG_PATHS[system_name]))
        planning_times = list(config["planning_times"])
        for run_number in range(1, num_runs + 1):
            master_seed = stable_seed(base_seed, system_name, run_number)
            for planner_name in PLANNERS:
                for planning_time in planning_times:
                    jobs.append(
                        {
                            "kind": "planner",
                            "system": system_name,
                            "planner": planner_name,
                            "run_number": run_number,
                            "planning_time": planning_time,
                            "seed": master_seed,
                        }
                    )

    print(f"[orchestrate] {len(jobs)} total jobs across {len(systems)} systems", flush=True)
    scratch_root.mkdir(parents=True, exist_ok=True)

    def launch(job: dict) -> tuple[dict, bool, str]:
        output_path = (
            scratch_root
            / job["system"]
            / job["kind"]
            / f"{job['planner']}_run{job['run_number']:03d}_pt{job['planning_time']:.1f}.json"
        )
        if resume and output_path.exists():
            return job, True, "cached"
        command = _worker_command(
            "planner",
            config=CONFIG_PATHS[job["system"]],
            planner=job["planner"],
            run_number=job["run_number"],
            planning_time=job["planning_time"],
            seed=job["seed"],
            output=output_path,
        )
        ok, log = _run_subprocess(command, timeout=job_timeout)
        return job, ok, log

    completed = 0
    failed_logs: list[str] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(launch, job): job for job in jobs}
        for future in as_completed(futures):
            job, ok, log = future.result()
            completed += 1
            if not ok:
                failed_logs.append(f"{job}: {log[-500:]}")
            if completed % 100 == 0 or completed == len(jobs):
                elapsed = time.monotonic() - started
                rate = completed / max(elapsed, 1e-9)
                remaining = (len(jobs) - completed) / max(rate, 1e-9)
                print(
                    f"[orchestrate] {completed}/{len(jobs)} done "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                    f"{len(failed_logs)} failed)",
                    flush=True,
                )

    if failed_logs:
        print(f"[orchestrate] {len(failed_logs)} jobs failed (subprocess error, not planner failure):")
        for entry in failed_logs[:20]:
            print(f"  {entry}")

    _collect_results(systems, num_runs, results_root, scratch_root)
    if not failed_logs:
        shutil.rmtree(scratch_root, ignore_errors=True)


def _collect_results(
    systems: list[str], num_runs: int, results_root: Path, scratch_root: Path
) -> None:
    for system_name in systems:
        planner_dir = results_root / system_name
        planner_dir.mkdir(parents=True, exist_ok=True)
        for planner_name in PLANNERS:
            for run_number in range(1, num_runs + 1):
                rows = []
                json_dir = scratch_root / system_name / "planner"
                for path in sorted(json_dir.glob(f"{planner_name}_run{run_number:03d}_pt*.json")):
                    rows.append(json.loads(path.read_text()))
                if not rows:
                    continue
                rows.sort(key=lambda r: float(r["planning_time"]))
                csv_path = planner_dir / f"{planner_name}_{run_number:03d}.csv"
                # Keep diagnostic continuation counts in scratch JSON while
                # writing only the stable public CSV schema.
                trimmed_rows = [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("continuations_from_refined_tree", "continuations_from_fallback")
                    }
                    for row in rows
                ]
                write_csv(csv_path, trimmed_rows, CSV_FIELDS)

    print(f"[collect] wrote per-system results under {results_root}")


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-planner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--planner", default=None)
    parser.add_argument("--system", default=None)
    parser.add_argument("--run-number", type=int, default=None)
    parser.add_argument("--planning-time", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)

    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--max-parallel", type=int, default=12)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results/trajectory_cost_comparison",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=None,
        help="Temporary worker output (removed after a successful campaign).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--job-timeout", type=float, default=600.0)
    args = parser.parse_args(arguments)

    if args.worker_planner:
        _worker_planner(args)
        return
    orchestrate(
        list(args.systems),
        int(args.base_seed),
        int(args.num_runs),
        int(args.max_parallel),
        Path(args.results_root),
        Path(args.scratch_root or (Path(args.results_root) / ".work")),
        job_timeout=float(args.job_timeout),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
