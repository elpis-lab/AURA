#!/usr/bin/env python3
"""Run the AURA initial-planning-time/control-duration sensitivity grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml
from ompl import util as ou

from experiment.task_time_efficiency import plan_until_solution, run_aura
from methods.plan import OMPLPlanner
from propagators import get_system
from utils.utils import arrayDistance, is_state_array_valid, normalize_obstacle_config


DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "experiments" / "initial_time_sensitivity.yaml"
)
SYSTEM_CONFIG = REPO_ROOT / "configs" / "systems" / "kinematic_car.yaml"
FIELDS = (
    "run_number",
    "planner",
    "planning_time",
    "control_duration",
    "status",
    "failure_reason",
    "collision_step",
    "planning_wall_time",
    "aura_wall_time",
    "aura_time",
    "cost",
    "tracking_error_mean",
    "goal_distance",
    "planned_goal_distance",
    "final_to_planned_distance",
    "num_controls",
    "final_x",
    "final_y",
    "final_theta",
    "planned_final_x",
    "planned_final_y",
    "planned_final_theta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    experiment = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    system = yaml.safe_load(SYSTEM_CONFIG.read_text(encoding="utf-8")) or {}
    system_values = {
        key: value
        for key, value in system.items()
        if key not in {"environments", "title"}
    }
    config = {
        **system_values,
        **system["environments"]["gaussian"],
        **experiment,
        "planner_name": experiment["planner"],
        "num_runs": experiment["num_trials"],
        "seed": experiment["base_seed"],
    }
    required = (
        "planner_name",
        "results_dir",
        "control_durations",
        "planning_times",
        "num_runs",
        "start_state",
        "goal_state",
        "state_bounds",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing config values: {', '.join(missing)}")
    if int(config["num_runs"]) <= 0:
        raise ValueError("num_runs must be positive")
    return config


def resolve_results_dir(config: dict, override: Path | None) -> Path:
    path = override or Path(config["results_dir"])
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def duration_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def result_path(results_dir: Path, config: dict, duration: float, run: int) -> Path:
    return results_dir / (
        f"{config['system_name']}_{config['planner_name']}_"
        f"cd{duration_token(duration)}_{run:02d}.csv"
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)
    os.replace(temporary, path)


def stable_seed(base_seed: int, run: int, duration: float, planning_time: float) -> int:
    payload = f"{base_seed}:{run}:{duration:.12g}:{planning_time:.12g}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (2**31 - 1)


def trial_config(config: dict, duration: float, seed: int) -> dict:
    trial = dict(config)
    validation_step = float(config["motion_validation_step_size"])
    duration_steps = max(1, int(round(float(duration) / validation_step)))
    trial["propagation_step_size"] = float(duration) / duration_steps
    trial["min_control_duration"] = int(
        config.get("minimum_control_duration_steps", 1)
    )
    trial["max_control_duration"] = duration_steps
    if trial["min_control_duration"] > trial["max_control_duration"]:
        raise ValueError("minimum control duration exceeds the tested maximum")
    trial["disturbance_seed"] = int(seed)
    return trial


def build_planner(config: dict, planning_time: float) -> OMPLPlanner:
    system = get_system(str(config["system_name"]))
    system.set_state_bounds(config["state_bounds"])
    planner = OMPLPlanner(
        system=system,
        start_state=np.asarray(config["start_state"], dtype=float),
        goal_state=np.asarray(config["goal_state"], dtype=float),
        planner_method=str(config["planner_name"]),
        goal_threshold=float(config["goal_threshold"]),
        min_max_control_duration=(
            int(config["min_control_duration"]),
            int(config["max_control_duration"]),
        ),
        propagation_step_size=float(config["propagation_step_size"]),
        initial_planning_time=float(planning_time),
        pruning_radius=float(config["pruning_radius"]),
        goal_bias=float(config.get("goal_bias", 0.05)),
        obstacle_config=config.get("obstacles"),
        single_solve=True,
    )
    planner.replanning_time = float(config["replanning_time_budget"])
    planner.recovery_replanning_time = float(config["recovery_replanning_time"])
    planner.motion_validation_step_size = float(
        config.get("motion_validation_step_size", config["propagation_step_size"])
    )
    planner.solution_continuity_max_distance = float(
        config["solution_continuity_max_distance"]
    )
    planner.optimizer_num_states = int(config["optimizer_num_states"])
    planner.optimizer_num_steps = int(config["optimizer_num_steps"])
    planner.optimizer_learning_rate = float(config["optimizer_learning_rate"])
    planner.optimizer_pos_std = float(
        config.get("optimizer_position_std", config["sampling_position_std"])
    )
    planner.optimizer_rot_std = float(
        config.get("optimizer_rotation_std", config["sampling_rotation_std"])
    )
    planner.optimizer_vel_std = float(config["sampling_velocity_std"])
    planner.optimizer_max_children = int(config.get("optimizer_max_children", 0))
    planner.optimizer_device = "cuda" if torch.cuda.is_available() else "cpu"
    planner.opt_model = None
    return planner


def failure_row(config: dict, run: int, duration: float, planning_time: float, reason: str) -> dict:
    penalty = float(config["failure_time"])
    return {
        "run_number": run,
        "planner": config["planner_name"],
        "planning_time": planning_time,
        "control_duration": duration,
        "status": "failure",
        "failure_reason": " ".join(reason.split())[:500],
        "collision_step": -1,
        "planning_wall_time": 0.0,
        "aura_wall_time": penalty,
        "aura_time": penalty,
        "num_controls": 0,
    }


def first_collision(states: list[np.ndarray], config: dict) -> int | None:
    obstacles = normalize_obstacle_config(config.get("obstacles"))
    for index, state in enumerate(states[1:], start=1):
        if not is_state_array_valid(
            np.asarray(state, dtype=float),
            system=str(config["system_name"]),
            config=config,
            obstacle_config=obstacles,
            safety_radius_override=0.0,
        ):
            return index
    return None


def run_trial(config: dict, run: int, duration: float, planning_time: float) -> dict:
    seed = stable_seed(int(config["seed"]), run, duration, planning_time)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    trial = trial_config(config, duration, seed)
    trial["planning_time"] = float(planning_time)
    planner = build_planner(trial, planning_time)

    initial_solution, planning_wall, _, planning_error = plan_until_solution(
        planner,
        attempt_budget_seconds=float(planning_time),
        total_budget_seconds=float(config["task_time_limit_seconds"]),
    )
    if initial_solution is None:
        row = failure_row(
            config,
            run,
            duration,
            planning_time,
            f"initial planning failed: {planning_error}",
        )
        row["planning_wall_time"] = planning_wall
        return row
    planner.solutions = [initial_solution]

    streams = {
        "execution": stable_seed(seed, 1, duration, planning_time),
        "optimization": stable_seed(seed, 2, duration, planning_time),
    }
    payload, _ = run_aura(
        planner,
        trial,
        streams,
        str(config["planner_name"]),
        float(planning_wall),
    )
    aura_wall = float(payload["raw_process_wall_seconds"])
    final = np.asarray(payload["final_state"], dtype=float)
    planned = (
        np.asarray(payload["planned_final_state"], dtype=float)
        if payload.get("planned_final_state") is not None
        else np.full(3, np.nan)
    )
    goal = np.asarray(config["goal_state"], dtype=float)
    goal_distance = float(arrayDistance(final, goal, system=str(config["system_name"])))
    planned_goal_distance = float(arrayDistance(planned, goal, system=str(config["system_name"])))
    final_to_planned = float(arrayDistance(final, planned, system=str(config["system_name"])))
    collision = first_collision(payload.get("primitive_states", []), trial)
    success = (
        str(payload["status"]) == "success"
        and collision is None
        and goal_distance <= float(config["goal_threshold"])
    )
    penalty = float(config["failure_time"])
    return {
        "run_number": run,
        "planner": config["planner_name"],
        "planning_time": planning_time,
        "control_duration": duration,
        "status": "success" if success else "failure",
        "failure_reason": "" if success else (payload["failure_reason"] or "goal not reached"),
        "collision_step": -1 if collision is None else collision,
        "planning_wall_time": planning_wall,
        "aura_wall_time": aura_wall if success else penalty,
        "aura_time": planning_wall + aura_wall if success else penalty,
        "cost": float(payload["cost"]),
        "tracking_error_mean": float(payload["tracking_error_mean"]),
        "goal_distance": goal_distance,
        "planned_goal_distance": planned_goal_distance,
        "final_to_planned_distance": final_to_planned,
        "num_controls": int(payload["num_controls"]),
        "final_x": final[0],
        "final_y": final[1],
        "final_theta": final[2],
        "planned_final_x": planned[0],
        "planned_final_y": planned[1],
        "planned_final_theta": planned[2],
    }


def main() -> None:
    args = parse_args()
    try:
        ou.setLogLevel(ou.LOG_ERROR)
    except Exception:
        pass
    config = load_config(args.config.resolve())
    ou.RNG.setSeed(int(config["seed"]))
    results_dir = resolve_results_dir(config, args.results_dir)
    durations = [float(value) for value in config["control_durations"]]
    planning_times = [float(value) for value in config["planning_times"]]
    jobs = []
    requested_times = set(planning_times)
    for duration in durations:
        for run in range(1, int(config["num_runs"]) + 1):
            path = result_path(results_dir, config, duration, run)
            loaded_rows = [] if args.overwrite else read_rows(path)
            rows = list(loaded_rows)
            rows[:] = [
                row
                for row in rows
                if float(row["planning_time"]) in requested_times
            ]
            if len(rows) != len(loaded_rows) and not args.dry_run:
                write_rows(path, rows)
            completed = {float(row["planning_time"]) for row in rows}
            jobs.extend(
                (path, rows, run, duration, planning_time)
                for planning_time in planning_times
                if planning_time not in completed
            )

    print(f"[initial-time] grid={len(durations) * len(planning_times) * int(config['num_runs'])} "
          f"complete={len(durations) * len(planning_times) * int(config['num_runs']) - len(jobs)} "
          f"remaining={len(jobs)}")
    if args.dry_run:
        return

    for index, (path, rows, run, duration, planning_time) in enumerate(jobs, start=1):
        print(
            f"[initial-time] {index}/{len(jobs)} run={run:02d} "
            f"duration={duration:g} planning={planning_time:g}s",
            flush=True,
        )
        try:
            row = run_trial(config, run, duration, planning_time)
        except Exception as error:
            traceback.print_exc()
            row = failure_row(config, run, duration, planning_time, repr(error))
        rows[:] = [item for item in rows if float(item["planning_time"]) != planning_time]
        rows.append(row)
        rows.sort(key=lambda item: float(item["planning_time"]))
        write_rows(path, rows)
        print(
            f"[initial-time] status={row['status']} task_time={float(row['aura_time']):.3f}s",
            flush=True,
        )

    print(f"[initial-time] complete: {results_dir}")


if __name__ == "__main__":
    main()
