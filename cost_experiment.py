#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import random
import re
import sys
import time
from typing import Any

import numpy as np
import ompl.util as ou

from systems import get_system, plan as PlanSession
from utils.configHandler import parse_args_and_config
from utils.solutionsHandler import getSolutionsInfo


def _cfg(config: dict, snake: str, camel: str, default: Any):
    return config.get(snake, config.get(camel, default))


def get_next_run_number(results_dir: str, planner_name: str) -> int:
    os.makedirs(results_dir, exist_ok=True)
    existing_files = glob.glob(f"{results_dir}/{planner_name}_*.csv")
    run_numbers = []
    for file in existing_files:
        match = re.search(rf"{planner_name}_(\d+)\.csv", os.path.basename(file))
        if match:
            run_numbers.append(int(match.group(1)))
    return (max(run_numbers) + 1) if run_numbers else 1


def get_best_cost(ss) -> float:
    info = getSolutionsInfo(ss)
    if info is None or len(info) == 0:
        return float("inf")
    return float(info[0]["cost"])


def replan_until_short_solution(ss, step_size: float, max_iter: int = 200, max_time: float = 30.0):
    """
    Keep calling resolve(step_size) to improve the solution until:
      - control count <= 1, or
      - safeguards trigger.
    """
    start_t = time.time()
    info = getSolutionsInfo(ss)
    if info is None or len(info) == 0:
        return ss

    num_controls = info[0]["control_count"]
    prev_controls = num_controls
    no_improve = 0

    while num_controls > 1:
        if (time.time() - start_t) > max_time:
            break
        if max_iter <= 0:
            break

        ss.getPlanner().resolve(step_size)
        info = getSolutionsInfo(ss)
        if info is None or len(info) == 0:
            break

        num_controls = info[0]["control_count"]
        max_iter -= 1

        if num_controls < prev_controls:
            no_improve = 0
            prev_controls = num_controls
        else:
            no_improve += 1
            if no_improve >= 25:
                break

    return ss


def run_experiment_for_planner(config: dict, planner_name: str, planning_times: np.ndarray, run_number: int):
    system_name = _cfg(config, "system", "system", "simple_car")
    object_name = _cfg(config, "objectName", "objectName", "crackerBox")

    start_state = np.array(_cfg(config, "start_state", "startState", [0.0, 0.0, 0.0]), dtype=float)
    goal_state = np.array(_cfg(config, "goal_state", "goalState", [1.0, 1.0, 0.0]), dtype=float)

    goal_threshold = float(_cfg(config, "goal_threshold", "goal_threshold", 0.1))
    min_cd = int(_cfg(config, "min_control_duration", "min_control_duration", 1))
    max_cd = int(_cfg(config, "max_control_duration", "max_control_duration", 5))
    step_size = float(_cfg(config, "propagation_step_size", "propagation_step_size", 1.0))
    pruning_radius = float(_cfg(config, "pruning_radius", "pruning_radius", 0.1))

    system = get_system(system_name, object_name=object_name)
    rows = []

    for planning_time in planning_times:
        session = PlanSession(
            system=system,
            start_state=start_state,
            goal_state=goal_state,
            planner_name=planner_name,
            goal_threshold=goal_threshold,
            min_control_duration=min_cd,
            max_control_duration=max_cd,
            propagation_step_size=step_size,
            planning_time=float(planning_time),
            pruning_radius=pruning_radius,
            config=config,
            visualize=bool(_cfg(config, "visualize", "visualize", False)),
        )

        t0 = time.time()
        solutions, ss = session.plan()
        planning_wall_time = time.time() - t0

        if solutions is None or len(solutions) == 0:
            return None

        initial_cost = get_best_cost(ss)
        ss = replan_until_short_solution(ss, step_size=step_size)
        final_cost = get_best_cost(ss)

        rows.append(
            {
                "run_number": run_number,
                "planner": planner_name,
                "planning_time": float(planning_time),
                "planning_wall_time": planning_wall_time,
                "initial_cost": initial_cost,
                "final_cost": final_cost,
                "improvement": initial_cost - final_cost,
                "seed": ou.RNG.getSeed(),
                "system": system_name,
                "start_state": str(start_state.tolist()),
                "goal_state": str(goal_state.tolist()),
            }
        )

    return rows


def save_run_csv(results_dir: str, planner_name: str, run_number: int, rows: list[dict]):
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{results_dir}/{planner_name}_{run_number:02d}.csv"
    fieldnames = [
        "run_number",
        "planner",
        "planning_time",
        "planning_wall_time",
        "initial_cost",
        "final_cost",
        "improvement",
        "seed",
        "system",
        "start_state",
        "goal_state",
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return filename


def main():
    parser = argparse.ArgumentParser(description="Cost experiment using new System/Plan classes")
    parser.add_argument(
        "--planner-name",
        type=str,
        required=True,
        choices=["sststar", "aorrt", "aoest", "sst", "rrt"],
        help="Planner to use.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/planning/cost_experiment",
        help="Directory for CSV output files.",
    )
    args, remaining = parser.parse_known_args()

    # Keep compatibility with existing config parser behavior.
    sys.argv = [sys.argv[0], "--planner-name", args.planner_name] + remaining
    config = parse_args_and_config()

    run_number = get_next_run_number(args.results_dir, args.planner_name)
    seed = random.randint(1, 2147483647)
    ou.RNG.setSeed(seed)

    planning_times = np.arange(2.0, 10.5, 2.0)
    rows = run_experiment_for_planner(config, args.planner_name, planning_times, run_number)

    if rows is None:
        print(f"[FATAL] Planning failed for run {run_number}.")
        sys.exit(1)

    out_file = save_run_csv(args.results_dir, args.planner_name, run_number, rows)
    print(f"✅ Run {run_number} results saved to {out_file}")


if __name__ == "__main__":
    main()
