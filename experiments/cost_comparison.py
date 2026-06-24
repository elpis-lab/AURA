#!/usr/bin/env python3
"""Compare planner solution costs across planning-time budgets."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml
from ompl import base as ob
from ompl import util as ou

from plan import OMPL_Planner
from systems import get_system

PLANNERS = ("aorrt", "aoest", "sststar")
PLANNER_ALIASES = {"sst": "sststar", "sst*": "sststar"}

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
    "planning_times": [2.0, 4.0, 6.0, 8.0, 10.0],
    "num_runs": 1,
    "results_dir": "results/planning/cost_comparison",
}


def _normalize_planner(name: str) -> str:
    planner = PLANNER_ALIASES.get(str(name).lower(), str(name).lower())
    if planner not in PLANNERS:
        raise ValueError(f"planner must be one of {PLANNERS} or all, got {name!r}")
    return planner


def _config_value(config: dict, *names: str, default=None):
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    return default


def load_config(path: str | None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config file must contain a mapping: {path}")
        config.update(loaded)

    config["system_name"] = str(
        _config_value(config, "system_name", "system", default=DEFAULT_CONFIG["system_name"])
    )
    config["start_state"] = np.asarray(
        _config_value(config, "start_state", "startState", default=DEFAULT_CONFIG["start_state"]),
        dtype=float,
    )
    config["goal_state"] = np.asarray(
        _config_value(config, "goal_state", "goalState", default=DEFAULT_CONFIG["goal_state"]),
        dtype=float,
    )
    config["state_bounds"] = [
        (float(low), float(high))
        for low, high in _config_value(
            config,
            "state_bounds",
            "stateBounds",
            default=DEFAULT_CONFIG["state_bounds"],
        )
    ]
    config["goal_threshold"] = float(_config_value(config, "goal_threshold", default=0.1))
    config["min_control_duration"] = int(
        _config_value(config, "min_control_duration", default=1)
    )
    config["max_control_duration"] = int(
        _config_value(config, "max_control_duration", default=1)
    )
    config["propagation_step_size"] = float(
        _config_value(config, "propagation_step_size", default=1.0)
    )
    config["pruning_radius"] = float(_config_value(config, "pruning_radius", default=0.1))
    config["planning_times"] = [
        float(x)
        for x in _config_value(config, "planning_times", "planningTimes", default=[2, 4, 6, 8, 10])
    ]
    config["num_runs"] = int(_config_value(config, "num_runs", "numRuns", default=1))
    config["results_dir"] = str(
        _config_value(config, "results_dir", "resultsDir", default=DEFAULT_CONFIG["results_dir"])
    )
    return config


def apply_state_bounds(system, state_bounds: list[tuple[float, float]]) -> None:
    if not state_bounds:
        return
    if len(state_bounds) != len(system.state_bounds):
        raise ValueError(
            f"{system.name} expects {len(system.state_bounds)} state bounds, got {len(state_bounds)}"
        )
    system.state_bounds = [(float(low), float(high)) for low, high in state_bounds]
    bounds = ob.RealVectorBounds(len(system.state_bounds))
    for i, (low, high) in enumerate(system.state_bounds):
        bounds.setLow(i, float(low))
        bounds.setHigh(i, float(high))
    system.state_space.setBounds(bounds)


def build_planner(system, config: dict, planner_name: str, planning_time: float) -> OMPL_Planner:
    planner = OMPL_Planner(
        system=system,
        start_state=np.asarray(config["start_state"], dtype=float),
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
    planner._single_solve = True
    return planner


def best_solution_metrics(solutions: list[dict]) -> dict[str, Any]:
    if not solutions:
        return {
            "status": "failure",
            "cost": float("nan"),
            "ompl_cost": float("nan"),
            "control_count": 0,
            "state_count": 0,
            "goal_distance": float("nan"),
        }

    def path_cost(solution: dict) -> float:
        times = solution.get("time") or []
        if times:
            return float(np.sum(np.asarray(times, dtype=float)))
        return float(solution.get("control_count", 0))

    best = min(solutions, key=path_cost)
    return {
        "status": "success",
        "cost": path_cost(best),
        "ompl_cost": float(best.get("cost", float("nan"))),
        "control_count": int(best["control_count"]),
        "state_count": int(best["state_count"]),
        "goal_distance": float(best["goal_distance"]),
    }


def run_once(
    config: dict,
    planner_name: str,
    planning_time: float,
    run_number: int,
    seed: int,
) -> dict[str, Any]:
    np.random.seed(int(seed))
    ou.RNG.setSeed(int(seed))

    system = get_system(config["system_name"])
    apply_state_bounds(system, config["state_bounds"])
    planner = build_planner(system, config, planner_name, planning_time)

    started_at = time.time()
    try:
        solutions, _ = planner.plan()
        row = best_solution_metrics(solutions)
    except Exception as exc:
        row = {
            "status": "failure",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "cost": float("nan"),
            "ompl_cost": float("nan"),
            "control_count": 0,
            "state_count": 0,
            "goal_distance": float("nan"),
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
            "system": str(config["system_name"]),
            "start_state": np.asarray(config["start_state"], dtype=float).tolist(),
            "goal_state": np.asarray(config["goal_state"], dtype=float).tolist(),
        }
    )
    return row


def output_csv_path(results_dir: str, system_name: str, planner_name: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"{system_name}_{planner_name}_cost_comparison.csv")


def append_rows(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "run_number",
        "planner",
        "planning_time",
        "planning_wall_time",
        "status",
        "failure_reason",
        "cost",
        "ompl_cost",
        "control_count",
        "state_count",
        "goal_distance",
        "seed",
        "system",
        "start_state",
        "goal_state",
    ]
    existing = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not existing:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config with system, bounds, timing, and output defaults.",
    )
    parser.add_argument(
        "--planner-name",
        default="all",
        help="Planner to run: aorrt, aoest, sststar, sst, or all.",
    )
    parser.add_argument(
        "--planning-times",
        type=float,
        nargs="+",
        default=None,
        help="Planning-time budgets to compare.",
    )
    parser.add_argument("--num-runs", type=int, default=None, help="Number of repeats.")
    parser.add_argument("--run-number", type=int, default=None, help="Run one repeat number.")
    parser.add_argument("--results-dir", default=None, help="Output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Print requested runs only.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.planning_times is not None:
        config["planning_times"] = [float(x) for x in args.planning_times]
    if args.num_runs is not None:
        config["num_runs"] = int(args.num_runs)
    if args.results_dir is not None:
        config["results_dir"] = str(args.results_dir)

    if args.planner_name.lower() == "all":
        planners = list(PLANNERS)
    else:
        planners = [_normalize_planner(args.planner_name)]

    if args.run_number is not None:
        run_numbers = [int(args.run_number)]
    else:
        run_numbers = list(range(1, int(config["num_runs"]) + 1))

    if not config["planning_times"]:
        raise ValueError("planning_times must contain at least one value")
    if not run_numbers:
        raise ValueError("num_runs must be positive")

    base_seed = int(args.seed if args.seed is not None else random.randint(1, 2_147_483_647))
    print("Cost comparison experiment")
    print(f"  system:         {config['system_name']}")
    print(f"  planners:       {' '.join(planners)}")
    print(f"  planning times: {' '.join(str(x) for x in config['planning_times'])}")
    print(f"  runs:           {' '.join(str(x) for x in run_numbers)}")
    print(f"  results dir:    {config['results_dir']}")

    if args.dry_run:
        return

    failures = 0
    for planner_index, planner_name in enumerate(planners):
        rows = []
        for run_number in run_numbers:
            for planning_time in config["planning_times"]:
                seed = base_seed + planner_index * 100_000 + int(run_number) * 1_000 + int(
                    round(float(planning_time) * 100)
                )
                row = run_once(config, planner_name, float(planning_time), int(run_number), seed)
                rows.append(row)
                if row["status"] == "success":
                    print(
                        f"[{planner_name}][run {run_number:02d}][{float(planning_time):.2f}s] "
                        f"cost={float(row['cost']):.6f} controls={row['control_count']} "
                        f"wall={float(row['planning_wall_time']):.3f}s"
                    )
                else:
                    failures += 1
                    print(
                        f"[{planner_name}][run {run_number:02d}][{float(planning_time):.2f}s] "
                        f"failed: {row['failure_reason']}"
                    )

        csv_path = output_csv_path(
            str(config["results_dir"]),
            str(config["system_name"]),
            planner_name,
        )
        append_rows(csv_path, rows)
        print(f"[save] {csv_path}")

    if failures:
        print(f"[warning] Completed with {failures} failed planning run(s).")


if __name__ == "__main__":
    main()
