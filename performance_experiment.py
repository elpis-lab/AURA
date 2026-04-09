#!/usr/bin/env python3

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass

import numpy as np

from AURA import AURA
from simulators import create_simulator
from systems import get_system, plan as PlanSession
from utils.configHandler import load_and_normalize_config
from utils.utils import arrayDistance, log, state2list


@dataclass
class ReplanningResult:
    num_controls: int
    final_state: np.ndarray
    cost: float
    tracking_error_mean: float
    tracking_error_list: list[float]
    trajectory: list[np.ndarray]
    planned_final_state: np.ndarray


def _state_to_numpy(state_obj, system_name: str, state_dim: int) -> np.ndarray:
    if state_obj is None:
        return None
    if hasattr(state_obj, "cpu"):
        arr = state_obj.detach().cpu().numpy()
    elif isinstance(state_obj, np.ndarray):
        arr = state_obj
    elif isinstance(state_obj, (list, tuple)):
        arr = np.array(state_obj, dtype=float)
    else:
        arr = np.array(state2list(state_obj, system_name), dtype=float)
    if arr.shape[0] > state_dim:
        arr = arr[:state_dim]
    return arr.astype(float)


def run_replanning(
    system_name: str,
    planner_name: str,
    config: dict,
    simulator_mode: str,
    max_steps: int = 2000,
) -> ReplanningResult:
    system = get_system(system_name, object_name=config.get("objectName", "crackerBox"))
    simulator = create_simulator(system_name, simulator_mode, config=config)

    simulator.reset()
    if system_name == "pushing_object":
        simulator.set_obj_init_pose(np.asarray(config["start_state"], dtype=float).tolist())

    start_state = _state_to_numpy(simulator.get_state(), system_name, system.state_dim)
    goal_state = np.asarray(config["goal_state"], dtype=float)
    goal_threshold = float(config["goal_threshold"])
    replanning_max_distance = float(
        config.get("replanningMaxDistance", config.get("sampling_max_distance", 0.05))
    )
    step_size = float(config["propagation_step_size"])

    def make_planner(start, planning_time):
        return PlanSession(
            system=system,
            start_state=start,
            goal_state=goal_state,
            planner_name=planner_name,
            goal_threshold=goal_threshold,
            min_control_duration=int(config["min_control_duration"]),
            max_control_duration=int(config["max_control_duration"]),
            propagation_step_size=step_size,
            planning_time=planning_time,
            pruning_radius=float(config["pruning_radius"]),
            config=config,
            visualize=bool(config.get("visualize", False)),
        )

    planner = make_planner(start_state, float(config["planning_time"]))
    solutions, _ = planner.plan()
    if solutions is None or len(solutions) == 0:
        raise RuntimeError("Initial replanning baseline plan failed.")

    current_solution = solutions[0]
    current_plan_idx = 0
    last_planned_final = _state_to_numpy(
        current_solution["states"][-1], system_name, system.state_dim
    )

    current_state = start_state.copy()
    trajectory = [current_state.copy()]
    tracking_errors: list[float] = []
    steps = 0

    while steps < max_steps:
        controls = current_solution.get("controls", [])
        states = current_solution.get("states", [])
        times = current_solution.get("time", [])

        if len(controls) == 0 or current_plan_idx >= len(controls) or len(states) < 2:
            # No controls left and goal not reached -> exact replanning from current state.
            simulator.stop()
            planner = make_planner(current_state, -1.0)
            solutions, _ = planner.replan(current_state, planning_time=-1.0)
            if solutions is None or len(solutions) == 0 or len(solutions[0].get("controls", [])) == 0:
                break
            current_solution = solutions[0]
            current_plan_idx = 0
            last_planned_final = _state_to_numpy(
                current_solution["states"][-1], system_name, system.state_dim
            )
            continue

        control = controls[current_plan_idx]
        next_planned = _state_to_numpy(states[current_plan_idx + 1], system_name, system.state_dim)
        duration = float(times[current_plan_idx]) if current_plan_idx < len(times) else step_size

        executed = simulator.execute_segment(control, duration)
        current_state = _state_to_numpy(executed, system_name, system.state_dim)
        if current_state is None:
            break

        trajectory.append(current_state.copy())
        step_err = float(arrayDistance(current_state, next_planned, system=system_name))
        tracking_errors.append(step_err)

        dist_to_goal = float(arrayDistance(current_state, goal_state, system=system_name))
        if dist_to_goal < goal_threshold:
            break

        if step_err > replanning_max_distance:
            simulator.stop()
            planner = make_planner(current_state, -1.0)
            solutions, _ = planner.replan(current_state, planning_time=-1.0)
            if solutions is None or len(solutions) == 0 or len(solutions[0].get("controls", [])) == 0:
                break
            current_solution = solutions[0]
            current_plan_idx = 0
            last_planned_final = _state_to_numpy(
                current_solution["states"][-1], system_name, system.state_dim
            )
        else:
            current_plan_idx += 1

        steps += 1

    cost = 0.0
    for i in range(len(trajectory) - 1):
        cost += arrayDistance(trajectory[i], trajectory[i + 1], system=system_name)

    return ReplanningResult(
        num_controls=max(0, len(trajectory) - 1),
        final_state=np.asarray(trajectory[-1], dtype=float),
        cost=float(cost),
        tracking_error_mean=float(np.mean(tracking_errors)) if tracking_errors else float("inf"),
        tracking_error_list=tracking_errors,
        trajectory=trajectory,
        planned_final_state=last_planned_final,
    )


def write_csv(
    csv_filename: str,
    run_number: int,
    planner_name: str,
    system_name: str,
    aura_result,
    replanning_result: ReplanningResult,
    planned_final_state: np.ndarray,
    aura_time: float,
    replanning_time: float,
):
    if system_name == "double_integrator":
        header = "run_number,planner,method,time,cost,tracking_error,actual_final_0,actual_final_1,actual_final_2,actual_final_3,actual_final_4,actual_final_5,planned_final_0,planned_final_1,planned_final_2,planned_final_3,planned_final_4,planned_final_5,num_controls\n"
        aura_row = (
            f"{run_number},{planner_name},aura,{aura_time:.6f},{aura_result.cost:.6f},{aura_result.tracking_error_mean:.6f},"
            + ",".join([f"{x:.6f}" for x in aura_result.final_state[:6]])
            + ","
            + ",".join([f"{x:.6f}" for x in planned_final_state[:6]])
            + f",{aura_result.num_controls}\n"
        )
        replan_row = (
            f"{run_number},{planner_name},replanning,{replanning_time:.6f},{replanning_result.cost:.6f},{replanning_result.tracking_error_mean:.6f},"
            + ",".join([f"{x:.6f}" for x in replanning_result.final_state[:6]])
            + ","
            + ",".join([f"{x:.6f}" for x in replanning_result.planned_final_state[:6]])
            + f",{replanning_result.num_controls}\n"
        )
    else:
        header = "run_number,planner,method,time,cost,tracking_error,actual_final_x,actual_final_y,actual_final_theta,planned_final_x,planned_final_y,planned_final_theta,num_controls\n"
        aura_row = (
            f"{run_number},{planner_name},aura,{aura_time:.6f},{aura_result.cost:.6f},{aura_result.tracking_error_mean:.6f},"
            f"{aura_result.final_state[0]:.6f},{aura_result.final_state[1]:.6f},{aura_result.final_state[2]:.6f},"
            f"{planned_final_state[0]:.6f},{planned_final_state[1]:.6f},{planned_final_state[2]:.6f},"
            f"{aura_result.num_controls}\n"
        )
        replan_row = (
            f"{run_number},{planner_name},replanning,{replanning_time:.6f},{replanning_result.cost:.6f},{replanning_result.tracking_error_mean:.6f},"
            f"{replanning_result.final_state[0]:.6f},{replanning_result.final_state[1]:.6f},{replanning_result.final_state[2]:.6f},"
            f"{replanning_result.planned_final_state[0]:.6f},{replanning_result.planned_final_state[1]:.6f},{replanning_result.planned_final_state[2]:.6f},"
            f"{replanning_result.num_controls}\n"
        )

    temp = csv_filename + ".tmp"
    with open(temp, "w") as f:
        f.write(header)
        f.write(aura_row)
        f.write(replan_row)
        f.flush()
    os.rename(temp, csv_filename)
    try:
        os.sync()
    except AttributeError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Performance experiment using new classes")
    parser.add_argument("planner_name", type=str)
    parser.add_argument(
        "system_name",
        type=str,
        choices=["kinematic_car", "pushing_object", "double_integrator"],
    )
    parser.add_argument("run_number", type=int)
    parser.add_argument(
        "--simulator-mode",
        type=str,
        default="gaussian",
        choices=["gaussian", "mujoco"],
        help="Simulation backend for execution.",
    )
    args = parser.parse_args()

    system_to_config = {
        "kinematic_car": "configs/car.yaml",
        "pushing_object": "configs/pushing.yaml",
        "double_integrator": "configs/double_integrator.yaml",
    }
    config = load_and_normalize_config(
        system_to_config[args.system_name],
        system_name=args.system_name,
        planner_name=args.planner_name,
    )

    if args.simulator_mode == "mujoco":
        system_to_folder = {
            "kinematic_car": "wholeTime_car",
            "pushing_object": "wholeTime_pushing",
            "double_integrator": "wholeTime_double",
        }
    else:
        system_to_folder = {
            "kinematic_car": "wholeTime_car_gaussian",
            "pushing_object": "wholeTime_pushing_gaussian",
            "double_integrator": "wholeTime_double_gaussian",
        }

    results_dir = f"results/planning/{system_to_folder[args.system_name]}"
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = f"{results_dir}/{args.system_name}_{args.planner_name}_{args.run_number:02d}.csv"

    def timeout_handler(signum, frame):
        raise KeyboardInterrupt("Timeout signal received")

    signal.signal(signal.SIGTERM, timeout_handler)

    # Use a single initial plan summary for planned_final_state reference in output.
    system = get_system(args.system_name, object_name=config.get("objectName", "crackerBox"))
    initial_plan = PlanSession(
        system=system,
        start_state=np.asarray(config["start_state"], dtype=float),
        goal_state=np.asarray(config["goal_state"], dtype=float),
        planner_name=args.planner_name,
        goal_threshold=float(config["goal_threshold"]),
        min_control_duration=int(config["min_control_duration"]),
        max_control_duration=int(config["max_control_duration"]),
        propagation_step_size=float(config["propagation_step_size"]),
        planning_time=float(config["planning_time"]),
        pruning_radius=float(config["pruning_radius"]),
        config=config,
        visualize=bool(config.get("visualize", False)),
    )
    solutions, _ = initial_plan.plan()
    if solutions is None or len(solutions) == 0:
        log("[ERROR] Initial plan failed.", "error")
        sys.exit(1)
    planned_final_state = _state_to_numpy(
        solutions[0]["states"][-1], args.system_name, system.state_dim
    )

    # Aura
    aura_runner = AURA(
        system_name=args.system_name,
        planner_name=args.planner_name,
        config=config,
        simulator_mode=args.simulator_mode,
    )
    t0 = time.time()
    aura_result = aura_runner.run()
    aura_time = time.time() - t0
    print(
        f"[DEBUG] Aura tracking error list ({len(aura_result.tracking_error_list)}): {aura_result.tracking_error_list}"
    )

    # Replanning baseline
    t1 = time.time()
    replanning_result = run_replanning(
        args.system_name, args.planner_name, config, simulator_mode=args.simulator_mode
    )
    replanning_time = time.time() - t1
    print(
        f"[DEBUG] Replanning tracking error list ({len(replanning_result.tracking_error_list)}): {replanning_result.tracking_error_list}"
    )

    write_csv(
        csv_filename=csv_filename,
        run_number=args.run_number,
        planner_name=args.planner_name,
        system_name=args.system_name,
        aura_result=aura_result,
        replanning_result=replanning_result,
        planned_final_state=planned_final_state,
        aura_time=aura_time,
        replanning_time=replanning_time,
    )
    print(f"📁 Results saved to {csv_filename}")


if __name__ == "__main__":
    main()
