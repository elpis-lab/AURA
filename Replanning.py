from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from ompl import base as ob

from plan import OMPL_Planner
from simulation.pushing_dynamics import get_pushing_model
from simulation.simulators import create_simulator
from systems import get_system
from train_model import load_opt_model_2
from utils.utils import arrayDistance, state2list


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


class ReplanningRunner:
    def __init__(
        self,
        system_name: str,
        planner_name: str,
        config: dict,
        simulator_mode: str,
        max_steps: int | None = 2000,
        initial_solution: dict | None = None,
        plan_update_callback=None,
        system_override=None,
        simulator_override=None,
    ):
        self.system_name = system_name
        self.planner_name = planner_name
        self.config = config
        self.simulator_mode = simulator_mode
        self.max_steps = max_steps
        self.initial_solution = deepcopy(initial_solution) if initial_solution is not None else None
        self.plan_update_callback = plan_update_callback

        self.system = system_override if system_override is not None else get_system(system_name)
        self._apply_control_bounds(config.get("control_bounds"))
        self.state_dim = int(len(config["start_state"]))
        self.simulator = (
            simulator_override
            if simulator_override is not None
            else create_simulator(system_name, simulator_mode, config=config)
        )

    def _apply_control_bounds(self, control_bounds) -> None:
        if not control_bounds:
            return
        if len(control_bounds) != len(self.system.control_bounds):
            raise ValueError(
                f"{self.system.name} expected {len(self.system.control_bounds)} control bounds, "
                f"got {len(control_bounds)}"
            )
        self.system.control_bounds = [(float(a), float(b)) for a, b in control_bounds]
        bounds = ob.RealVectorBounds(len(self.system.control_bounds))
        for i, (low, high) in enumerate(self.system.control_bounds):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        self.system.control_space.setBounds(bounds)

    def _publish_plan_update(self, solution: dict) -> None:
        if self.plan_update_callback is None or solution is None:
            return
        try:
            self.plan_update_callback(deepcopy(solution))
        except Exception as exc:
            print(f"[WARNING] Replanning plan visualization update failed: {exc}")

    def _build_planner(self, start_state: np.ndarray, planning_time: float) -> OMPL_Planner:
        planner = OMPL_Planner(
            system=self.system,
            start_state=np.asarray(start_state, dtype=float),
            goal_state=np.asarray(self.config["goal_state"], dtype=float),
            planner_method=self.planner_name,
            goal_threshold=float(
                self.config.get(
                    "planner_goal_threshold",
                    self.config["goal_threshold"],
                )
            ),
            min_max_control_duration=(
                int(self.config["min_control_duration"]),
                int(self.config["max_control_duration"]),
            ),
            propagation_step_size=float(self.config["propagation_step_size"]),
            initial_planning_time=float(planning_time),
            pruning_radius=float(self.config.get("pruning_radius", 0.1)),
            obstacle_config=self.config.get("obstacles"),
        )
        planner.replanning_time = float(
            self.config.get(
                "replanning_time_budget",
                self.config.get("control_duration_seconds", self.config.get("planning_time", planning_time)),
            )
        )
        if self.system.name == "pushing_object":
            planner.opt_model = load_opt_model_2(
                get_pushing_model(
                    self.system.object_shape,
                    model_name=getattr(self.system, "model_name", "cracker_box_flipped"),
                    model_path=getattr(self.system, "model_path", None),
                )
            )
        else:
            planner.opt_model = None
        return planner

    def run(self) -> ReplanningResult:
        self.simulator.reset()
        if self.system_name == "pushing_object":
            self.simulator.set_obj_init_pose(
                np.asarray(self.config["start_state"], dtype=float).tolist()
            )

        start_state = _state_to_numpy(self.simulator.get_state(), self.system_name, self.state_dim)
        goal_state = np.asarray(self.config["goal_state"], dtype=float)
        goal_threshold = float(
            self.config.get("actual_goal_threshold", self.config["goal_threshold"])
        )
        replanning_max_distance = float(
            self.config.get("replanningMaxDistance", self.config.get("sampling_max_distance", 0.05))
        )
        step_size = float(self.config["propagation_step_size"])
        start_goal_distance = float(
            arrayDistance(start_state, goal_state, system=self.system_name)
        )
        if start_goal_distance < goal_threshold:
            return ReplanningResult(
                num_controls=0,
                final_state=np.asarray(start_state, dtype=float),
                cost=0.0,
                tracking_error_mean=0.0,
                tracking_error_list=[],
                trajectory=[np.asarray(start_state, dtype=float).copy()],
                planned_final_state=np.asarray(start_state, dtype=float),
            )

        if self.initial_solution is not None:
            current_solution = deepcopy(self.initial_solution)
            if current_solution.get("states"):
                states = [np.asarray(s, dtype=float).copy() for s in current_solution["states"]]
                states[0] = start_state.copy()
                current_solution["states"] = states
            print(
                "[replanning] using shared initial solution "
                f"({len(current_solution.get('controls', []))} controls)"
            )
        else:
            planner = self._build_planner(start_state, float(self.config["planning_time"]))
            solutions, _ = planner.plan()
            if solutions is None or len(solutions) == 0:
                raise RuntimeError("Initial replanning baseline plan failed.")
            current_solution = solutions[0]
        current_plan_idx = 0
        self._publish_plan_update(current_solution)
        last_planned_final = _state_to_numpy(
            current_solution["states"][-1], self.system_name, self.state_dim
        )

        current_state = start_state.copy()
        trajectory = [current_state.copy()]
        tracking_errors: list[float] = []
        steps = 0

        while self.max_steps is None or steps < self.max_steps:
            controls = current_solution.get("controls", [])
            states = current_solution.get("states", [])
            times = current_solution.get("time", [])

            if len(controls) == 0 or current_plan_idx >= len(controls) or len(states) < 2:
                self.simulator.stop()
                planner = self._build_planner(current_state, -1.0)
                solutions, _ = planner.plan()
                if (
                    solutions is None
                    or len(solutions) == 0
                    or len(solutions[0].get("controls", [])) == 0
                ):
                    break
                current_solution = solutions[0]
                current_plan_idx = 0
                self._publish_plan_update(current_solution)
                last_planned_final = _state_to_numpy(
                    current_solution["states"][-1], self.system_name, self.state_dim
                )
                continue

            control = controls[current_plan_idx]
            next_planned = _state_to_numpy(
                states[current_plan_idx + 1], self.system_name, self.state_dim
            )
            duration = float(times[current_plan_idx]) if current_plan_idx < len(times) else step_size

            executed = self.simulator.execute_segment(control, duration)
            current_state = _state_to_numpy(executed, self.system_name, self.state_dim)
            if current_state is None:
                break

            trajectory.append(current_state.copy())
            step_err = float(arrayDistance(current_state, next_planned, system=self.system_name))
            tracking_errors.append(step_err)

            dist_to_goal = float(arrayDistance(current_state, goal_state, system=self.system_name))
            if dist_to_goal < goal_threshold:
                break

            if step_err > replanning_max_distance:
                self.simulator.stop()
                planner = self._build_planner(current_state, -1.0)
                solutions, _ = planner.plan()
                if (
                    solutions is None
                    or len(solutions) == 0
                    or len(solutions[0].get("controls", [])) == 0
                ):
                    break
                current_solution = solutions[0]
                current_plan_idx = 0
                self._publish_plan_update(current_solution)
                last_planned_final = _state_to_numpy(
                    current_solution["states"][-1], self.system_name, self.state_dim
                )
            elif current_plan_idx >= len(controls) - 1:
                self.simulator.stop()
                planner = self._build_planner(current_state, -1.0)
                solutions, _ = planner.plan()
                if (
                    solutions is None
                    or len(solutions) == 0
                    or len(solutions[0].get("controls", [])) == 0
                ):
                    break
                current_solution = solutions[0]
                current_plan_idx = 0
                self._publish_plan_update(current_solution)
                last_planned_final = _state_to_numpy(
                    current_solution["states"][-1], self.system_name, self.state_dim
                )
            else:
                current_plan_idx += 1

            steps += 1

        final_goal_distance = float(
            arrayDistance(current_state, goal_state, system=self.system_name)
        )
        if (
            self.max_steps is not None
            and steps >= self.max_steps
            and final_goal_distance >= goal_threshold
        ):
            print(
                "[WARNING] Replanning hit max_steps before reaching the goal: "
                f"distance {final_goal_distance:.6f} > threshold {goal_threshold:.6f}. "
                "Increase replanning_min_steps_to_goal or --max-steps if this persists."
            )

        cost = 0.0
        for i in range(len(trajectory) - 1):
            cost += arrayDistance(trajectory[i], trajectory[i + 1], system=self.system_name)

        return ReplanningResult(
            num_controls=max(0, len(trajectory) - 1),
            final_state=np.asarray(trajectory[-1], dtype=float),
            cost=float(cost),
            tracking_error_mean=float(np.mean(tracking_errors)) if tracking_errors else float("inf"),
            tracking_error_list=tracking_errors,
            trajectory=trajectory,
            planned_final_state=last_planned_final,
        )
