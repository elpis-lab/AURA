from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from systems import get_system, plan as PlanConfig
from simulators import Simulator, create_simulator
from utils.utils import arrayDistance, log, state2list


@dataclass
class AURAResult:
    success: bool
    num_controls: int
    final_state: np.ndarray
    cost: float
    tracking_error_mean: float
    tracking_error_list: list[float]
    trajectory: list[np.ndarray]


class AURA:
    """
    AURA execution loop built on top of:
      - systems.System subclasses
      - systems.plan planning wrapper
      - simulators.Simulator backends
    """

    def __init__(
        self,
        system_name: str,
        planner_name: str,
        config: dict,
        simulator_mode: str = "gaussian",
        simulator: Optional[Simulator] = None,
    ):
        self.config = config or {}
        self.system_name = system_name
        self.planner_name = planner_name
        self.system = get_system(system_name, object_name=self.config.get("objectName", "crackerBox"))
        self.simulator = (
            simulator
            if simulator is not None
            else create_simulator(system_name, simulator_mode, config=self.config)
        )

        self.start_state = np.asarray(self.config.get("start_state"), dtype=float)
        self.goal_state = np.asarray(self.config.get("goal_state"), dtype=float)

        self.goal_threshold = float(self.config.get("goal_threshold", 0.1))
        self.propagation_step_size = float(self.config.get("propagation_step_size", 1.0))
        self.min_control_duration = int(self.config.get("min_control_duration", 1))
        self.max_control_duration = int(self.config.get("max_control_duration", 5))
        self.initial_planning_time = float(self.config.get("planning_time", 10.0))
        self.replanning_time = float(self.config.get("replanning_time", -1.0))
        self.pruning_radius = float(self.config.get("pruning_radius", 0.1))
        self.max_replan_distance = float(
            self.config.get(
                "replanningMaxDistance",
                self.config.get("sampling_max_distance", 0.05),
            )
        )

    def _state_to_numpy(self, state_obj) -> np.ndarray:
        if state_obj is None:
            return None
        if hasattr(state_obj, "cpu"):
            arr = state_obj.detach().cpu().numpy()
        elif isinstance(state_obj, np.ndarray):
            arr = state_obj
        elif isinstance(state_obj, (list, tuple)):
            arr = np.array(state_obj, dtype=float)
        else:
            arr = np.array(state2list(state_obj, self.system_name), dtype=float)

        # Trim simulator extras (e.g., simple_car may return [x, y, yaw, v]).
        if arr.shape[0] > self.system.state_dim:
            arr = arr[: self.system.state_dim]
        return arr.astype(float)

    def _make_planner(self, start_state: np.ndarray, planning_time: float) -> PlanConfig:
        return PlanConfig(
            system=self.system,
            start_state=start_state,
            goal_state=self.goal_state,
            planner_name=self.planner_name,
            goal_threshold=self.goal_threshold,
            min_control_duration=self.min_control_duration,
            max_control_duration=self.max_control_duration,
            propagation_step_size=self.propagation_step_size,
            planning_time=planning_time,
            pruning_radius=self.pruning_radius,
            config=self.config,
            visualize=bool(self.config.get("visualize", False)),
        )

    def _safe_first_plan(self, solutions_info):
        if solutions_info is None or len(solutions_info) == 0:
            return None
        if "controls" not in solutions_info[0] or "states" not in solutions_info[0]:
            return None
        return solutions_info[0]

    def run(self, max_steps: int = 1000, reset_sim: bool = True) -> AURAResult:
        if reset_sim:
            self.simulator.reset()
            if self.system_name == "pushing":
                self.simulator.set_obj_init_pose(self.start_state.tolist())

        current_state = self._state_to_numpy(self.simulator.get_state())
        if current_state is None:
            current_state = self.start_state.copy()

        trajectory = [current_state.copy()]
        tracking_errors: list[float] = []

        planner = self._make_planner(current_state, planning_time=self.initial_planning_time)
        solutions_info, _ = planner.plan()

        step = 0
        while step < max_steps:
            best = self._safe_first_plan(solutions_info)
            if best is None or len(best["controls"]) == 0:
                log("[WARNING] No controls available in current plan, stopping AURA run.", "warning")
                break

            control = best["controls"][0]
            duration = (
                float(best["time"][0])
                if "time" in best and len(best["time"]) > 0
                else self.propagation_step_size
            )

            if len(best["states"]) > 1:
                next_planned = self._state_to_numpy(best["states"][1])
            else:
                next_planned = self.goal_state

            executed = self.simulator.execute_segment(control, duration)
            current_state = self._state_to_numpy(executed)
            if current_state is None:
                log("[ERROR] Simulator returned None state; aborting AURA run.", "error")
                break

            trajectory.append(current_state.copy())
            step_error = arrayDistance(current_state, next_planned, system=self.system_name)
            tracking_errors.append(float(step_error))

            dist_to_goal = arrayDistance(current_state, self.goal_state, system=self.system_name)
            if dist_to_goal < self.goal_threshold:
                break

            need_replan = step_error > self.max_replan_distance
            next_plan_time = self.replanning_time if need_replan else self.propagation_step_size
            if need_replan:
                log(
                    f"[WARNING] Step error {step_error:.6f} > max {self.max_replan_distance:.6f}; replanning from scratch.",
                    "warning",
                )
                self.simulator.stop()

            planner = self._make_planner(current_state, planning_time=next_plan_time)
            solutions_info, _ = planner.replan(current_state, planning_time=next_plan_time)
            step += 1

        cost = 0.0
        for i in range(len(trajectory) - 1):
            cost += arrayDistance(trajectory[i], trajectory[i + 1], system=self.system_name)

        final_state = trajectory[-1] if trajectory else current_state
        final_dist = arrayDistance(final_state, self.goal_state, system=self.system_name)
        success = final_dist < self.goal_threshold
        tracking_mean = float(np.mean(tracking_errors)) if len(tracking_errors) > 0 else float("inf")

        return AURAResult(
            success=success,
            num_controls=max(0, len(trajectory) - 1),
            final_state=np.asarray(final_state, dtype=float),
            cost=float(cost),
            tracking_error_mean=tracking_mean,
            tracking_error_list=tracking_errors,
            trajectory=[np.asarray(s, dtype=float) for s in trajectory],
        )
