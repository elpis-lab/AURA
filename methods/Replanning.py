"""Blocking restart-replanning baseline built on the shared OMPL planner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import time

import numpy as np
from ompl import base as ob

from methods.plan import OMPLPlanner
from simulation.pushing_model import get_pushing_model
from simulation.simulator import create_simulator
from systems import get_system
from train_model import load_opt_model_2
from utils.control_duration import duration_seconds_to_steps
from utils.utils import arrayDistance, state2list


@dataclass
class ReplanningResult:
    num_controls: int
    num_replanning: int
    final_state: np.ndarray
    cost: float
    tracking_error_mean: float
    tracking_error_list: list[float]
    trajectory: list[np.ndarray]
    primitive_trajectory: list[np.ndarray]
    controls_trajectory: list[np.ndarray]
    control_duration_steps_trajectory: list[int]
    control_duration_seconds_trajectory: list[float]
    planned_final_state: np.ndarray
    nominal_execution_seconds: float = 0.0
    actual_execution_seconds: float = 0.0
    blocking_replanning_seconds: float = 0.0
    compute_overrun_seconds: float = 0.0
    status: str = "success"
    failure_reason: str = ""


@dataclass
class ReplanningHistory:
    trajectory: list[np.ndarray]
    tracking_errors: list[float] = field(default_factory=list)
    controls: list[np.ndarray] = field(default_factory=list)
    duration_steps: list[int] = field(default_factory=list)
    duration_seconds: list[float] = field(default_factory=list)
    steps: int = 0
    num_replanning: int = 0
    nominal_execution_seconds: float = 0.0
    actual_execution_seconds: float = 0.0
    blocking_replanning_seconds: float = 0.0
    compute_overrun_seconds: float = 0.0
    last_execution_duration: float = 0.0
    failure_reason: str = ""


@dataclass
class ReplanningContext:
    solution: dict
    plan_index: int
    current_state: np.ndarray
    planned_final_state: np.ndarray
    history: ReplanningHistory


@dataclass(frozen=True)
class PlanSegment:
    control: np.ndarray
    planned_state: np.ndarray
    duration_seconds: float
    duration_steps: int


def state_to_numpy(
    state,
    system_name: str,
    state_dimension: int,
) -> np.ndarray | None:
    """Convert simulator, Torch, NumPy, or OMPL state objects to NumPy."""

    if state is None:
        return None
    if hasattr(state, "cpu"):
        values = state.detach().cpu().numpy()
    elif isinstance(state, np.ndarray):
        values = state
    elif isinstance(state, (list, tuple)):
        values = np.asarray(state, dtype=float)
    else:
        values = np.asarray(state2list(state, system_name), dtype=float)
    return np.asarray(values[:state_dimension], dtype=float)


class ReplanningRunner:
    """Execute controls and replace the plan whenever tracking diverges."""

    def __init__(
        self,
        system_name: str,
        planner_name: str,
        config: dict,
        simulator_mode: str,
        max_steps: int | None = 2000,
        initial_solution: dict | None = None,
        task_time_budget_seconds: float | None = None,
        plan_update_callback=None,
        system_override=None,
        simulator_override=None,
    ):
        self.system_name = system_name
        self.planner_name = planner_name
        self.config = config
        self.simulator_mode = simulator_mode
        self.max_steps = max_steps
        self.initial_solution = (
            deepcopy(initial_solution) if initial_solution is not None else None
        )
        self.task_time_budget_seconds = (
            None
            if task_time_budget_seconds is None
            else max(0.0, float(task_time_budget_seconds))
        )
        self.plan_update_callback = plan_update_callback
        self.system = (
            system_override if system_override is not None else get_system(system_name)
        )
        self.apply_control_bounds(config.get("control_bounds"))
        self.state_dimension = len(config["start_state"])
        self.simulator = (
            simulator_override
            if simulator_override is not None
            else create_simulator(
                system_name,
                simulator_mode,
                config=config,
            )
        )

    def apply_control_bounds(self, control_bounds) -> None:
        if not control_bounds:
            return
        if len(control_bounds) != len(self.system.control_bounds):
            raise ValueError(
                f"{self.system.name} expected {len(self.system.control_bounds)} "
                f"control bounds, got {len(control_bounds)}"
            )
        self.system.control_bounds = [
            (float(lower), float(upper)) for lower, upper in control_bounds
        ]
        bounds = ob.RealVectorBounds(len(self.system.control_bounds))
        for index, (lower, upper) in enumerate(self.system.control_bounds):
            bounds.setLow(index, lower)
            bounds.setHigh(index, upper)
        self.system.control_space.setBounds(bounds)

    def publish_plan_update(self, solution: dict) -> None:
        if self.plan_update_callback is None or solution is None:
            return
        try:
            self.plan_update_callback(deepcopy(solution))
        except Exception as error:
            print(f"[WARNING] Replanning visualization update failed: {error}")

    def build_planner(
        self,
        start_state: np.ndarray,
        planning_time: float,
    ) -> OMPLPlanner:
        planner = OMPLPlanner(
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
            optimization_objective=str(
                self.config.get("optimization_objective", "control_duration")
            ),
            single_solve=True,
        )
        planner.replanning_time = float(planning_time)
        if self.system.name == "pushing_object":
            planner.opt_model = load_opt_model_2(
                get_pushing_model(
                    self.system.object_shape,
                    model_name=getattr(
                        self.system,
                        "model_name",
                        "cracker_box_flipped",
                    ),
                    model_path=getattr(self.system, "model_path", None),
                )
            )
        else:
            planner.opt_model = None
        return planner

    def remaining_task_time(self, history: ReplanningHistory) -> float:
        if self.task_time_budget_seconds is None:
            return float("inf")
        return max(
            0.0,
            self.task_time_budget_seconds
            - history.nominal_execution_seconds
            - history.blocking_replanning_seconds,
        )

    def retry_plan_from_state(
        self,
        measured_state: np.ndarray,
        per_attempt_budget: float,
        history: ReplanningHistory,
    ) -> dict | None:
        """Retry fresh planning until success or task-time exhaustion."""

        history.num_replanning += 1
        planner = None
        first_attempt = True
        immediate_returns = 0
        step_size = float(self.config["propagation_step_size"])
        while self.remaining_task_time(history) >= step_size - 1e-9:
            budget = min(
                max(step_size, float(per_attempt_budget)),
                self.remaining_task_time(history),
            )
            time_before_attempt = self.remaining_task_time(history)
            started = time.monotonic()
            try:
                if planner is None:
                    planner = self.build_planner(measured_state, budget)
                if first_attempt:
                    solutions, _ = planner.plan()
                    first_attempt = False
                else:
                    solutions, _ = planner.retry_initial_plan(time_budget=budget)
            except Exception:
                solutions = []
                planner = None
                first_attempt = True
            elapsed = time.monotonic() - started
            history.blocking_replanning_seconds += min(time_before_attempt, elapsed)
            solution = self.first_executable_solution(solutions)
            if solution is not None:
                return solution

            immediate_threshold = min(0.05, max(0.005, 0.05 * budget))
            immediate_returns = (
                immediate_returns + 1 if elapsed < immediate_threshold else 0
            )
            if immediate_returns >= 3:
                return None
        return None

    @staticmethod
    def first_executable_solution(solutions) -> dict | None:
        return next(
            (
                solution
                for solution in (solutions or [])
                if solution.get("controls") and len(solution.get("states", [])) >= 2
            ),
            None,
        )

    def prepare_simulator(self, reset_sim: bool) -> np.ndarray:
        if reset_sim:
            self.simulator.reset()
        if self.system_name == "pushing_object":
            self.simulator.set_state(
                np.asarray(self.config["start_state"], dtype=float).tolist()
            )
        return state_to_numpy(
            self.simulator.get_state(),
            self.system_name,
            self.state_dimension,
        )

    def prepare_initial_solution(self, start_state: np.ndarray) -> dict:
        if self.initial_solution is None:
            planner = self.build_planner(
                start_state,
                float(self.config["planning_time"]),
            )
            solutions, _ = planner.plan()
            if not solutions:
                raise RuntimeError("Initial replanning baseline plan failed.")
            return solutions[0]

        solution = deepcopy(self.initial_solution)
        if solution.get("states"):
            states = [
                np.asarray(state, dtype=float).copy() for state in solution["states"]
            ]
            states[0] = start_state.copy()
            solution["states"] = states
        return solution

    def plan_segment(self, solution: dict, index: int) -> PlanSegment | None:
        controls = solution.get("controls", [])
        states = solution.get("states", [])
        if not controls or index >= len(controls) or len(states) < index + 2:
            return None
        step_size = float(self.config["propagation_step_size"])
        durations = solution.get("time", [])
        duration = float(durations[index]) if index < len(durations) else step_size
        return PlanSegment(
            control=np.asarray(controls[index], dtype=float),
            planned_state=state_to_numpy(
                states[index + 1],
                self.system_name,
                self.state_dimension,
            ),
            duration_seconds=duration,
            duration_steps=duration_seconds_to_steps(
                duration,
                step_size,
                min_steps=int(self.config["min_control_duration"]),
                max_steps=int(self.config["max_control_duration"]),
            ),
        )

    def execute_segment(
        self,
        context: ReplanningContext,
        segment: PlanSegment,
    ) -> float | None:
        started = time.monotonic()
        executed = self.simulator.execute_segment(
            segment.control,
            segment.duration_seconds,
        )
        elapsed = time.monotonic() - started
        history = context.history
        history.actual_execution_seconds += elapsed
        history.nominal_execution_seconds += segment.duration_seconds
        history.compute_overrun_seconds += max(0.0, elapsed - segment.duration_seconds)
        history.last_execution_duration = segment.duration_seconds
        current_state = state_to_numpy(
            executed,
            self.system_name,
            self.state_dimension,
        )
        if current_state is None:
            history.failure_reason = "simulator_state_unavailable"
            return None

        context.current_state = current_state
        history.trajectory.append(current_state.copy())
        history.controls.append(segment.control.copy())
        history.duration_steps.append(segment.duration_steps)
        history.duration_seconds.append(segment.duration_seconds)
        tracking_error = float(
            arrayDistance(
                current_state,
                segment.planned_state,
                system=self.system_name,
            )
        )
        history.tracking_errors.append(tracking_error)
        return tracking_error

    def refresh_plan(
        self,
        context: ReplanningContext,
        planning_budget: float,
    ) -> bool:
        self.simulator.stop()
        solution = self.retry_plan_from_state(
            context.current_state,
            planning_budget,
            context.history,
        )
        if solution is None:
            context.history.failure_reason = "task_time_limit_reached"
            return False
        context.solution = solution
        context.plan_index = 0
        context.planned_final_state = state_to_numpy(
            solution["states"][-1],
            self.system_name,
            self.state_dimension,
        )
        self.publish_plan_update(solution)
        return True

    def run_cycle(
        self,
        context: ReplanningContext,
        goal_state: np.ndarray,
        goal_threshold: float,
        replanning_max_distance: float,
    ) -> str:
        segment = self.plan_segment(context.solution, context.plan_index)
        if segment is None:
            refreshed = self.refresh_plan(
                context,
                context.history.last_execution_duration,
            )
            return "continue" if refreshed else "failure"
        if segment.duration_seconds > self.remaining_task_time(context.history) + 1e-9:
            context.history.failure_reason = "task_time_limit_reached"
            return "failure"

        tracking_error = self.execute_segment(context, segment)
        if tracking_error is None:
            return "failure"
        goal_distance = float(
            arrayDistance(
                context.current_state,
                goal_state,
                system=self.system_name,
            )
        )
        if goal_distance < goal_threshold:
            return "goal"

        controls = context.solution.get("controls", [])
        needs_replan = (
            tracking_error > replanning_max_distance
            or context.plan_index >= len(controls) - 1
        )
        if needs_replan:
            if not self.refresh_plan(context, segment.duration_seconds):
                return "failure"
        else:
            context.plan_index += 1
        context.history.steps += 1
        return "continue"

    def stationary_result(self, start_state: np.ndarray) -> ReplanningResult:
        state = np.asarray(start_state, dtype=float)
        return ReplanningResult(
            num_controls=0,
            num_replanning=0,
            final_state=state,
            cost=0.0,
            tracking_error_mean=0.0,
            tracking_error_list=[],
            trajectory=[state.copy()],
            primitive_trajectory=[state.copy()],
            controls_trajectory=[],
            control_duration_steps_trajectory=[],
            control_duration_seconds_trajectory=[],
            planned_final_state=state,
        )

    def build_result(
        self,
        context: ReplanningContext,
        goal_state: np.ndarray,
        goal_threshold: float,
    ) -> ReplanningResult:
        history = context.history
        goal_distance = float(
            arrayDistance(
                context.current_state,
                goal_state,
                system=self.system_name,
            )
        )
        if (
            self.max_steps is not None
            and history.steps >= self.max_steps
            and goal_distance >= goal_threshold
        ):
            history.failure_reason = history.failure_reason or "max_steps_reached"
        if goal_distance >= goal_threshold:
            history.failure_reason = history.failure_reason or (
                f"goal_not_reached:{goal_distance:.6g}"
            )
        cost = sum(
            arrayDistance(first, second, system=self.system_name)
            for first, second in zip(history.trajectory, history.trajectory[1:])
        )
        return ReplanningResult(
            num_controls=max(0, len(history.trajectory) - 1),
            num_replanning=history.num_replanning,
            final_state=np.asarray(history.trajectory[-1], dtype=float),
            cost=float(cost),
            tracking_error_mean=(
                float(np.mean(history.tracking_errors))
                if history.tracking_errors
                else float("inf")
            ),
            tracking_error_list=history.tracking_errors,
            trajectory=history.trajectory,
            primitive_trajectory=[
                np.asarray(state, dtype=float).copy()
                for state in getattr(
                    self.simulator,
                    "primitive_states",
                    history.trajectory,
                )
            ],
            controls_trajectory=history.controls,
            control_duration_steps_trajectory=history.duration_steps,
            control_duration_seconds_trajectory=history.duration_seconds,
            planned_final_state=context.planned_final_state,
            nominal_execution_seconds=history.nominal_execution_seconds,
            actual_execution_seconds=history.actual_execution_seconds,
            blocking_replanning_seconds=history.blocking_replanning_seconds,
            compute_overrun_seconds=history.compute_overrun_seconds,
            status="success" if not history.failure_reason else "failure",
            failure_reason=history.failure_reason,
        )

    def run(self, *, reset_sim: bool = True) -> ReplanningResult:
        """Execute the restart-replanning baseline until success or a limit."""

        start_state = self.prepare_simulator(reset_sim)
        goal_state = np.asarray(self.config["goal_state"], dtype=float)
        goal_threshold = float(
            self.config.get("actual_goal_threshold", self.config["goal_threshold"])
        )
        if (
            arrayDistance(start_state, goal_state, system=self.system_name)
            < goal_threshold
        ):
            return self.stationary_result(start_state)

        solution = self.prepare_initial_solution(start_state)
        self.publish_plan_update(solution)
        step_size = float(self.config["propagation_step_size"])
        context = ReplanningContext(
            solution=solution,
            plan_index=0,
            current_state=start_state.copy(),
            planned_final_state=state_to_numpy(
                solution["states"][-1],
                self.system_name,
                self.state_dimension,
            ),
            history=ReplanningHistory(
                trajectory=[start_state.copy()],
                last_execution_duration=step_size,
            ),
        )
        replanning_max_distance = float(
            self.config.get(
                "replanningMaxDistance",
                self.config.get(
                    "replanning_max_distance",
                    self.config.get("sampling_max_distance", 0.05),
                ),
            )
        )
        while self.max_steps is None or context.history.steps < self.max_steps:
            outcome = self.run_cycle(
                context,
                goal_state,
                goal_threshold,
                replanning_max_distance,
            )
            if outcome != "continue":
                break
        return self.build_result(context, goal_state, goal_threshold)
