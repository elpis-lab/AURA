from __future__ import annotations

from dataclasses import dataclass
import os
import time
import traceback
from typing import Callable, Optional

import threading
import numpy as np
import torch

from utils import auraHandler
from propagators import (
    System,
    double_integrator,
    dubins_airplane,
    kinematic_car,
    pushing_object,
)
from simulation.pushing_model import get_pushing_model
from aura.optimization import optimize_controls, warmup_optimizer_device
from methods.plan import OMPLPlanner
from utils.childrenHandler import getChildEdges
from utils.control_duration import (
    ControlEdge,
    ControlSelection,
)
from simulation.simulator import Simulator
from utils.utils import (
    arrayDistance,
    is_state_array_valid,
    sample_control_curve,
)


class AURA:
    """
    AURA execution loop built on top of:
      - propagators.System subclasses
      - methods.plan.OMPLPlanner
      - simulation.simulator.Simulator backends
    """

    @dataclass
    class AURAResult:
        num_controls: int
        num_replanning: int
        final_state: np.ndarray
        cost: float
        tracking_error_mean: float
        tracking_error_list: list[float]
        controls_trajectory: list[np.ndarray]
        control_duration_steps_trajectory: list[int]
        control_duration_seconds_trajectory: list[float]
        states_trajectory: list[np.ndarray]
        dense_states_trajectory: list[np.ndarray]
        cycle_timing: list[dict]
        nominal_execution_seconds: float
        actual_execution_seconds: float
        online_replanning_seconds: float
        optimizer_seconds: float
        compute_overrun_seconds: float
        final_plan_states: list[np.ndarray] | None = None
        final_plan_controls: list[np.ndarray] | None = None
        final_planned_state: np.ndarray | None = None
        status: str = "success"
        failure_reason: str = ""

    @dataclass
    class PlanUpdate:
        trajectory: dict | None
        optimization_result: dict | None
        continuity: dict
        candidate_paths: list[dict]
        goal_solution_paths: list[dict]
        caption: str
        additional_replanning: int = 0
        failure_reason: str = ""

    @dataclass
    class ControlUpdate:
        trajectory: dict
        selection: ControlSelection
        continuity: dict
        candidate_paths: list[dict]
        goal_solution_paths: list[dict]
        caption: str
        failure_reason: str = ""

    @dataclass
    class RunHistory:
        actual_states: list[np.ndarray]
        dense_states: list[np.ndarray]
        controls: list[np.ndarray]
        duration_steps: list[int]
        duration_seconds: list[float]
        tracking_errors: list[float]
        cycle_timing: list[dict]
        num_replanning: int = 0

    def __init__(
        self,
        system: System,
        planner: OMPLPlanner,
        simulator: Simulator,
    ):
        self.system = system
        self.planner = planner
        self.simulator = simulator

        self.start_state = planner.start_state
        self.goal_state = planner.goal_state
        self.goal_threshold = planner.goal_threshold
        self.propagation_step_size = planner.propagation_step_size
        self.initial_planning_time = planner.initial_planning_time
        self.replanning_time = getattr(planner, "replanning_time", planner.propagation_step_size)
        self.pruning_radius = planner.pruning_radius
        self.opt_model = getattr(planner, "opt_model", None)
        self.optimizer_num_states = int(getattr(planner, "optimizer_num_states", 1000))
        self.optimizer_num_steps = int(getattr(planner, "optimizer_num_steps", 25))
        self.optimizer_learning_rate = float(getattr(planner, "optimizer_learning_rate", 0.05))
        self.optimizer_pos_std = float(getattr(planner, "optimizer_pos_std", 0.003))
        self.optimizer_rot_std = float(getattr(planner, "optimizer_rot_std", 0.05))
        self.optimizer_vel_std = getattr(planner, "optimizer_vel_std", None)
        self.optimizer_max_children = int(getattr(planner, "optimizer_max_children", 0))
        self.optimizer_device = getattr(planner, "optimizer_device", None)

        self.execution_thread_result = {"result": None, "completed": False, "error": None}
        self.optimization_thread_result = {"result": None, "completed": False, "error": None}
        self.replanning_thread_result = {"result": None, "completed": False, "error": None}
        self.last_control_decision: dict = {}
        self._timing_lock = threading.Lock()
        self._timing_totals = {
            "execution": 0.0,
            "replanning": 0.0,
            "optimization": 0.0,
        }

    def record_timing(self, category: str, elapsed: float) -> None:
        with self._timing_lock:
            self._timing_totals[category] = (
                float(self._timing_totals.get(category, 0.0)) + float(elapsed)
            )

    def reset_thread_results(self) -> None:
        self.execution_thread_result = {
            "result": None,
            "completed": False,
            "error": None,
        }
        self.optimization_thread_result = {
            "result": None,
            "partial_result": None,
            "completed": False,
            "error": None,
        }
        self.replanning_thread_result = {
            "result": None,
            "completed": False,
            "error": None,
        }

    def append_executed_primitives(
        self,
        dense_states: list[np.ndarray],
        primitive_count_before: int,
        predicted_curve: list[np.ndarray],
        executed_endpoint,
    ) -> None:
        primitive_history = getattr(self.simulator, "primitive_states", None)
        if (
            primitive_history is not None
            and len(primitive_history) > primitive_count_before
        ):
            samples = primitive_history[primitive_count_before:]
        else:
            samples = predicted_curve
        dense_states.extend(
            np.asarray(sample, dtype=float).copy() for sample in samples
        )
        endpoint = np.asarray(executed_endpoint, dtype=float).copy()
        if not dense_states or arrayDistance(
            dense_states[-1], endpoint, system=self.system.name
        ) > 1e-12:
            dense_states.append(endpoint)

    def fresh_recovery_plan(
        self,
        measured_state: np.ndarray,
        time_budget: float,
    ) -> tuple[dict | None, str, float]:
        """Run one bounded exact solve from the measured state."""
        started = time.monotonic()
        try:
            recovery_solutions, _ = self.planner.plan_from_state(
                np.asarray(measured_state, dtype=float),
                time_budget=float(time_budget),
            )
            error = ""
        except Exception as exc:
            recovery_solutions = []
            error = repr(exc)
        elapsed = time.monotonic() - started
        self.record_timing("replanning", elapsed)
        solution = next(
            (
                candidate
                for candidate in (recovery_solutions or [])
                if candidate.get("controls")
                and len(candidate.get("states", [])) >= 2
            ),
            None,
        )
        if solution is None:
            return None, error or "no_exact_solution", float(elapsed)
        aligned = auraHandler.align_plan_root_to_pose(
            solution,
            np.asarray(measured_state, dtype=float),
        )
        aligned["_selection_source"] = "recovery fresh plan"
        aligned["_trim_index"] = 0
        aligned["_continuity_dist"] = 0.0
        self.planner.solutions = [aligned]
        return aligned, "", float(elapsed)

    def optimizer_edges(
        self,
        trajectory: dict,
        next_nominal_state: np.ndarray,
        execution_duration: float,
    ) -> list[ControlEdge]:
        """Return the bounded branch set optimized during one control cycle."""
        edges, _ = getChildEdges(
            self.planner.setup,
            next_nominal_state,
            system=self.system.name,
        )
        if (
            not edges
            and len(trajectory.get("states", [])) > 2
            and len(trajectory.get("controls", [])) > 1
        ):
            fallback_duration = auraHandler.plan_control_duration(
                trajectory,
                1,
                fallback=execution_duration,
                default_duration=self.planner.propagation_step_size,
            )
            edges = [
                ControlEdge(
                    source_state=np.asarray(trajectory["states"][1], dtype=float),
                    target_state=np.asarray(trajectory["states"][2], dtype=float),
                    control=np.asarray(trajectory["controls"][1], dtype=float),
                    duration_steps=self.planner.duration_seconds_to_steps(
                        fallback_duration
                    ),
                    duration_seconds=float(fallback_duration),
                    edge_id="solution-path:1",
                )
            ]
        if not edges or self.optimizer_max_children <= 0:
            return edges

        max_children = int(self.optimizer_max_children)
        if len(edges) <= max_children:
            return edges
        reference_state = (
            np.asarray(trajectory["states"][2], dtype=float)
            if len(trajectory.get("states", [])) > 2
            else np.asarray(self.goal_state, dtype=float)
        )
        goal_state = np.asarray(self.goal_state, dtype=float)
        scores = []
        for index, edge in enumerate(edges):
            target = np.asarray(edge.target_state, dtype=float)
            progress = arrayDistance(target, goal_state, system=self.system.name)
            local = arrayDistance(target, reference_state, system=self.system.name)
            scores.append((0.65 * progress + 0.35 * local, index))
        keep = [index for _, index in sorted(scores)[:max_children]]
        return [edges[index] for index in keep]

    def optimizer_compatible_candidate(
        self,
        candidates: list[dict],
        optimization_result: dict | None,
        optimized_edges: list[ControlEdge],
    ) -> dict:
        """Prefer the cheapest candidate whose first edge was optimized."""
        if not optimization_result or not optimized_edges:
            return candidates[0]
        tolerance = float(
            getattr(self.planner, "optimizer_child_match_tolerance", 1e-5)
        )
        for candidate in candidates:
            states = candidate.get("states") or []
            if len(states) < 2:
                continue
            target = np.asarray(states[1], dtype=float)
            nearest = min(
                float(
                    arrayDistance(
                        target,
                        edge.target_state,
                        system=self.system.name,
                    )
                )
                for edge in optimized_edges
            )
            if nearest <= tolerance:
                return candidate
        return candidates[0]

    def select_replanned_trajectory(
        self,
        previous_trajectory: dict,
        current_state: np.ndarray,
        optimization_result: dict | None,
        optimized_edges: list[ControlEdge],
        execution_duration: float,
        continuity_max_distance: float,
    ) -> AURA.PlanUpdate:
        """Select a continuous plan, using one fresh solve when necessary."""
        replanning_completed = bool(self.replanning_thread_result["completed"])
        if replanning_completed:
            solutions = self.replanning_thread_result["result"] or []
            failure_prefix = "no_continuous_resolved_solution_and_recovery_failed"
        else:
            error = self.replanning_thread_result.get("error")
            recoverable = (
                error is None
                or str(error).startswith("timeout_after_")
                or str(error) == "no_replan_solutions"
            )
            if not recoverable:
                return self.PlanUpdate(
                    None, optimization_result, {}, [], [], "",
                    failure_reason=f"replanning_failed:{error}",
                )
            solutions = []
            failure_prefix = "no_continuous_previous_plan_and_recovery_failed"

        candidates, continuity = auraHandler.reachable_plan_candidates(
            solutions,
            previous_trajectory,
            np.asarray(current_state, dtype=float),
            system_name=self.system.name,
            propagation_step_size=self.propagation_step_size,
            continuity_max_distance=continuity_max_distance,
        )
        additional_replanning = 0
        if not candidates:
            recovery_budget = float(
                getattr(
                    self.planner,
                    "recovery_replanning_time",
                    execution_duration,
                )
            )
            additional_replanning = 1
            recovered, recovery_error, _ = self.fresh_recovery_plan(
                np.asarray(current_state, dtype=float),
                recovery_budget,
            )
            if recovered is None:
                return self.PlanUpdate(
                    None,
                    optimization_result,
                    continuity,
                    [],
                    [],
                    "",
                    additional_replanning=additional_replanning,
                    failure_reason=f"{failure_prefix}:{recovery_error}",
                )
            candidates = [recovered]
            optimization_result = None
            continuity = {
                "max_dist": continuity_max_distance,
                "accepted": 1,
                "rejected": int(continuity.get("rejected", 0)),
                "best_rejected_dist": continuity.get(
                    "best_rejected_dist", float("inf")
                ),
                "recovery": "fresh_plan",
            }

        chosen = self.optimizer_compatible_candidate(
            candidates,
            optimization_result,
            optimized_edges,
        )
        goal_path_source = (
            candidates
            if additional_replanning and replanning_completed
            else solutions
        )
        goal_paths = auraHandler.visual_goal_solution_paths(goal_path_source)
        return self.PlanUpdate(
            trajectory=chosen,
            optimization_result=optimization_result,
            continuity=continuity,
            candidate_paths=auraHandler.visual_candidate_paths(candidates),
            goal_solution_paths=goal_paths,
            caption=auraHandler.plan_change_caption(
                chosen,
                continuity,
                candidates,
                goal_solution_count=len(solutions),
            ),
            additional_replanning=additional_replanning,
        )

    def select_safe_control(
        self,
        plan_update: AURA.PlanUpdate,
        current_state: np.ndarray,
        optimized_edges: list[ControlEdge],
        execution_duration: float,
    ) -> AURA.ControlUpdate:
        """Choose the optimizer/planner control and recover if its curve is invalid."""
        trajectory = plan_update.trajectory
        duration = auraHandler.plan_control_duration(
            trajectory,
            0,
            default_duration=self.planner.propagation_step_size,
        )
        next_state = trajectory["states"][1]
        planned_edge = ControlEdge(
            source_state=np.asarray(trajectory["states"][0], dtype=float),
            target_state=np.asarray(next_state, dtype=float),
            control=np.asarray(trajectory["controls"][0], dtype=float),
            duration_steps=self.planner.duration_seconds_to_steps(duration),
            duration_seconds=float(duration),
            edge_id=str(trajectory.get("_edge_id", "selected-plan:0")),
        )
        selection = self.pick_next_control(
            self.system,
            plan_update.optimization_result,
            current_state,
            next_state,
            optimized_edges,
            fallback_edge=planned_edge,
            optimization_error=self.optimization_thread_result.get("error"),
        )
        curve_valid = selection.control is not None and self.control_curve_valid(
            np.asarray(current_state, dtype=float),
            np.asarray(selection.control, dtype=float),
            float(selection.duration_seconds),
        )
        self.last_control_decision["chosen_curve_valid"] = bool(curve_valid)
        if curve_valid:
            return self.ControlUpdate(
                trajectory,
                selection,
                plan_update.continuity,
                plan_update.candidate_paths,
                plan_update.goal_solution_paths,
                plan_update.caption,
            )

        self.last_control_decision["reason"] = (
            f"{self.last_control_decision.get('reason', 'unknown')}; "
            "chosen_curve_in_collision; recovery_replan_triggered"
        )
        recovery_budget = float(
            getattr(
                self.planner,
                "recovery_replanning_time",
                max(float(execution_duration), float(selection.duration_seconds)),
            )
        )
        recovery_info = {
            "attempted": True,
            "budget": recovery_budget,
            "status": "failed",
            "solutions": 0,
            "selected_controls": 0,
        }
        try:
            recovery_solutions, _ = self.planner.plan_from_state(
                np.asarray(current_state, dtype=float),
                time_budget=recovery_budget,
            )
        except Exception as exc:
            recovery_info["status"] = f"exception:{exc!r}"
            recovery_solutions = []

        recovery_info["solutions"] = len(recovery_solutions or [])
        for recovery_solution in recovery_solutions or []:
            if not recovery_solution.get("controls"):
                continue
            recovery_solution = auraHandler.align_plan_root_to_pose(
                recovery_solution,
                np.asarray(current_state, dtype=float),
            )
            recovery_duration = auraHandler.plan_control_duration(
                recovery_solution,
                0,
                fallback=max(
                    float(execution_duration),
                    float(selection.duration_seconds),
                ),
                default_duration=self.planner.propagation_step_size,
            )
            recovery_control = np.asarray(
                recovery_solution["controls"][0], dtype=float
            )
            if not self.control_curve_valid(
                np.asarray(current_state, dtype=float),
                recovery_control,
                float(recovery_duration),
            ):
                continue

            recovery_solution["_selection_source"] = "recovery fresh plan"
            recovery_solution["_trim_index"] = 0
            recovery_solution["_continuity_dist"] = 0.0
            selection = ControlSelection(
                control=recovery_control,
                duration_steps=self.planner.duration_seconds_to_steps(
                    recovery_duration
                ),
                duration_seconds=float(recovery_duration),
                edge_id="recovery-plan:0",
                source="recovery",
                target_state=np.asarray(
                    recovery_solution["states"][1], dtype=float
                ),
            )
            recovery_info["status"] = "success"
            recovery_info["selected_controls"] = int(
                recovery_solution.get(
                    "control_count",
                    len(recovery_solution.get("controls", [])),
                )
            )
            continuity = {
                "accepted": 1,
                "rejected": 0,
                "best_rejected_dist": float("inf"),
                "max_dist": 0.0,
            }
            self.last_control_decision.update(
                source="recovery",
                reason="fresh_replan_from_actual_pose_after_invalid_candidates",
                chosen_curve_valid=True,
            )
            candidate_paths = auraHandler.visual_candidate_paths(
                [recovery_solution]
            )
            goal_paths = auraHandler.visual_goal_solution_paths(
                [recovery_solution]
            )
            caption = auraHandler.plan_change_caption(
                recovery_solution,
                continuity,
                [recovery_solution],
                recovery_info=recovery_info,
                goal_solution_count=1,
            )
            return self.ControlUpdate(
                recovery_solution,
                selection,
                continuity,
                candidate_paths,
                goal_paths,
                caption,
            )

        return self.ControlUpdate(
            trajectory,
            selection,
            plan_update.continuity,
            plan_update.candidate_paths,
            plan_update.goal_solution_paths,
            plan_update.caption,
            failure_reason="recovery_replan_failed_after_invalid_candidates",
        )

    def wait_for_cycle_tasks(
        self,
        execution_thread: threading.Thread,
        optimization_thread: threading.Thread | None,
        optimization_stop_event: threading.Event | None,
        optimization_slot: dict | None,
        replanning_thread: threading.Thread,
        deadline: float,
        control_duration: float,
    ) -> None:
        """Collect the parallel workers without exceeding the execution window."""
        execution_thread.join(max(0.0, deadline - time.monotonic()))
        timeout_slack = float(
            getattr(self.simulator, "config", {}).get(
                "execution_timeout_slack", 0.0
            )
        )
        if execution_thread.is_alive() and timeout_slack > 0.0:
            execution_thread.join(timeout_slack)
        if execution_thread.is_alive():
            try:
                self.simulator.stop()
            except Exception:
                pass
            execution_thread.join()
            if not self.execution_thread_result.get("completed"):
                self.execution_thread_result = {
                    "result": None,
                    "completed": False,
                    "error": f"timeout_after_{float(control_duration):.3f}s",
                }

        if optimization_thread is not None:
            optimization_thread.join(max(0.0, deadline - time.monotonic()))
            if optimization_thread.is_alive():
                if optimization_stop_event is not None:
                    optimization_stop_event.set()
                partial_result = optimization_slot.get("partial_result")
                if partial_result is not None:
                    partial_result = dict(partial_result)
                    partial_result.update(
                        partial_result=True,
                        timed_out=True,
                        stopped=True,
                    )
                    optimization_slot["result"] = partial_result
                    optimization_slot["completed"] = True
                    optimization_slot["error"] = (
                        f"partial_after_{float(control_duration):.3f}s"
                    )
                else:
                    optimization_slot["result"] = None
                    optimization_slot["completed"] = False
                    optimization_slot["error"] = (
                        f"timeout_after_{float(control_duration):.3f}s"
                    )
                optimization_thread.join()

        replanning_thread.join(max(0.0, deadline - time.monotonic()))
        if replanning_thread.is_alive():
            replanning_thread.join()
            timeout_message = f"timeout_after_{float(control_duration):.3f}s"
            if not self.replanning_thread_result.get("completed"):
                self.replanning_thread_result.update(
                    result=None,
                    completed=False,
                    error=timeout_message,
                )
            elif self.replanning_thread_result.get("error") is None:
                self.replanning_thread_result["error"] = (
                    f"late_after_{float(control_duration):.3f}s"
                )

    def run_cycle_tasks(
        self,
        history: AURA.RunHistory,
        control: np.ndarray,
        duration: float,
        optimization_anchor: np.ndarray,
        children_edges: list[ControlEdge],
    ) -> int:
        """Execute, optimize, and replan concurrently for one control edge."""
        deadline = time.monotonic() + float(duration)
        execution_thread = threading.Thread(
            target=self.execute_control,
            args=(control, duration),
        )
        primitive_count_before = len(
            getattr(self.simulator, "primitive_states", [])
        )
        execution_thread.start()

        if children_edges:
            optimization_slot = self.optimization_thread_result
            optimization_stop_event = threading.Event()
            optimization_ready_event = threading.Event()
            wall_budget = max(
                0.05,
                float(duration)
                * (
                    0.90
                    if self.system.name == "dubins_airplane"
                    else 1.0
                ),
            )
            optimization_thread = threading.Thread(
                target=self.run_optimizer,
                args=(
                    self.system,
                    optimization_anchor,
                    children_edges,
                    wall_budget,
                ),
                kwargs={
                    "result_slot": optimization_slot,
                    "stop_event": optimization_stop_event,
                    "ready_event": optimization_ready_event,
                },
            )
            optimization_thread.start()
            if self.system.name == "dubins_airplane":
                optimization_ready_event.wait(
                    min(0.20, max(0.0, deadline - time.monotonic()))
                )
        else:
            optimization_thread = None
            optimization_stop_event = None
            optimization_slot = None
            self.optimization_thread_result = {
                "result": None,
                "partial_result": None,
                "completed": True,
                "error": "no_branch_children",
            }

        replanning_thread = threading.Thread(
            target=self.run_replanning,
            args=(self.planner, max(0.001, deadline - time.monotonic())),
            kwargs={"result_slot": self.replanning_thread_result},
        )
        history.num_replanning += 1
        replanning_thread.start()
        self.wait_for_cycle_tasks(
            execution_thread,
            optimization_thread,
            optimization_stop_event,
            optimization_slot,
            replanning_thread,
            deadline,
            duration,
        )
        return primitive_count_before

    def record_cycle_execution(
        self,
        history: AURA.RunHistory,
        control: np.ndarray,
        duration: float,
        predicted_curve: list[np.ndarray],
        primitive_count_before: int,
        nominal_target: np.ndarray,
    ) -> np.ndarray | None:
        """Store the measured outcome of a completed execution worker."""
        if not self.execution_thread_result["completed"]:
            return None
        current_state = self.execution_thread_result["result"]
        history.actual_states.append(current_state.copy())
        history.controls.append(np.asarray(control, dtype=float).copy())
        history.duration_seconds.append(float(duration))
        history.duration_steps.append(
            self.planner.duration_seconds_to_steps(duration)
        )
        self.append_executed_primitives(
            history.dense_states,
            primitive_count_before,
            predicted_curve,
            current_state,
        )
        history.tracking_errors.append(
            float(
                arrayDistance(
                    current_state,
                    nominal_target,
                    system=self.system.name,
                )
            )
        )
        return current_state

    def publish_planning_update(
        self,
        callback: Optional[Callable[[int, dict, np.ndarray], None]],
        index: int,
        trajectory: dict,
        current_state: np.ndarray,
        dense_states: list[np.ndarray],
        reference_context: dict,
        cost_context: dict,
        *,
        candidate_paths: list[dict] | None = None,
        goal_solution_paths: list[dict] | None = None,
        caption: str = "",
        initial: bool = False,
        final: bool = False,
    ) -> None:
        if callback is None:
            return
        update = auraHandler.add_visual_reference(
            auraHandler.snapshot_plan(trajectory),
            **reference_context,
        )
        update["actual_states"] = [
            np.asarray(state, dtype=float).copy() for state in dense_states
        ]
        if candidate_paths is not None:
            update["candidate_paths"] = candidate_paths
        if goal_solution_paths is not None:
            update["goal_solution_paths"] = goal_solution_paths
        if initial:
            caption = auraHandler.plan_change_caption(
                trajectory, {}, [], initial=True
            )
        elif final:
            update["_final_update"] = True
            caption = auraHandler.plan_change_caption(
                trajectory, {}, [], final=True
            )
        update["plan_caption"] = caption
        auraHandler.annotate_visual_cost(
            update,
            **cost_context,
            **({"final": True} if final else {}),
        )
        callback(index, update, np.asarray(current_state, dtype=float))

    def plot_optimizer_loss(
        self,
        optimization_result: dict | None,
        step_index: int,
        output_directory: str | None,
        show_live: bool,
    ) -> None:
        if not optimization_result:
            return
        loss_history = optimization_result.get("loss_history")
        if loss_history is None or np.size(loss_history) == 0:
            return
        save_path = None
        if output_directory:
            save_path = os.path.join(
                output_directory, f"loss_step_{step_index:04d}.png"
            )
        auraHandler.plot_optimization_loss_history(
            loss_history,
            step_index=step_index,
            save_path=save_path,
            show_live=show_live,
        )

    def parallel_cycle_timing(
        self,
        cycle_started: float,
        control_duration: float,
        optimization_result: dict | None,
    ) -> dict:
        elapsed = time.monotonic() - cycle_started
        result = optimization_result or {}
        optimizer_elapsed = result.get(
            "total_optimizer_seconds",
            self.optimization_thread_result.get("elapsed_seconds", 0.0),
        )
        states = int(result.get("effective_num_states", 0) or 0)
        children = int(result.get("num_children", 0) or 0)
        return {
            "duration_seconds": float(control_duration),
            "wall_seconds": float(elapsed),
            "compute_overrun_seconds": float(
                max(0.0, elapsed - float(control_duration))
            ),
            "mode": "parallel",
            "optimizer_candidate_batch_seconds": float(
                result.get("candidate_batch_construction_seconds", 0.0)
            ),
            "optimizer_dynamics_gradient_seconds": float(
                result.get("dynamics_gradient_seconds", 0.0)
            ),
            "optimizer_total_seconds": float(optimizer_elapsed),
            "optimizer_b": states,
            "optimizer_c": children,
            "optimizer_candidate_rows": states * children,
            "optimizer_steps_completed": int(
                result.get("steps_completed", 0) or 0
            ),
            "optimizer_steps_requested": int(
                result.get("requested_num_steps", 0) or 0
            ),
            "optimizer_device": str(result.get("device", "")),
            "optimizer_candidate_batch_shape": result.get(
                "candidate_batch_shape"
            ),
        }

    def execute_final_segment(
        self,
        history: AURA.RunHistory,
        current_state: np.ndarray,
        trajectory: dict,
        selected_control: np.ndarray | None,
        selected_duration: float,
        max_nominal_execution_seconds: float | None,
        callback: Optional[Callable[[int, dict, np.ndarray], None]],
        callback_index: int,
        reference_context: dict,
        cost_context: dict,
    ) -> tuple[np.ndarray, np.ndarray | None, str]:
        """Execute and record the final selected control."""
        control = (
            np.asarray(selected_control, dtype=float)
            if selected_control is not None
            else np.asarray(trajectory["controls"][0], dtype=float)
        )
        duration = float(selected_duration)
        if (
            max_nominal_execution_seconds is not None
            and sum(history.duration_seconds) + duration
            > float(max_nominal_execution_seconds) + 1e-9
        ):
            return current_state, None, "task_time_limit_reached"

        predicted_curve = sample_control_curve(
            self.system,
            current_state,
            control,
            duration,
            float(self.propagation_step_size),
            include_start=False,
        )
        started = time.monotonic()
        primitive_count_before = len(
            getattr(self.simulator, "primitive_states", [])
        )
        self.execution_thread_result = {
            "result": None,
            "completed": False,
            "error": None,
        }
        execution_thread = threading.Thread(
            target=self.execute_control,
            args=(control, duration),
        )
        execution_thread.start()
        execution_thread.join()
        if not self.execution_thread_result["completed"]:
            error = self.execution_thread_result.get("error")
            return current_state, None, f"final_execution_failed:{error}"

        current_state = self.execution_thread_result["result"]
        history.actual_states.append(current_state.copy())
        self.append_executed_primitives(
            history.dense_states,
            primitive_count_before,
            predicted_curve,
            current_state,
        )
        history.controls.append(control.copy())
        history.duration_seconds.append(duration)
        history.duration_steps.append(
            self.planner.duration_seconds_to_steps(duration)
        )
        final_planned_state = np.asarray(trajectory["states"][-1], dtype=float)
        history.tracking_errors.append(
            float(
                arrayDistance(
                    current_state,
                    final_planned_state,
                    system=self.system.name,
                )
            )
        )
        self.publish_planning_update(
            callback,
            callback_index,
            trajectory,
            current_state,
            history.dense_states,
            reference_context,
            cost_context,
            final=True,
        )
        elapsed = time.monotonic() - started
        history.cycle_timing.append(
            {
                "duration_seconds": duration,
                "wall_seconds": float(elapsed),
                "compute_overrun_seconds": float(max(0.0, elapsed - duration)),
                "mode": "final",
            }
        )
        return current_state, final_planned_state, ""

    def recover_terminal_goal(
        self,
        history: AURA.RunHistory,
        trajectory: dict,
        final_planned_state: np.ndarray | None,
        *,
        aborted: bool,
        failure_reason: str,
        reached_step_limit: bool,
        max_steps: int | None,
        max_nominal_execution_seconds: float | None,
        terminal_recovery_depth: int,
        callback: Optional[Callable[[int, dict, np.ndarray], None]],
        pause_each_step: bool,
        optimization_loss_plot_dir: str | None,
        show_optimization_loss_plot: bool,
    ) -> tuple[dict, np.ndarray | None, bool, str, bool]:
        """Continue from a measured terminal miss and merge the child result."""
        final_goal_distance = float(
            arrayDistance(
                history.actual_states[-1],
                self.goal_state,
                system=self.system.name,
            )
        )
        remaining_steps = (
            None
            if max_steps is None
            else max(0, int(max_steps) - len(history.controls))
        )
        should_recover = (
            not aborted
            and not reached_step_limit
            and final_goal_distance > float(self.goal_threshold)
            and (remaining_steps is None or remaining_steps > 0)
            and int(terminal_recovery_depth) < 32
        )
        if not should_recover:
            return (
                trajectory,
                final_planned_state,
                aborted,
                failure_reason,
                reached_step_limit,
            )

        recovery_budget = float(
            history.duration_seconds[-1]
            if history.duration_seconds
            else self.replanning_time
        )
        started = time.monotonic()
        history.num_replanning += 1
        try:
            recovery_solutions, _ = self.planner.plan_from_state(
                np.asarray(history.actual_states[-1], dtype=float),
                time_budget=recovery_budget,
            )
            recovery_error = ""
        except Exception as exc:
            recovery_solutions = []
            recovery_error = repr(exc)
        elapsed = time.monotonic() - started
        self.record_timing("replanning", elapsed)
        history.cycle_timing.append(
            {
                "duration_seconds": 0.0,
                "wall_seconds": float(elapsed),
                "compute_overrun_seconds": float(elapsed),
                "mode": "terminal-recovery-planning",
            }
        )
        recovery_solution = next(
            (
                solution
                for solution in (recovery_solutions or [])
                if solution.get("controls")
                and len(solution.get("states", [])) >= 2
            ),
            None,
        )
        if recovery_solution is None:
            reason = (
                "terminal_recovery_plan_failed"
                if not recovery_error
                else f"terminal_recovery_plan_failed:{recovery_error}"
            )
            return trajectory, final_planned_state, True, reason, reached_step_limit

        recovery_solution = auraHandler.align_plan_root_to_pose(
            recovery_solution,
            np.asarray(history.actual_states[-1], dtype=float),
        )
        self.planner.solutions = [recovery_solution]
        recovery_runner = AURA(self.system, self.planner, self.simulator)
        child_callback = None
        if callback is not None:
            callback_offset = len(history.controls)

            def forward_child_update(child_index, plan_update, state):
                callback(callback_offset + int(child_index), plan_update, state)

            child_callback = forward_child_update

        recovery_result = recovery_runner.run(
            reset_sim=False,
            on_planning_update=child_callback,
            pause_each_step=pause_each_step,
            optimization_loss_plot_dir=optimization_loss_plot_dir,
            show_optimization_loss_plot=show_optimization_loss_plot,
            max_steps=remaining_steps,
            max_nominal_execution_seconds=(
                None
                if max_nominal_execution_seconds is None
                else max(
                    0.0,
                    float(max_nominal_execution_seconds)
                    - sum(history.duration_seconds),
                )
            ),
            terminal_recovery_depth=int(terminal_recovery_depth) + 1,
        )
        history.actual_states.extend(
            np.asarray(state, dtype=float).copy()
            for state in recovery_result.states_trajectory[1:]
        )
        history.dense_states.extend(
            np.asarray(state, dtype=float).copy()
            for state in recovery_result.dense_states_trajectory[1:]
        )
        history.controls.extend(
            np.asarray(control, dtype=float).copy()
            for control in recovery_result.controls_trajectory
        )
        history.duration_steps.extend(
            int(steps)
            for steps in recovery_result.control_duration_steps_trajectory
        )
        history.duration_seconds.extend(
            float(seconds)
            for seconds in recovery_result.control_duration_seconds_trajectory
        )
        history.tracking_errors.extend(
            float(error) for error in recovery_result.tracking_error_list
        )
        history.num_replanning += int(recovery_result.num_replanning)
        history.cycle_timing.extend(
            dict(row) for row in recovery_result.cycle_timing
        )
        self._timing_totals["execution"] += float(
            recovery_result.actual_execution_seconds
        )
        self._timing_totals["replanning"] += float(
            recovery_result.online_replanning_seconds
        )
        self._timing_totals["optimization"] += float(
            recovery_result.optimizer_seconds
        )
        final_planned_state = (
            None
            if recovery_result.final_planned_state is None
            else np.asarray(recovery_result.final_planned_state, dtype=float)
        )
        trajectory = {
            "states": [
                np.asarray(state, dtype=float).copy()
                for state in (recovery_result.final_plan_states or [])
            ],
            "controls": [
                np.asarray(control, dtype=float).copy()
                for control in (recovery_result.final_plan_controls or [])
            ],
        }
        trajectory["state_count"] = len(trajectory["states"])
        trajectory["control_count"] = len(trajectory["controls"])
        failure_reason = recovery_result.failure_reason
        return (
            trajectory,
            final_planned_state,
            recovery_result.status != "success",
            failure_reason,
            failure_reason == "max_steps_reached",
        )

    def build_result(
        self,
        *,
        history: AURA.RunHistory,
        final_trajectory: dict,
        final_planned_state: np.ndarray | None,
        aborted: bool,
        failure_reason: str,
        reached_step_limit: bool,
    ) -> AURA.AURAResult:
        final_goal_distance = float(
            arrayDistance(
                history.actual_states[-1],
                self.goal_state,
                system=self.system.name,
            )
        )
        if final_goal_distance > float(self.goal_threshold):
            aborted = True
            if not failure_reason:
                failure_reason = (
                    "max_steps_reached"
                    if reached_step_limit
                    else f"goal_not_reached:{final_goal_distance:.9g}"
                )
        cost = sum(
            arrayDistance(first, second, system=self.system.name)
            for first, second in zip(
                history.actual_states, history.actual_states[1:]
            )
        )
        if final_planned_state is None and final_trajectory.get("states"):
            final_planned_state = final_trajectory["states"][-1]
        return self.AURAResult(
            num_controls=len(history.controls),
            num_replanning=int(history.num_replanning),
            final_state=np.asarray(history.actual_states[-1], dtype=float),
            cost=float(cost),
            tracking_error_mean=(
                float(np.mean(history.tracking_errors))
                if history.tracking_errors
                else 0.0
            ),
            tracking_error_list=list(history.tracking_errors),
            controls_trajectory=[
                np.asarray(control, dtype=float) for control in history.controls
            ],
            control_duration_steps_trajectory=list(history.duration_steps),
            control_duration_seconds_trajectory=list(history.duration_seconds),
            states_trajectory=[
                np.asarray(state, dtype=float) for state in history.actual_states
            ],
            dense_states_trajectory=[
                np.asarray(state, dtype=float) for state in history.dense_states
            ],
            cycle_timing=list(history.cycle_timing),
            nominal_execution_seconds=float(sum(history.duration_seconds)),
            actual_execution_seconds=float(self._timing_totals["execution"]),
            online_replanning_seconds=float(self._timing_totals["replanning"]),
            optimizer_seconds=float(self._timing_totals["optimization"]),
            compute_overrun_seconds=float(
                sum(
                    row["compute_overrun_seconds"]
                    for row in history.cycle_timing
                )
            ),
            final_plan_states=[
                np.asarray(state, dtype=float)
                for state in final_trajectory.get("states", [])
            ],
            final_plan_controls=[
                np.asarray(control, dtype=float)
                for control in final_trajectory.get("controls", [])
            ],
            final_planned_state=(
                None
                if final_planned_state is None
                else np.asarray(final_planned_state, dtype=float)
            ),
            status="failure" if aborted else "success",
            failure_reason=failure_reason,
        )

    def run(
        self,
        reset_sim: bool = True,
        *,
        on_planning_update: Optional[Callable[[int, dict, np.ndarray], None]] = None,
        pause_each_step: bool = True,
        optimization_loss_plot_dir: Optional[str] = None,
        show_optimization_loss_plot: bool = False,
        max_steps: int | None = None,
        max_nominal_execution_seconds: float | None = None,
        terminal_recovery_depth: int = 0,
    ) -> AURA.AURAResult:
        # CUDA/Adam initialization is intentionally outside the first one-second
        # online recovery window.  Otherwise the first control cycle can spend
        # its entire deadline compiling kernels and complete zero optimizer steps.
        warmup_optimizer_device(self.system.name, self.optimizer_device)
        if reset_sim:
            self.simulator.reset()
            self.simulator.set_state(self.start_state.tolist())

        current_state = np.array(self.simulator.get_state(), dtype=float)

        history = self.RunHistory(
            actual_states=[current_state.copy()],
            dense_states=[current_state.copy()],
            controls=[],
            duration_steps=[],
            duration_seconds=[],
            tracking_errors=[],
            cycle_timing=[],
        )
        actual_trajectory = history.actual_states
        dense_actual_trajectory = history.dense_states
        controls_trajectory = history.controls
        control_duration_steps_trajectory = history.duration_steps
        control_duration_seconds_trajectory = history.duration_seconds
        tracking_errors = history.tracking_errors
        cycle_timing = history.cycle_timing
        self._timing_totals = {
            "execution": 0.0,
            "replanning": 0.0,
            "optimization": 0.0,
        }
        default_control_duration = self.planner.propagation_step_size
        continuity_max_distance = float(
            getattr(
                self.planner,
                "solution_continuity_max_distance",
                max(0.25, float(getattr(self, "replanning_time", 0.075))),
            )
        )
        visual_cost_context = {
            "controls_trajectory": controls_trajectory,
            "dense_actual_trajectory": dense_actual_trajectory,
            "system": self.system,
            "propagation_step_size": self.propagation_step_size,
            "planner": self.planner,
        }

        best_trajectory = self.planner.best_solution()
        if not best_trajectory or not best_trajectory.get("controls"):
            raise RuntimeError("AURA cannot run without an initial planner solution.")
        best_trajectory = auraHandler.align_plan_root_to_pose(best_trajectory, current_state)
        initial_reference_plan = auraHandler.snapshot_plan(best_trajectory)
        visual_reference_context = {
            "initial_reference_plan": initial_reference_plan,
            "planner": self.planner,
            "goal_state": self.goal_state,
            "goal_threshold": self.goal_threshold,
        }

        next_control = best_trajectory["controls"][0]
        control_duration = auraHandler.plan_control_duration(
            best_trajectory,
            0,
            default_duration=default_control_duration,
        )
        aborted = False
        failure_reason = ""
        reached_step_limit = False
        final_report_state = None

        self.publish_planning_update(
            on_planning_update,
            0,
            best_trajectory,
            current_state,
            dense_actual_trajectory,
            visual_reference_context,
            visual_cost_context,
            initial=True,
        )

        index = 0
        while best_trajectory["control_count"] > 1:
            cycle_started = time.monotonic()
            if max_steps is not None and index >= int(max_steps):
                reached_step_limit = True
                break

            current_state = self.simulator.get_state()
            actual_trajectory.append(current_state.copy())

            # Fresh result slots each iteration (avoid stale completed/result between threads)
            self.reset_thread_results()
            previous_plan_for_continuity = auraHandler.snapshot_plan(best_trajectory)

            execution_control_duration = auraHandler.plan_control_duration(
                best_trajectory,
                0,
                default_duration=default_control_duration,
            )
            if (
                max_nominal_execution_seconds is not None
                and sum(control_duration_seconds_trajectory)
                + float(execution_control_duration)
                > float(max_nominal_execution_seconds) + 1e-9
            ):
                aborted = True
                failure_reason = "task_time_limit_reached"
                break
            # Tree branch set from the pre-replan path (read before replan mutates the tree in parallel)
            next_nominal_state = np.asarray(best_trajectory["states"][1], dtype=float)
            execution_nominal_target = next_nominal_state.copy()
            children_edges = self.optimizer_edges(
                best_trajectory,
                next_nominal_state,
                execution_control_duration,
            )
            # While execution runs, optimize around the selected-control prediction of the
            # post-segment state. The optimizer's targets are still planner-derived children.
            pre_exec = np.asarray(current_state, dtype=float)
            u = np.asarray(next_control, dtype=float)
            executed_curve = sample_control_curve(
                self.system,
                pre_exec,
                u,
                float(execution_control_duration),
                float(self.propagation_step_size),
                include_start=False,
            )
            selected_control_prediction = (
                np.asarray(executed_curve[-1], dtype=float)
                if executed_curve
                else pre_exec.copy()
            )
            optimization_anchor = selected_control_prediction
            primitive_count_before = self.run_cycle_tasks(
                history,
                next_control,
                execution_control_duration,
                optimization_anchor,
                children_edges,
            )

            # Extract execution result
            current_state = self.record_cycle_execution(
                history,
                u,
                execution_control_duration,
                executed_curve,
                primitive_count_before,
                execution_nominal_target,
            )
            if current_state is None:
                aborted = True
                failure_reason = f"execution_failed:{self.execution_thread_result.get('error')}"
                break

            if (
                arrayDistance(
                    np.asarray(current_state, dtype=float),
                    self.goal_state,
                    system=self.system.name,
                )
                <= float(self.goal_threshold)
            ):
                # The task predicate is defined on the measured state.  A
                # concurrently improved nominal path may still contain
                # controls after the robot has entered the goal; executing
                # them can drive it back out again.
                try:
                    self.simulator.stop()
                except Exception:
                    pass
                final_report_state = np.asarray(
                    execution_nominal_target,
                    dtype=float,
                )
                best_trajectory = {
                    "states": [np.asarray(current_state, dtype=float).copy()],
                    "controls": [],
                    "time": [],
                    "time_steps": [],
                    "state_count": 1,
                    "control_count": 0,
                    "_selection_source": "measured goal reached",
                }
                cycle_elapsed = time.monotonic() - cycle_started
                cycle_timing.append(
                    {
                        "duration_seconds": float(execution_control_duration),
                        "wall_seconds": float(cycle_elapsed),
                        "compute_overrun_seconds": float(
                            max(
                                0.0,
                                cycle_elapsed - float(execution_control_duration),
                            )
                        ),
                        "mode": "goal-reached",
                    }
                )
                break

            # Extract optimization result
            if self.optimization_thread_result["completed"]:
                optimization_result = self.optimization_thread_result["result"]
            else:
                optimization_result = None

            plan_update = self.select_replanned_trajectory(
                previous_plan_for_continuity,
                current_state,
                optimization_result,
                children_edges,
                execution_control_duration,
                continuity_max_distance,
            )
            history.num_replanning += plan_update.additional_replanning
            if plan_update.trajectory is None:
                aborted = True
                failure_reason = plan_update.failure_reason
                break
            best_trajectory = plan_update.trajectory
            optimization_result = plan_update.optimization_result
            control_update = self.select_safe_control(
                plan_update,
                current_state,
                children_edges,
                execution_control_duration,
            )
            if control_update.failure_reason:
                aborted = True
                failure_reason = control_update.failure_reason
                break
            best_trajectory = control_update.trajectory
            next_control = control_update.selection.control
            control_duration = control_update.selection.duration_seconds

            self.plot_optimizer_loss(
                optimization_result,
                index,
                optimization_loss_plot_dir,
                show_optimization_loss_plot,
            )
            self.publish_planning_update(
                on_planning_update,
                index + 1,
                best_trajectory,
                current_state,
                dense_actual_trajectory,
                visual_reference_context,
                visual_cost_context,
                candidate_paths=control_update.candidate_paths,
                goal_solution_paths=control_update.goal_solution_paths,
                caption=control_update.caption,
            )
            cycle_timing.append(
                self.parallel_cycle_timing(
                    cycle_started,
                    execution_control_duration,
                    optimization_result,
                )
            )
            index += 1
            if pause_each_step:
                input("Press Enter to continue...")

        # Execute the selected final edge when one segment remains.
        if not aborted and not reached_step_limit and best_trajectory.get("controls"):
            current_state, final_report_state, final_error = (
                self.execute_final_segment(
                    history,
                    current_state,
                    best_trajectory,
                    next_control,
                    control_duration,
                    max_nominal_execution_seconds,
                    on_planning_update,
                    index + 1,
                    visual_reference_context,
                    visual_cost_context,
                )
            )
            if final_error:
                aborted = True
                failure_reason = final_error

        (
            best_trajectory,
            final_report_state,
            aborted,
            failure_reason,
            reached_step_limit,
        ) = self.recover_terminal_goal(
            history,
            best_trajectory,
            final_report_state,
            aborted=aborted,
            failure_reason=failure_reason,
            reached_step_limit=reached_step_limit,
            max_steps=max_steps,
            max_nominal_execution_seconds=max_nominal_execution_seconds,
            terminal_recovery_depth=terminal_recovery_depth,
            callback=on_planning_update,
            pause_each_step=pause_each_step,
            optimization_loss_plot_dir=optimization_loss_plot_dir,
            show_optimization_loss_plot=show_optimization_loss_plot,
        )

        result = self.build_result(
            history=history,
            final_trajectory=best_trajectory,
            final_planned_state=final_report_state,
            aborted=aborted,
            failure_reason=failure_reason,
            reached_step_limit=reached_step_limit,
        )
        if result.status == "failure" and terminal_recovery_depth == 0:
            print(f"[AURA] failed: {result.failure_reason}")
        return result

    def run_optimizer(
        self,
        system: System,
        next_nominal_state: np.ndarray,
        children_edges: list[ControlEdge],
        wall_time_budget: float | None = None,
        result_slot: dict | None = None,
        stop_event=None,
        ready_event=None,
    ):
        slot = self.optimization_thread_result if result_slot is None else result_slot
        started = time.monotonic()
        try:
            def publish_partial(partial_result):
                auraHandler.store_optimizer_partial_result(slot, partial_result)
                if ready_event is not None:
                    ready_event.set()

            result = optimize_controls(
                system=system,
                next_state=np.asarray(next_nominal_state, dtype=float),
                child_edges=children_edges,
                model=self.opt_model,
                num_states=self.optimizer_num_states,
                position_std=self.optimizer_pos_std,
                rotation_std=self.optimizer_rot_std,
                velocity_std=self.optimizer_vel_std,
                integration_step_size=float(self.propagation_step_size),
                num_steps=self.optimizer_num_steps,
                learning_rate=self.optimizer_learning_rate,
                max_wall_time=wall_time_budget,
                stop_event=stop_event,
                partial_result_callback=publish_partial,
                requested_device=self.optimizer_device,
            )
            slot["result"] = result
            slot["completed"] = True
            slot["error"] = None if result is not None else "no_result"
            return result
        except Exception as e:
            slot["result"] = None
            slot["completed"] = False
            slot["error"] = repr(e)
            traceback.print_exc()
            return None
        finally:
            elapsed = time.monotonic() - started
            slot["elapsed_seconds"] = float(elapsed)
            self.record_timing("optimization", elapsed)

    def execute_control(self, control: np.ndarray, duration: float):
        started = time.monotonic()
        try:
            self.simulator.execute_segment(control, duration)
            result = self.simulator.get_state()
            self.execution_thread_result["result"] = result
            self.execution_thread_result["completed"] = result is not None
            self.execution_thread_result["error"] = None if result is not None else "no_state"
            return result
        except Exception as e:
            self.execution_thread_result["result"] = None
            self.execution_thread_result["completed"] = False
            self.execution_thread_result["error"] = repr(e)
            traceback.print_exc()
            return None
        finally:
            elapsed = time.monotonic() - started
            self.execution_thread_result["elapsed_seconds"] = float(elapsed)
            self.record_timing("execution", elapsed)

    def run_replanning(
        self,
        planner: OMPLPlanner,
        time_budget: float,
        result_slot: dict | None = None,
    ):
        """``OMPLPlanner.replan`` returns solutions and the OMPL setup."""
        slot = self.replanning_thread_result if result_slot is None else result_slot
        started = time.monotonic()
        try:
            replan_solutions, _ = planner.replan(
                time_budget=float(time_budget),
            )
            slot["result"] = replan_solutions
            slot["completed"] = bool(replan_solutions)
            slot["error"] = (
                None if replan_solutions else "no_replan_solutions"
            )
            return replan_solutions
        except Exception as e:
            slot["result"] = None
            slot["completed"] = False
            slot["error"] = repr(e)
            traceback.print_exc()
            return None
        finally:
            elapsed = time.monotonic() - started
            slot["elapsed_seconds"] = float(elapsed)
            self.record_timing("replanning", elapsed)

    def predict_control_endpoints(
        self,
        system: System,
        start_state: np.ndarray,
        controls: np.ndarray,
        duration: float,
    ) -> np.ndarray:
        """Vectorized endpoint prediction for choosing among optimizer samples."""
        controls_np = np.asarray(controls, dtype=float)
        if controls_np.ndim == 1:
            controls_np = controls_np.reshape(1, -1)
        start = np.asarray(start_state, dtype=float).reshape(-1)
        duration = float(duration)

        if system.name == "kinematic_car":
            starts = np.repeat(
                start[:3].reshape(1, 3), controls_np.shape[0], axis=0
            )
            return kinematic_car.propagate_numpy(starts, controls_np, duration)

        if system.name == "double_integrator":
            starts = np.repeat(
                start[:6].reshape(1, 6), controls_np.shape[0], axis=0
            )
            return double_integrator.propagate_numpy(starts, controls_np, duration)

        if system.name == "dubins_airplane":
            starts = np.repeat(
                start[:6].reshape(1, 6), controls_np.shape[0], axis=0
            )
            return dubins_airplane.propagate_numpy(starts, controls_np, duration)

        if system.name in ("pushing", "pushing_object"):
            canonical_controls = np.asarray(
                [system.canonical_control(control) for control in controls_np],
                dtype=float,
            )
            model = getattr(self.opt_model, "model", None)
            if model is None:
                model = get_pushing_model(
                    system.object_shape,
                    model_name=getattr(system, "model_name", "cracker_box_flipped"),
                    model_path=getattr(system, "model_path", None),
                )
            device = next(model.parameters()).device
            model = model.to(device)
            starts = torch.as_tensor(
                np.repeat(start[:3].reshape(1, 3), len(canonical_controls), axis=0),
                dtype=torch.float32,
                device=device,
            )
            control_tensor = torch.as_tensor(
                canonical_controls, dtype=torch.float32, device=device
            )
            duration_steps = self.planner.duration_seconds_to_steps(duration)
            steps = torch.full(
                (len(canonical_controls),),
                int(duration_steps),
                dtype=torch.long,
                device=device,
            )
            with torch.no_grad():
                predicted = pushing_object.propagate_torch(
                    starts,
                    control_tensor,
                    steps,
                    model,
                )
            return predicted.detach().cpu().numpy()

        predicted_states = []
        step_size = float(
            getattr(
                self.planner,
                "motion_validation_step_size",
                self.propagation_step_size,
            )
        )
        for ctrl in controls_np:
            curve = sample_control_curve(
                system,
                start.copy(),
                ctrl,
                duration,
                step_size,
                include_start=False,
            )
            predicted_states.append(curve[-1] if curve else start.copy())
        return np.asarray(predicted_states, dtype=float)

    def batch_state_distance(
        self, target_state: np.ndarray, states: np.ndarray, system_name: str
    ) -> np.ndarray:
        target = np.asarray(target_state, dtype=float).reshape(-1)
        states_np = np.asarray(states, dtype=float)
        if states_np.ndim == 1:
            states_np = states_np.reshape(1, -1)
        if system_name in ("simple_car", "kinematic_car", "pushing", "pushing_object"):
            dxy = np.linalg.norm(states_np[:, :2] - target[:2], axis=1)
            dtheta = (states_np[:, 2] - target[2] + np.pi) % (2 * np.pi) - np.pi
            return dxy + 0.5 * np.abs(dtheta)
        if system_name == "double_integrator":
            return np.linalg.norm(states_np[:, :6] - target[:6], axis=1)
        if system_name == "dubins_airplane":
            residual = states_np[:, :6] - target[:6]
            residual[:, 3] = (residual[:, 3] + np.pi) % (2.0 * np.pi) - np.pi
            return np.linalg.norm(residual, axis=1)
        return np.asarray(
            [arrayDistance(target, pred, system=system_name) for pred in states_np],
            dtype=float,
        )

    def control_curve_valid(
        self, start_state: np.ndarray, control: np.ndarray, duration: float
    ) -> bool:
        config = dict(getattr(self.simulator, "config", {}) or {})
        if "obstacles" not in config and getattr(self.planner, "obstacle_config", None) is not None:
            config["obstacles"] = self.planner.obstacle_config
        if "state_bounds" not in config:
            config["state_bounds"] = self.system.state_bounds
        execution_safety_radius = float(config.get("execution_safety_radius", 0.0))
        step_size = float(getattr(self.planner, "motion_validation_step_size", self.propagation_step_size))
        if self.system.name in ("pushing", "pushing_object"):
            # The learned model represents exactly one primitive at h. Validate
            # every primitive boundary without feeding it a fractional h.
            duration_steps = self.planner.duration_seconds_to_steps(duration)
            state = np.asarray(start_state, dtype=float).reshape(-1).copy()
            curve = [state.copy()]
            for _ in range(duration_steps):
                state = np.asarray(
                    self.system.propagate(
                        state,
                        control,
                        float(self.propagation_step_size),
                    ),
                    dtype=float,
                ).reshape(-1)
                curve.append(state.copy())
        else:
            curve = sample_control_curve(
                self.system,
                start_state,
                control,
                float(duration),
                step_size,
                include_start=True,
            )
        return all(
            is_state_array_valid(
                s,
                system=self.system.name,
                config=config,
                safety_radius_override=execution_safety_radius,
            )
            for s in curve
        )

    def pick_next_control(
        self,
        system: System,
        optimization_result: dict,
        current_state: np.ndarray,
        next_state: np.ndarray,
        children_edges: list[ControlEdge],
        *,
        fallback_edge: ControlEdge,
        optimization_error: str | None = None,
        target_tolerance: float | None = None,
    ) -> ControlSelection:
        """Select one inseparable control-duration pair for the replanned edge."""

        def original_selection(edge: ControlEdge, reason: str) -> ControlSelection:
            self.last_control_decision = {
                "source": "planner",
                "reason": reason,
                "edge_id": edge.edge_id,
                "duration_steps": edge.duration_steps,
                "duration_seconds": edge.duration_seconds,
                "original_distance": float("nan"),
                "optimized_distance": float("nan"),
                "between_pred_ompl": float("nan"),
                "between_pred_dxy": float("nan"),
                "between_pred_dtheta": float("nan"),
                "original_curve_valid": None,
                "optimized_curve_valid": None,
            }
            return ControlSelection(
                control=edge.control,
                duration_steps=edge.duration_steps,
                duration_seconds=edge.duration_seconds,
                edge_id=edge.edge_id,
                source="planner",
                target_state=edge.target_state,
            )

        if not optimization_result or "optimized_controls" not in optimization_result:
            reason = "missing_optimization_result"
            if optimization_error:
                reason = f"{reason}:{optimization_error}"
            return original_selection(fallback_edge, reason)

        raw_optimized_controls = optimization_result.get("optimized_controls")
        optimized_control_dtype = np.asarray(raw_optimized_controls).dtype
        optimized_controls = auraHandler.to_numpy_state_control(
            raw_optimized_controls, system.name
        )
        start_states = auraHandler.to_numpy_state_control(
            optimization_result.get("start_states"), system.name
        )
        target_states = auraHandler.to_numpy_state_control(
            optimization_result.get("target_states"), system.name
        )
        duration_seconds = auraHandler.to_numpy_state_control(
            optimization_result.get("duration_seconds"), system.name
        )
        duration_steps = auraHandler.to_numpy_state_control(
            optimization_result.get("duration_steps"), system.name
        )
        edge_ids = list(optimization_result.get("edge_ids", []))
        row_metadata = list(optimization_result.get("row_metadata", []))
        if (
            optimized_controls is None
            or start_states is None
            or target_states is None
            or duration_seconds is None
            or duration_steps is None
            or not children_edges
        ):
            return original_selection(
                fallback_edge, "invalid_optimizer_tensors_or_no_children"
            )

        optimized_controls = np.asarray(optimized_controls, dtype=float)
        start_states = np.asarray(start_states, dtype=float)
        target_states = np.asarray(target_states, dtype=float)
        if optimized_controls.ndim == 1:
            optimized_controls = optimized_controls.reshape(1, -1)
        if start_states.ndim == 1:
            start_states = start_states.reshape(1, -1)
        if target_states.ndim == 1:
            target_states = target_states.reshape(1, -1)
        duration_seconds = np.asarray(duration_seconds, dtype=float).reshape(-1)
        duration_steps = np.asarray(duration_steps, dtype=float).reshape(-1)
        row_count = optimized_controls.shape[0]
        if not (
            start_states.shape[0] == row_count
            and target_states.shape[0] == row_count
            and len(duration_seconds) == row_count
            and len(duration_steps) == row_count
            and len(edge_ids) == row_count
        ):
            return original_selection(fallback_edge, "optimizer_row_contract_mismatch")

        actual_next = auraHandler.to_numpy_state_control(
            next_state, system.name
        ).reshape(-1)
        actual_current = auraHandler.to_numpy_state_control(
            current_state, system.name
        ).reshape(-1)
        tolerance = float(
            target_tolerance
            if target_tolerance is not None
            else getattr(self.planner, "optimizer_child_match_tolerance", 1e-5)
        )
        child_distances = [
            float(arrayDistance(actual_next, edge.target_state, system=system.name))
            for edge in children_edges
        ]
        matched_index = int(np.argmin(child_distances))
        if child_distances[matched_index] > tolerance:
            return original_selection(
                fallback_edge,
                f"unseen_replanned_edge:nearest_distance={child_distances[matched_index]:.6g}",
            )

        matched_edge = children_edges[matched_index]
        candidate_indices = [
            index
            for index, edge_id in enumerate(edge_ids)
            if str(edge_id) == matched_edge.edge_id
        ]
        if not candidate_indices:
            return original_selection(matched_edge, "matched_edge_has_no_optimizer_rows")
        if row_metadata:
            candidate_indices = [
                index
                for index in candidate_indices
                if index < len(row_metadata)
                and bool(row_metadata[index].get("finite", False))
            ]
        if not candidate_indices:
            return original_selection(matched_edge, "matched_edge_optimizer_rows_nonfinite")

        valid_indices = []
        for row_index in candidate_indices:
            if (
                int(round(float(duration_steps[row_index])))
                == matched_edge.duration_steps
                and np.isclose(
                    float(duration_seconds[row_index]),
                    matched_edge.duration_seconds,
                    rtol=0.0,
                    atol=1e-8,
                )
            ):
                valid_indices.append(row_index)
        if not valid_indices:
            return original_selection(matched_edge, "optimizer_duration_identity_mismatch")

        original_control = np.asarray(matched_edge.control, dtype=float).reshape(-1)
        control_duration = matched_edge.duration_seconds
        predicted_original = self.predict_control_endpoints(
            system,
            actual_current.copy(),
            original_control,
            control_duration,
        )[0]
        # Match the original AURA implementation: the gradient batch is
        # precomputed before the observation arrives, then every optimized
        # control for the selected child is evaluated from the measured state.
        # Selecting only the row whose sampled start is nearest can discard a
        # better recovery control when the observed disturbance lies between
        # samples or near the edge of the sampled neighborhood.
        candidate_controls = np.asarray(
            optimized_controls[valid_indices], dtype=float
        )
        candidate_predictions = self.predict_control_endpoints(
            system,
            actual_current.copy(),
            candidate_controls,
            control_duration,
        )
        candidate_distances = self.batch_state_distance(
            actual_next, candidate_predictions, system.name
        )
        ranked_rows = np.argsort(candidate_distances)
        best_row = valid_indices[int(ranked_rows[0])]
        best_optimized_control = candidate_controls[int(ranked_rows[0])]
        predicted_optimized = candidate_predictions[int(ranked_rows[0])]
        original_distance = float(
            self.batch_state_distance(
                actual_next,
                np.asarray(predicted_original, dtype=float).reshape(1, -1),
                system.name,
            )[0]
        )
        optimized_distance = float(candidate_distances[int(ranked_rows[0])])
        original_curve_valid = self.control_curve_valid(
            actual_current.copy(), original_control, control_duration
        )
        optimized_finite = bool(np.all(np.isfinite(best_optimized_control)))
        control_bounds = np.asarray(system.control_bounds, dtype=float)
        dtype_epsilon = (
            float(np.finfo(optimized_control_dtype).eps)
            if np.issubdtype(optimized_control_dtype, np.floating)
            else 0.0
        )
        bound_scale = max(1.0, float(np.max(np.abs(control_bounds))))
        bound_tolerance = max(1e-9, 4.0 * dtype_epsilon * bound_scale)
        optimized_in_bounds = optimized_finite and bool(
            np.all(best_optimized_control >= control_bounds[:, 0] - bound_tolerance)
            and np.all(best_optimized_control <= control_bounds[:, 1] + bound_tolerance)
        )
        if optimized_in_bounds:
            # Torch optimizes float32 controls. Exact decimal actuator limits
            # such as 0.3 are not exactly representable, so project the tiny
            # representation error away before prediction and execution.
            best_optimized_control = np.clip(
                best_optimized_control,
                control_bounds[:, 0],
                control_bounds[:, 1],
            )
            predicted_optimized = self.predict_control_endpoints(
                system,
                actual_current.copy(),
                best_optimized_control,
                control_duration,
            )[0]
            optimized_distance = float(
                self.batch_state_distance(
                    actual_next,
                    np.asarray(predicted_optimized, dtype=float).reshape(1, -1),
                    system.name,
                )[0]
            )
        bp_ompl, bp_dxy, bp_dth = auraHandler.se2_breakdown(
            np.asarray(predicted_original, dtype=float),
            np.asarray(predicted_optimized, dtype=float),
            system.name,
        )
        optimized_curve_valid = optimized_in_bounds and self.control_curve_valid(
            actual_current.copy(),
            best_optimized_control,
            control_duration,
        )
        choose_optimized = optimized_curve_valid and (
            not original_curve_valid or optimized_distance <= original_distance
        )
        self.last_control_decision = {
            "source": "optimized" if choose_optimized else "planner",
            "reason": (
                "optimized_curve_valid_and_planner_curve_invalid"
                if choose_optimized and not original_curve_valid
                else "optimized_distance_better_or_equal"
                if choose_optimized
                else "optimized_control_out_of_bounds"
                if not optimized_in_bounds and original_curve_valid
                else "optimized_curve_in_collision"
                if not optimized_curve_valid and original_curve_valid
                else "planner_pair_safer_or_better"
            ),
            "edge_id": matched_edge.edge_id,
            "duration_steps": matched_edge.duration_steps,
            "duration_seconds": matched_edge.duration_seconds,
            "optimizer_row": best_row,
            "original_distance": float(original_distance),
            "optimized_distance": float(optimized_distance),
            "between_pred_ompl": float(bp_ompl),
            "between_pred_dxy": float(bp_dxy),
            "between_pred_dtheta": float(bp_dth),
            "original_curve_valid": bool(original_curve_valid),
            "optimized_curve_valid": bool(optimized_curve_valid),
            "optimized_finite": optimized_finite,
            "optimized_in_bounds": optimized_in_bounds,
        }
        return ControlSelection(
            control=best_optimized_control if choose_optimized else original_control,
            duration_steps=matched_edge.duration_steps,
            duration_seconds=matched_edge.duration_seconds,
            edge_id=matched_edge.edge_id,
            source="optimized" if choose_optimized else "planner",
            target_state=matched_edge.target_state,
        )
