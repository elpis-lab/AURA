from __future__ import annotations

from dataclasses import dataclass
import os
import time
import traceback
from typing import Callable, Optional

import threading
import numpy as np

from utils import auraHandler
from optimization import runOptimizer
from systems import System
from plan import OMPL_Planner
from utils.childrenHandler import getChildrenStates
from simulators import Simulator
from utils.utils import (
    arrayDistance,
    is_state_array_valid,
    sample_control_curve,
    state2list,
)


class AURA:
    """
    AURA execution loop built on top of:
      - systems.System subclasses
      - systems.plan planning wrapper
      - simulators.Simulator backends
    """

    @dataclass
    class AURAResult:
        num_controls: int
        final_state: np.ndarray
        cost: float
        tracking_error_mean: float
        tracking_error_list: list[float]
        controls_trajectory: list[np.ndarray]
        states_trajectory: list[np.ndarray]
        dense_states_trajectory: list[np.ndarray]
        final_plan_states: list[np.ndarray] | None = None
        final_plan_controls: list[np.ndarray] | None = None
        final_planned_state: np.ndarray | None = None
        status: str = "success"
        failure_reason: str = ""

    def __init__(
        self,
        system: System,
        planner: OMPL_Planner,
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
        if self.optimizer_max_children > 0:
            self.optimizer_max_children = min(self.optimizer_max_children, 64)

        self.execution_thread_result = {"result": None, "completed": False, "error": None}
        self.optimization_thread_result = {"result": None, "completed": False, "error": None}
        self.replanning_thread_result = {"result": None, "completed": False, "error": None}
        self.last_control_decision: dict = {}

    def run(
        self,
        reset_sim: bool = True,
        *,
        on_planning_update: Optional[Callable[[int, dict, np.ndarray], None]] = None,
        pause_each_step: bool = True,
        optimization_loss_plot_dir: Optional[str] = None,
        show_optimization_loss_plot: bool = False,
        max_steps: int | None = None,
    ) -> AURA.AURAResult:
        if reset_sim:
            self.simulator.reset()

        self.simulator.set_obj_init_pose(self.start_state.tolist())

        current_state = np.array(self.simulator.get_state(), dtype=float)

        actual_trajectory = [current_state.copy()]
        dense_actual_trajectory = [current_state.copy()]
        nominal_trajectory = [self.start_state.copy()]
        controls_trajectory: list[np.ndarray] = []
        tracking_errors: list[float] = []
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

        best_trajectory = self.planner.getBestSolution()
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
        previous_step_diff = None
        aborted = False
        failure_reason = ""
        reached_step_limit = False
        final_report_state = None

        if on_planning_update is not None:
            plan_for_update = auraHandler.add_visual_reference(
                auraHandler.snapshot_plan(best_trajectory),
                **visual_reference_context,
            )
            plan_for_update["actual_states"] = [
                np.asarray(s, dtype=float).copy() for s in dense_actual_trajectory
            ]
            plan_for_update["plan_caption"] = auraHandler.plan_change_caption(
                best_trajectory,
                {},
                [],
                initial=True,
            )
            auraHandler.annotate_visual_cost(plan_for_update, **visual_cost_context)
            on_planning_update(0, plan_for_update, np.asarray(current_state, dtype=float))

        index = 0
        while best_trajectory["control_count"] > 1:
            if max_steps is not None and index >= int(max_steps):
                reached_step_limit = True
                break

            current_state = self.simulator.get_state()
            actual_trajectory.append(current_state.copy())
            nominal_trajectory.append(best_trajectory["states"][0].copy())

            state_diff_pre, dxy_vs_plan, dth_vs_plan = auraHandler.se2_breakdown(
                actual_trajectory[-1], nominal_trajectory[-1], self.system.name
            )

            # Fresh result slots each iteration (avoid stale completed/result between threads)
            self.execution_thread_result = {"result": None, "completed": False, "error": None}
            self.optimization_thread_result = {
                "result": None,
                "partial_result": None,
                "completed": False,
                "error": None,
            }
            self.replanning_thread_result = {"result": None, "completed": False, "error": None}
            previous_plan_for_continuity = auraHandler.snapshot_plan(best_trajectory)

            execution_control_duration = auraHandler.plan_control_duration(
                best_trajectory,
                0,
                default_duration=default_control_duration,
            )
            optimization_control_duration = auraHandler.plan_control_duration(
                best_trajectory,
                1,
                fallback=execution_control_duration,
                default_duration=default_control_duration,
            )

            # Tree branch set from the pre-replan path (read before replan mutates the tree in parallel)
            next_nominal_state = np.asarray(best_trajectory["states"][1], dtype=float)
            execution_nominal_target = next_nominal_state.copy()
            if int(best_trajectory.get("control_count", len(best_trajectory.get("controls", [])))) == 2:
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
                print(
                    "[AURA] one control remains after this segment; "
                    "skipping penultimate tree query/replan/optimizer and executing the saved final control next",
                    flush=True,
                )
                self.execution(next_control, execution_control_duration)
                if self.execution_thread_result["completed"]:
                    current_state = self.execution_thread_result["result"]
                    actual_trajectory.append(current_state.copy())
                    controls_trajectory.append(u.copy())
                    dense_actual_trajectory.extend(
                        [np.asarray(s, dtype=float).copy() for s in executed_curve]
                    )
                    if len(dense_actual_trajectory) == 0 or arrayDistance(
                        dense_actual_trajectory[-1], current_state, system=self.system.name
                    ) > 1e-12:
                        dense_actual_trajectory.append(np.asarray(current_state, dtype=float).copy())
                    tracking_errors.append(
                        float(
                            arrayDistance(
                                np.asarray(current_state, dtype=float),
                                execution_nominal_target,
                                system=self.system.name,
                            )
                        )
                    )

                    suffix_controls = [
                        np.asarray(best_trajectory["controls"][1], dtype=float).copy()
                    ]
                    suffix_times = [
                        auraHandler.plan_control_duration(
                            best_trajectory,
                            1,
                            fallback=execution_control_duration,
                            default_duration=default_control_duration,
                        )
                    ]
                    suffix_states = [np.asarray(current_state, dtype=float).copy()]
                    if len(best_trajectory.get("states", [])) > 2:
                        suffix_states.append(
                            np.asarray(best_trajectory["states"][2], dtype=float).copy()
                        )
                    else:
                        suffix_curve = sample_control_curve(
                            self.system,
                            np.asarray(current_state, dtype=float),
                            suffix_controls[0],
                            float(suffix_times[0]),
                            float(self.propagation_step_size),
                            include_start=False,
                        )
                        suffix_states.append(
                            np.asarray(
                                suffix_curve[-1] if suffix_curve else current_state,
                                dtype=float,
                            ).copy()
                        )

                    best_trajectory = dict(best_trajectory)
                    best_trajectory["states"] = suffix_states
                    best_trajectory["controls"] = suffix_controls
                    best_trajectory["time"] = suffix_times
                    best_trajectory["control_count"] = 1
                    best_trajectory["state_count"] = len(suffix_states)
                    best_trajectory["_selection_source"] = "final saved suffix"
                    next_control = suffix_controls[0]
                    control_duration = float(suffix_times[0])

                    if on_planning_update is not None:
                        plan_for_update = auraHandler.add_visual_reference(
                            auraHandler.snapshot_plan(best_trajectory),
                            **visual_reference_context,
                        )
                        plan_for_update["actual_states"] = [
                            np.asarray(s, dtype=float).copy() for s in dense_actual_trajectory
                        ]
                        plan_for_update["candidate_paths"] = []
                        plan_for_update["goal_solution_paths"] = []
                        plan_for_update["plan_caption"] = (
                            "Final saved suffix selected; no extra tree query or replan is "
                            "needed before the last control."
                        )
                        auraHandler.annotate_visual_cost(plan_for_update, **visual_cost_context)
                        on_planning_update(
                            index + 1,
                            plan_for_update,
                            np.asarray(current_state, dtype=float),
                        )
                    index += 1
                    break

                print(f"Execution thread failed: {self.execution_thread_result.get('error')}")
                aborted = True
                failure_reason = f"execution_failed:{self.execution_thread_result.get('error')}"
                break

            children_states, children_controls, children_lookup_info = getChildrenStates(
                self.planner.ss,
                next_nominal_state,
                system=self.system.name,
                return_metadata=True,
            )
            children_source = "planner tree"
            if (
                (not children_states or not children_controls)
                and len(best_trajectory.get("states", [])) > 2
                and len(best_trajectory.get("controls", [])) > 1
            ):
                children_states = [np.asarray(best_trajectory["states"][2], dtype=float)]
                children_controls = [np.asarray(best_trajectory["controls"][1], dtype=float)]
                children_source = "solution path fallback"
                children_lookup_info = {
                    "match": "solution_path_fallback",
                    "nearest_distance": float("nan"),
                    "num_children": len(children_states),
                    "num_vertices": children_lookup_info.get("num_vertices", 0),
                }
            if children_states and children_controls and self.optimizer_max_children > 0:
                max_children = int(self.optimizer_max_children)
                num_children_before = min(len(children_states), len(children_controls))
                if num_children_before > max_children:
                    reference_state = (
                        np.asarray(best_trajectory["states"][2], dtype=float)
                        if len(best_trajectory.get("states", [])) > 2
                        else np.asarray(self.goal_state, dtype=float)
                    )
                    goal_state_np = np.asarray(self.goal_state, dtype=float)
                    scored_children = []
                    for child_idx, child_state in enumerate(children_states[:num_children_before]):
                        child_np = np.asarray(child_state, dtype=float)
                        progress_score = float(
                            arrayDistance(child_np, goal_state_np, system=self.system.name)
                        )
                        local_score = float(
                            arrayDistance(child_np, reference_state, system=self.system.name)
                        )
                        scored_children.append((0.65 * progress_score + 0.35 * local_score, child_idx))
                    keep_indices = [
                        idx for _, idx in sorted(scored_children, key=lambda item: item[0])[:max_children]
                    ]
                    children_states = [children_states[idx] for idx in keep_indices]
                    children_controls = [children_controls[idx] for idx in keep_indices]
                    children_source = (
                        f"{children_source} (top {len(keep_indices)}/{num_children_before})"
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
            optimization_wall_budget = max(0.05, float(execution_control_duration))

            # Execute, replan, and runOptimizer in parallel
            executeThread = threading.Thread(
                target=self.execution,
                args=(next_control, execution_control_duration),
            )
            executeThread.start()

            optimization_ran_this_step = bool(children_states and children_controls)
            if optimization_ran_this_step:
                optimization_slot = self.optimization_thread_result
                optimization_stop_event = threading.Event()
                optimizationThread = threading.Thread(
                    target=self.optimization,
                    args=(
                        self.system,
                        optimization_anchor,
                        children_states,
                        children_controls,
                        optimization_control_duration,
                        optimization_wall_budget,
                    ),
                    kwargs={
                        "result_slot": optimization_slot,
                        "stop_event": optimization_stop_event,
                    },
                )
                optimizationThread.start()
            else:
                optimizationThread = None
                optimization_stop_event = None
                self.optimization_thread_result = {
                    "result": None,
                    "partial_result": None,
                    "completed": True,
                    "error": "no_branch_children",
                }

            replanningThread = threading.Thread(
                target=self.replanning,
                args=(self.planner, execution_control_duration),
                kwargs={"result_slot": self.replanning_thread_result},
            )
            replanningThread.start()

            # Wait for one control-duration wall-clock window, then use whatever
            # finished. AURA must not block indefinitely on an optimizer iteration.
            parallel_deadline = time.monotonic() + float(execution_control_duration)
            executeThread.join(max(0.0, parallel_deadline - time.monotonic()))
            execution_timeout_slack = float(
                getattr(self.simulator, "config", {}).get("execution_timeout_slack", 0.0)
            )
            if executeThread.is_alive() and execution_timeout_slack > 0.0:
                executeThread.join(float(execution_timeout_slack))
            if executeThread.is_alive():
                try:
                    self.simulator.stop()
                except Exception:
                    pass
                executeThread.join()
                if not self.execution_thread_result.get("completed"):
                    self.execution_thread_result = {
                        "result": None,
                        "completed": False,
                        "error": f"timeout_after_{float(execution_control_duration):.3f}s",
                    }
            if optimizationThread is not None:
                optimizationThread.join(max(0.0, parallel_deadline - time.monotonic()))
                if optimizationThread.is_alive():
                    if optimization_stop_event is not None:
                        optimization_stop_event.set()
                    partial_result = optimization_slot.get("partial_result")
                    if partial_result is not None:
                        partial_result = dict(partial_result)
                        partial_result["partial_result"] = True
                        partial_result["timed_out"] = True
                        partial_result["stopped"] = True
                        optimization_slot["result"] = partial_result
                        optimization_slot["completed"] = True
                        optimization_slot["error"] = (
                            f"partial_after_{float(execution_control_duration):.3f}s"
                        )
                    else:
                        optimization_slot["result"] = None
                        optimization_slot["completed"] = False
                        optimization_slot["error"] = (
                            f"timeout_after_{float(execution_control_duration):.3f}s"
                        )
                    optimizationThread.join()
            replanningThread.join(max(0.0, parallel_deadline - time.monotonic()))
            if replanningThread.is_alive():
                timeout_msg = f"timeout_after_{float(execution_control_duration):.3f}s"
                print(
                    "[AURA] replanning exceeded the control window; "
                    "waiting for OMPL thread to finish before touching the planner again",
                    flush=True,
                )
                replanningThread.join()
                if not self.replanning_thread_result.get("completed"):
                    self.replanning_thread_result["result"] = None
                    self.replanning_thread_result["completed"] = False
                    self.replanning_thread_result["error"] = timeout_msg
                elif self.replanning_thread_result.get("error") is None:
                    self.replanning_thread_result["error"] = (
                        f"late_after_{float(execution_control_duration):.3f}s"
                    )

            # Extract execution result
            noise_row: dict | None = None
            planner_target_row: dict | None = None
            if self.execution_thread_result["completed"]:
                current_state = self.execution_thread_result["result"]
                actual_trajectory.append(current_state.copy())
                controls_trajectory.append(u.copy())
                dense_actual_trajectory.extend(
                    [np.asarray(s, dtype=float).copy() for s in executed_curve]
                )
                if len(dense_actual_trajectory) == 0 or arrayDistance(
                    dense_actual_trajectory[-1], current_state, system=self.system.name
                ) > 1e-12:
                    dense_actual_trajectory.append(np.asarray(current_state, dtype=float).copy())
                selected_prediction_state = np.asarray(
                    optimization_anchor,
                    dtype=float,
                ).reshape(-1)
                actual_state_np = np.asarray(current_state, dtype=float).reshape(-1)
                noise_xy = actual_state_np[:2] - selected_prediction_state[:2]
                noise_theta = auraHandler.wrap_to_pi(
                    actual_state_np[2] - selected_prediction_state[2]
                )
                step_ompl = float(
                    arrayDistance(
                        actual_state_np,
                        selected_prediction_state,
                        system=self.system.name,
                    )
                )
                _, dxy_step, _ = auraHandler.se2_breakdown(
                    actual_state_np,
                    selected_prediction_state,
                    self.system.name,
                )
                noise_row = {
                    "predicted": np.round(selected_prediction_state[:3], 6).tolist(),
                    "actual": np.round(actual_state_np[:3], 6).tolist(),
                    "dx": float(noise_xy[0]),
                    "dy": float(noise_xy[1]),
                    "dtheta": float(noise_theta),
                    "ompl": step_ompl,
                    "dxy": dxy_step,
                }
                target_ompl, target_dxy, target_dtheta = auraHandler.se2_breakdown(
                    actual_state_np,
                    execution_nominal_target,
                    self.system.name,
                )
                planner_target_row = {
                    "target": np.round(execution_nominal_target[:3], 6).tolist(),
                    "ompl": target_ompl,
                    "dxy": target_dxy,
                    "dtheta": target_dtheta,
                }
                tracking_errors.append(step_ompl)
            else:
                print(f"Execution thread failed: {self.execution_thread_result.get('error')}")
                aborted = True
                failure_reason = f"execution_failed:{self.execution_thread_result.get('error')}"
                break

            # Extract optimization result
            if self.optimization_thread_result["completed"]:
                optimization_result = self.optimization_thread_result["result"]
            else:
                print(
                    "Optimization thread failed; falling back to planner control: "
                    f"{self.optimization_thread_result.get('error')}"
                )
                optimization_result = None

            # Extract replanning result
            raw_ompl_root_dist: float | None = None
            plan_continuity_info: dict = {}
            visual_candidate_paths: list[dict] = []
            visual_goal_solution_paths: list[dict] = []
            plan_change_caption = ""
            if self.replanning_thread_result["completed"]:
                solutions = self.replanning_thread_result["result"]
                visual_goal_solution_paths = auraHandler.visual_goal_solution_paths(solutions)
                candidates, plan_continuity_info = auraHandler.reachable_plan_candidates(
                    solutions,
                    previous_plan_for_continuity,
                    np.asarray(current_state, dtype=float),
                    system_name=self.system.name,
                    propagation_step_size=self.propagation_step_size,
                    continuity_max_distance=continuity_max_distance,
                )
                if not candidates:
                    print(
                        "[WARNING] No resolved solution is continuous from the executed pose; "
                        "stopping before selecting a disconnected branch."
                    )
                    aborted = True
                    failure_reason = "no_continuous_resolved_solution"
                    break
                chosen_sol = candidates[0]
                raw_ompl_root_dist = float(chosen_sol.get("_continuity_dist", 0.0))
                best_trajectory = chosen_sol
                visual_candidate_paths = auraHandler.visual_candidate_paths(candidates)
                plan_change_caption = auraHandler.plan_change_caption(
                    best_trajectory,
                    plan_continuity_info,
                    candidates,
                    goal_solution_count=len(solutions or []),
                )
            else:
                replanning_error = self.replanning_thread_result.get("error")
                recoverable_replanning_miss = (
                    replanning_error is None
                    or str(replanning_error).startswith("timeout_after_")
                    or str(replanning_error) == "no_replan_solutions"
                )
                if recoverable_replanning_miss:
                    print(
                        "[WARNING] Replanning did not produce a usable update "
                        f"({replanning_error}); continuing with the previous plan suffix."
                    )
                    candidates, plan_continuity_info = auraHandler.reachable_plan_candidates(
                        [],
                        previous_plan_for_continuity,
                        np.asarray(current_state, dtype=float),
                        system_name=self.system.name,
                        propagation_step_size=self.propagation_step_size,
                        continuity_max_distance=continuity_max_distance,
                    )
                    if not candidates:
                        print(
                            "[WARNING] Replanning missed and no previous plan suffix is "
                            "continuous from the executed pose; stopping."
                        )
                        aborted = True
                        failure_reason = (
                            "no_continuous_previous_plan_after_replanning_miss"
                        )
                        break
                    chosen_sol = candidates[0]
                    raw_ompl_root_dist = float(chosen_sol.get("_continuity_dist", 0.0))
                    best_trajectory = chosen_sol
                    visual_candidate_paths = auraHandler.visual_candidate_paths(candidates)
                    plan_change_caption = auraHandler.plan_change_caption(
                        best_trajectory,
                        plan_continuity_info,
                        candidates,
                        goal_solution_count=0,
                    )
                else:
                    print(f"Replanning thread failed: {replanning_error}")
                    aborted = True
                    failure_reason = f"replanning_failed:{replanning_error}"
                    break

            control_duration = auraHandler.plan_control_duration(
                best_trajectory,
                0,
                fallback=optimization_control_duration,
                default_duration=default_control_duration,
            )

            next_state = best_trajectory["states"][1]

            next_control = self.pick_next_control(
                self.system,
                optimization_result,
                current_state,
                next_state,
                children_states,
                children_controls,
                control_duration,
                fallback_control=best_trajectory["controls"][0],
                optimization_error=self.optimization_thread_result.get("error"),
            )
            chosen_curve_valid = next_control is not None and self._control_curve_valid(
                np.asarray(current_state, dtype=float),
                np.asarray(next_control, dtype=float),
                float(control_duration),
            )
            self.last_control_decision["chosen_curve_valid"] = bool(chosen_curve_valid)
            stop_after_table = False
            recovery_info = {"attempted": False}
            if not chosen_curve_valid:
                self.last_control_decision["reason"] = (
                    f"{self.last_control_decision.get('reason', 'unknown')}; "
                    "chosen_curve_in_collision; recovery_replan_triggered"
                )
                recovery_budget = float(
                    getattr(
                        self.planner,
                        "recovery_replanning_time",
                        max(float(execution_control_duration), float(control_duration)),
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
                        fallback=max(float(execution_control_duration), float(control_duration)),
                        default_duration=default_control_duration,
                    )
                    recovery_control = np.asarray(
                        recovery_solution["controls"][0],
                        dtype=float,
                    )
                    if not self._control_curve_valid(
                        np.asarray(current_state, dtype=float),
                        recovery_control,
                        float(recovery_duration),
                    ):
                        continue

                    recovery_solution["_selection_source"] = "recovery fresh plan"
                    recovery_solution["_trim_index"] = 0
                    recovery_solution["_continuity_dist"] = 0.0
                    best_trajectory = recovery_solution
                    control_duration = float(recovery_duration)
                    next_control = recovery_control
                    chosen_curve_valid = True
                    raw_ompl_root_dist = 0.0
                    recovery_info["status"] = "success"
                    recovery_info["selected_controls"] = int(
                        recovery_solution.get("control_count", len(recovery_solution.get("controls", [])))
                    )
                    plan_continuity_info = {
                        "accepted": 1,
                        "rejected": 0,
                        "best_rejected_dist": float("inf"),
                        "max_dist": 0.0,
                    }
                    self.last_control_decision["source"] = "recovery"
                    self.last_control_decision[
                        "reason"
                    ] = "fresh_replan_from_actual_pose_after_invalid_candidates"
                    self.last_control_decision["chosen_curve_valid"] = True
                    visual_candidate_paths = auraHandler.visual_candidate_paths([best_trajectory])
                    visual_goal_solution_paths = auraHandler.visual_goal_solution_paths([best_trajectory])
                    plan_change_caption = auraHandler.plan_change_caption(
                        best_trajectory,
                        plan_continuity_info,
                        [best_trajectory],
                        recovery_info=recovery_info,
                        goal_solution_count=1,
                    )
                    break

                if not chosen_curve_valid:
                    stop_after_table = True
                    failure_reason = "recovery_replan_failed_after_invalid_candidates"
            state_diff_post = float(
                arrayDistance(
                    np.asarray(current_state, dtype=float),
                    np.asarray(best_trajectory["states"][0], dtype=float),
                    system=self.system.name,
                )
            )
            track_warn = (
                previous_step_diff is not None
                and state_diff_post > previous_step_diff + 1e-6
                and self.last_control_decision.get("source") == "optimized"
            )
            previous_step_diff = state_diff_post

            # --- per-step summary table ---
            lcd = self.last_control_decision
            opt_res = optimization_result
            opt_init = None
            opt_fin = None
            if opt_res:
                opt_init = opt_res.get("initial_loss")
                lh = opt_res.get("loss_history") or []
                if opt_init is None and lh:
                    opt_init = lh[0]
                opt_fin = opt_res.get("final_loss")
                if opt_fin is None and lh:
                    opt_fin = lh[-1]

            if not optimization_ran_this_step:
                loss_str = "(optimizer not run — no branch children)"
            elif opt_res is None:
                opt_error = self.optimization_thread_result.get("error")
                if opt_error:
                    loss_str = f"(optimizer returned no result: {opt_error})"
                else:
                    loss_str = "(optimizer returned no result)"
            elif opt_fin is None and opt_init is None:
                loss_str = "(loss not recorded)"
            else:
                loss_str = f"{auraHandler.fmt_scalar(opt_init)} → {auraHandler.fmt_scalar(opt_fin)}"
                if opt_res:
                    requested_states = opt_res.get("requested_num_states")
                    effective_states = opt_res.get("effective_num_states")
                    requested_steps = opt_res.get("requested_num_steps")
                    effective_steps = opt_res.get("steps_completed")
                    num_children = opt_res.get("num_children")
                    device = opt_res.get("device")
                    if effective_states is not None or effective_steps is not None:
                        loss_str += (
                            f"  (used {effective_states or '—'}/{requested_states or '—'} "
                            f"samples, {effective_steps or '—'}/{requested_steps or '—'} "
                            f"Adam steps, children {num_children or '—'}, "
                            f"device {device or '—'})"
                        )
                if opt_res and opt_res.get("timed_out"):
                    loss_str += (
                        f"  (wall budget hit after "
                        f"{opt_res.get('steps_completed', '—')} Adam steps)"
                    )
                if opt_res and opt_res.get("partial_result"):
                    loss_str += "  (partial anytime result used)"

            pre_track = (
                f"OMPL {auraHandler.fmt_scalar(state_diff_pre)}  |dxy| {auraHandler.fmt_scalar(dxy_vs_plan)}  "
                f"|dθ| {auraHandler.fmt_scalar(dth_vs_plan)} rad"
            )
            noise_block = "—"
            if noise_row:
                noise_block = (
                    f"OMPL {auraHandler.fmt_scalar(noise_row['ompl'])}  |dxy| {auraHandler.fmt_scalar(noise_row['dxy'])}\n"
                    f"dx={noise_row['dx']:.6f} dy={noise_row['dy']:.6f} dθ={noise_row['dtheta']:+.6f} rad\n"
                    f"predicted {noise_row['predicted']}\n"
                    f"actual   {noise_row['actual']}"
                )
            planner_target_block = "—"
            if planner_target_row:
                planner_target_block = (
                    f"OMPL {auraHandler.fmt_scalar(planner_target_row['ompl'])}  "
                    f"|dxy| {auraHandler.fmt_scalar(planner_target_row['dxy'])}  "
                    f"|dθ| {auraHandler.fmt_scalar(planner_target_row['dtheta'])} rad\n"
                    f"planner target {planner_target_row['target']}"
                )

            lookup_match = children_lookup_info.get("match", "—")
            lookup_dist = children_lookup_info.get("nearest_distance")
            if lookup_match == "exact":
                lookup_desc = "exact tree vertex"
            elif lookup_match == "nearest":
                lookup_desc = f"nearest tree vertex, distance {auraHandler.fmt_scalar(lookup_dist)}"
            elif lookup_match == "solution_path_fallback":
                lookup_desc = "solution path fallback"
            else:
                lookup_desc = f"{lookup_match}, nearest distance {auraHandler.fmt_scalar(lookup_dist)}"
            branch_target_block = (
                f"target {np.round(next_nominal_state[:3], 6).tolist()}\n"
                f"{lookup_desc}; children {len(children_states)}"
            )

            control_block = (
                f"to nominal child:  original u  {auraHandler.fmt_scalar(lcd.get('original_distance'))}   "
                f"optimized u  {auraHandler.fmt_scalar(lcd.get('optimized_distance'))}  "
                f"(OMPL vs plan.states[1])\n"
                f"predicted state gap ‖f(s,u_orig)−f(s,u_opt)‖:  "
                f"OMPL {auraHandler.fmt_scalar(lcd.get('between_pred_ompl'))}  "
                f"|dxy| {auraHandler.fmt_scalar(lcd.get('between_pred_dxy'))}  "
                f"|dθ| {auraHandler.fmt_scalar(lcd.get('between_pred_dtheta'))} rad\n"
                f"execution curve valid: original {lcd.get('original_curve_valid', '—')}   "
                f"optimized {lcd.get('optimized_curve_valid', '—')}   "
                f"chosen {lcd.get('chosen_curve_valid', '—')}"
            )

            table_rows: list[tuple[str, str]] = [
                (
                    "Durations",
                    (
                        f"execute/replan {auraHandler.fmt_scalar(execution_control_duration)} s; "
                        f"optimize horizon {auraHandler.fmt_scalar(optimization_control_duration)} s; "
                        f"optimizer budget {auraHandler.fmt_scalar(optimization_wall_budget)} s; "
                        f"next control {auraHandler.fmt_scalar(control_duration)} s"
                    ),
                ),
                ("Optimizer branch source", children_source),
                ("Optimizer branch target", branch_target_block),
                (
                    "Pre-step track (actual vs plan root)",
                    pre_track + "  [root aligned to executed pose]",
                ),
                ("Execution vs selected-control prediction", noise_block),
                ("Execution vs planner target", planner_target_block),
                ("Optimizer MSE (sampled starts → child targets)", loss_str),
                ("Control comparison", control_block),
                (
                    "Post-step vs plan root",
                    (
                        f"aligned root OMPL {auraHandler.fmt_scalar(state_diff_post)} (should be ~0);  "
                        f"selected OMPL path vertex vs pose {auraHandler.fmt_scalar(raw_ompl_root_dist)}"
                    ),
                ),
                (
                    "Plan continuity",
                    (
                        f"{best_trajectory.get('_selection_source', '—')}  "
                        f"trim idx {best_trajectory.get('_trim_index', '—')}  "
                        f"closest OMPL {auraHandler.fmt_scalar(best_trajectory.get('_continuity_dist'))}  "
                        f"accepted {plan_continuity_info.get('accepted', '—')}  "
                        f"rejected {plan_continuity_info.get('rejected', '—')}  "
                        f"max {auraHandler.fmt_scalar(plan_continuity_info.get('max_dist'))}"
                    ),
                ),
                (
                    "Chosen control",
                    f"{lcd.get('source', '?')} — {lcd.get('reason', '')}",
                ),
            ]
            if recovery_info.get("attempted"):
                table_rows.append(
                    (
                        "Recovery replan",
                        (
                            f"status {recovery_info.get('status')}  "
                            f"budget {auraHandler.fmt_scalar(recovery_info.get('budget'))} s  "
                            f"solutions {recovery_info.get('solutions')}  "
                            f"selected controls {recovery_info.get('selected_controls')}"
                        ),
                    )
                )
            if stop_after_table:
                table_rows.append(
                    (
                        "Stop reason",
                        "chosen control curve is invalid from the current state and recovery "
                        "planning did not produce a valid first curve",
                    )
                )
            if track_warn:
                table_rows.append(
                    (
                        "Note",
                        "post track increased with optimized control selected",
                    )
                )
            auraHandler.print_step_table(index, table_rows)

            if stop_after_table:
                aborted = True
                break

            lh_plot = None
            if optimization_ran_this_step and opt_res:
                lh_plot = opt_res.get("loss_history")
            if lh_plot is not None and np.size(lh_plot) > 0:
                loss_png = None
                if optimization_loss_plot_dir:
                    loss_png = os.path.join(
                        optimization_loss_plot_dir, f"loss_step_{index:04d}.png"
                    )
                auraHandler.plot_optimization_loss_history(
                    lh_plot,
                    step_index=index,
                    save_path=loss_png,
                    show_live=show_optimization_loss_plot,
                )

            if on_planning_update is not None:
                plan_for_update = auraHandler.add_visual_reference(
                    auraHandler.snapshot_plan(best_trajectory),
                    **visual_reference_context,
                )
                plan_for_update["actual_states"] = [
                    np.asarray(s, dtype=float).copy() for s in dense_actual_trajectory
                ]
                plan_for_update["candidate_paths"] = visual_candidate_paths
                plan_for_update["goal_solution_paths"] = visual_goal_solution_paths
                plan_for_update["plan_caption"] = plan_change_caption
                auraHandler.annotate_visual_cost(plan_for_update, **visual_cost_context)
                on_planning_update(index + 1, plan_for_update, np.asarray(current_state, dtype=float))

            index += 1
            if pause_each_step:
                input("Press Enter to continue...")

        # Execute the control selected by the final AURA table when the loop exits
        # normally with one segment left. This may be optimized or recovery-selected,
        # not necessarily the planner's stored first control.
        if not aborted and not reached_step_limit and best_trajectory.get("controls"):
            final_control = (
                np.asarray(next_control, dtype=float)
                if next_control is not None
                else np.asarray(best_trajectory["controls"][0], dtype=float)
            )
            final_control_duration = auraHandler.plan_control_duration(
                best_trajectory,
                0,
                fallback=control_duration,
                default_duration=default_control_duration,
            )
            final_pre_exec = np.asarray(current_state, dtype=float)
            print(
                "[AURA final] executing selected final control "
                f"for {float(final_control_duration):.3f}s"
            )
            final_curve = sample_control_curve(
                self.system,
                final_pre_exec,
                final_control,
                float(final_control_duration),
                float(self.propagation_step_size),
                include_start=False,
            )
            self.execution_thread_result = {"result": None, "completed": False, "error": None}
            executionThread = threading.Thread(
                target=self.execution,
                args=(final_control, final_control_duration),
            )
            executionThread.start()
            executionThread.join()
            if self.execution_thread_result["completed"]:
                current_state = self.execution_thread_result["result"]
                actual_trajectory.append(current_state.copy())
                dense_actual_trajectory.extend(
                    [np.asarray(s, dtype=float).copy() for s in final_curve]
                )
                if len(dense_actual_trajectory) == 0 or arrayDistance(
                    dense_actual_trajectory[-1], current_state, system=self.system.name
                ) > 1e-12:
                    dense_actual_trajectory.append(np.asarray(current_state, dtype=float).copy())
                nominal_trajectory.append(best_trajectory["states"][-1].copy())
                controls_trajectory.append(np.asarray(final_control, dtype=float).copy())
                final_expected = final_curve[-1] if final_curve else final_pre_exec
                final_report_state = np.asarray(final_expected, dtype=float)
                tracking_errors.append(
                    float(
                        arrayDistance(
                            np.asarray(current_state, dtype=float),
                            np.asarray(final_expected, dtype=float),
                            system=self.system.name,
                        )
                    )
                )
                if on_planning_update is not None:
                    plan_for_update = auraHandler.add_visual_reference(
                        auraHandler.snapshot_plan(best_trajectory),
                        **visual_reference_context,
                    )
                    plan_for_update["actual_states"] = [
                        np.asarray(s, dtype=float).copy() for s in dense_actual_trajectory
                    ]
                    plan_for_update["_final_update"] = True
                    plan_for_update["plan_caption"] = auraHandler.plan_change_caption(
                        best_trajectory,
                        {},
                        [],
                        final=True,
                    )
                    auraHandler.annotate_visual_cost(
                        plan_for_update,
                        **visual_cost_context,
                        final=True,
                    )
                    on_planning_update(
                        index + 1,
                        plan_for_update,
                        np.asarray(current_state, dtype=float),
                    )
            else:
                aborted = True
                failure_reason = f"final_execution_failed:{self.execution_thread_result.get('error')}"

        cost = 0.0
        for i in range(len(actual_trajectory) - 1):
            cost += arrayDistance(
                actual_trajectory[i], actual_trajectory[i + 1], system=self.system.name
            )

        return self.AURAResult(
            num_controls=max(0, len(controls_trajectory)),
            final_state=np.asarray(actual_trajectory[-1], dtype=float),
            cost=float(cost),
            tracking_error_mean=float(np.mean(tracking_errors)) if tracking_errors else 0.0,
            tracking_error_list=tracking_errors,
            controls_trajectory=[np.asarray(c, dtype=float) for c in controls_trajectory],
            states_trajectory=[np.asarray(s, dtype=float) for s in actual_trajectory],
            dense_states_trajectory=[
                np.asarray(s, dtype=float) for s in dense_actual_trajectory
            ],
            final_plan_states=[
                np.asarray(s, dtype=float) for s in best_trajectory.get("states", [])
            ],
            final_plan_controls=[
                np.asarray(c, dtype=float) for c in best_trajectory.get("controls", [])
            ],
            final_planned_state=(
                np.asarray(final_report_state, dtype=float)
                if final_report_state is not None
                else (
                    np.asarray(best_trajectory["states"][-1], dtype=float)
                    if best_trajectory.get("states")
                    else None
                )
            ),
            status="failure" if aborted else "success",
            failure_reason=failure_reason,
        )

    def optimization(
        self,
        system: System,
        next_nominal_state: np.ndarray,
        children_states: list[np.ndarray],
        children_controls: list[np.ndarray],
        control_duration: float,
        wall_time_budget: float | None = None,
        result_slot: dict | None = None,
        stop_event=None,
    ):
        slot = self.optimization_thread_result if result_slot is None else result_slot
        try:
            result = runOptimizer(
                system=system.name,
                nextState=np.asarray(next_nominal_state, dtype=float),
                childrenStatesArray=children_states,
                childrenControlsArray=children_controls,
                optModel=self.opt_model,
                numStates=self.optimizer_num_states,
                posSTD=self.optimizer_pos_std,
                rotSTD=self.optimizer_rot_std,
                velSTD=self.optimizer_vel_std,
                controlDuration=float(control_duration),
                integrationStepSize=float(self.propagation_step_size),
                numSteps=self.optimizer_num_steps,
                learningRate=self.optimizer_learning_rate,
                maxWallTime=wall_time_budget,
                stopEvent=stop_event,
                partialResultCallback=lambda partial_result: auraHandler.store_optimizer_partial_result(
                    slot,
                    partial_result,
                ),
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

    def execution(self, control: np.ndarray, duration: float):
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

    def replanning(
        self,
        planner: OMPL_Planner,
        time_budget: float,
        result_slot: dict | None = None,
    ):
        """`OMPL_Planner.replan` returns (solutions, ss); downstream expects the list only."""
        slot = self.replanning_thread_result if result_slot is None else result_slot
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

    def sample_random_state(
        self,
        system: str,
        state: np.ndarray,
        num_states: int = 1000,
        pos_std: float = 0.003,
        rot_std: float = 0.05,
    ):
        sampled_states = []
        system_key = {
            "kinematic_car": "kinematic_car",
            "pushing_object": "pushing_object",
            "double_integrator": "double_integrator",
        }.get(system, system)

        if system_key in ("kinematic_car", "pushing_object"):
            if hasattr(state, "getX"):
                # utils.state2list uses legacy labels for SE2 extraction.
                state_list = state2list(state, "simple_car")
            else:
                state_list = np.asarray(state, dtype=float).reshape(-1).tolist()

            for _ in range(num_states):
                noisy_x = state_list[0] + np.random.normal(0.0, pos_std)
                noisy_y = state_list[1] + np.random.normal(0.0, pos_std)
                noisy_yaw = state_list[2] + np.random.normal(0.0, rot_std)
                while noisy_yaw > np.pi:
                    noisy_yaw -= 2 * np.pi
                while noisy_yaw < -np.pi:
                    noisy_yaw += 2 * np.pi
                sampled_states.append([noisy_x, noisy_y, noisy_yaw])
            return sampled_states

        if system_key == "double_integrator":
            if isinstance(state, (list, tuple, np.ndarray)):
                state_list = np.asarray(state, dtype=float).reshape(-1).tolist()
            else:
                # OMPL RealVectorState fallback.
                state_list = [float(state[i]) for i in range(3)]
            if len(state_list) < 3:
                raise ValueError(
                    f"double_integrator expects state with at least 3 values, got {len(state_list)}"
                )

            for _ in range(num_states):
                noisy = [state_list[i] + np.random.normal(0.0, pos_std) for i in range(3)]
                sampled_states.append(noisy)
            return sampled_states

        raise ValueError(f"Unsupported system for sampling: {system_key}")

    def _predict_control_endpoints(
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
        step_size = max(float(self.propagation_step_size), 1e-9)
        n_steps = max(1, int(np.ceil(duration / step_size)))
        dt = duration / float(n_steps)

        if system.name == "kinematic_car":
            states = np.repeat(start[:3].reshape(1, 3), controls_np.shape[0], axis=0)
            u_vel = controls_np[:, 0]
            u_phi = controls_np[:, 1]
            wheelbase = getattr(system, "_wheelbase", 0.1385 + 0.158)
            for _ in range(n_steps):
                yaw = states[:, 2]
                states[:, 0] = states[:, 0] + u_vel * np.cos(yaw) * dt
                states[:, 1] = states[:, 1] + u_vel * np.sin(yaw) * dt
                yaw_dot = (u_vel / wheelbase) * np.tan(u_phi)
                states[:, 2] = (states[:, 2] + yaw_dot * dt + np.pi) % (2 * np.pi) - np.pi
            return states

        if system.name == "double_integrator":
            states = np.repeat(start[:6].reshape(1, 6), controls_np.shape[0], axis=0)
            acc = controls_np[:, :3]
            states[:, :3] = states[:, :3] + states[:, 3:6] * duration + 0.5 * acc * (duration**2)
            states[:, 3:6] = states[:, 3:6] + acc * duration
            return states

        if system.name in ("pushing", "pushing_object"):
            # The learned pushing propagator maps one push control to one SE2
            # delta. It is not a continuous-time dynamics model, so sampling a
            # "curve" would apply the same push repeatedly and badly distort
            # candidate ranking.
            return np.asarray(
                [
                    np.asarray(system.propagate(start.copy(), ctrl, duration), dtype=float).reshape(-1)
                    for ctrl in controls_np
                ],
                dtype=float,
            )

        predicted_states = []
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

    def _batch_state_distance(
        self, target_state: np.ndarray, states: np.ndarray, system_name: str
    ) -> np.ndarray:
        target = np.asarray(target_state, dtype=float).reshape(-1)
        states_np = np.asarray(states, dtype=float)
        if states_np.ndim == 1:
            states_np = states_np.reshape(1, -1)
        if system_name in ("simple_car", "kinematic_car", "pushing", "pushing_object"):
            dxy = np.linalg.norm(states_np[:, :2] - target[:2], axis=1)
            dtheta = (states_np[:, 2] - target[2] + np.pi) % (2 * np.pi) - np.pi
            return np.sqrt(dxy * dxy + dtheta * dtheta)
        if system_name == "double_integrator":
            return np.linalg.norm(states_np[:, :6] - target[:6], axis=1)
        return np.asarray(
            [arrayDistance(target, pred, system=system_name) for pred in states_np],
            dtype=float,
        )

    def _control_curve_valid(
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
            endpoint = np.asarray(
                self.system.propagate(start_state, control, float(duration)),
                dtype=float,
            ).reshape(-1)
            return all(
                is_state_array_valid(
                    s,
                    system=self.system.name,
                    config=config,
                    safety_radius_override=execution_safety_radius,
                )
                for s in (np.asarray(start_state, dtype=float).reshape(-1), endpoint)
            )
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
        children_states: list[np.ndarray],
        children_controls: list[np.ndarray],
        control_duration: float,
        fallback_control: np.ndarray | None = None,
        optimization_error: str | None = None,
    ):
        if not optimization_result or "optimized_controls" not in optimization_result:
            reason = "missing_optimization_result"
            if optimization_error:
                reason = f"{reason}:{optimization_error}"
            self.last_control_decision = {
                "source": "planner",
                "reason": reason,
                "original_distance": float("nan"),
                "optimized_distance": float("nan"),
                "between_pred_ompl": float("nan"),
                "between_pred_dxy": float("nan"),
                "between_pred_dtheta": float("nan"),
                "original_curve_valid": None,
                "optimized_curve_valid": None,
            }
            return auraHandler.planner_fallback_control(children_controls, fallback_control)

        optimized_controls = auraHandler.to_numpy_state_control(
            optimization_result.get("optimized_controls"),
            system.name,
        )
        start_states = auraHandler.to_numpy_state_control(
            optimization_result.get("start_states"),
            system.name,
        )
        target_states = auraHandler.to_numpy_state_control(
            optimization_result.get("target_states"),
            system.name,
        )

        if (
            optimized_controls is None
            or start_states is None
            or target_states is None
            or len(children_states) == 0
        ):
            self.last_control_decision = {
                "source": "planner",
                "reason": "invalid_optimizer_tensors_or_no_children",
                "original_distance": float("nan"),
                "optimized_distance": float("nan"),
                "between_pred_ompl": float("nan"),
                "between_pred_dxy": float("nan"),
                "between_pred_dtheta": float("nan"),
                "original_curve_valid": None,
                "optimized_curve_valid": None,
            }
            return auraHandler.planner_fallback_control(children_controls, fallback_control)

        num_children = len(children_states)
        if optimized_controls.ndim == 1:
            optimized_controls = optimized_controls.reshape(1, -1)
        if start_states.ndim == 1:
            start_states = start_states.reshape(1, -1)
        if target_states.ndim == 1:
            target_states = target_states.reshape(1, -1)

        if optimized_controls.shape[0] < num_children:
            self.last_control_decision = {
                "source": "planner",
                "reason": "optimizer_output_too_small",
                "original_distance": float("nan"),
                "optimized_distance": float("nan"),
                "between_pred_ompl": float("nan"),
                "between_pred_dxy": float("nan"),
                "between_pred_dtheta": float("nan"),
                "original_curve_valid": None,
                "optimized_curve_valid": None,
            }
            return auraHandler.planner_fallback_control(children_controls, fallback_control)

        sampling_num_states = max(1, optimized_controls.shape[0] // num_children)
        control_dim = optimized_controls.shape[-1]
        state_dim = start_states.shape[-1]

        optimized_controls = optimized_controls.reshape(
            num_children, sampling_num_states, control_dim
        )
        start_states = start_states.reshape(num_children, sampling_num_states, state_dim)
        target_states = target_states.reshape(num_children, sampling_num_states, state_dim)

        children_samples = target_states[:, 0, :]
        actual_next = auraHandler.to_numpy_state_control(next_state, system.name).reshape(-1)
        actual_current = auraHandler.to_numpy_state_control(current_state, system.name).reshape(-1)

        closest_child_idx = int(
            np.argmin(
                [
                    arrayDistance(actual_next, child_sample, system=system.name)
                    for child_sample in children_samples
                ]
            )
        )

        candidate_controls = optimized_controls[closest_child_idx]
        predicted_states = self._predict_control_endpoints(
            system,
            actual_current.copy(),
            candidate_controls,
            float(control_duration),
        )
        distances = self._batch_state_distance(actual_next, predicted_states, system.name)
        best_idx = int(np.argmin(distances))
        best_optimized_control = np.asarray(candidate_controls[best_idx], dtype=float)

        # Compare optimized candidate against original branch control and pick the better one.
        original_control = None
        if len(children_controls) > closest_child_idx:
            original_control = auraHandler.to_numpy_state_control(
                children_controls[closest_child_idx],
                system.name,
            ).reshape(-1)
        if original_control is None:
            self.last_control_decision = {
                "source": "optimized",
                "reason": "no_original_control_for_branch",
                "original_distance": float("nan"),
                "optimized_distance": float("nan"),
                "between_pred_ompl": float("nan"),
                "between_pred_dxy": float("nan"),
                "between_pred_dtheta": float("nan"),
                "original_curve_valid": None,
                "optimized_curve_valid": None,
            }
            return best_optimized_control

        predicted_pair = self._predict_control_endpoints(
            system,
            actual_current.copy(),
            np.vstack([original_control, best_optimized_control]),
            float(control_duration),
        )
        predicted_original = predicted_pair[0]
        predicted_optimized = predicted_pair[1]
        original_distance = arrayDistance(actual_next, predicted_original, system=system.name)
        optimized_distance = arrayDistance(actual_next, predicted_optimized, system=system.name)
        bp_ompl, bp_dxy, bp_dth = auraHandler.se2_breakdown(
            np.asarray(predicted_original, dtype=float),
            np.asarray(predicted_optimized, dtype=float),
            system.name,
        )
        original_curve_valid = self._control_curve_valid(
            actual_current.copy(), original_control, float(control_duration)
        )
        optimized_curve_valid = self._control_curve_valid(
            actual_current.copy(), best_optimized_control, float(control_duration)
        )

        if optimized_curve_valid and (not original_curve_valid or optimized_distance <= original_distance):
            self.last_control_decision = {
                "source": "optimized",
                "reason": (
                    "optimized_curve_valid_and_planner_curve_invalid"
                    if not original_curve_valid
                    else "optimized_distance_better_or_equal"
                ),
                "original_distance": float(original_distance),
                "optimized_distance": float(optimized_distance),
                "between_pred_ompl": float(bp_ompl),
                "between_pred_dxy": float(bp_dxy),
                "between_pred_dtheta": float(bp_dth),
                "original_curve_valid": bool(original_curve_valid),
                "optimized_curve_valid": bool(optimized_curve_valid),
            }
            return best_optimized_control
        self.last_control_decision = {
            "source": "planner",
            "reason": (
                "optimized_curve_in_collision"
                if not optimized_curve_valid and original_curve_valid
                else "planner_distance_better"
            ),
            "original_distance": float(original_distance),
            "optimized_distance": float(optimized_distance),
            "between_pred_ompl": float(bp_ompl),
            "between_pred_dxy": float(bp_dxy),
            "between_pred_dtheta": float(bp_dth),
            "original_curve_valid": bool(original_curve_valid),
            "optimized_curve_valid": bool(optimized_curve_valid),
        }
        return original_control
