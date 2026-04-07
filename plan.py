from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, TYPE_CHECKING

import numpy as np
from ompl import base as ob
from ompl import control as oc

from planning.planning_utils import ControlCountObjective
from utils.utils import isStateValid, log

if TYPE_CHECKING:
    from systems import System


@dataclass
class plan:
    """Planning wrapper that owns all planning settings for plan/replan."""

    system: System
    start_state: np.ndarray
    goal_state: np.ndarray
    planner_name: str = "aorrt"
    goal_threshold: float = 0.1
    min_control_duration: int = 1
    max_control_duration: int = 5
    propagation_step_size: float = 1.0
    planning_time: float = 10.0
    pruning_radius: float = 0.1
    config: Optional[dict] = None
    visualize: bool = False

    def __post_init__(self):
        self.start_state = np.asarray(self.start_state, dtype=float)
        self.goal_state = np.asarray(self.goal_state, dtype=float)
        self.solutions = None
        self.ss = None
        self.bestSolution = None
        # Backward-compat aliases used by older call sites.
        self.latest_solutions = None
        self.latest_ss = None

    def _get_path_info(self, solution_path):
        solution_info = {}
        solution_info["state_count"] = solution_path.getStateCount()
        solution_info["control_count"] = solution_path.getControlCount()

        control_space = self.ss.getControlSpace()
        control_dimension = control_space.getDimension()

        controls_list = []
        time_list = []
        for i in range(solution_info["control_count"]):
            control = solution_path.getControl(i)
            control_values = [control[j] for j in range(control_dimension)]
            controls_list.append(control_values)
            time_list.append(solution_path.getControlDuration(i))

        solution_info["controls"] = controls_list
        solution_info["time"] = time_list

        states_list = []
        space = self.ss.getSpaceInformation().getStateSpace()
        space_type = space.getType()

        for i in range(solution_info["state_count"]):
            state = solution_path.getState(i)

            if space_type == ob.STATE_SPACE_SE2:
                state_values = [state.getX(), state.getY(), state.getYaw()]
            elif space_type == ob.STATE_SPACE_SE3:
                state_values = [
                    state.getX(),
                    state.getY(),
                    state.getZ(),
                    state.rotation().x,
                    state.rotation().y,
                    state.rotation().z,
                    state.rotation().w,
                ]
            else:
                dim = space.getDimension()
                state_values = []
                try:
                    if space_type == ob.STATE_SPACE_REAL_VECTOR:
                        state_values = [float(state[j]) for j in range(dim)]
                    else:
                        raise ValueError("not real-vector")
                except Exception:
                    try:
                        state_values = [
                            state[0][0],
                            state[0][1],
                            state[0][2],
                            state[1].w,
                            state[1].x,
                            state[1].y,
                            state[1].z,
                        ]
                    except Exception:
                        try:
                            state_values = [float(state[j]) for j in range(dim)]
                        except Exception:
                            state_values = [0.0] * dim

            states_list.append(state_values)

        solution_info["states"] = states_list
        return solution_info

    def _get_all_planner_solutions_info(self, planner):
        try:
            all_solutions = planner.getAllSolutions()
            all_solution_infos = []

            for i, solution in enumerate(all_solutions):
                try:
                    path_control = oc.PathControl(solution.path_)
                    info = self._get_path_info(path_control)
                    info["cost"] = solution.cost_.value()
                    info["solution_index"] = i
                    all_solution_infos.append(info)
                except Exception as e:
                    all_solution_infos.append(
                        {
                            "state_count": 0,
                            "control_count": 0,
                            "states": [],
                            "controls": [],
                            "cost": solution.cost_.value(),
                            "solution_index": i,
                            "error": str(e),
                        }
                    )

            all_solution_infos.sort(key=lambda x: x["cost"])
            return all_solution_infos
        except Exception:
            return []

    def getSolutionsInfo(self):
        if self.ss is None:
            return []

        planner = self.ss.getPlanner()

        try:
            if hasattr(planner, "getAllSolutions"):
                all_solution_infos = self._get_all_planner_solutions_info(planner)
                if len(all_solution_infos) > 0:
                    return all_solution_infos
        except Exception as e:
            print(f"WARNING: Error with enhanced solution tracking: {e}; using fallback.")

        solutions = self.ss.getProblemDefinition().getSolutions()
        all_solution_infos = []
        for solution in solutions:
            info = self._get_path_info(solution.path_)
            info["cost"] = solution.cost_.value()
            all_solution_infos.append(info)

        if len(all_solution_infos) == 0:
            try:
                solution_path = self.ss.getSolutionPath()
                if solution_path and solution_path.getStateCount() > 0:
                    info = self._get_path_info(solution_path)
                    try:
                        opt = self.ss.getProblemDefinition().getOptimizationObjective()
                        info["cost"] = opt.cost(solution_path).value() if opt else 0.0
                    except Exception:
                        info["cost"] = 0.0
                    all_solution_infos.append(info)
            except Exception:
                pass

        all_solution_infos.sort(key=lambda x: x["cost"])
        return all_solution_infos

    def getBestSolution(self):
        if self.solutions is None:
            return None
        if len(self.solutions) == 0:
            return None
        return self.solutions[0]

    def _assign_state_values(self, ompl_state, values: np.ndarray):
        values = np.asarray(values, dtype=float)
        if hasattr(ompl_state, "setX") and hasattr(ompl_state, "setY"):
            ompl_state.setX(float(values[0]))
            ompl_state.setY(float(values[1]))
            if hasattr(ompl_state, "setYaw") and len(values) > 2:
                ompl_state.setYaw(float(values[2]))
            return ompl_state

        for i in range(len(values)):
            ompl_state[i] = float(values[i])
        return ompl_state

    def _make_ompl_state(self, space, values: np.ndarray):
        state = space.allocState()
        return self._assign_state_values(state, values)

    def _make_goal(self, ss):
        goal_state = self._make_ompl_state(ss.getStateSpace(), self.goal_state)
        goal = ob.GoalState(ss.getSpaceInformation())
        goal.setState(goal_state)
        goal.setThreshold(float(self.goal_threshold))
        return goal

    def _make_planner(self, ss):
        planner_name = self.planner_name.lower()
        planner_type_candidates = {
            "rrt": ["RRT"],
            "est": ["EST"],
            "sst": ["SST"],
            "sststar": ["SSTstar", "SSTStar"],
            "aorrt": ["AORRT"],
            "aoest": ["AOEST"],
        }
        class_names = planner_type_candidates.get(planner_name, [])
        planner_cls = None
        for class_name in class_names:
            planner_cls = getattr(oc, class_name, None)
            if planner_cls is not None:
                break
        if planner_cls is None:
            planner_cls = oc.RRT
            log(
                f"[WARNING] Planner '{self.planner_name}' unavailable in OMPL bindings; falling back to RRT.",
                "warning",
            )

        planner = planner_cls(ss.getSpaceInformation())
        if hasattr(planner, "setPruningRadius"):
            planner.setPruningRadius(float(self.pruning_radius))
        return planner

    def _run_plan(self):
        system_name = self.system.name
        config = self.config or {}

        space, cspace = self.system.create_spaces()
        ss = oc.SimpleSetup(cspace)

        obstacle_config = config.get("obstacles", None)
        if isinstance(obstacle_config, dict):
            if (
                obstacle_config.get("circles")
                or obstacle_config.get("aabbs")
                or obstacle_config.get("boxes")
            ):
                obstacle_config["enabled"] = obstacle_config.get("enabled", True)

        ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(
                partial(
                    isStateValid,
                    ss.getSpaceInformation(),
                    system=system_name,
                    config=config,
                    obstacle_config=obstacle_config,
                )
            )
        )

        ss.setStatePropagator(oc.StatePropagatorFn(self.system.propagator_fn))
        ss.getSpaceInformation().setMinMaxControlDuration(
            int(self.min_control_duration), int(self.max_control_duration)
        )
        ss.getSpaceInformation().setPropagationStepSize(float(self.propagation_step_size))

        start = self._make_ompl_state(space, self.start_state)
        ss.setStartState(start)

        goal = self._make_goal(ss)
        ss.setGoal(goal)

        planner = self._make_planner(ss)
        ss.setPlanner(planner)

        ss.setOptimizationObjective(
            ControlCountObjective(
                ss.getSpaceInformation(), cost_per_control=float(self.propagation_step_size)
            )
        )

        if float(self.planning_time) < 0:
            print("[INFO] Using exact solution termination condition")
            ptc = ob.exactSolnPlannerTerminationCondition(ss.getProblemDefinition())
        else:
            ptc = ob.timedPlannerTerminationCondition(float(self.planning_time))

        ss.solve(ptc)

        if float(self.planning_time) < 0:
            if ss.haveExactSolutionPath():
                control_count = ss.getSolutionPath().getControlCount()
                print(
                    f"Solution found with {self.planner_name} "
                    f"({control_count} controls) - accepting first exact solution"
                )
            else:
                log(
                    "[WARNING] No exact solution found with exact solution termination condition",
                    "warning",
                )
        else:
            max_retries = 1000 if system_name == "simple_car" else 100
            max_attempts = 10
            attempts = 0

            while attempts < max_attempts:
                attempts += 1
                if ss.haveExactSolutionPath():
                    control_count = ss.getSolutionPath().getControlCount()
                    if control_count <= max_retries:
                        print(
                            f"Solution found in {float(self.planning_time):.1f}s with {self.planner_name} "
                            f"({control_count} controls)"
                        )
                        break
                    print(
                        f"[WARNING] Solution found but has {control_count} controls (> {max_retries})... "
                        f"retrying... (attempt {attempts}/{max_attempts})"
                    )
                else:
                    print(
                        f"[WARNING] No exact solution found in {float(self.planning_time):.1f}s... "
                        f"retrying... (attempt {attempts}/{max_attempts})"
                    )

                ss.getPlanner().clear()
                ss.solve(float(self.planning_time))

            if attempts >= max_attempts:
                log(
                    f"[WARNING] Maximum retry attempts ({max_attempts}) reached, using current solution",
                    "warning",
                )

        return ss

    def plan(self):
        self.ss = self._run_plan()
        self.solutions = self.getSolutionsInfo()
        self.bestSolution = self.getBestSolution()
        self.latest_solutions = self.solutions
        self.latest_ss = self.ss
        return self.solutions, self.ss

    def replan(self, new_start_state: np.ndarray, planning_time: Optional[float] = None):
        self.start_state = np.asarray(new_start_state, dtype=float)
        if planning_time is not None:
            self.planning_time = float(planning_time)
        return self.plan()
