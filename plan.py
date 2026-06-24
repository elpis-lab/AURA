from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
from ompl import base as ob
from ompl import control as oc

from utils.utils import arrayDistance, isStateValid, log, normalize_obstacle_config

from systems import System


class InterpolatingMotionValidator(ob.MotionValidator):
    """Checks straight state-space interpolation; control edges are checked by substep propagation."""

    def __init__(self, si, max_segment_length: float):
        super().__init__(si)
        self.si = si
        self.max_segment_length = max(float(max_segment_length), 1e-9)

    def checkMotion(self, s1, s2, *args):
        if not self.si.isValid(s1) or not self.si.isValid(s2):
            return False
        distance = float(self.si.distance(s1, s2))
        steps = max(1, int(np.ceil(distance / self.max_segment_length)))
        state_space = self.si.getStateSpace()
        test_state = state_space.allocState()
        try:
            for i in range(1, steps + 1):
                state_space.interpolate(s1, s2, float(i) / float(steps), test_state)
                if not self.si.isValid(test_state):
                    return False
        finally:
            state_space.freeState(test_state)
        return True


class ControlCountObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_per_control=1.0):
        super().__init__(si)
        self.cost_per_control = cost_per_control

    def stateCost(self, s):
        return ob.Cost(0.0)

    def motionCost(self, s1, s2):
        # Penalize each control application uniformly.
        return ob.Cost(self.cost_per_control)


@dataclass
class OMPL_Planner:
    """Planning wrapper that owns all planning settings for plan/replan."""

    system: System
    start_state: np.ndarray
    goal_state: np.ndarray
    planner_method: str = "aorrt"
    goal_threshold: float = 0.1
    min_max_control_duration: tuple[int, int] = (1, 1)
    propagation_step_size: float = 1.0
    initial_planning_time: float = 10.0
    pruning_radius: float = 0.1
    goal_bias: float = 0.05
    obstacle_config: Optional[dict] = None
    planner: Optional[oc.Planner] = None

    def __post_init__(self):
        self.start_state = np.asarray(self.start_state, dtype=float)
        self.goal_state = np.asarray(self.goal_state, dtype=float)
        self.space, self.cspace = self.system.state_space, self.system.control_space
        self.ss = oc.SimpleSetup(self.cspace)
        self.solutions = None

    def set_state_values(self, ompl_state, values: np.ndarray):
        """Set the state values for the ompl state."""
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

    def set_planner(self):
        """Set the planner for the planner."""
        planner_name = self.planner_method.lower()
        if planner_name == "sststar":
            planner = oc.SSTStar(self.ss.getSpaceInformation())
            planner.setPruningRadius(float(self.pruning_radius))
        elif planner_name == "aorrt":
            planner = oc.AORRT(self.ss.getSpaceInformation())
        elif planner_name == "aoest":
            planner = oc.AOEST(self.ss.getSpaceInformation())
        else:
            raise ValueError(f"Unknown planner: {self.planner_method}")
        planner.setGoalBias(float(self.goal_bias))
        self.ss.setPlanner(planner)
        return planner

    def run_planner(self):
        """Run the planner."""
        config = {"obstacles": self.obstacle_config} if self.obstacle_config is not None else {}
        obstacle_config = normalize_obstacle_config(config.get("obstacles", None))

        # Validity checker
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(
                partial(
                    isStateValid,
                    self.ss.getSpaceInformation(),
                    system=self.system.name,
                    config=config,
                    obstacle_config=obstacle_config,
                )
            )
        )

        # State propagator
        self.ss.setStatePropagator(oc.StatePropagatorFn(self.system.propagator_fn))

        # Min max control duration
        self.ss.getSpaceInformation().setMinMaxControlDuration(
            int(self.min_max_control_duration[0]), int(self.min_max_control_duration[1])
        )

        # Propagation step size
        self.ss.getSpaceInformation().setPropagationStepSize(float(self.propagation_step_size))
        self.ss.getSpaceInformation().setMotionValidator(
            InterpolatingMotionValidator(
                self.ss.getSpaceInformation(),
                max_segment_length=float(self.propagation_step_size),
            )
        )

        # Start and goal: ProblemDefinition.setStartAndGoalStates accepts raw State*.
        # SimpleSetup.setStartState / GoalState expect ScopedState, which these bindings
        # do not expose to Python.
        start = self.space.allocState()
        self.set_state_values(start, self.start_state)
        goal = self.space.allocState()
        self.set_state_values(goal, self.goal_state)
        self.ss.getProblemDefinition().setStartAndGoalStates(
            start, goal, float(self.goal_threshold)
        )

        # Planner
        planner = self.set_planner()

        # Optimization objective
        self.cost_mode = "control_count"
        self.cost_label = "best total controls"
        objective = ControlCountObjective(
            self.ss.getSpaceInformation(), cost_per_control=float(self.propagation_step_size)
        )
        self.ss.setOptimizationObjective(objective)

        # Termination condition
        if float(self.initial_planning_time) < 0:
            print("[INFO] Using exact solution termination condition")
            ptc = ob.exactSolnPlannerTerminationCondition(self.ss.getProblemDefinition())
        else:
            ptc = ob.timedPlannerTerminationCondition(float(self.initial_planning_time))

        # Solve
        self.ss.solve(ptc)

        if float(self.initial_planning_time) < 0:
            if self.ss.haveExactSolutionPath():
                control_count = self.ss.getSolutionPath().getControlCount()
                print(
                    f"Solution found with {self.planner_method} "
                    f"({control_count} controls) - accepting first exact solution"
                )
            else:
                log(
                    "[WARNING] No exact solution found with exact solution termination condition",
                    "warning",
                )
        else:
            max_retries = 1000 if self.system.name == "kinematic_car" else 100
            max_attempts = 1 if getattr(self, "_single_solve", False) else 10
            attempts = 0
            accepted_solution = False

            while attempts < max_attempts:
                attempts += 1
                if self.ss.haveExactSolutionPath():
                    control_count = self.ss.getSolutionPath().getControlCount()
                    if control_count <= max_retries:
                        print(
                            f"Solution found in {float(self.initial_planning_time):.1f}s with {self.planner_method} "
                            f"({control_count} controls)"
                        )
                        accepted_solution = True
                        break
                    print(
                        f"[WARNING] Solution found but has {control_count} controls (> {max_retries})... "
                        f"retrying... (attempt {attempts}/{max_attempts})"
                    )
                else:
                    print(
                        f"[WARNING] No exact solution found in {float(self.initial_planning_time):.1f}s... "
                        f"retrying... (attempt {attempts}/{max_attempts})"
                    )

                self.ss.getPlanner().clear()
                self.ss.solve(float(self.initial_planning_time))

            if not accepted_solution:
                log(
                    f"[WARNING] Maximum retry attempts ({max_attempts}) reached, using current solution",
                    "warning",
                )

        return self.ss

    def plan(self):
        """Plan the path."""
        self.ss = self.run_planner()
        self.solutions = self.get_solutions()
        return self.solutions, self.ss

    def replan(
        self,
        planning_time: float | None = None,
        time_budget: float | None = None,
    ):
        """Continue planning from the existing OMPL setup/tree."""
        budget = time_budget
        if budget is None:
            budget = planning_time
        if budget is None:
            budget = self.propagation_step_size

        self.ss.getPlanner().resolve(float(budget))
        self.solutions = self.get_solutions()
        return self.solutions, self.ss

    def plan_from_state(self, start_state: np.ndarray, time_budget: float | None = None):
        """Explicit fresh solve from a new start state; AURA's replanning loop does not use this."""
        self.start_state = np.asarray(start_state, dtype=float)
        old_planning_time = self.initial_planning_time
        old_single_solve = getattr(self, "_single_solve", False)
        self.initial_planning_time = (
            float(self.propagation_step_size) if time_budget is None else float(time_budget)
        )
        self._single_solve = True
        self.ss = oc.SimpleSetup(self.cspace)
        try:
            self.ss = self.run_planner()
            self.solutions = self.get_solutions()
        finally:
            self.initial_planning_time = old_planning_time
            self._single_solve = old_single_solve
        return self.solutions, self.ss

    def _duration_to_seconds(self, duration: float) -> float:
        duration = float(duration)
        step = float(self.propagation_step_size)
        max_steps = max(int(self.min_max_control_duration[0]), int(self.min_max_control_duration[1]))
        max_seconds = max_steps * step
        if step < 1.0 and duration > max_seconds + 1e-9 and abs(duration - round(duration)) < 1e-9:
            return duration * step
        return duration

    def extract_solution_info(self, solution_path):
        """Extract the solution information."""
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
            time_list.append(self._duration_to_seconds(solution_path.getControlDuration(i)))

        solution_info["controls"] = controls_list
        solution_info["time"] = time_list

        states_list = []
        space = self.ss.getSpaceInformation().getStateSpace()
        space_type = space.getType()

        for i in range(solution_info["state_count"]):
            state = solution_path.getState(i)

            if space_type == ob.STATE_SPACE_SE2:
                state_values = [state.getX(), state.getY(), state.getYaw()]
            else:
                dim = space.getDimension()
                state_values = []
                try:
                    if space_type == ob.STATE_SPACE_REAL_VECTOR:
                        state_values = [float(state[j]) for j in range(dim)]
                    else:
                        raise ValueError("not real-vector")
                except Exception:
                    raise ValueError("not real-vector")

            states_list.append(state_values)

        solution_info["states"] = states_list
        return solution_info

    def get_solutions(self):
        """Get the solutions."""
        self.solutions = []
        rejected_approx = 0
        best_rejected_goal_distance = float("inf")

        for solution in self.ss.getProblemDefinition().getSolutions():
            info = self.extract_solution_info(solution.path_)
            info["cost"] = solution.cost_.value()
            info["approximate"] = bool(getattr(solution, "approximate_", False))
            info["solution_difference"] = float(getattr(solution, "difference_", float("nan")))
            if info.get("states"):
                info["goal_distance"] = float(
                    arrayDistance(
                        np.asarray(info["states"][-1], dtype=float),
                        self.goal_state,
                        system=self.system.name,
                    )
                )
            else:
                info["goal_distance"] = float("inf")

            # OMPL may keep approximate solutions in ProblemDefinition.getSolutions().
            # AURA's experiments require plans that actually terminate inside the goal
            # region, so reject any path whose final state misses the configured goal
            # threshold, regardless of cost.
            if info["goal_distance"] <= float(self.goal_threshold) + 1e-9:
                self.solutions.append(info)
            else:
                rejected_approx += 1
                best_rejected_goal_distance = min(
                    best_rejected_goal_distance,
                    float(info["goal_distance"]),
                )

        if rejected_approx:
            log(
                f"[WARNING] Ignored {rejected_approx} approximate/non-goal solution(s); "
                f"best rejected final distance {best_rejected_goal_distance:.6f} "
                f"> goal threshold {float(self.goal_threshold):.6f}.",
                "warning",
            )

        # Sort solutions by cost
        self.solutions.sort(key=lambda x: x["cost"])
        return self.solutions

    def getBestSolution(self):
        """Get the best solution."""
        if self.solutions is None:
            return None
        if len(self.solutions) == 0:
            return None
        return self.solutions[0]
