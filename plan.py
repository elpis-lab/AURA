from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
from ompl import base as ob
from ompl import control as oc

from utils.utils import isStateValid, log, normalize_obstacle_config

from systems import System


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

    def set_goal(self):
        """Set the goal for the planner."""
        goal_state = self.space.allocState()
        self.set_state_values(goal_state, self.goal_state)
        goal = ob.GoalState(self.ss.getSpaceInformation(), goal_state)
        goal.setThreshold(float(self.goal_threshold))
        self.ss.setGoal(goal)
        return goal

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

        # Start state
        start = self.space.allocState()
        self.set_state_values(start, self.start_state)
        self.ss.setStartState(start)

        # Goal state
        goal = self.set_goal()

        # Planner
        planner = self.set_planner()

        # Optimization objective
        self.ss.setOptimizationObjective(
            ControlCountObjective(
                self.ss.getSpaceInformation(), cost_per_control=float(self.propagation_step_size)
            )
        )

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
            max_attempts = 10
            attempts = 0

            while attempts < max_attempts:
                attempts += 1
                if self.ss.haveExactSolutionPath():
                    control_count = self.ss.getSolutionPath().getControlCount()
                    if control_count <= max_retries:
                        print(
                            f"Solution found in {float(self.initial_planning_time):.1f}s with {self.planner_method} "
                            f"({control_count} controls)"
                        )
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

            if attempts >= max_attempts:
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

    def replan(self):
        """Replan the path."""
        self.ss.resolve(float(self.propagation_step_size))
        self.solutions = self.get_solutions()
        return self.solutions, self.ss

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

        for solution in self.ss.getProblemDefinition().getSolutions():
            info = self.extract_solution_info(solution.path_)
            info["cost"] = solution.cost_.value()
            self.solutions.append(info)

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
