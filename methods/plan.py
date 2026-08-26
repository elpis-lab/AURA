"""OMPL configuration, planning, replanning, and solution extraction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from ompl import base as ob
from ompl import control as oc

from methods.RandUpRRT import RandUpRRT, RandUpRRTConfig
from utils.utils import arrayDistance, isStateValid, log, normalize_obstacle_config

if TYPE_CHECKING:
    from propagators import System


DURATION_TOLERANCE = 1e-8


def validate_duration_range(min_steps: int, max_steps: int) -> tuple[int, int]:
    """Validate and normalize an OMPL control-duration range."""

    minimum = int(min_steps)
    maximum = int(max_steps)
    if minimum < 1:
        raise ValueError(f"min_control_duration must be at least 1, got {minimum}")
    if maximum < minimum:
        raise ValueError(
            "max_control_duration must be greater than or equal to "
            f"min_control_duration, got [{minimum}, {maximum}]"
        )
    return minimum, maximum


def duration_steps_to_seconds(steps: int, propagation_step_size: float) -> float:
    """Convert an OMPL propagation-step count to physical seconds."""

    step_count = int(steps)
    step_size = float(propagation_step_size)
    if step_count < 1:
        raise ValueError(f"duration steps must be at least 1, got {step_count}")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError(
            "propagation_step_size must be finite and positive, "
            f"got {step_size}"
        )
    return float(step_count * step_size)


def duration_seconds_to_steps(
    duration_seconds: float,
    propagation_step_size: float,
    *,
    min_steps: int | None = None,
    max_steps: int | None = None,
    atol: float = DURATION_TOLERANCE,
) -> int:
    """Convert an OMPL edge duration to an exact propagation-step count."""

    duration = float(duration_seconds)
    step_size = float(propagation_step_size)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(
            f"duration_seconds must be finite and positive, got {duration}"
        )
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError(
            "propagation_step_size must be finite and positive, "
            f"got {step_size}"
        )

    ratio = duration / step_size
    rounded = int(round(ratio))
    allowed_error = max(float(atol), abs(ratio) * float(atol))
    if rounded < 1 or abs(ratio - rounded) > allowed_error:
        raise ValueError(
            f"duration {duration:.17g}s is not an integer multiple of "
            f"propagation_step_size {step_size:.17g}s (ratio={ratio:.17g})"
        )
    if min_steps is not None and rounded < int(min_steps):
        raise ValueError(
            f"duration has {rounded} steps, below configured minimum "
            f"{int(min_steps)}"
        )
    if max_steps is not None and rounded > int(max_steps):
        raise ValueError(
            f"duration has {rounded} steps, above configured maximum "
            f"{int(max_steps)}"
        )
    return rounded


@dataclass(frozen=True)
class ControlEdge:
    """A planner edge whose control and duration remain inseparable."""

    source_state: np.ndarray
    target_state: np.ndarray
    control: np.ndarray
    duration_steps: int
    duration_seconds: float
    source_vertex: int | None = None
    target_vertex: int | None = None
    edge_id: str = ""

    def __post_init__(self) -> None:
        source = np.asarray(self.source_state, dtype=float).reshape(-1).copy()
        target = np.asarray(self.target_state, dtype=float).reshape(-1).copy()
        control = np.asarray(self.control, dtype=float).reshape(-1).copy()
        steps = int(self.duration_steps)
        seconds = float(self.duration_seconds)
        if not source.size or not np.all(np.isfinite(source)):
            raise ValueError("ControlEdge source_state must be finite and non-empty")
        if not target.size or not np.all(np.isfinite(target)):
            raise ValueError("ControlEdge target_state must be finite and non-empty")
        if not control.size or not np.all(np.isfinite(control)):
            raise ValueError("ControlEdge control must be finite and non-empty")
        if steps < 1:
            raise ValueError(
                f"ControlEdge duration_steps must be at least 1, got {steps}"
            )
        if not np.isfinite(seconds) or seconds <= 0.0:
            raise ValueError(
                "ControlEdge duration_seconds must be finite and positive, "
                f"got {seconds}"
            )

        source.setflags(write=False)
        target.setflags(write=False)
        control.setflags(write=False)
        object.__setattr__(self, "source_state", source)
        object.__setattr__(self, "target_state", target)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "duration_steps", steps)
        object.__setattr__(self, "duration_seconds", seconds)
        if not self.edge_id:
            source_id = (
                "?" if self.source_vertex is None else str(int(self.source_vertex))
            )
            target_id = (
                "?" if self.target_vertex is None else str(int(self.target_vertex))
            )
            object.__setattr__(self, "edge_id", f"{source_id}->{target_id}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_state": self.source_state.tolist(),
            "target_state": self.target_state.tolist(),
            "control": self.control.tolist(),
            "duration_steps": self.duration_steps,
            "duration_seconds": self.duration_seconds,
            "source_vertex": self.source_vertex,
            "target_vertex": self.target_vertex,
            "edge_id": self.edge_id,
        }


@dataclass(frozen=True)
class ControlSelection:
    """A control-duration pair selected for the next execution cycle."""

    control: np.ndarray
    duration_steps: int
    duration_seconds: float
    edge_id: str
    source: str
    target_state: np.ndarray

    def __post_init__(self) -> None:
        control = np.asarray(self.control, dtype=float).reshape(-1).copy()
        target = np.asarray(self.target_state, dtype=float).reshape(-1).copy()
        if not control.size or not np.all(np.isfinite(control)):
            raise ValueError("ControlSelection control must be finite and non-empty")
        if not target.size or not np.all(np.isfinite(target)):
            raise ValueError(
                "ControlSelection target_state must be finite and non-empty"
            )
        if int(self.duration_steps) < 1:
            raise ValueError("ControlSelection duration_steps must be at least 1")
        if (
            not np.isfinite(float(self.duration_seconds))
            or float(self.duration_seconds) <= 0.0
        ):
            raise ValueError(
                "ControlSelection duration_seconds must be finite and positive"
            )
        control.setflags(write=False)
        target.setflags(write=False)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "target_state", target)
        object.__setattr__(self, "duration_steps", int(self.duration_steps))
        object.__setattr__(self, "duration_seconds", float(self.duration_seconds))


class InterpolatingMotionValidator(ob.MotionValidator):
    """Validate interpolated states between two OMPL states."""

    def __init__(self, space_information, max_segment_length: float):
        super().__init__(space_information)
        self.space_information = space_information
        self.max_segment_length = max(float(max_segment_length), 1e-9)

    def checkMotion(self, start, goal, *args):
        """OMPL callback; the camel-case name is required by OMPL."""

        if not self.space_information.isValid(start):
            return False
        if not self.space_information.isValid(goal):
            return False

        distance = float(self.space_information.distance(start, goal))
        segments = max(1, int(np.ceil(distance / self.max_segment_length)))
        state_space = self.space_information.getStateSpace()
        test_state = state_space.allocState()
        try:
            for index in range(1, segments + 1):
                state_space.interpolate(
                    start,
                    goal,
                    float(index) / float(segments),
                    test_state,
                )
                if not self.space_information.isValid(test_state):
                    return False
        finally:
            state_space.freeState(test_state)
        return True


class ControlDurationObjective(ob.OptimizationObjective):
    """Minimize elapsed control time instead of geometric path length."""

    def __init__(self, space_information, propagation_step_size: float):
        super().__init__(space_information)
        self.propagation_step_size = float(propagation_step_size)

    def stateCost(self, state):
        """OMPL callback; states do not add duration cost."""

        return ob.Cost(0.0)

    def motionCost(self, start, goal):
        """OMPL callback; geometric motion does not add duration cost."""

        return ob.Cost(0.0)

    def controlCost(self, control, steps):
        """OMPL callback; charge the physical duration of a control edge."""

        return ob.Cost(float(steps) * self.propagation_step_size)


@dataclass
class OMPLPlanner:
    """Own one fully configured OMPL control-planning problem."""

    system: System
    start_state: np.ndarray
    goal_state: np.ndarray
    planner_method: str = "aorrt"
    goal_threshold: float = 0.1
    min_max_control_duration: tuple[int, int] = (1, 5)
    propagation_step_size: float = 1.0
    initial_planning_time: float = 10.0
    pruning_radius: float = 0.1
    goal_bias: float = 0.05
    selection_radius: float | None = None
    obstacle_config: dict | None = None
    optimization_objective: str = "control_duration"
    randup_config: RandUpRRTConfig | None = None
    single_solve: bool = False

    def __post_init__(self) -> None:
        self.start_state = np.asarray(self.start_state, dtype=float)
        self.goal_state = np.asarray(self.goal_state, dtype=float)
        self.min_max_control_duration = validate_duration_range(
            *self.min_max_control_duration
        )
        if float(self.propagation_step_size) <= 0.0:
            raise ValueError("propagation_step_size must be positive")

        self.system.configure_duration_contract(
            float(self.propagation_step_size),
            *self.min_max_control_duration,
        )
        self.state_space = self.system.state_space
        self.control_space = self.system.control_space
        self.setup = oc.SimpleSetup(self.control_space)
        self.solutions: list[dict] | None = None
        self.randup_planner: RandUpRRT | None = None
        self.cost_mode = self.optimization_objective
        self.cost_label = ""

    def configure_problem(self):
        """Configure validity, dynamics, durations, endpoints, and objective."""

        config = (
            {"obstacles": self.obstacle_config}
            if self.obstacle_config is not None
            else {}
        )
        obstacles = normalize_obstacle_config(config.get("obstacles"))
        validity_checker = partial(
            isStateValid,
            self.setup.getSpaceInformation(),
            system=self.system.name,
            config=config,
            obstacle_config=obstacles,
        )
        validity_wrapper = getattr(ob, "StateValidityCheckerFn", None)
        self.setup.setStateValidityChecker(
            validity_wrapper(validity_checker)
            if validity_wrapper is not None
            else validity_checker
        )

        propagator_wrapper = getattr(oc, "StatePropagatorFn", None)
        self.setup.setStatePropagator(
            propagator_wrapper(self.system.propagator_fn)
            if propagator_wrapper is not None
            else self.system.propagator_fn
        )

        space_information = self.setup.getSpaceInformation()
        space_information.setMinMaxControlDuration(*self.min_max_control_duration)
        space_information.setPropagationStepSize(float(self.propagation_step_size))
        space_information.setMotionValidator(
            InterpolatingMotionValidator(
                space_information,
                max_segment_length=float(self.propagation_step_size),
            )
        )

        start = self.state_space.allocState()
        goal = self.state_space.allocState()
        self.system.set_ompl_state(start, self.start_state)
        self.system.set_ompl_state(goal, self.goal_state)
        self.setup.getProblemDefinition().setStartAndGoalStates(
            start,
            goal,
            float(self.goal_threshold),
        )

        planner = self.create_planner()
        self.setup.setOptimizationObjective(self.create_objective())
        return planner

    def create_planner(self):
        """Create the single planner selected by ``planner_method``."""

        planner_name = self.planner_method.lower()
        space_information = self.setup.getSpaceInformation()
        if planner_name == "sststar":
            planner = oc.SSTStar(space_information)
            planner.setPruningRadius(float(self.pruning_radius))
        elif planner_name == "aorrt":
            planner = oc.AORRT(space_information)
        elif planner_name == "aoest":
            planner = oc.AOEST(space_information)
        elif planner_name == "randup_rrt":
            config = self.randup_config or RandUpRRTConfig(
                planning_time=float(self.initial_planning_time),
                control_duration_min=self.min_max_control_duration[0],
                control_duration_max=self.min_max_control_duration[1],
                goal_bias=float(self.goal_bias),
            )
            planner = RandUpRRT(
                space_information,
                system=self.system,
                config=config,
                start_state=self.start_state,
                goal_state=self.goal_state,
                goal_threshold=float(self.goal_threshold),
                obstacle_config=self.obstacle_config,
            )
            planner.setProblemDefinition(self.setup.getProblemDefinition())
            self.randup_planner = planner
            return planner
        else:
            raise ValueError(f"Unknown planner: {self.planner_method}")

        if self.selection_radius is not None:
            planner.setSelectionRadius(float(self.selection_radius))
        if hasattr(planner, "setGoalBias"):
            planner.setGoalBias(float(self.goal_bias))
        self.setup.setPlanner(planner)
        return planner

    def create_objective(self):
        """Create the configured OMPL optimization objective."""

        if self.optimization_objective == "path_length":
            self.cost_mode = "path_length"
            self.cost_label = "best total path length"
            return ob.PathLengthOptimizationObjective(self.setup.getSpaceInformation())
        if self.optimization_objective == "control_duration":
            self.cost_mode = "control_duration"
            self.cost_label = "best total control duration"
            return ControlDurationObjective(
                self.setup.getSpaceInformation(),
                self.propagation_step_size,
            )
        raise ValueError(
            f"Unknown optimization objective {self.optimization_objective!r}; "
            "expected 'path_length' or 'control_duration'"
        )

    def termination_condition(self):
        if float(self.initial_planning_time) < 0.0:
            return ob.exactSolnPlannerTerminationCondition(
                self.setup.getProblemDefinition()
            )
        return ob.timedPlannerTerminationCondition(float(self.initial_planning_time))

    def solve_randup(self, planner: RandUpRRT, condition) -> None:
        self.setup.getSpaceInformation().setup()
        status = planner.solve(condition)
        if not bool(status):
            log(
                f"[WARNING] RandUp-RRT did not find a robust solution: "
                f"{planner.failure_reason}",
                "warning",
            )

    def solve_stock_planner(self, condition) -> None:
        self.setup.solve(condition)
        if float(self.initial_planning_time) < 0.0:
            if not self.setup.haveExactSolutionPath():
                log("[WARNING] No exact solution found", "warning")
            return

        control_limit = 1000 if self.system.name == "kinematic_car" else 100
        max_attempts = 1 if self.single_solve else 10
        for attempt in range(max_attempts):
            if self.setup.haveExactSolutionPath():
                control_count = self.setup.getSolutionPath().getControlCount()
                if control_count <= control_limit:
                    return
            if attempt + 1 < max_attempts:
                self.setup.getPlanner().clear()
                self.setup.solve(float(self.initial_planning_time))

        log(
            f"[WARNING] Maximum retry attempts ({max_attempts}) reached, "
            "using the current solution",
            "warning",
        )

    def run_planner(self):
        """Configure and solve the current planning problem."""

        planner = self.configure_problem()
        condition = self.termination_condition()
        if isinstance(planner, RandUpRRT):
            self.solve_randup(planner, condition)
        else:
            self.solve_stock_planner(condition)
        return self.setup

    def plan(self):
        """Run the initial solve and return all accepted solutions."""

        self.run_planner()
        self.solutions = self.get_solutions()
        return self.solutions, self.setup

    def replan(
        self,
        planning_time: float | None = None,
        time_budget: float | None = None,
    ):
        """Continue solving from the existing OMPL planner tree."""

        budget = time_budget if time_budget is not None else planning_time
        if budget is None:
            budget = self.propagation_step_size
        self.setup.getPlanner().resolve(float(budget))
        self.solutions = self.get_solutions()
        return self.solutions, self.setup

    def retry_initial_plan(self, time_budget: float):
        """Restart a bounded solve when the initial solve has no exact path."""

        budget = max(1e-3, float(time_budget))
        self.setup.getProblemDefinition().clearSolutionPaths()
        self.setup.getPlanner().clear()
        self.setup.solve(ob.timedPlannerTerminationCondition(budget))
        self.solutions = self.get_solutions()
        return self.solutions, self.setup

    def plan_from_state(
        self,
        start_state: np.ndarray,
        time_budget: float | None = None,
    ):
        """Run one fresh bounded solve from a new start state."""

        self.start_state = np.asarray(start_state, dtype=float)
        previous_time = self.initial_planning_time
        previous_single_solve = self.single_solve
        self.initial_planning_time = (
            self.propagation_step_size if time_budget is None else float(time_budget)
        )
        self.single_solve = True
        self.setup = oc.SimpleSetup(self.control_space)
        try:
            self.run_planner()
            self.solutions = self.get_solutions()
        finally:
            self.initial_planning_time = previous_time
            self.single_solve = previous_single_solve
        return self.solutions, self.setup

    def duration_seconds_to_steps(self, duration: float) -> int:
        """Convert one OMPL edge duration to its validated integer step count."""

        return duration_seconds_to_steps(
            duration,
            self.propagation_step_size,
            min_steps=self.min_max_control_duration[0],
            max_steps=self.min_max_control_duration[1],
        )

    def extract_solution_info(self, solution_path) -> dict:
        """Convert an OMPL control path to repository arrays and durations."""

        control_count = solution_path.getControlCount()
        control_dimension = self.setup.getControlSpace().getDimension()
        controls = []
        durations = []
        duration_steps = []
        for index in range(control_count):
            control = solution_path.getControl(index)
            controls.append(
                [control[dimension] for dimension in range(control_dimension)]
            )
            duration = float(solution_path.getControlDuration(index))
            durations.append(duration)
            duration_steps.append(self.duration_seconds_to_steps(duration))

        states = [
            self.system.ompl_state_to_numpy(solution_path.getState(index)).tolist()
            for index in range(solution_path.getStateCount())
        ]
        return {
            "state_count": solution_path.getStateCount(),
            "control_count": control_count,
            "controls": controls,
            "time": durations,
            "time_steps": duration_steps,
            "states": states,
        }

    def fallback_solution(self) -> dict | None:
        """Read the direct OMPL path when PlannerSolution conversion is unavailable."""

        if not self.setup.haveSolutionPath():
            return None
        path = self.setup.getSolutionPath()
        info = self.extract_solution_info(path)
        if self.cost_mode == "control_duration":
            info["cost"] = float(sum(info["time"]))
        else:
            planner = self.setup.getPlanner()
            info["cost"] = (
                float(planner.getBestSolutionCost().value())
                if hasattr(planner, "getBestSolutionCost")
                else float(path.length())
            )
        info["approximate"] = not bool(self.setup.haveExactSolutionPath())
        info["solution_difference"] = float("nan")
        return info

    def raw_solutions(self) -> tuple[list, dict | None]:
        randup_info = (
            self.randup_planner.solution_info()
            if self.randup_planner is not None
            else None
        )
        if randup_info is not None:
            return [randup_info], randup_info
        try:
            return list(self.setup.getProblemDefinition().getSolutions()), None
        except TypeError:
            fallback = self.fallback_solution()
            return ([] if fallback is None else [fallback]), None

    def normalize_solution(self, solution) -> dict:
        if isinstance(solution, dict):
            return solution
        info = self.extract_solution_info(solution.path_)
        info["cost"] = solution.cost_.value()
        info["approximate"] = bool(getattr(solution, "approximate_", False))
        info["solution_difference"] = float(
            getattr(solution, "difference_", float("nan"))
        )
        return info

    def goal_distance(self, states: list) -> float:
        if not states:
            return float("inf")
        endpoint = np.asarray(states[-1], dtype=float)
        if self.system.name == "dubins_airplane":
            return self.system.state_distance(endpoint, self.goal_state)
        return float(arrayDistance(endpoint, self.goal_state, system=self.system.name))

    def get_solutions(self) -> list[dict]:
        """Return exact/goal-reaching solutions ordered by objective cost."""

        accepted = []
        rejected_distances = []
        raw_solutions, randup_info = self.raw_solutions()
        for solution in raw_solutions:
            info = self.normalize_solution(solution)
            info["goal_distance"] = self.goal_distance(info.get("states", []))
            robust_randup = randup_info is not None and solution is randup_info
            reaches_goal = info["goal_distance"] <= self.goal_threshold + 1e-9
            if robust_randup or reaches_goal:
                planner = self.randup_planner or self.setup.getPlanner()
                metadata = getattr(planner, "solution_metadata", None)
                if callable(metadata):
                    info["randup"] = metadata()
                accepted.append(info)
            else:
                rejected_distances.append(info["goal_distance"])

        if rejected_distances:
            log(
                f"[WARNING] Ignored {len(rejected_distances)} approximate/non-goal "
                f"solution(s); best final distance {min(rejected_distances):.6f} "
                f"> goal threshold {self.goal_threshold:.6f}.",
                "warning",
            )
        accepted.sort(key=lambda solution: solution["cost"])
        self.solutions = accepted
        return accepted

    def best_solution(self) -> dict | None:
        """Return the lowest-cost accepted solution."""

        return self.solutions[0] if self.solutions else None
