from __future__ import annotations

import numpy as np
from types import SimpleNamespace

import aura.AURA as aura_module
from aura.AURA import AURA
from methods.Replanning import ReplanningRunner
from propagators import get_system
from methods.plan import ControlEdge


class _Simulator:
    def __init__(self):
        self.states = iter(
            (
                [0.5, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            )
        )
        self.current = [0.0, 0.0, 0.0]
        self.primitive_states = [np.asarray(self.current)]

    def reset(self):
        self.current = [0.0, 0.0, 0.0]

    def get_state(self):
        return self.current

    def execute_segment(self, control, duration):
        del control, duration
        self.current = next(self.states)
        self.primitive_states.append(np.asarray(self.current))
        return self.current

    def stop(self):
        return None


class _Planner:
    def __init__(self, solution):
        self.solution = solution

    def plan(self):
        return ([] if self.solution is None else [self.solution]), None

    def retry_initial_plan(self, *, time_budget):
        del time_budget
        return self.plan()


def _solution(start_x: float, target_x: float, duration: float) -> dict:
    return {
        "states": [
            [start_x, 0.0, 0.0],
            [target_x, 0.0, 0.0],
        ],
        "controls": [[0.1, 0.0]],
        "time": [duration],
    }


def test_restart_replanning_budget_tracks_each_executed_duration(monkeypatch):
    config = {
        "start_state": [0.0, 0.0, 0.0],
        "goal_state": [1.0, 0.0, 0.0],
        "goal_threshold": 0.05,
        "actual_goal_threshold": 0.05,
        "replanning_max_distance": 0.01,
        "planning_time": 1.0,
        "replanning_time_budget": 99.0,
        "propagation_step_size": 1.0,
        "min_control_duration": 1,
        "max_control_duration": 5,
    }
    runner = ReplanningRunner(
        "kinematic_car",
        "aorrt",
        config,
        "gaussian",
        max_steps=10,
        initial_solution=_solution(0.0, 0.1, 5.0),
        system_override=get_system("kinematic_car"),
        simulator_override=_Simulator(),
    )
    budgets = []
    replans = iter(
        (
            _solution(0.5, 0.6, 1.0),
            _solution(0.8, 1.0, 1.0),
        )
    )

    def build_planner(start_state, planning_time):
        del start_state
        budgets.append(float(planning_time))
        return _Planner(next(replans))

    monkeypatch.setattr(runner, "build_planner", build_planner)
    result = runner.run(reset_sim=False)
    assert budgets == [5.0, 1.0]
    assert result.num_replanning == 2
    assert result.control_duration_steps_trajectory == [5, 1, 1]


def test_restart_replanning_retries_a_missed_fresh_solve(monkeypatch):
    config = {
        "start_state": [0.0, 0.0, 0.0],
        "goal_state": [1.0, 0.0, 0.0],
        "goal_threshold": 0.05,
        "actual_goal_threshold": 0.05,
        "replanning_max_distance": 0.01,
        "planning_time": 1.0,
        "replanning_time_budget": 1.0,
        "propagation_step_size": 1.0,
        "min_control_duration": 1,
        "max_control_duration": 5,
    }

    class Simulator(_Simulator):
        def __init__(self):
            self.states = iter(([0.5, 0.0, 0.0], [1.0, 0.0, 0.0]))
            self.current = [0.0, 0.0, 0.0]
            self.primitive_states = [np.asarray(self.current)]

    runner = ReplanningRunner(
        "kinematic_car",
        "aorrt",
        config,
        "gaussian",
        max_steps=None,
        initial_solution=_solution(0.0, 0.1, 5.0),
        task_time_budget_seconds=30.0,
        system_override=get_system("kinematic_car"),
        simulator_override=Simulator(),
    )
    attempted_solutions = iter((None, _solution(0.5, 1.0, 1.0)))
    budgets = []
    retry_budgets = []

    class RetryPlanner:
        def plan(self):
            solution = next(attempted_solutions)
            return ([] if solution is None else [solution]), None

        def retry_initial_plan(self, *, time_budget):
            retry_budgets.append(float(time_budget))
            return self.plan()

    def build_planner(start_state, planning_time):
        del start_state
        budgets.append(float(planning_time))
        return RetryPlanner()

    monkeypatch.setattr(runner, "build_planner", build_planner)
    result = runner.run(reset_sim=False)
    assert result.status == "success"
    assert result.failure_reason == ""
    assert result.num_replanning == 1
    assert budgets == [5.0]
    assert retry_budgets == [5.0]


def test_aura_forwards_dynamic_segment_budget_to_replan_and_optimizer(monkeypatch):
    system = get_system("kinematic_car")
    planner = SimpleNamespace(
        start_state=np.zeros(3),
        goal_state=np.ones(3),
        goal_threshold=0.1,
        propagation_step_size=1.0,
        initial_planning_time=1.0,
        replanning_time=1.0,
        pruning_radius=0.1,
        opt_model=None,
    )
    aura = AURA(system, planner, SimpleNamespace())
    optimizer_budgets = []
    replanning_budgets = []

    def fake_optimizer(**kwargs):
        optimizer_budgets.append(float(kwargs["max_wall_time"]))
        return {"selection": None}

    class Replanner:
        def replan(self, *, time_budget):
            replanning_budgets.append(float(time_budget))
            return [{"controls": [[0.1, 0.0]]}], None

    monkeypatch.setattr(aura_module, "optimize_controls", fake_optimizer)
    edge = ControlEdge(
        source_state=np.zeros(3),
        target_state=np.ones(3),
        control=np.asarray([0.1, 0.0]),
        duration_steps=5,
        duration_seconds=5.0,
    )
    for duration in (5.0, 1.0):
        aura.run_optimizer(
            system,
            np.zeros(3),
            [edge],
            wall_time_budget=duration,
            result_slot={},
        )
        aura.run_replanning(Replanner(), duration, result_slot={})

    assert optimizer_budgets == [5.0, 1.0]
    assert replanning_budgets == [5.0, 1.0]


def test_aura_final_segment_preserves_result_contract_without_normal_output(capsys):
    system = get_system("kinematic_car")
    solution = {
        "states": [np.zeros(3), np.array([0.1, 0.0, 0.0])],
        "controls": [np.array([0.1, 0.0])],
        "time": [1.0],
        "time_steps": [1],
        "state_count": 2,
        "control_count": 1,
    }

    class Planner:
        start_state = np.zeros(3)
        goal_state = np.array([0.1, 0.0, 0.0])
        goal_threshold = 1e-9
        propagation_step_size = 1.0
        initial_planning_time = 1.0
        replanning_time = 1.0
        pruning_radius = 0.1
        opt_model = None

        def best_solution(self):
            return solution

        def duration_seconds_to_steps(self, duration):
            return int(round(float(duration)))

    class Simulator:
        config = {}

        def __init__(self):
            self.current = np.zeros(3)
            self.primitive_states = [self.current.copy()]

        def get_state(self):
            return self.current.copy()

        def execute_segment(self, control, duration):
            self.current = system.propagate(self.current, control, duration)
            self.primitive_states.append(self.current.copy())

    result = AURA(system, Planner(), Simulator()).run(
        reset_sim=False,
        pause_each_step=False,
    )

    assert capsys.readouterr().out == ""
    assert result.status == "success"
    assert result.failure_reason == ""
    assert result.num_controls == 1
    assert result.num_replanning == 0
    assert result.control_duration_steps_trajectory == [1]
    assert result.control_duration_seconds_trajectory == [1.0]
    assert result.cycle_timing[0]["mode"] == "final"
    assert all(isinstance(state, np.ndarray) for state in result.states_trajectory)
    assert all(
        isinstance(control, np.ndarray) for control in result.controls_trajectory
    )
    np.testing.assert_allclose(result.final_state, [0.1, 0.0, 0.0])
    np.testing.assert_allclose(result.final_planned_state, [0.1, 0.0, 0.0])
    assert result.tracking_error_list == [0.0]


def test_aura_parallel_cycle_and_final_segment_keep_the_same_contract(
    monkeypatch, capsys
):
    system = get_system("kinematic_car")
    duration = 0.02
    control = np.array([0.5, 0.0])
    start = np.zeros(3)
    middle = system.propagate(start, control, duration)
    goal = system.propagate(middle, control, duration)
    initial_solution = {
        "states": [start, middle, goal],
        "controls": [control, control],
        "time": [duration, duration],
        "time_steps": [1, 1],
        "state_count": 3,
        "control_count": 2,
    }
    replanned_solution = {
        "states": [middle, goal],
        "controls": [control],
        "time": [duration],
        "time_steps": [1],
        "state_count": 2,
        "control_count": 1,
    }

    class Planner:
        start_state = start
        goal_state = goal
        goal_threshold = 1e-9
        propagation_step_size = duration
        initial_planning_time = 1.0
        replanning_time = duration
        pruning_radius = 0.1
        opt_model = None
        setup = object()
        solutions = [initial_solution]

        def best_solution(self):
            return initial_solution

        def duration_seconds_to_steps(self, seconds):
            return int(round(float(seconds) / duration))

        def replan(self, *, time_budget):
            assert 0.0 < time_budget <= duration
            return [replanned_solution], self.setup

    class Simulator:
        config = {}

        def __init__(self):
            self.current = start.copy()
            self.primitive_states = [self.current.copy()]

        def get_state(self):
            return self.current.copy()

        def execute_segment(self, selected_control, selected_duration):
            self.current = system.propagate(
                self.current, selected_control, selected_duration
            )
            self.primitive_states.append(self.current.copy())

        def stop(self):
            return None

    edge = ControlEdge(
        source_state=middle,
        target_state=goal,
        control=control,
        duration_steps=1,
        duration_seconds=duration,
        edge_id="remaining-edge",
    )
    monkeypatch.setattr(
        aura_module,
        "getChildEdges",
        lambda *args, **kwargs: ([edge], {"match": "exact"}),
    )
    monkeypatch.setattr(aura_module, "optimize_controls", lambda **kwargs: None)

    result = AURA(system, Planner(), Simulator()).run(
        reset_sim=False,
        pause_each_step=False,
    )

    assert capsys.readouterr().out == ""
    assert result.status == "success"
    assert result.failure_reason == ""
    assert result.num_controls == 2
    assert result.num_replanning == 1
    assert result.control_duration_steps_trajectory == [1, 1]
    assert result.control_duration_seconds_trajectory == [duration, duration]
    assert [row["mode"] for row in result.cycle_timing] == ["parallel", "final"]
    np.testing.assert_allclose(result.final_state, goal)
    np.testing.assert_allclose(result.final_planned_state, goal)
