from __future__ import annotations

import numpy as np
import pytest
from ompl import control as oc
from ompl import util as ou

from experiment.task_time_efficiency import _duration_audit
from methods.plan import OMPLPlanner
from propagators import KinematicCar
from utils.childrenHandler import getChildEdges, getOutgoingEdgeIndices
from methods.plan import duration_seconds_to_steps
from utils.utils import state2list


class _StockRRTPlanner(OMPLPlanner):
    def create_planner(self):
        planner = oc.RRT(self.setup.getSpaceInformation())
        planner.setGoalBias(0.25)
        self.setup.setPlanner(planner)
        return planner


def _planned_stock_rrt(
    duration_range: tuple[int, int] = (1, 3),
    objective: str = "path_length",
) -> tuple[_StockRRTPlanner, dict]:
    planner = _StockRRTPlanner(
        system=KinematicCar(),
        start_state=np.array([0.0, 0.0, 0.0]),
        goal_state=np.array([0.7, 0.0, 0.0]),
        goal_threshold=0.16,
        min_max_control_duration=duration_range,
        propagation_step_size=0.1,
        initial_planning_time=1.0,
        goal_bias=0.25,
        optimization_objective=objective,
    )
    solutions, _ = planner.plan()
    assert solutions
    return planner, solutions[0]


def test_pathcontrol_preserves_physical_durations() -> None:
    _, solution = _planned_stock_rrt()
    assert solution["time_steps"]
    assert all(1 <= value <= 3 for value in solution["time_steps"])
    np.testing.assert_allclose(
        solution["time"],
        np.asarray(solution["time_steps"], dtype=float) * 0.1,
        atol=1e-8,
    )


def test_explicit_fixed_duration_ablation_remains_supported() -> None:
    _, solution = _planned_stock_rrt((1, 1))
    assert solution["time_steps"]
    assert set(solution["time_steps"]) == {1}
    np.testing.assert_allclose(solution["time"], 0.1, atol=1e-8)


def test_step_aware_duration_objective_matches_extracted_path_time() -> None:
    _, solution = _planned_stock_rrt((1, 3), "control_duration")
    assert solution["cost"] == pytest.approx(sum(solution["time"]))


def test_plannerdata_edges_preserve_physical_durations() -> None:
    try:
        planner, solution = _planned_stock_rrt()
        root_edges, metadata = getChildEdges(
            planner.setup,
            solution["states"][0],
            system="kinematic_car",
            nearest_match_max_dist=None,
        )
    except TypeError as exc:
        if "PlannerData: no constructor defined" in str(exc):
            pytest.skip(
                "installed OMPL nanobind build cannot construct control.PlannerData; "
                "dependency preflight rejects it for production"
            )
        raise
    assert metadata["match"] == "exact"
    assert root_edges
    assert all(1 <= edge.duration_steps <= 3 for edge in root_edges)
    assert all(
        edge.duration_seconds == pytest.approx(edge.duration_steps * 0.1)
        for edge in root_edges
    )


@pytest.mark.parametrize("planner_class", ["SSTStar", "AORRT", "AOEST"])
def test_required_custom_planner_classes_are_integration_tested_when_available(
    planner_class: str,
) -> None:
    if not hasattr(oc, planner_class):
        pytest.skip(
            f"custom OMPL class {planner_class} unavailable; dependency preflight "
            "must fail before production experiments"
        )
    planner = OMPLPlanner(
        system=KinematicCar(),
        start_state=np.array([0.0, 0.0, 0.0]),
        goal_state=np.array([0.5, 0.0, 0.0]),
        goal_threshold=0.25,
        planner_method=planner_class.lower(),
        min_max_control_duration=(1, 3),
        propagation_step_size=0.1,
        initial_planning_time=1.0,
        goal_bias=0.2,
    )
    planner.single_solve = True
    solutions, _ = planner.plan()
    assert solutions
    solution = solutions[0]
    assert all(1 <= value <= 3 for value in solution["time_steps"])
    audit = _duration_audit(planner)
    assert audit["malformed_edges"] == []
    assert audit["edge_count"] > 20
    assert len(audit["distinct_duration_steps"]) >= 2

    data = oc.PlannerData(planner.setup.getSpaceInformation())
    planner.setup.getPlanner().getPlannerData(data)
    tree_edges = []
    for source_index in range(int(data.numVertices())):
        targets = getOutgoingEdgeIndices(data, source_index)
        source = np.asarray(
            state2list(
                data.getVertex(source_index).getState(), "kinematic_car"
            ),
            dtype=float,
        )
        for target_index_raw in targets:
            target_index = int(target_index_raw)
            target = np.asarray(
                state2list(
                    data.getVertex(target_index).getState(), "kinematic_car"
                ),
                dtype=float,
            )
            seconds = float(
                data.getEdge(source_index, target_index).getDuration()
            )
            tree_edges.append((source, target, seconds))
    for index, seconds in enumerate(solution["time"]):
        matching = [
            edge_seconds
            for source, target, edge_seconds in tree_edges
            if np.linalg.norm(source - solution["states"][index]) < 1e-6
            and np.linalg.norm(target - solution["states"][index + 1]) < 1e-6
        ]
        assert matching
        assert any(value == pytest.approx(seconds) for value in matching)
        assert duration_seconds_to_steps(
            seconds, 0.1, min_steps=1, max_steps=3
        ) in {1, 2, 3}
