from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from aura.AURA import AURA
from aura.optimization import weighted_state_loss
from propagators import KinematicCar, kinematic_car
from utils import auraHandler
from methods.plan import ControlEdge
from utils.utils import arrayDistance


def _aura(*, obstacles=None) -> AURA:
    aura = AURA.__new__(AURA)
    aura.system = KinematicCar()
    aura.system.configure_propagation_step_size(0.2)
    aura.propagation_step_size = 0.2
    aura.planner = SimpleNamespace(
        motion_validation_step_size=0.2,
        propagation_step_size=0.2,
        optimizer_child_match_tolerance=1e-6,
        obstacle_config=obstacles,
        duration_seconds_to_steps=lambda seconds: int(round(seconds / 0.2)),
    )
    aura.simulator = SimpleNamespace(
        config={
            "propagation_step_size": 0.2,
            "state_bounds": [(-2.0, 3.0), (-2.0, 2.0)],
            "obstacles": obstacles,
            "execution_safety_radius": 0.0,
        }
    )
    aura.last_control_decision = {}
    return aura


def _edge(
    edge_id: str,
    source: np.ndarray,
    target: np.ndarray,
    control: np.ndarray,
    steps: int,
) -> ControlEdge:
    return ControlEdge(
        source_state=source,
        target_state=target,
        control=control,
        duration_steps=steps,
        duration_seconds=steps * 0.2,
        edge_id=edge_id,
    )


def _optimizer_row(
    edge: ControlEdge, start: np.ndarray, optimized_control: np.ndarray
) -> dict:
    return {
        "optimized_controls": np.asarray([optimized_control], dtype=float),
        "start_states": np.asarray([start], dtype=float),
        "target_states": np.asarray([edge.target_state], dtype=float),
        "duration_seconds": np.asarray([edge.duration_seconds]),
        "duration_steps": np.asarray([edge.duration_steps]),
        "edge_ids": [edge.edge_id],
        "row_metadata": [{"finite": True}],
    }


def test_selected_optimized_control_retains_its_child_duration() -> None:
    aura = _aura()
    nominal_source = np.array([0.0, 0.0, 0.0])
    original_control = np.array([0.4, 0.0])
    edge = _edge(
        "branch:2",
        nominal_source,
        kinematic_car.propagate_numpy(nominal_source, original_control, 0.4),
        original_control,
        2,
    )
    actual_current = np.array([-0.04, 0.0, 0.0])
    result = _optimizer_row(edge, actual_current, np.array([0.5, 0.0]))
    fallback = _edge(
        "new-plan",
        actual_current,
        edge.target_state,
        np.array([0.3, 0.0]),
        3,
    )
    selected = aura.pick_next_control(
        aura.system,
        result,
        actual_current,
        edge.target_state,
        [edge],
        fallback_edge=fallback,
    )
    assert selected.source == "optimized"
    np.testing.assert_allclose(selected.control, [0.5, 0.0])
    assert selected.duration_steps == 2
    assert selected.duration_seconds == 0.4
    assert selected.edge_id == edge.edge_id


def test_unseen_replanned_edge_uses_that_new_edges_original_pair() -> None:
    aura = _aura()
    source = np.zeros(3)
    old_edge = _edge("old", source, np.array([0.2, 0.0, 0.0]), np.array([0.5, 0]), 2)
    new_edge = _edge("new", source, np.array([0.0, 0.5, 0.0]), np.array([0.2, 0.1]), 3)
    selected = aura.pick_next_control(
        aura.system,
        _optimizer_row(old_edge, source, np.array([0.4, 0.0])),
        source,
        new_edge.target_state,
        [old_edge],
        fallback_edge=new_edge,
    )
    assert selected.source == "planner"
    assert selected.edge_id == "new"
    assert selected.duration_steps == 3
    np.testing.assert_allclose(selected.control, new_edge.control)
    assert "unseen_replanned_edge" in aura.last_control_decision["reason"]


def test_invalid_optimized_control_falls_back_to_same_edge_pair() -> None:
    aura = _aura()
    source = np.zeros(3)
    edge = _edge("edge", source, np.array([0.16, 0.0, 0.0]), np.array([0.4, 0]), 2)
    selected = aura.pick_next_control(
        aura.system,
        _optimizer_row(edge, source, np.array([4.0, 0.0])),
        source,
        edge.target_state,
        [edge],
        fallback_edge=edge,
    )
    assert selected.source == "planner"
    assert selected.edge_id == edge.edge_id
    assert selected.duration_steps == edge.duration_steps
    np.testing.assert_allclose(selected.control, edge.control)
    assert aura.last_control_decision["optimized_in_bounds"] is False


def test_float32_roundoff_at_control_bound_is_clipped_and_accepted() -> None:
    aura = _aura()
    source = np.zeros(3)
    optimized = np.asarray([0.5, np.float32(0.3)], dtype=np.float32)
    target = kinematic_car.propagate_numpy(source, [0.5, 0.3], 0.2)
    edge = _edge("edge", source, target, np.array([0.4, 0.0]), 1)
    result = _optimizer_row(edge, source, optimized)
    result["optimized_controls"] = np.asarray([optimized], dtype=np.float32)
    selected = aura.pick_next_control(
        aura.system,
        result,
        source,
        target,
        [edge],
        fallback_edge=edge,
    )
    assert selected.source == "optimized"
    assert selected.control[1] == 0.3
    assert aura.last_control_decision["optimized_in_bounds"] is True


def test_optimizer_selector_and_report_use_the_same_se2_metric() -> None:
    aura = _aura()
    target = np.zeros(3)
    state = np.array([0.6, 0.8, 0.4])
    expected = 1.0 + 0.5 * 0.4
    assert np.isclose(
        arrayDistance(target, state, system="kinematic_car"), expected
    )
    assert np.isclose(aura.system.state_distance(target, state), expected)
    assert np.isclose(
        aura.batch_state_distance(target, state, "kinematic_car")[0], expected
    )
    loss = weighted_state_loss(
        "kinematic_car",
        torch.as_tensor(state, dtype=torch.float64).reshape(1, -1),
        torch.as_tensor(target, dtype=torch.float64).reshape(1, -1),
        reduction="none",
    )
    assert np.isclose(float(torch.sqrt(loss)[0]), expected)


def test_full_curve_rejects_valid_endpoint_with_colliding_middle() -> None:
    obstacles = {
        "enabled": True,
        "safety_radius": 0.0,
        "circles": [(0.4, 0.0, 0.08)],
    }
    aura = _aura(obstacles=obstacles)
    start = np.zeros(3)
    control = np.array([0.5, 0.0])
    duration = 1.6
    endpoint = kinematic_car.propagate_numpy(start, control, duration)
    assert endpoint[0] == 0.8
    assert aura.control_curve_valid(start, control, duration) is False


def test_disconnected_previous_suffix_is_not_reported_as_continuous() -> None:
    previous = {
        "states": [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),
        ],
        "controls": [
            np.array([0.5, 0.0]),
            np.array([0.5, 0.0]),
        ],
        "time": [0.2, 0.2],
    }
    candidates, info = auraHandler.reachable_plan_candidates(
        [],
        previous,
        np.array([0.0, 1.0, 0.0]),
        system_name="kinematic_car",
        propagation_step_size=0.2,
        continuity_max_distance=0.1,
    )
    assert candidates == []
    assert info["accepted"] == 0
    assert info["rejected"] == 1
