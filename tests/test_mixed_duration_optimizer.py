from __future__ import annotations

import numpy as np

from propagators import double_integrator
from aura.optimization import optimize_controls
from propagators import get_system
from methods.plan import ControlEdge


def _edge(edge_id: str, control: np.ndarray, steps: int, target: np.ndarray) -> ControlEdge:
    return ControlEdge(
        source_state=np.zeros(6),
        target_state=target,
        control=control,
        duration_steps=steps,
        duration_seconds=0.2 * steps,
        edge_id=edge_id,
    )


def test_optimizer_keeps_mixed_duration_rows_aligned_and_improves() -> None:
    start = np.zeros(6)
    desired_controls = [np.array([0.12, -0.08, 0.05]), np.array([-0.1, 0.09, 0.07])]
    edges = [
        _edge(
            "short",
            np.zeros(3),
            1,
            double_integrator.propagate_numpy(start, desired_controls[0], 0.2),
        ),
        _edge(
            "long",
            np.zeros(3),
            4,
            double_integrator.propagate_numpy(start, desired_controls[1], 0.8),
        ),
    ]
    result = optimize_controls(
        get_system("double_integrator"),
        next_state=start,
        child_edges=edges,
        num_states=3,
        position_std=0.0,
        velocity_std=0.0,
        integration_step_size=0.2,
        num_steps=120,
        learning_rate=0.08,
        requested_device="cpu",
    )
    assert result is not None
    assert result["edge_ids"] == ["short"] * 3 + ["long"] * 3
    assert result["duration_steps"].tolist() == [1] * 3 + [4] * 3
    for row in result["row_metadata"]:
        assert row["finite"]
        assert row["optimized_loss"] < row["original_loss"]


def test_changing_one_child_duration_only_changes_its_rows() -> None:
    start = np.zeros(6)
    target = np.array([0.02, -0.01, 0.03, 0.1, -0.05, 0.08])
    base = [
        _edge("a", np.zeros(3), 1, target),
        _edge("b", np.zeros(3), 2, target),
    ]
    changed = [
        base[0],
        _edge("b", np.zeros(3), 4, target),
    ]
    kwargs = dict(
        system=get_system("double_integrator"),
        next_state=start,
        num_states=1,
        position_std=0.0,
        velocity_std=0.0,
        integration_step_size=0.2,
        num_steps=40,
        learning_rate=0.05,
        requested_device="cpu",
    )
    torch_seed = 17
    np.random.seed(torch_seed)
    first = optimize_controls(child_edges=base, **kwargs)
    np.random.seed(torch_seed)
    second = optimize_controls(child_edges=changed, **kwargs)
    first_controls = first["optimized_controls"].numpy()
    second_controls = second["optimized_controls"].numpy()
    np.testing.assert_allclose(first_controls[0], second_controls[0], atol=1e-7)
    assert not np.allclose(first_controls[1], second_controls[1], atol=1e-4)


def test_each_child_batch_contains_the_exact_measured_state() -> None:
    start = np.array([0.2, -0.1, 0.4, 0.03, -0.02, 0.01])
    target = double_integrator.propagate_numpy(
        start, np.array([0.1, -0.05, 0.02]), 0.2
    )
    edges = [
        _edge("a", np.zeros(3), 1, target),
        _edge("b", np.zeros(3), 2, target),
    ]
    np.random.seed(23)
    result = optimize_controls(
        get_system("double_integrator"),
        next_state=start,
        child_edges=edges,
        num_states=4,
        position_std=0.1,
        velocity_std=0.05,
        integration_step_size=0.2,
        num_steps=2,
        learning_rate=0.01,
        requested_device="cpu",
    )
    starts = result["start_states"].numpy()
    np.testing.assert_allclose(starts[0], start)
    np.testing.assert_allclose(starts[4], start)
