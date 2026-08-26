from __future__ import annotations

import numpy as np
import pytest
import torch

from aura.optimization import clamp_controls, optimize_controls
from propagators import dubins_airplane
from propagators import get_system
from methods.plan import ControlEdge


def direct_ompl_propagation(system, start, control, duration):
    ompl_start = system.state_space.allocState()
    ompl_result = system.state_space.allocState()
    ompl_control = system.control_space.allocControl()
    system.set_ompl_state(ompl_start, np.asarray(start, dtype=float))
    for index, value in enumerate(control):
        ompl_control[index] = float(value)
    system.ompl_propagator(ompl_start, ompl_control, duration, ompl_result)
    return system.ompl_state_to_numpy(ompl_result)


def test_derivative_matches_equations() -> None:
    state = np.array([[0.1, 0.2, 0.3, np.pi / 2.0, np.pi / 6.0, 0.24]])
    control = np.array([[0.08, -0.3, 0.2]])
    derivative = dubins_airplane.derivative_numpy(state, control)[0]
    expected = np.array(
        [
            0.24 * np.cos(np.pi / 6.0) * np.cos(np.pi / 2.0),
            0.24 * np.cos(np.pi / 6.0) * np.sin(np.pi / 2.0),
            0.24 * np.sin(np.pi / 6.0),
            -0.3,
            0.2,
            0.08,
        ]
    )
    np.testing.assert_allclose(derivative, expected, atol=1e-14)


@pytest.mark.parametrize("duration", [0.0, 0.1, 0.37, 0.8])
def test_numpy_torch_batch_and_ompl_agree(duration: float) -> None:
    starts = np.array(
        [
            [0.2, 0.3, 0.4, 0.2, -0.1, 0.18],
            [0.6, 0.4, 0.2, -0.4, 0.15, 0.12],
        ],
        dtype=float,
    )
    controls = np.array([[0.03, 0.2, -0.1], [-0.02, -0.15, 0.12]])
    expected = dubins_airplane.propagate_numpy(starts, controls, duration)
    actual = (
        dubins_airplane.propagate_torch(
            torch.tensor(starts, dtype=torch.float64),
            torch.tensor(controls, dtype=torch.float64),
            duration,
        )
        .detach()
        .numpy()
    )
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
    system = get_system("dubins_airplane")
    ompl = direct_ompl_propagation(system, starts[0], controls[0], duration)
    np.testing.assert_allclose(ompl, expected[0], atol=2e-12, rtol=2e-12)


def test_rk4_semigroup_and_control_gradients() -> None:
    start = np.array([0.2, 0.3, 0.4, 0.2, -0.1, 0.18])
    control = np.array([0.03, 0.2, -0.1])
    one_call = dubins_airplane.propagate_numpy(start, control, 0.8)
    repeated = start.copy()
    for _ in range(8):
        repeated = dubins_airplane.propagate_numpy(repeated, control, 0.1)
    np.testing.assert_allclose(one_call, repeated, atol=2e-12, rtol=2e-12)

    torch_control = torch.tensor(
        control[None], dtype=torch.float64, requires_grad=True
    )
    loss = dubins_airplane.propagate_torch(
        torch.tensor(start[None], dtype=torch.float64), torch_control, 0.4
    ).square().sum()
    gradient = torch.autograd.grad(loss, torch_control)[0]
    assert torch.all(torch.isfinite(gradient))
    assert torch.all(torch.abs(gradient) > 1e-10)


def test_optimizer_control_dimensions_are_clipped() -> None:
    system = get_system("dubins_airplane")
    bounds = system.control_bounds
    controls = torch.tensor(
        [
            [high + 100.0 for _low, high in bounds],
            [low - 100.0 for low, _high in bounds],
        ],
        dtype=torch.float64,
    )
    original = torch.zeros_like(controls)
    clamp_controls(
        controls,
        bounds,
        original_controls=original,
        system=system.name,
    )
    np.testing.assert_allclose(
        controls[0].numpy(),
        np.asarray([high for _low, high in bounds]),
        atol=0.0,
    )
    np.testing.assert_allclose(
        controls[1].numpy(),
        np.asarray([low for low, _high in bounds]),
        atol=0.0,
    )


def test_optimizer_state_sampling_respects_bounds() -> None:
    system = get_system("dubins_airplane")
    state = [0.0, 1.0, 0.5, 0.0, 0.0, 0.15]
    samples = np.asarray(
        system.sample_random_states(
            state,
            256,
            position_std=10.0,
            rotation_std=10.0,
            velocity_std=100.0,
        )
    )
    assert np.all(np.isfinite(samples))
    for index, (low, high) in enumerate(system.state_bounds):
        assert np.all(samples[:, index] >= low)
        assert np.all(samples[:, index] <= high)


def test_dimensions_and_actuator_bounds() -> None:
    system = get_system("dubins_airplane")
    assert system.state_space.getDimension() == 6
    np.testing.assert_allclose(
        system.control_bounds,
        [(-0.3, 0.3), (-np.pi / 4, np.pi / 4), (-np.pi / 4, np.pi / 4)],
    )


def test_optimizer_preserves_one_to_five_second_edge_durations() -> None:
    system = get_system("dubins_airplane")
    system.configure_duration_contract(1.0, 1, 5)
    start = np.asarray([0.2, 0.2, 0.2, 0.0, 0.0, 0.15], dtype=float)
    controls = [
        np.asarray([0.01, 0.05, 0.02], dtype=float),
        np.asarray([0.0, 0.1, -0.02], dtype=float),
    ]
    durations = [1.0, 5.0]
    edges = [
        ControlEdge(
            source_state=start,
            target_state=system.propagate(start, control, duration),
            control=control,
            duration_steps=int(duration),
            duration_seconds=duration,
            edge_id=f"edge-{int(duration)}",
        )
        for control, duration in zip(controls, durations)
    ]
    result = optimize_controls(
        system=system,
        next_state=start,
        num_states=3,
        position_std=0.001,
        rotation_std=0.001,
        velocity_std=0.001,
        num_steps=2,
        learning_rate=0.001,
        child_edges=edges,
        integration_step_size=1.0,
        requested_device="cpu",
    )
    assert result is not None and result["optimization_success"]
    assert result["max_duration_steps"] == 5
    assert result["edge_ids"] == ["edge-1"] * 3 + ["edge-5"] * 3
    np.testing.assert_allclose(
        result["duration_seconds"].numpy(), [1] * 3 + [5] * 3
    )
    np.testing.assert_array_equal(
        result["duration_steps"].numpy(), [1] * 3 + [5] * 3
    )
    optimized = result["optimized_controls"].numpy()
    assert np.all(np.isfinite(optimized))
    for index, (low, high) in enumerate(system.control_bounds):
        assert np.all(optimized[:, index] >= low)
        assert np.all(optimized[:, index] <= high)
