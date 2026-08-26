"""Constant-acceleration double-integrator propagation."""

from __future__ import annotations

import numpy as np
import torch
from ompl import base as ob
from ompl import control as oc

from propagators.propagator import (
    System,
    prepare_numpy_batch,
    prepare_torch_duration,
)


class DoubleIntegrator(System):
    """Three-dimensional double-integrator system used by OMPL and AURA."""

    def __init__(self):
        state_space = ob.RealVectorStateSpace(6)
        super().__init__(
            name="double_integrator",
            state_space=state_space,
            control_space=oc.RealVectorControlSpace(state_space, 3),
            state_bounds=[
                (-3.0, 3.0),
                (-3.0, 3.0),
                (0.0, 3.0),
                (-0.5, 0.5),
                (-0.5, 0.5),
                (-0.5, 0.5),
            ],
            control_bounds=[(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)],
            dynamics_fn=self.propagator,
            propagator_fn=self.ompl_propagator,
        )
        self.set_state_bounds(self.state_bounds)
        self.set_control_bounds(self.control_bounds)

    def propagator(
        self,
        start: np.ndarray,
        control: np.ndarray,
        duration: float,
    ) -> np.ndarray:
        return propagate_numpy(start, control, duration)

    def sample_random_states(
        self,
        state: np.ndarray,
        num_states: int,
        position_std: float,
        rotation_std: float,
        velocity_std: float | None = None,
    ) -> np.ndarray:
        del rotation_std
        count = int(num_states)
        state_array = np.asarray(state, dtype=float).reshape(-1)
        if state_array.size < 6:
            raise ValueError("double_integrator expects six state values")
        samples = np.repeat(state_array[:6].reshape(1, 6), count, axis=0)
        samples[:, :3] += np.random.normal(
            0.0,
            float(position_std),
            size=(count, 3),
        )
        resolved_velocity_std = float(
            position_std if velocity_std is None else velocity_std
        )
        samples[:, 3:6] += np.random.normal(
            0.0,
            resolved_velocity_std,
            size=(count, 3),
        )
        return samples

    def ompl_propagator(self, start, control, duration, state) -> None:
        start_array = np.array([start[index] for index in range(6)], dtype=float)
        control_array = np.array(
            [control[index] for index in range(3)],
            dtype=float,
        )
        result = self.propagator(start_array, control_array, duration)
        for index in range(6):
            state[index] = result[index]


def propagate_numpy(
    start: np.ndarray, control: np.ndarray, duration
) -> np.ndarray:
    """Return the exact 3-D double-integrator endpoint."""
    state, command, durations, single = prepare_numpy_batch(
        start, control, duration, state_dim=6, control_dim=3
    )
    duration_columns = durations[:, None]
    position = (
        state[:, :3]
        + state[:, 3:6] * duration_columns
        + 0.5 * command * duration_columns**2
    )
    velocity = state[:, 3:6] + command * duration_columns
    result = np.concatenate([position, velocity], axis=1)
    return result[0] if single else result


def propagate_torch(
    start: torch.Tensor, control: torch.Tensor, duration
) -> torch.Tensor:
    """Differentiable counterpart of :func:`propagate_numpy`."""
    if start.ndim != 2 or control.ndim != 2:
        raise ValueError("double-integrator tensors must be aligned 2-D batches")
    if start.shape != (control.shape[0], 6) or control.shape[1] != 3:
        raise ValueError(
            "double-integrator tensors must have shapes (N,6) and (N,3)"
        )
    durations = prepare_torch_duration(duration, start).unsqueeze(1)
    position = start[:, :3] + start[:, 3:6] * durations + 0.5 * control * durations**2
    velocity = start[:, 3:6] + control * durations
    return torch.cat([position, velocity], dim=1)
