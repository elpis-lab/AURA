"""Kino-PAX Dubins-airplane propagation."""

from __future__ import annotations

import numpy as np
import torch
from ompl import base as ob
from ompl import control as oc

from propagators.propagator import (
    System,
    prepare_numpy_batch,
    rk4_numpy,
    rk4_torch,
    wrap_angle_numpy,
    wrap_angle_torch,
)


class DubinsAirplane(System):
    """Six-dimensional Dubins-airplane system used by OMPL and AURA."""

    def __init__(self):
        self.position_space = ob.RealVectorStateSpace(3)
        self.yaw_space = ob.SO2StateSpace()
        self.pitch_space = ob.RealVectorStateSpace(1)
        self.speed_space = ob.RealVectorStateSpace(1)
        state_space = ob.CompoundStateSpace()
        state_space.addSubspace(self.position_space, 1.0)
        state_space.addSubspace(self.yaw_space, 0.5)
        state_space.addSubspace(self.pitch_space, 0.5)
        state_space.addSubspace(self.speed_space, 1.0)
        state_space.lock()
        super().__init__(
            name="dubins_airplane",
            state_space=state_space,
            control_space=oc.RealVectorControlSpace(state_space, 3),
            state_bounds=[
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (-np.pi, np.pi),
                (-np.pi / 3.0, np.pi / 3.0),
                (0.0, 0.3),
            ],
            control_bounds=[
                (-0.3, 0.3),
                (-np.pi / 4.0, np.pi / 4.0),
                (-np.pi / 4.0, np.pi / 4.0),
            ],
            dynamics_fn=self.propagator,
            propagator_fn=self.ompl_propagator,
        )
        self.set_state_bounds(self.state_bounds)
        self.set_control_bounds(self.control_bounds)

    def set_state_bounds(self, bounds_values) -> None:
        bounds_list = [(float(low), float(high)) for low, high in bounds_values]
        if len(bounds_list) != 6:
            raise ValueError("dubins_airplane requires six state bounds")
        if not np.allclose(bounds_list[3], (-np.pi, np.pi), atol=1e-6):
            raise ValueError(
                "dubins_airplane yaw is periodic and must span [-pi, pi]"
            )
        if bounds_list[4][0] <= -np.pi / 2.0 or bounds_list[4][1] >= np.pi / 2.0:
            raise ValueError(
                "dubins_airplane pitch bounds must avoid +/-pi/2"
            )
        self.state_bounds = bounds_list

        position_bounds = ob.RealVectorBounds(3)
        for index in range(3):
            position_bounds.setLow(index, bounds_list[index][0])
            position_bounds.setHigh(index, bounds_list[index][1])
        self.position_space.setBounds(position_bounds)

        pitch_bounds = ob.RealVectorBounds(1)
        pitch_bounds.setLow(0, bounds_list[4][0])
        pitch_bounds.setHigh(0, bounds_list[4][1])
        self.pitch_space.setBounds(pitch_bounds)

        speed_bounds = ob.RealVectorBounds(1)
        speed_bounds.setLow(0, bounds_list[5][0])
        speed_bounds.setHigh(0, bounds_list[5][1])
        self.speed_space.setBounds(speed_bounds)

    def set_ompl_state(self, state, values: np.ndarray):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size < 6:
            raise ValueError("dubins_airplane state must contain six values")
        for index in range(3):
            state[0][index] = float(values[index])
        state[1].value = float(values[3])
        state[2][0] = float(values[4])
        state[3][0] = float(values[5])
        return state

    def ompl_state_to_numpy(self, state) -> np.ndarray:
        return np.array(
            [
                state[0][0],
                state[0][1],
                state[0][2],
                state[1].value,
                state[2][0],
                state[3][0],
            ],
            dtype=float,
        )

    def state_distance(self, state_a, state_b) -> float:
        first = np.asarray(state_a, dtype=float).reshape(-1)
        second = np.asarray(state_b, dtype=float).reshape(-1)
        difference = first[:6] - second[:6]
        difference[3] = wrap_angle_numpy(difference[3])
        return float(np.linalg.norm(difference))

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
        count = int(num_states)
        state_array = np.asarray(state, dtype=float).reshape(-1)
        if state_array.size < 6:
            raise ValueError("dubins_airplane expects six state values")
        samples = np.repeat(state_array[:6].reshape(1, 6), count, axis=0)
        samples[:, :3] += np.random.normal(
            0.0,
            float(position_std),
            size=(count, 3),
        )
        samples[:, 3:5] += np.random.normal(
            0.0,
            float(rotation_std),
            size=(count, 2),
        )
        samples[:, 3] = wrap_angle_numpy(samples[:, 3])
        resolved_velocity_std = float(
            position_std if velocity_std is None else velocity_std
        )
        samples[:, 5] += np.random.normal(
            0.0,
            resolved_velocity_std,
            size=count,
        )
        for index in (0, 1, 2, 4, 5):
            low, high = self.state_bounds[index]
            samples[:, index] = np.clip(samples[:, index], low, high)
        return samples

    def ompl_propagator(self, start, control, duration, state) -> None:
        start_array = self.ompl_state_to_numpy(start)
        control_array = np.array(
            [control[index] for index in range(3)],
            dtype=float,
        )
        self.set_ompl_state(
            state,
            self.propagator(start_array, control_array, duration),
        )


def derivative_numpy(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Evaluate the six-dimensional Dubins-airplane state derivative."""
    yaw, pitch, speed = state[:, 3], state[:, 4], state[:, 5]
    acceleration, yaw_rate, pitch_rate = control[:, 0], control[:, 1], control[:, 2]
    return np.stack(
        [
            speed * np.cos(pitch) * np.cos(yaw),
            speed * np.cos(pitch) * np.sin(yaw),
            speed * np.sin(pitch),
            yaw_rate,
            pitch_rate,
            acceleration,
        ],
        axis=1,
    )


def derivative_torch(
    state: torch.Tensor, control: torch.Tensor
) -> torch.Tensor:
    """Differentiable counterpart of :func:`derivative_numpy`."""
    yaw, pitch, speed = state[:, 3], state[:, 4], state[:, 5]
    acceleration, yaw_rate, pitch_rate = control[:, 0], control[:, 1], control[:, 2]
    return torch.stack(
        [
            speed * torch.cos(pitch) * torch.cos(yaw),
            speed * torch.cos(pitch) * torch.sin(yaw),
            speed * torch.sin(pitch),
            yaw_rate,
            pitch_rate,
            acceleration,
        ],
        dim=1,
    )


def propagate_numpy(
    start: np.ndarray, control: np.ndarray, duration
) -> np.ndarray:
    """Return the endpoint using RK4 steps no larger than 0.1 seconds."""
    state, command, durations, single = prepare_numpy_batch(
        start, control, duration, state_dim=6, control_dim=3
    )
    result = rk4_numpy(state, command, durations, derivative_numpy)
    result[:, 3] = wrap_angle_numpy(result[:, 3])
    return result[0] if single else result


def propagate_torch(
    start: torch.Tensor, control: torch.Tensor, duration
) -> torch.Tensor:
    """Differentiable counterpart of :func:`propagate_numpy`."""
    if start.ndim != 2 or control.ndim != 2:
        raise ValueError("Dubins-airplane tensors must be aligned 2-D batches")
    if start.shape != (control.shape[0], 6) or control.shape[1] != 3:
        raise ValueError("Dubins-airplane tensors must have shapes (N,6) and (N,3)")
    result = rk4_torch(start, control, duration, derivative_torch).clone()
    result[:, 3] = wrap_angle_torch(result[:, 3])
    return result
