"""Constant-control kinematic-car propagation."""

from __future__ import annotations

import numpy as np
import torch
from ompl import base as ob
from ompl import control as oc

from propagators.propagator import (
    System,
    prepare_numpy_batch,
    prepare_torch_duration,
    wrap_angle_numpy,
    wrap_angle_torch,
)


WHEELBASE = 0.1385 + 0.158


class KinematicCar(System):
    """SE(2) bicycle-model system used by OMPL and AURA."""

    def __init__(self):
        state_space = ob.SE2StateSpace()
        super().__init__(
            name="kinematic_car",
            state_space=state_space,
            control_space=oc.RealVectorControlSpace(state_space, 2),
            state_bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            control_bounds=[(-0.2, 0.95), (-0.30, 0.30)],
            dynamics_fn=self.propagator,
            propagator_fn=self.ompl_propagator,
        )
        self.set_state_bounds(self.state_bounds)
        self.set_control_bounds(self.control_bounds)

    def set_state_bounds(self, bounds_values) -> None:
        bounds_list = [(float(low), float(high)) for low, high in bounds_values]
        if len(bounds_list) != 2:
            raise ValueError("kinematic_car requires two position bounds")
        self.state_bounds = bounds_list
        bounds = ob.RealVectorBounds(2)
        for index, (low, high) in enumerate(bounds_list):
            bounds.setLow(index, low)
            bounds.setHigh(index, high)
        self.state_space.setBounds(bounds)

    def set_ompl_state(self, state, values: np.ndarray):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size < 3:
            raise ValueError("kinematic_car state must contain x, y, and yaw")
        state.setX(float(values[0]))
        state.setY(float(values[1]))
        state.setYaw(float(values[2]))
        return state

    def ompl_state_to_numpy(self, state) -> np.ndarray:
        return np.array(
            [state.getX(), state.getY(), state.getYaw()],
            dtype=float,
        )

    def state_distance(self, state_a, state_b) -> float:
        first = np.asarray(state_a, dtype=float).reshape(-1)
        second = np.asarray(state_b, dtype=float).reshape(-1)
        angle = wrap_angle_numpy(first[2] - second[2])
        return float(np.linalg.norm(first[:2] - second[:2]) + 0.5 * abs(angle))

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
        del velocity_std
        count = int(num_states)
        state_array = np.asarray(state, dtype=float).reshape(-1)
        if state_array.size < 3:
            raise ValueError("kinematic_car expects three state values")
        samples = np.repeat(state_array[:3].reshape(1, 3), count, axis=0)
        samples[:, :2] += np.random.normal(
            0.0,
            float(position_std),
            size=(count, 2),
        )
        samples[:, 2] += np.random.normal(
            0.0,
            float(rotation_std),
            size=count,
        )
        samples[:, 2] = wrap_angle_numpy(samples[:, 2])
        return samples

    def ompl_propagator(self, start, control, duration, state) -> None:
        start_array = np.array(
            [start.getX(), start.getY(), start.getYaw()],
            dtype=float,
        )
        control_array = np.array([control[0], control[1]], dtype=float)
        result = self.propagator(start_array, control_array, duration)
        state.setX(result[0])
        state.setY(result[1])
        state.setYaw(result[2])


def propagate_numpy(
    start: np.ndarray,
    control: np.ndarray,
    duration,
    *,
    wheelbase: float = WHEELBASE,
) -> np.ndarray:
    """Return the closed-form bicycle-model endpoint."""
    state, command, durations, single = prepare_numpy_batch(
        start, control, duration, state_dim=3, control_dim=2
    )
    x, y, yaw = state[:, 0], state[:, 1], state[:, 2]
    velocity, steering = command[:, 0], command[:, 1]
    yaw_rate = velocity * np.tan(steering) / float(wheelbase)
    end_yaw = yaw + yaw_rate * durations
    straight = np.abs(yaw_rate) < 1e-9
    safe_rate = np.where(straight, 1.0, yaw_rate)
    curve_x = x + velocity / safe_rate * (np.sin(end_yaw) - np.sin(yaw))
    curve_y = y - velocity / safe_rate * (np.cos(end_yaw) - np.cos(yaw))
    line_x = x + velocity * np.cos(yaw) * durations
    line_y = y + velocity * np.sin(yaw) * durations
    result = np.stack(
        [
            np.where(straight, line_x, curve_x),
            np.where(straight, line_y, curve_y),
            wrap_angle_numpy(end_yaw),
        ],
        axis=1,
    )
    return result[0] if single else result


def propagate_torch(
    start: torch.Tensor,
    control: torch.Tensor,
    duration,
    *,
    wheelbase: float = WHEELBASE,
) -> torch.Tensor:
    """Differentiable counterpart of :func:`propagate_numpy`."""
    if start.ndim != 2 or control.ndim != 2:
        raise ValueError("car tensors must be aligned 2-D batches")
    if start.shape != (control.shape[0], 3) or control.shape[1] != 2:
        raise ValueError("car tensors must have shapes (N,3) and (N,2)")
    durations = prepare_torch_duration(duration, start)
    x, y, yaw = start[:, 0], start[:, 1], start[:, 2]
    velocity, steering = control[:, 0], control[:, 1]
    yaw_rate = velocity * torch.tan(steering) / float(wheelbase)
    end_yaw = yaw + yaw_rate * durations
    straight = torch.abs(yaw_rate) < 1e-7
    safe_rate = torch.where(straight, torch.ones_like(yaw_rate), yaw_rate)
    curve_x = x + velocity / safe_rate * (torch.sin(end_yaw) - torch.sin(yaw))
    curve_y = y - velocity / safe_rate * (torch.cos(end_yaw) - torch.cos(yaw))
    line_x = x + velocity * torch.cos(yaw) * durations
    line_y = y + velocity * torch.sin(yaw) * durations
    return torch.stack(
        [
            torch.where(straight, line_x, curve_x),
            torch.where(straight, line_y, curve_y),
            wrap_angle_torch(end_yaw),
        ],
        dim=1,
    )
