"""Common OMPL system contract and numerical propagation helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from ompl import base as ob
from ompl import control as oc

from utils.control_duration import validate_duration_range


INTEGRATION_STEP = 0.1


@dataclass
class System:
    """Bridge one nominal propagator to OMPL planning and state sampling."""

    name: str
    state_space: ob.StateSpace
    control_space: oc.ControlSpace
    state_bounds: list[tuple[float, float]]
    control_bounds: list[tuple[float, float]]
    dynamics_fn: Callable
    propagator_fn: Callable
    propagation_step_size: float | None = None
    min_control_duration: int = 1
    max_control_duration: int | None = None

    def create_spaces(self):
        return self.state_space, self.control_space

    def propagate(
        self,
        state: np.ndarray,
        control: np.ndarray,
        duration: float,
    ) -> np.ndarray:
        """Return the nominal endpoint for one state-control-duration input."""

        return self.dynamics_fn(state, control, duration)

    def sample_random_states(
        self,
        state: np.ndarray,
        num_states: int,
        position_std: float,
        rotation_std: float,
        velocity_std: float | None = None,
    ) -> np.ndarray:
        raise NotImplementedError(
            f"{self.name} does not define optimizer state sampling"
        )

    def configure_propagation_step_size(self, step_size: float) -> None:
        step = float(step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError(
                "propagation_step_size must be finite and positive, "
                f"got {step}"
            )
        self.propagation_step_size = step

    def configure_duration_contract(
        self,
        step_size: float,
        minimum_steps: int,
        maximum_steps: int,
    ) -> None:
        minimum, maximum = validate_duration_range(
            minimum_steps,
            maximum_steps,
        )
        self.configure_propagation_step_size(step_size)
        self.min_control_duration = minimum
        self.max_control_duration = maximum

    def set_state_bounds(self, bounds_values) -> None:
        """Set bounds for a real-vector state space."""

        bounds_list = [(float(low), float(high)) for low, high in bounds_values]
        if len(bounds_list) != len(self.state_bounds):
            raise ValueError(
                f"{self.name} expects {len(self.state_bounds)} state bounds, "
                f"got {len(bounds_list)}"
            )
        self.state_bounds = bounds_list
        bounds = ob.RealVectorBounds(len(bounds_list))
        for index, (low, high) in enumerate(bounds_list):
            bounds.setLow(index, low)
            bounds.setHigh(index, high)
        self.state_space.setBounds(bounds)

    def set_control_bounds(self, bounds_values) -> None:
        bounds_list = [(float(low), float(high)) for low, high in bounds_values]
        if len(bounds_list) != len(self.control_bounds):
            raise ValueError(
                f"{self.name} expects {len(self.control_bounds)} control bounds, "
                f"got {len(bounds_list)}"
            )
        self.control_bounds = bounds_list
        self.set_control_sampling_bounds(bounds_list)

    def set_control_sampling_bounds(self, bounds_values) -> None:
        """Restrict OMPL proposals without changing actuator feasibility."""

        bounds_list = [(float(low), float(high)) for low, high in bounds_values]
        if len(bounds_list) != len(self.control_bounds):
            raise ValueError(
                f"{self.name} expects {len(self.control_bounds)} sampling bounds, "
                f"got {len(bounds_list)}"
            )
        for index, ((low, high), (actuator_low, actuator_high)) in enumerate(
            zip(bounds_list, self.control_bounds)
        ):
            if low < actuator_low or high > actuator_high or low > high:
                raise ValueError(
                    f"{self.name} control sampling bound {index} {(low, high)} "
                    f"must be inside actuator bound {(actuator_low, actuator_high)}"
                )
        bounds = ob.RealVectorBounds(len(bounds_list))
        for index, (low, high) in enumerate(bounds_list):
            bounds.setLow(index, low)
            bounds.setHigh(index, high)
        self.control_space.setBounds(bounds)

    def set_ompl_state(self, state, values: np.ndarray):
        values = np.asarray(values, dtype=float).reshape(-1)
        for index, value in enumerate(values):
            state[index] = float(value)
        return state

    def ompl_state_to_numpy(self, state) -> np.ndarray:
        return np.array(
            [
                float(state[index])
                for index in range(self.state_space.getDimension())
            ],
            dtype=float,
        )

    def state_distance(self, state_a, state_b) -> float:
        first = np.asarray(state_a, dtype=float).reshape(-1)
        second = np.asarray(state_b, dtype=float).reshape(-1)
        return float(np.linalg.norm(first - second))


def wrap_angle_numpy(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def prepare_torch_duration(duration, reference: torch.Tensor) -> torch.Tensor:
    """Return one duration per row on the reference tensor's device and dtype."""
    if isinstance(duration, torch.Tensor):
        values = duration.to(device=reference.device, dtype=reference.dtype)
    else:
        values = torch.as_tensor(
            duration, device=reference.device, dtype=reference.dtype
        )
    if values.ndim == 0:
        values = values.repeat(reference.shape[0])
    values = values.reshape(-1)
    if values.numel() == 1 and reference.shape[0] > 1:
        values = values.repeat(reference.shape[0])
    if values.numel() != reference.shape[0]:
        raise ValueError("duration batch must have one value per state")
    return values


def prepare_numpy_batch(
    start: np.ndarray,
    control: np.ndarray,
    duration,
    *,
    state_dim: int,
    control_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Validate and align NumPy state, control, and duration batches."""
    state = np.asarray(start, dtype=float)
    command = np.asarray(control, dtype=float)
    single = state.ndim == 1
    state = np.atleast_2d(state)
    command = np.atleast_2d(command)
    if state.shape[1] != int(state_dim):
        raise ValueError(f"expected {state_dim} state values, got shape {state.shape}")
    if command.shape[1] != int(control_dim):
        raise ValueError(
            f"expected {control_dim} control values, got shape {command.shape}"
        )
    if len(command) == 1 and len(state) > 1:
        command = np.repeat(command, len(state), axis=0)

    durations = np.asarray(duration, dtype=float)
    if durations.ndim == 0:
        durations = np.full(len(state), float(durations), dtype=float)
    durations = durations.reshape(-1)
    if len(durations) == 1 and len(state) > 1:
        durations = np.repeat(durations, len(state))
    if len(state) != len(command) or len(state) != len(durations):
        raise ValueError("state, control, and duration batches must align")
    if np.any(~np.isfinite(durations)) or np.any(durations < 0.0):
        raise ValueError("propagation durations must be finite and nonnegative")
    return state, command, durations, single


def rk4_numpy(
    state: np.ndarray,
    control: np.ndarray,
    duration: np.ndarray,
    derivative: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    max_step: float = INTEGRATION_STEP,
) -> np.ndarray:
    """Integrate a NumPy batch with per-row durations and bounded RK4 steps."""
    step_counts = np.maximum(1, np.ceil(duration / float(max_step)).astype(int))
    row_steps = duration / step_counts
    result = state.copy()
    for step_index in range(int(step_counts.max(initial=1))):
        active = step_index < step_counts
        dt = np.where(active, row_steps, 0.0)[:, None]
        k1 = derivative(result, control)
        k2 = derivative(result + 0.5 * dt * k1, control)
        k3 = derivative(result + 0.5 * dt * k2, control)
        k4 = derivative(result + dt * k3, control)
        result = result + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return result


def rk4_torch(
    state: torch.Tensor,
    control: torch.Tensor,
    duration,
    derivative: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    max_step: float = INTEGRATION_STEP,
) -> torch.Tensor:
    """Differentiable counterpart of :func:`rk4_numpy`."""
    if state.ndim != 2 or control.ndim != 2 or len(state) != len(control):
        raise ValueError("Torch state and control inputs must be aligned 2-D batches")
    durations = prepare_torch_duration(duration, state)
    if not bool(torch.all(torch.isfinite(durations))) or bool(
        torch.any(durations < 0.0)
    ):
        raise ValueError("propagation durations must be finite and nonnegative")
    step_counts = torch.clamp(
        torch.ceil(durations / float(max_step)).to(dtype=torch.long), min=1
    )
    row_steps = durations / step_counts.to(dtype=state.dtype)
    result = state
    for step_index in range(int(step_counts.max().detach().cpu())):
        active = step_counts > step_index
        dt = torch.where(active, row_steps, torch.zeros_like(row_steps)).unsqueeze(1)
        k1 = derivative(result, control)
        k2 = derivative(result + 0.5 * dt * k1, control)
        k3 = derivative(result + 0.5 * dt * k2, control)
        k4 = derivative(result + dt * k3, control)
        result = result + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return result
