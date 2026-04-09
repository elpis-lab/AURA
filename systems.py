from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import numpy as np
from ompl import base as ob
from ompl import control as oc

from geometry.pose import SE2Pose
from pushing_dynamics import get_pushing_model


@dataclass
class System:
    """Base system definition for planning/execution."""

    name: str
    state_space: ob.StateSpace
    control_space: oc.ControlSpace
    state_bounds: list[tuple[float, float]]
    control_bounds: list[tuple[float, float]]
    dynamics_fn: Callable
    propagator_fn: Callable

    def create_spaces(self):
        return self.state_space, self.control_space

    def propagate(self, state: np.ndarray, control: np.ndarray, duration: float) -> np.ndarray:
        """Numpy propagation helper."""
        return self.dynamics_fn(state, control, duration)


class kinematicCar(System):
    """SE2 kinematic car system."""

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
        bounds = ob.RealVectorBounds(2)
        for i, (low, high) in enumerate(self.state_bounds):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        self.state_space.setBounds(bounds)
        bounds = ob.RealVectorBounds(2)

        for i, (low, high) in enumerate(self.control_bounds):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        self.control_space.setBounds(bounds)

        self._wheelbase = 0.1385 + 0.158

    def propagator(self, start: np.ndarray, control: np.ndarray, duration: float) -> np.ndarray:
        """Kinematic car propagation."""
        x, y, yaw = start
        u_vel, u_phi = control

        x_dot = u_vel * np.cos(yaw)
        y_dot = u_vel * np.sin(yaw)
        yaw_dot = (u_vel / self._wheelbase) * np.tan(u_phi)

        new_x = x + x_dot * duration
        new_y = y + y_dot * duration
        new_yaw = (yaw + yaw_dot * duration + np.pi) % (2 * np.pi) - np.pi

        return np.array([new_x, new_y, new_yaw], dtype=float)

    def ompl_propagator(self, start, control, duration, state):
        start_np = np.array([start.getX(), start.getY(), start.getYaw()], dtype=float)
        control_np = np.array([control[0], control[1]], dtype=float)
        result = self.propagator(start_np, control_np, duration)
        state.setX(result[0])
        state.setY(result[1])
        state.setYaw(result[2])


class doubleIntegrator(System):
    """3D double integrator in R^6 with acceleration control in R^3."""

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
        bounds = ob.RealVectorBounds(6)
        for i, (low, high) in enumerate(self.state_bounds):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        self.state_space.setBounds(bounds)

        bounds = ob.RealVectorBounds(3)
        for i, (low, high) in enumerate(self.control_bounds):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        self.control_space.setBounds(bounds)

    def propagator(self, start: np.ndarray, control: np.ndarray, duration: float) -> np.ndarray:
        """3D double-integrator propagation."""
        x, y, z, vx, vy, vz = start
        ax, ay, az = control

        x_new = x + vx * duration + 0.5 * ax * (duration**2)
        y_new = y + vy * duration + 0.5 * ay * (duration**2)
        z_new = z + vz * duration + 0.5 * az * (duration**2)
        vx_new = vx + ax * duration
        vy_new = vy + ay * duration
        vz_new = vz + az * duration

        return np.array([x_new, y_new, z_new, vx_new, vy_new, vz_new], dtype=float)

    def ompl_propagator(self, start, control, duration, state):
        start_np = np.array([start[i] for i in range(6)], dtype=float)
        control_np = np.array([control[i] for i in range(3)], dtype=float)
        result = self.propagator(start_np, control_np, duration)
        for i in range(6):
            state[i] = result[i]


class pushingObject(System):
    """SE2 pushing system with learned-model dynamics."""

    def __init__(self):
        self.object_shape = np.array([0.1628, 0.2139, 0.0676], dtype=float)
        state_space = ob.SE2StateSpace()

        super().__init__(
            name="pushing_object",
            state_space=state_space,
            control_space=oc.RealVectorControlSpace(state_space, 3),
            state_bounds=[(-0.9, 0.76), (-0.9, -0.3)],
            control_bounds=[(0.0, 4.0), (-0.4, 0.4), (0.0, 0.25)],
            dynamics_fn=self.propagator,
            propagator_fn=self.ompl_propagator,
        )
        bounds = ob.RealVectorBounds(2)
        for i, (low, high) in enumerate(self.state_bounds):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        self.state_space.setBounds(bounds)

        bounds = ob.RealVectorBounds(3)
        for i, (low, high) in enumerate(self.control_bounds):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        self.control_space.setBounds(bounds)

    def propagator(self, start: np.ndarray, control: np.ndarray, duration: float) -> np.ndarray:
        """Pushing object propagation."""
        model = get_pushing_model(self.object_shape)
        device = next(model.parameters()).device
        control_tensor = torch.tensor(
            [[float(control[0]), float(control[1]), float(control[2])]],
            dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            output = model(control_tensor)
            delta = output[0].detach().cpu().numpy()

        start_pose = SE2Pose(np.array([start[0], start[1]]), start[2])
        delta_pose = SE2Pose(np.array([delta[0], delta[1]]), delta[2])
        next_pose = start_pose @ delta_pose

        return np.array([next_pose.position[0], next_pose.position[1], next_pose.euler[2]])

    def ompl_propagator(self, start, control, duration, state):
        start_np = np.array([start.getX(), start.getY(), start.getYaw()], dtype=float)
        control_np = np.array([control[0], control[1], control[2]], dtype=float)
        result = self.propagator(start_np, control_np, duration)
        state.setX(result[0])
        state.setY(result[1])
        state.setYaw(result[2])


def system_dynamics_torch_wrapper(system_ctor, doc: str):
    """Create a torch dynamics wrapper from a System class propagator."""
    system_instance = None

    def _wrapper(start: torch.Tensor, control: torch.Tensor, *, duration: float) -> torch.Tensor:
        nonlocal system_instance
        if system_instance is None:
            system_instance = system_ctor()

        start_np = start.detach().cpu().numpy()
        control_np = control.detach().cpu().numpy()

        if start_np.ndim == 1:
            result = system_instance.propagator(start_np, control_np, duration)
        else:
            result = np.stack(
                [system_instance.propagator(s, u, duration) for s, u in zip(start_np, control_np)],
                axis=0,
            )

        return torch.tensor(result, dtype=start.dtype, device=start.device)

    _wrapper.__doc__ = doc
    return _wrapper


def get_system(system_name: str) -> System:
    if system_name == "kinematic_car":
        return kinematicCar()
    if system_name == "double_integrator":
        return doubleIntegrator()
    if system_name == "pushing_object":
        return pushingObject()
    raise ValueError(f"Unknown system: {system_name}")
