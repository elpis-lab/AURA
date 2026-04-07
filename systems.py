from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from ompl import base as ob
from ompl import control as oc

from factories import pickObjectShape
from plan import plan as run_plan
from planning.propagators import (
    pushingDynamics,
    pushingDynamicsTorch,
)


@dataclass
class System:
    """Base system definition for planning/execution."""

    name: str
    state_dim: int
    control_dim: int
    state_bounds: list[tuple[float, float]]
    control_bounds: list[tuple[float, float]]
    dynamics_fn: Callable
    propagator_fn: Callable
    goal_threshold: float = 0.1

    def create_spaces(self) -> Tuple[ob.StateSpace, oc.ControlSpace]:
        """Create OMPL state/control spaces with configured bounds."""
        space = self._create_state_space()
        cspace = oc.RealVectorControlSpace(space, self.control_dim)
        cbounds = ob.RealVectorBounds(self.control_dim)
        for i, (low, high) in enumerate(self.control_bounds):
            cbounds.setLow(i, float(low))
            cbounds.setHigh(i, float(high))
        cspace.setBounds(cbounds)
        return space, cspace

    def _create_state_space(self) -> ob.StateSpace:
        raise NotImplementedError

    def propagate(self, state: np.ndarray, control: np.ndarray, duration: float) -> np.ndarray:
        """Numpy propagation helper."""
        return self.dynamics_fn(state, control, duration)


class kinamticCar(System):
    """SE2 kinematic car system."""

    def __init__(self):
        super().__init__(
            name="simple_car",
            state_dim=3,
            control_dim=2,
            state_bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            control_bounds=[(0.2, 0.95), (-0.30, 0.30)],
            dynamics_fn=self.dynamics,
            propagator_fn=self.ompl_propagator,
            goal_threshold=0.1,
        )
        self._wheelbase = 0.1385 + 0.158

    def _create_state_space(self) -> ob.StateSpace:
        space = ob.SE2StateSpace()
        bounds = ob.RealVectorBounds(2)
        for i, (low, high) in enumerate(self.state_bounds):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        space.setBounds(bounds)
        return space

    def dynamics(self, start: np.ndarray, control: np.ndarray, duration: float) -> np.ndarray:
        """Kinematic car propagation (same formulation as planning/propagators.py)."""
        x, y, yaw = start
        u_vel, u_phi = control
        x_dot = u_vel * np.cos(yaw)
        y_dot = u_vel * np.sin(yaw)
        yaw_dot = (u_vel / self._wheelbase) * np.tan(u_phi)
        x_state = x + x_dot * duration
        y_state = y + y_dot * duration
        yaw_state = (yaw + yaw_dot * duration + np.pi) % (2 * np.pi) - np.pi
        return np.array([x_state, y_state, yaw_state], dtype=float)

    def ompl_propagator(self, start, control, duration, state):
        start_np = np.array([start.getX(), start.getY(), start.getYaw()], dtype=float)
        control_np = np.array([control[0], control[1]], dtype=float)
        result = self.dynamics(start_np, control_np, duration)
        state.setX(result[0])
        state.setY(result[1])
        state.setYaw(result[2])


class doubleIntegrator(System):
    """3D double integrator in R^6 with acceleration control in R^3."""

    def __init__(self):
        super().__init__(
            name="double_integrator",
            state_dim=6,
            control_dim=3,
            state_bounds=[
                (-3.0, 3.0),
                (-3.0, 3.0),
                (0.0, 3.0),
                (-0.5, 0.5),
                (-0.5, 0.5),
                (-0.5, 0.5),
            ],
            control_bounds=[(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)],
            dynamics_fn=self.dynamics,
            propagator_fn=self.ompl_propagator,
            goal_threshold=0.1,
        )

    def _create_state_space(self) -> ob.StateSpace:
        space = ob.RealVectorStateSpace(self.state_dim)
        bounds = ob.RealVectorBounds(self.state_dim)
        for i, (low, high) in enumerate(self.state_bounds):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        space.setBounds(bounds)
        return space

    def dynamics(self, start: np.ndarray, control: np.ndarray, duration: float) -> np.ndarray:
        """3D double-integrator propagation (same formulation as planning/propagators.py)."""
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
        result = self.dynamics(start_np, control_np, duration)
        for i in range(6):
            state[i] = result[i]


class pushingObject(System):
    """SE2 pushing system with learned-model dynamics."""

    def __init__(self, object_name: str = "crackerBox"):
        self.object_name = object_name
        self.object_shape = pickObjectShape(object_name)

        super().__init__(
            name="pushing",
            state_dim=3,
            control_dim=3,
            state_bounds=[(-0.9, 0.76), (-0.9, -0.3)],
            control_bounds=[(0.0, 4.0), (-0.4, 0.4), (0.0, 0.25)],
            dynamics_fn=self.dynamics,
            propagator_fn=self.ompl_propagator,
            goal_threshold=0.1,
        )

    def _create_state_space(self) -> ob.StateSpace:
        space = ob.SE2StateSpace()
        bounds = ob.RealVectorBounds(2)
        for i, (low, high) in enumerate(self.state_bounds):
            bounds.setLow(i, float(low))
            bounds.setHigh(i, float(high))
        space.setBounds(bounds)
        return space

    def dynamics(self, start: np.ndarray, control: np.ndarray, duration: float) -> np.ndarray:
        # Learned-model propagation uses object shape.
        return pushingDynamics(start, control, duration, object_shape=self.object_shape)

    def ompl_propagator(self, start, control, duration, state):
        start_np = np.array([start.getX(), start.getY(), start.getYaw()], dtype=float)
        control_np = np.array([control[0], control[1], control[2]], dtype=float)
        result = self.dynamics(start_np, control_np, duration)
        state.setX(result[0])
        state.setY(result[1])
        state.setYaw(result[2])


def kinamticCarDynamicsTorch(
    start: torch.Tensor, control: torch.Tensor, *, duration: float
) -> torch.Tensor:
    """Torch wrapper for kinematic car dynamics (outside class by design)."""
    wheelbase = torch.tensor(0.1385 + 0.158, dtype=start.dtype, device=start.device)
    x = start[:, 0]
    y = start[:, 1]
    yaw = start[:, 2]
    v = control[:, 0]
    phi = control[:, 1]
    new_x = x + v * torch.cos(yaw) * duration
    new_y = y + v * torch.sin(yaw) * duration
    angle = yaw + (v / wheelbase) * torch.tan(phi) * duration
    new_yaw = (angle + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.stack([new_x, new_y, new_yaw], dim=1)


def doubleIntegratorDynamicsTorch(
    start: torch.Tensor, control: torch.Tensor, *, duration: float
) -> torch.Tensor:
    """Torch wrapper for 3D double-integrator dynamics (outside class by design)."""
    x = start[:, 0]
    y = start[:, 1]
    z = start[:, 2]
    vx = start[:, 3]
    vy = start[:, 4]
    vz = start[:, 5]
    ax = control[:, 0]
    ay = control[:, 1]
    az = control[:, 2]
    dt = torch.as_tensor(duration, dtype=start.dtype, device=start.device)
    half_dt2 = 0.5 * dt * dt
    x_new = x + vx * dt + ax * half_dt2
    y_new = y + vy * dt + ay * half_dt2
    z_new = z + vz * dt + az * half_dt2
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    vz_new = vz + az * dt
    return torch.stack([x_new, y_new, z_new, vx_new, vy_new, vz_new], dim=1)


def pushingObjectDynamicsTorch(
    start: torch.Tensor, control: torch.Tensor, *, duration: float, object_name: str = "crackerBox"
) -> torch.Tensor:
    """Torch wrapper for pushing learned dynamics (outside class by design)."""
    object_shape = pickObjectShape(object_name)
    return pushingDynamicsTorch(start, control, duration=duration, object_shape=object_shape)


def get_system(system_name: str, object_name: str = "crackerBox") -> System:
    if system_name == "simple_car":
        return kinamticCar()
    if system_name == "double_integrator":
        return doubleIntegrator()
    if system_name == "pushing":
        return pushingObject(object_name=object_name)
    raise ValueError(f"Unknown system: {system_name}")


@dataclass
class plan:
    """Planning wrapper that owns all planning settings for plan/replan."""

    system: System
    start_state: np.ndarray
    goal_state: np.ndarray
    planner_name: str = "aorrt"
    goal_threshold: float = 0.1
    min_control_duration: int = 1
    max_control_duration: int = 5
    propagation_step_size: float = 1.0
    planning_time: float = 10.0
    pruning_radius: float = 0.1
    config: Optional[dict] = None
    visualize: bool = False

    def __post_init__(self):
        self.start_state = np.asarray(self.start_state, dtype=float)
        self.goal_state = np.asarray(self.goal_state, dtype=float)
        self.latest_solutions = None
        self.latest_ss = None

    def plan(self):
        solutions, ss = run_plan(
            system=self.system.name,
            startState=self.start_state,
            goalState=self.goal_state,
            goalThreshold=float(self.goal_threshold),
            propagator=self.system.propagator_fn,
            minControlDuration=int(self.min_control_duration),
            maxControlDuration=int(self.max_control_duration),
            propagationStepSize=float(self.propagation_step_size),
            planningTime=float(self.planning_time),
            plannerName=self.planner_name,
            pruningRadius=float(self.pruning_radius),
            config=self.config,
            visualize=self.visualize,
        )
        self.latest_solutions = solutions
        self.latest_ss = ss
        return solutions, ss

    def replan(self, new_start_state: np.ndarray, planning_time: Optional[float] = None):
        self.start_state = np.asarray(new_start_state, dtype=float)
        if planning_time is not None:
            self.planning_time = float(planning_time)
        return self.plan()
