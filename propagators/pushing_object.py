"""Learned SE(2) pushing-object propagation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from ompl import base as ob
from ompl import control as oc

from propagators.propagator import System, wrap_angle_numpy, wrap_angle_torch
from simulation.pushing_model import CRACKER_BOX_FLIPPED_SHAPE, get_pushing_model


class PushingObject(System):
    """Learned SE(2) pushing system used by OMPL and AURA."""

    def __init__(self):
        self.object_shape = CRACKER_BOX_FLIPPED_SHAPE.copy()
        self.model_name = "cracker_box_flipped"
        self.model_path = None
        state_space = ob.SE2StateSpace()
        super().__init__(
            name="pushing_object",
            state_space=state_space,
            control_space=oc.RealVectorControlSpace(state_space, 3),
            state_bounds=[(-0.9, 0.76), (-0.9, -0.3)],
            control_bounds=[(0.0, 0.75), (-0.4, 0.4), (0.0, 0.25)],
            dynamics_fn=self.propagator,
            propagator_fn=self.ompl_propagator,
        )
        self.set_state_bounds(self.state_bounds)
        self.set_control_bounds(self.control_bounds)

    def set_state_bounds(self, bounds_values) -> None:
        bounds_list = [(float(low), float(high)) for low, high in bounds_values]
        if len(bounds_list) != 2:
            raise ValueError("pushing_object requires two position bounds")
        self.state_bounds = bounds_list
        bounds = ob.RealVectorBounds(2)
        for index, (low, high) in enumerate(bounds_list):
            bounds.setLow(index, low)
            bounds.setHigh(index, high)
        self.state_space.setBounds(bounds)

    def set_ompl_state(self, state, values: np.ndarray):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size < 3:
            raise ValueError("pushing_object state must contain x, y, and yaw")
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

    def canonical_control(self, control: np.ndarray) -> np.ndarray:
        command = np.asarray(control, dtype=float).reshape(-1).copy()
        if command.size < 3:
            raise ValueError(
                f"pushing_object control must contain three values, got {command}"
            )

        face_raw = float(command[0])
        normalized_faces = np.array([0.0, 0.25, 0.5, 0.75], dtype=float)
        radian_faces = np.array(
            [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0],
            dtype=float,
        )
        if 0.0 <= face_raw <= 0.75:
            face_index = int(np.round(face_raw * 4.0)) % 4
        elif abs(face_raw - round(face_raw)) < 1e-9 and 0 <= round(face_raw) <= 3:
            face_index = int(round(face_raw)) % 4
        elif np.min(np.abs(face_raw - radian_faces)) < 1e-6:
            face_index = int(np.argmin(np.abs(face_raw - radian_faces)))
        elif 0.0 <= face_raw < 4.0:
            face_index = int(face_raw) % 4
        else:
            face_index = int(face_raw / (np.pi / 2.0)) % 4

        command[0] = normalized_faces[face_index]
        command[1] = float(np.clip(command[1], -0.4, 0.4))
        command[2] = float(
            np.clip(command[2], 0.0, self.control_bounds[2][1])
        )
        return command[:3]

    def propagator(
        self,
        start: np.ndarray,
        control: np.ndarray,
        duration: float,
    ) -> np.ndarray:
        command = self.canonical_control(control)
        if self.propagation_step_size is None:
            raise RuntimeError(
                "pushing_object requires a configured propagation step size"
            )
        duration_steps = self.propagation_step_count(duration)
        model = get_pushing_model(
            self.object_shape,
            model_name=self.model_name,
            model_path=self.model_path,
        )
        device = next(model.parameters()).device

        def predict_one_step(control_batch: np.ndarray) -> np.ndarray:
            control_tensor = torch.as_tensor(
                control_batch,
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                return model(control_tensor)[:, :3].detach().cpu().numpy()

        return propagate_numpy(
            start,
            command,
            duration_steps,
            predict_one_step,
        )

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
            raise ValueError("pushing_object expects three state values")
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
        control_array = np.array(
            [control[0], control[1], control[2]],
            dtype=float,
        )
        result = self.propagator(start_array, control_array, duration)
        state.setX(result[0])
        state.setY(result[1])
        state.setYaw(result[2])


def compose_numpy(pose: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """Apply a body-frame relative pose to an SE(2) pose batch."""
    pose_array = np.asarray(pose, dtype=float)
    relative_array = np.asarray(relative, dtype=float)
    single = pose_array.ndim == 1
    pose_array = np.atleast_2d(pose_array)
    relative_array = np.atleast_2d(relative_array)
    if len(relative_array) == 1 and len(pose_array) > 1:
        relative_array = np.repeat(relative_array, len(pose_array), axis=0)
    if pose_array.shape[1] != 3 or relative_array.shape != pose_array.shape:
        raise ValueError("pose and relative pose batches must have matching (N,3) shapes")
    yaw = pose_array[:, 2]
    cosine, sine = np.cos(yaw), np.sin(yaw)
    result = np.stack(
        [
            pose_array[:, 0]
            + cosine * relative_array[:, 0]
            - sine * relative_array[:, 1],
            pose_array[:, 1]
            + sine * relative_array[:, 0]
            + cosine * relative_array[:, 1],
            wrap_angle_numpy(yaw + relative_array[:, 2]),
        ],
        axis=1,
    )
    return result[0] if single else result


def compose_torch(pose: torch.Tensor, relative: torch.Tensor) -> torch.Tensor:
    """Differentiable counterpart of :func:`compose_numpy`."""
    if pose.ndim != 2 or relative.ndim != 2 or pose.shape != relative.shape:
        raise ValueError("pose and relative pose tensors must have matching (N,3) shapes")
    if pose.shape[1] != 3:
        raise ValueError("pushing poses must contain three values")
    yaw = pose[:, 2]
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        [
            pose[:, 0] + cosine * relative[:, 0] - sine * relative[:, 1],
            pose[:, 1] + sine * relative[:, 0] + cosine * relative[:, 1],
            wrap_angle_torch(yaw + relative[:, 2]),
        ],
        dim=1,
    )


def propagate_numpy(
    start: np.ndarray,
    control: np.ndarray,
    duration_steps: int,
    model: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply learned relative-pose transitions for a fixed number of steps."""
    steps = int(duration_steps)
    if steps < 1:
        raise ValueError(f"duration_steps must be at least 1, got {steps}")
    start_array = np.asarray(start, dtype=float)
    single = start_array.ndim == 1
    pose = np.atleast_2d(start_array).copy()
    if pose.ndim != 2 or pose.shape[1] != 3:
        raise ValueError("pushing states must have shape (3,) or (N,3)")
    command = np.atleast_2d(np.asarray(control, dtype=float))
    if len(command) == 1 and len(pose) > 1:
        command = np.repeat(command, len(pose), axis=0)
    if len(command) != len(pose):
        raise ValueError("pushing state and control batches must align")
    for _ in range(steps):
        relative = np.asarray(model(command), dtype=float).reshape(-1, 3)
        if len(relative) != len(pose):
            raise ValueError("pushing model output batch does not match the state batch")
        pose = compose_numpy(pose, relative)
    return pose[0] if single else pose


def propagate_torch(
    start: torch.Tensor,
    control: torch.Tensor,
    duration_steps,
    model: torch.nn.Module,
) -> torch.Tensor:
    """Differentiably unroll learned transitions with per-row durations."""
    if start.ndim != 2 or control.ndim != 2 or len(start) != len(control):
        raise ValueError("pushing tensors must be aligned 2-D batches")
    if start.shape[1] != 3:
        raise ValueError("pushing state tensors must have shape (N,3)")
    if isinstance(duration_steps, torch.Tensor):
        steps = duration_steps.to(device=start.device, dtype=torch.long).reshape(-1)
    else:
        steps = torch.as_tensor(
            duration_steps, device=start.device, dtype=torch.long
        ).reshape(-1)
    if steps.numel() == 1 and start.shape[0] > 1:
        steps = steps.repeat(start.shape[0])
    if steps.numel() != start.shape[0] or bool(torch.any(steps < 1)):
        raise ValueError("duration_steps must contain one positive integer per state")

    pose = start
    for step_index in range(int(steps.max().detach().cpu())):
        candidate = compose_torch(pose, model(control)[:, :3])
        pose = torch.where((steps > step_index).unsqueeze(1), candidate, pose)
    return pose
