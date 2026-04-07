from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from systems import kinamticCar, doubleIntegrator, pushingObject
from utils.utils import addNoise
from sim_network import SimClient


class Simulator(ABC):
    """Base simulator interface shared by Gaussian and MuJoCo-backed simulators."""

    def __init__(self, system_name: str, config: Optional[dict] = None):
        self.system_name = system_name
        self.config = config or {}
        self.dt = float(self.config.get("propagation_step_size", 0.1))
        self.pos_std = float(self.config.get("sampling_position_std", 0.0))
        self.rot_std = float(self.config.get("sampling_rotation_std", 0.0))
        self.running = True

        default_state = [0.0, 0.0, 0.0]
        if system_name == "double_integrator":
            default_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.start_state = np.array(self.config.get("start_state", default_state), dtype=float)
        self.current_state = self.start_state.copy()

    def reset(self):
        self.current_state = self.start_state.copy()
        self.running = True
        return self.get_state()

    def stop(self):
        self.running = False
        return True

    def get_state(self):
        return self.current_state.tolist()

    def set_obj_init_pose(self, pose):
        pose_np = np.array(pose, dtype=float).reshape(-1)
        if pose_np.shape[0] < 3:
            raise ValueError(f"Expected pose with at least 3 values, got {pose}")
        self.current_state = np.array([pose_np[0], pose_np[1], pose_np[2]], dtype=float)
        self.start_state = self.current_state.copy()
        return self.get_state()

    def set_obj_init_poses(self, env_id, poses):
        if poses is None or len(poses) == 0:
            raise ValueError("poses must contain at least one pose")
        return self.set_obj_init_pose(poses[0])

    def get_sim_info(self):
        return (1, self.dt, None)

    def execute_waypoints(self, pos_waypoints):
        if pos_waypoints is None or len(pos_waypoints) == 0:
            return self.get_state()
        final_wp = np.array(pos_waypoints[-1], dtype=float).reshape(-1)
        if final_wp.shape[0] >= 3:
            self.current_state = np.array([final_wp[0], final_wp[1], final_wp[2]], dtype=float)
        elif final_wp.shape[0] >= 2:
            self.current_state[0] = final_wp[0]
            self.current_state[1] = final_wp[1]
        return self.get_state()

    @abstractmethod
    def execute_segment(self, control, duration):
        raise NotImplementedError


class KinematicCarGaussianNoise(Simulator):
    def __init__(self, config: Optional[dict] = None):
        super().__init__("simple_car", config=config)
        self.system = kinamticCar()

    def execute_segment(self, control, duration):
        control_np = np.array(control, dtype=float).reshape(-1)
        duration = float(duration)
        if duration <= 0.0:
            return self.get_state()

        n_steps = max(1, int(np.ceil(duration / self.dt)))
        dt_step = duration / n_steps
        for _ in range(n_steps):
            self.current_state = self.system.propagate(self.current_state, control_np, dt_step)

        self.current_state = np.array(
            addNoise(self.system_name, self.current_state.copy(), self.pos_std, self.rot_std),
            dtype=float,
        )
        self.running = True
        return self.get_state()


class DoubleIntegratorGaussianNoise(Simulator):
    def __init__(self, config: Optional[dict] = None):
        super().__init__("double_integrator", config=config)
        self.system = doubleIntegrator()

    def execute_segment(self, control, duration):
        control_np = np.array(control, dtype=float).reshape(-1)
        duration = float(duration)
        if duration <= 0.0:
            return self.get_state()

        n_steps = max(1, int(np.ceil(duration / self.dt)))
        dt_step = duration / n_steps
        for _ in range(n_steps):
            self.current_state = self.system.propagate(self.current_state, control_np, dt_step)

        # For double integrator, treat all dimensions as positional noise scale for now.
        noise = np.random.normal(0.0, self.pos_std, size=self.current_state.shape)
        self.current_state = (self.current_state + noise).astype(float)
        self.running = True
        return self.get_state()


class PushingObjectGaussianNoise(Simulator):
    def __init__(self, config: Optional[dict] = None):
        super().__init__("pushing", config=config)
        object_name = self.config.get("objectName", "crackerBox")
        self.system = pushingObject(object_name=object_name)

    def execute_segment(self, control, duration):
        control_np = np.array(control, dtype=float).reshape(-1)
        duration = float(duration)
        if duration <= 0.0:
            return self.get_state()

        n_steps = max(1, int(np.ceil(duration / self.dt)))
        dt_step = duration / n_steps
        for _ in range(n_steps):
            self.current_state = self.system.propagate(self.current_state, control_np, dt_step)

        self.current_state = np.array(
            addNoise(self.system_name, self.current_state.copy(), self.pos_std, self.rot_std),
            dtype=float,
        )
        self.running = True
        return self.get_state()


class KinematicCarMujoco(Simulator):
    def __init__(self, config: Optional[dict] = None, client: Optional[SimClient] = None):
        super().__init__("simple_car", config=config)
        self.client = client if client is not None else SimClient()

    def reset(self):
        return self.client.execute("reset")

    def stop(self):
        return self.client.execute("stop")

    def get_state(self):
        return self.client.execute("get_state")

    def execute_segment(self, control, duration):
        return self.client.execute("execute_segment", [control, duration])


class PushingObjectMujoco(Simulator):
    def __init__(self, config: Optional[dict] = None, client: Optional[SimClient] = None):
        super().__init__("pushing", config=config)
        self.client = client if client is not None else SimClient()

    def reset(self):
        return self.client.execute("reset")

    def stop(self):
        return self.client.execute("stop")

    def get_state(self):
        return self.client.execute("get_state")

    def set_obj_init_pose(self, pose):
        return self.client.execute("set_obj_init_pose", [pose])

    def set_obj_init_poses(self, env_id, poses):
        return self.client.execute("set_obj_init_poses", [env_id, poses])

    def execute_segment(self, control, duration):
        return self.client.execute("execute_segment", [control, duration])


def create_simulator(
    system_name: str,
    mode: str,
    config: Optional[dict] = None,
    client: Optional[SimClient] = None,
) -> Simulator:
    """
    Factory selector by system name + backend mode.
    mode: "gaussian" or "mujoco"
    """
    mode = mode.lower()
    system_name = system_name.lower()

    if mode == "gaussian":
        if system_name == "simple_car":
            return KinematicCarGaussianNoise(config=config)
        if system_name == "double_integrator":
            return DoubleIntegratorGaussianNoise(config=config)
        if system_name == "pushing":
            return PushingObjectGaussianNoise(config=config)
    elif mode == "mujoco":
        if system_name == "simple_car":
            return KinematicCarMujoco(config=config, client=client)
        if system_name == "pushing":
            return PushingObjectMujoco(config=config, client=client)

    raise ValueError(f"Unsupported simulator combination: system={system_name}, mode={mode}")
