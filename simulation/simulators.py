from __future__ import annotations

import argparse
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from systems import kinematicCar, doubleIntegrator, pushingObject
from utils.utils import addNoise

try:
    from simulation.mujoco_car import Sim as MujocoCarSimulator
except ImportError:
    MujocoCarSimulator = None

try:
    from simulation.mujoco_pushing import Sim as MujocoPushingSimulator
except ImportError:
    MujocoPushingSimulator = None


class Simulator(ABC):
    """Base simulator interface shared by Gaussian and MuJoCo-backed simulators."""

    def __init__(self, system_name: str, config: Optional[dict] = None):
        self.system_name = system_name
        self.config = config or {}
        self.dt = float(self.config.get("propagation_step_size", 0.1))
        self.pos_std = float(self.config.get("sampling_position_std", 0.0))
        self.rot_std = float(self.config.get("sampling_rotation_std", 0.0))
        self.vel_std = float(self.config.get("sampling_velocity_std", self.pos_std))
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
        expected_dim = int(self.start_state.shape[0])
        if pose_np.shape[0] < expected_dim:
            raise ValueError(f"Expected pose with at least {expected_dim} values, got {pose}")
        self.current_state = np.array(pose_np[:expected_dim], dtype=float)
        self.start_state = self.current_state.copy()
        return self.get_state()

    def set_object_init_pose(self, pose):
        """Compatibility alias used by experiment scripts."""
        return self.set_obj_init_pose(pose)

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
        super().__init__("kinematic_car", config=config)
        self.system = kinematicCar()

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

        vel_noise = np.random.normal(0.0, self.vel_std, size=3)
        self.current_state[3:6] = self.current_state[3:6] + vel_noise
        self.current_state = self.current_state.astype(float)
        self.running = True
        return self.get_state()


class PushingObjectGaussianNoise(Simulator):
    def __init__(self, config: Optional[dict] = None):
        super().__init__("pushing_object", config=config)
        self.system = pushingObject()

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
    """Local in-process simulator implementation for Mujoco mode."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__("kinematic_car", config=config)
        if MujocoCarSimulator is None:
            raise ImportError("mujoco is required for kinematic_car mujoco simulation mode")
        self.simulator = MujocoCarSimulator()
        self.simulator.throttle_ctrl_scale = float(
            self.config.get("mujoco_car_throttle_ctrl_scale", self.simulator.throttle_ctrl_scale)
        )
        self.simulator.steering_ctrl_scale = float(
            self.config.get("mujoco_car_steering_ctrl_scale", self.simulator.steering_ctrl_scale)
        )
        self._viewer_thread = None
        self._start_viewer_thread()

    def _start_viewer_thread(self):
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            return
        self._viewer_thread = threading.Thread(
            target=self.simulator.run_viewer,
            daemon=True,
        )
        self._viewer_thread.start()
        # Let passive viewer loop initialize before queueing controls.
        time.sleep(0.5)

    def reset(self):
        self._start_viewer_thread()
        self.simulator.reset()
        self.current_state = np.asarray(self.simulator.get_state(), dtype=float).reshape(-1)[:3]
        self.running = True
        return self.get_state()

    def execute_segment(self, control, duration):
        control_np = np.array(control, dtype=float).reshape(-1)
        duration = float(duration)
        if duration <= 0.0:
            return self.get_state()
        self._start_viewer_thread()
        self.simulator.execute_segment(control_np, duration)
        self.current_state = np.asarray(self.simulator.get_state(), dtype=float).reshape(-1)[:3]
        self.running = True
        return self.get_state()

    def close(self):
        self.running = False
        try:
            self.simulator.close()
        finally:
            if self._viewer_thread is not None and self._viewer_thread.is_alive():
                self._viewer_thread.join(timeout=5.0)


class PushingObjectMujoco(Simulator):
    """Local in-process simulator implementation for Mujoco mode."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__("pushing_object", config=config)
        if MujocoPushingSimulator is None:
            raise ImportError("mujoco is required for pushing_object mujoco simulation mode")
        self.simulator = MujocoPushingSimulator()
        self._viewer_thread = None
        self._start_viewer_thread()

    def _start_viewer_thread(self):
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            return
        self._viewer_thread = threading.Thread(
            target=self.simulator.run_viewer,
            daemon=True,
        )
        self._viewer_thread.start()
        # Give the passive viewer loop a brief moment to initialize before queueing controls.
        time.sleep(0.5)

    def reset(self):
        self._start_viewer_thread()
        self.simulator.reset()
        self.running = True
        # Keep wrapper and underlying simulator state aligned after reset.
        return self.set_obj_init_pose(self.start_state.tolist())

    def set_obj_init_pose(self, pose):
        state = super().set_obj_init_pose(pose)
        self._start_viewer_thread()
        self.simulator.set_obj_init_pose(np.asarray(state, dtype=float))
        return self.get_state()

    def execute_segment(self, control, duration):
        control_np = np.array(control, dtype=float).reshape(-1)
        duration = float(duration)
        if duration <= 0.0:
            return self.get_state()
        self._start_viewer_thread()
        result = self.simulator.execute_segment(control_np, duration)
        result_np = np.asarray(result, dtype=float)
        if result_np.ndim > 1:
            result_np = result_np[0]
        self.current_state = result_np
        self.running = True
        return self.get_state()

    def close(self):
        self.running = False
        try:
            self.simulator.close()
        finally:
            if self._viewer_thread is not None and self._viewer_thread.is_alive():
                self._viewer_thread.join(timeout=5.0)


def create_simulator(
    system_name: str,
    mode: str,
    config: Optional[dict] = None,
    client: Optional[object] = None,
) -> Simulator:
    """
    Factory selector by system name + backend mode.
    mode: "gaussian" or "mujoco"
    """
    mode = mode.lower()
    system_name = system_name.lower()
    system_name = {
        "simple_car": "kinematic_car",
        "car": "kinematic_car",
        "pushing": "pushing_object",
        "push": "pushing_object",
    }.get(system_name, system_name)

    if mode == "gaussian":
        if system_name == "kinematic_car":
            return KinematicCarGaussianNoise(config=config)
        if system_name == "double_integrator":
            return DoubleIntegratorGaussianNoise(config=config)
        if system_name == "pushing_object":
            return PushingObjectGaussianNoise(config=config)
    elif mode == "mujoco":
        if system_name == "kinematic_car":
            return KinematicCarMujoco(config=config)
        if system_name == "pushing_object":
            return PushingObjectMujoco(config=config)

    raise ValueError(f"Unsupported simulator combination: system={system_name}, mode={mode}")


def main():
    parser = argparse.ArgumentParser(description="Open a simulator.")
    parser.add_argument(
        "system_name",
        choices=["kinematic_car", "double_integrator", "pushing_object"],
        help="System name.",
    )
    parser.add_argument("mode", choices=["gaussian", "mujoco"], help="Simulator mode.")

    args = parser.parse_args()
    simulator = create_simulator(args.system_name, args.mode, config={})

    simulator.reset()
    print(f"[INFO] Opened simulator: system={args.system_name}, mode={args.mode}")
    print(f"[INFO] Current state: {simulator.get_state()}")
    if args.mode == "mujoco":
        print("[INFO] Launching MuJoCo viewer... Press Ctrl+C to exit.")
        try:
            simulator.simulator.run_viewer()
        except KeyboardInterrupt:
            print("[INFO] MuJoCo viewer stopped.")


if __name__ == "__main__":
    main()
