from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from propagators import (
    DoubleIntegrator,
    DubinsAirplane,
    KinematicCar,
    PushingObject,
)
from utils.control_duration import duration_seconds_to_steps


class Simulator(ABC):
    """Base simulator interface shared by Gaussian and MuJoCo-backed simulators."""

    def __init__(self, system_name: str, config: dict | None = None):
        self.system_name = system_name
        self.config = config or {}
        self.dt = float(self.config.get("propagation_step_size", 0.1))
        self.pos_std = float(self.config.get("sampling_position_std", 0.0))
        self.rot_std = float(self.config.get("sampling_rotation_std", 0.0))
        self.vel_std = float(self.config.get("sampling_velocity_std", self.pos_std))
        self.running = True
        self._disturbance_seed = int(
            self.config.get("disturbance_seed", self.config.get("seed", 0))
        )
        self._disturbance_schedule = self.config.get("disturbance_schedule")
        self._disturbance_index = 0
        self._rng = np.random.default_rng(self._disturbance_seed)

        default_state = [0.0, 0.0, 0.0]
        if system_name == "double_integrator":
            default_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif system_name == "dubins_airplane":
            default_state = [0.2, 0.2, 0.2, 0.0, 0.0, 0.15]
        self.start_state = np.array(
            self.config.get("start_state", default_state), dtype=float
        )
        self.current_state = self.start_state.copy()
        self.primitive_states = [self.current_state.copy()]

    def reset(self):
        self.current_state = self.start_state.copy()
        self.primitive_states = [self.current_state.copy()]
        self._disturbance_index = 0
        self._rng = np.random.default_rng(self._disturbance_seed)
        self.running = True
        return self.get_state()

    def stop(self):
        self.running = False
        return True

    def get_state(self):
        return self.current_state.tolist()

    def set_state(self, state):
        pose_np = np.array(state, dtype=float).reshape(-1)
        expected_dim = int(self.start_state.shape[0])
        if pose_np.shape[0] < expected_dim:
            raise ValueError(
                f"Expected state with at least {expected_dim} values, got {state}"
            )
        self.current_state = np.array(pose_np[:expected_dim], dtype=float)
        self.start_state = self.current_state.copy()
        return self.get_state()

    def _duration_steps(self, duration: float) -> int:
        return duration_seconds_to_steps(
            duration,
            self.dt,
            min_steps=int(self.config.get("min_control_duration", 1)),
            max_steps=(
                int(self.config["max_control_duration"])
                if "max_control_duration" in self.config
                else None
            ),
        )

    def _next_standard_disturbance(self, dimension: int) -> np.ndarray:
        if self._disturbance_schedule is None:
            value = self._rng.standard_normal(int(dimension))
        else:
            if self._disturbance_index >= len(self._disturbance_schedule):
                raise RuntimeError(
                    "disturbance schedule exhausted at primitive step "
                    f"{self._disturbance_index}"
                )
            value = np.asarray(
                self._disturbance_schedule[self._disturbance_index], dtype=float
            ).reshape(-1)
            if len(value) < int(dimension):
                raise ValueError(
                    f"disturbance row {self._disturbance_index} has {len(value)} "
                    f"values; expected at least {dimension}"
                )
            value = value[: int(dimension)]
        self._disturbance_index += 1
        return np.asarray(value, dtype=float)

    def _record_primitive_state(self) -> None:
        self.primitive_states.append(self.current_state.copy())

    @abstractmethod
    def execute_segment(self, control, duration):
        raise NotImplementedError


class KinematicCarGaussianNoise(Simulator):
    def __init__(self, config: dict | None = None):
        super().__init__("kinematic_car", config=config)
        self.system = KinematicCar()
        self.system.configure_duration_contract(
            self.dt,
            int(self.config.get("min_control_duration", 1)),
            int(self.config.get("max_control_duration", 1)),
        )

    def execute_segment(self, control, duration):
        control_np = np.array(control, dtype=float).reshape(-1)
        duration = float(duration)
        if duration <= 0.0:
            return self.get_state()

        for _ in range(self._duration_steps(duration)):
            self.current_state = self.system.propagate(
                self.current_state, control_np, self.dt
            )
            disturbance = self._next_standard_disturbance(3)
            self.current_state[:2] += disturbance[:2] * self.pos_std
            self.current_state[2] += disturbance[2] * self.rot_std
            self.current_state[2] = (
                self.current_state[2] + np.pi
            ) % (2.0 * np.pi) - np.pi
            self._record_primitive_state()
        self.running = True
        return self.get_state()


class DoubleIntegratorGaussianNoise(Simulator):
    def __init__(self, config: dict | None = None):
        super().__init__("double_integrator", config=config)
        self.system = DoubleIntegrator()
        self.system.configure_duration_contract(
            self.dt,
            int(self.config.get("min_control_duration", 1)),
            int(self.config.get("max_control_duration", 1)),
        )

    def execute_segment(self, control, duration):
        control_np = np.array(control, dtype=float).reshape(-1)
        duration = float(duration)
        if duration <= 0.0:
            return self.get_state()

        for _ in range(self._duration_steps(duration)):
            self.current_state = self.system.propagate(
                self.current_state, control_np, self.dt
            )
            self.current_state[3:6] += (
                self._next_standard_disturbance(3) * self.vel_std
            )
            self._record_primitive_state()
        self.current_state = self.current_state.astype(float)
        self.running = True
        return self.get_state()


class PushingObjectGaussianNoise(Simulator):
    def __init__(self, config: dict | None = None):
        super().__init__("pushing_object", config=config)
        self.system = PushingObject()
        self.system.configure_duration_contract(
            self.dt,
            int(self.config.get("min_control_duration", 1)),
            int(self.config.get("max_control_duration", 1)),
        )

    def execute_segment(self, control, duration):
        control_np = np.array(control, dtype=float).reshape(-1)
        duration = float(duration)
        if duration <= 0.0:
            return self.get_state()

        for _ in range(self._duration_steps(duration)):
            self.current_state = self.system.propagate(
                self.current_state, control_np, self.dt
            )
            disturbance = self._next_standard_disturbance(3)
            self.current_state[:2] += disturbance[:2] * self.pos_std
            self.current_state[2] += disturbance[2] * self.rot_std
            self.current_state[2] = (
                self.current_state[2] + np.pi
            ) % (2.0 * np.pi) - np.pi
            self._record_primitive_state()
        self.running = True
        return self.get_state()


class DubinsAirplaneGaussianNoise(Simulator):
    def __init__(self, config: dict | None = None):
        super().__init__("dubins_airplane", config=config)
        self.system = DubinsAirplane()
        self.system.configure_duration_contract(
            self.dt,
            int(self.config.get("min_control_duration", 1)),
            int(self.config.get("max_control_duration", 1)),
        )

    def execute_segment(self, control, duration):
        control_np = np.asarray(control, dtype=float).reshape(-1)
        control_np = np.clip(
            control_np,
            np.asarray([bound[0] for bound in self.system.control_bounds]),
            np.asarray([bound[1] for bound in self.system.control_bounds]),
        )
        duration = float(duration)
        if duration <= 0.0:
            return self.get_state()
        for _ in range(self._duration_steps(duration)):
            self.current_state = self.system.propagate(
                self.current_state, control_np, self.dt
            )
            disturbance = self._next_standard_disturbance(6)
            self.current_state[:3] += disturbance[:3] * self.pos_std
            self.current_state[3:5] += disturbance[3:5] * self.rot_std
            self.current_state[5] += disturbance[5] * self.vel_std
            self.current_state[3] = (
                self.current_state[3] + np.pi
            ) % (2.0 * np.pi) - np.pi
            state_low = np.asarray([bound[0] for bound in self.system.state_bounds])
            state_high = np.asarray([bound[1] for bound in self.system.state_bounds])
            self.current_state = np.clip(self.current_state, state_low, state_high)
            self._record_primitive_state()
        self.running = True
        return self.get_state()


class KinematicCarMujoco(Simulator):
    """Local in-process simulator implementation for Mujoco mode."""

    def __init__(self, config: dict | None = None):
        super().__init__("kinematic_car", config=config)
        from simulation.mujoco_car import MujocoCarSimulator

        self.simulator = MujocoCarSimulator()
        self.simulator.throttle_ctrl_scale = float(
            self.config.get(
                "mujoco_car_throttle_ctrl_scale",
                self.simulator.throttle_ctrl_scale,
            )
        )
        self.simulator.steering_ctrl_scale = float(
            self.config.get(
                "mujoco_car_steering_ctrl_scale",
                self.simulator.steering_ctrl_scale,
            )
        )
        self.headless = bool(self.config.get("headless", False))
        self._viewer_thread = None
        if not self.headless:
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
        super().reset()
        if self.headless:
            self.simulator.reset_immediately()
        else:
            self._start_viewer_thread()
            self.simulator.reset()
        self.running = True
        return self.set_state(self.start_state.tolist())

    def set_state(self, state):
        state = super().set_state(state)
        actual = self.simulator.set_state(np.asarray(state, dtype=float))
        self.current_state = np.asarray(actual, dtype=float).reshape(-1)[:3]
        self.start_state = self.current_state.copy()
        return self.get_state()

    def execute_segment(self, control, duration):
        control_np = np.array(control, dtype=float).reshape(-1)
        duration = float(duration)
        if duration <= 0.0:
            return self.get_state()
        if not self.headless:
            self._start_viewer_thread()
        for _ in range(self._duration_steps(duration)):
            if self.headless:
                segment_end = float(self.simulator.d.time) + self.dt
                while float(self.simulator.d.time) < segment_end:
                    with self.simulator.sim_lock:
                        self.simulator.d.ctrl[self.simulator.steering_id] = (
                            self.simulator.steering_ctrl_scale * control_np[1]
                        )
                        wheel_omega = control_np[0] / self.simulator.r
                        self.simulator.d.ctrl[self.simulator.throttle_id] = (
                            wheel_omega * self.simulator.throttle_ctrl_scale
                        )
                    self.simulator.step()
                self.simulator.stop()
            else:
                self.simulator.execute_segment(control_np, self.dt)
            self.current_state = np.asarray(
                self.simulator.get_state(), dtype=float
            ).reshape(-1)[:3]
            self._record_primitive_state()
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

    def __init__(self, config: dict | None = None):
        super().__init__("pushing_object", config=config)
        from simulation.mujoco_pushing import MujocoPushingSimulator

        self.simulator = MujocoPushingSimulator()
        self.headless = bool(self.config.get("headless", False))
        self._viewer_thread = None
        if not self.headless:
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
        super().reset()
        if not self.headless:
            self._start_viewer_thread()
        self.simulator.reset(wait_time=0.0 if self.headless else 0.5)
        self.running = True
        # Keep wrapper and underlying simulator state aligned after reset.
        return self.set_state(self.start_state.tolist())

    def set_state(self, state):
        state = super().set_state(state)
        if not self.headless:
            self._start_viewer_thread()
        self.simulator.set_state(np.asarray(state, dtype=float))
        return self.get_state()

    def execute_segment(self, control, duration):
        control_np = np.array(control, dtype=float).reshape(-1)
        duration = float(duration)
        if duration <= 0.0:
            return self.get_state()
        if not self.headless:
            self._start_viewer_thread()
        for _ in range(self._duration_steps(duration)):
            if self.headless:
                ws_path = self.simulator.generate_ws_path(control_np, self.dt)
                waypoints = self.simulator.compute_waypoints_for_path(
                    self.dt, ws_path
                )
                result = self.simulator.execute_waypoints(waypoints)
            else:
                result = self.simulator.execute_segment(control_np, self.dt)
            result_np = np.asarray(result, dtype=float)
            if result_np.ndim > 1:
                result_np = result_np[0]
            self.current_state = result_np
            self._record_primitive_state()
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
    config: dict | None = None,
) -> Simulator:
    """Create a Gaussian or MuJoCo simulator for a canonical system name."""
    mode = mode.lower()
    system_name = system_name.lower()
    if mode == "gaussian":
        if system_name == "kinematic_car":
            return KinematicCarGaussianNoise(config=config)
        if system_name == "double_integrator":
            return DoubleIntegratorGaussianNoise(config=config)
        if system_name == "pushing_object":
            return PushingObjectGaussianNoise(config=config)
        if system_name == "dubins_airplane":
            return DubinsAirplaneGaussianNoise(config=config)
    elif mode == "mujoco":
        if system_name == "kinematic_car":
            return KinematicCarMujoco(config=config)
        if system_name == "pushing_object":
            return PushingObjectMujoco(config=config)

    raise ValueError(
        f"Unsupported simulator combination: system={system_name}, mode={mode}"
    )
