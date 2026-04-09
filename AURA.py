from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import threading
import numpy as np

from optimization import runOptimizer
from systems import System, get_system
from plan import OMPL_Planner
from utils.childrenHandler import getChildrenStates
from simulators import Simulator, create_simulator
from utils.utils import arrayDistance, log, state2list


@dataclass
class AURAResult:
    num_controls: int
    final_state: np.ndarray
    cost: float
    tracking_error_mean: float
    tracking_error_list: list[float]
    controls_trajectory: list[np.ndarray]
    states_trajectory: list[np.ndarray]


class AURA:
    """
    AURA execution loop built on top of:
      - systems.System subclasses
      - systems.plan planning wrapper
      - simulators.Simulator backends
    """

    def __init__(
        self,
        system: System,
        planner: OMPL_Planner,
        simulator: Simulator,
    ):
        self.system = system
        self.planner = planner
        self.simulator = simulator

        self.start_state = planner.start_state
        self.goal_state = planner.goal_state
        self.goal_threshold = planner.goal_threshold
        self.propagation_step_size = planner.propagation_step_size
        self.initial_planning_time = planner.initial_planning_time
        self.replanning_time = planner.replanning_time
        self.pruning_radius = planner.pruning_radius
        self.opt_model = getattr(planner, "opt_model", None)

        self.execution_thread_result = {"result": None, "completed": False}
        self.optimization_thread_result = {"result": None, "completed": False}
        self.replanning_thread_result = {"result": None, "completed": False}

    def run(self, reset_sim: bool = True) -> AURAResult:
        if reset_sim:
            self.simulator.reset()

        self.simulator.set_obj_init_pose(self.start_state.tolist())

        current_state = np.array(self.simulator.get_state(), dtype=float)

        actual_trajectory = [current_state.copy()]
        nominal_trajectory = [self.start_state.copy()]
        controls_trajectory: list[np.ndarray] = []

        best_trajectory = self.planner.getBestSolution()[0]
        next_control = best_trajectory["controls"][0]
        control_duration = self.planner.propagation_step_size

        index = 0
        while best_trajectory["control_count"] > 1:

            print(f"Executing control {index}: {next_control}")

            current_state = self.simulator.get_state()
            actual_trajectory.append(current_state.copy())
            nominal_trajectory.append(best_trajectory["states"][0].copy())

            print(
                f"State difference: {arrayDistance(actual_trajectory[-1], nominal_trajectory[-1], system=self.system.name)}"
            )

            # Execute thread
            executeThread = threading.Thread(
                target=self.execution,
                args=(next_control, control_duration),
            )
            executeThread.start()

            # Optimization thread
            next_nominal_state = best_trajectory["states"][1]

            children_states, children_controls = getChildrenStates(
                self.planner.ss, next_nominal_state, system=self.system.name
            )

            optimizationThread = threading.Thread(
                target=self.optimization,
                args=(self.system, next_nominal_state, children_states, children_controls),
            )
            optimizationThread.start()

            # Replanning thread
            replanningThread = threading.Thread(
                target=self.replanning,
                args=(self.planner),
            )
            replanningThread.start()

            # Wait for all threads to complete
            executeThread.join()
            optimizationThread.join()
            replanningThread.join()
            print("All threads completed")

            # Extract execution result
            if self.execution_thread_result["completed"]:
                current_state = self.execution_thread_result["result"]
                actual_trajectory.append(current_state.copy())
            else:
                print("Execution thread failed")
                break

            # Extract optimization result
            if self.optimization_thread_result["completed"]:
                optimization_result = self.optimization_thread_result["result"]
            else:
                print("Optimization thread failed")
                break

            # Extract replanning result
            if self.replanning_thread_result["completed"]:
                solutions = self.replanning_thread_result["result"]
                for i, solution in enumerate(solutions):
                    planned_dist = arrayDistance(
                        solution["states"][0], solution["states"][1], system=self.system.name
                    )
                    actual_dist = arrayDistance(
                        current_state, solution["states"][1], system=self.system.name
                    )
                    solution["cost"] = solution["cost"] - planned_dist + actual_dist
                    print(f"Solution {i} cost change: {actual_dist - planned_dist}")

                solutions.sort(key=lambda x: x["cost"])
                best_trajectory = solutions[0]
            else:
                print("Replanning thread failed")
                break

            next_state = best_trajectory["states"][1]

            next_control = self.pick_next_control(
                self.system,
                optimization_result,
                current_state,
                next_state,
                children_states,
                children_controls,
            )

            index += 1
            input("Press Enter to continue...")

        # Execute the last control
        final_control = best_trajectory["controls"][0]
        executionThread = threading.Thread(
            target=self.execution,
            args=(final_control, control_duration),
        )
        executionThread.start()
        executionThread.join()
        if self.execution_thread_result["completed"]:
            current_state = self.execution_thread_result["result"]
            actual_trajectory.append(current_state.copy())
            nominal_trajectory.append(best_trajectory["states"][-1].copy())

        cost = 0.0
        for i in range(len(actual_trajectory) - 1):
            cost += arrayDistance(
                actual_trajectory[i], actual_trajectory[i + 1], system=self.system.name
            )

        # Compute tracking errors after execution from nominal vs actual trajectory pairs.
        tracking_errors = []
        for i in range(len(actual_trajectory)):
            tracking_errors.append(
                float(
                    arrayDistance(
                        actual_trajectory[i], nominal_trajectory[i], system=self.system.name
                    )
                )
            )

        return AURAResult(
            num_controls=max(0, len(controls_trajectory)),
            final_state=np.asarray(actual_trajectory[-1], dtype=float),
            cost=float(cost),
            tracking_error_mean=float(np.mean(tracking_errors)),
            tracking_error_list=tracking_errors,
            controls_trajectory=[np.asarray(c, dtype=float) for c in controls_trajectory],
            states_trajectory=[np.asarray(s, dtype=float) for s in actual_trajectory],
        )

    def optimization(
        self,
        system: System,
        next_nominal_state: np.ndarray,
        children_states: list[np.ndarray],
        children_controls: list[np.ndarray],
    ):
        result = runOptimizer(
            system=system.name,
            nextState=np.asarray(next_nominal_state, dtype=float),
            childrenStatesArray=children_states,
            childrenControlsArray=children_controls,
            optModel=self.opt_model,
            numStates=1000,
            posSTD=0.003,
            rotSTD=0.05,
        )
        self.optimization_thread_result["result"] = result
        self.optimization_thread_result["completed"] = result is not None
        return result

    def execution(self, control: np.ndarray, duration: float):
        self.simulator.execute_segment(control, duration)
        result = self.simulator.get_state()
        self.execution_thread_result["result"] = result
        self.execution_thread_result["completed"] = result is not None
        return result

    def replanning(self, planner: OMPL_Planner):
        result = self.planner.replan()
        self.replanning_thread_result["result"] = result
        self.replanning_thread_result["completed"] = result is not None
        return result

    def sample_random_state(
        self,
        system: str,
        state: np.ndarray,
        num_states: int = 1000,
        pos_std: float = 0.003,
        rot_std: float = 0.05,
    ):
        sampled_states = []
        system_key = {
            "kinematic_car": "kinematic_car",
            "pushing_object": "pushing_object",
            "double_integrator": "double_integrator",
        }.get(system, system)

        if system_key in ("kinematic_car", "pushing_object"):
            if hasattr(state, "getX"):
                # utils.state2list uses legacy labels for SE2 extraction.
                state_list = state2list(state, "simple_car")
            else:
                state_list = np.asarray(state, dtype=float).reshape(-1).tolist()

            for _ in range(num_states):
                noisy_x = state_list[0] + np.random.normal(0.0, pos_std)
                noisy_y = state_list[1] + np.random.normal(0.0, pos_std)
                noisy_yaw = state_list[2] + np.random.normal(0.0, rot_std)
                while noisy_yaw > np.pi:
                    noisy_yaw -= 2 * np.pi
                while noisy_yaw < -np.pi:
                    noisy_yaw += 2 * np.pi
                sampled_states.append([noisy_x, noisy_y, noisy_yaw])
            return sampled_states

        if system_key == "double_integrator":
            if isinstance(state, (list, tuple, np.ndarray)):
                state_list = np.asarray(state, dtype=float).reshape(-1).tolist()
            else:
                # OMPL RealVectorState fallback.
                state_list = [float(state[i]) for i in range(3)]
            if len(state_list) < 3:
                raise ValueError(
                    f"double_integrator expects state with at least 3 values, got {len(state_list)}"
                )

            for _ in range(num_states):
                noisy = [state_list[i] + np.random.normal(0.0, pos_std) for i in range(3)]
                sampled_states.append(noisy)
            return sampled_states

        raise ValueError(f"Unsupported system for sampling: {system_key}")

    def pick_next_control(
        self,
        system: System,
        optimization_result: dict,
        current_state: np.ndarray,
        next_state: np.ndarray,
        children_states: list[np.ndarray],
        children_controls: list[np.ndarray],
    ):
        def _to_numpy(x):
            if x is None:
                return None
            if hasattr(x, "detach"):
                x = x.detach()
            if hasattr(x, "cpu"):
                x = x.cpu()
            if hasattr(x, "numpy"):
                return np.asarray(x.numpy(), dtype=float)
            if isinstance(x, np.ndarray):
                return x.astype(float)
            if isinstance(x, (list, tuple)):
                return np.asarray(x, dtype=float)
            return np.asarray(state2list(x, system.name), dtype=float)

        if not optimization_result or "optimized_controls" not in optimization_result:
            return (
                np.asarray(children_controls[0], dtype=float)
                if len(children_controls) > 0
                else None
            )

        optimized_controls = _to_numpy(optimization_result.get("optimized_controls"))
        start_states = _to_numpy(optimization_result.get("start_states"))
        target_states = _to_numpy(optimization_result.get("target_states"))

        if (
            optimized_controls is None
            or start_states is None
            or target_states is None
            or len(children_states) == 0
        ):
            return (
                np.asarray(children_controls[0], dtype=float)
                if len(children_controls) > 0
                else None
            )

        num_children = len(children_states)
        if optimized_controls.ndim == 1:
            optimized_controls = optimized_controls.reshape(1, -1)
        if start_states.ndim == 1:
            start_states = start_states.reshape(1, -1)
        if target_states.ndim == 1:
            target_states = target_states.reshape(1, -1)

        if optimized_controls.shape[0] < num_children:
            return (
                np.asarray(children_controls[0], dtype=float)
                if len(children_controls) > 0
                else None
            )

        sampling_num_states = max(1, optimized_controls.shape[0] // num_children)
        control_dim = optimized_controls.shape[-1]
        state_dim = start_states.shape[-1]

        optimized_controls = optimized_controls.reshape(
            num_children, sampling_num_states, control_dim
        )
        start_states = start_states.reshape(num_children, sampling_num_states, state_dim)
        target_states = target_states.reshape(num_children, sampling_num_states, state_dim)

        children_samples = target_states[:, 0, :]
        actual_next = _to_numpy(next_state).reshape(-1)
        actual_current = _to_numpy(current_state).reshape(-1)

        closest_child_idx = int(
            np.argmin(
                [
                    arrayDistance(actual_next, child_sample, system=system.name)
                    for child_sample in children_samples
                ]
            )
        )

        candidate_controls = optimized_controls[closest_child_idx]
        predicted_states = [
            system.propagate(actual_current.copy(), ctrl, self.propagation_step_size)
            for ctrl in candidate_controls
        ]
        distances = [
            arrayDistance(actual_next, pred, system=system.name) for pred in predicted_states
        ]
        best_idx = int(np.argmin(distances))
        best_optimized_control = np.asarray(candidate_controls[best_idx], dtype=float)

        # Compare optimized candidate against original branch control and pick the better one.
        original_control = None
        if len(children_controls) > closest_child_idx:
            original_control = _to_numpy(children_controls[closest_child_idx]).reshape(-1)
        if original_control is None:
            return best_optimized_control

        predicted_original = system.propagate(
            actual_current.copy(), original_control, self.propagation_step_size
        )
        predicted_optimized = system.propagate(
            actual_current.copy(), best_optimized_control, self.propagation_step_size
        )
        original_distance = arrayDistance(actual_next, predicted_original, system=system.name)
        optimized_distance = arrayDistance(actual_next, predicted_optimized, system=system.name)

        if optimized_distance <= original_distance:
            return best_optimized_control
        return original_control
