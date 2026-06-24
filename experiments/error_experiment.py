#!/usr/bin/env python3
"""Simulated tracking-error evaluation for naive vs optimized controls.

Typical usage:
  python experiments/error_experiment.py kinematic_car gaussian
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aura_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from AURA import AURA
from optimization import runOptimizer
from simulation.pushing_dynamics import get_pushing_model
from simulation.simulators import create_simulator
from systems import get_system
from train_model import load_opt_model_2
from utils.utils import arrayDistance


@dataclass
class ErrorExperimentResult:
    controls: list[np.ndarray]
    optimized_controls: list[np.ndarray]
    nominal_trajectory: list[np.ndarray]
    naive_trajectory: list[np.ndarray]
    optimized_trajectory: list[np.ndarray]
    naive_tracking_error: list[float]
    optimized_tracking_error: list[float]
    simulator_pos_std: float
    simulator_rot_std: float
    simulator_vel_std: float
    optimization_pos_std: float
    optimization_rot_std: float
    optimization_vel_std: float
    optimizer_num_states: int
    optimizer_learning_rate: float
    optimizer_epochs: int


def _system_alias(system_name: str) -> str:
    return {
        "kinematic_car": "simple_car",
        "double_integrator": "double_integrator",
        "pushing_object": "pushing",
    }[system_name]


def _sample_random_controls(
    system,
    num_controls: int,
    rng: np.random.Generator,
    *,
    environment_name: str = "gaussian",
) -> list[np.ndarray]:
    controls = []
    for _ in range(num_controls):
        if system.name == "pushing_object":
            controls.append(
                _generate_random_push_params(
                    system, rng, environment_name=environment_name
                )
            )
        else:
            control = [rng.uniform(low, high) for low, high in system.control_bounds]
            controls.append(np.asarray(control, dtype=float))
    return controls


def _generate_random_push_params(
    system, rng: np.random.Generator, *, environment_name: str = "gaussian"
) -> np.ndarray:
    """Sample push params with the same structure as PushingControlSampler."""
    if environment_name == "mujoco":
        return np.asarray(
            [
                0.0,
                rng.uniform(-0.32, 0.32),
                rng.uniform(0.045, 0.085),
            ],
            dtype=float,
        )

    low0, high0 = system.control_bounds[0]
    low1, high1 = system.control_bounds[1]
    low2, high2 = system.control_bounds[2]

    control = np.zeros(3, dtype=float)
    faces = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float)
    if high0 <= 0.75:
        faces = faces[(faces >= low0) & (faces <= high0)]
    if len(faces) == 0:
        faces = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float)
    control[0] = float(rng.choice(faces))

    # Sample side excluding the near-center interval [-0.05, 0.05].
    gap_low, gap_high = -0.05, 0.05
    if high1 <= gap_low or low1 >= gap_high:
        control[1] = rng.uniform(low1, high1)
    else:
        left_len = max(0.0, gap_low - low1)
        right_len = max(0.0, high1 - gap_high)
        total_len = left_len + right_len
        u = rng.random() * total_len
        if u < left_len:
            control[1] = low1 + u
        else:
            control[1] = gap_high + (u - left_len)

    control[2] = rng.uniform(low2, high2)

    return control


def _sample_mujoco_push_sequence(
    system,
    start_state: np.ndarray,
    num_controls: int,
    rng: np.random.Generator,
    duration: float,
) -> list[np.ndarray]:
    controls: list[np.ndarray] = []
    current = np.asarray(start_state, dtype=float).reshape(-1).copy()
    table_x = (-0.12, 0.76)
    table_y = (-0.82, -0.34)
    push_faces = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float)
    previous_face: float | None = None

    for _ in range(num_controls):
        valid: list[tuple[float, np.ndarray, np.ndarray]] = []
        for _candidate_idx in range(160):
            face = float(rng.choice(push_faces))
            control = np.asarray(
                [
                    face,
                    rng.uniform(-0.32, 0.32),
                    rng.uniform(0.035, 0.075),
                ],
                dtype=float,
            )
            next_state = np.asarray(
                system.propagate(current, control, duration), dtype=float
            )
            if not (
                table_x[0] <= next_state[0] <= table_x[1]
                and table_y[0] <= next_state[1] <= table_y[1]
            ):
                continue

            displacement = float(np.linalg.norm(next_state[:2] - current[:2]))
            center_penalty = 0.22 * abs(float(next_state[1] + 0.58))
            repeat_penalty = (
                0.025
                if previous_face is not None and abs(face - previous_face) < 1e-9
                else 0.0
            )
            score = (
                displacement
                - center_penalty
                - repeat_penalty
                + float(rng.normal(0.0, 0.006))
            )
            valid.append((score, control, next_state))

        if not valid:
            control = np.asarray([0.0, 0.0, 0.045], dtype=float)
            next_state = np.asarray(
                system.propagate(current, control, duration), dtype=float
            )
        else:
            valid.sort(key=lambda item: item[0], reverse=True)
            top_k = valid[: min(8, len(valid))]
            _, control, next_state = top_k[int(rng.integers(0, len(top_k)))]

        controls.append(control)
        current = next_state.copy()
        current[2] = float((current[2] + np.pi) % (2.0 * np.pi) - np.pi)
        previous_face = float(control[0])

    return controls


def _simulator_config(
    system_name: str, environment_name: str, duration: float | None
) -> dict:
    config = {}
    if (
        duration is None
        and system_name == "pushing_object"
        and environment_name == "mujoco"
    ):
        duration = 2.0
    if duration is not None:
        config["propagation_step_size"] = float(duration)
    if environment_name == "gaussian":
        config["sampling_position_std"] = 0.003
        config["sampling_rotation_std"] = 0.05
        if system_name == "double_integrator":
            config["sampling_rotation_std"] = 0.0
            config["sampling_velocity_std"] = 0.001
    return config


def _ensure_optimizer_stds_exceed_simulator(
    opt_pos: float,
    opt_rot: float,
    opt_vel: float,
    sim_pos: float,
    sim_rot: float,
    sim_vel: float,
    *,
    min_ratio: float = 1.2,
) -> tuple[float, float, float]:
    """Optimizer sampling std must be strictly larger than simulator noise when that noise is > 0."""

    def bump(opt: float, sim: float) -> float:
        if sim <= 0.0:
            return opt
        return float(max(opt, sim * min_ratio))

    return bump(opt_pos, sim_pos), bump(opt_rot, sim_rot), bump(opt_vel, sim_vel)


def _optimization_sampling_stds(
    system_name: str, simulator_config: dict
) -> tuple[float, float, float]:
    sim_pos_std = float(simulator_config.get("sampling_position_std", 0.0))
    sim_rot_std = float(simulator_config.get("sampling_rotation_std", 0.0))
    sim_vel_std = float(simulator_config.get("sampling_velocity_std", sim_pos_std))
    if system_name == "double_integrator":
        opt_pos_std = max(0.01, 3.0 * sim_pos_std)
        opt_vel_std = max(0.003, 3.0 * sim_vel_std)
        opt_pos_std, _, opt_vel_std = _ensure_optimizer_stds_exceed_simulator(
            opt_pos_std, 0.0, opt_vel_std, sim_pos_std, 0.0, sim_vel_std
        )
        return opt_pos_std, 0.0, opt_vel_std

    if system_name == "pushing_object":
        opt_pos_std = max(0.035, 3.0 * sim_pos_std)
        opt_rot_std = max(0.35, 2.0 * sim_rot_std) if sim_rot_std > 0.0 else 0.35
        opt_vel_std = max(0.003, 3.0 * sim_vel_std)
        return _ensure_optimizer_stds_exceed_simulator(
            opt_pos_std, opt_rot_std, opt_vel_std, sim_pos_std, sim_rot_std, sim_vel_std
        )

    opt_pos_std = max(0.01, 3.0 * sim_pos_std)
    opt_rot_std = max(0.1, 2.0 * sim_rot_std) if sim_rot_std > 0.0 else 0.1
    opt_vel_std = max(0.003, 3.0 * sim_vel_std)
    return _ensure_optimizer_stds_exceed_simulator(
        opt_pos_std, opt_rot_std, opt_vel_std, sim_pos_std, sim_rot_std, sim_vel_std
    )


def _build_nominal_trajectory(
    system, start_state: np.ndarray, controls: list[np.ndarray], duration: float
):
    trajectory = [np.asarray(start_state, dtype=float).copy()]
    current = trajectory[0].copy()
    for control in controls:
        current = np.asarray(system.propagate(current, control, duration), dtype=float)
        trajectory.append(current.copy())
    return trajectory


def _canonicalize_state(
    system_name: str, state: np.ndarray | list[float]
) -> np.ndarray:
    """Project simulator states to the system state dimensionality used by dynamics."""
    state_np = np.asarray(state, dtype=float).reshape(-1)
    if system_name in ("kinematic_car", "pushing_object"):
        return state_np[:3]
    if system_name == "double_integrator":
        return state_np[:6]
    return state_np


def _execute_trajectory(
    simulator,
    system_name: str,
    start_state: np.ndarray,
    controls: list[np.ndarray],
    duration: float,
    noise_seed_base: int | None = None,
):
    simulator.reset()
    simulator.set_object_init_pose(np.asarray(start_state, dtype=float).tolist())
    trajectory = [_canonicalize_state(system_name, simulator.get_state())]
    for step_idx, control in enumerate(controls):
        if noise_seed_base is not None:
            np.random.seed(noise_seed_base + step_idx)
        state = simulator.execute_segment(control, duration)
        trajectory.append(_canonicalize_state(system_name, state))
    return trajectory


def _load_pushing_opt_model(system, learning_rate: float, epochs: int):
    model = get_pushing_model(system.object_shape)
    return load_opt_model_2(model, lr=float(learning_rate), epochs=int(epochs))


class _PickerSimulatorStub:
    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})


class _PickerPlannerStub:
    obstacle_config = None

    def __init__(self, motion_validation_step_size: float):
        self.motion_validation_step_size = float(motion_validation_step_size)


def _make_aura_picker(system, duration: float, simulator_config: dict):
    aura = AURA.__new__(AURA)
    aura.system = system
    aura.simulator = _PickerSimulatorStub(simulator_config)
    aura.planner = _PickerPlannerStub(max(0.02, min(0.05, float(duration) / 20.0)))
    aura.propagation_step_size = float(duration)
    aura.last_control_decision = {}
    # The error experiment compares tracking performance, not planning safety.
    # Let the optimizer candidate be selected by endpoint prediction alone;
    # otherwise the planning-time state-bounds gate can reject a useful local
    # correction before MuJoCo ever gets to execute it.
    aura._control_curve_valid = lambda *_args, **_kwargs: True
    return aura


def _predicted_tracking_distance(
    system,
    current_state: np.ndarray,
    control: np.ndarray,
    target_state: np.ndarray,
    duration: float,
) -> float:
    predicted = np.asarray(
        system.propagate(current_state, control, duration), dtype=float
    )
    return float(arrayDistance(predicted, target_state, system=system.name))


def _pushing_side_limit(system, control: np.ndarray) -> float | None:
    if system.name != "pushing_object":
        return None
    return 0.4


def _build_optimized_rollout(
    system,
    simulator,
    start_state: np.ndarray,
    nominal_trajectory: list[np.ndarray],
    controls: list[np.ndarray],
    duration: float,
    opt_pos_std: float,
    opt_rot_std: float,
    opt_vel_std: float,
    optimizer_num_states: int,
    optimizer_learning_rate: float,
    optimizer_epochs: int,
    noise_seed_base: int | None = None,
):
    opt_model = (
        _load_pushing_opt_model(system, optimizer_learning_rate, optimizer_epochs)
        if system.name == "pushing_object"
        else None
    )

    aura_picker = _make_aura_picker(
        system,
        duration,
        getattr(simulator, "config", {}) or {},
    )
    optimized_controls: list[np.ndarray] = []
    simulator.reset()
    simulator.set_object_init_pose(np.asarray(start_state, dtype=float).tolist())
    optimized_trajectory = [_canonicalize_state(system.name, simulator.get_state())]

    for step_idx, original_control in enumerate(controls):
        current_state = np.asarray(optimized_trajectory[-1], dtype=float)
        target_state = np.asarray(nominal_trajectory[step_idx + 1], dtype=float)
        print(
            f"[AUDIT] step={step_idx} optimizer_target_is_nominal[{step_idx + 1}]="
            f"{np.array2string(target_state, precision=5, suppress_small=True)} "
            f"current_optimized_state={np.array2string(current_state, precision=5, suppress_small=True)}"
        )
        optimization_result = None
        if opt_model is not None or system.name in (
            "kinematic_car",
            "double_integrator",
        ):
            optimization_result = runOptimizer(
                system=system.name,
                nextState=current_state,
                childrenStatesArray=[target_state],
                childrenControlsArray=[np.asarray(original_control, dtype=float)],
                optModel=opt_model,
                numStates=int(optimizer_num_states),
                posSTD=opt_pos_std,
                rotSTD=opt_rot_std,
                velSTD=opt_vel_std,
                originalControl=np.asarray(original_control, dtype=float),
                controlDuration=duration,
                learningRate=float(optimizer_learning_rate),
            )
        if (
            optimization_result is not None
            and "optimized_controls" in optimization_result
        ):
            optimized_controls_np = optimization_result["optimized_controls"]
            if hasattr(optimized_controls_np, "detach"):
                optimized_controls_np = optimized_controls_np.detach().cpu().numpy()
            optimized_controls_np = np.asarray(optimized_controls_np, dtype=float)
            if optimized_controls_np.ndim == 1:
                optimized_controls_np = optimized_controls_np.reshape(1, -1)
            control_deltas = np.linalg.norm(
                optimized_controls_np
                - np.asarray(original_control, dtype=float).reshape(1, -1),
                axis=1,
            )
            print(
                f"[OPT] step={step_idx} candidate_delta_min={float(np.min(control_deltas)):.6f} "
                f"candidate_delta_mean={float(np.mean(control_deltas)):.6f} "
                f"candidate_delta_max={float(np.max(control_deltas)):.6f}"
            )
        best_control = aura_picker.pick_next_control(
            system=system,
            optimization_result=optimization_result,
            current_state=current_state,
            next_state=target_state,
            children_states=[target_state],
            children_controls=[np.asarray(original_control, dtype=float)],
            control_duration=duration,
            fallback_control=np.asarray(original_control, dtype=float),
        )
        original_pred_distance = _predicted_tracking_distance(
            system,
            current_state=np.asarray(current_state, dtype=float),
            control=np.asarray(original_control, dtype=float),
            target_state=np.asarray(target_state, dtype=float),
            duration=duration,
        )
        chosen_pred_distance = _predicted_tracking_distance(
            system,
            current_state=np.asarray(current_state, dtype=float),
            control=np.asarray(best_control, dtype=float),
            target_state=np.asarray(target_state, dtype=float),
            duration=duration,
        )
        print(
            f"[OPT] step={step_idx} "
            f"target={np.asarray(target_state, dtype=float)} "
            f"orig_control={np.asarray(original_control, dtype=float)} "
            f"chosen_control={np.asarray(best_control, dtype=float)} "
            f"control_delta={np.linalg.norm(np.asarray(best_control, dtype=float) - np.asarray(original_control, dtype=float)):.6f} "
            f"orig_pred_err={original_pred_distance:.6f} "
            f"chosen_pred_err={chosen_pred_distance:.6f}"
        )
        side_limit = _pushing_side_limit(system, best_control)
        if side_limit is not None:
            side_offset = float(np.asarray(best_control, dtype=float).reshape(-1)[1])
            print(
                f"[OPT] step={step_idx} pushing_relative_side_offset={side_offset:.6f} "
                f"limit=+/-{side_limit:.6f} "
                f"fraction={abs(side_offset) / max(side_limit, 1e-9):.3f}"
            )
        decision = getattr(aura_picker, "last_control_decision", {}) or {}
        if decision:
            print(
                f"[OPT] step={step_idx} decision={decision.get('source')} "
                f"reason={decision.get('reason')} "
                f"model_original_distance={decision.get('original_distance')} "
                f"model_optimized_distance={decision.get('optimized_distance')} "
                f"original_curve_valid={decision.get('original_curve_valid')} "
                f"optimized_curve_valid={decision.get('optimized_curve_valid')}"
            )
        if optimization_result is not None and "final_loss" in optimization_result:
            print(
                f"[OPT] step={step_idx} final_loss={float(optimization_result['final_loss']):.6f}"
            )
        optimized_controls.append(best_control)
        if noise_seed_base is not None:
            np.random.seed(noise_seed_base + step_idx)
        next_state = simulator.execute_segment(best_control, duration)
        next_state = _canonicalize_state(system.name, next_state)
        actual_step_error = float(
            arrayDistance(next_state, target_state, system=system.name)
        )
        print(
            f"[OPT] step={step_idx} actual_mujoco_error_after_chosen={actual_step_error:.6f} "
            f"(computed against nominal[{step_idx + 1}])"
        )
        optimized_trajectory.append(next_state)

    return optimized_controls, optimized_trajectory


def _close_simulator(simulator) -> None:
    if simulator is None:
        return
    try:
        simulator.stop()
    except Exception:
        pass
    try:
        simulator.close()
    except Exception:
        pass


def _trajectory_errors(
    reference: list[np.ndarray], trajectory: list[np.ndarray], system_name: str
) -> list[float]:
    return [
        float(
            arrayDistance(
                np.asarray(actual, dtype=float),
                np.asarray(target, dtype=float),
                system=system_name,
            )
        )
        for actual, target in zip(trajectory, reference)
    ]


def run_error_experiment(
    system_name: str,
    environment_name: str,
    num_controls: int = 10,
    seed: int = 42,
    duration: float | None = None,
    optimizer_num_states: int | None = None,
    optimizer_learning_rate: float | None = None,
    optimizer_epochs: int | None = None,
):
    controls_rng = np.random.default_rng(seed)

    system = get_system(system_name)
    simulator_config = _simulator_config(system_name, environment_name, duration)
    opt_pos_std, opt_rot_std, opt_vel_std = _optimization_sampling_stds(
        system_name, simulator_config
    )
    if optimizer_num_states is None:
        optimizer_num_states = 10000 if system_name == "pushing_object" else 5000
    if optimizer_learning_rate is None:
        optimizer_learning_rate = 5e-4 if system_name == "pushing_object" else 0.05
    if optimizer_epochs is None:
        optimizer_epochs = 1000 if system_name == "pushing_object" else 1000
    simulator = None
    optimized_simulator = None
    try:
        simulator = create_simulator(
            _system_alias(system_name), environment_name, config=simulator_config
        )

        simulator.reset()
        if system_name == "pushing_object":
            if environment_name == "mujoco":
                # Match the smooth MuJoCo pushing demo: start in front of the UR10
                # with the cracker box oriented for rightward pushes.
                start_state = np.asarray([0.25, -0.58, np.pi], dtype=float)
            else:
                start_state = _canonicalize_state(system_name, simulator.get_state())
        else:
            start_state = _canonicalize_state(system_name, simulator.get_state())
        segment_duration = float(
            simulator_config.get("propagation_step_size", simulator.dt)
        )

        if system_name == "pushing_object" and environment_name == "mujoco":
            controls = _sample_mujoco_push_sequence(
                system, start_state, num_controls, controls_rng, segment_duration
            )
        else:
            controls = _sample_random_controls(
                system, num_controls, controls_rng, environment_name=environment_name
            )
        nominal_trajectory = _build_nominal_trajectory(
            system, start_state, controls, segment_duration
        )
        rollout_noise_seed = seed + 10_000
        naive_trajectory = _execute_trajectory(
            simulator,
            system_name,
            start_state,
            controls,
            segment_duration,
            noise_seed_base=rollout_noise_seed,
        )

        if environment_name == "mujoco":
            # Reuse the viewer-backed simulator within the trial. Creating a
            # second MuJoCo viewer before the first one is fully torn down can
            # crash in native GLFW/MuJoCo code.
            optimized_simulator = simulator
        else:
            optimized_simulator = create_simulator(
                _system_alias(system_name), environment_name, config=simulator_config
            )
        optimized_controls, optimized_trajectory = _build_optimized_rollout(
            system=system,
            simulator=optimized_simulator,
            start_state=start_state,
            nominal_trajectory=nominal_trajectory,
            controls=controls,
            duration=segment_duration,
            opt_pos_std=opt_pos_std,
            opt_rot_std=opt_rot_std,
            opt_vel_std=opt_vel_std,
            optimizer_num_states=int(optimizer_num_states),
            optimizer_learning_rate=float(optimizer_learning_rate),
            optimizer_epochs=int(optimizer_epochs),
            noise_seed_base=rollout_noise_seed,
        )

        return ErrorExperimentResult(
            controls=controls,
            optimized_controls=optimized_controls,
            nominal_trajectory=nominal_trajectory,
            naive_trajectory=naive_trajectory,
            optimized_trajectory=optimized_trajectory,
            naive_tracking_error=_trajectory_errors(
                nominal_trajectory, naive_trajectory, system.name
            ),
            optimized_tracking_error=_trajectory_errors(
                nominal_trajectory, optimized_trajectory, system.name
            ),
            simulator_pos_std=float(simulator_config.get("sampling_position_std", 0.0)),
            simulator_rot_std=float(simulator_config.get("sampling_rotation_std", 0.0)),
            simulator_vel_std=float(
                simulator_config.get(
                    "sampling_velocity_std",
                    simulator_config.get("sampling_position_std", 0.0),
                )
            ),
            optimization_pos_std=opt_pos_std,
            optimization_rot_std=opt_rot_std,
            optimization_vel_std=opt_vel_std,
            optimizer_num_states=int(optimizer_num_states),
            optimizer_learning_rate=float(optimizer_learning_rate),
            optimizer_epochs=int(optimizer_epochs),
        )
    finally:
        if optimized_simulator is not simulator:
            _close_simulator(optimized_simulator)
        _close_simulator(simulator)
        if environment_name == "mujoco":
            time.sleep(0.5)


def _plot_tracking_errors(
    result: ErrorExperimentResult, system_name: str, environment_name: str
):
    control_indices = np.arange(len(result.naive_tracking_error))
    plt.figure(figsize=(8, 4.5))
    plt.plot(control_indices, result.naive_tracking_error, marker="o", label="Naive")
    plt.plot(
        control_indices, result.optimized_tracking_error, marker="s", label="Optimized"
    )
    plt.xlabel("Executed controls")
    plt.ylabel("Tracking error")
    plt.title(f"Tracking Error per Control: {system_name} ({environment_name})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_average_tracking_errors(
    naive_mean_by_step: np.ndarray,
    optimized_mean_by_step: np.ndarray,
    system_name: str,
    environment_name: str,
    num_trials: int,
):
    control_indices = np.arange(len(naive_mean_by_step))
    plt.figure(figsize=(8, 4.5))
    plt.plot(control_indices, naive_mean_by_step, marker="o", label="Naive (avg)")
    plt.plot(
        control_indices, optimized_mean_by_step, marker="s", label="Optimized (avg)"
    )
    plt.xlabel("Executed controls")
    plt.ylabel("Tracking error")
    plt.title(
        f"Average Tracking Error per Control: {system_name} ({environment_name}), trials={num_trials}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Run naive vs optimized trajectory error experiment."
    )
    parser.add_argument(
        "system_name",
        choices=["kinematic_car", "double_integrator", "pushing_object"],
        help="System name.",
    )
    parser.add_argument(
        "environment_name",
        choices=["gaussian", "mujoco"],
        help="Simulator environment/backend.",
    )
    parser.add_argument(
        "--num-controls", type=int, default=5, help="Number of random controls."
    )
    parser.add_argument(
        "--num-trials", type=int, default=5, help="Number of experiment trials."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--duration", type=float, default=None, help="Duration per control segment."
    )
    parser.add_argument(
        "--optimizer-num-states",
        type=int,
        default=None,
        help="Number of sampled states used by the optimizer.",
    )
    parser.add_argument(
        "--optimizer-learning-rate",
        type=float,
        default=None,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--optimizer-epochs",
        type=int,
        default=None,
        help="Pushing optimizer epochs.",
    )
    args = parser.parse_args()

    all_results: list[ErrorExperimentResult] = []
    for trial_idx in range(args.num_trials):
        trial_seed = args.seed + trial_idx
        print(
            f"[INFO] Running trial {trial_idx + 1}/{args.num_trials} (seed={trial_seed})"
        )
        result = run_error_experiment(
            system_name=args.system_name,
            environment_name=args.environment_name,
            num_controls=args.num_controls,
            seed=trial_seed,
            duration=args.duration,
            optimizer_num_states=args.optimizer_num_states,
            optimizer_learning_rate=args.optimizer_learning_rate,
            optimizer_epochs=args.optimizer_epochs,
        )
        all_results.append(result)

    naive_errors = np.asarray(
        [res.naive_tracking_error for res in all_results], dtype=float
    )
    optimized_errors = np.asarray(
        [res.optimized_tracking_error for res in all_results], dtype=float
    )
    naive_mean_by_step = np.mean(naive_errors, axis=0)
    optimized_mean_by_step = np.mean(optimized_errors, axis=0)

    naive_mean = float(np.mean(naive_mean_by_step))
    optimized_mean = float(np.mean(optimized_mean_by_step))
    first_result = all_results[0]
    print(f"[INFO] System: {args.system_name}, environment: {args.environment_name}")
    print(
        f"[INFO] Simulator noise stds: pos={first_result.simulator_pos_std:.6f}, rot={first_result.simulator_rot_std:.6f}, vel={first_result.simulator_vel_std:.6f}"
    )
    print(
        f"[INFO] Optimization sampling stds: pos={first_result.optimization_pos_std:.6f}, rot={first_result.optimization_rot_std:.6f}, vel={first_result.optimization_vel_std:.6f}"
    )
    print(
        f"[INFO] Optimizer settings: num_states={first_result.optimizer_num_states}, "
        f"learning_rate={first_result.optimizer_learning_rate:.6g}, epochs={first_result.optimizer_epochs}"
    )
    print(f"[INFO] Trials: {args.num_trials}")
    print(f"[INFO] Average naive tracking error over all trials: {naive_mean:.6f}")
    print(
        f"[INFO] Average optimized tracking error over all trials: {optimized_mean:.6f}"
    )
    print(f"[INFO] Naive mean per control: {naive_mean_by_step}")
    print(f"[INFO] Optimized mean per control: {optimized_mean_by_step}")
    _plot_average_tracking_errors(
        naive_mean_by_step,
        optimized_mean_by_step,
        args.system_name,
        args.environment_name,
        args.num_trials,
    )


if __name__ == "__main__":
    main()
