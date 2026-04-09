from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from AURA import AURA
from systems import get_system
from optimization import runOptimizer
from utils.utils import arrayDistance
from simulators import create_simulator
from train_model import load_opt_model_2
from pushing_dynamics import get_pushing_model

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


def _sample_random_controls(
    system, num_controls: int, rng: np.random.Generator
) -> list[np.ndarray]:
    controls = []
    for _ in range(num_controls):
        if system.name == "pushing_object":
            controls.append(_generate_random_push_params(system, rng))
        else:
            control = [rng.uniform(low, high) for low, high in system.control_bounds]
            controls.append(np.asarray(control, dtype=float))
    return controls


def _generate_random_push_params(system, rng: np.random.Generator) -> np.ndarray:
    """Sample push params with the same structure as PushingControlSampler."""
    low0, high0 = system.control_bounds[0]
    low1, high1 = system.control_bounds[1]
    low2, high2 = system.control_bounds[2]

    control = np.zeros(3, dtype=float)
    control[0] = rng.uniform(low0, high0)

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

    # Puna-style normalization used by the pushing control sampler.
    control[0] = int(control[0]) / 4.0
    mask = (int(control[0] * 4) % 2) == 0
    control[1] *= system.object_shape[1] if mask else system.object_shape[0]
    return control


def _simulator_config(system_name: str, environment_name: str, duration: float | None) -> dict:
    config = {}
    if duration is not None:
        config["propagation_step_size"] = float(duration)
    if environment_name == "gaussian":
        config["sampling_position_std"] = 0.003
        config["sampling_rotation_std"] = 0.05
        if system_name == "double_integrator":
            config["sampling_position_std"] = 0.003
            config["sampling_rotation_std"] = 0.0
            config["sampling_velocity_std"] = 0.003
    return config


def _optimization_sampling_stds(
    system_name: str, simulator_config: dict
) -> tuple[float, float, float]:
    sim_pos_std = float(simulator_config.get("sampling_position_std", 0.0))
    sim_rot_std = float(simulator_config.get("sampling_rotation_std", 0.0))
    sim_vel_std = float(simulator_config.get("sampling_velocity_std", sim_pos_std))
    if system_name == "double_integrator":
        # Keep optimizer sampling broader than simulator process noise.
        opt_pos_std = max(0.01, 3.0 * sim_pos_std)
        opt_vel_std = max(0.003, 3.0 * sim_vel_std)
        return opt_pos_std, 0.0, opt_vel_std

    if system_name == "pushing_object":
        opt_pos_std = max(0.01, 3.0 * sim_pos_std)
        opt_rot_std = max(0.1, 2.0 * sim_rot_std) if sim_rot_std > 0.0 else 0.1
        # Pushing does not use velocity state, keep legacy magnitude.
        opt_vel_std = max(0.003, 3.0 * sim_vel_std)
        return opt_pos_std, opt_rot_std, opt_vel_std

    opt_pos_std = max(0.01, 3.0 * sim_pos_std)
    opt_rot_std = max(0.1, 2.0 * sim_rot_std) if sim_rot_std > 0.0 else 0.1
    opt_vel_std = max(0.003, 3.0 * sim_vel_std)
    return opt_pos_std, opt_rot_std, opt_vel_std


def _build_nominal_trajectory(
    system, start_state: np.ndarray, controls: list[np.ndarray], duration: float
):
    trajectory = [np.asarray(start_state, dtype=float).copy()]
    current = trajectory[0].copy()
    for control in controls:
        current = np.asarray(system.propagate(current, control, duration), dtype=float)
        trajectory.append(current.copy())
    return trajectory


def _canonicalize_state(system_name: str, state: np.ndarray | list[float]) -> np.ndarray:
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


def _load_pushing_opt_model(system):
    model = get_pushing_model(system.object_shape)
    return load_opt_model_2(model)


def _make_aura_picker(duration: float):
    aura = AURA.__new__(AURA)
    aura.propagation_step_size = float(duration)
    return aura


def _predicted_tracking_distance(
    system,
    current_state: np.ndarray,
    control: np.ndarray,
    target_state: np.ndarray,
    duration: float,
) -> float:
    predicted = np.asarray(system.propagate(current_state, control, duration), dtype=float)
    return float(arrayDistance(predicted, target_state, system=system.name))


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
    noise_seed_base: int | None = None,
):
    opt_model = _load_pushing_opt_model(system) if system.name == "pushing_object" else None

    aura_picker = _make_aura_picker(duration)
    optimized_controls: list[np.ndarray] = []
    simulator.reset()
    simulator.set_object_init_pose(np.asarray(start_state, dtype=float).tolist())
    optimized_trajectory = [_canonicalize_state(system.name, simulator.get_state())]

    for step_idx, original_control in enumerate(controls):
        current_state = np.asarray(optimized_trajectory[-1], dtype=float)
        target_state = np.asarray(nominal_trajectory[step_idx + 1], dtype=float)
        optimization_result = None
        if opt_model is not None or system.name in ("kinematic_car", "double_integrator"):
            optimization_result = runOptimizer(
                system=system.name,
                nextState=current_state,
                childrenStatesArray=[target_state],
                childrenControlsArray=[np.asarray(original_control, dtype=float)],
                optModel=opt_model,
                numStates=5000,
                posSTD=opt_pos_std,
                rotSTD=opt_rot_std,
                velSTD=opt_vel_std,
                originalControl=np.asarray(original_control, dtype=float),
                controlDuration=duration,
            )
        if optimization_result is not None and "optimized_controls" in optimization_result:
            optimized_controls_np = optimization_result["optimized_controls"]
            if hasattr(optimized_controls_np, "detach"):
                optimized_controls_np = optimized_controls_np.detach().cpu().numpy()
            optimized_controls_np = np.asarray(optimized_controls_np, dtype=float)
            if optimized_controls_np.ndim == 1:
                optimized_controls_np = optimized_controls_np.reshape(1, -1)
            control_deltas = np.linalg.norm(
                optimized_controls_np - np.asarray(original_control, dtype=float).reshape(1, -1),
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
            f"orig_pred_err={original_pred_distance:.6f} "
            f"chosen_pred_err={chosen_pred_distance:.6f}"
        )
        if optimization_result is not None and "final_loss" in optimization_result:
            print(
                f"[OPT] step={step_idx} final_loss={float(optimization_result['final_loss']):.6f}"
            )
        optimized_controls.append(best_control)
        if noise_seed_base is not None:
            np.random.seed(noise_seed_base + step_idx)
        next_state = simulator.execute_segment(best_control, duration)
        optimized_trajectory.append(_canonicalize_state(system.name, next_state))

    return optimized_controls, optimized_trajectory


def _trajectory_errors(
    reference: list[np.ndarray], trajectory: list[np.ndarray], system_name: str
) -> list[float]:
    return [
        float(
            arrayDistance(
                np.asarray(actual, dtype=float), np.asarray(target, dtype=float), system=system_name
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
    simulator=None,
    optimized_simulator=None,
):
    controls_rng = np.random.default_rng(seed)

    system = get_system(system_name)
    simulator_config = _simulator_config(system_name, environment_name, duration)
    opt_pos_std, opt_rot_std, opt_vel_std = _optimization_sampling_stds(
        system_name, simulator_config
    )
    if simulator is None:
        simulator = create_simulator(system_name, environment_name, config=simulator_config)

    simulator.reset()
    if system_name == "pushing_object":
        # Keep the pushed object away from the robot base for Mujoco pushing runs.
        start_state = np.asarray([0.0, -0.7, 0.0], dtype=float)
    else:
        start_state = _canonicalize_state(system_name, simulator.get_state())
    segment_duration = float(simulator.dt if duration is None else duration)

    controls = _sample_random_controls(system, num_controls, controls_rng)
    nominal_trajectory = _build_nominal_trajectory(system, start_state, controls, segment_duration)
    rollout_noise_seed = seed + 10_000
    naive_trajectory = _execute_trajectory(
        simulator,
        system_name,
        start_state,
        controls,
        segment_duration,
        noise_seed_base=rollout_noise_seed,
    )

    if optimized_simulator is None:
        if environment_name == "mujoco":
            # Reuse a single MuJoCo simulator/viewer instance to avoid
            # crashes from multiple concurrent viewer-backed simulators.
            optimized_simulator = simulator
        else:
            optimized_simulator = create_simulator(
                system_name, environment_name, config=simulator_config
            )
    elif environment_name == "mujoco":
        optimized_simulator = simulator
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
        noise_seed_base=rollout_noise_seed,
    )

    return ErrorExperimentResult(
        controls=controls,
        optimized_controls=optimized_controls,
        nominal_trajectory=nominal_trajectory,
        naive_trajectory=naive_trajectory,
        optimized_trajectory=optimized_trajectory,
        naive_tracking_error=_trajectory_errors(nominal_trajectory, naive_trajectory, system.name),
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
    )


def _plot_tracking_errors(result: ErrorExperimentResult, system_name: str, environment_name: str):
    control_indices = np.arange(1, len(result.naive_tracking_error) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(control_indices, result.naive_tracking_error, marker="o", label="Naive")
    plt.plot(control_indices, result.optimized_tracking_error, marker="s", label="Optimized")
    plt.xlabel("Number of controls")
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
    control_indices = np.arange(1, len(naive_mean_by_step) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(control_indices, naive_mean_by_step, marker="o", label="Naive (avg)")
    plt.plot(control_indices, optimized_mean_by_step, marker="s", label="Optimized (avg)")
    plt.xlabel("Number of controls")
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
    parser.add_argument("--num-controls", type=int, default=5, help="Number of random controls.")
    parser.add_argument("--num-trials", type=int, default=5, help="Number of experiment trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--duration", type=float, default=None, help="Duration per control segment."
    )
    args = parser.parse_args()

    shared_simulator = None
    shared_optimized_simulator = None
    if args.environment_name == "mujoco":
        simulator_config = _simulator_config(
            args.system_name, args.environment_name, args.duration
        )
        shared_simulator = create_simulator(
            args.system_name, args.environment_name, config=simulator_config
        )
        shared_optimized_simulator = shared_simulator

    all_results: list[ErrorExperimentResult] = []
    for trial_idx in range(args.num_trials):
        trial_seed = args.seed + trial_idx
        print(f"[INFO] Running trial {trial_idx + 1}/{args.num_trials} (seed={trial_seed})")
        result = run_error_experiment(
            system_name=args.system_name,
            environment_name=args.environment_name,
            num_controls=args.num_controls,
            seed=trial_seed,
            duration=args.duration,
            simulator=shared_simulator,
            optimized_simulator=shared_optimized_simulator,
        )
        all_results.append(result)

    naive_errors = np.asarray([res.naive_tracking_error for res in all_results], dtype=float)
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
    print(f"[INFO] Trials: {args.num_trials}")
    print(f"[INFO] Average naive tracking error over all trials: {naive_mean:.6f}")
    print(f"[INFO] Average optimized tracking error over all trials: {optimized_mean:.6f}")
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
