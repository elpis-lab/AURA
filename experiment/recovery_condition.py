#!/usr/bin/env python3
"""Rerun the Proposition 2 conditional recovery experiment.

For every executed step, the experiment first measures
``e_execution = d(x_executed, Gamma(x_current, u_optimized))``.  Only steps
with ``e_execution <= delta`` enter the reported denominator.  A qualifying
step satisfies the Proposition 2 certificate when
``e_optimized + e_execution <= delta``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from aura.optimization import optimize_controls, optimizer_device_info
from experiment.deviation_error import (
    build_simulator_config,
    close_simulator,
)
from simulation.simulator import create_simulator
from systems import get_system
from utils.control_duration import ControlEdge
from utils.deviation import (
    canonicalize_state,
    create_aura_picker,
    generate_valid_reference,
    load_pushing_optimizer,
)
from utils.experiment_io import write_csv, write_json
from utils.utils import arrayDistance


DEFAULT_SOURCE = ROOT / "results/error_experiment/proposition2_step_metrics.csv"
DEFAULT_CALIBRATION_ROOT = ROOT / "results/error_experiment"
NUMERICAL_TOLERANCE = 1.0e-9
NUM_CONTROLS = 10
NUM_TRIALS = 5
BASE_SEED = 20_260_824

CONDITIONS = (
    {
        "system": "double_integrator",
        "environment": "gaussian",
        "condition": "Double Integrator — Gaussian Noise",
        "duration": 0.1,
        "aura": {
            "batch_size": 5_000,
            "position_std": 0.01,
            "rotation_std": 0.0,
            "velocity_std": 0.003,
            "gradient_iterations": 25,
            "learning_rate": 0.05,
        },
    },
    {
        "system": "kinematic_car",
        "environment": "gaussian",
        "condition": "Kinematic Car — Gaussian Noise",
        "duration": 1.0,
        "aura": {
            "batch_size": 60_000,
            "position_std": 0.2,
            "rotation_std": 0.8,
            "velocity_std": 0.009,
            "gradient_iterations": 400,
            "learning_rate": 0.01,
        },
    },
    {
        "system": "pushing_object",
        "environment": "gaussian",
        "condition": "Pushing Dynamics — Gaussian Noise",
        "duration": 0.1,
        "aura": {
            "batch_size": 10_000,
            "position_std": 0.035,
            "rotation_std": 0.35,
            "velocity_std": 0.009,
            "gradient_iterations": 1_000,
            "learning_rate": 5.0e-4,
        },
    },
    {
        "system": "kinematic_car",
        "environment": "mujoco",
        "condition": "Kinematic Car — MuJoCo",
        "duration": 1.0,
        "simulator_overrides": {
            "mujoco_car_throttle_ctrl_scale": 0.08,
            "mujoco_car_steering_ctrl_scale": 1.1,
        },
        "aura": {
            "batch_size": 30_000,
            "position_std": 0.04,
            "rotation_std": 0.25,
            "velocity_std": 0.003,
            "gradient_iterations": 100,
            "learning_rate": 0.03,
        },
    },
    {
        "system": "pushing_object",
        "environment": "mujoco",
        "condition": "Pushing Dynamics — MuJoCo",
        "duration": 2.0,
        "aura": {
            "batch_size": 30_000,
            "position_std": 0.05,
            "rotation_std": 0.4,
            "velocity_std": 0.003,
            "gradient_iterations": 1_000,
            "learning_rate": 5.0e-4,
        },
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun all five simulation conditions for Proposition 2."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=DEFAULT_CALIBRATION_ROOT,
        help="completed deviation experiment used only to calibrate Delta",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--condition",
        choices=tuple(
            f"{spec['system']}__{spec['environment']}" for spec in CONDITIONS
        ),
        help="rerun one condition and preserve the other saved condition rows",
    )
    return parser.parse_args()


def json_vector(value: np.ndarray) -> str:
    return json.dumps(
        np.asarray(value, dtype=float).reshape(-1).tolist(), separators=(",", ":")
    )


def state_distance(system_name: str, first: np.ndarray, second: np.ndarray) -> float:
    return float(arrayDistance(first, second, system=system_name))


def calibrated_spec(
    spec: dict[str, Any], calibration_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze independent optimization and execution budgets from prior data."""

    system_name = str(spec["system"])
    environment = str(spec["environment"])
    condition_key = f"{system_name}__{environment}"
    condition_dir = calibration_root / f"{system_name}_{environment}"
    reference_path = condition_dir / "reference_trajectories.jsonl"
    metrics_path = condition_dir / "tracking_step_metrics.csv"
    config_path = calibration_root / "config.json"
    for path in (reference_path, metrics_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing Delta calibration source: {path}")

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    condition_config = saved_config.get("conditions", {}).get(condition_key)
    if not isinstance(condition_config, dict):
        raise ValueError(f"calibration config is missing {condition_key}")
    if not math.isclose(
        float(condition_config["control_duration"]),
        float(spec["duration"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"calibration duration mismatch for {condition_key}")
    for name, expected in spec["aura"].items():
        actual = condition_config["aura"].get(name)
        if not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"calibration AURA setting mismatch for {condition_key}: "
                f"{name}={actual}, expected {expected}"
            )
    saved_simulator = condition_config.get("simulator_config", {})
    for name, expected in spec.get("simulator_overrides", {}).items():
        actual = saved_simulator.get(name)
        if not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"calibration simulator setting mismatch for {condition_key}: "
                f"{name}={actual}, expected {expected}"
            )

    references: dict[int, dict[str, Any]] = {}
    with reference_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                reference = json.loads(line)
                references[int(reference["trial"])] = reference

    rows_by_trial: dict[int, list[dict[str, str]]] = {}
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("method") == "aura":
                rows_by_trial.setdefault(int(row["trial"]), []).append(row)
    if set(rows_by_trial) != set(references):
        raise ValueError(f"calibration trial mismatch for {condition_key}")

    system = get_system(system_name)
    duration = float(spec["duration"])
    system.configure_duration_contract(duration, 1, 1)
    max_optimized = 0.0
    max_execution = 0.0
    calibration_steps = 0
    for trial, rows in rows_by_trial.items():
        rows.sort(key=lambda row: int(row["step"]))
        if len(rows) != NUM_CONTROLS:
            raise ValueError(
                f"calibration trial {trial} for {condition_key} has "
                f"{len(rows)} steps, expected {NUM_CONTROLS}"
            )
        current = np.asarray(references[trial]["initial_state"], dtype=float)
        for row in rows:
            target = np.asarray(json.loads(row["x_reference"]), dtype=float)
            control = np.asarray(json.loads(row["u_executed"]), dtype=float)
            executed = np.asarray(json.loads(row["x_executed"]), dtype=float)
            predicted = np.asarray(
                system.propagate(current, control, duration), dtype=float
            )
            max_optimized = max(
                max_optimized,
                state_distance(system_name, predicted, target),
            )
            max_execution = max(
                max_execution,
                state_distance(system_name, executed, predicted),
            )
            current = executed
            calibration_steps += 1

    if calibration_steps != len(references) * NUM_CONTROLS:
        raise RuntimeError(f"incomplete Delta calibration for {condition_key}")
    delta = max_optimized + max_execution
    if not math.isfinite(delta) or delta <= 0.0:
        raise RuntimeError(f"invalid calibrated Delta for {condition_key}: {delta}")

    calibrated = dict(spec)
    calibrated.update(
        {
            "delta": delta,
            "delta_source": (
                f"sum of separately calibrated maxima from {calibration_steps} "
                "AURA steps in the completed 100-trial deviation experiment; "
                f"Delta_opt={max_optimized:.15g}, "
                f"Delta_exec={max_execution:.15g}"
            ),
            "delta_kind": "independently calibrated empirical recovery-tube radius",
            "delta_is_theoretical_bound": False,
        }
    )
    calibration = {
        "system": system_name,
        "environment": environment,
        "calibration_trials": len(references),
        "calibration_steps": calibration_steps,
        "delta_optimization": max_optimized,
        "delta_execution": max_execution,
        "delta": delta,
        "reference_source": str(reference_path.resolve()),
        "metrics_source": str(metrics_path.resolve()),
        "evaluation_seed_base": BASE_SEED,
        "calibration_and_evaluation_seeds_disjoint": True,
    }
    return calibrated, calibration


def optimizer_seed(trial_seed: int, step: int) -> int:
    sequence = np.random.SeedSequence([int(trial_seed), int(step), 0xA0A2])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def seed_optimizer(seed: int) -> None:
    normalized = int(seed) % (2**32 - 1)
    np.random.seed(normalized)
    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)


def build_condition(spec: dict[str, Any], device: str) -> tuple[Any, dict[str, Any]]:
    system_name = str(spec["system"])
    environment = str(spec["environment"])
    duration = float(spec["duration"])
    system = get_system(system_name)
    simulator_config = build_simulator_config(
        system_name, environment, duration=duration
    )
    simulator_config.update(
        {
            "propagation_step_size": duration,
            "min_control_duration": 1,
            "max_control_duration": 1,
            "state_bounds": [list(bound) for bound in system.state_bounds],
            "headless": True,
        }
    )
    simulator_config.update(spec.get("simulator_overrides", {}))
    system.configure_duration_contract(duration, 1, 1)
    # These are the exact condition-specific settings recorded in the completed
    # deviation experiment.  Do not silently replace them with generic defaults:
    # the car experiments in particular use a longer edge duration and a larger
    # optimization budget.
    aura = dict(spec["aura"])
    config = {
        "system": system_name,
        "environment": environment,
        "control_duration": duration,
        "state_bounds": [list(bound) for bound in system.state_bounds],
        "simulator_config": simulator_config,
        "obstacles": {"enabled": False},
        "resolved_device": device,
        "aura": aura,
    }
    if system_name == "pushing_object" and environment == "mujoco":
        config["initial_state_bounds"] = [[-0.12, 0.76], [-0.82, -0.34]]
    return system, config


def make_reference(system: Any, config: dict[str, Any], trial_seed: int):
    rng = np.random.default_rng(trial_seed)
    return generate_valid_reference(
        system,
        NUM_CONTROLS,
        rng,
        config,
        boundary_margin_fraction=0.20,
    )


def calculate_metrics(
    system_name: str,
    target: np.ndarray,
    predicted_nominal: np.ndarray,
    predicted_optimized: np.ndarray,
    executed: np.ndarray,
    delta: float,
) -> dict[str, Any]:
    e_nominal = state_distance(system_name, predicted_nominal, target)
    e_optimized = state_distance(system_name, predicted_optimized, target)
    e_execution = state_distance(system_name, executed, predicted_optimized)
    e_final = state_distance(system_name, executed, target)
    certificate_margin = delta - (e_optimized + e_execution)
    tube_margin = delta - e_final
    lower = e_optimized / delta
    upper = 1.0 - e_execution / delta
    intersection_lower = max(0.0, lower)
    intersection_upper = min(1.0, upper)
    return {
        "e_nominal": e_nominal,
        "e_optimized": e_optimized,
        "e_execution": e_execution,
        "e_final": e_final,
        "relative_improvement": (
            (e_nominal - e_optimized) / max(e_nominal, 1.0e-12)
        ),
        "improved": e_optimized < e_nominal,
        "certificate_satisfied": certificate_margin >= -NUMERICAL_TOLERANCE,
        "tube_retained": tube_margin >= -NUMERICAL_TOLERANCE,
        "lambda_lower": lower,
        "lambda_upper": upper,
        "lambda_intersection_lower": intersection_lower,
        "lambda_intersection_upper": intersection_upper,
        "lambda_interval_valid": (
            intersection_lower <= intersection_upper + NUMERICAL_TOLERANCE
            and intersection_upper > NUMERICAL_TOLERANCE
            and intersection_lower < 1.0 - NUMERICAL_TOLERANCE
        ),
        "certificate_margin": certificate_margin,
        "tube_margin": tube_margin,
        "triangle_upper_bound": e_optimized + e_execution,
        "triangle_slack": e_optimized + e_execution - e_final,
        "triangle_inequality_satisfied": (
            e_final <= e_optimized + e_execution + NUMERICAL_TOLERANCE
        ),
    }


def run_trial(
    spec: dict[str, Any],
    trial: int,
    condition_index: int,
    device: str,
    system: Any,
    config: dict[str, Any],
    optimization_model: Any,
) -> list[dict[str, Any]]:
    trial_seed = BASE_SEED + 1_000_003 * condition_index + 10_007 * trial
    duration = float(config["control_duration"])
    reference = make_reference(system, config, trial_seed)
    plant_config = dict(config["simulator_config"])
    plant_config["start_state"] = reference.initial_state.tolist()
    plant_config["disturbance_seed"] = trial_seed + 10_000

    plant = None
    rows: list[dict[str, Any]] = []
    try:
        plant = create_simulator(
            str(spec["system"]), str(spec["environment"]), config=plant_config
        )
        plant.reset()
        plant.set_state(reference.initial_state.tolist())
        picker = create_aura_picker(system, duration, plant_config)
        picker.opt_model = optimization_model
        aura = config["aura"]

        for step, (nominal_control, target) in enumerate(
            zip(reference.controls, reference.states[1:])
        ):
            started = time.perf_counter()
            current = canonicalize_state(system.name, plant.get_state())
            seed = optimizer_seed(trial_seed, step)
            edge = ControlEdge(
                source_state=current,
                target_state=target,
                control=nominal_control,
                duration_steps=1,
                duration_seconds=duration,
                edge_id=f"recovery:{trial}:{step}",
            )
            seed_optimizer(seed)
            optimization_started = time.perf_counter()
            optimization = optimize_controls(
                system=system,
                next_state=current,
                child_edges=[edge],
                integration_step_size=duration,
                model=optimization_model,
                num_states=int(aura["batch_size"]),
                position_std=float(aura["position_std"]),
                rotation_std=float(aura["rotation_std"]),
                velocity_std=float(aura["velocity_std"]),
                num_steps=int(aura["gradient_iterations"]),
                learning_rate=float(aura["learning_rate"]),
                requested_device=device,
            )
            optimization_runtime = time.perf_counter() - optimization_started
            selection = picker.pick_next_control(
                system=system,
                optimization_result=optimization,
                current_state=current,
                next_state=target,
                children_edges=[edge],
                fallback_edge=edge,
            )
            selected_control = np.asarray(selection.control, dtype=float).reshape(-1)
            predicted_nominal = np.asarray(
                system.propagate(current, nominal_control, duration), dtype=float
            )
            predicted_optimized = np.asarray(
                system.propagate(current, selected_control, duration), dtype=float
            )
            execution_started = time.perf_counter()
            executed = canonicalize_state(
                system.name, plant.execute_segment(selected_control, duration)
            )
            execution_runtime = time.perf_counter() - execution_started
            metrics = calculate_metrics(
                system.name,
                target,
                predicted_nominal,
                predicted_optimized,
                executed,
                float(spec["delta"]),
            )
            decision = dict(picker.last_control_decision or {})
            rows.append(
                {
                    "system": system.name,
                    "environment": spec["environment"],
                    "condition": spec["condition"],
                    "trial": trial,
                    "step": step,
                    "seed": trial_seed,
                    "optimizer_seed": seed,
                    "delta": spec["delta"],
                    "delta_source": spec["delta_source"],
                    "delta_kind": spec["delta_kind"],
                    "delta_is_theoretical_bound": spec[
                        "delta_is_theoretical_bound"
                    ],
                    "x_current": json_vector(current),
                    "x_target": json_vector(target),
                    "u_nominal": json_vector(nominal_control),
                    "u_optimized": json_vector(selected_control),
                    "x_pred_nominal": json_vector(predicted_nominal),
                    "x_pred_optimized": json_vector(predicted_optimized),
                    "x_executed": json_vector(executed),
                    **metrics,
                    "optimizer_iterations": optimization.get(
                        "steps_completed", aura["gradient_iterations"]
                    ),
                    "optimizer_iterations_requested": aura["gradient_iterations"],
                    "optimizer_learning_rate": aura["learning_rate"],
                    "optimizer_num_states": aura["batch_size"],
                    "optimizer_num_states_effective": optimization.get(
                        "effective_num_states", aura["batch_size"]
                    ),
                    "optimizer_device": optimization.get("device", device),
                    "optimizer_selection_source": decision.get(
                        "source", selection.source
                    ),
                    "optimizer_selection_reason": decision.get("reason", ""),
                    "optimizer_hyperparameters_inferred": False,
                    "propagation_duration": duration,
                    "naive_tracking_error": "",
                    "optimization_runtime_seconds": optimization_runtime,
                    "execution_runtime_seconds": execution_runtime,
                    "runtime_seconds": time.perf_counter() - started,
                    "data_source": "experiment/recovery_condition.py full rerun",
                }
            )
        return rows
    finally:
        close_simulator(plant)


def read_source_fields(source: Path) -> list[str]:
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
    if not fields:
        raise RuntimeError(f"source has no CSV header: {source}")
    return fields


def run_all_conditions(
    source: Path,
    device: str,
    calibration_root: Path,
    selected_condition: str | None = None,
) -> None:
    fields = read_source_fields(source)
    selected_uncalibrated = [
        (index, spec)
        for index, spec in enumerate(CONDITIONS)
        if selected_condition is None
        or f"{spec['system']}__{spec['environment']}" == selected_condition
    ]
    if not selected_uncalibrated:
        raise ValueError(f"unknown recovery condition: {selected_condition}")
    selected_specs: list[tuple[int, dict[str, Any]]] = []
    calibrations: list[dict[str, Any]] = []
    for index, spec in selected_uncalibrated:
        frozen_spec, calibration = calibrated_spec(spec, calibration_root)
        selected_specs.append((index, frozen_spec))
        calibrations.append(calibration)

    all_rows: list[dict[str, Any]] = []
    if selected_condition is not None:
        selected_keys = {
            (str(spec["system"]), str(spec["environment"]))
            for _, spec in selected_specs
        }
        with source.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
        all_rows.extend(
            row
            for row in existing
            if (str(row["system"]), str(row["environment"])) not in selected_keys
        )
    for condition_index, spec in selected_specs:
        system, config = build_condition(spec, device)
        aura = config["aura"]
        optimization_model = (
            load_pushing_optimizer(
                system,
                float(aura["learning_rate"]),
                int(aura["gradient_iterations"]),
            )
            if system.name == "pushing_object"
            else None
        )
        for trial in range(NUM_TRIALS):
            rows = run_trial(
                spec,
                trial,
                condition_index,
                device,
                system,
                config,
                optimization_model,
            )
            eligible = [
                row
                for row in rows
                if float(row["e_execution"])
                <= float(row["delta"]) + NUMERICAL_TOLERANCE
            ]
            successes = [
                row
                for row in eligible
                if float(row["e_optimized"]) + float(row["e_execution"])
                <= float(row["delta"]) + NUMERICAL_TOLERANCE
            ]
            print(
                f"{spec['condition']}, trial {trial + 1}/{NUM_TRIALS}: "
                f"eligible={len(eligible)}/{len(rows)}, "
                f"certificate={len(successes)}/{len(eligible)}"
            )
            all_rows.extend(rows)

    expected = len(CONDITIONS) * NUM_TRIALS * NUM_CONTROLS
    if len(all_rows) != expected:
        raise RuntimeError(f"expected {expected} consolidated rows, got {len(all_rows)}")

    normalized = [{field: row.get(field, "") for field in fields} for row in all_rows]
    normalized.sort(
        key=lambda row: (
            str(row["system"]),
            str(row["environment"]),
            str(row["trial"]),
            int(row["step"]),
        )
    )
    write_csv(source, normalized, fieldnames=fields)
    calibration_path = source.parent / "proposition2_delta_calibration.json"
    if selected_condition is None or not calibration_path.is_file():
        calibration_payload = calibrations
    else:
        existing_calibrations = json.loads(
            calibration_path.read_text(encoding="utf-8")
        )
        selected_keys = {
            (item["system"], item["environment"]) for item in calibrations
        }
        calibration_payload = [
            item
            for item in existing_calibrations
            if (item["system"], item["environment"]) not in selected_keys
        ] + calibrations
    calibration_payload.sort(key=lambda item: (item["system"], item["environment"]))
    write_json(calibration_path, calibration_payload)
    print(f"Saved {len(normalized)} newly executed simulation steps to {source}")
    print(f"Saved frozen Delta calibration to {calibration_path}")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Proposition 2 source not found: {source}")
    device_info = optimizer_device_info(None if args.device == "auto" else args.device)
    device = str(device_info["device"])
    if args.device != "auto" and device != str(torch.device(args.device)):
        raise RuntimeError(
            f"requested device {args.device!r} is unavailable: "
            f"{device_info.get('error', '')}"
        )
    run_all_conditions(
        source,
        device,
        args.calibration_root.resolve(),
        args.condition,
    )


if __name__ == "__main__":
    main()
