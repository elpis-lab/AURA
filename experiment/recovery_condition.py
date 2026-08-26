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
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import yaml
from utils.experiment_io import write_csv, write_json


DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "recovery_condition.yaml"
SYSTEM_CONFIG_DIR = ROOT / "configs" / "systems"
NUMERICAL_TOLERANCE = 1.0e-9
METRIC_TOLERANCE = 1.0e-12
NUM_CONTROLS = 10
NUM_TRIALS = 5
BASE_SEED = 20_260_824

def load_conditions() -> tuple[dict[str, Any], ...]:
    experiment = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
    conditions = []
    for system_name, environment in (
        ("double_integrator", "gaussian"),
        ("kinematic_car", "gaussian"),
        ("pushing_object", "gaussian"),
        ("kinematic_car", "mujoco"),
        ("pushing_object", "mujoco"),
    ):
        path = SYSTEM_CONFIG_DIR / f"{system_name}.yaml"
        system = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        settings = experiment["condition_hyperparameters"][
            f"{system_name}_{environment}"
        ]
        environment_config = system["environments"][environment]
        spec = {
            "system": system_name,
            "environment": environment,
            "condition": f"{system['title']} — {environment_config['panel_subtitle']}",
            "duration": float(settings["control_duration"]),
            "aura": {
                key: value for key, value in settings.items() if key != "control_duration"
            },
        }
        simulator_overrides = {
            key: value
            for key, value in environment_config.items()
            if key.startswith("mujoco_")
        }
        if simulator_overrides:
            spec["simulator_overrides"] = simulator_overrides
        conditions.append(spec)
    return tuple(conditions)


CONDITIONS = load_conditions()

CONDITION_ORDER = tuple(
    (str(spec["system"]), str(spec["environment"])) for spec in CONDITIONS
)
REQUIRED_SUMMARY_COLUMNS = {
    "system",
    "environment",
    "condition",
    "trial",
    "step",
    "delta",
    "delta_source",
    "delta_kind",
    "x_target",
    "x_pred_nominal",
    "x_pred_optimized",
    "x_executed",
    "e_nominal",
    "e_optimized",
    "e_execution",
    "e_final",
}
SUMMARY_COLUMNS = (
    "system_environment",
    "certificate_given_execution_bound",
)


def parse_args() -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = preliminary.parse_known_args()
    defaults = yaml.safe_load(known.config.read_text(encoding="utf-8")) or {}
    parser = argparse.ArgumentParser(
        description=(
            "Run or summarize the five-condition Proposition 2 recovery experiment."
        )
    )
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument("--source", type=Path, default=ROOT / defaults["source"])
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / defaults["output_dir"]
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="audit and summarize the saved measurements without rerunning them",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=ROOT / defaults["calibration_root"],
        help="completed deviation experiment used only to calibrate Delta",
    )
    parser.add_argument("--device", default=str(defaults["device"]))
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
    first = np.asarray(first, dtype=float).reshape(-1)
    second = np.asarray(second, dtype=float).reshape(-1)
    if system_name == "double_integrator":
        if len(first) < 6 or len(second) < 6:
            raise ValueError("double-integrator states must have six components")
        return float(np.linalg.norm(first[:6] - second[:6]))
    if system_name in {"kinematic_car", "pushing_object"}:
        if len(first) < 3 or len(second) < 3:
            raise ValueError("SE(2) states must have three components")
        position = float(np.linalg.norm(first[:2] - second[:2]))
        yaw = float(abs((first[2] - second[2] + np.pi) % (2.0 * np.pi) - np.pi))
        return position + 0.5 * yaw
    raise ValueError(f"unsupported recovery-condition system: {system_name}")


def calibrated_spec(
    spec: dict[str, Any], calibration_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze independent optimization and execution budgets from prior data."""

    from propagators import get_system

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
    import torch

    normalized = int(seed) % (2**32 - 1)
    np.random.seed(normalized)
    torch.manual_seed(normalized)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)


def build_condition(spec: dict[str, Any], device: str) -> tuple[Any, dict[str, Any]]:
    from experiment.deviation_error import build_simulator_config
    from propagators import get_system

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
    from utils.deviation import generate_valid_reference

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
    from aura.optimization import optimize_controls
    from experiment.deviation_error import close_simulator
    from simulation.simulator import create_simulator
    from methods.plan import ControlEdge
    from utils.deviation import canonicalize_state, create_aura_picker

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
    from utils.deviation import load_pushing_optimizer

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


def parse_saved_vector(value: str) -> np.ndarray:
    vector = json.loads(value)
    if not isinstance(vector, list) or not vector:
        raise ValueError(f"invalid state vector: {value!r}")
    parsed = np.asarray(vector, dtype=float).reshape(-1)
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"non-finite state vector: {value!r}")
    return parsed


def read_and_audit(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load saved step metrics and independently verify every reported error."""

    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing_columns = REQUIRED_SUMMARY_COLUMNS.difference(
            reader.fieldnames or ()
        )
        if missing_columns:
            raise ValueError(f"source is missing columns: {sorted(missing_columns)}")
        raw_rows = list(reader)

    evaluated: list[dict[str, Any]] = []
    invalid_records: list[str] = []
    maximum_metric_difference = 0.0
    delta_by_condition: dict[tuple[str, str], set[float]] = defaultdict(set)
    rows_by_trial: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    metric_vectors = {
        "e_nominal": ("x_pred_nominal", "x_target"),
        "e_optimized": ("x_pred_optimized", "x_target"),
        "e_execution": ("x_executed", "x_pred_optimized"),
        "e_final": ("x_executed", "x_target"),
    }

    for line_number, raw in enumerate(raw_rows, start=2):
        identity = (
            f"line {line_number} ({raw.get('condition', '?')}, "
            f"trial={raw.get('trial', '?')}, step={raw.get('step', '?')})"
        )
        try:
            system_name = str(raw["system"])
            environment = str(raw["environment"])
            delta = float(raw["delta"])
            if not math.isfinite(delta) or delta <= 0.0:
                raise ValueError(f"invalid delta {delta}")
            metrics = {
                name: float(raw[name])
                for name in ("e_nominal", "e_optimized", "e_execution", "e_final")
            }
            if not all(
                math.isfinite(value) and value >= 0.0 for value in metrics.values()
            ):
                raise ValueError("one or more error metrics are invalid")
            vectors = {
                name: parse_saved_vector(raw[name])
                for name in {item for pair in metric_vectors.values() for item in pair}
            }
            for metric_name, (first_name, second_name) in metric_vectors.items():
                recomputed = state_distance(
                    system_name, vectors[first_name], vectors[second_name]
                )
                difference = abs(recomputed - metrics[metric_name])
                maximum_metric_difference = max(maximum_metric_difference, difference)
                if difference > METRIC_TOLERANCE:
                    raise ValueError(
                        f"{metric_name} differs from the saved metric by "
                        f"{difference:.3g}"
                    )
            trial = str(raw["trial"])
            step = int(raw["step"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            invalid_records.append(f"{identity}: {error}")
            continue

        row = dict(raw)
        row.update(metrics)
        row["delta"] = delta
        row["execution_bound_satisfied"] = (
            metrics["e_execution"] <= delta + NUMERICAL_TOLERANCE
        )
        row["certificate_satisfied"] = (
            row["execution_bound_satisfied"]
            and metrics["e_optimized"] + metrics["e_execution"]
            <= delta + NUMERICAL_TOLERANCE
        )
        evaluated.append(row)
        key = (system_name, environment)
        delta_by_condition[key].add(delta)
        rows_by_trial[(system_name, environment, trial)].append(step)

    inconsistent_deltas = {
        key: sorted(values)
        for key, values in delta_by_condition.items()
        if len(values) != 1
    }
    if inconsistent_deltas:
        raise ValueError(f"inconsistent delta within condition: {inconsistent_deltas}")

    incomplete_trials = []
    for (system_name, environment, trial), steps in rows_by_trial.items():
        expected_steps = list(range(NUM_CONTROLS))
        actual_steps = sorted(steps)
        if actual_steps != expected_steps:
            incomplete_trials.append(
                f"{system_name}/{environment} trial {trial}: expected "
                f"{expected_steps}, found {actual_steps}"
            )
    if invalid_records or incomplete_trials:
        details = "\n".join(invalid_records + incomplete_trials)
        raise ValueError(f"source audit failed:\n{details}")

    return evaluated, {
        "source_records": len(raw_rows),
        "evaluated_steps": len(evaluated),
        "invalid_records": invalid_records,
        "incomplete_trials": incomplete_trials,
        "maximum_metric_recalculation_difference": maximum_metric_difference,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else math.nan


def count_text(numerator: int, denominator: int) -> str:
    return (
        f"{numerator} / {denominator} ({percentage(numerator, denominator):.1f}%)"
        if denominator
        else "N/A"
    )


def summarize_rows(
    rows: list[dict[str, Any]], *, combined: bool = False
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty condition")
    eligible = [row for row in rows if row["execution_bound_satisfied"]]
    if not eligible:
        raise ValueError("no steps satisfy the execution-error assumption")
    successful = [row for row in eligible if row["certificate_satisfied"]]
    trials = {
        (row["system"], row["environment"], str(row["trial"])) for row in rows
    }
    first = rows[0]
    return {
        "system_environment": (
            "All evaluated conditions" if combined else first["condition"]
        ),
        "system": "all" if combined else first["system"],
        "environment": "all" if combined else first["environment"],
        "trials": len(trials),
        "evaluated_steps": len(rows),
        "eligible_steps": len(eligible),
        "excluded_steps": len(rows) - len(eligible),
        "certificate_steps": len(successful),
        "certificate_pct": percentage(len(successful), len(eligible)),
        "certificate_given_execution_bound": count_text(
            len(successful), len(eligible)
        ),
        "delta": "condition-specific" if combined else first["delta"],
        "delta_source": "See source rows" if combined else first["delta_source"],
        "delta_kind": "mixed" if combined else first["delta_kind"],
    }


def build_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["environment"])].append(row)
    unexpected = set(grouped).difference(CONDITION_ORDER)
    missing = set(CONDITION_ORDER).difference(grouped)
    if unexpected or missing:
        raise ValueError(
            f"condition mismatch; unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )
    summaries = [summarize_rows(grouped[key]) for key in CONDITION_ORDER]
    summaries.append(summarize_rows(rows, combined=True))
    return summaries


def latex_label(label: str) -> str:
    return label.replace("—", "--").replace(
        "All evaluated conditions", "All conditions"
    )


def write_latex(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Empirical evaluation of the sufficient approximate-recovery condition in Proposition~2. A step is evaluated only when $e_{\mathrm{exec}}\leq\Delta$, and the certificate is satisfied when $e_{\mathrm{opt}}+e_{\mathrm{exec}}\leq\Delta$.}",
        r"\label{tab:recovery-condition-validation}",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"System / Environment & $e_{\mathrm{opt}}+e_{\mathrm{exec}}\leq\Delta$ \\",
        r"\midrule",
    ]
    for index, summary in enumerate(summaries):
        if index == len(summaries) - 1:
            lines.append(r"\midrule")
        value = summary["certificate_given_execution_bound"].replace("%", r"\%")
        lines.append(
            "{} & {} \\\\".format(
                latex_label(str(summary["system_environment"])), value
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def interpretation(summary: dict[str, Any]) -> str:
    return (
        f"For {summary['system_environment']}, {summary['eligible_steps']}/"
        f"{summary['evaluated_steps']} steps satisfied e_exec <= Delta. Among "
        f"those steps, e_opt + e_exec <= Delta in "
        f"{summary['certificate_steps']}/{summary['eligible_steps']} steps "
        f"({summary['certificate_pct']:.1f}%)."
    )


def write_text_summary(
    path: Path,
    source: Path,
    summaries: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    lines = [
        "Proposition 2 conditional empirical evaluation",
        "",
        "Reported test",
        "- First evaluate e_exec = d(x_executed, Gamma(x_current, u_optimized)).",
        "- Exclude steps for which e_exec > Delta.",
        "- On the remaining steps, test e_opt + e_exec <= Delta.",
        "",
        "Source data",
        f"- Working-tree source: {source.resolve()}",
        "- All five simulation conditions were run by experiment/recovery_condition.py.",
        f"- SHA-256: {audit['source_sha256']}",
        "- No real-world execution is included.",
        "",
        "Consistency audit",
        f"- Source records: {audit['source_records']}",
        f"- Evaluated execution steps: {audit['evaluated_steps']}",
        f"- Distinct trials/trajectories: {summaries[-1]['trials']}",
        f"- Invalid records: {len(audit['invalid_records'])}",
        f"- Incomplete trial sequences: {len(audit['incomplete_trials'])}",
        "- Counts are per execution step, not per trial.",
        f"- Maximum error-metric recalculation difference: {audit['maximum_metric_recalculation_difference']:.3g}",
        "",
        "Delta definitions",
    ]
    for summary in summaries[:-1]:
        lines.append(
            f"- {summary['system_environment']}: "
            f"Delta={float(summary['delta']):.15g}; {summary['delta_kind']}; "
            f"{summary['delta_source']}."
        )
    lines.extend(["", "Results"])
    lines.extend(f"- {interpretation(summary)}" for summary in summaries)
    lines.extend(
        [
            "",
            "Important limitation",
            "Each Delta is an empirical recovery-tube radius frozen from a disjoint calibration set, not a universal mathematical disturbance bound.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_results(source: Path, output_dir: Path) -> None:
    """Audit saved measurements and write the paper-ready recovery summaries."""

    rows, audit = read_and_audit(source)
    summaries = build_summaries(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "proposition2_summary.csv"
    latex_path = output_dir / "proposition2_table.tex"
    text_path = output_dir / "proposition2_summary.txt"
    write_csv(
        csv_path,
        ({field: summary[field] for field in SUMMARY_COLUMNS} for summary in summaries),
        fieldnames=SUMMARY_COLUMNS,
    )
    write_latex(latex_path, summaries)
    write_text_summary(text_path, source, summaries, audit)
    for summary in summaries:
        print(interpretation(summary))
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")
    print(f"Summary: {text_path}")


def main() -> None:
    global NUM_TRIALS, NUM_CONTROLS, BASE_SEED

    args = parse_args()
    experiment_config = yaml.safe_load(
        args.config.resolve().read_text(encoding="utf-8")
    ) or {}
    NUM_TRIALS = int(experiment_config["num_trials"])
    NUM_CONTROLS = int(experiment_config["num_controls"])
    BASE_SEED = int(experiment_config["base_seed"])
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Proposition 2 source not found: {source}")
    if args.summarize_only:
        summarize_results(source, args.output_dir.resolve())
        return
    import torch

    from aura.optimization import optimizer_device_info

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
    summarize_results(source, args.output_dir.resolve())


if __name__ == "__main__":
    main()
