#!/usr/bin/env python3
"""Compare AURA-local, MPPI, and open-loop fixed-reference tracking."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aura_closed_loop_tracking_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import yaml

from methods.MPPI import default_parameters, parameters_from_config
from aura.optimization import optimizer_device_info
from utils.deviation import (
    AURALocalTrackingAdapter,
    METHOD_ORDER,
    MethodResult,
    make_mppi_controller,
    mppi_cost_weights,
    run_aura_tracking,
    run_mppi_tracking,
    run_open_loop,
)
from scripts.plot_deviation_error import plot_tracking_statistics
from utils.deviation import (
    compute_tracking_statistics,
    method_result_to_step_rows,
    method_result_to_trial_row,
    write_summary_tables,
)
from utils.deviation import (
    generate_valid_reference,
)
from simulation.simulator import create_simulator
from propagators import get_system
from utils.experiment_io import (
    read_csv,
    read_json_lines,
    write_csv,
    write_json,
    write_json_lines,
)


ALL_SYSTEMS = ("double_integrator", "kinematic_car", "pushing_object")
ALL_ENVIRONMENTS = ("gaussian", "mujoco")
STEP_FILE = "tracking_step_metrics.csv"
TRIAL_FILE = "trial_tracking_metrics.csv"
REFERENCE_FILE = "reference_trajectories.jsonl"
BOOTSTRAP_SEED = 91_337
DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "experiments" / "deviation_error.yaml"
)


def condition_output_dir(root: Path, system: str, environment: str) -> Path:
    return Path(root) / f"{system}_{environment}"


def build_simulator_config(
    system_name: str, environment: str, duration: float | None = None
) -> dict[str, Any]:
    system_path = REPO_ROOT / "configs" / "systems" / f"{system_name}.yaml"
    system_config = yaml.safe_load(system_path.read_text(encoding="utf-8")) or {}
    environment_config = system_config.get("environments", {}).get(environment, {})
    if duration is None and system_name == "pushing_object" and environment == "mujoco":
        duration = 2.0
    config: dict[str, Any] = {
        key: value
        for key, value in environment_config.items()
        if key.startswith("sampling_") or key.startswith("mujoco_") or key == "headless"
    }
    if duration is not None:
        config["propagation_step_size"] = float(duration)
    if system_name == "dubins_airplane":
        config["start_state"] = [0.1, 0.1, 0.15, 0.0, 0.0, 0.15]
    return config


def optimizer_sampling_stds(
    system_name: str, config: dict
) -> tuple[float, float, float]:
    sim_position = float(config.get("sampling_position_std", 0.0))
    sim_rotation = float(config.get("sampling_rotation_std", 0.0))
    sim_velocity = float(config.get("sampling_velocity_std", sim_position))

    def larger(value: float, simulator_value: float) -> float:
        return float(max(value, 1.2 * simulator_value)) if simulator_value > 0 else value

    if system_name == "double_integrator":
        return (
            larger(max(0.01, 3.0 * sim_position), sim_position),
            0.0,
            larger(max(0.003, 3.0 * sim_velocity), sim_velocity),
        )
    if system_name == "pushing_object":
        minimums = (0.035, 0.35, 0.003)
    elif system_name == "dubins_airplane":
        minimums = (0.01, 0.1, 0.01)
    else:
        minimums = (0.01, 0.1, 0.003)
    return (
        larger(max(minimums[0], 3.0 * sim_position), sim_position),
        larger(max(minimums[1], 2.0 * sim_rotation), sim_rotation),
        larger(max(minimums[2], 3.0 * sim_velocity), sim_velocity),
    )


def close_simulator(simulator) -> None:
    if simulator is None:
        return
    for method_name in ("close", "stop"):
        method = getattr(simulator, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
            return


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively apply explicit experiment overrides to a copied config."""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _load_condition_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("condition overrides must be a JSON object")
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"override {key!r} must be a JSON object")
        normalized[str(key)] = value
    return normalized


def _condition_with_overrides(
    system_name: str,
    environment: str,
    resolved_device: str,
    overrides: dict[str, dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    system, config = _condition_config(system_name, environment, resolved_device)
    selected = overrides.get(f"{system_name}__{environment}", {})
    _deep_update(config, selected)
    simulator_config = config["simulator_config"]
    system.configure_duration_contract(
        float(simulator_config["propagation_step_size"]),
        int(simulator_config["min_control_duration"]),
        int(simulator_config["max_control_duration"]),
    )
    return system, config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command_output(*command: str) -> str:
    try:
        return subprocess.check_output(
            command, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _git_metadata() -> dict[str, Any]:
    status = _command_output("git", "status", "--porcelain")
    return {
        "commit": _command_output("git", "rev-parse", "HEAD"),
        "branch": _command_output("git", "branch", "--show-current"),
        "dirty": bool(status and not status.startswith("<unavailable:")),
        "status_porcelain": status.splitlines() if status else [],
    }


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "torch", "matplotlib", "scipy", "PyYAML", "mujoco", "pytest")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _resolve_systems(requested: str) -> tuple[str, ...]:
    value = requested.lower()
    if value == "all":
        return ALL_SYSTEMS
    if value not in ALL_SYSTEMS:
        raise ValueError(f"unsupported system {requested!r}")
    return (value,)


def _resolve_environments(requested: str) -> tuple[str, ...]:
    value = requested.lower()
    if value == "all":
        return ALL_ENVIRONMENTS
    if value not in ALL_ENVIRONMENTS:
        raise ValueError(f"unsupported environment {requested!r}")
    return (value,)


def _resolve_device(requested: str) -> tuple[str, dict[str, Any]]:
    info = optimizer_device_info(None if requested == "auto" else requested)
    resolved = str(info["device"])
    if requested != "auto" and resolved != requested:
        raise RuntimeError(
            f"requested device {requested!r} is unavailable: {info.get('error', '')}"
        )
    return resolved, info


def _condition_config(
    system_name: str, environment: str, resolved_device: str
) -> tuple[Any, dict[str, Any]]:
    if environment == "mujoco" and system_name == "double_integrator":
        raise ValueError("double_integrator has no MuJoCo execution adapter")
    system_path = REPO_ROOT / "configs" / "systems" / f"{system_name}.yaml"
    system_config = yaml.safe_load(system_path.read_text(encoding="utf-8")) or {}
    environment_config = system_config["environments"][environment]
    system = get_system(system_name)
    system.set_state_bounds(
        environment_config.get("state_bounds", system_config["state_bounds"])
    )
    system.set_control_bounds(system_config["control_bounds"])
    simulator_config = build_simulator_config(system_name, environment)
    duration = float(simulator_config.get("propagation_step_size", 0.1))
    if system_name == "pushing_object" and environment == "mujoco":
        duration = 2.0
    simulator_config.update(
        {
            "propagation_step_size": duration,
            "min_control_duration": 1,
            "max_control_duration": 1,
            "state_bounds": [list(bound) for bound in system.state_bounds],
        }
    )
    system.configure_duration_contract(duration, 1, 1)
    pos_std, rot_std, vel_std = optimizer_sampling_stds(
        system_name, simulator_config
    )
    pushing = system_name == "pushing_object"
    aura = {
        "implementation": "aura.optimization.optimize_controls",
        "selection": "AURA.pick_next_control",
        "global_replanning_enabled": False,
        "batch_size": 10_000 if pushing else 5_000,
        "neighborhood_radius": 0.025,
        "neighborhood_radius_effective": False,
        "neighborhood_radius_note": (
            "The optimizer samples the Gaussian neighborhood without radius clipping"
        ),
        "position_std": float(pos_std),
        "rotation_std": float(rot_std),
        "velocity_std": float(vel_std),
        "learning_rate": 5.0e-4 if pushing else 0.05,
        "gradient_iterations": 1_000 if pushing else 25,
        "control_initialization": "nominal edge control",
        "projection": "aura.optimization.clamp_controls with native system bounds",
    }
    params = default_parameters(system_name)
    mppi = {
        **asdict(params),
        "implementation": "methods.MPPI.MPPIController via ReferenceTrackingMPPI adapter",
        "cost_weights": mppi_cost_weights(system_name),
        "warm_start": "initial nominal control window, then shifted optimized sequence",
        "reference_horizon": "min(horizon_steps, num_controls - step)",
        "optimization_iterations": 1,
        "nominal_deadband": -1.0,
    }
    if system_name in ("kinematic_car", "pushing_object"):
        distance = {
            "implementation": "utils.utils.arrayDistance / OMPL SE2StateSpace.distance",
            "translation_weight": 1.0,
            "wrapped_angular_weight": 0.5,
            "form": "Euclidean planar translation plus 0.5 times wrapped absolute yaw",
        }
    else:
        distance = {
            "implementation": "utils.utils.arrayDistance",
            "weights": [1.0] * 6,
            "form": "Euclidean R6 norm",
        }
    config: dict[str, Any] = {
        "system": system_name,
        "environment": environment,
        "control_duration": duration,
        "integration_timestep": duration if environment == "gaussian" else "simulator_internal",
        "nominal_integration": (
            "closed_form"
            if system_name in ("kinematic_car", "double_integrator")
            else "one learned primitive per propagation tick"
        ),
        "state_bounds": [list(bound) for bound in system.state_bounds],
        "control_bounds": [list(bound) for bound in system.control_bounds],
        "obstacles": {"enabled": False},
        "simulator_config": simulator_config,
        "process_noise": {
            "distribution": "standard Gaussian scaled component-wise",
            "bounded": False,
            "position_std": float(simulator_config.get("sampling_position_std", 0.0)),
            "rotation_std": float(simulator_config.get("sampling_rotation_std", 0.0)),
            "velocity_std": float(
                simulator_config.get(
                    "sampling_velocity_std",
                    simulator_config.get("sampling_position_std", 0.0),
                )
            ),
        },
        "state_distance": distance,
        "aura": aura,
        "mppi": mppi,
        "resolved_device": resolved_device,
    }
    if pushing:
        config.update(
            {
                "model_name": system_config["model_name"],
                "model_path": system_config["model_path"],
            }
        )
    if pushing and environment == "mujoco":
        # These are the table bounds used by the submitted MuJoCo push
        # sequence sampler, narrowed further by the generic 20% margin.
        config["initial_state_bounds"] = [[-0.12, 0.76], [-0.82, -0.34]]
    return system, config


def _trial_seed(base_seed: int, condition_index: int, trial: int) -> int:
    return int(base_seed + 1_000_003 * condition_index + 10_007 * trial)


def _disturbances(trial_seed: int, count: int) -> np.ndarray:
    return np.random.default_rng(trial_seed + 20_000_033).standard_normal((count, 3))


def _disturbance_hash(disturbances: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(disturbances, dtype=np.float64).tobytes()).hexdigest()


def _make_plant(
    system_name: str,
    environment: str,
    config: dict[str, Any],
    initial_state: np.ndarray,
    disturbances: np.ndarray,
    seed: int,
):
    plant_config = dict(config["simulator_config"])
    plant_config["start_state"] = np.asarray(initial_state, dtype=float).tolist()
    plant_config["disturbance_seed"] = int(seed)
    if environment == "gaussian":
        plant_config["disturbance_schedule"] = disturbances.tolist()
    if environment == "mujoco":
        plant_config["headless"] = True
    return create_simulator(system_name, environment, config=plant_config)


def _failed_method(method: str, initial_state: np.ndarray, exc: Exception) -> MethodResult:
    initial = np.asarray(initial_state, dtype=float).copy()
    return MethodResult(
        method=method,
        initial_state=initial,
        states=[initial.copy()],
        success=False,
        failure_reason=repr(exc),
    )


def _checkpoint(
    output_dir: Path,
    step_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> None:
    conditions = {
        (str(row["system"]), str(row["environment"]))
        for row in (*trial_rows, *references)
    }
    for system, environment in conditions:
        condition_dir = condition_output_dir(output_dir, system, environment)
        write_csv(
            condition_dir / STEP_FILE,
            (
                row
                for row in step_rows
                if row["system"] == system and row["environment"] == environment
            ),
        )
        write_csv(
            condition_dir / TRIAL_FILE,
            (
                row
                for row in trial_rows
                if row["system"] == system and row["environment"] == environment
            ),
        )
        write_json_lines(
            condition_dir / REFERENCE_FILE,
            (
                row
                for row in references
                if row["system"] == system and row["environment"] == environment
            ),
        )


def _completed_trials(trial_rows: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    counts: dict[tuple[str, str, int], set[str]] = {}
    for row in trial_rows:
        key = (str(row["system"]), str(row["environment"]), int(row["trial"]))
        counts.setdefault(key, set()).add(str(row["method"]))
    return {key for key, methods in counts.items() if methods == set(METHOD_ORDER)}


def _print_summary(summary: list[dict[str, Any]]) -> None:
    for condition in sorted({(row["system"], row["environment"]) for row in summary}):
        print(f"\n{condition[0]} / {condition[1]}")
        print("Method       Mean Tracking Error   Std. Dev.   95% CI")
        for method in METHOD_ORDER:
            row = next(
                item
                for item in summary
                if (item["system"], item["environment"], item["method"])
                == (*condition, method)
            )
            print(
                f"{row['method_label']:<12} {float(row['mean_tracking_error']):>19.6f} "
                f"{float(row['std_dev']):>11.6f}   "
                f"[{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}]"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = preliminary.parse_known_args(argv)
    defaults = yaml.safe_load(known.config.read_text(encoding="utf-8")) or {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument("--system", default="double_integrator")
    parser.add_argument(
        "--environment", choices=("gaussian", "mujoco", "all"), default="gaussian"
    )
    parser.add_argument("--num-trials", type=int, default=int(defaults["num_trials"]))
    parser.add_argument("--num-controls", type=int, default=int(defaults["num_controls"]))
    parser.add_argument("--seed", type=int, default=int(defaults["base_seed"]))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / str(defaults["results_dir"]),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=str(defaults["device"]))
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=int(defaults["bootstrap_resamples"]),
    )
    parser.add_argument(
        "--overrides-json",
        type=Path,
        help="per-condition nested controller overrides keyed as system__environment",
    )
    args = parser.parse_args(argv)
    if args.num_trials < 1 or args.num_controls < 1:
        parser.error("--num-trials and --num-controls must be positive")
    if args.bootstrap_resamples < 10_000:
        parser.error("--bootstrap-resamples must be at least 10000")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    systems = _resolve_systems(args.system)
    environments = _resolve_environments(args.environment)
    experiment_config = yaml.safe_load(
        args.config.resolve().read_text(encoding="utf-8")
    ) or {}
    condition_overrides = {
        str(key): dict(value)
        for key, value in experiment_config.get(
            "condition_hyperparameters", {}
        ).items()
    }
    for key, value in _load_condition_overrides(args.overrides_json).items():
        _deep_update(condition_overrides.setdefault(key, {}), value)
    resolved_device, device_info = _resolve_device(args.device)
    condition_configs: list[tuple[str, str, dict[str, Any]]] = []
    unavailable: list[dict[str, str]] = []
    for system_name in systems:
        for environment in environments:
            try:
                _, config = _condition_with_overrides(
                    system_name,
                    environment,
                    resolved_device,
                    condition_overrides,
                )
                condition_configs.append((system_name, environment, config))
            except Exception as exc:
                if len(systems) == 1 and len(environments) == 1:
                    raise
                unavailable.append(
                    {
                        "system": system_name,
                        "environment": environment,
                        "reason": repr(exc),
                    }
                )
    if args.environment == "all":
        unavailable.append(
            {
                "system": "pushing_object",
                "environment": "real_world",
                "reason": "requires external robot/camera hardware; no autonomous plant adapter exists",
            }
        )

    run_config = {
        "run": {
            "systems": list(systems),
            "environments": list(environments),
            "environment_requested": args.environment,
            "num_trials": args.num_trials,
            "num_controls": args.num_controls,
            "seed": args.seed,
            "seed_derivation": {
                "trial_seed": "base_seed + 1000003 * condition_index + 10007 * trial_index",
                "disturbance_seed": "trial_seed + 20000033",
                "aura_seed": "trial_seed + 30000049",
                "aura_step_seed": "aura_seed + 104729 * (step_index + 1)",
                "mppi_seed": "trial_seed + 40000063",
            },
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "device_requested": args.device,
            "device_resolved": resolved_device,
            "resume": bool(args.resume),
            "dry_run": bool(args.dry_run),
            "overrides_json": (
                str(args.overrides_json.resolve()) if args.overrides_json else None
            ),
            "experiment_config": str(args.config.resolve()),
            "condition_overrides": condition_overrides,
        },
        "conditions": {
            f"{name}__{environment}": config
            for name, environment, config in condition_configs
        },
        "reused_files": [
            "aura/optimization.py",
            "aura/AURA.py",
            "methods/MPPI.py",
            "methods/plan.py",
            "propagators/propagator.py",
            "propagators/double_integrator.py",
            "propagators/dubins_airplane.py",
            "propagators/kinematic_car.py",
            "propagators/pushing_object.py",
            "simulation/simulator.py",
            "simulation/pushing_model.py",
            "utils/utils.py",
        ],
    }
    if args.dry_run:
        print(json.dumps(run_config, indent=2, sort_keys=True))
        return 0

    if not args.resume:
        for system_name, environment, _ in condition_configs:
            shutil.rmtree(
                condition_output_dir(output_dir, system_name, environment),
                ignore_errors=True,
            )
        for pattern in (
            "tracking_summary.*",
            "tracking_step_statistics.csv",
            "paired_comparisons.csv",
            "success_rates.csv",
            "control_timing_*",
            "tracking_error_by_step*",
            "cumulative_tracking_error*",
        ):
            for path in output_dir.glob(pattern):
                if path.is_file():
                    path.unlink()

    config_path = output_dir / "config.json"
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        keys = (
            "systems",
            "environments",
            "num_trials",
            "num_controls",
            "seed",
            "condition_overrides",
        )
        mismatch = [
            key for key in keys if previous.get("run", {}).get(key) != run_config["run"].get(key)
        ]
        if mismatch:
            raise RuntimeError(f"resume configuration mismatch for: {', '.join(mismatch)}")
    write_json(config_path, run_config)

    metadata = {
        "started_at": _utc_now(),
        "finished_at": None,
        "elapsed_runtime_seconds": None,
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": device_info,
        "package_versions": _package_versions(),
        "git": _git_metadata(),
        "unavailable_conditions": unavailable,
        "completed_trial_count": 0,
        "status": "dry_run" if args.dry_run else "running",
    }
    metadata_path = output_dir / "run_metadata.json"
    write_json(metadata_path, metadata)

    step_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for system_name in ALL_SYSTEMS:
        for environment in ALL_ENVIRONMENTS:
            condition_dir = condition_output_dir(
                output_dir, system_name, environment
            )
            step_rows.extend(read_csv(condition_dir / STEP_FILE))
            trial_rows.extend(read_csv(condition_dir / TRIAL_FILE))
            references.extend(read_json_lines(condition_dir / REFERENCE_FILE))
    completed = _completed_trials(trial_rows)

    for condition_index, (system_name, environment, config) in enumerate(
        condition_configs
    ):
        for trial in range(args.num_trials):
            key = (system_name, environment, trial)
            if key in completed:
                print(
                    f"[resume] {system_name}/{environment} trial "
                    f"{trial + 1} already saved"
                )
                continue
            trial_seed = _trial_seed(args.seed, condition_index, trial)
            print(
                f"[run] {system_name}/{environment} trial "
                f"{trial + 1}/{args.num_trials} seed={trial_seed}",
                flush=True,
            )
            try:
                system, config = _condition_with_overrides(
                    system_name,
                    environment,
                    resolved_device,
                    condition_overrides,
                )
                reference_rng = np.random.default_rng(trial_seed)
                reference = generate_valid_reference(
                    system, args.num_controls, reference_rng, config
                )
                disturbance_schedule = _disturbances(trial_seed, args.num_controls)
                reference_row = reference.as_dict()
                reference_row.update(
                    {
                        "trial": trial,
                        "seed": trial_seed,
                        "disturbance_schedule_sha256": _disturbance_hash(
                            disturbance_schedule
                        ),
                    }
                )
                method_results = []

                plant = None
                try:
                    plant = _make_plant(
                        system_name,
                        environment,
                        config,
                        reference.initial_state,
                        disturbance_schedule,
                        trial_seed,
                    )
                    method_results.append(run_open_loop(reference, plant, config))
                except Exception as exc:
                    method_results.append(
                        _failed_method("open_loop", reference.initial_state, exc)
                    )
                finally:
                    close_simulator(plant)

                plant = None
                try:
                    aura = AURALocalTrackingAdapter(system, config)
                    plant = _make_plant(
                        system_name,
                        environment,
                        config,
                        reference.initial_state,
                        disturbance_schedule,
                        trial_seed,
                    )
                    method_results.append(
                        run_aura_tracking(
                            reference,
                            aura,
                            plant,
                            config,
                            seed=trial_seed + 30_000_049,
                        )
                    )
                except Exception as exc:
                    method_results.append(
                        _failed_method("aura", reference.initial_state, exc)
                    )
                finally:
                    close_simulator(plant)

                plant = None
                try:
                    params = parameters_from_config(
                        system_name, {"mppi": config["mppi"]}
                    )
                    controller = make_mppi_controller(
                        system,
                        reference,
                        params,
                        config,
                        seed=trial_seed + 40_000_063,
                    )
                    plant = _make_plant(
                        system_name,
                        environment,
                        config,
                        reference.initial_state,
                        disturbance_schedule,
                        trial_seed,
                    )
                    method_results.append(
                        run_mppi_tracking(reference, controller, plant, config)
                    )
                except Exception as exc:
                    method_results.append(
                        _failed_method("mppi", reference.initial_state, exc)
                    )
                finally:
                    close_simulator(plant)

                initial_states = [result.initial_state for result in method_results]
                if not all(
                    np.allclose(initial_states[0], state, rtol=0.0, atol=1e-10)
                    for state in initial_states[1:]
                ):
                    raise RuntimeError("methods did not begin from an identical plant state")
                for result in method_results:
                    step_rows.extend(
                        method_result_to_step_rows(
                            reference=reference,
                            result=result,
                            trial=trial,
                            seed=trial_seed,
                        )
                    )
                    trial_rows.append(
                        method_result_to_trial_row(
                            reference=reference,
                            result=result,
                            trial=trial,
                            seed=trial_seed,
                        )
                    )
                references.append(reference_row)
            except Exception as exc:
                traceback.print_exc()
                # A nominal-reference failure happens before any method can be
                # compared, so it is an explicit unavailable trial rather than
                # a selectively dropped controller outcome.
                unavailable.append(
                    {
                        "system": system_name,
                        "environment": environment,
                        "trial": str(trial),
                        "reason": repr(exc),
                    }
                )
                raise

            _checkpoint(output_dir, step_rows, trial_rows, references)
            # AURA's pushing adapter owns large optimizer tensors and MuJoCo
            # wrappers can participate in Python reference cycles.  Release
            # per-trial objects before the next 10,000-state batch so a
            # 100-trial campaign does not progressively slow down.
            try:
                del method_results
                del aura
                del controller
            except UnboundLocalError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            metadata["completed_trial_count"] = len(_completed_trials(trial_rows))
            metadata["unavailable_conditions"] = unavailable
            metadata["elapsed_runtime_seconds"] = time.perf_counter() - started
            write_json(metadata_path, metadata)

    statistics = compute_tracking_statistics(
        step_rows,
        trial_rows,
        num_controls=args.num_controls,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    write_summary_tables(output_dir, statistics["summary"])
    write_csv(
        output_dir / "tracking_step_statistics.csv", statistics["step_statistics"]
    )
    write_csv(
        output_dir / "paired_comparisons.csv", statistics["paired_comparisons"]
    )
    write_csv(output_dir / "success_rates.csv", statistics["success_rates"])
    plot_paths = plot_tracking_statistics(statistics["step_statistics"], output_dir)

    metadata.update(
        {
            "finished_at": _utc_now(),
            "elapsed_runtime_seconds": time.perf_counter() - started,
            "completed_trial_count": len(_completed_trials(trial_rows)),
            "unavailable_conditions": unavailable,
            "status": "complete",
            "plot_paths": {key: str(value) for key, value in plot_paths.items()},
        }
    )
    write_json(metadata_path, metadata)
    _print_summary(statistics["summary"])
    print(f"\nOutputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
