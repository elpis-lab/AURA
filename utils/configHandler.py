import argparse
import os
import shlex
import sys
from typing import Any

import numpy as np


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "initial_time_experiment.yaml",
)

DEFAULT_CONFIG = {
    "planner_name": "aorrt",
    "results_dir": "results/planning/initial_time_car_gaussian",
    "control_durations": [1, 2, 3, 4, 5],
    "planning_times": [3.0, 6.0, 9.0, 12.0, 15.0],
    "num_runs": 10,
}


def load_config(config_file: str) -> dict:
    import yaml

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args_and_config():
    """Parse command line arguments and load configuration from YAML file."""
    import yaml

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run fusion planning with YAML configuration")
    parser.add_argument(
        "--planning-time",
        type=float,
        help="Planning time in seconds (overrides YAML)",
    )
    parser.add_argument(
        "--replanning-time",
        type=float,
        help="Replanning time in seconds (overrides YAML)",
    )
    parser.add_argument("--planner-name", type=str, help="Planner name (overrides YAML)")
    parser.add_argument("--system", type=str, help="System name (overrides YAML)")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization (overrides YAML)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization (overrides YAML)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for optimization (overrides YAML)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of epochs for optimization (overrides YAML)",
    )
    parser.add_argument(
        "--sampling-num-states",
        type=int,
        help="Number of states for sampling (overrides YAML)",
    )
    parser.add_argument(
        "--sampling-position-std",
        type=float,
        help="Position standard deviation for sampling (overrides YAML)",
    )
    parser.add_argument(
        "--sampling-rotation-std",
        type=float,
        help="Rotation standard deviation for sampling (overrides YAML)",
    )
    parser.add_argument(
        "--propagation-step-size",
        type=float,
        help="Propagation step size (overrides YAML)",
    )
    parser.add_argument(
        "--min-control-duration",
        type=int,
        help="Minimum control duration in steps (overrides YAML)",
    )
    parser.add_argument(
        "--max-control-duration",
        type=int,
        help="Maximum control duration in steps (overrides YAML)",
    )
    parser.add_argument(
        "--goal-threshold",
        type=float,
        help="Goal threshold for planning (overrides YAML)",
    )
    parser.add_argument(
        "--pruning-radius",
        type=float,
        help="Pruning radius for planner (overrides YAML)",
    )

    args = parser.parse_args()

    # Use fixed launcher config file name
    launcher_config_file = "config.yaml"

    # Load configuration from YAML file
    try:
        launcher_config = load_config(launcher_config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{launcher_config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    # If the top-level config contains a 'config_file' pointer, load it and merge
    if isinstance(launcher_config, dict) and launcher_config.get("config_file"):
        try:
            pointed_config = load_config(launcher_config["config_file"])
        except Exception as e:
            print(f"Error loading referenced config '{launcher_config['config_file']}': {e}")
            sys.exit(1)
        # Merge: values in launcher_config override pointed_config (except 'config_file')
        config = dict(pointed_config or {})
        for k, v in launcher_config.items():
            if k == "config_file":
                continue
            if v is not None:
                config[k] = v
    else:
        config = launcher_config

    # Extract parameters from config with command line overrides
    system = args.system if args.system is not None else config.get("system")
    objectName = config.get("objectName")
    startState = (
        np.array(config.get("startState")) if config.get("startState") is not None else None
    )
    goalState = np.array(config.get("goalState")) if config.get("goalState") is not None else None

    # Use command line args if provided, otherwise use YAML values
    planningTime = (
        args.planning_time if args.planning_time is not None else config.get("planningTime")
    )
    replanningTime = (
        args.replanning_time if args.replanning_time is not None else config.get("replanningTime")
    )
    plannerName = args.planner_name if args.planner_name is not None else config.get("plannerName")

    # Handle visualize flag with explicit override options
    if args.visualize:
        visualize = True
    elif args.no_visualize:
        visualize = False
    else:
        visualize = config.get("visualize")

    # Extract optimization parameters
    learningRate = (
        args.learning_rate
        if args.learning_rate is not None
        else config.get("optimizer_learning_rate", 0.001)
    )
    numEpochs = (
        args.num_epochs if args.num_epochs is not None else config.get("optimizer_num_steps", 1000)
    )

    # Ensure we have valid values
    if learningRate is None or learningRate <= 0:
        print(f"[WARNING] Invalid learningRate: {learningRate}, using default: 0.001")
        learningRate = 0.001
    if numEpochs is None or numEpochs <= 0:
        print(f"[WARNING] Invalid numEpochs: {numEpochs}, using default: 1000")
        numEpochs = 1000

    # Debug: Print loaded values
    print(f"[DEBUG] Config loading:")
    print(f"  - optimizer_learning_rate from YAML: {config.get('optimizer_learning_rate')}")
    print(f"  - optimizer_num_steps from YAML: {config.get('optimizer_num_steps')}")
    print(f"  - learning_rate from args: {args.learning_rate}")
    print(f"  - num_epochs from args: {args.num_epochs}")
    print(f"  - Final learningRate: {learningRate}")
    print(f"  - Final numEpochs: {numEpochs}")

    # Extract sampling parameters
    sampling_config = config.get("sampling", {}) or {}
    sampling_num_states = (
        args.sampling_num_states
        if args.sampling_num_states is not None
        else sampling_config.get("num_states", 1000)
    )
    sampling_position_std = (
        args.sampling_position_std
        if args.sampling_position_std is not None
        else sampling_config.get("position_std", 0.005)
    )
    sampling_rotation_std = (
        args.sampling_rotation_std
        if args.sampling_rotation_std is not None
        else sampling_config.get("rotation_std", 0.1)
    )
    sampling_max_distance = sampling_config.get("max_distance", 0.025)

    # Ensure we have valid sampling values
    if sampling_num_states is None or sampling_num_states <= 0:
        print(f"[WARNING] Invalid sampling_num_states: {sampling_num_states}, using default: 1000")
        sampling_num_states = 1000
    if sampling_position_std is None or sampling_position_std <= 0:
        print(
            f"[WARNING] Invalid sampling_position_std: {sampling_position_std}, using default: 0.005"
        )
        sampling_position_std = 0.005
    if sampling_rotation_std is None or sampling_rotation_std <= 0:
        print(
            f"[WARNING] Invalid sampling_rotation_std: {sampling_rotation_std}, using default: 0.1"
        )
        sampling_rotation_std = 0.1
    if sampling_max_distance is None or sampling_max_distance <= 0:
        print(
            f"[WARNING] Invalid sampling_max_distance: {sampling_max_distance}, using default: 0.025"
        )
        sampling_max_distance = 0.025

    # Extract propagation step size from config
    propagation_step_size = (
        args.propagation_step_size
        if args.propagation_step_size is not None
        else config.get("propagation_step_size", 1.0)
    )

    # Extract control duration parameters from config
    min_control_duration = (
        args.min_control_duration
        if args.min_control_duration is not None
        else config.get("min_control_duration", 1)
    )
    max_control_duration = (
        args.max_control_duration
        if args.max_control_duration is not None
        else config.get("max_control_duration", 5)
    )

    # Extract goal threshold and pruning radius parameters
    goal_threshold = (
        args.goal_threshold
        if args.goal_threshold is not None
        else config.get("goal_threshold", 0.1)
    )
    pruning_radius = (
        args.pruning_radius
        if args.pruning_radius is not None
        else config.get("pruning_radius", 0.1)
    )

    # Ensure we have valid propagation and control values
    if propagation_step_size is None or propagation_step_size <= 0:
        print(
            f"[WARNING] Invalid propagation_step_size: {propagation_step_size}, using default: 1.0"
        )
        propagation_step_size = 1.0
    if min_control_duration is None or min_control_duration <= 0:
        print(f"[WARNING] Invalid min_control_duration: {min_control_duration}, using default: 1")
        min_control_duration = 1
    if max_control_duration is None or max_control_duration <= 0:
        print(f"[WARNING] Invalid max_control_duration: {max_control_duration}, using default: 5")
        max_control_duration = 5
    if min_control_duration > max_control_duration:
        print(
            f"[WARNING] min_control_duration ({min_control_duration}) > max_control_duration ({max_control_duration}), swapping"
        )
        min_control_duration, max_control_duration = max_control_duration, min_control_duration

    # Ensure we have valid goal threshold and pruning radius values
    if goal_threshold is None or goal_threshold <= 0:
        print(f"[WARNING] Invalid goal_threshold: {goal_threshold}, using default: 0.1")
        goal_threshold = 0.1
    if pruning_radius is None or pruning_radius <= 0:
        print(f"[WARNING] Invalid pruning_radius: {pruning_radius}, using default: 0.1")
        pruning_radius = 0.1

    # Debug: Print final returned values
    print(f"[DEBUG] Final config values:")
    print(f"  - learningRate: {learningRate}")
    print(f"  - numEpochs: {numEpochs}")
    print(f"  - sampling_num_states: {sampling_num_states}")
    print(f"  - sampling_max_distance: {sampling_max_distance}")
    print(f"  - sampling_position_std: {sampling_position_std}")
    print(f"  - sampling_rotation_std: {sampling_rotation_std}")
    print(f"  - propagation_step_size: {propagation_step_size}")
    print(f"  - min_control_duration: {min_control_duration}")
    print(f"  - max_control_duration: {max_control_duration}")
    print(f"  - goal_threshold: {goal_threshold}")
    print(f"  - pruning_radius: {pruning_radius}")

    return {
        "system": system,
        "objectName": objectName,
        "startState": startState,
        "goalState": goalState,
        "planningTime": planningTime,
        "replanningTime": replanningTime,
        "plannerName": plannerName,
        "visualize": visualize,
        "learningRate": learningRate,
        "numEpochs": numEpochs,
        "sampling_num_states": sampling_num_states,
        "sampling_max_distance": sampling_max_distance,
        "sampling_position_std": sampling_position_std,
        "sampling_rotation_std": sampling_rotation_std,
        "propagation_step_size": propagation_step_size,
        "min_control_duration": min_control_duration,
        "max_control_duration": max_control_duration,
        "goal_threshold": goal_threshold,
        "pruning_radius": pruning_radius,
    }


def load_and_normalize_config(
    config_file: str, system_name: str = None, planner_name: str = None
) -> dict:
    """Load configuration from YAML file and normalize all values.

    Args:
        config_file: Path to the YAML configuration file
        system_name: System name to override in config (optional)
        planner_name: Planner name to override in config (optional)

    Returns:
        Normalized configuration dictionary with all values properly typed
    """
    import yaml

    # Load configuration from YAML file
    try:
        config = load_config(config_file)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML file '{config_file}': {e}")
        sys.exit(1)

    # Override system and planner if provided
    if system_name is not None:
        config["system"] = system_name
    if planner_name is not None:
        config["plannerName"] = planner_name

    # Extract nested configs
    sampling_config = config.get("sampling", {}) or {}
    optimization_config = config.get("optimization", {}) or {}

    # Normalize planning parameters
    config["goal_threshold"] = float(config.get("goal_threshold", 0.1))
    config["min_control_duration"] = int(config.get("min_control_duration", 1))
    config["max_control_duration"] = int(config.get("max_control_duration", 5))
    config["propagation_step_size"] = float(config.get("propagation_step_size", 1.0))
    config["pruning_radius"] = float(config.get("pruning_radius", 0.1))
    config["planningTime"] = float(config.get("planningTime", 10.0))
    config["replanningMaxDistance"] = float(config.get("replanningMaxDistance", 0.05))
    config["max_total_time"] = float(config.get("max_total_time", 300.0))

    # Normalize sampling parameters
    config["sampling_max_distance"] = float(sampling_config.get("max_distance", 0.05))
    config["sampling_position_std"] = float(sampling_config.get("position_std", 0.003))
    config["sampling_rotation_std"] = float(sampling_config.get("rotation_std", 0.05))
    config["sampling_num_states"] = int(sampling_config.get("num_states", 1000))

    # Normalize optimization parameters
    # First normalize the optimizer_* keys if they exist (from YAML)
    optimizer_lr = None
    optimizer_steps = None
    optimizer_factor = None
    optimizer_patience = None
    optimizer_min_lr = None

    if "optimizer_learning_rate" in config:
        optimizer_lr = float(config["optimizer_learning_rate"])
        config["optimizer_learning_rate"] = optimizer_lr

    if "optimizer_num_steps" in config:
        optimizer_steps = int(config["optimizer_num_steps"])
        config["optimizer_num_steps"] = optimizer_steps

    if "optimizer_plateau_factor" in config:
        optimizer_factor = float(config["optimizer_plateau_factor"])
        config["optimizer_plateau_factor"] = optimizer_factor

    if "optimizer_plateau_patience" in config:
        optimizer_patience = int(config["optimizer_plateau_patience"])
        config["optimizer_plateau_patience"] = optimizer_patience

    if "optimizer_plateau_min_lr" in config:
        optimizer_min_lr = float(config["optimizer_plateau_min_lr"])
        config["optimizer_plateau_min_lr"] = optimizer_min_lr

    # Now normalize the base keys, using optimizer_* as fallback if available
    config["learningRate"] = float(
        optimization_config.get(
            "learning_rate",
            config.get("learning_rate", optimizer_lr if optimizer_lr is not None else 0.001),
        )
    )
    config["numEpochs"] = int(
        optimization_config.get(
            "num_epochs",
            config.get("num_epochs", optimizer_steps if optimizer_steps is not None else 1000),
        )
    )
    config["plateau_factor"] = float(
        optimization_config.get(
            "plateau_factor",
            config.get("plateau_factor", optimizer_factor if optimizer_factor is not None else 0.5),
        )
    )
    config["plateau_patience"] = int(
        optimization_config.get(
            "plateau_patience",
            config.get(
                "plateau_patience", optimizer_patience if optimizer_patience is not None else 2
            ),
        )
    )
    config["plateau_min_lr"] = float(
        optimization_config.get(
            "plateau_min_lr",
            config.get(
                "plateau_min_lr", optimizer_min_lr if optimizer_min_lr is not None else 1e-6
            ),
        )
    )

    # If optimizer_* keys weren't in config, set them to normalized base values for backward compatibility
    if optimizer_lr is None:
        config["optimizer_learning_rate"] = config["learningRate"]
    if optimizer_steps is None:
        config["optimizer_num_steps"] = config["numEpochs"]
    if optimizer_factor is None:
        config["optimizer_plateau_factor"] = config["plateau_factor"]
    if optimizer_patience is None:
        config["optimizer_plateau_patience"] = config["plateau_patience"]
    if optimizer_min_lr is None:
        config["optimizer_plateau_min_lr"] = config["plateau_min_lr"]

    # Convert numpy arrays
    if config.get("startState"):
        config["startState"] = np.array(config["startState"])
    if config.get("goalState"):
        config["goalState"] = np.array(config["goalState"])

    # Handle objectName (can be object_name or objectName in YAML)
    if "objectName" not in config and "object_name" in config:
        config["objectName"] = config["object_name"]
    elif "objectName" not in config:
        # Set default if neither is present
        config["objectName"] = None

    # Set defaults for optional fields
    config["visualize"] = config.get("visualize", False)

    return config


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if (text[0], text[-1]) in {('"', '"'), ("'", "'")}:
        return text[1:-1]
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if any(char in text for char in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_inline_list(value: str) -> list[Any]:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"expected inline list, got {value!r}")
    body = text[1:-1].strip()
    if not body:
        return []
    return [_parse_scalar(item) for item in body.split(",")]


def _parse_yaml_subset(path: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    current_list_key: str | None = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = _strip_comment(raw_line).rstrip()
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith("- "):
                if current_list_key is None:
                    raise ValueError(f"list item without a key in {path}: {raw_line!r}")
                parsed.setdefault(current_list_key, []).append(
                    _parse_scalar(stripped[2:].strip())
                )
                continue

            current_list_key = None
            if ":" not in stripped:
                raise ValueError(f"expected key: value in {path}: {raw_line!r}")
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError(f"empty key in {path}: {raw_line!r}")
            if not value:
                parsed[key] = []
                current_list_key = key
            elif value.startswith("["):
                parsed[key] = _parse_inline_list(value)
            else:
                parsed[key] = _parse_scalar(value)
    return parsed


def _as_float_list(value: Any, key: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    out = [float(item) for item in value]
    if not out:
        raise ValueError(f"{key} must contain at least one value")
    return out


def load_experiment_config(path: str | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config_path = path or DEFAULT_CONFIG_PATH
    if config_path and os.path.exists(config_path):
        config.update(_parse_yaml_subset(config_path))
    elif path:
        raise FileNotFoundError(config_path)

    config["planner_name"] = str(config["planner_name"])
    config["results_dir"] = str(config["results_dir"])
    config["control_durations"] = _as_float_list(
        config["control_durations"], "control_durations"
    )
    config["planning_times"] = _as_float_list(config["planning_times"], "planning_times")
    config["num_runs"] = int(config["num_runs"])

    if config["num_runs"] <= 0:
        raise ValueError("num_runs must be positive")
    if any(duration <= 0 for duration in config["control_durations"]):
        raise ValueError("control_durations must be positive")
    if any(time <= 0.0 for time in config["planning_times"]):
        raise ValueError("planning_times must be positive")
    return config


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def emit_bash_exports(config: dict[str, Any]) -> None:
    print(f"CONFIG_PLANNER_NAME={shlex.quote(str(config['planner_name']))}")
    print(f"CONFIG_RESULTS_DIR={shlex.quote(str(config['results_dir']))}")
    print(f"CONFIG_NUM_RUNS={shlex.quote(str(int(config['num_runs'])))}")
    control_values = " ".join(
        shlex.quote(_format_number(float(value))) for value in config["control_durations"]
    )
    planning_values = " ".join(
        shlex.quote(_format_number(float(value))) for value in config["planning_times"]
    )
    print(f"CONFIG_CONTROL_DURATIONS=({control_values})")
    print(f"CONFIG_PLANNING_TIMES=({planning_values})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read AURA configuration files.")
    parser.add_argument(
        "--experiment-grid",
        action="store_true",
        help="Read the initial-time experiment sweep config.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Experiment-grid YAML config path.",
    )
    parser.add_argument(
        "--emit-bash",
        action="store_true",
        help="Print bash assignments for run_initial_time_experiments.sh.",
    )
    args = parser.parse_args()

    if not args.experiment_grid:
        parser.error("--experiment-grid is required when running configHandler.py directly")

    config = load_experiment_config(args.config)
    if args.emit_bash:
        emit_bash_exports(config)
    else:
        for key in (
            "planner_name",
            "results_dir",
            "control_durations",
            "planning_times",
            "num_runs",
        ):
            print(f"{key}: {config[key]}")


if __name__ == "__main__":
    main()
