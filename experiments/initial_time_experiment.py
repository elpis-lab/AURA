#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import re
import sys
import time
import traceback
from collections import Counter
from copy import deepcopy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aura_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

BASE_CONFIG = {
    "start_state": [0.25, 0.25, 0.0],
    "goal_state": [4.75, 4.75, 0.0],
    "goal_threshold": 0.1,
    "state_bounds": [(0.0, 5.0), (0.0, 5.0)],
    "propagation_step_size": 1.0,
    "motion_validation_step_size": 0.05,
    "pruning_radius": 0.1,
    "replanning_max_distance": 0.075,
    "sampling_position_std": 0.003,
    "sampling_rotation_std": 0.05,
    "sampling_velocity_std": 0.003,
    "optimizer_num_states": 10000,
    "optimizer_num_steps": 500,
    "optimizer_learning_rate": 0.1,
    "execution_safety_radius": 0.0,
    "recovery_replanning_time": 1.0,
    "visualize": True,
    "obstacles": {
        "enabled": True,
        "safety_radius": 0.15,
        "circles": [
            (2.1, 2.1, 0.50),
            (2.8, 2.9, 0.45),
            (2.8, 0.7, 0.40),
            # (0.5, 1.8, 0.35),
            (3.5, 4.0, 0.45),
            (4.5, 2.8, 0.35),
            (1.0, 4.0, 0.42),
            (4.0, 1.2, 0.45),
        ],
        "aabbs": [],
        "boxes": [],
    },
}


def _configure_matplotlib_backend() -> str:
    """Set backend before pyplot. Must run before imports that may touch pyplot."""
    if "--no-show" in sys.argv:
        matplotlib.use("Agg")
        return "Agg"
    if not BASE_CONFIG.get("visualize", True):
        matplotlib.use("Agg")
        return "Agg"
    envb = (os.environ.get("MPLBACKEND") or "").strip()
    candidates: list[str] = []
    if envb:
        candidates.append(envb)
    for b in ("TkAgg", "Qt5Agg", "QtAgg", "Gtk3Agg"):
        if b not in candidates:
            candidates.append(b)
    for b in candidates:
        try:
            matplotlib.use(b, force=True)
            return b
        except Exception:
            continue
    matplotlib.use("Agg")
    print(
        "[plotting] No interactive matplotlib backend worked (tried $MPLBACKEND, TkAgg, Qt5Agg, …). "
        "Only PNGs will be saved. Install a GUI toolkit (e.g. python3-tk on Debian/Ubuntu) or set MPLBACKEND.",
    )
    return "Agg"


_CONFIGURED_BACKEND = _configure_matplotlib_backend()
import matplotlib.pyplot as plt
import numpy as np
from ompl import base as ob
from ompl import util as ou

from AURA import AURA
from utils.auraHandler import reset_optimizer_loss_live_figure
from utils.configHandler import DEFAULT_CONFIG_PATH, load_experiment_config
from optimization import optimizer_device_info, warmup_optimizer_device
from plan import OMPL_Planner
from simulation.simulators import create_simulator
from systems import get_system
from utils.utils import arrayDistance, is_state_array_valid, normalize_obstacle_config

SYSTEM_NAME = "kinematic_car"
SIMULATOR_MODE = "gaussian"
WORKSPACE_REPLAY_RE = re.compile(
    r"(?P<planner>.+)_cd(?P<control_duration>-?\d+(?:\.\d+)?)_r(?P<run_number>\d+)_"
    r"pt(?P<planning_time>-?\d+(?:\.\d+)?)\.npz$"
)


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _colorize(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def print_section(message: str) -> None:
    rule = "=" * 72
    print()
    print(_colorize(rule, "36"), flush=True)
    print(_colorize(message, "1;36"), flush=True)
    print(_colorize(rule, "36"), flush=True)


def print_step(message: str) -> None:
    rule = "-" * 72
    print()
    print(_colorize(rule, "90"), flush=True)
    print(_colorize(message, "1;33"), flush=True)
    print(_colorize(rule, "90"), flush=True)


def print_status(message: str) -> None:
    print(_colorize(message, "33"), flush=True)


def print_success(message: str) -> None:
    print(_colorize(message, "32"), flush=True)


def format_state3(state: np.ndarray | list[float]) -> str:
    arr = np.asarray(state, dtype=float).reshape(-1)
    return np.array2string(
        arr[:3],
        precision=6,
        floatmode="fixed",
        separator=", ",
    )


def get_next_run_number(
    results_dir: str, planner_name: str, control_duration: float
) -> int:
    os.makedirs(results_dir, exist_ok=True)
    cd_token = control_duration_token(control_duration)
    existing = glob.glob(
        f"{results_dir}/{SYSTEM_NAME}_{planner_name}_cd{cd_token}_*.csv"
    )
    run_numbers = []
    pattern = rf"{SYSTEM_NAME}_{planner_name}_cd{re.escape(cd_token)}_(\d+)\.csv"
    for filename in existing:
        match = re.search(pattern, os.path.basename(filename))
        if match:
            run_numbers.append(int(match.group(1)))
    return max(run_numbers) + 1 if run_numbers else 1


def control_duration_token(control_duration: float) -> str:
    value = float(control_duration)
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def planning_time_token(planning_time: float) -> str:
    return f"{float(planning_time):.2f}"


def trial_tag(planner_name: str, control_duration: float, run_number: int, planning_time: float) -> str:
    return (
        f"{planner_name}_cd{control_duration_token(control_duration)}_r{int(run_number):02d}_"
        f"pt{planning_time_token(planning_time)}"
    )


def workspace_replay_path(
    results_dir: str,
    planner_name: str,
    control_duration: float,
    run_number: int,
    planning_time: float,
) -> str:
    return os.path.join(
        results_dir,
        "workspace_replays",
        f"{trial_tag(planner_name, control_duration, run_number, planning_time)}.npz",
    )


def parse_workspace_replay_filename(path: str) -> tuple[str, float, int, float] | None:
    match = WORKSPACE_REPLAY_RE.search(os.path.basename(path))
    if not match:
        return None
    return (
        match.group("planner"),
        float(match.group("control_duration")),
        int(match.group("run_number")),
        float(match.group("planning_time")),
    )


def workspace_video_path(
    results_dir: str,
    planner_name: str,
    control_duration: float,
    run_number: int,
    planning_time: float,
) -> str:
    return os.path.join(
        results_dir,
        "workspace_videos",
        f"{trial_tag(planner_name, control_duration, run_number, planning_time)}.mp4",
    )


def workspace_frames_dir(
    results_dir: str,
    planner_name: str,
    control_duration: float,
    run_number: int,
    planning_time: float,
) -> str:
    return os.path.join(
        results_dir,
        "workspace_frames",
        trial_tag(planner_name, control_duration, run_number, planning_time),
    )


def optimization_loss_dir(
    results_dir: str,
    planner_name: str,
    control_duration: float,
    run_number: int,
    planning_time: float,
) -> str:
    return os.path.join(
        results_dir,
        "optimization_loss",
        trial_tag(planner_name, control_duration, run_number, planning_time),
    )


def replay_is_complete(replay_path: str) -> bool:
    if not os.path.exists(replay_path) or os.path.getsize(replay_path) <= 0:
        return False
    try:
        with np.load(replay_path, allow_pickle=True) as data:
            if "frames" not in data.files:
                return False
            return len(data["frames"]) > 0
    except Exception:
        return False


def write_missing_replay_file(
    path: str,
    missing: list[tuple[float, int, float, str]],
) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for control_duration, run_number, planning_time, replay_path in missing:
            f.write(
                f"{float(control_duration):.12g}\t{int(run_number)}\t"
                f"{float(planning_time):.12g}\t{replay_path}\n"
            )


def experiment_config(control_duration: float) -> dict:
    config = deepcopy(BASE_CONFIG)
    config["start_state"] = np.asarray(config["start_state"], dtype=float)
    config["goal_state"] = np.asarray(config["goal_state"], dtype=float)
    config["state_bounds"] = [
        (float(low), float(high)) for (low, high) in config["state_bounds"]
    ]
    control_duration_seconds = float(control_duration)
    validation_step = float(
        config.get("motion_validation_step_size", config["propagation_step_size"])
    )
    if validation_step <= 0.0:
        raise ValueError("motion_validation_step_size must be positive")
    control_steps = max(1, int(round(control_duration_seconds / validation_step)))
    config["propagation_step_size"] = control_duration_seconds / float(control_steps)
    config["control_duration_seconds"] = control_duration_seconds
    config["min_control_duration"] = int(control_steps)
    config["max_control_duration"] = int(control_steps)
    config["replanningMaxDistance"] = float(config["replanning_max_distance"])
    return config


def apply_state_bounds(system, state_bounds: list[tuple[float, float]]) -> None:
    if system.name != SYSTEM_NAME:
        raise ValueError(f"Unsupported system for custom state bounds: {system.name}")
    if len(state_bounds) != 2:
        raise ValueError(
            f"{SYSTEM_NAME} expects 2 planar bounds entries, got {len(state_bounds)}"
        )

    system.state_bounds = [(float(low), float(high)) for (low, high) in state_bounds]
    bounds = ob.RealVectorBounds(2)
    for i, (low, high) in enumerate(system.state_bounds):
        bounds.setLow(i, float(low))
        bounds.setHigh(i, float(high))
    system.state_space.setBounds(bounds)


def build_planner(
    system, config: dict, planner_name: str, planning_time: float
) -> OMPL_Planner:
    planner = OMPL_Planner(
        system=system,
        start_state=np.asarray(config["start_state"], dtype=float),
        goal_state=np.asarray(config["goal_state"], dtype=float),
        planner_method=planner_name,
        goal_threshold=float(config["goal_threshold"]),
        min_max_control_duration=(
            int(config["min_control_duration"]),
            int(config["max_control_duration"]),
        ),
        propagation_step_size=float(config["propagation_step_size"]),
        initial_planning_time=float(planning_time),
        pruning_radius=float(config["pruning_radius"]),
        obstacle_config=config.get("obstacles"),
    )
    planner.replanning_time = float(config["replanningMaxDistance"])
    planner.opt_model = None
    planner.optimizer_num_states = int(config["optimizer_num_states"])
    planner.optimizer_num_steps = int(config["optimizer_num_steps"])
    planner.optimizer_learning_rate = float(config["optimizer_learning_rate"])
    planner.optimizer_pos_std = float(config["sampling_position_std"])
    planner.optimizer_rot_std = float(config["sampling_rotation_std"])
    planner.optimizer_vel_std = float(config["sampling_velocity_std"])
    planner.motion_validation_step_size = float(config["propagation_step_size"])
    planner.control_duration_seconds = float(config["control_duration_seconds"])
    planner.recovery_replanning_time = float(config["recovery_replanning_time"])
    return planner


def state_in_collision(state: np.ndarray, config: dict) -> bool:
    return not is_state_array_valid(
        np.asarray(state, dtype=float),
        system=SYSTEM_NAME,
        config=config,
        obstacle_config=normalize_obstacle_config(config.get("obstacles")),
        safety_radius_override=float(config.get("execution_safety_radius", 0.0)),
    )


def first_collision_step(
    states_trajectory: list[np.ndarray], config: dict
) -> int | None:
    for step_idx, state in enumerate(states_trajectory[1:], start=1):
        if state_in_collision(np.asarray(state, dtype=float), config):
            return step_idx
    return None


def run_single_experiment(
    planner_name: str,
    planning_time: float,
    control_duration: float,
    *,
    workspace_frames_dir: str | None = None,
    workspace_live: bool = False,
    workspace_replay_path: str | None = None,
    workspace_video_path: str | None = None,
    workspace_video_fps: float = 3.0,
    workspace_video_dpi: int = 150,
    optimization_loss_plot_dir: str | None = None,
    show_optimization_loss_plot: bool = False,
) -> tuple[dict, list[float]]:
    config = experiment_config(control_duration)
    system = get_system(SYSTEM_NAME)
    apply_state_bounds(system, config["state_bounds"])
    planner = build_planner(system, config, planner_name, planning_time)

    plan_start = time.time()
    solutions, _ = planner.plan()
    planning_wall_time = time.time() - plan_start
    if solutions is None or len(solutions) == 0:
        raise RuntimeError(
            f"Initial planning failed for planner={planner_name}, planning_time={planning_time}"
        )

    simulator = create_simulator(SYSTEM_NAME, SIMULATOR_MODE, config=config)
    aura_runner = AURA(system=system, planner=planner, simulator=simulator)

    aura_start = time.time()
    run_kwargs: dict = {
        "pause_each_step": False,
        "optimization_loss_plot_dir": optimization_loss_plot_dir,
        "show_optimization_loss_plot": show_optimization_loss_plot,
    }
    callbacks = []
    replay_recorder = None
    if config.get("visualize", True):
        if workspace_live and _CONFIGURED_BACKEND != "Agg":
            from scripts.plotting import workspace_planning_callback_interactive

            callbacks.append(
                workspace_planning_callback_interactive(
                    config["state_bounds"],
                    config.get("obstacles"),
                    system=system,
                    planner_name=planner_name,
                    curve_step_size=float(config["propagation_step_size"]),
                    frames_dir=workspace_frames_dir,
                    title_prefix=f"pt={float(planning_time):.1f}s",
                    goal_state=config["goal_state"],
                    goal_threshold=float(config["goal_threshold"]),
                )
            )
        elif workspace_frames_dir:
            from scripts.plotting import workspace_planning_callback

            callbacks.append(
                workspace_planning_callback(
                    workspace_frames_dir,
                    config["state_bounds"],
                    config.get("obstacles"),
                    system=system,
                    planner_name=planner_name,
                    curve_step_size=float(config["propagation_step_size"]),
                    title_prefix=f"planning_time={float(planning_time):.1f}s",
                    goal_state=config["goal_state"],
                    goal_threshold=float(config["goal_threshold"]),
                )
            )
    if workspace_video_path and not workspace_replay_path:
        workspace_replay_path = os.path.splitext(workspace_video_path)[0] + ".npz"
    if workspace_replay_path:
        from scripts.plotting import WorkspaceReplayRecorder

        replay_recorder = WorkspaceReplayRecorder(
            workspace_replay_path,
            config["state_bounds"],
            config.get("obstacles"),
            system_name=SYSTEM_NAME,
            planner_name=planner_name,
            curve_step_size=float(config["propagation_step_size"]),
            title_prefix=(
                f"{planner_name} cd={control_duration_token(control_duration)} "
                f"pt={float(planning_time):.1f}s"
            ),
            goal_state=config["goal_state"],
            goal_threshold=float(config["goal_threshold"]),
        )
        callbacks.append(replay_recorder)

    if callbacks:

        def _on_planning_update(step_id: int, best_tr: dict, pose: np.ndarray) -> None:
            for cb in callbacks:
                cb(step_id, best_tr, pose)

        run_kwargs["on_planning_update"] = _on_planning_update

    aura_result = aura_runner.run(**run_kwargs)
    aura_run_wall_time = time.time() - aura_start
    if replay_recorder is not None:
        replay_recorder.save()
        if workspace_video_path:
            from scripts.plotting import render_workspace_replay

            render_workspace_replay(
                replay_recorder.replay_path,
                video_path=workspace_video_path,
                fps=float(workspace_video_fps),
                dpi=int(workspace_video_dpi),
            )
    aura_time = planning_wall_time + aura_run_wall_time
    collision_states = getattr(
        aura_result, "dense_states_trajectory", aura_result.states_trajectory
    )
    collision_step = first_collision_step(collision_states, config)
    goal_distance = float(
        arrayDistance(
            np.asarray(aura_result.final_state, dtype=float),
            np.asarray(config["goal_state"], dtype=float),
            system=SYSTEM_NAME,
        )
    )
    final_planned_state = getattr(aura_result, "final_planned_state", None)
    planned_goal_distance = (
        float(
            arrayDistance(
                np.asarray(final_planned_state, dtype=float),
                np.asarray(config["goal_state"], dtype=float),
                system=SYSTEM_NAME,
            )
        )
        if final_planned_state is not None
        else float("inf")
    )
    final_to_planned_distance = (
        float(
            arrayDistance(
                np.asarray(aura_result.final_state, dtype=float),
                np.asarray(final_planned_state, dtype=float),
                system=SYSTEM_NAME,
            )
        )
        if final_planned_state is not None
        else float("inf")
    )
    final_state_np = np.asarray(aura_result.final_state, dtype=float)
    final_planned_np = (
        np.asarray(final_planned_state, dtype=float)
        if final_planned_state is not None
        else np.full_like(final_state_np, np.nan, dtype=float)
    )
    goal_state_np = np.asarray(config["goal_state"], dtype=float)
    goal_threshold = float(config["goal_threshold"])
    final_plan_states = getattr(aura_result, "final_plan_states", None) or []
    print_status(
        "[final] actual "
        f"{format_state3(final_state_np)}  |  planned final "
        f"{format_state3(final_planned_np)}  |  goal "
        f"{format_state3(goal_state_np)}"
    )
    print_status(
        "[final] distances: actual->goal "
        f"{goal_distance:.6f}, planned->goal {planned_goal_distance:.6f}, "
        f"actual->planned {final_to_planned_distance:.6f}, "
        f"threshold {goal_threshold:.6f}"
    )
    if final_plan_states:
        print_status(
            f"[final plan] {len(final_plan_states)} states from AURA's final best plan:"
        )
        for plan_index, plan_state in enumerate(final_plan_states):
            marker = (
                "  <-- final planned state"
                if plan_index == len(final_plan_states) - 1
                else ""
            )
            print_status(
                f"[final plan] {plan_index:02d}: {format_state3(plan_state)}{marker}"
            )
    status = str(getattr(aura_result, "status", "success") or "success")
    failure_reason = str(getattr(aura_result, "failure_reason", "") or "")
    if collision_step is not None:
        print_status(
            f"[WARNING] Collision detected at executed step {collision_step}; "
            "marking trial as failure."
        )
        status = "failure"
        failure_reason = f"collision_step_{collision_step}"
        aura_run_wall_time = 360.0
        aura_time = 360.0
    elif status != "success":
        print_status(
            f"[WARNING] AURA stopped early ({failure_reason or 'unknown'}); "
            "marking trial as failure."
        )
        status = "failure"
        aura_run_wall_time = 360.0
        aura_time = 360.0
    elif goal_distance <= goal_threshold:
        pass
    elif final_to_planned_distance <= goal_threshold:
        print_status(
            "[INFO] Final state reached AURA's final planned state "
            f"(actual->planned {final_to_planned_distance:.6f} <= "
            f"threshold {goal_threshold:.6f}). actual->goal {goal_distance:.6f}; "
            f"planned->goal {planned_goal_distance:.6f}."
        )
    else:
        print_status(
            "[WARNING] Goal/planner final not reached "
            f"(actual->goal {goal_distance:.6f}, planned->goal "
            f"{planned_goal_distance:.6f}, actual->planned "
            f"{final_to_planned_distance:.6f}, threshold "
            f"{goal_threshold:.6f}); marking trial as failure."
        )
        status = "failure"
        failure_reason = (
            f"goal_not_reached_{goal_distance:.6f}_"
            f"final_to_planned_{final_to_planned_distance:.6f}"
        )
        aura_run_wall_time = 360.0
        aura_time = 360.0

    row = {
        "planner": planner_name,
        "planning_time": float(planning_time),
        "control_duration": float(control_duration),
        "status": status,
        "failure_reason": failure_reason,
        "collision_step": int(collision_step) if collision_step is not None else -1,
        "planning_wall_time": float(planning_wall_time),
        "aura_wall_time": float(aura_run_wall_time),
        "aura_time": float(aura_time),
        "cost": float(aura_result.cost),
        "tracking_error_mean": float(aura_result.tracking_error_mean),
        "goal_distance": goal_distance,
        "num_controls": int(aura_result.num_controls),
        "final_x": float(aura_result.final_state[0]),
        "final_y": float(aura_result.final_state[1]),
        "final_theta": float(aura_result.final_state[2]),
        "planned_final_x": float(final_planned_np[0]),
        "planned_final_y": float(final_planned_np[1]),
        "planned_final_theta": float(final_planned_np[2]),
    }
    row["planned_goal_distance"] = planned_goal_distance
    row["final_to_planned_distance"] = final_to_planned_distance
    return row, list(aura_result.tracking_error_list)


def save_csv(output_path: str, rows: list[dict]) -> None:
    fieldnames = [
        "run_number",
        "planner",
        "planning_time",
        "control_duration",
        "status",
        "failure_reason",
        "collision_step",
        "planning_wall_time",
        "aura_wall_time",
        "aura_time",
        "cost",
        "tracking_error_mean",
        "goal_distance",
        "planned_goal_distance",
        "final_to_planned_distance",
        "num_controls",
        "final_x",
        "final_y",
        "final_theta",
        "planned_final_x",
        "planned_final_y",
        "planned_final_theta",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
    try:
        os.sync()
    except AttributeError:
        pass


def _coerce_csv_row(row: dict) -> dict:
    int_fields = {"run_number", "collision_step", "num_controls"}
    float_fields = {
        "planning_time",
        "control_duration",
        "planning_wall_time",
        "aura_wall_time",
        "aura_time",
        "cost",
        "tracking_error_mean",
        "goal_distance",
        "planned_goal_distance",
        "final_to_planned_distance",
        "final_x",
        "final_y",
        "final_theta",
        "planned_final_x",
        "planned_final_y",
        "planned_final_theta",
    }
    out = dict(row)
    for key in int_fields:
        if key in out and out[key] not in (None, ""):
            try:
                out[key] = int(float(out[key]))
            except (TypeError, ValueError):
                pass
    for key in float_fields:
        if key in out and out[key] not in (None, ""):
            try:
                out[key] = float(out[key])
            except (TypeError, ValueError):
                pass
    return out


def load_csv_rows(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [_coerce_csv_row(row) for row in csv.DictReader(f)]


def upsert_planning_time_row(rows: list[dict], new_row: dict) -> list[dict]:
    new_pt = float(new_row["planning_time"])
    kept = [
        row
        for row in rows
        if abs(float(row.get("planning_time", float("nan"))) - new_pt) > 1e-9
    ]
    kept.append(new_row)
    kept.sort(key=lambda row: float(row.get("planning_time", float("inf"))))
    return kept


def make_failure_row(
    planner_name: str,
    planning_time: float,
    control_duration: float,
    reason: str,
) -> dict:
    reason = " ".join(str(reason).split())
    return {
        "planner": planner_name,
        "planning_time": float(planning_time),
        "control_duration": float(control_duration),
        "status": "failure",
        "failure_reason": reason[:240],
        "collision_step": -1,
        "planning_wall_time": 0.0,
        "aura_wall_time": 360.0,
        "aura_time": 360.0,
        "cost": float("nan"),
        "tracking_error_mean": float("nan"),
        "goal_distance": float("nan"),
        "planned_goal_distance": float("nan"),
        "final_to_planned_distance": float("nan"),
        "num_controls": 0,
        "final_x": float("nan"),
        "final_y": float("nan"),
        "final_theta": float("nan"),
        "planned_final_x": float("nan"),
        "planned_final_y": float("nan"),
        "planned_final_theta": float("nan"),
    }


class _SummaryLiveState:
    fig = None
    axes = None


def reset_summary_live_figure() -> None:
    """Call when starting a new sweep (e.g. new control_duration) so a fresh window is used."""
    if _SummaryLiveState.fig is not None:
        try:
            plt.close(_SummaryLiveState.fig)
        except Exception:
            pass
    _SummaryLiveState.fig = None
    _SummaryLiveState.axes = None


def _draw_sweep_summary_axes(axes, rows: list[dict]) -> None:
    planning_times = [row["planning_time"] for row in rows]
    tracking_errors = [row["tracking_error_mean"] for row in rows]
    aura_times = [row["aura_time"] for row in rows]
    costs = [row["cost"] for row in rows]
    goal_distances = [row["goal_distance"] for row in rows]
    num_controls = [row["num_controls"] for row in rows]

    axes[0, 0].plot(planning_times, tracking_errors, marker="o")
    axes[0, 0].set_title("AURA tracking error")
    axes[0, 0].set_xlabel("Initial planning time")
    axes[0, 0].set_ylabel("Mean tracking error")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(planning_times, aura_times, marker="o")
    axes[0, 1].set_title("AURA time")
    axes[0, 1].set_xlabel("Initial planning time")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(planning_times, goal_distances, marker="o")
    axes[1, 0].set_title("Goal distance")
    axes[1, 0].set_xlabel("Initial planning time")
    axes[1, 0].set_ylabel("Distance")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(planning_times, num_controls, marker="o")
    axes[1, 1].set_title("Executed controls")
    axes[1, 1].set_xlabel("Initial planning time")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(planning_times, costs, marker="o")
    axes[2, 0].set_title("AURA cost")
    axes[2, 0].set_xlabel("Initial planning time")
    axes[2, 0].set_ylabel("Cost")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].axis("off")


def save_summary_plot(
    plot_path: str, rows: list[dict], *, show: bool = True, live: bool = True
) -> None:
    """
    3x2 summary metrics. *live* reuses one window and refreshes after each new row
    (needs a GUI backend). *show* = False only writes PNGs.
    """
    ab = os.path.abspath(plot_path)
    if not show:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        _draw_sweep_summary_axes(axes, rows)
        fig.suptitle("AURA initial-time sweep (summary)", fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        return

    if _CONFIGURED_BACKEND == "Agg":
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        _draw_sweep_summary_axes(axes, rows)
        fig.suptitle("AURA initial-time sweep", fontsize=11)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(
            f"[plotting] Wrote (no on-screen display; use --no-show to silence): {ab}"
        )
        return

    if not live:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        _draw_sweep_summary_axes(axes, rows)
        fig.suptitle("AURA initial-time sweep (modal — close to continue)", fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        try:
            plt.show(block=True)
        except Exception as e:
            print(f"[plotting] Could not open window: {e!s}. File: {ab}")
        finally:
            plt.close(fig)
        return

    if _SummaryLiveState.fig is None or _SummaryLiveState.axes is None:
        _SummaryLiveState.fig, _SummaryLiveState.axes = plt.subplots(
            3, 2, figsize=(10, 10)
        )
    fig, axes = _SummaryLiveState.fig, _SummaryLiveState.axes
    for a in np.ravel(axes):
        a.clear()
    _draw_sweep_summary_axes(axes, rows)
    fig.suptitle("AURA sweep (live) — each row adds a planning point", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    if not plt.isinteractive():
        plt.ion()
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=False)
        plt.pause(0.25)
    except Exception as e:
        print(f"[plotting] Live window: {e!s}. See PNG: {ab}")
    else:
        print(f"[plotting] Updated live summary → {ab}")


def configure_ompl_logging() -> None:
    try:
        ou.setLogLevel(ou.LOG_WARN)
    except Exception:
        pass


def print_optimizer_device_info(*, require_gpu: bool = False) -> None:
    info = optimizer_device_info()
    device = str(info.get("device", "cpu"))
    cuda_visible = str(info.get("cuda_visible_devices", "<unset>"))
    details = (
        f"[optimizer] device={device}"
        f"  torch={info.get('torch_version')}"
        f"  torch_cuda={info.get('torch_cuda')}"
        f"  cuda_available={info.get('cuda_available')}"
        f"  device_count={info.get('device_count')}"
        f"  CUDA_VISIBLE_DEVICES={cuda_visible}"
    )
    if str(device).startswith("cuda"):
        name = str(info.get("device_name") or "")
        print_success(f"{details}  name={name}")
        try:
            warmup_optimizer_device(SYSTEM_NAME)
            print_success("[optimizer] CUDA warmup complete")
        except Exception as exc:
            if require_gpu:
                raise RuntimeError("CUDA optimizer warmup failed.") from exc
            print_status(f"[optimizer] CUDA warmup failed: {exc!r}")
    else:
        error = str(info.get("error") or "")
        suffix = f"  cuda_error={error}" if error else ""
        print_status(f"{details}{suffix}")
        if require_gpu:
            raise RuntimeError(
                "--require-gpu was set, but the optimizer selected CPU. "
                "The startup device line above shows the CUDA visibility state."
            )


def run_missing_replay_grid(
    args,
    control_durations: list[float],
    planning_times: np.ndarray,
    *,
    show_summary: bool,
    summary_live: bool,
) -> None:
    run_numbers = (
        [int(args.run_number)]
        if args.run_number is not None
        else list(range(1, int(args.num_runs) + 1))
    )
    expected: list[tuple[float, int, float, str]] = []
    missing: list[tuple[float, int, float, str]] = []
    requested_keys: set[tuple[float, int, float]] = set()

    for run_number in run_numbers:
        for control_duration in control_durations:
            for planning_time in planning_times:
                replay_path = workspace_replay_path(
                    args.results_dir,
                    args.planner_name,
                    float(control_duration),
                    int(run_number),
                    float(planning_time),
                )
                item = (
                    round(float(control_duration), 2),
                    int(run_number),
                    float(planning_time),
                    replay_path,
                )
                expected.append(item)
                requested_keys.add(
                    (
                        round(float(control_duration), 2),
                        int(run_number),
                        round(float(planning_time), 2),
                    )
                )
                if not replay_is_complete(replay_path):
                    missing.append(item)

    replay_dir = os.path.join(args.results_dir, "workspace_replays")
    parsed_replays: list[tuple[str, float, int, float, str]] = []
    if os.path.isdir(replay_dir):
        for replay_file in glob.glob(os.path.join(replay_dir, "*.npz")):
            parsed = parse_workspace_replay_filename(replay_file)
            if parsed is None:
                continue
            planner, control_duration, run_number, planning_time = parsed
            parsed_replays.append(
                (
                    planner,
                    round(float(control_duration), 2),
                    int(run_number),
                    round(float(planning_time), 2),
                    replay_file,
                )
            )
    planner_replays = [
        item for item in parsed_replays if item[0] == str(args.planner_name)
    ]
    matching_replays = [
        item for item in planner_replays if (item[1], item[2], item[3]) in requested_keys
    ]
    outside_replays = [
        item for item in planner_replays if (item[1], item[2], item[3]) not in requested_keys
    ]
    planning_time_counts = Counter(item[3] for item in planner_replays)

    print_section("Workspace replay resume scan")
    print_status(f"Replay files in folder: {len(parsed_replays)}")
    print_status(f"Replay files for planner '{args.planner_name}': {len(planner_replays)}")
    if planning_time_counts:
        counts = ", ".join(
            f"{pt:.2f}s:{count}" for pt, count in sorted(planning_time_counts.items())
        )
        print_status(f"Planner files by planning time: {counts}")
    print_status(f"Files matching requested grid: {len(matching_replays)}")
    if outside_replays:
        print_status(
            "Files outside requested grid: "
            f"{len(outside_replays)} (kept, but ignored by this fill pass)"
        )
    print_status(f"Expected replay trials: {len(expected)}")
    print_status(f"Complete replay files:  {len(expected) - len(missing)}")
    print_status(f"Missing/corrupt files:  {len(missing)}")
    if missing:
        preview = ", ".join(
            f"cd{control_duration_token(cd)}/r{run_number:02d}/pt{planning_time_token(pt)}"
            for cd, run_number, pt, _ in missing[:12]
        )
        more = "" if len(missing) <= 12 else f", ... +{len(missing) - 12} more"
        print_status(f"First missing: {preview}{more}")
    if args.missing_replays_file:
        write_missing_replay_file(args.missing_replays_file, missing)
        print_status(f"Missing replay list: {args.missing_replays_file}")

    if bool(args.dry_run_missing):
        return

    if not missing:
        print_success("Nothing to run; workspace_replays already contains the requested grid.")
        return

    for missing_idx, (control_duration, run_number, planning_time, replay_path) in enumerate(
        missing,
        start=1,
    ):
        print_section(
            (
                f"Missing replay {missing_idx}/{len(missing)} | "
                f"Planner={args.planner_name} | control_duration={control_duration} | "
                f"run={run_number:02d}"
            )
        )
        print_step(f"Planning time {planning_time:.1f}s")

        csv_path = (
            f"{args.results_dir}/{SYSTEM_NAME}_{args.planner_name}_"
            f"cd{control_duration_token(control_duration)}_{int(run_number):02d}.csv"
        )
        plot_path = (
            f"{args.results_dir}/{SYSTEM_NAME}_{args.planner_name}_"
            f"cd{control_duration_token(control_duration)}_{int(run_number):02d}.png"
        )
        rows = load_csv_rows(csv_path)
        ws_dir = (
            workspace_frames_dir(
                args.results_dir,
                args.planner_name,
                control_duration,
                run_number,
                planning_time,
            )
            if args.workspace_frames
            else None
        )
        opt_loss_dir = (
            optimization_loss_dir(
                args.results_dir,
                args.planner_name,
                control_duration,
                run_number,
                planning_time,
            )
            if args.opt_loss_frames
            else None
        )
        video_path = (
            workspace_video_path(
                args.results_dir,
                args.planner_name,
                control_duration,
                run_number,
                planning_time,
            )
            if args.workspace_video
            else None
        )

        try:
            row, _ = run_single_experiment(
                planner_name=args.planner_name,
                planning_time=float(planning_time),
                control_duration=float(control_duration),
                workspace_frames_dir=ws_dir,
                workspace_live=show_summary and summary_live,
                workspace_replay_path=replay_path,
                workspace_video_path=video_path,
                workspace_video_fps=float(args.workspace_video_fps),
                workspace_video_dpi=int(args.workspace_video_dpi),
                optimization_loss_plot_dir=opt_loss_dir,
                show_optimization_loss_plot=bool(args.show_opt_loss)
                and _CONFIGURED_BACKEND != "Agg",
            )
        except Exception as exc:
            print_status(f"[ERROR] Trial failed before replay save: {exc}")
            traceback.print_exc()
            row = make_failure_row(
                args.planner_name,
                float(planning_time),
                float(control_duration),
                f"{type(exc).__name__}: {exc}",
            )

        row["run_number"] = int(run_number)
        rows = upsert_planning_time_row(rows, row)
        save_csv(csv_path, rows)
        save_summary_plot(plot_path, rows, show=show_summary, live=summary_live)

        if replay_is_complete(replay_path):
            print_success(f"Filled replay: {os.path.abspath(replay_path)}")
        else:
            print_status(
                "[WARNING] Replay still missing after this attempt; it will be retried "
                "the next time fill mode runs."
            )

    print_success("Missing-replay fill pass finished.")


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Shared sweep YAML config.",
    )
    config_args, _ = config_parser.parse_known_args()
    sweep_config = load_experiment_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Sweep AURA over initial planning times for kinematic_car with gaussian noise.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--planner-name",
        type=str,
        default=str(sweep_config["planner_name"]),
        choices=["aorrt", "aoest", "sststar"],
        help="Planner to use for the AURA runs.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(sweep_config["results_dir"]),
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=None,
        help="Optional run number. If omitted, the next available run number is used.",
    )
    parser.add_argument(
        "--control-durations",
        type=float,
        nargs="+",
        default=list(sweep_config["control_durations"]),
        help="Control durations to sweep.",
    )
    parser.add_argument(
        "--planning-times",
        type=float,
        nargs="+",
        default=list(sweep_config["planning_times"]),
        help="Initial planning times in seconds to sweep.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=int(sweep_config["num_runs"]),
        help=(
            "Numbered repeats used by --fill-missing-replays. Expected run numbers "
            "are 1..N unless --run-number is provided."
        ),
    )
    parser.add_argument(
        "--fill-missing-replays",
        action="store_true",
        help=(
            "Scan results-dir/workspace_replays for the requested grid and run only "
            "missing or unreadable replay files."
        ),
    )
    parser.add_argument(
        "--dry-run-missing",
        action="store_true",
        help="With --fill-missing-replays, only print the missing replay slots.",
    )
    parser.add_argument(
        "--missing-replays-file",
        default=None,
        help=(
            "With --fill-missing-replays, write missing slots as TSV rows: "
            "control_duration, run_number, planning_time, replay_path."
        ),
    )
    parser.add_argument(
        "--motion-validation-step-size",
        type=float,
        default=float(BASE_CONFIG["motion_validation_step_size"]),
        help="Internal propagation substep used for curve collision checking.",
    )
    parser.add_argument(
        "--optimizer-num-states",
        type=int,
        default=int(BASE_CONFIG["optimizer_num_states"]),
        help="Number of sampled states per optimizer call.",
    )
    parser.add_argument(
        "--optimizer-num-steps",
        type=int,
        default=int(BASE_CONFIG["optimizer_num_steps"]),
        help="Adam steps per optimizer call.",
    )
    parser.add_argument(
        "--optimizer-learning-rate",
        type=float,
        default=float(BASE_CONFIG["optimizer_learning_rate"]),
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail before running if PyTorch cannot use CUDA for the optimizer.",
    )
    parser.add_argument(
        "--recovery-replanning-time",
        type=float,
        default=float(BASE_CONFIG["recovery_replanning_time"]),
        help="Fresh recovery planning budget used only when both next-control candidates are invalid.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Backward-compatible alias for showing plots (now default unless --no-show).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Headless: no plot windows, only write PNG/CSV (uses Agg). Default is to show plots on screen when possible.",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Use a blocking (modal) plot each time instead of a single live-refreshing window. Implies a window to close after each sweep step if only one point.",
    )
    parser.add_argument(
        "--workspace-frames",
        action="store_true",
        help="Save workspace PNGs (obstacles, path, pose) each AURA replan; combine with not --no-live to also show that view live.",
    )
    parser.add_argument(
        "--workspace-replay",
        action="store_true",
        help="Save compact .npz workspace replay data for each trial; can be rendered later without rerunning AURA.",
    )
    parser.add_argument(
        "--workspace-video",
        action="store_true",
        help="Save a compact workspace replay and immediately render an MP4 video for each trial.",
    )
    parser.add_argument(
        "--workspace-video-fps",
        type=float,
        default=3.0,
        help="Frames per second for videos rendered from workspace replay data.",
    )
    parser.add_argument(
        "--workspace-video-dpi",
        type=int,
        default=150,
        help="DPI for workspace replay video rendering.",
    )
    parser.add_argument(
        "--opt-loss-frames",
        action="store_true",
        help="Save optimizer MSE vs Adam step (loss_step_####.png) under results-dir/optimization_loss/... for each trial.",
    )
    parser.add_argument(
        "--show-opt-loss",
        action="store_true",
        help="Live matplotlib window updating optimizer loss each AURA step (skipped if backend is Agg).",
    )
    args = parser.parse_args()
    if bool(args.fill_missing_replays):
        args.workspace_replay = True
    BASE_CONFIG["motion_validation_step_size"] = float(args.motion_validation_step_size)
    BASE_CONFIG["optimizer_num_states"] = int(args.optimizer_num_states)
    BASE_CONFIG["optimizer_num_steps"] = int(args.optimizer_num_steps)
    BASE_CONFIG["optimizer_learning_rate"] = float(args.optimizer_learning_rate)
    BASE_CONFIG["recovery_replanning_time"] = float(args.recovery_replanning_time)
    control_durations = [float(x) for x in args.control_durations]
    planning_times = np.asarray([float(x) for x in args.planning_times], dtype=float)
    show_summary = bool(args.show) or (not bool(args.no_show))
    summary_live = show_summary and not bool(args.no_live)

    if SYSTEM_NAME != "kinematic_car" or SIMULATOR_MODE != "gaussian":
        raise RuntimeError(
            "initial_time_experiment.py only supports kinematic_car with gaussian noise."
        )
    if len(control_durations) == 0:
        raise ValueError("control_durations must contain at least one entry")
    if len(planning_times) == 0:
        raise ValueError("planning_times must contain at least one entry")
    if int(args.num_runs) <= 0:
        raise ValueError("num-runs must be positive")
    if any(float(duration) <= 0.0 for duration in control_durations):
        raise ValueError("all control_durations must be positive")
    if float(BASE_CONFIG["motion_validation_step_size"]) <= 0.0:
        raise ValueError("motion_validation_step_size must be positive")
    if int(BASE_CONFIG["optimizer_num_states"]) <= 0:
        raise ValueError("optimizer_num_states must be positive")
    if int(BASE_CONFIG["optimizer_num_steps"]) <= 0:
        raise ValueError("optimizer_num_steps must be positive")
    if float(BASE_CONFIG["recovery_replanning_time"]) <= 0.0:
        raise ValueError("recovery_replanning_time must be positive")
    if float(args.workspace_video_fps) <= 0.0:
        raise ValueError("workspace-video-fps must be positive")
    if int(args.workspace_video_dpi) <= 0:
        raise ValueError("workspace-video-dpi must be positive")
    print_optimizer_device_info(require_gpu=bool(args.require_gpu))

    configure_ompl_logging()
    os.makedirs(args.results_dir, exist_ok=True)
    reset_summary_live_figure()
    reset_optimizer_loss_live_figure()
    if bool(args.fill_missing_replays):
        run_missing_replay_grid(
            args,
            control_durations,
            planning_times,
            show_summary=show_summary,
            summary_live=summary_live,
        )
        if show_summary and summary_live and _CONFIGURED_BACKEND != "Agg":
            print_status("Live plot ready. Close the figure window to exit.")
            try:
                plt.ioff()
                plt.show(block=True)
            except Exception as e:
                print_status(f"[plotting] Could not keep window open: {e}")
        return

    for control_duration in control_durations:
        run_number = (
            int(args.run_number)
            if args.run_number is not None
            else get_next_run_number(
                args.results_dir, args.planner_name, float(control_duration)
            )
        )
        rows: list[dict] = []
        cd_token = control_duration_token(control_duration)
        csv_path = f"{args.results_dir}/{SYSTEM_NAME}_{args.planner_name}_cd{cd_token}_{run_number:02d}.csv"
        plot_path = f"{args.results_dir}/{SYSTEM_NAME}_{args.planner_name}_cd{cd_token}_{run_number:02d}.png"
        print_section(
            f"Planner={args.planner_name} | control_duration={cd_token} | run={run_number:02d}"
        )

        for step_idx, planning_time in enumerate(planning_times, start=1):
            print_step(
                f"Planning time {float(planning_time):.1f}s ({step_idx}/{len(planning_times)})"
            )
            ws_dir = None
            if args.workspace_frames:
                ws_dir = workspace_frames_dir(
                    args.results_dir,
                    args.planner_name,
                    float(control_duration),
                    int(run_number),
                    float(planning_time),
                )
            opt_loss_dir = None
            if args.opt_loss_frames:
                opt_loss_dir = optimization_loss_dir(
                    args.results_dir,
                    args.planner_name,
                    float(control_duration),
                    int(run_number),
                    float(planning_time),
                )
            replay_path = None
            video_path = None
            if args.workspace_replay or args.workspace_video:
                replay_path = workspace_replay_path(
                    args.results_dir,
                    args.planner_name,
                    float(control_duration),
                    int(run_number),
                    float(planning_time),
                )
            if args.workspace_video:
                video_path = workspace_video_path(
                    args.results_dir,
                    args.planner_name,
                    float(control_duration),
                    int(run_number),
                    float(planning_time),
                )
            try:
                row, _ = run_single_experiment(
                    planner_name=args.planner_name,
                    planning_time=float(planning_time),
                    control_duration=float(control_duration),
                    workspace_frames_dir=ws_dir,
                    workspace_live=show_summary and summary_live,
                    workspace_replay_path=replay_path,
                    workspace_video_path=video_path,
                    workspace_video_fps=float(args.workspace_video_fps),
                    workspace_video_dpi=int(args.workspace_video_dpi),
                    optimization_loss_plot_dir=opt_loss_dir,
                    show_optimization_loss_plot=bool(args.show_opt_loss)
                    and _CONFIGURED_BACKEND != "Agg",
                )
            except Exception as exc:
                print_status(f"[ERROR] Trial failed: {exc}")
                traceback.print_exc()
                row = make_failure_row(
                    args.planner_name,
                    float(planning_time),
                    float(control_duration),
                    f"{type(exc).__name__}: {exc}",
                )
            row["run_number"] = run_number
            rows.append(row)
            save_csv(csv_path, rows)
            save_summary_plot(
                plot_path,
                rows,
                show=show_summary,
                live=summary_live,
            )
            print_status(
                f"Saved intermediate results to {csv_path}  |  summary figure: {plot_path}"
            )
            if args.workspace_frames and ws_dir:
                print_status(f"Workspace frames: {os.path.abspath(ws_dir)}")
            if replay_path:
                print_status(f"Workspace replay: {os.path.abspath(replay_path)}")
            if video_path and os.path.exists(video_path):
                print_status(f"Workspace video: {os.path.abspath(video_path)}")
            elif video_path:
                fallback_frames = (
                    os.path.splitext(os.path.abspath(video_path))[0] + "_frames"
                )
                if os.path.isdir(fallback_frames):
                    print_status(
                        "Workspace video writer unavailable; replay frames: "
                        f"{fallback_frames}"
                    )
            if args.opt_loss_frames and opt_loss_dir:
                print_status(f"Optimizer loss plots: {os.path.abspath(opt_loss_dir)}")

        print_success(f"Results saved to {csv_path}")
        print_success(f"Summary plot saved to {plot_path}")

    # Keep live figure visible after script finishes, especially when only one
    # planning point is plotted (otherwise the process exits immediately).
    if show_summary and summary_live and _CONFIGURED_BACKEND != "Agg":
        print_status("Live plot ready. Close the figure window to exit.")
        try:
            plt.ioff()
            plt.show(block=True)
        except Exception as e:
            print_status(f"[plotting] Could not keep window open: {e}")


if __name__ == "__main__":
    main()
