#!/usr/bin/env python3
"""Generate annotated videos for AURA-AORRT, Replanning-AORRT, and MPPI.

The videos are rendered as a clean top-down workspace view so we can add method
status, the chosen plan/control, obstacles, and the goal region consistently.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aura_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, PillowWriter, writers
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.transforms import Affine2D
from ompl import base as ob
from ompl import util as ou

from AURA import AURA
from Replanning import ReplanningRunner
from experiments.initial_time_experiment import BASE_CONFIG, build_planner
from plan import OMPL_Planner
from simulation.pushing_dynamics import get_pushing_model
from simulation.pushing_object_specs import CRACKER_BOX_FLIPPED_SHAPE
from simulation.simulators import create_simulator
from systems import get_system
from train_model import load_opt_model_2
from utils.utils import arrayDistance, is_state_array_valid, sample_control_curve

ABSTRACT_FONT_FAMILY = "Times New Roman"
ABSTRACT_TEXT_DARK = "#17202A"
ABSTRACT_TEXT_MUTED = "#55616D"
ABSTRACT_SIDE_X = 1.08

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [ABSTRACT_FONT_FAMILY, "Times", "DejaVu Serif"],
        "font.size": 15,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
    }
)


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(int(seed))
    try:
        ou.RNG.setSeed(int(seed))
    except Exception:
        pass


@dataclass
class VideoFrame:
    task: str
    method: str
    status: str
    step: int
    pose: np.ndarray
    actual_path: list[np.ndarray]
    current_plan: list[np.ndarray]
    initial_plan: list[np.ndarray]
    candidate_paths: list[list[np.ndarray]]
    distance_to_goal: float
    chosen_control: np.ndarray | None


def _blend_angle(a: float, b: float, alpha: float) -> float:
    return float(a + alpha * _wrap_angle(b - a))


def _interpolate_pose(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)[:3]
    b = np.asarray(b, dtype=float).reshape(-1)[:3]
    out = a + float(alpha) * (b - a)
    out[2] = _blend_angle(float(a[2]), float(b[2]), float(alpha))
    return out


def _car_config(control_duration: float) -> dict:
    config = deepcopy(BASE_CONFIG)
    config["task"] = "car"
    config["system_name"] = "kinematic_car"
    config["start_state"] = np.asarray([-0.16, 0.0, 0.0], dtype=float)
    config["goal_state"] = np.asarray([3.0, 4.0, 1.57], dtype=float)
    config["state_bounds"] = [(-0.75, 3.45), (-0.65, 5.45)]
    config["control_duration_seconds"] = 1.0
    config["propagation_step_size"] = 1.0
    config["min_control_duration"] = 1
    config["max_control_duration"] = 1
    config["goal_threshold"] = 0.1
    config["visual_goal_threshold"] = 0.3
    config["mppi_goal_threshold"] = config["visual_goal_threshold"]
    config["obstacles"] = {"circles": [], "aabbs": [], "safety_radius": 0.0}
    config["replanningMaxDistance"] = 0.25
    config["replanning_time_budget"] = 1.0
    config["execution_timeout_slack"] = 2.0
    config["mujoco_car_throttle_ctrl_scale"] = 0.07
    config["mujoco_car_steering_ctrl_scale"] = 0.85
    config["sampling_position_std"] = 0.05
    config["sampling_rotation_std"] = 0.20
    config["sampling_velocity_std"] = 0.003
    config["optimizer_num_states"] = 10000
    config["optimizer_num_steps"] = 250
    config["optimizer_learning_rate"] = 0.005
    config["optimizer_max_children"] = 1000
    config["visualize"] = False
    return config


def _pushing_config(control_duration: float) -> dict:
    system = get_system("pushing_object")
    return {
        "task": "pushing",
        "system_name": "pushing_object",
        "start_state": np.asarray([0.0, -0.60, -np.pi / 3], dtype=float),
        "goal_state": np.asarray([0.72, -0.60, -np.pi / 2], dtype=float),
        "state_bounds": [(float(a), float(b)) for a, b in system.state_bounds],
        "control_duration_seconds": 2.0,
        "propagation_step_size": 2.0,
        "min_control_duration": 1,
        "max_control_duration": 1,
        "control_bounds": [(0.0, 0.75), (-0.4, 0.4), (0.0, 0.12)],
        "goal_threshold": 0.15,
        "visual_goal_threshold": 0.09,
        "mppi_goal_threshold": 0.09,
        "replanning_time_budget": 2.0,
        "replanningMaxDistance": 50.0,
        "optimizer_num_states": 2000,
        "optimizer_num_steps": 300,
        "optimizer_learning_rate": 5e-3,
        "sampling_position_std": 0.035,
        "sampling_rotation_std": 0.2,
        "sampling_velocity_std": 0.003,
        "optimizer_max_children": 100,
        "mujoco_camera_distance": 1.35,
        "mujoco_camera_azimuth": 180.0,
        "mujoco_camera_elevation": -35.0,
        "obstacles": {"circles": [], "aabbs": [], "safety_radius": 0.0},
        "visualize": False,
    }


def _config(task: str, control_duration: float) -> dict:
    if task == "car":
        return _car_config(control_duration)
    if task == "pushing":
        return _pushing_config(control_duration)
    raise ValueError(f"Unknown task: {task}")


def _apply_optimizer_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.optimizer_num_states is not None:
        config["optimizer_num_states"] = int(args.optimizer_num_states)
    if args.optimizer_num_steps is not None:
        config["optimizer_num_steps"] = int(args.optimizer_num_steps)
    if args.optimizer_learning_rate is not None:
        config["optimizer_learning_rate"] = float(args.optimizer_learning_rate)
    if args.optimizer_pos_std is not None:
        config["sampling_position_std"] = float(args.optimizer_pos_std)
    if args.optimizer_rot_std is not None:
        config["sampling_rotation_std"] = float(args.optimizer_rot_std)
    if args.optimizer_vel_std is not None:
        config["sampling_velocity_std"] = float(args.optimizer_vel_std)
    if args.optimizer_max_children is not None:
        config["optimizer_max_children"] = min(
            64, max(1, int(args.optimizer_max_children))
        )


def _system_name_alias(name: str) -> str:
    aliases = {
        "car": "kinematic_car",
        "simple_car": "kinematic_car",
        "kinematic_car": "kinematic_car",
        "pushing": "pushing_object",
        "push": "pushing_object",
        "pushing_object": "pushing_object",
    }
    key = str(name).strip().lower()
    if key not in aliases:
        raise ValueError(
            f"Unknown system '{name}'. Use kinematic_car or pushing_object."
        )
    return aliases[key]


def _apply_mppi_goal_threshold_cli_overrides(
    config: dict, raw_overrides: list[str]
) -> None:
    if not raw_overrides:
        return
    system_name = _system_name_alias(config["system_name"])
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError(
                "--mppi-goal-threshold must be SYSTEM=VALUE, "
                "for example kinematic_car=0.30"
            )
        raw_system, raw_value = raw.split("=", 1)
        if _system_name_alias(raw_system) != system_name:
            continue
        value = float(raw_value)
        if value <= 0.0:
            raise ValueError("MPPI goal threshold must be positive.")
        config["mppi_goal_threshold"] = value


def _method_requested(requested: str, method: str) -> bool:
    requested = requested.lower().replace("_", "-")
    method = method.lower().replace("_", "-")
    aliases = {
        "aura": "aura",
        "aura-aorrt": "aura",
        "replanning": "replanning",
        "replanning-aorrt": "replanning",
        "mppi": "mppi",
        "baseline": "baseline",
        "open-loop": "baseline",
        "openloop": "baseline",
        "open-loop-aorrt": "baseline",
        "openloop-aorrt": "baseline",
    }
    requested = aliases.get(requested, requested)
    method = aliases.get(method, method)
    return requested == "all" or requested == method


def _apply_bounds(system, state_bounds):
    system.state_bounds = [(float(a), float(b)) for a, b in state_bounds]
    bounds = ob.RealVectorBounds(2)
    for i, (low, high) in enumerate(system.state_bounds):
        bounds.setLow(i, float(low))
        bounds.setHigh(i, float(high))
    system.state_space.setBounds(bounds)


def _apply_control_bounds(system, control_bounds):
    if not control_bounds:
        return
    if len(control_bounds) != len(system.control_bounds):
        raise ValueError(
            f"{system.name} expected {len(system.control_bounds)} control bounds, "
            f"got {len(control_bounds)}"
        )
    system.control_bounds = [(float(a), float(b)) for a, b in control_bounds]
    bounds = ob.RealVectorBounds(len(system.control_bounds))
    for i, (low, high) in enumerate(system.control_bounds):
        bounds.setLow(i, float(low))
        bounds.setHigh(i, float(high))
    system.control_space.setBounds(bounds)


def _as_state_array_list(
    states, system_name: str = "kinematic_car"
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for state in states or []:
        if isinstance(state, np.ndarray):
            arr = np.asarray(state, dtype=float).reshape(-1)
        elif isinstance(state, (list, tuple)):
            arr = np.asarray(state, dtype=float).reshape(-1)
        elif hasattr(state, "getX"):
            arr = np.asarray([state.getX(), state.getY(), state.getYaw()], dtype=float)
        else:
            arr = np.asarray(state, dtype=float).reshape(-1)
        out.append(arr[:3].copy())
    return out


def _copy_solution(solution: dict) -> dict:
    copied = dict(solution)
    for key in ("states", "controls"):
        if copied.get(key) is not None:
            copied[key] = [np.asarray(x, dtype=float).copy() for x in copied[key]]
    if copied.get("time") is not None:
        copied["time"] = [float(t) for t in copied["time"]]
    return copied


def _plan_states(plan: dict | None) -> list[np.ndarray]:
    if not plan:
        return []
    return _as_state_array_list(plan.get("states", []))


def _plan_candidates(plan: dict | None, limit: int = 6) -> list[list[np.ndarray]]:
    if not plan:
        return []
    candidates = []
    for item in (plan.get("candidate_paths") or [])[:limit]:
        candidates.append(_as_state_array_list(item.get("states", [])))
    for item in (plan.get("goal_solution_paths") or [])[
        : max(0, limit - len(candidates))
    ]:
        candidates.append(_as_state_array_list(item.get("states", [])))
    return candidates


def _best_goal_solution_path(plan: dict | None) -> list[np.ndarray]:
    if not plan:
        return []
    goal_paths = plan.get("goal_solution_paths") or []
    if not goal_paths:
        return []
    best = min(
        goal_paths,
        key=lambda item: float(item.get("cost", float("inf"))),
    )
    return _plan_states(best)


def _display_states(plan: dict | None) -> list[np.ndarray]:
    if not plan:
        return []
    return _as_state_array_list(plan.get("display_states", []))


def _display_plan(
    raw_plan: list[np.ndarray],
    previous_plan: list[np.ndarray],
    current: np.ndarray,
    continuity_threshold: float = 0.70,
) -> list[np.ndarray]:
    if not raw_plan:
        return []
    current = np.asarray(current, dtype=float).reshape(-1)[:3]
    raw = [current.copy()] + [
        np.asarray(s, dtype=float).reshape(-1)[:3].copy() for s in raw_plan[1:]
    ]
    if len(previous_plan) < 3:
        return raw

    prev = [np.asarray(s, dtype=float).reshape(-1)[:3] for s in previous_plan]
    distances = [np.linalg.norm(p[:2] - current[:2]) for p in prev]
    trim_idx = int(np.argmin(distances))
    if distances[trim_idx] <= continuity_threshold and trim_idx < len(prev) - 1:
        return [current.copy()] + [p.copy() for p in prev[trim_idx + 1 :]]
    return raw


def _mujoco_display_plan(
    raw_plan: list[np.ndarray],
    current: np.ndarray,
) -> list[np.ndarray]:
    """Return a MuJoCo overlay path anchored at the executed XY pose.

    This is deliberately visual-only: trim by XY distance, not the OMPL metric.
    For pushing, heading can dominate the OMPL distance and make the overlay jump
    to the goal-side tail even when the object is still in the middle of the table.
    """
    raw = [
        np.asarray(s, dtype=float).reshape(-1)[:3].copy()
        for s in (raw_plan or [])
        if np.asarray(s, dtype=float).size >= 2
    ]
    if not raw:
        return []

    current_np = np.asarray(current, dtype=float).reshape(-1)[:3].copy()
    if len(raw) == 1:
        return [current_np, raw[0].copy()]

    xy_dists = [float(np.linalg.norm(s[:2] - current_np[:2])) for s in raw]
    closest_idx = int(np.argmin(xy_dists))
    if closest_idx >= len(raw) - 1:
        closest_idx = max(0, len(raw) - 2)

    display = [current_np]
    display.extend(s.copy() for s in raw[closest_idx + 1 :])
    return display


def _animated_frames(frames: list[VideoFrame], fps: float) -> list[VideoFrame]:
    if len(frames) <= 1:
        return frames
    out: list[VideoFrame] = []
    for idx, frame in enumerate(frames[:-1]):
        nxt = frames[idx + 1]
        repeats = max(1, int(round(float(fps))))
        for j in range(repeats):
            alpha = float(j) / float(repeats)
            pose = _interpolate_pose(frame.pose, nxt.pose, alpha)
            actual_path = list(frame.actual_path)
            actual_path.append(pose.copy())
            out.append(
                VideoFrame(
                    task=frame.task,
                    method=frame.method,
                    status=frame.status,
                    step=frame.step,
                    pose=pose,
                    actual_path=actual_path,
                    current_plan=frame.current_plan,
                    initial_plan=frame.initial_plan,
                    candidate_paths=frame.candidate_paths,
                    distance_to_goal=float(
                        (1.0 - alpha) * frame.distance_to_goal
                        + alpha * nxt.distance_to_goal
                    ),
                    chosen_control=frame.chosen_control,
                )
            )
    out.append(frames[-1])
    return out


def _path_from_controls(
    system, start: np.ndarray, controls, duration: float
) -> list[np.ndarray]:
    states = [np.asarray(start, dtype=float).reshape(-1)[:3].copy()]
    state = states[0].copy()
    for control in controls or []:
        control_np = np.asarray(control, dtype=float)
        if system.name == "pushing_object":
            state = _normalize_se2_state(system.propagate(state, control_np, duration))
            states.append(state.copy())
            continue
        curve = sample_control_curve(
            system,
            state,
            control_np,
            duration,
            max(0.03, duration / 25.0),
            include_start=False,
        )
        if curve:
            states.extend(
                np.asarray(s, dtype=float).reshape(-1)[:3].copy() for s in curve
            )
            state = states[-1].copy()
    return states


def _valid_state(config: dict, state: np.ndarray) -> bool:
    return is_state_array_valid(
        state,
        system=config.get("system_name", "kinematic_car"),
        config=config,
        obstacle_config=config.get("obstacles"),
        safety_radius_override=float(config.get("execution_safety_radius", 0.0)),
    )


def _get_pushing_opt_model(system, config: dict):
    if system.name != "pushing_object":
        return None
    if config.get("_pushing_opt_model") is None:
        model = get_pushing_model(system.object_shape)
        config["_pushing_opt_model"] = load_opt_model_2(model)
    return config["_pushing_opt_model"]


def _build_planner_from_state(
    system, config: dict, start_state: np.ndarray, planning_time: float
) -> OMPL_Planner:
    planner_start = _normalize_se2_state(start_state)
    planner_goal = _normalize_se2_state(config["goal_state"])
    if system.name == "kinematic_car":
        planner = build_planner(system, config, "aorrt", planning_time)
        planner.start_state = planner_start.copy()
        planner.goal_state = planner_goal.copy()
        planner.optimizer_max_children = int(config.get("optimizer_max_children", 0))
        return planner
    planner = OMPL_Planner(
        system=system,
        start_state=planner_start.copy(),
        goal_state=planner_goal.copy(),
        planner_method="aorrt",
        goal_threshold=float(config.get("goal_threshold", 0.1)),
        min_max_control_duration=(
            int(config.get("min_control_duration", 1)),
            int(config.get("max_control_duration", 1)),
        ),
        propagation_step_size=float(config.get("propagation_step_size", 1.0)),
        initial_planning_time=float(planning_time),
        pruning_radius=float(config.get("pruning_radius", 0.1)),
        goal_bias=float(config.get("goal_bias", 0.10)),
        obstacle_config=config.get("obstacles"),
    )
    planner.opt_model = _get_pushing_opt_model(system, config)
    planner.optimizer_num_states = int(config.get("optimizer_num_states", 1000))
    planner.optimizer_num_steps = int(config.get("optimizer_num_steps", 25))
    planner.optimizer_learning_rate = float(config.get("optimizer_learning_rate", 0.05))
    planner.optimizer_pos_std = float(config.get("sampling_position_std", 0.003))
    planner.optimizer_rot_std = float(config.get("sampling_rotation_std", 0.05))
    planner.optimizer_vel_std = float(config.get("sampling_velocity_std", 0.003))
    planner.optimizer_max_children = int(config.get("optimizer_max_children", 0))
    planner.replanning_time = float(
        config.get(
            "replanning_time_budget",
            config.get("control_duration_seconds", planner.propagation_step_size),
        )
    )
    return planner


def _fallback_car_control(current: np.ndarray, goal: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    desired = np.arctan2(goal[1] - current[1], goal[0] - current[0])
    heading_error = _wrap_angle(float(desired - current[2]))
    return np.asarray([0.65, np.clip(0.42 * heading_error, -0.30, 0.30)], dtype=float)


def _car_mppi_control_sample(
    rng: np.random.Generator,
    system,
    state: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    vel_low, vel_high = map(float, system.control_bounds[0])
    steer_low, steer_high = map(float, system.control_bounds[1])

    nominal_heading = np.arctan2(goal[1] - state[1], goal[0] - state[0])
    forward_error = _wrap_angle(float(nominal_heading - state[2]))
    reverse_error = _wrap_angle(float(nominal_heading - _wrap_angle(state[2] + np.pi)))
    reverse_is_better = vel_low < 0.0 and abs(reverse_error) + 0.20 < abs(forward_error)

    if reverse_is_better and rng.random() < 0.70:
        reverse_high = min(-0.02, vel_high)
        if vel_low < reverse_high:
            vel = rng.uniform(vel_low, reverse_high)
            steer_center = np.clip(-0.38 * reverse_error, steer_low, steer_high)
        else:
            vel = rng.uniform(max(0.05, vel_low), min(0.85, vel_high))
            steer_center = np.clip(0.38 * forward_error, steer_low, steer_high)
    else:
        vel = rng.uniform(max(0.05, vel_low), min(0.85, vel_high))
        steer_center = np.clip(0.38 * forward_error, steer_low, steer_high)

    steer = np.clip(rng.normal(steer_center, 0.16), steer_low, steer_high)
    return np.asarray([vel, steer], dtype=float)


def _open_mujoco_simulator(config: dict):
    simulator = create_simulator(config["system_name"], "mujoco", config=config)
    if config["system_name"] == "pushing_object":
        simulator.set_obj_init_pose(config["start_state"])
    else:
        simulator.reset()
    _set_mujoco_goal_region(simulator, config)
    _set_mujoco_camera(simulator, config)
    _start_mujoco_recording(simulator, config)
    wait_s = float(config.get("mujoco_start_wait", 0.0))
    if wait_s > 0.0:
        print(f"[mujoco] viewer ready; waiting {wait_s:.1f}s before controls")
        time.sleep(wait_s)
    return simulator


def _start_mujoco_recording(simulator, config: dict) -> None:
    video_path = config.get("mujoco_record_path")
    backend = getattr(simulator, "simulator", None)
    if not video_path or backend is None or not hasattr(backend, "start_recording"):
        return
    backend.start_recording(
        video_path,
        fps=float(config.get("mujoco_record_fps", 24.0)),
        width=int(config.get("mujoco_record_width", 1920)),
        height=int(config.get("mujoco_record_height", 1080)),
    )


def _attach_mujoco_video(simulator, config: dict) -> None:
    _set_mujoco_goal_region(simulator, config)
    _set_mujoco_camera(simulator, config)
    _start_mujoco_recording(simulator, config)


def _fmt_state(arr) -> str:
    x = np.asarray(arr, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{v:.6f}" for v in x[:3]) + "]"


def _print_method_final(method: str, actual, expected, goal, system_name: str) -> None:
    actual_np = np.asarray(actual, dtype=float).reshape(-1)[:3]
    expected_np = np.asarray(expected, dtype=float).reshape(-1)[:3]
    goal_np = np.asarray(goal, dtype=float).reshape(-1)[:3]
    print(
        f"[final:{method}] actual={_fmt_state(actual_np)}  "
        f"expected/planned={_fmt_state(expected_np)}  goal={_fmt_state(goal_np)}"
    )
    print(
        f"[final:{method}] actual->goal={arrayDistance(actual_np, goal_np, system=system_name):.6f}  "
        f"expected->goal={arrayDistance(expected_np, goal_np, system=system_name):.6f}  "
        f"actual->expected={arrayDistance(actual_np, expected_np, system=system_name):.6f}"
    )


def _set_mujoco_goal_region(simulator, config: dict) -> None:
    backend = getattr(simulator, "simulator", None)
    if backend is None:
        return
    try:
        backend.goal_region = (
            float(config["goal_state"][0]),
            float(config["goal_state"][1]),
            float(
                config.get("visual_goal_threshold", config.get("goal_threshold", 0.1))
            ),
        )
    except Exception:
        pass


def _set_mujoco_camera(simulator, config: dict) -> None:
    backend = getattr(simulator, "simulator", None)
    if backend is None or not hasattr(backend, "set_fixed_camera"):
        return

    start = np.asarray(config["start_state"], dtype=float).reshape(-1)
    goal = np.asarray(config["goal_state"], dtype=float).reshape(-1)
    if config.get("system_name") == "pushing_object":
        center_xy = 0.5 * (start[:2] + goal[:2])
        backend.set_fixed_camera(
            [float(center_xy[0]), float(center_xy[1]), 0.12],
            distance=float(config.get("mujoco_camera_distance", 1.35)),
            azimuth=float(config.get("mujoco_camera_azimuth", 180.0)),
            elevation=float(config.get("mujoco_camera_elevation", -35.0)),
        )
        return

    if config.get("system_name") != "kinematic_car":
        return

    center_xy = 0.5 * (start[:2] + goal[:2])
    span_xy = np.abs(goal[:2] - start[:2])
    distance = max(
        5.0,
        1.35 * float(np.linalg.norm(goal[:2] - start[:2])),
        1.55 * float(np.max(span_xy)),
    )
    backend.set_fixed_camera(
        [float(center_xy[0]), float(center_xy[1]), 0.10],
        distance=distance,
        azimuth=90.0,
        elevation=-90.0,
    )


def _set_mujoco_plan_path(simulator, states) -> None:
    backend = getattr(simulator, "simulator", None)
    if backend is None or not hasattr(backend, "set_plan_path"):
        return
    try:
        backend.set_plan_path(states)
    except Exception as exc:
        print(f"[WARNING] Could not update MuJoCo plan path: {exc}")


def _close_simulator(simulator) -> None:
    if simulator is None:
        return
    backend = getattr(simulator, "simulator", None)
    try:
        simulator.stop()
    except Exception:
        pass
    try:
        simulator.close()
    except Exception:
        pass
    if backend is not None and hasattr(backend, "save_recording"):
        try:
            backend.save_recording()
        except Exception as exc:
            print(f"[WARNING] Could not save MuJoCo recording: {exc}")


def _execute_control(
    system, simulator, current: np.ndarray, control: np.ndarray, duration: float
) -> np.ndarray:
    if simulator is not None:
        return _normalize_se2_state(simulator.execute_segment(control, duration))
    return _normalize_se2_state(system.propagate(current, control, duration))


def run_aura_aorrt(
    config: dict, planning_time: float, max_steps: int
) -> list[VideoFrame]:
    system_name = config["system_name"]
    system = get_system(system_name)
    _apply_bounds(system, config["state_bounds"])
    _apply_control_bounds(system, config.get("control_bounds"))
    simulator = (
        _open_mujoco_simulator(config) if config.get("use_mujoco", True) else None
    )
    current = np.asarray(
        simulator.get_state() if simulator is not None else config["start_state"],
        dtype=float,
    ).reshape(-1)[:3]
    frames: list[VideoFrame] = []
    goal = np.asarray(config["goal_state"], dtype=float)
    mujoco_plan_path: list[np.ndarray] = []

    try:
        planner = _build_planner_from_state(system, config, current, planning_time)
        solutions, _ = planner.plan()
        if not solutions:
            raise RuntimeError("AURA-AORRT initial plan failed.")
        config["_shared_initial_solution"] = _copy_solution(solutions[0])
        aura = AURA(system=system, planner=planner, simulator=simulator)

        def _callback(step_id: int, best_tr: dict, pose: np.ndarray) -> None:
            nonlocal mujoco_plan_path
            pose_np = np.asarray(pose, dtype=float).reshape(-1)[:3]
            actual_path = _as_state_array_list(best_tr.get("actual_states", []))
            current_plan = _plan_states(best_tr)
            raw_display_plan = _display_states(best_tr) or current_plan
            display_plan = _mujoco_display_plan(raw_display_plan, pose_np)
            initial_plan = _as_state_array_list(best_tr.get("initial_states", []))
            if display_plan:
                mujoco_plan_path = display_plan
            _set_mujoco_plan_path(simulator, mujoco_plan_path)
            frames.append(
                VideoFrame(
                    task=config["task"],
                    method="AURA-AORRT",
                    status="AURA",
                    step=int(step_id),
                    pose=pose_np,
                    actual_path=actual_path,
                    current_plan=display_plan,
                    initial_plan=initial_plan,
                    candidate_paths=_plan_candidates(best_tr),
                    distance_to_goal=float(
                        arrayDistance(
                            np.asarray(pose, dtype=float), goal, system=system_name
                        )
                    ),
                    chosen_control=None,
                )
            )

        result = aura.run(
            reset_sim=False,
            on_planning_update=_callback,
            pause_each_step=False,
            max_steps=max_steps,
        )
        current = np.asarray(result.final_state, dtype=float).reshape(-1)[:3]
        planned_final = (
            np.asarray(result.final_planned_state, dtype=float).reshape(-1)[:3]
            if getattr(result, "final_planned_state", None) is not None
            else current.copy()
        )
        _print_method_final("AURA-AORRT", current, planned_final, goal, system_name)
    finally:
        _close_simulator(simulator)

    if not frames or np.linalg.norm(frames[-1].pose[:2] - current[:2]) > 1e-9:
        frames.append(
            VideoFrame(
                task=config["task"],
                method="AURA-AORRT",
                status="DONE",
                step=len(frames),
                pose=current.copy(),
                actual_path=frames[-1].actual_path if frames else [current.copy()],
                current_plan=[],
                initial_plan=frames[0].initial_plan if frames else [],
                candidate_paths=[],
                distance_to_goal=float(
                    arrayDistance(current, goal, system=system_name)
                ),
                chosen_control=None,
            )
        )
    return frames


def run_replanning_runner(
    config: dict, planning_time: float, max_steps: int
) -> list[VideoFrame]:
    run_config = deepcopy(config)
    run_config["planning_time"] = float(planning_time)
    run_config["simulator_mode"] = "mujoco"
    run_config["replanning_time_budget"] = float(run_config["control_duration_seconds"])
    runner_max_steps = max(
        int(max_steps),
        int(run_config.get("replanning_min_steps_to_goal", 120)),
    )
    run_config["max_steps"] = runner_max_steps
    runner = None
    published_plans: list[list[np.ndarray]] = []

    def _plan_update(solution: dict) -> None:
        plan_states = _plan_states(solution)
        if plan_states:
            published_plans.append(plan_states)
        if runner is not None:
            _set_mujoco_plan_path(runner.simulator, plan_states)

    runner = ReplanningRunner(
        system_name=run_config["system_name"],
        planner_name="aorrt",
        config=run_config,
        simulator_mode="mujoco",
        max_steps=runner_max_steps,
        initial_solution=run_config.get("_shared_initial_solution"),
        plan_update_callback=_plan_update,
    )
    _attach_mujoco_video(runner.simulator, run_config)
    try:
        result = runner.run()
        _print_method_final(
            "Replanning-AORRT",
            result.final_state,
            result.planned_final_state,
            run_config["goal_state"],
            run_config["system_name"],
        )
    finally:
        _close_simulator(runner.simulator)
    trajectory = [
        np.asarray(state, dtype=float).reshape(-1)[:3].copy()
        for state in result.trajectory
    ]
    goal = np.asarray(run_config["goal_state"], dtype=float)
    initial_plan = published_plans[0] if published_plans else []
    final_plan = published_plans[-1] if published_plans else []
    frames: list[VideoFrame] = []
    for step_idx, state in enumerate(trajectory):
        plan_idx = min(step_idx, len(published_plans) - 1) if published_plans else -1
        frames.append(
            VideoFrame(
                task=run_config["task"],
                method="Replanning-AORRT",
                status="DONE" if step_idx == len(trajectory) - 1 else "REPLAN",
                step=step_idx,
                pose=state.copy(),
                actual_path=trajectory[: step_idx + 1],
                current_plan=published_plans[plan_idx] if plan_idx >= 0 else final_plan,
                initial_plan=initial_plan if step_idx == 0 else [],
                candidate_paths=[],
                distance_to_goal=float(
                    arrayDistance(state, goal, system=run_config["system_name"])
                ),
                chosen_control=None,
            )
        )
    return frames


def run_open_loop_aorrt(
    config: dict, planning_time: float, max_steps: int | None = None
) -> list[VideoFrame]:
    system_name = config["system_name"]
    system = get_system(system_name)
    _apply_bounds(system, config["state_bounds"])
    _apply_control_bounds(system, config.get("control_bounds"))
    simulator = (
        _open_mujoco_simulator(config) if config.get("use_mujoco", True) else None
    )
    current = np.asarray(
        simulator.get_state() if simulator is not None else config["start_state"],
        dtype=float,
    ).reshape(-1)[:3]
    goal = np.asarray(config["goal_state"], dtype=float).reshape(-1)[:3]
    actual_path = [current.copy()]
    frames: list[VideoFrame] = []
    raw_plan: list[np.ndarray] = []
    planned_final = current.copy()

    try:
        planner = _build_planner_from_state(system, config, current, planning_time)
        solutions, _ = planner.plan()
        if not solutions:
            raise RuntimeError("Open-loop AORRT initial plan failed.")

        plan = solutions[0]
        raw_plan = _plan_states(plan)
        controls = [np.asarray(u, dtype=float).copy() for u in plan.get("controls", [])]
        times = [float(t) for t in (plan.get("time") or [])]
        if raw_plan:
            planned_final = np.asarray(raw_plan[-1], dtype=float).reshape(-1)[:3].copy()
        _set_mujoco_plan_path(simulator, raw_plan)
        print(f"[open-loop] executing fixed plan ({len(controls)} controls)")

        for step, control in enumerate(controls):
            duration = (
                float(times[step])
                if step < len(times)
                else float(config["control_duration_seconds"])
            )
            _set_mujoco_plan_path(simulator, raw_plan)
            frames.append(
                VideoFrame(
                    task=config["task"],
                    method="OpenLoop-AORRT",
                    status="OPEN-LOOP",
                    step=step,
                    pose=current.copy(),
                    actual_path=actual_path.copy(),
                    current_plan=raw_plan,
                    initial_plan=raw_plan if step == 0 else [],
                    candidate_paths=[],
                    distance_to_goal=float(
                        arrayDistance(current, goal, system=system_name)
                    ),
                    chosen_control=control.copy(),
                )
            )
            current = _execute_control(system, simulator, current, control, duration)
            actual_path.append(current.copy())
    finally:
        _close_simulator(simulator)

    frames.append(
        VideoFrame(
            task=config["task"],
            method="OpenLoop-AORRT",
            status="DONE",
            step=len(frames),
            pose=current.copy(),
            actual_path=actual_path.copy(),
            current_plan=raw_plan,
            initial_plan=raw_plan if not frames else [],
            candidate_paths=[],
            distance_to_goal=float(arrayDistance(current, goal, system=system_name)),
            chosen_control=None,
        )
    )
    _print_method_final("OpenLoop-AORRT", current, planned_final, goal, system_name)
    return frames


def run_replanning_aorrt(
    config: dict,
    planning_time: float,
    max_steps: int,
    *,
    method: str = "Replanning-AORRT",
    status: str = "REPLAN",
    replan_time: float | None = None,
) -> list[VideoFrame]:
    system = get_system("kinematic_car")
    _apply_bounds(system, config["state_bounds"])
    _apply_control_bounds(system, config.get("control_bounds"))
    duration = float(config["control_duration_seconds"])
    simulator = (
        _open_mujoco_simulator(config) if config.get("use_mujoco", True) else None
    )
    current = np.asarray(
        simulator.get_state() if simulator is not None else config["start_state"],
        dtype=float,
    ).reshape(-1)[:3]
    actual_path = [current.copy()]
    initial_plan: list[np.ndarray] = []
    previous_display_plan: list[np.ndarray] = []
    frames: list[VideoFrame] = []

    try:
        for step in range(max_steps):
            step_planning_time = (
                planning_time
                if step == 0
                else float(planning_time if replan_time is None else replan_time)
            )
            planner = _build_planner_from_state(
                system, config, current, step_planning_time
            )
            solutions, _ = planner.plan()
            if not solutions:
                break
            plan = solutions[0]
            if step == 0:
                initial_plan = _plan_states(plan)
            controls = plan.get("controls", [])
            if not controls:
                control = _fallback_car_control(current, config["goal_state"])
            else:
                control = np.asarray(controls[0], dtype=float)
            raw_plan = _plan_states(plan)
            current_plan = _display_plan(raw_plan, previous_display_plan, current)
            _set_mujoco_plan_path(simulator, raw_plan)
            previous_display_plan = current_plan
            frames.append(
                VideoFrame(
                    task="car",
                    method=method,
                    status=status,
                    step=step,
                    pose=current.copy(),
                    actual_path=actual_path.copy(),
                    current_plan=current_plan,
                    initial_plan=initial_plan,
                    candidate_paths=_plan_candidates(plan),
                    distance_to_goal=float(
                        arrayDistance(
                            current, config["goal_state"], system="kinematic_car"
                        )
                    ),
                    chosen_control=control.copy(),
                )
            )
            current = _execute_control(system, simulator, current, control, duration)
            actual_path.append(current.copy())
            if arrayDistance(
                current, config["goal_state"], system="kinematic_car"
            ) <= float(config["goal_threshold"]):
                break
    finally:
        _close_simulator(simulator)

    frames.append(
        VideoFrame(
            task="car",
            method=method,
            status="DONE",
            step=len(frames),
            pose=current.copy(),
            actual_path=actual_path.copy(),
            current_plan=[],
            initial_plan=initial_plan,
            candidate_paths=[],
            distance_to_goal=float(
                arrayDistance(current, config["goal_state"], system="kinematic_car")
            ),
            chosen_control=None,
        )
    )
    _print_method_final(
        method, current, config["goal_state"], config["goal_state"], "kinematic_car"
    )
    return frames


def run_mppi(config: dict, max_steps: int, seed: int) -> list[VideoFrame]:
    rng = np.random.default_rng(seed)
    system = get_system("kinematic_car")
    _apply_bounds(system, config["state_bounds"])
    _apply_control_bounds(system, config.get("control_bounds"))
    duration = float(config["control_duration_seconds"])
    simulator = (
        _open_mujoco_simulator(config) if config.get("use_mujoco", True) else None
    )
    current = np.asarray(
        simulator.get_state() if simulator is not None else config["start_state"],
        dtype=float,
    ).reshape(-1)[:3]
    actual_path = [current.copy()]
    frames: list[VideoFrame] = []
    horizon = 12
    rollouts = 420
    goal = np.asarray(config["goal_state"], dtype=float)
    mppi_goal_threshold = float(
        config.get(
            "mppi_goal_threshold",
            config.get("visual_goal_threshold", config["goal_threshold"]),
        )
    )
    best_rollout: list[np.ndarray] = []

    try:
        for step in range(max_steps):
            if np.linalg.norm(current[:2] - goal[:2]) <= mppi_goal_threshold:
                break
            best_cost = float("inf")
            best_controls = None
            best_rollout = []
            for _ in range(rollouts):
                state = current.copy()
                controls = []
                rollout_states = [state.copy()]
                cost = 0.0
                for h in range(horizon):
                    control = _car_mppi_control_sample(rng, system, state, goal)
                    state = np.asarray(
                        system.propagate(state, control, duration), dtype=float
                    )
                    controls.append(control)
                    rollout_states.append(state.copy())
                    dist = np.linalg.norm(state[:2] - goal[:2])
                    outside_goal = max(0.0, dist - mppi_goal_threshold)
                    yaw_error = abs(_wrap_angle(float(state[2] - goal[2])))
                    cost += outside_goal + (
                        0.2 * yaw_error if outside_goal > 0.0 else 0.0
                    )
                    cost += 0.02 * abs(float(control[0])) + 0.03 * abs(
                        float(control[1])
                    )
                    if not _valid_state(config, state):
                        cost += 100.0
                final_dist = np.linalg.norm(rollout_states[-1][:2] - goal[:2])
                cost += 4.0 * max(0.0, final_dist - mppi_goal_threshold)
                if cost < best_cost:
                    best_cost = cost
                    best_controls = controls
                    best_rollout = rollout_states

            if not best_controls:
                break
            control = np.asarray(best_controls[0], dtype=float)
            _set_mujoco_plan_path(simulator, best_rollout)
            frames.append(
                VideoFrame(
                    task="car",
                    method="MPPI",
                    status="MPPI",
                    step=step,
                    pose=current.copy(),
                    actual_path=actual_path.copy(),
                    current_plan=best_rollout,
                    initial_plan=best_rollout if step == 0 else [],
                    candidate_paths=[],
                    distance_to_goal=float(
                        arrayDistance(current, goal, system="kinematic_car")
                    ),
                    chosen_control=control.copy(),
                )
            )
            current = _execute_control(system, simulator, current, control, duration)
            actual_path.append(current.copy())
            if np.linalg.norm(current[:2] - goal[:2]) <= mppi_goal_threshold:
                break
    finally:
        _close_simulator(simulator)

    frames.append(
        VideoFrame(
            task="car",
            method="MPPI",
            status="DONE",
            step=len(frames),
            pose=current.copy(),
            actual_path=actual_path.copy(),
            current_plan=best_rollout,
            initial_plan=[],
            candidate_paths=[],
            distance_to_goal=float(
                arrayDistance(current, goal, system="kinematic_car")
            ),
            chosen_control=None,
        )
    )
    _print_method_final("MPPI", current, goal, goal, "kinematic_car")
    return frames


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _normalize_se2_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(-1).copy()
    if state.size >= 3:
        state[2] = _wrap_angle(float(state[2]))
    return state[:3]


def _pushing_control_toward_goal(
    rng: np.random.Generator, system, state: np.ndarray, goal: np.ndarray
) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(1, -1)[:, :3]
    goal = np.asarray(goal, dtype=float).reshape(-1)[:3]
    candidate_states = np.repeat(state, 256, axis=0)
    candidate_controls = _sample_pushing_controls_batch(
        rng,
        system,
        candidate_states,
        goal,
        exploration=0.45,
    )
    next_states = _pushing_propagate_batch(system, candidate_states, candidate_controls)
    xy_dist = np.linalg.norm(next_states[:, :2] - goal[:2], axis=1)
    yaw_error = np.abs((next_states[:, 2] - goal[2] + np.pi) % (2.0 * np.pi) - np.pi)
    valid = _valid_states_batch(system, next_states)
    score = xy_dist + 0.05 * yaw_error + np.where(valid, 0.0, 100.0)
    return candidate_controls[int(np.argmin(score))].copy()


def _pushing_goal_face_indices(states: np.ndarray, goal: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float).reshape(-1, 3)
    goal = np.asarray(goal, dtype=float).reshape(-1)[:3]
    delta_world = goal[None, :2] - states[:, :2]
    c = np.cos(states[:, 2])
    s = np.sin(states[:, 2])
    delta_local = np.column_stack(
        [
            c * delta_world[:, 0] + s * delta_world[:, 1],
            -s * delta_world[:, 0] + c * delta_world[:, 1],
        ]
    )
    local_motion_dirs = np.asarray(
        [
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    return np.argmax(delta_local @ local_motion_dirs.T, axis=1)


def _sample_pushing_controls_batch(
    rng: np.random.Generator,
    system,
    states: np.ndarray,
    goal: np.ndarray,
    *,
    exploration: float = 0.30,
) -> np.ndarray:
    states = np.asarray(states, dtype=float).reshape(-1, 3)
    n = states.shape[0]
    face_values = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float)
    face_idx = _pushing_goal_face_indices(states, goal)
    random_mask = rng.random(n) < float(exploration)
    face_idx[random_mask] = rng.integers(0, 4, size=int(np.count_nonzero(random_mask)))

    side_low, side_high = map(float, system.control_bounds[1])
    dist_low, dist_high = map(float, system.control_bounds[2])
    sides = np.clip(rng.normal(0.0, 0.18, size=n), side_low, side_high)
    distances = rng.uniform(
        max(dist_low, 0.045),
        min(dist_high, 0.18),
        size=n,
    )
    return np.column_stack([face_values[face_idx], sides, distances]).astype(float)


def _pushing_propagate_batch(
    system, states: np.ndarray, controls: np.ndarray
) -> np.ndarray:
    states = np.asarray(states, dtype=float).reshape(-1, 3)
    controls = np.asarray(controls, dtype=float).reshape(-1, 3).copy()
    controls[:, 0] = np.mod(np.round(controls[:, 0] * 4.0), 4.0) / 4.0
    controls[:, 1] = np.clip(
        controls[:, 1],
        float(system.control_bounds[1][0]),
        float(system.control_bounds[1][1]),
    )
    controls[:, 2] = np.clip(
        controls[:, 2],
        float(system.control_bounds[2][0]),
        float(system.control_bounds[2][1]),
    )

    model = get_pushing_model(system.object_shape)
    device = next(model.parameters()).device
    control_tensor = torch.as_tensor(controls, dtype=torch.float32, device=device)
    with torch.no_grad():
        deltas = model(control_tensor).detach().cpu().numpy()

    theta = states[:, 2]
    c = np.cos(theta)
    s = np.sin(theta)
    next_states = np.empty_like(states, dtype=float)
    next_states[:, 0] = states[:, 0] + c * deltas[:, 0] - s * deltas[:, 1]
    next_states[:, 1] = states[:, 1] + s * deltas[:, 0] + c * deltas[:, 1]
    next_states[:, 2] = (theta + deltas[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    return next_states


def _valid_states_batch(system, states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float).reshape(-1, 3)
    (x_low, x_high), (y_low, y_high) = system.state_bounds[:2]
    return (
        (states[:, 0] >= float(x_low))
        & (states[:, 0] <= float(x_high))
        & (states[:, 1] >= float(y_low))
        & (states[:, 1] <= float(y_high))
    )


def _pushing_goal_distance_batch(states: np.ndarray, goal: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float).reshape(-1, 3)
    goal = np.asarray(goal, dtype=float).reshape(-1)[:3]
    xy_dist = np.linalg.norm(states[:, :2] - goal[None, :2], axis=1)
    yaw_error = np.abs((states[:, 2] - goal[2] + np.pi) % (2.0 * np.pi) - np.pi)
    return xy_dist + 0.5 * yaw_error


def run_pushing_planner(
    config: dict, planning_time: float, max_steps: int, method: str
) -> list[VideoFrame]:
    rng = np.random.default_rng(17 if method == "AURA-AORRT" else 23)
    system = get_system("pushing_object")
    _apply_bounds(system, config["state_bounds"])
    _apply_control_bounds(system, config.get("control_bounds"))
    duration = float(config["control_duration_seconds"])
    simulator = (
        _open_mujoco_simulator(config) if config.get("use_mujoco", True) else None
    )
    current = np.asarray(
        simulator.get_state() if simulator is not None else config["start_state"],
        dtype=float,
    ).reshape(-1)[:3]
    goal = np.asarray(config["goal_state"], dtype=float)
    actual_path = [current.copy()]
    initial_plan: list[np.ndarray] = []
    previous_display_plan: list[np.ndarray] = []
    frames: list[VideoFrame] = []

    try:
        for step in range(max_steps):
            plan = None
            if method != "AURA-AORRT" or step % 2 == 0:
                planner = _build_planner_from_state(
                    system, config, current, planning_time if step == 0 else 0.35
                )
                solutions, _ = planner.plan()
                if solutions:
                    plan = solutions[0]
            if plan is None:
                control = _pushing_control_toward_goal(rng, system, current, goal)
                current_plan = [
                    current.copy(),
                    np.asarray(
                        system.propagate(current, control, duration), dtype=float
                    ),
                ]
                candidates = []
            else:
                raw_plan = _plan_states(plan)
                current_plan = _display_plan(raw_plan, previous_display_plan, current)
                _set_mujoco_plan_path(simulator, raw_plan)
                previous_display_plan = current_plan
                candidates = _plan_candidates(plan)
                if step == 0:
                    initial_plan = current_plan
                controls = plan.get("controls", [])
                control = (
                    np.asarray(controls[0], dtype=float)
                    if controls
                    else _pushing_control_toward_goal(rng, system, current, goal)
                )

            frames.append(
                VideoFrame(
                    task="pushing",
                    method=method,
                    status="AURA" if method == "AURA-AORRT" else "REPLAN",
                    step=step,
                    pose=current.copy(),
                    actual_path=actual_path.copy(),
                    current_plan=current_plan,
                    initial_plan=initial_plan,
                    candidate_paths=candidates,
                    distance_to_goal=float(
                        arrayDistance(current, goal, system="pushing_object")
                    ),
                    chosen_control=control.copy(),
                )
            )
            current = _execute_control(system, simulator, current, control, duration)
            actual_path.append(current.copy())
            if arrayDistance(current, goal, system="pushing_object") <= float(
                config["goal_threshold"]
            ):
                break
    finally:
        _close_simulator(simulator)

    frames.append(
        VideoFrame(
            task="pushing",
            method=method,
            status="DONE",
            step=len(frames),
            pose=current.copy(),
            actual_path=actual_path.copy(),
            current_plan=[],
            initial_plan=initial_plan,
            candidate_paths=[],
            distance_to_goal=float(
                arrayDistance(current, goal, system="pushing_object")
            ),
            chosen_control=None,
        )
    )
    _print_method_final("MPPI", current, goal, goal, "pushing_object")
    return frames


def run_pushing_mppi(config: dict, max_steps: int, seed: int) -> list[VideoFrame]:
    rng = np.random.default_rng(seed + 1000)
    system = get_system("pushing_object")
    _apply_bounds(system, config["state_bounds"])
    _apply_control_bounds(system, config.get("control_bounds"))
    duration = float(config["control_duration_seconds"])
    simulator = (
        _open_mujoco_simulator(config) if config.get("use_mujoco", True) else None
    )
    current = np.asarray(
        simulator.get_state() if simulator is not None else config["start_state"],
        dtype=float,
    ).reshape(-1)[:3]
    goal = np.asarray(config["goal_state"], dtype=float)
    actual_path = [current.copy()]
    frames: list[VideoFrame] = []
    horizon = int(config.get("pushing_mppi_horizon", 7))
    rollouts = int(config.get("pushing_mppi_rollouts", 640))
    mppi_goal_threshold = float(
        config.get(
            "mppi_goal_threshold",
            config.get("visual_goal_threshold", config["goal_threshold"]),
        )
    )
    best_rollout: list[np.ndarray] = []

    try:
        for step in range(max_steps):
            current_goal_dist = float(
                arrayDistance(current, goal, system="pushing_object")
            )
            if current_goal_dist <= mppi_goal_threshold:
                break
            rollout_states = np.zeros((rollouts, horizon + 1, 3), dtype=float)
            rollout_controls = np.zeros((rollouts, horizon, 3), dtype=float)
            rollout_states[:, 0, :] = current[None, :]
            states = np.repeat(current[None, :], rollouts, axis=0)
            costs = np.zeros(rollouts, dtype=float)
            previous_goal_dist = _pushing_goal_distance_batch(states, goal)

            for h in range(horizon):
                exploration = 0.45 if h == 0 else 0.30
                controls = _sample_pushing_controls_batch(
                    rng,
                    system,
                    states,
                    goal,
                    exploration=exploration,
                )
                states = _pushing_propagate_batch(system, states, controls)
                rollout_controls[:, h, :] = controls
                rollout_states[:, h + 1, :] = states

                goal_dist = _pushing_goal_distance_batch(states, goal)
                outside_goal = np.maximum(0.0, goal_dist - mppi_goal_threshold)
                progress = previous_goal_dist - goal_dist
                costs += outside_goal
                costs -= 0.20 * np.maximum(0.0, progress)
                costs += 0.02 * np.abs(controls[:, 1])
                costs += 0.03 * controls[:, 2]
                costs += np.where(_valid_states_batch(system, states), 0.0, 100.0)
                previous_goal_dist = goal_dist

            final_dist = _pushing_goal_distance_batch(states, goal)
            costs += 4.5 * np.maximum(0.0, final_dist - mppi_goal_threshold)
            best_idx = int(np.argmin(costs))
            best_control = rollout_controls[best_idx, 0, :].copy()
            best_rollout = [
                rollout_states[best_idx, i, :].copy() for i in range(horizon + 1)
            ]

            if best_control is None:
                break
            print(
                f"[MPPI:pushing] step {step} dist={current_goal_dist:.4f} "
                f"best_cost={float(costs[best_idx]):.4f} control={np.round(best_control, 4).tolist()}",
                flush=True,
            )
            _set_mujoco_plan_path(simulator, best_rollout)
            frames.append(
                VideoFrame(
                    task="pushing",
                    method="MPPI",
                    status="MPPI",
                    step=step,
                    pose=current.copy(),
                    actual_path=actual_path.copy(),
                    current_plan=best_rollout,
                    initial_plan=best_rollout if step == 0 else [],
                    candidate_paths=[],
                    distance_to_goal=float(
                        arrayDistance(current, goal, system="pushing_object")
                    ),
                    chosen_control=best_control.copy(),
                )
            )
            current = _execute_control(
                system, simulator, current, best_control, duration
            )
            actual_path.append(current.copy())
            if (
                arrayDistance(current, goal, system="pushing_object")
                <= mppi_goal_threshold
            ):
                break
    finally:
        _close_simulator(simulator)

    final_goal_dist = float(arrayDistance(current, goal, system="pushing_object"))
    if final_goal_dist > mppi_goal_threshold:
        print(
            "[WARNING] MPPI stopped before reaching its goal threshold: "
            f"distance {final_goal_dist:.6f} > threshold {mppi_goal_threshold:.6f}. "
            "Increase --max-steps or --mppi-goal-threshold pushing_object=VALUE.",
            flush=True,
        )

    frames.append(
        VideoFrame(
            task="pushing",
            method="MPPI",
            status="DONE",
            step=len(frames),
            pose=current.copy(),
            actual_path=actual_path.copy(),
            current_plan=best_rollout,
            initial_plan=[],
            candidate_paths=[],
            distance_to_goal=final_goal_dist,
            chosen_control=None,
        )
    )
    _print_method_final("MPPI", current, goal, goal, "pushing_object")
    return frames


def _plot_path(ax, path, **kwargs):
    if not path:
        return
    arr = np.asarray(path, dtype=float)
    if arr.ndim == 2 and arr.shape[0] > 0:
        ax.plot(arr[:, 0], arr[:, 1], **kwargs)


def _draw_car(ax, pose: np.ndarray) -> None:
    x, y, theta = pose[:3]
    length = 0.34
    start = np.asarray([x, y], dtype=float)
    end = start + length * np.asarray([np.cos(theta), np.sin(theta)])
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=24,
        linewidth=3.2,
        color="#EB5B00",
        zorder=20,
    )
    ax.add_patch(arrow)
    ax.add_patch(
        Circle(
            (x, y),
            0.045,
            facecolor="white",
            edgecolor="#EB5B00",
            linewidth=2.0,
            zorder=21,
        )
    )


def _draw_pushing_box(ax, pose: np.ndarray) -> None:
    x, y, theta = pose[:3]
    width, length = CRACKER_BOX_FLIPPED_SHAPE[:2]
    box = Rectangle(
        (-width / 2.0, -length / 2.0),
        width,
        length,
        facecolor="#EB5B00",
        edgecolor="#111111",
        linewidth=1.8,
        zorder=20,
    )
    box.set_transform(
        Affine2D().rotate(float(theta)).translate(float(x), float(y)) + ax.transData
    )
    ax.add_patch(box)
    nose = np.asarray([x, y]) + 0.16 * np.asarray([np.cos(theta), np.sin(theta)])
    ax.plot([x, nose[0]], [y, nose[1]], color="white", linewidth=2.4, zorder=21)


def _draw_pose(ax, frame: VideoFrame) -> None:
    if frame.task == "pushing":
        _draw_pushing_box(ax, frame.pose)
    else:
        _draw_car(ax, frame.pose)


def _control_label(frame: VideoFrame) -> str:
    if frame.chosen_control is None:
        return ""
    if frame.task == "pushing":
        return f"u={frame.chosen_control[0]:.2f},{frame.chosen_control[1]:+.2f},{frame.chosen_control[2]:.2f}"
    return f"u={frame.chosen_control[0]:.2f},{frame.chosen_control[1]:+.2f}"


def render_video(
    frames: list[VideoFrame], config: dict, video_path: str, fps: float, dpi: int
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.6, 6.6), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.68, top=0.90, bottom=0.13)
    writer = None
    ext = os.path.splitext(video_path)[1].lower()
    if ext == ".mp4" and writers.is_available("ffmpeg"):
        writer = FFMpegWriter(fps=fps, bitrate=24000)
    elif ext == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        raise RuntimeError(
            "No video writer available. Install ffmpeg or use a .gif output."
        )

    bounds = config["state_bounds"]
    goal = np.asarray(config["goal_state"], dtype=float)
    goal_threshold = float(
        config.get("visual_goal_threshold", config["goal_threshold"])
    )
    obstacles = config.get("obstacles") or {}

    display_frames = _animated_frames(frames, fps)

    def draw(frame: VideoFrame) -> None:
        ax.clear()
        fig.subplots_adjust(left=0.08, right=0.68, top=0.90, bottom=0.13)
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#F7F8FA")
        ax.grid(True, color="#DDE3EA", linewidth=0.75, alpha=0.95)
        ax.set_axisbelow(True)
        ax.set_xlabel("x", labelpad=12)
        ax.set_ylabel("y", labelpad=10)
        ax.tick_params(colors="#000000", width=1.05, length=4.0)
        for spine in ax.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(1.05)
        ax.set_title(
            frame.method,
            fontsize=20,
            fontweight="semibold",
            color=ABSTRACT_TEXT_DARK,
            loc="left",
            pad=12,
        )

        for cx, cy, r in obstacles.get("circles", []):
            safety = float(obstacles.get("safety_radius", 0.0))
            if safety > 0:
                ax.add_patch(
                    Circle(
                        (cx, cy),
                        r + safety,
                        facecolor="#E8EEF8",
                        edgecolor="#9CB3D1",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.75,
                    )
                )
            ax.add_patch(
                Circle(
                    (cx, cy),
                    r,
                    facecolor="#505861",
                    edgecolor="#262B31",
                    linewidth=1.4,
                    zorder=5,
                )
            )
        for x0, y0, x1, y1 in obstacles.get("aabbs", []):
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="#505861",
                    edgecolor="#262B31",
                    zorder=5,
                )
            )

        ax.add_patch(
            Circle(
                goal[:2],
                goal_threshold,
                facecolor="#A0C878",
                edgecolor="#4E8B45",
                alpha=0.30,
                linewidth=2.0,
                zorder=3,
            )
        )
        ax.scatter([goal[0]], [goal[1]], marker="*", s=180, color="#4E8B45", zorder=15)

        _plot_path(
            ax,
            frame.initial_plan,
            color="#8A8F98",
            linewidth=2.0,
            alpha=0.65,
            label="Initial plan",
        )
        for candidate in frame.candidate_paths:
            _plot_path(ax, candidate, color="#143D60", linewidth=1.0, alpha=0.16)
        _plot_path(
            ax,
            frame.current_plan,
            color="#143D60",
            linewidth=3.2,
            alpha=0.95,
            label="Current plan",
        )
        _plot_path(
            ax,
            frame.actual_path,
            color="#9B59B6",
            linewidth=3.0,
            alpha=0.95,
            label="Executed path",
        )
        _draw_pose(ax, frame)

        control_text = _control_label(frame)
        info_lines = [
            frame.status.upper(),
            f"step {frame.step}",
            f"goal dist. {frame.distance_to_goal:.3f}",
        ]
        if control_text:
            info_lines.append(control_text)
        info = "\n".join(info_lines)
        ax.text(
            ABSTRACT_SIDE_X,
            0.96,
            info,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13.5,
            color=ABSTRACT_TEXT_DARK,
            linespacing=1.30,
            bbox=dict(
                boxstyle="round,pad=0.35,rounding_size=0.08",
                facecolor="white",
                edgecolor="#D4DAE2",
                linewidth=0.9,
                alpha=0.94,
            ),
            clip_on=False,
            zorder=30,
        )
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                unique.setdefault(label, handle)
        if unique:
            legend = ax.legend(
                unique.values(),
                unique.keys(),
                loc="upper left",
                bbox_to_anchor=(ABSTRACT_SIDE_X, 0.70),
                bbox_transform=ax.transAxes,
                borderaxespad=0.0,
                framealpha=0.94,
                borderpad=0.55,
                labelspacing=0.45,
                handlelength=2.4,
                handletextpad=0.7,
            )
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("#D4DAE2")
            legend.get_frame().set_linewidth(0.9)

    with writer.saving(fig, video_path, dpi=dpi):
        for frame in display_frames:
            draw(frame)
            writer.grab_frame()
    plt.close(fig)
    print(f"[video] saved {video_path}")


def _save_abstract_video(
    frames: list[VideoFrame],
    config: dict,
    output_dir: str,
    name: str,
    ext: str,
    fps: float,
    dpi: int,
) -> None:
    if not frames:
        print(f"[WARNING] No abstract frames for {name}; skipping abstract video.")
        return
    abstract_path = os.path.join(output_dir, f"{name}_abstract{ext}")
    render_video(frames, config, abstract_path, fps=fps, dpi=dpi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render videos for AURA-AORRT, Replanning-AORRT, MPPI, and open-loop baselines."
    )
    parser.add_argument("--output-dir", default="videos")
    parser.add_argument("--task", choices=["car", "pushing", "both"], default="both")
    parser.add_argument(
        "--method",
        choices=[
            "all",
            "aura",
            "aura-aorrt",
            "replanning",
            "replanning-aorrt",
            "mppi",
            "baseline",
            "open-loop",
            "openloop",
        ],
        default="all",
        help="Which method to render.",
    )
    parser.add_argument("--planning-time", type=float, default=3.0)
    parser.add_argument("--control-duration", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument(
        "--mujoco-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after each MuJoCo viewer opens.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--video-size",
        type=int,
        default=1080,
        help="Square MuJoCo video size in pixels, e.g. 1080 means 1080x1080.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--optimizer-num-states", type=int, default=None)
    parser.add_argument("--optimizer-num-steps", type=int, default=None)
    parser.add_argument("--optimizer-learning-rate", type=float, default=None)
    parser.add_argument("--optimizer-pos-std", type=float, default=None)
    parser.add_argument("--optimizer-rot-std", type=float, default=None)
    parser.add_argument("--optimizer-vel-std", type=float, default=None)
    parser.add_argument("--optimizer-max-children", type=int, default=None)
    parser.add_argument("--pushing-camera-distance", type=float, default=None)
    parser.add_argument("--pushing-camera-azimuth", type=float, default=None)
    parser.add_argument("--pushing-camera-elevation", type=float, default=None)
    parser.add_argument(
        "--mppi-goal-threshold",
        action="append",
        default=[],
        metavar="SYSTEM=VALUE",
        help=(
            "Override MPPI stopping/cost threshold for one system. "
            "Examples: kinematic_car=0.30, pushing_object=0.09. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--skip-aura",
        action="store_true",
        help="Debug helper: only render replanning and MPPI.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ext = f".{args.format}"
    print(f"[config] initial planning time: {float(args.planning_time):.2f}s")
    print("[config] control duration: car=1.00s, pushing=2.00s")
    print("[config] replanning/optimization budget follows control duration")
    print(f"[config] method: {args.method}")
    print(
        f"[config] saving square MuJoCo videos ({int(args.video_size)}x{int(args.video_size)}) and abstract videos"
    )

    tasks = ["car", "pushing"] if args.task == "both" else [args.task]
    for task in tasks:
        config = _config(task, args.control_duration)
        _apply_optimizer_cli_overrides(config, args)
        if task == "pushing":
            if args.pushing_camera_distance is not None:
                config["mujoco_camera_distance"] = float(args.pushing_camera_distance)
            if args.pushing_camera_azimuth is not None:
                config["mujoco_camera_azimuth"] = float(args.pushing_camera_azimuth)
            if args.pushing_camera_elevation is not None:
                config["mujoco_camera_elevation"] = float(args.pushing_camera_elevation)
        try:
            _apply_mppi_goal_threshold_cli_overrides(config, args.mppi_goal_threshold)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            f"[config] thresholds ({task}): "
            f"planner={float(config['goal_threshold']):.3f}, "
            f"visual={float(config['visual_goal_threshold']):.3f}, "
            f"mppi={float(config.get('mppi_goal_threshold', config['visual_goal_threshold'])):.3f}"
        )
        print(
            "[config] optimizer: "
            f"states={int(config.get('optimizer_num_states', 1000))}, "
            f"steps={int(config.get('optimizer_num_steps', 25))}, "
            f"lr={float(config.get('optimizer_learning_rate', 0.05)):.4g}, "
            f"pos_std={float(config.get('sampling_position_std', 0.003)):.4g}, "
            f"rot_std={float(config.get('sampling_rotation_std', 0.05)):.4g}, "
            f"max_children={int(config.get('optimizer_max_children', 0))}"
        )
        config["mujoco_start_wait"] = float(args.mujoco_wait)
        config["mujoco_record_width"] = int(args.video_size)
        config["mujoco_record_height"] = int(args.video_size)
        if task == "car":
            if _method_requested(args.method, "aura") and not args.skip_aura:
                print("[run] car / AURA-AORRT")
                name = "car_aura_aorrt"
                config["mujoco_record_path"] = os.path.join(
                    args.output_dir, f"{name}{ext}"
                )
                config["mujoco_record_fps"] = float(args.fps)
                frames = run_aura_aorrt(config, args.planning_time, args.max_steps)
                _save_abstract_video(
                    frames, config, args.output_dir, name, ext, args.fps, args.dpi
                )
            if _method_requested(args.method, "replanning"):
                print("[run] car / Replanning-AORRT")
                name = "car_replanning_aorrt"
                config["mujoco_record_path"] = os.path.join(
                    args.output_dir, f"{name}{ext}"
                )
                config["mujoco_record_fps"] = float(args.fps)
                frames = run_replanning_runner(
                    config, args.planning_time, args.max_steps
                )
                _save_abstract_video(
                    frames, config, args.output_dir, name, ext, args.fps, args.dpi
                )
            if _method_requested(args.method, "baseline"):
                print("[run] car / OpenLoop-AORRT")
                name = "car_open_loop_aorrt"
                config["mujoco_record_path"] = os.path.join(
                    args.output_dir, f"{name}{ext}"
                )
                config["mujoco_record_fps"] = float(args.fps)
                frames = run_open_loop_aorrt(config, args.planning_time, args.max_steps)
                _save_abstract_video(
                    frames, config, args.output_dir, name, ext, args.fps, args.dpi
                )
            if _method_requested(args.method, "mppi"):
                print("[run] car / MPPI")
                name = "car_mppi"
                config["mujoco_record_path"] = os.path.join(
                    args.output_dir, f"{name}{ext}"
                )
                config["mujoco_record_fps"] = float(args.fps)
                frames = run_mppi(config, args.max_steps, args.seed)
                _save_abstract_video(
                    frames, config, args.output_dir, name, ext, args.fps, args.dpi
                )
        else:
            if _method_requested(args.method, "aura") and not args.skip_aura:
                print("[run] pushing / AURA-AORRT")
                name = "pushing_aura_aorrt"
                config["mujoco_record_path"] = os.path.join(
                    args.output_dir, f"{name}{ext}"
                )
                config["mujoco_record_fps"] = float(args.fps)
                frames = run_aura_aorrt(config, args.planning_time, args.max_steps)
                _save_abstract_video(
                    frames, config, args.output_dir, name, ext, args.fps, args.dpi
                )
            if _method_requested(args.method, "replanning"):
                print("[run] pushing / Replanning-AORRT")
                name = "pushing_replanning_aorrt"
                config["mujoco_record_path"] = os.path.join(
                    args.output_dir, f"{name}{ext}"
                )
                config["mujoco_record_fps"] = float(args.fps)
                frames = run_replanning_runner(
                    config, args.planning_time, args.max_steps
                )
                _save_abstract_video(
                    frames, config, args.output_dir, name, ext, args.fps, args.dpi
                )
            if _method_requested(args.method, "baseline"):
                print("[run] pushing / OpenLoop-AORRT")
                name = "pushing_open_loop_aorrt"
                config["mujoco_record_path"] = os.path.join(
                    args.output_dir, f"{name}{ext}"
                )
                config["mujoco_record_fps"] = float(args.fps)
                frames = run_open_loop_aorrt(config, args.planning_time, args.max_steps)
                _save_abstract_video(
                    frames, config, args.output_dir, name, ext, args.fps, args.dpi
                )
            if _method_requested(args.method, "mppi"):
                print("[run] pushing / MPPI")
                name = "pushing_mppi"
                config["mujoco_record_path"] = os.path.join(
                    args.output_dir, f"{name}{ext}"
                )
                config["mujoco_record_fps"] = float(args.fps)
                frames = run_pushing_mppi(config, args.max_steps, args.seed)
                _save_abstract_video(
                    frames, config, args.output_dir, name, ext, args.fps, args.dpi
                )

    print(f"[done] videos saved in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
