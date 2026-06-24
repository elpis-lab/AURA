#!/usr/bin/env python3
"""Real-robot execution of AURA and replanning pushing to a fixed goal.

This script starts from the currently detected object pose, then lets the actual
AURA or ReplanningRunner class drive real UR10 pushes through the same
motion-generation path used by experiments/real_error_experiment.py: two-step
object detection, relative side-offset push params, pre-push/pre-grasp lift,
push waypoints, post-push lift, and optional movement pauses.

Typical usage:
  /usr/bin/python3 real_world/real_execution.py --method aura
  /usr/bin/python3 real_world/real_execution.py --method replanning --goal 0.0 0.7 0.0
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ompl import base as ob
from scipy.spatial.transform import Rotation as R

from AURA import AURA
from Replanning import ReplanningRunner
from geometry.object_model import get_obj_shape
from geometry.pose import Pose, matrix_to_quat
from geometry.random_push import generate_path_form_params
from plan import OMPL_Planner
from pushing_dynamics import get_pushing_model
from experiments.real_error_experiment import (
    _wrap_angle,
    matrix_to_flat,
    project_se3_pose,
)
from real_world.physical_robot import PhysicalUR10
from simulators import Simulator
from systems import get_system
from train_model import load_opt_model_2
from utils.utils import arrayDistance

_LATEST_OBJECT_STATE_FOR_PAUSES: np.ndarray | None = None

# Keep real-world tolerances and distance limits in one place. CLI flags below
# still override these, but this block is the source of truth for defaults.
REAL_EXECUTION_LIMITS = {
    # OMPL must produce a plan ending inside this tighter goal region.
    "planner_goal_threshold": 0.075,
    # Real camera/robot execution stops once the measured object is within this.
    "actual_goal_threshold": 0.15,
    # Warn if measured execution drifts farther than this from AURA's prediction.
    "tracking_error_threshold": 0.1,
    # Max continuity/tracking miss before replanning treats the path as stale.
    "replanning_max_distance": 0.1,
    # Push distance is absolute meters. Keep this at the real data/model limit.
    "max_push_distance": 0.30,
    "planner_margin": 0.25,
    "pruning_radius": 0.08,
}


@dataclass
class RealExecutionStep:
    step: int
    method: str
    measured_start_x: float
    measured_start_y: float
    measured_start_theta: float
    target_x: float
    target_y: float
    target_theta: float
    planned_final_x: float
    planned_final_y: float
    planned_final_theta: float
    original_rotation: float
    original_side: float
    original_distance: float
    chosen_rotation: float
    chosen_side: float
    chosen_distance: float
    original_pred_error: float
    chosen_pred_error: float
    measured_end_x: float
    measured_end_y: float
    measured_end_theta: float
    tracking_error: float
    goal_distance_after: float
    planning_time_s: float
    executed_duration_s: float


def _fmt(arr: np.ndarray) -> str:
    return np.array2string(
        np.asarray(arr, dtype=float), precision=5, suppress_small=True
    )


def _set_pause_object_state(state: np.ndarray | None) -> None:
    global _LATEST_OBJECT_STATE_FOR_PAUSES
    if state is None:
        _LATEST_OBJECT_STATE_FOR_PAUSES = None
        return
    arr = np.asarray(state, dtype=float)
    if arr.shape == (4, 4):
        arr = project_se3_pose(matrix_to_flat(arr))
    else:
        arr = arr.reshape(-1)[:3]
    arr = arr.astype(float, copy=True)
    arr[2] = _wrap_angle(float(arr[2]))
    _LATEST_OBJECT_STATE_FOR_PAUSES = arr


def _pause_between_movements(enabled: bool, label: str) -> None:
    if not enabled:
        return
    if _LATEST_OBJECT_STATE_FOR_PAUSES is None:
        print("[REAL] object state before prompt: unknown/not detected yet")
    else:
        print(
            "[REAL] object state before prompt:", _fmt(_LATEST_OBJECT_STATE_FOR_PAUSES)
        )
    print(f"[PAUSE] {label}. Sleeping 0.5s before continuing.")
    time.sleep(0.5)


def pose_valid(pose) -> bool:
    return (
        pose is not None
        and pose.get("pose") is not None
        and isinstance(pose["pose"], np.ndarray)
        and pose["pose"].dtype.kind in {"f", "i"}
    )


def adjust_joint_limit(robot: PhysicalUR10) -> None:
    joint_val = robot.get_q_values()
    if joint_val[5] >= np.pi or joint_val[5] <= -np.pi:
        robot.move_joint(np.array([*joint_val[:5], 0.0]), speed=3.0, acceleration=2.0)


def to_se3_matrix(se2_pose: np.ndarray | list[float]) -> np.ndarray:
    se2_pose = np.asarray(se2_pose, dtype=float).reshape(-1)[:3]
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_euler("z", se2_pose[2], degrees=False).as_matrix()
    matrix[:2, 3] = se2_pose[:2]
    return matrix


def get_object_pose(
    robot: PhysicalUR10,
    max_retries: int = 5,
    rough_detect_pose: np.ndarray | None = None,
    debug_img_id: str | int | None = None,
    debug_dir: str = "debug",
) -> tuple[np.ndarray, object]:
    """Match collect_push_data_real.py: hand-camera local pose -> robot base SE2."""
    tcp_base_pose = robot.get_ee_transform()
    pose = robot.get_object_pose_hand()

    c = 0
    while not pose_valid(pose):
        if c >= max_retries - 1 and rough_detect_pose is not None:
            print("(Camera) Moving to rough pose to try last time.")
            two_step_detect(robot, rough_detect_pose, 0.35)
        if c >= max_retries:
            raise RuntimeError(
                "Camera detection failed. Manual adjustment is required."
            )
        print(f"(Camera) Detection failed, retry {c + 1}/{max_retries}")
        time.sleep(1.0)
        pose = robot.get_object_pose_hand()
        c += 1

    obj_pose_camera = np.asarray(pose["pose"], dtype=float)
    bounding_box = pose.get("bounding_box")
    img = pose.get("result_image")
    if img is not None:
        os.makedirs(debug_dir, exist_ok=True)
        img.save(
            os.path.join(debug_dir, f"{debug_img_id or 'real'}_{time.time():.3f}.jpg")
        )

    tcp_ee_pose = np.eye(4)
    tcp_ee_pose[2, 3] = 0.260
    obj_pose_base = tcp_base_pose @ np.linalg.inv(tcp_ee_pose) @ obj_pose_camera
    obj_se2 = project_se3_pose(matrix_to_flat(obj_pose_base))
    obj_pose_base_se2 = to_se3_matrix(obj_se2)
    _set_pause_object_state(obj_pose_base_se2)

    print("[REAL] Raw hand-camera object translation:", _fmt(obj_pose_camera[:3, 3]))
    print("[REAL] Converted robot/base object SE2:", _fmt(obj_se2))
    return obj_pose_base_se2, bounding_box


def two_step_detect(
    robot: PhysicalUR10,
    rough_detect_pose: np.ndarray,
    height: float = 0.12,
    *,
    pause_between_movements: bool = False,
    label: str = "object detection",
):
    """Match collect_push_data_real.py, with prompts before every robot move."""
    adjust_joint_limit(robot)
    _pause_between_movements(
        pause_between_movements, f"{label}: about to move to rough detection pose"
    )
    robot.move_tool(rough_detect_pose)
    obj_pose, bounding_box = get_object_pose(robot, max_retries=0)
    print(
        f"[REAL] {label}: rough-detected global SE2",
        _fmt(project_se3_pose(matrix_to_flat(obj_pose))),
    )

    _pause_between_movements(
        pause_between_movements, f"{label}: about to move above detected object"
    )
    robot.move_tool(list(obj_pose[:2, 3]) + [0.35, 0.0, np.pi, 0.0])
    obj_pose, bounding_box = get_object_pose(robot, max_retries=0)
    print(
        f"[REAL] {label}: refined global SE2",
        _fmt(project_se3_pose(matrix_to_flat(obj_pose))),
    )

    _pause_between_movements(
        pause_between_movements, f"{label}: about to move to inspection height"
    )
    robot.move_tool(list(obj_pose[:2, 3]) + [height, 0.0, np.pi, 0.0])
    return obj_pose, bounding_box


def execute_push(
    robot: PhysicalUR10,
    ws_path: np.ndarray,
    push_param: np.ndarray,
    *,
    pause_between_movements: bool = False,
) -> None:
    """Match collect_push_data_real.py execute_push, with step prompts."""
    ws_path = np.asarray(ws_path, dtype=float).copy()
    for i in range(len(ws_path)):
        r1 = R.from_quat(ws_path[i][[4, 5, 6, 3]])
        r2 = R.from_euler("z", np.pi / 2, degrees=False)
        r = r1 * r2
        ws_path[i] = np.array([*ws_path[i][:3], *r.as_quat()[[3, 0, 1, 2]]])

    pre_push_pose_flat = ws_path[0].copy()
    pre_push_pose_flat[2] += 0.1
    post_push_pose_flat = ws_path[-1].copy()
    post_push_pose_flat[2] = 0.35

    print("[REAL] Robot motion preview:")
    print("  pre-push:   ", _fmt(pre_push_pose_flat))
    print("  waypoint[0]:", _fmt(ws_path[0]))
    print("  waypoint[-1]:", _fmt(ws_path[-1]))
    print("  post-push:  ", _fmt(post_push_pose_flat))

    adjust_joint_limit(robot)
    _pause_between_movements(pause_between_movements, "About to move to pre-push pose")
    robot.move_tool(pre_push_pose_flat, to_rotvec=True, speed=1.0, acceleration=2.0)
    _pause_between_movements(
        pause_between_movements,
        "At pre-push pose; about to move to contact waypoint[0]",
    )
    robot.move_tool(ws_path[0], to_rotvec=True)
    _pause_between_movements(
        pause_between_movements,
        "At contact waypoint[0]; about to execute push to waypoint[-1]",
    )
    robot.execute_ee_waypoints(ws_path, to_rotvec=True)
    _pause_between_movements(
        pause_between_movements,
        "At push endpoint waypoint[-1]; about to move to post-push pose",
    )
    robot.move_tool(post_push_pose_flat, to_rotvec=True, speed=1.0, acceleration=2.0)

    print("Push Param:", _fmt(push_param))
    if (push_param[2] > 0.15 and abs(push_param[1]) > 0.25) or push_param[2] > 0.2:
        angle = -np.sign(push_param[1]) * np.pi / 2
        final_r = R.from_quat(post_push_pose_flat[[4, 5, 6, 3]])
        final_r = final_r * R.from_euler("z", angle, degrees=False)
        post_push_pose_flat[3:] = final_r.as_quat()[[3, 0, 1, 2]]
        adjust_joint_limit(robot)
        _pause_between_movements(
            pause_between_movements, "About to move to adjusted post-push camera pose"
        )
        robot.move_tool(
            post_push_pose_flat, to_rotvec=True, speed=1.0, acceleration=2.0
        )


def _apply_state_bounds(system, state_bounds: list[tuple[float, float]]) -> None:
    system.state_bounds = [(float(a), float(b)) for a, b in state_bounds]
    bounds = ob.RealVectorBounds(2)
    for i, (low, high) in enumerate(system.state_bounds):
        bounds.setLow(i, float(low))
        bounds.setHigh(i, float(high))
    system.state_space.setBounds(bounds)


def _apply_control_bounds(system, control_bounds: list[tuple[float, float]]) -> None:
    system.control_bounds = [(float(a), float(b)) for a, b in control_bounds]
    bounds = ob.RealVectorBounds(3)
    for i, (low, high) in enumerate(system.control_bounds):
        bounds.setLow(i, float(low))
        bounds.setHigh(i, float(high))
    system.control_space.setBounds(bounds)


def _auto_state_bounds(
    start: np.ndarray, goal: np.ndarray, margin: float
) -> list[tuple[float, float]]:
    start = np.asarray(start, dtype=float).reshape(-1)[:3]
    goal = np.asarray(goal, dtype=float).reshape(-1)[:3]
    xy_min = np.minimum(start[:2], goal[:2]) - float(margin)
    xy_max = np.maximum(start[:2], goal[:2]) + float(margin)
    return [
        (float(xy_min[0]), float(xy_max[0])),
        (float(xy_min[1]), float(xy_max[1])),
    ]


def _resolve_planner_bounds(
    start: np.ndarray,
    goal: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[float, float]]:
    if args.planner_bounds is None:
        return _auto_state_bounds(start, goal, float(args.planner_margin))

    xmin, xmax, ymin, ymax = [float(v) for v in args.planner_bounds]
    if not xmin < xmax:
        raise ValueError("--object-bounds/--planner-bounds requires xmin < xmax")
    if not ymin < ymax:
        raise ValueError("--object-bounds/--planner-bounds requires ymin < ymax")
    start = np.asarray(start, dtype=float).reshape(-1)[:3]
    goal = np.asarray(goal, dtype=float).reshape(-1)[:3]
    for name, state in (("start", start), ("goal", goal)):
        if not (xmin <= state[0] <= xmax and ymin <= state[1] <= ymax):
            raise ValueError(
                f"{name} state {_fmt(state)} is outside planner bounds "
                f"x=[{xmin:.3f}, {xmax:.3f}], y=[{ymin:.3f}, {ymax:.3f}]"
            )
    return [(xmin, xmax), (ymin, ymax)]


def _bounds_center_radius(
    state_bounds: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(state_bounds, dtype=float)
    center = 0.5 * (bounds[:, 0] + bounds[:, 1])
    radius = 0.5 * (bounds[:, 1] - bounds[:, 0])
    return center, radius


def _build_planner(
    system,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    args: argparse.Namespace,
    planning_time: float,
) -> OMPL_Planner:
    planner = OMPL_Planner(
        system=system,
        start_state=np.asarray(start_state, dtype=float).reshape(-1)[:3],
        goal_state=np.asarray(goal_state, dtype=float).reshape(-1)[:3],
        planner_method="aorrt",
        goal_threshold=float(args.planner_goal_threshold),
        min_max_control_duration=(1, 1),
        propagation_step_size=float(args.duration),
        initial_planning_time=float(planning_time),
        pruning_radius=float(args.pruning_radius),
        obstacle_config=None,
    )
    planner.replanning_time = float(args.replanning_time)
    planner.motion_validation_step_size = float(args.duration)
    planner.optimizer_num_states = int(args.optimizer_num_states)
    planner.optimizer_num_steps = int(args.optimizer_epochs)
    planner.optimizer_learning_rate = float(args.optimizer_learning_rate)
    planner.optimizer_pos_std = float(args.optimizer_pos_std)
    planner.optimizer_rot_std = float(args.optimizer_rot_std)
    planner.optimizer_vel_std = 0.003
    planner.optimizer_max_children = int(args.optimizer_max_children)
    planner.solution_continuity_max_distance = float(args.replanning_max_distance)
    planner.recovery_replanning_time = float(args.replanning_time)
    return planner


def _fresh_plan(
    system,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    args: argparse.Namespace,
    planning_time: float,
) -> tuple[OMPL_Planner, dict, float]:
    planner = _build_planner(system, start_state, goal_state, args, planning_time)
    t0 = time.time()
    solutions, _ = planner.plan()
    wall = time.time() - t0
    if not solutions:
        raise RuntimeError(
            f"Planner failed from {_fmt(start_state)} to {_fmt(goal_state)} "
            f"with budget {float(planning_time):.3f}s"
        )
    solution = solutions[0]
    if not solution.get("controls"):
        raise RuntimeError("Planner returned a solution with no controls.")
    return planner, solution, wall


def _predict(
    system, state: np.ndarray, control: np.ndarray, duration: float
) -> np.ndarray:
    pred = np.asarray(system.propagate(state, control, duration), dtype=float).reshape(
        -1
    )[:3]
    pred[2] = _wrap_angle(float(pred[2]))
    return pred


def _oriented_box_corners(state: np.ndarray, obj_shape: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(-1)[:3]
    width, length = np.asarray(obj_shape, dtype=float).reshape(-1)[:2]
    local = np.array(
        [
            [-0.5 * width, -0.5 * length],
            [0.5 * width, -0.5 * length],
            [0.5 * width, 0.5 * length],
            [-0.5 * width, 0.5 * length],
            [-0.5 * width, -0.5 * length],
        ],
        dtype=float,
    )
    c, s = np.cos(state[2]), np.sin(state[2])
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return state[:2] + local @ rot.T


def _draw_state_box(
    ax,
    state: np.ndarray,
    obj_shape: np.ndarray,
    *,
    edgecolor: str,
    facecolor: str = "none",
    linewidth: float = 1.4,
    alpha: float = 1.0,
    label: str | None = None,
) -> None:
    corners = _oriented_box_corners(state, obj_shape)
    ax.plot(
        corners[:, 0],
        corners[:, 1],
        color=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )
    if facecolor != "none":
        ax.fill(corners[:, 0], corners[:, 1], color=facecolor, alpha=0.12)
    state = np.asarray(state, dtype=float).reshape(-1)[:3]
    arrow_len = 0.08
    ax.arrow(
        state[0],
        state[1],
        arrow_len * np.cos(state[2]),
        arrow_len * np.sin(state[2]),
        color=edgecolor,
        width=0.003,
        head_width=0.018,
        length_includes_head=True,
        alpha=alpha,
    )


def _save_plan_xy_plot(
    out_dir: Path,
    method: str,
    step: int,
    states: list[np.ndarray],
    measured_start: np.ndarray,
    goal: np.ndarray,
    state_bounds: list[tuple[float, float]],
    obj_shape: np.ndarray,
    model_name: str,
) -> Path:
    plan_dir = out_dir / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    path = plan_dir / f"{method}_step_{step:03d}_xy_plan.png"

    states_arr = np.asarray(states, dtype=float).reshape(-1, 3)
    measured_start = np.asarray(measured_start, dtype=float).reshape(-1)[:3]
    goal = np.asarray(goal, dtype=float).reshape(-1)[:3]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.set_title(f"{method.upper()} planned states - {model_name}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.88", linewidth=0.8)

    bounds = np.asarray(state_bounds, dtype=float)
    ax.add_patch(
        plt.Rectangle(
            (bounds[0, 0], bounds[1, 0]),
            bounds[0, 1] - bounds[0, 0],
            bounds[1, 1] - bounds[1, 0],
            fill=False,
            edgecolor="0.55",
            linewidth=1.0,
            linestyle="--",
            label="planner bounds",
        )
    )

    ax.plot(
        states_arr[:, 0],
        states_arr[:, 1],
        "-o",
        color="#0b4fba",
        linewidth=2.4,
        markersize=4.0,
        label="planned state path",
    )
    ax.scatter(
        [measured_start[0]],
        [measured_start[1]],
        s=90,
        color="#1b9e77",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label="measured start",
    )
    ax.scatter(
        [goal[0]],
        [goal[1]],
        s=120,
        marker="*",
        color="#d95f02",
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
        label="goal",
    )

    _draw_state_box(
        ax,
        measured_start,
        obj_shape,
        edgecolor="#1b9e77",
        facecolor="#1b9e77",
        linewidth=2.0,
        label="start box",
    )
    _draw_state_box(
        ax,
        goal,
        obj_shape,
        edgecolor="#d95f02",
        linewidth=2.0,
        label="goal box",
    )
    for idx, state in enumerate(states_arr[1:-1], start=1):
        _draw_state_box(
            ax,
            state,
            obj_shape,
            edgecolor="#0b4fba",
            linewidth=1.0,
            alpha=0.35,
            label="planned boxes" if idx == 1 else None,
        )

    all_xy = np.vstack([states_arr[:, :2], measured_start[:2], goal[:2], bounds.T])
    pad = max(0.12, 0.2 * float(np.max(np.ptp(all_xy, axis=0))))
    ax.set_xlim(float(np.min(all_xy[:, 0]) - pad), float(np.max(all_xy[:, 0]) + pad))
    ax.set_ylim(float(np.min(all_xy[:, 1]) - pad), float(np.max(all_xy[:, 1]) + pad))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _show_plan_plot_for_review(path: Path, args: argparse.Namespace) -> None:
    if not args.show_plan_plot:
        return

    print(f"[REAL] Opening XY plan visualization: {path}")
    try:
        subprocess.Popen(
            ["xdg-open", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        print(f"[WARNING] Could not open plan plot automatically: {exc}")
        print(f"[WARNING] Open this file manually: {path}")

    if args.confirm_plan_plot:
        print("[REAL] Review the XY plan plot; sleeping 0.5s before continuing.")
        time.sleep(0.5)


def _canonical_push_control(control: np.ndarray) -> np.ndarray:
    control = np.asarray(control, dtype=float).reshape(-1).copy()
    if control.size < 3:
        raise ValueError(f"Expected 3D push control, got {control}")
    normalized_faces = np.array([0.0, 0.25, 0.5, 0.75], dtype=float)
    rad_faces = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    face_raw = float(control[0])
    if 0.0 <= face_raw <= 0.75:
        face_idx = int(np.round(face_raw * 4.0)) % 4
    elif np.min(np.abs(face_raw - rad_faces)) < 1e-6:
        face_idx = int(np.argmin(np.abs(face_raw - rad_faces)))
    elif abs(face_raw - round(face_raw)) < 1e-9 and 0 <= round(face_raw) <= 3:
        face_idx = int(round(face_raw)) % 4
    elif 0.0 <= face_raw < 4.0:
        face_idx = int(face_raw) % 4
    else:
        face_idx = int(face_raw / (np.pi / 2.0)) % 4
    control[0] = normalized_faces[face_idx]
    control[1] = float(np.clip(control[1], -0.4, 0.4))
    control[2] = float(np.clip(control[2], 0.0, 0.30))
    return control[:3]


def _push_face_summary(control: np.ndarray, obj_shape: np.ndarray) -> str:
    control = _canonical_push_control(control)
    face_idx = int(round(float(control[0]) * 4.0)) % 4
    width, length = np.asarray(obj_shape, dtype=float).reshape(-1)[:2]
    push_side = ["+x", "+y", "-x", "-y"][face_idx]
    contacted_edge = float(length if face_idx % 2 == 0 else width)
    side_kind = "longer" if contacted_edge >= max(width, length) - 1e-9 else "shorter"
    return (
        f"face={face_idx} ({push_side}), normalized={control[0]:.2f}, "
        f"contact edge={contacted_edge:.4f}m ({side_kind} side)"
    )


def _print_plan_summary(
    states: list[np.ndarray],
    controls: list[np.ndarray],
    duration: float,
    max_push_distance: float,
) -> None:
    states_arr = np.asarray(states, dtype=float).reshape(-1, 3)
    controls_arr = np.asarray(controls, dtype=float).reshape(-1, 3)
    xy_steps = np.linalg.norm(np.diff(states_arr[:, :2], axis=0), axis=1)
    total_xy = float(np.sum(xy_steps))
    direct_xy = float(np.linalg.norm(states_arr[-1, :2] - states_arr[0, :2]))
    max_xy_step = float(np.max(xy_steps)) if len(xy_steps) else 0.0
    mean_xy_step = float(np.mean(xy_steps)) if len(xy_steps) else 0.0
    max_control_distance = (
        float(np.max(controls_arr[:, 2])) if len(controls_arr) else 0.0
    )
    print(
        "[REAL] plan summary: "
        f"{len(controls)} controls, direct_xy={direct_xy:.3f}m, "
        f"path_xy={total_xy:.3f}m, mean_step_xy={mean_xy_step:.3f}m, "
        f"max_step_xy={max_xy_step:.3f}m, max_control_distance={max_control_distance:.3f}m, "
        f"duration/control={float(duration):.3f}s"
    )
    if len(controls) > 12:
        print(
            "[REAL] plan has many controls. Current max push distance is "
            f"{float(max_push_distance):.3f}m; if you expect 7-8 controls, "
            "this bound is usually the first thing to check."
        )


def _save_rows(out_dir: Path, rows: list[RealExecutionStep], metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out_dir / "real_execution_steps.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    np.savez(
        out_dir / "real_execution_results.npz",
        rows=np.array([asdict(row) for row in rows], dtype=object),
        metadata=np.array(metadata, dtype=object),
    )


def _execute_real_push_from_control(
    robot: PhysicalUR10,
    obj_pose: np.ndarray,
    obj_shape: np.ndarray,
    tool_offset: Pose,
    control: np.ndarray,
    args: argparse.Namespace,
    duration: float | None = None,
) -> None:
    _set_pause_object_state(obj_pose)
    # Keep the learned-model/collect_push_data convention: face ids are
    # normalized discrete values 0, 0.25, 0.5, 0.75.  The planner/optimizer may
    # return a nearby continuous value, so snap before building the real path.
    path_control = _canonical_push_control(control)
    print("[REAL] planner/optimizer raw control:", _fmt(control))
    print("[REAL] canonical real path control:", _fmt(path_control))
    print(
        "[REAL] canonical real push face:", _push_face_summary(path_control, obj_shape)
    )
    _, ws_path = generate_path_form_params(
        Pose(obj_pose[:3, 3], matrix_to_quat(obj_pose[:3, :3])),
        obj_shape,
        path_control,
        tool_offset=tool_offset,
        total_time=float(args.duration if duration is None else duration),
        pre_push_offset=float(args.pre_push_offset),
        push_height=float(args.push_height),
        dt=float(args.execution_dt),
        relative_push_offset=True,
    )
    execute_push(
        robot,
        ws_path,
        path_control,
        pause_between_movements=bool(args.pause_between_movements),
    )


class RealPushingSimulator(Simulator):
    """Simulator adapter that lets AURA/Replanning drive the real UR10."""

    def __init__(
        self,
        *,
        robot: PhysicalUR10,
        initial_pose: np.ndarray,
        obj_shape: np.ndarray,
        tool_offset: Pose,
        rough_detect_pose: np.ndarray,
        goal_state: np.ndarray,
        args: argparse.Namespace,
        method_label: str,
    ):
        initial_state = project_se3_pose(matrix_to_flat(initial_pose))
        initial_state[2] = _wrap_angle(float(initial_state[2]))
        config = {
            "start_state": initial_state.tolist(),
            "goal_state": np.asarray(goal_state, dtype=float).reshape(-1)[:3].tolist(),
            "propagation_step_size": float(args.duration),
            # Real execution includes user prompts and robot motion, so AURA must
            # wait for the execution thread instead of treating wall time as a
            # simulated one-control deadline.
            "execution_timeout_slack": float(args.execution_timeout_slack),
        }
        super().__init__("pushing_object", config=config)
        self.robot = robot
        self.current_pose = np.asarray(initial_pose, dtype=float).copy()
        self.obj_shape = np.asarray(obj_shape, dtype=float).copy()
        self.tool_offset = tool_offset
        self.rough_detect_pose = np.asarray(rough_detect_pose, dtype=float).copy()
        self.goal_state = np.asarray(goal_state, dtype=float).reshape(-1)[:3].copy()
        self.args = args
        self.method_label = method_label
        self.executed_segments = 0
        _set_pause_object_state(self.current_state)

    def reset(self):
        self.running = True
        self.current_state = self.start_state.copy()
        self.current_pose = to_se3_matrix(self.current_state)
        _set_pause_object_state(self.current_state)
        return self.get_state()

    def get_state(self):
        return np.asarray(self.current_state, dtype=float).copy()

    def set_obj_init_pose(self, pose):
        pose_np = np.asarray(pose, dtype=float).reshape(-1)[:3]
        pose_np[2] = _wrap_angle(float(pose_np[2]))
        self.current_state = pose_np.copy()
        self.start_state = pose_np.copy()
        self.current_pose = to_se3_matrix(pose_np)
        _set_pause_object_state(self.current_state)
        return self.get_state()

    def execute_segment(self, control, duration):
        self.running = True
        self.executed_segments += 1
        control_np = np.asarray(control, dtype=float).reshape(-1)[:3]
        duration = float(duration)
        print(
            f"\n[REAL] ===== {self.method_label} segment {self.executed_segments} ====="
        )
        print("[REAL] measured start:", _fmt(self.current_state))
        print("[REAL] control:", _fmt(control_np))
        print(
            "[REAL] canonical real push face:",
            _push_face_summary(control_np, self.obj_shape),
        )

        _execute_real_push_from_control(
            self.robot,
            self.current_pose,
            self.obj_shape,
            self.tool_offset,
            control_np,
            self.args,
            duration=duration,
        )
        _pause_between_movements(
            bool(self.args.pause_between_movements),
            "Post-push motion complete; about to run two-step object detection",
        )
        time.sleep(float(self.args.detection_wait))
        self.current_pose, _ = two_step_detect(
            self.robot,
            self.rough_detect_pose,
            height=float(self.args.post_detect_height),
            pause_between_movements=bool(self.args.pause_between_movements),
            label=f"{self.method_label} segment {self.executed_segments} post-push two-step detection",
        )
        measured = project_se3_pose(matrix_to_flat(self.current_pose))
        measured[2] = _wrap_angle(float(measured[2]))
        self.current_state = measured.copy()
        _set_pause_object_state(self.current_state)
        goal_distance = float(
            arrayDistance(self.current_state, self.goal_state, system=self.system_name)
        )
        print("[REAL] measured end:  ", _fmt(self.current_state))
        print(f"[REAL] goal distance after segment: {goal_distance:.6f}")
        return self.get_state()


def _plan_states_from_update(plan: dict) -> list[np.ndarray]:
    states = plan.get("states") or plan.get("display_states") or []
    return [np.asarray(s, dtype=float).reshape(-1)[:3].copy() for s in states]


def _plan_controls_from_update(plan: dict) -> list[np.ndarray]:
    controls = plan.get("controls") or []
    return [np.asarray(u, dtype=float).reshape(-1)[:3].copy() for u in controls]


def _make_plan_update_callback(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    method: str,
    goal: np.ndarray,
    state_bounds: list[tuple[float, float]],
    obj_shape: np.ndarray,
    model_name: str,
):
    counter = {"value": 0}

    def _callback(step: int, plan: dict, current_state: np.ndarray) -> None:
        states = _plan_states_from_update(plan)
        controls = _plan_controls_from_update(plan)
        if not states:
            return
        if controls:
            duration = float((plan.get("time") or [args.duration])[0])
            _print_plan_summary(
                states, controls, duration, float(args.max_push_distance)
            )
        if args.plan_plot:
            plot_path = _save_plan_xy_plot(
                out_dir,
                method,
                counter["value"],
                states,
                np.asarray(current_state, dtype=float).reshape(-1)[:3],
                goal,
                state_bounds,
                obj_shape,
                model_name,
            )
            print(f"[REAL] XY plan visualization saved: {plot_path}")
            _show_plan_plot_for_review(plot_path, args)
        counter["value"] += 1

    return _callback


def _make_replanning_update_callback(aura_style_callback):
    counter = {"step": 0}

    def _callback(solution: dict) -> None:
        current = solution.get("states", [None])[0]
        if current is None:
            return
        aura_style_callback(counter["step"], solution, np.asarray(current, dtype=float))
        counter["step"] += 1

    return _callback


def _planner_config(
    *,
    start_state: np.ndarray,
    goal: np.ndarray,
    args: argparse.Namespace,
    control_bounds: list[tuple[float, float]],
    state_bounds: list[tuple[float, float]],
) -> dict:
    return {
        "start_state": np.asarray(start_state, dtype=float).reshape(-1)[:3].tolist(),
        "goal_state": np.asarray(goal, dtype=float).reshape(-1)[:3].tolist(),
        "goal_threshold": float(args.goal_threshold),
        "actual_goal_threshold": float(args.goal_threshold),
        "planner_goal_threshold": float(args.planner_goal_threshold),
        "planning_time": float(args.planning_time),
        "replanning_time_budget": float(args.replanning_time),
        "control_duration_seconds": float(args.duration),
        "min_control_duration": 1,
        "max_control_duration": 1,
        "propagation_step_size": float(args.duration),
        "pruning_radius": float(args.pruning_radius),
        "control_bounds": control_bounds,
        "state_bounds": state_bounds,
        "replanningMaxDistance": float(args.replanning_max_distance),
    }


def run_real_execution(args: argparse.Namespace) -> dict:
    method = str(args.method).lower()
    system = get_system("pushing_object")
    control_bounds = [(0.0, 0.75), (-0.4, 0.4), (0.0, float(args.max_push_distance))]
    _apply_control_bounds(system, control_bounds)

    goal = np.asarray(args.goal, dtype=float).reshape(3)
    rough_detect_pose = np.asarray(args.rough_detect_pose, dtype=float)
    obj_shape = get_obj_shape(f"assets/{args.obj_name}/textured.obj")
    system.object_shape = np.asarray(obj_shape, dtype=float).copy()
    system.model_name = str(args.model_name)
    system.model_path = args.model_path
    tool_offset = Pose([0.0, 0.0, -float(args.push_offset)], [1.0, 0.0, 0.0, 0.0])
    out_dir = Path(args.output_dir)
    rows: list[RealExecutionStep] = []

    opt_model = None
    if method == "aura":
        pushing_model = get_pushing_model(
            system.object_shape,
            model_name=system.model_name,
            model_path=system.model_path,
        )
        opt_model = load_opt_model_2(
            pushing_model,
            lr=float(args.optimizer_learning_rate),
            epochs=int(args.optimizer_epochs),
        )

    print("[REAL] Connecting to PhysicalUR10...")
    robot = PhysicalUR10()
    _pause_between_movements(
        bool(args.pause_between_movements),
        "Connected to robot; about to move to home joint pose",
    )
    robot.move_joint(np.asarray(args.robot_home, dtype=float))

    print("[REAL] Detecting current object pose; this is the execution start.")
    obj_pose, _ = two_step_detect(
        robot,
        rough_detect_pose,
        height=float(args.detect_height),
        pause_between_movements=bool(args.pause_between_movements),
        label=f"{method} initial two-step detection",
    )
    measured_start = project_se3_pose(matrix_to_flat(obj_pose))
    measured_start[2] = _wrap_angle(float(measured_start[2]))
    _set_pause_object_state(measured_start)
    state_bounds = _resolve_planner_bounds(measured_start, goal, args)
    _apply_state_bounds(system, state_bounds)

    metadata = {
        "method": method,
        "obj_name": args.obj_name,
        "goal": goal.tolist(),
        "start_state": measured_start.tolist(),
        "state_bounds": state_bounds,
        "control_bounds": control_bounds,
        "duration": float(args.duration),
        "goal_threshold": float(args.goal_threshold),
        "actual_goal_threshold": float(args.goal_threshold),
        "planner_goal_threshold": float(args.planner_goal_threshold),
        "tracking_error_threshold": float(args.tracking_error_threshold),
        "planning_time": float(args.planning_time),
        "replanning_time": float(args.replanning_time),
        "push_offset": float(args.push_offset),
        "push_height": float(args.push_height),
        "pre_push_offset": float(args.pre_push_offset),
        "relative_side_offset": True,
        "model_name": system.model_name,
        "model_path": system.model_path,
        "max_steps": None if args.max_steps is None else int(args.max_steps),
        "replanning_max_distance": float(args.replanning_max_distance),
    }
    _save_rows(out_dir, rows, metadata)

    print("[REAL] Start:", _fmt(measured_start))
    print("[REAL] Goal: ", _fmt(goal))
    print("[REAL] Planner state bounds:", state_bounds)
    if args.planner_bounds is None:
        print(
            "[REAL] Planner bounds source: auto from start/goal "
            f"with margin {float(args.planner_margin):.3f} m"
        )
    else:
        print("[REAL] Planner bounds source: explicit --object-bounds/--planner-bounds")
    print("[REAL] Control bounds:", control_bounds)
    print(
        "[REAL] Thresholds: "
        f"planner={float(args.planner_goal_threshold):.6f}, "
        f"actual/final={float(args.goal_threshold):.6f}, "
        f"tracking={float(args.tracking_error_threshold):.6f}"
    )
    print("[REAL] Object mesh shape used for real path:", _fmt(obj_shape))
    print("[REAL] Learned model shape:", _fmt(system.object_shape))
    print("[REAL] Learned model name:", system.model_name)
    if args.require_confirm:
        print("[REAL] object state before prompt:", _fmt(measured_start))
        print("[REAL] Starting real execution after 0.5s.")
        time.sleep(0.5)

    real_sim = RealPushingSimulator(
        robot=robot,
        initial_pose=obj_pose,
        obj_shape=obj_shape,
        tool_offset=tool_offset,
        rough_detect_pose=rough_detect_pose,
        goal_state=goal,
        args=args,
        method_label=method,
    )
    plan_callback = _make_plan_update_callback(
        args=args,
        out_dir=out_dir,
        method=method,
        goal=goal,
        state_bounds=state_bounds,
        obj_shape=obj_shape,
        model_name=system.model_name,
    )

    status = "not_started"
    final_state = measured_start.copy()
    last_planned_final = np.full(3, np.nan, dtype=float)
    total_controls = 0
    tracking_error_mean = 0.0
    tracking_error_max = 0.0
    all_tracking_errors: list[float] = []

    if method == "aura":
        episode = 0
        while True:
            current_state = np.asarray(real_sim.get_state(), dtype=float).reshape(-1)[
                :3
            ]
            final_state = current_state.copy()
            goal_distance = float(
                arrayDistance(current_state, goal, system=system.name)
            )
            print(f"\n[REAL] ===== AURA run {episode + 1} =====")
            print("[REAL] measured start:", _fmt(current_state))
            print(f"[REAL] goal distance before run: {goal_distance:.6f}")
            if goal_distance <= float(args.goal_threshold):
                status = "success"
                break
            if args.max_steps is not None and total_controls >= int(args.max_steps):
                status = "max_steps_reached"
                break

            budget = float(args.planning_time if episode == 0 else args.replanning_time)
            planner, _solution, planning_wall = _fresh_plan(
                system,
                current_state,
                goal,
                args,
                budget,
            )
            planner.opt_model = opt_model
            print(f"[REAL] AURA planner wall time: {planning_wall:.3f}s")
            aura = AURA(system, planner, real_sim)
            remaining = (
                None
                if args.max_steps is None
                else max(0, int(args.max_steps) - total_controls)
            )
            result = aura.run(
                reset_sim=False,
                on_planning_update=plan_callback,
                pause_each_step=False,
                max_steps=remaining,
            )
            total_controls += int(result.num_controls)
            run_tracking_errors = [float(v) for v in (result.tracking_error_list or [])]
            if run_tracking_errors:
                all_tracking_errors.extend(run_tracking_errors)
                tracking_error_mean = float(np.mean(all_tracking_errors))
                tracking_error_max = float(np.max(all_tracking_errors))
            final_state = np.asarray(real_sim.get_state(), dtype=float).reshape(-1)[:3]
            if result.final_planned_state is not None:
                last_planned_final = np.asarray(
                    result.final_planned_state, dtype=float
                ).reshape(-1)[:3]
            final_goal_distance = float(
                arrayDistance(final_state, goal, system=system.name)
            )
            print(
                f"[REAL] AURA run {episode + 1} finished: status={result.status}, "
                f"controls={result.num_controls}, final goal distance={final_goal_distance:.6f}, "
                f"tracking mean={tracking_error_mean:.6f}, tracking max={tracking_error_max:.6f}"
            )
            if tracking_error_max > float(args.tracking_error_threshold):
                print(
                    "[WARNING] AURA tracking error exceeded the requested threshold: "
                    f"max {tracking_error_max:.6f} > {float(args.tracking_error_threshold):.6f}. "
                    "Continuing from the freshly measured object pose."
                )
            if final_goal_distance <= float(args.goal_threshold):
                status = "success"
                break
            if result.status != "success":
                status = f"aura_{result.status}:{result.failure_reason}"
                break
            if result.num_controls <= 0:
                status = "aura_no_progress"
                break
            print(
                "[REAL] AURA completed its current plan outside the goal threshold; "
                "planning another AURA run from the freshly measured object pose."
            )
            episode += 1
    else:
        config = _planner_config(
            start_state=measured_start,
            goal=goal,
            args=args,
            control_bounds=control_bounds,
            state_bounds=state_bounds,
        )
        replanning_callback = _make_replanning_update_callback(plan_callback)
        runner = ReplanningRunner(
            "pushing_object",
            "aorrt",
            config,
            simulator_mode="real",
            max_steps=args.max_steps,
            plan_update_callback=replanning_callback,
            system_override=system,
            simulator_override=real_sim,
        )
        result = runner.run()
        total_controls = int(result.num_controls)
        tracking_error_mean = float(result.tracking_error_mean)
        tracking_error_max = (
            float(np.max(result.tracking_error_list))
            if result.tracking_error_list
            else 0.0
        )
        final_state = np.asarray(result.final_state, dtype=float).reshape(-1)[:3]
        last_planned_final = np.asarray(
            result.planned_final_state, dtype=float
        ).reshape(-1)[:3]
        final_goal_distance = float(
            arrayDistance(final_state, goal, system=system.name)
        )
        status = (
            "success"
            if final_goal_distance <= float(args.goal_threshold)
            else "replanning_stopped_outside_goal"
        )

    final_goal_distance = float(arrayDistance(final_state, goal, system=system.name))
    print("\n[REAL] Done.")
    print(f"[REAL] status: {status}")
    print("[REAL] final state: ", _fmt(final_state))
    print("[REAL] goal:        ", _fmt(goal))
    print(f"[REAL] final goal distance: {final_goal_distance:.6f}")
    print(
        f"[REAL] tracking mean/max: {tracking_error_mean:.6f} / {tracking_error_max:.6f}"
    )
    print(f"[REAL] executed controls: {total_controls}")
    print(f"[REAL] results saved in {out_dir}")

    metadata.update(
        {
            "status": status,
            "final_state": final_state.tolist(),
            "final_goal_distance": final_goal_distance,
            "planned_final_state": last_planned_final.tolist(),
            "executed_controls": int(total_controls),
            "tracking_error_mean": float(tracking_error_mean),
            "tracking_error_max": float(tracking_error_max),
        }
    )
    _save_rows(out_dir, rows, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    limits = REAL_EXECUTION_LIMITS
    parser = argparse.ArgumentParser(description="Real UR10 AURA/Replanning execution.")
    parser.add_argument("--method", choices=["aura", "replanning"], required=True)
    parser.add_argument("--obj-name", default="cracker_box_flipped")
    parser.add_argument(
        "--model-name",
        default="real_cracker_box",
        help="Learned pushing model name under learned_models/<name>_mlp_0.0_1000_0.pth.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Explicit learned pushing model checkpoint path. Overrides --model-name.",
    )
    parser.add_argument("--goal", type=float, nargs=3, default=[0.35, -0.7, 0.0])
    parser.add_argument(
        "--goal-threshold",
        type=float,
        default=limits["actual_goal_threshold"],
        help="Measured real-world stop threshold. This can be looser than planning.",
    )
    parser.add_argument(
        "--planner-goal-threshold",
        type=float,
        default=limits["planner_goal_threshold"],
        help="Tighter OMPL goal threshold used to accept planned solutions.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Optional safety cap on executed controls. By default there is no cap; "
            "AURA/Replanning run until the goal threshold is met or planning fails."
        ),
    )
    parser.add_argument("--planning-time", type=float, default=4.0)
    parser.add_argument("--replanning-time", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--execution-dt", type=float, default=0.008)
    parser.add_argument("--detection-wait", type=float, default=0.0)
    parser.add_argument("--push-offset", type=float, default=0.01)
    parser.add_argument("--push-height", type=float, default=0.035)
    parser.add_argument("--pre-push-offset", type=float, default=0.03)
    parser.add_argument("--post-detect-height", type=float, default=0.25)
    parser.add_argument("--detect-height", type=float, default=0.15)
    parser.add_argument(
        "--max-push-distance",
        type=float,
        default=limits["max_push_distance"],
    )
    parser.add_argument(
        "--object-bounds",
        "--planner-bounds",
        dest="planner_bounds",
        type=float,
        nargs=4,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help=(
            "Explicit XY bounds for object/planner states. If omitted, bounds are "
            "built from detected start, goal, and --planner-margin."
        ),
    )
    parser.add_argument(
        "--planner-margin", type=float, default=limits["planner_margin"]
    )
    parser.add_argument(
        "--pruning-radius", type=float, default=limits["pruning_radius"]
    )
    parser.add_argument(
        "--replanning-max-distance",
        type=float,
        default=limits["replanning_max_distance"],
    )
    parser.add_argument(
        "--tracking-error-threshold",
        type=float,
        default=limits["tracking_error_threshold"],
        help=(
            "Warn when AURA's measured-vs-predicted tracking error exceeds this "
            "threshold. The next run still starts from the freshly measured pose."
        ),
    )
    parser.add_argument("--optimizer-pos-std", type=float, default=0.035)
    parser.add_argument("--optimizer-rot-std", type=float, default=0.35)
    parser.add_argument("--optimizer-num-states", type=int, default=5000)
    parser.add_argument("--optimizer-max-children", type=int, default=500)
    parser.add_argument("--optimizer-learning-rate", type=float, default=5e-3)
    parser.add_argument("--optimizer-epochs", type=int, default=300)
    parser.add_argument(
        "--execution-timeout-slack",
        type=float,
        default=3600.0,
        help=(
            "Extra seconds AURA waits for real robot execution after the nominal "
            "control duration. Large default keeps manual prompts from being timed out."
        ),
    )
    parser.add_argument(
        "--rough-detect-pose",
        type=float,
        nargs=6,
        default=[-0.05, -0.65, 0.6, 0.0, np.pi, 0.0],
    )
    parser.add_argument(
        "--robot-home",
        type=float,
        nargs=6,
        default=[1.3, -1.571, 1.2, -1.2, -1.571, -0.271],
    )
    parser.add_argument("--output-dir", default="results/real_execution")
    parser.add_argument(
        "--pause-between-movements",
        dest="pause_between_movements",
        action="store_true",
        default=True,
        help="Prompt before each real robot motion phase so the setup can be inspected.",
    )
    parser.add_argument(
        "--no-pause-between-movements",
        dest="pause_between_movements",
        action="store_false",
    )
    parser.add_argument(
        "--plan-plot",
        dest="plan_plot",
        action="store_true",
        default=True,
        help="Save an XY image of each planned state sequence before execution.",
    )
    parser.add_argument("--no-plan-plot", dest="plan_plot", action="store_false")
    parser.add_argument(
        "--show-plan-plot",
        dest="show_plan_plot",
        action="store_true",
        default=True,
        help="Open the saved XY plan image before executing the planned push.",
    )
    parser.add_argument(
        "--no-show-plan-plot", dest="show_plan_plot", action="store_false"
    )
    parser.add_argument(
        "--confirm-plan-plot",
        dest="confirm_plan_plot",
        action="store_true",
        default=True,
        help="Wait for Enter after opening the XY plan plot.",
    )
    parser.add_argument(
        "--no-confirm-plan-plot",
        dest="confirm_plan_plot",
        action="store_false",
    )
    parser.add_argument("--no-confirm", dest="require_confirm", action="store_false")
    parser.set_defaults(require_confirm=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.goal_threshold <= 0:
        raise ValueError("--goal-threshold must be positive")
    if args.planner_goal_threshold <= 0:
        raise ValueError("--planner-goal-threshold must be positive")
    if args.goal_threshold < args.planner_goal_threshold:
        raise ValueError(
            "--goal-threshold is the real final-check threshold and should be "
            "greater than or equal to --planner-goal-threshold"
        )
    if args.tracking_error_threshold <= 0:
        raise ValueError("--tracking-error-threshold must be positive")
    if args.max_push_distance <= 0:
        raise ValueError("--max-push-distance must be positive")
    if args.replanning_max_distance <= 0:
        raise ValueError("--replanning-max-distance must be positive")
    if args.execution_timeout_slack < 0:
        raise ValueError("--execution-timeout-slack must be non-negative")
    run_real_execution(args)


if __name__ == "__main__":
    main()
