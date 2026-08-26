#!/usr/bin/env python3
"""Real-robot execution of AURA and replanning pushing to a fixed goal.

This script starts from the currently detected object pose, then lets the actual
AURA or ReplanningRunner class drive real UR10 pushes through the same
motion-generation path used by this repository: two-step object detection,
relative side-offset push params, pre-push/pre-grasp lift, push waypoints,
post-push lift, and optional movement pauses.

Typical usage:
  /usr/bin/python3 real_world/real_execution.py --method aura
  /usr/bin/python3 real_world/real_execution.py --method replanning --goal 0.35 -0.5 0.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
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
import yaml

from aura.AURA import AURA
from methods.Replanning import ReplanningRunner
from geometry.object_model import get_obj_shape
from geometry.pose import Pose, matrix_to_flat, project_se3_to_se2, wrap_to_pi
from geometry.random_push import generate_path_form_params
from methods.plan import OMPLPlanner, duration_seconds_to_steps
from simulation.pushing_model import get_pushing_model
from utils.experiment_io import result_path, upsert_result
from real_world.physical_robot import PhysicalUR10
from simulation.simulator import Simulator
from propagators import get_system
from train_model import load_opt_model_2
from utils.utils import arrayDistance

_LATEST_OBJECT_STATE_FOR_PAUSES: np.ndarray | None = None
_OPERATOR_PAUSE_SECONDS = 0.0
_OPERATOR_PAUSE_LOCK = threading.Lock()

# Keep real-world tolerances and distance limits in one place. CLI flags below
# still override these, but this block is the source of truth for defaults.
REAL_EXECUTION_LIMITS = {
    # OMPL must produce a plan ending inside this tighter goal region.
    "planner_goal_threshold": 0.075,
    # Real camera/robot execution stops once the measured object is within this.
    "actual_goal_threshold": 0.075,
    # Warn if measured execution drifts farther than this from AURA's prediction.
    "tracking_error_threshold": 0.1,
    # Max continuity/tracking miss before replanning treats the path as stale.
    "replanning_max_distance": 0.1,
    # Push distance is absolute meters. Keep this at the real data/model limit.
    "max_push_distance": 0.13,
    "planner_margin": 0.25,
    "pruning_radius": 0.08,
}

ROBOT_HOME_SPEED = 0.6
ROBOT_HOME_ACCELERATION = 0.8
DETECTION_MOVE_SPEED = 0.5
DETECTION_MOVE_ACCELERATION = 0.3
PUSH_TRANSIT_SPEED = 0.5
PUSH_TRANSIT_ACCELERATION = 1.0
PUSH_CONTACT_SPEED = 0.15
PUSH_CONTACT_ACCELERATION = 0.5
WRIST_ADJUSTMENT_SPEED = 1.0
WRIST_ADJUSTMENT_ACCELERATION = 1.0

# The hand-camera detector reports its local x axis along the cracker box's
# physical long dimension.  The pushing mesh/model uses local y for that long
# dimension, so rotate detections by +90 degrees before planning or execution.
CAMERA_TO_PUSHING_YAW_OFFSET = np.pi / 2.0


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


def recorded_task_time(status: str, elapsed: float, limit: float) -> float:
    """Store the exact experiment limit for a timed-out trial."""

    return float(limit) if status == "timeout" else float(elapsed)


def _set_pause_object_state(state: np.ndarray | None) -> None:
    global _LATEST_OBJECT_STATE_FOR_PAUSES
    if state is None:
        _LATEST_OBJECT_STATE_FOR_PAUSES = None
        return
    arr = np.asarray(state, dtype=float)
    if arr.shape == (4, 4):
        arr = project_se3_to_se2(matrix_to_flat(arr))
    else:
        arr = arr.reshape(-1)[:3]
    arr = arr.astype(float, copy=True)
    arr[2] = wrap_to_pi(float(arr[2]))
    _LATEST_OBJECT_STATE_FOR_PAUSES = arr


def _operator_pause_total() -> float:
    with _OPERATOR_PAUSE_LOCK:
        return float(_OPERATOR_PAUSE_SECONDS)


def _recorded_operator_pause(seconds: float) -> None:
    """Sleep deliberately and retain time that must be excluded from task time."""
    global _OPERATOR_PAUSE_SECONDS
    started = time.monotonic()
    time.sleep(float(seconds))
    elapsed = time.monotonic() - started
    with _OPERATOR_PAUSE_LOCK:
        _OPERATOR_PAUSE_SECONDS += elapsed


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
    _recorded_operator_pause(0.5)


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
        robot.move_joint(
            np.array([*joint_val[:5], 0.0]),
            speed=WRIST_ADJUSTMENT_SPEED,
            acceleration=WRIST_ADJUSTMENT_ACCELERATION,
            label="wrist joint-limit adjustment",
        )


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
    """Convert a hand-camera pose into the calibrated pushing/model frame."""
    tcp_base_pose = robot.get_ee_transform()
    pose = robot.get_object_pose_hand()

    c = 0
    while not pose_valid(pose):
        if c >= max_retries - 1 and rough_detect_pose is not None:
            print("(Camera) Moving to rough pose to try last time.")
            two_step_detection(robot, rough_detect_pose, 0.35)
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
    detected_se2 = project_se3_to_se2(matrix_to_flat(obj_pose_base))
    obj_se2 = np.asarray(detected_se2, dtype=float).copy()
    obj_se2[2] = wrap_to_pi(float(obj_se2[2]) + CAMERA_TO_PUSHING_YAW_OFFSET)
    obj_pose_base_se2 = to_se3_matrix(obj_se2)
    _set_pause_object_state(obj_pose_base_se2)

    return obj_pose_base_se2, bounding_box


def two_step_detection(
    robot: PhysicalUR10,
    rough_detect_pose: np.ndarray,
    height: float = 0.12,
    *,
    pause_between_movements: bool = False,
    label: str = "object detection",
):
    """Detect roughly, then detect again from directly above the object."""
    adjust_joint_limit(robot)
    _pause_between_movements(
        pause_between_movements, f"{label}: about to move to rough detection pose"
    )
    robot.move_tool(
        rough_detect_pose,
        speed=DETECTION_MOVE_SPEED,
        acceleration=DETECTION_MOVE_ACCELERATION,
        label=f"{label}: rough detection pose",
    )
    # Let the arm and hand-camera image settle before using the rough estimate.
    # This is part of real execution time, so do not record it as operator pause.
    time.sleep(5.0)
    obj_pose, bounding_box = get_object_pose(robot, max_retries=0)

    _pause_between_movements(
        pause_between_movements, f"{label}: about to move above detected object"
    )
    robot.move_tool(
        list(obj_pose[:2, 3]) + [0.35, 0.0, np.pi, 0.0],
        speed=DETECTION_MOVE_SPEED,
        acceleration=DETECTION_MOVE_ACCELERATION,
        label=f"{label}: refined overhead detection pose",
    )
    obj_pose, bounding_box = get_object_pose(robot, max_retries=0)

    return obj_pose, bounding_box


def detect_after_push(
    robot: PhysicalUR10,
    rough_detect_pose: np.ndarray,
    completed_pushes: int,
    *,
    height: float,
    debug_img_id: str,
):
    """Run the complete two-step detection sequence after every push."""

    del debug_img_id
    push_number = int(completed_pushes)
    if push_number < 1:
        raise ValueError("completed_pushes must be positive")
    return two_step_detection(
        robot,
        rough_detect_pose,
        height=height,
        label=f"post-push {push_number} two-step detection",
    )


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

    adjust_joint_limit(robot)
    _pause_between_movements(pause_between_movements, "About to move to pre-push pose")
    robot.move_tool(
        pre_push_pose_flat,
        to_rotvec=True,
        speed=PUSH_TRANSIT_SPEED,
        acceleration=PUSH_TRANSIT_ACCELERATION,
        label="push: pre-push clearance pose",
    )
    _pause_between_movements(
        pause_between_movements,
        "At pre-push pose; about to move to contact waypoint[0]",
    )
    robot.move_tool(
        ws_path[0],
        to_rotvec=True,
        speed=PUSH_CONTACT_SPEED,
        acceleration=PUSH_CONTACT_ACCELERATION,
        label="push: first contact waypoint",
    )
    _pause_between_movements(
        pause_between_movements,
        "At contact waypoint[0]; about to execute push to waypoint[-1]",
    )
    robot.execute_ee_waypoints(
        ws_path,
        to_rotvec=True,
        label="push: contact trajectory",
    )
    _pause_between_movements(
        pause_between_movements,
        "At push endpoint waypoint[-1]; about to move to post-push pose",
    )
    robot.move_tool(
        post_push_pose_flat,
        to_rotvec=True,
        speed=PUSH_TRANSIT_SPEED,
        acceleration=PUSH_TRANSIT_ACCELERATION,
        label="push: post-push clearance pose",
    )

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
            post_push_pose_flat,
            to_rotvec=True,
            speed=PUSH_TRANSIT_SPEED,
            acceleration=PUSH_TRANSIT_ACCELERATION,
            label="push: adjusted post-push camera pose",
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
) -> OMPLPlanner:
    planner = OMPLPlanner(
        system=system,
        start_state=np.asarray(start_state, dtype=float).reshape(-1)[:3],
        goal_state=np.asarray(goal_state, dtype=float).reshape(-1)[:3],
        planner_method=str(args.planner),
        goal_threshold=float(args.planner_goal_threshold),
        min_max_control_duration=(
            int(args.min_control_steps),
            int(args.max_control_steps),
        ),
        propagation_step_size=float(args.primitive_duration),
        initial_planning_time=float(planning_time),
        pruning_radius=float(args.pruning_radius),
        obstacle_config=None,
    )
    planner.replanning_time = float(args.replanning_time)
    planner.motion_validation_step_size = float(args.primitive_duration)
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
) -> tuple[OMPLPlanner, dict, float]:
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
    pred[2] = wrap_to_pi(float(pred[2]))
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
        _recorded_operator_pause(0.5)


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


def _save_rows(out_dir: Path, rows: list[RealExecutionStep], metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_path = out_dir / "real_execution_steps.csv"
    if rows:
        with steps_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    elif steps_path.exists():
        steps_path.unlink()
    np.savez(
        out_dir / "real_execution_results.npz",
        rows=np.array([asdict(row) for row in rows], dtype=object),
        metadata=np.array(metadata, dtype=object),
    )
    metadata_path = out_dir / "metadata.json"
    temporary_path = metadata_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(metadata_path)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_hash(value) -> str:
    rendered = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _solution_payload(
    solution: dict,
    *,
    planner: str,
    run_number: int,
    seed: int,
    planning_seconds: float,
) -> dict:
    payload = {
        "states": _jsonable(solution.get("states", [])),
        "controls": _jsonable(solution.get("controls", [])),
        "time": _jsonable(solution.get("time", [])),
        "time_steps": _jsonable(solution.get("time_steps", [])),
        "cost": float(solution["cost"]),
        "control_count": int(solution["control_count"]),
    }
    return {
        "schema_version": 1,
        "planner": planner,
        "run_number": int(run_number),
        "seed": int(seed),
        "planning_seconds": float(planning_seconds),
        "solution_hash": _canonical_hash(payload),
        "solution": payload,
    }


def _load_shared_initial_plan(
    path: Path,
    *,
    args: argparse.Namespace,
    measured_start: np.ndarray,
) -> tuple[dict, float, str]:
    if not path.is_file():
        raise RuntimeError(
            "restartReplanning must use the immutable initial solution created "
            f"by the paired AURA invocation; missing {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("planner") != str(args.planner)
        or int(payload.get("run_number", -1)) != int(args.run_number)
        or int(payload.get("seed", -1)) != int(args.seed)
    ):
        raise RuntimeError("shared initial-plan identity does not match this job")
    solution = payload["solution"]
    if _canonical_hash(solution) != payload.get("solution_hash"):
        raise RuntimeError("shared initial-plan hash is invalid")
    planned_start = np.asarray(solution["states"][0], dtype=float)
    start_error = float(
        arrayDistance(
            planned_start, measured_start, system="pushing_object"
        )
    )
    if start_error > float(args.pairing_start_tolerance):
        raise RuntimeError(
            "physical start is not paired closely enough with AURA's initial "
            f"state: distance {start_error:.6g} > "
            f"{float(args.pairing_start_tolerance):.6g}"
        )
    return (
        solution,
        float(payload["planning_seconds"]),
        str(payload["solution_hash"]),
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
    _, ws_path = generate_path_form_params(
        Pose(obj_pose[:3, 3], matrix_to_quat(obj_pose[:3, :3])),
        obj_shape,
        path_control,
        tool_offset=tool_offset,
        total_time=float(
            args.primitive_duration if duration is None else duration
        ),
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
        initial_state = project_se3_to_se2(matrix_to_flat(initial_pose))
        initial_state[2] = wrap_to_pi(float(initial_state[2]))
        config = {
            "start_state": initial_state.tolist(),
            "goal_state": np.asarray(goal_state, dtype=float).reshape(-1)[:3].tolist(),
            "propagation_step_size": float(args.primitive_duration),
            "min_control_duration": int(args.min_control_steps),
            "max_control_duration": int(args.max_control_steps),
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
        self.operator_pause_seconds = 0.0
        _set_pause_object_state(self.current_state)

    def reset(self):
        super().reset()
        self.current_pose = to_se3_matrix(self.current_state)
        _set_pause_object_state(self.current_state)
        return self.get_state()

    def get_state(self):
        return np.asarray(self.current_state, dtype=float).copy()

    def set_state(self, pose):
        pose_np = np.asarray(pose, dtype=float).reshape(-1)[:3]
        pose_np[2] = wrap_to_pi(float(pose_np[2]))
        self.current_state = pose_np.copy()
        self.start_state = pose_np.copy()
        self.current_pose = to_se3_matrix(pose_np)
        _set_pause_object_state(self.current_state)
        return self.get_state()

    def execute_segment(self, control, duration):
        self.running = True
        pause_before = _operator_pause_total()
        control_np = np.asarray(control, dtype=float).reshape(-1)[:3]
        duration = float(duration)
        primitive_count = duration_seconds_to_steps(
            duration,
            float(self.args.primitive_duration),
            min_steps=int(self.args.min_control_steps),
            max_steps=int(self.args.max_control_steps),
        )
        for primitive_index in range(primitive_count):
            self.executed_segments += 1
            _execute_real_push_from_control(
                self.robot,
                self.current_pose,
                self.obj_shape,
                self.tool_offset,
                control_np,
                self.args,
                duration=float(self.args.primitive_duration),
            )
            _pause_between_movements(
                bool(self.args.pause_between_movements),
                "Post-push motion complete; about to run two-step object detection",
            )
            time.sleep(float(self.args.detection_wait))
            self.current_pose, _ = two_step_detection(
                self.robot,
                self.rough_detect_pose,
                height=float(self.args.post_detect_height),
                pause_between_movements=bool(self.args.pause_between_movements),
                label=(
                    f"{self.method_label} macro primitive "
                    f"{primitive_index + 1}/{primitive_count} post-push detection"
                ),
            )
            measured = project_se3_to_se2(matrix_to_flat(self.current_pose))
            measured[2] = wrap_to_pi(float(measured[2]))
            self.current_state = measured.copy()
            self._record_primitive_state()
            _set_pause_object_state(self.current_state)
            goal_distance = float(
                arrayDistance(
                    self.current_state, self.goal_state, system=self.system_name
                )
            )
            print(
                f"[REAL] {self.method_label.upper()} step "
                f"{self.executed_segments}: pose={_fmt(self.current_state)}; "
                f"goal distance={goal_distance:.6f}"
            )
        self.operator_pause_seconds += max(
            0.0, _operator_pause_total() - pause_before
        )
        return self.get_state()


def _plan_states_from_update(plan: dict) -> list[np.ndarray]:
    states = plan.get("states") or plan.get("display_states") or []
    return [np.asarray(s, dtype=float).reshape(-1)[:3].copy() for s in states]


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
        if not states:
            return
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
        "control_duration_seconds": float(args.primitive_duration),
        "min_control_duration": int(args.min_control_steps),
        "max_control_duration": int(args.max_control_steps),
        "propagation_step_size": float(args.primitive_duration),
        "pruning_radius": float(args.pruning_radius),
        "control_bounds": control_bounds,
        "state_bounds": state_bounds,
        "replanningMaxDistance": float(args.replanning_max_distance),
    }


def run_real_execution(args: argparse.Namespace) -> dict:
    method = str(args.method).lower()
    if method == "restartreplanning":
        method = "replanning"
    np.random.seed(int(args.seed))
    try:
        from ompl import util as ou

        ou.RNG.setSeed(int(args.seed))
    except Exception:
        pass
    system = get_system("pushing_object")
    control_bounds = [(0.0, 0.75), (-0.4, 0.4), (0.0, float(args.max_push_distance))]
    _apply_control_bounds(system, control_bounds)

    goal = np.asarray(args.goal, dtype=float).reshape(3)
    task_limit = float(args.task_time_limit)
    rough_detect_pose = np.asarray(args.rough_detect_pose, dtype=float)
    obj_shape = get_obj_shape(f"simulation/assets/{args.obj_name}/textured.obj")
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

    setup_started = time.monotonic()
    print("[REAL] Connecting to PhysicalUR10...")
    robot = PhysicalUR10()
    _pause_between_movements(
        bool(args.pause_between_movements),
        "Connected to robot; about to move to home joint pose",
    )
    robot.move_joint(
        np.asarray(args.robot_home, dtype=float),
        speed=ROBOT_HOME_SPEED,
        acceleration=ROBOT_HOME_ACCELERATION,
        label="real execution setup: robot home",
    )

    print("[REAL] Detecting current object pose; this is the execution start.")
    obj_pose, _ = two_step_detection(
        robot,
        rough_detect_pose,
        height=float(args.detect_height),
        pause_between_movements=bool(args.pause_between_movements),
        label=f"{method} initial two-step detection",
    )
    measured_start = project_se3_to_se2(matrix_to_flat(obj_pose))
    measured_start[2] = wrap_to_pi(float(measured_start[2]))
    setup_reset_operator_pause_seconds = time.monotonic() - setup_started
    method_wall_started = time.monotonic()
    method_pause_baseline = _operator_pause_total()
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
        "primitive_duration": float(args.primitive_duration),
        "min_control_steps": int(args.min_control_steps),
        "max_control_steps": int(args.max_control_steps),
        "planner": str(args.planner),
        "run_number": int(args.run_number),
        "seed": int(args.seed),
        "manifest": args.manifest,
        "goal_threshold": float(args.goal_threshold),
        "actual_goal_threshold": float(args.goal_threshold),
        "planner_goal_threshold": float(args.planner_goal_threshold),
        "task_time_limit_seconds": task_limit,
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
    initial_goal_distance = float(
        arrayDistance(measured_start, goal, system=system.name)
    )
    print(
        f"[REAL] {method.upper()} step 0: pose={_fmt(measured_start)}; "
        f"goal distance={initial_goal_distance:.6f}"
    )
    if args.require_confirm:
        _recorded_operator_pause(0.5)

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
    total_replanning = 0
    tracking_error_mean = 0.0
    tracking_error_max = 0.0
    all_tracking_errors: list[float] = []
    all_controls: list[np.ndarray] = []
    all_duration_steps: list[int] = []
    all_duration_seconds: list[float] = []
    initial_planning_seconds = 0.0
    nominal_execution_seconds = 0.0
    actual_execution_seconds = 0.0
    online_replanning_seconds = 0.0
    optimizer_seconds = 0.0
    blocking_replanning_seconds = 0.0
    compute_overrun_seconds = 0.0
    total_cost = 0.0
    initial_plan_hash = None
    pending_initial_plan_payload = None
    shared_initial_path = Path(args.shared_initial_plan).expanduser().resolve()

    def task_time_so_far() -> float:
        return float(
            initial_planning_seconds
            + nominal_execution_seconds
            + compute_overrun_seconds
            + blocking_replanning_seconds
        )

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
            if goal_distance <= float(args.goal_threshold):
                status = (
                    "success" if task_time_so_far() <= task_limit else "timeout"
                )
                break
            if task_time_so_far() >= task_limit:
                status = "timeout"
                break
            if args.max_steps is not None and total_controls >= int(args.max_steps):
                status = "max_steps_reached"
                break

            requested_budget = float(
                args.planning_time if episode == 0 else args.replanning_time
            )
            budget = min(requested_budget, task_limit - task_time_so_far())
            if budget <= 0.0:
                status = "timeout"
                break
            planner, _solution, planning_wall = _fresh_plan(
                system,
                current_state,
                goal,
                args,
                budget,
            )
            planner.opt_model = opt_model
            if episode == 0:
                initial_plan_payload = _solution_payload(
                    _solution,
                    planner=str(args.planner),
                    run_number=int(args.run_number),
                    seed=int(args.seed),
                    planning_seconds=float(planning_wall),
                )
                pending_initial_plan_payload = initial_plan_payload
                initial_plan_hash = initial_plan_payload["solution_hash"]
                initial_planning_seconds = float(planning_wall)
            else:
                blocking_replanning_seconds += float(planning_wall)
                total_replanning += 1
            if task_time_so_far() >= task_limit:
                status = "timeout"
                break
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
                max_nominal_execution_seconds=max(
                    0.0, task_limit - task_time_so_far()
                ),
            )
            total_controls += int(result.num_controls)
            total_replanning += int(result.num_replanning)
            all_controls.extend(
                np.asarray(value, dtype=float)
                for value in result.controls_trajectory
            )
            all_duration_steps.extend(
                int(value)
                for value in result.control_duration_steps_trajectory
            )
            all_duration_seconds.extend(
                float(value)
                for value in result.control_duration_seconds_trajectory
            )
            nominal_execution_seconds += float(
                result.nominal_execution_seconds
            )
            actual_execution_seconds += float(
                result.actual_execution_seconds
            )
            online_replanning_seconds += float(
                result.online_replanning_seconds
            )
            optimizer_seconds += float(result.optimizer_seconds)
            compute_overrun_seconds += float(
                result.compute_overrun_seconds
            )
            total_cost += float(result.cost)
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
            if (
                final_goal_distance <= float(args.goal_threshold)
                and task_time_so_far() <= task_limit
            ):
                status = "success"
                break
            if (
                task_time_so_far() >= task_limit
                or result.failure_reason == "task_time_limit_reached"
            ):
                status = "timeout"
                break
            if result.status != "success":
                status = f"aura_{result.status}:{result.failure_reason}"
                break
            if result.num_controls <= 0:
                status = "aura_no_progress"
                break
            episode += 1
    else:
        (
            shared_initial_solution,
            initial_planning_seconds,
            initial_plan_hash,
        ) = _load_shared_initial_plan(
            shared_initial_path,
            args=args,
            measured_start=measured_start,
        )
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
            str(args.planner),
            config,
            simulator_mode="real",
            max_steps=args.max_steps,
            plan_update_callback=replanning_callback,
            system_override=system,
            simulator_override=real_sim,
            initial_solution=shared_initial_solution,
            task_time_budget_seconds=max(
                0.0, task_limit - initial_planning_seconds
            ),
        )
        result = runner.run()
        total_controls = int(result.num_controls)
        total_replanning = int(result.num_replanning)
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
        all_tracking_errors = [
            float(value) for value in result.tracking_error_list
        ]
        all_controls = [
            np.asarray(value, dtype=float)
            for value in result.controls_trajectory
        ]
        all_duration_steps = [
            int(value)
            for value in result.control_duration_steps_trajectory
        ]
        all_duration_seconds = [
            float(value)
            for value in result.control_duration_seconds_trajectory
        ]
        nominal_execution_seconds = float(
            result.nominal_execution_seconds
        )
        actual_execution_seconds = float(result.actual_execution_seconds)
        blocking_replanning_seconds = float(
            result.blocking_replanning_seconds
        )
        compute_overrun_seconds = float(result.compute_overrun_seconds)
        total_cost = float(result.cost)
        if (
            final_goal_distance <= float(args.goal_threshold)
            and task_time_so_far() <= task_limit
        ):
            status = "success"
        elif (
            task_time_so_far() >= task_limit
            or result.failure_reason == "task_time_limit_reached"
        ):
            status = "timeout"
        else:
            status = "replanning_stopped_outside_goal"

    timed_operator_pause_seconds = max(
        0.0, _operator_pause_total() - method_pause_baseline
    )
    execution_operator_pause_seconds = min(
        float(real_sim.operator_pause_seconds),
        float(actual_execution_seconds),
    )
    actual_execution_seconds = max(
        0.0, actual_execution_seconds - execution_operator_pause_seconds
    )
    compute_overrun_seconds = max(
        0.0, compute_overrun_seconds - timed_operator_pause_seconds
    )
    setup_reset_operator_pause_seconds += timed_operator_pause_seconds

    elapsed_task_time = task_time_so_far()
    if status != "success" and elapsed_task_time >= task_limit:
        status = "timeout"
    task_time_seconds = recorded_task_time(status, elapsed_task_time, task_limit)
    final_goal_distance = float(arrayDistance(final_state, goal, system=system.name))
    print("\n[REAL] Done.")
    print(f"[REAL] status: {status}")
    print("[REAL] final state: ", _fmt(final_state))
    print(f"[REAL] final goal distance: {final_goal_distance:.6f}")
    print(f"[REAL] task time: {task_time_seconds:.3f}s")
    print(f"[REAL] results saved in {out_dir}")

    if pending_initial_plan_payload is not None:
        _atomic_json(shared_initial_path, pending_initial_plan_payload)

    metadata.update(
        {
            "status": status,
            "final_state": final_state.tolist(),
            "final_goal_distance": final_goal_distance,
            "planned_final_state": last_planned_final.tolist(),
            "executed_controls": int(total_controls),
            "tracking_error_mean": float(tracking_error_mean),
            "tracking_error_max": float(tracking_error_max),
            "completed": True,
            "shared_initial_plan": str(shared_initial_path),
            "initial_plan_hash": initial_plan_hash,
            "initial_planning_seconds": initial_planning_seconds,
            "task_time_seconds": task_time_seconds,
        }
    )
    _save_rows(out_dir, rows, metadata)
    method_name = "aura" if method == "aura" else "restartReplanning"
    panel_config = {}
    config_hash = None
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if config_path.suffix.lower() == ".json":
            panel_config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            panel_config = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            )
        config_hash = _canonical_hash(panel_config)
    result_row = {
        "schema_version": 1,
        "panel_id": "pushing_real",
        "system": "pushing_object",
        "environment": "real",
        "planner": str(args.planner),
        "method": method_name,
        "run_number": int(args.run_number),
        "seed": int(args.seed),
        "rng_streams": {"real_trial": int(args.seed)},
        "method_order": ["aura", "restartReplanning"],
        "config_hash": config_hash,
        "initial_planning_seconds": float(initial_planning_seconds),
        "initial_plan_hash": initial_plan_hash,
        "initial_plan": (
            json.loads(shared_initial_path.read_text(encoding="utf-8"))[
                "solution"
            ]
            if shared_initial_path.is_file()
            else None
        ),
        "status": status if status in {"success", "timeout"} else "failure",
        "failure_reason": (
            ""
            if status == "success"
            else (
                "task_time_limit_reached" if status == "timeout" else str(status)
            )
        ),
        "nominal_execution_seconds": float(nominal_execution_seconds),
        "actual_execution_seconds": float(actual_execution_seconds),
        "online_replanning_seconds": float(online_replanning_seconds),
        "optimizer_seconds": float(optimizer_seconds),
        "blocking_replanning_seconds": float(
            blocking_replanning_seconds
        ),
        "compute_overrun_seconds": float(compute_overrun_seconds),
        "task_time_seconds": float(task_time_seconds),
        "raw_process_wall_seconds": float(
            time.monotonic() - method_wall_started
        ),
        "setup_reset_operator_pause_seconds": float(
            setup_reset_operator_pause_seconds
        ),
        "timed_operator_pause_seconds": float(timed_operator_pause_seconds),
        "num_controls": int(total_controls),
        "num_replanning": int(total_replanning),
        "cost": float(total_cost),
        "tracking_error_mean": float(tracking_error_mean),
        "tracking_error_list": all_tracking_errors,
        "goal_distance": float(final_goal_distance),
        "final_state": final_state.tolist(),
        "planned_final_state": last_planned_final.tolist(),
        "controls": _jsonable(all_controls),
        "control_duration_steps": all_duration_steps,
        "control_duration_seconds": all_duration_seconds,
        "primitive_states": _jsonable(real_sim.primitive_states),
        "hardware_artifact_directory": str(out_dir.resolve()),
        "duration_audit_initial_tree": {
            "range_steps": [
                int(args.min_control_steps),
                int(args.max_control_steps),
            ],
            "propagation_step_size_seconds": float(
                args.primitive_duration
            ),
            "duration_step_histogram": {},
        },
    }
    campaign_root = shared_initial_path.parents[4]
    paired_csv = result_path(
        campaign_root,
        "pushing_real",
        str(args.planner),
        int(args.run_number),
    )
    upsert_result(paired_csv, result_row)
    metadata["fig7_result_row"] = str(paired_csv)
    return metadata


def parse_args() -> argparse.Namespace:
    limits = REAL_EXECUTION_LIMITS
    parser = argparse.ArgumentParser(description="Real UR10 AURA/Replanning execution.")
    parser.add_argument(
        "--method",
        type=lambda value: str(value).lower(),
        choices=["aura", "replanning", "restartreplanning"],
        required=True,
    )
    parser.add_argument(
        "--planner",
        type=lambda value: str(value).lower(),
        choices=["aorrt", "sststar", "aoest"],
        required=True,
    )
    parser.add_argument("--run-number", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Frozen experiment manifest JSON used to authorize this run.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Frozen panel configuration YAML/JSON (recorded in output metadata).",
    )
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
    parser.add_argument("--goal", type=float, nargs=3, default=[0.35, -0.5, 0.0])
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
    parser.add_argument(
        "--task-time-limit",
        type=float,
        default=300.0,
        help="Stop outside the goal at this task time and record the exact limit.",
    )
    parser.add_argument(
        "--primitive-duration",
        "--duration",
        dest="primitive_duration",
        type=float,
        default=1.0,
        help="Physical duration h of one complete push/reposition primitive.",
    )
    parser.add_argument("--min-control-steps", type=int, default=1)
    parser.add_argument("--max-control-steps", type=int, default=5)
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
        "--shared-initial-plan",
        required=True,
        help=(
            "Immutable pair initial-plan JSON. AURA must run first and creates "
            "this file; restartReplanning consumes it."
        ),
    )
    parser.add_argument(
        "--pairing-start-tolerance",
        type=float,
        default=0.03,
        help="Maximum measured start-state distance from the paired AURA start.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the exact run without importing/commanding robot state.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print whether this method/trial already has a completed metadata file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Return without robot access if this exact output is already complete.",
    )
    parser.add_argument(
        "--pause-between-movements",
        dest="pause_between_movements",
        action="store_true",
        default=False,
        help="Add a short inspection pause before each real robot motion phase.",
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


def _file_fingerprint(path_value: str | None) -> dict | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def dry_run_spec(args: argparse.Namespace) -> dict:
    primitive_sequence = [
        {
            "primitive_index": index + 1,
            "control": "<same planner edge control>",
            "duration_seconds": float(args.primitive_duration),
            "reposition_and_redetect_after": True,
            "replan_inside_macro_edge": False,
        }
        for index in range(int(args.max_control_steps))
    ]
    return {
        "dry_run": True,
        "robot_connection_attempted": False,
        "method": str(args.method),
        "planner": str(args.planner),
        "run_number": int(args.run_number),
        "seed": int(args.seed),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "shared_initial_plan": str(
            Path(args.shared_initial_plan).expanduser().resolve()
        ),
        "pairing_protocol": (
            "AURA creates or replaces the paired plan after its rerun; "
            "restartReplanning reuses it after the operator resets the object "
            "within tolerance"
        ),
        "goal_state": [float(value) for value in args.goal],
        "goal_threshold": float(args.goal_threshold),
        "planner_goal_threshold": float(args.planner_goal_threshold),
        "task_time_limit_seconds": float(args.task_time_limit),
        "existing_output_will_be_replaced": (
            Path(args.output_dir).expanduser() / "metadata.json"
        ).is_file(),
        "replacement_occurs_after_rerun_returns_a_result": True,
        "primitive_duration_seconds": float(args.primitive_duration),
        "min_control_steps": int(args.min_control_steps),
        "max_control_steps": int(args.max_control_steps),
        "example_max_duration_macro_edge": primitive_sequence,
        "manifest": _file_fingerprint(args.manifest),
        "config": _file_fingerprint(args.config),
    }


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
    if args.task_time_limit <= 0:
        raise ValueError("--task-time-limit must be positive")
    if args.run_number < 1:
        raise ValueError("--run-number must be positive")
    if args.min_control_steps < 1:
        raise ValueError("--min-control-steps must be at least 1")
    if args.max_control_steps <= args.min_control_steps:
        raise ValueError(
            "production real-world configuration requires "
            "--min-control-steps < --max-control-steps"
        )
    if args.primitive_duration <= 0.0:
        raise ValueError("--primitive-duration must be positive")
    if args.pairing_start_tolerance <= 0.0:
        raise ValueError("--pairing-start-tolerance must be positive")
    output_metadata = Path(args.output_dir).expanduser() / "metadata.json"
    if args.status:
        completed = False
        metadata = None
        if output_metadata.is_file():
            metadata = json.loads(output_metadata.read_text(encoding="utf-8"))
            completed = bool(metadata.get("completed", False))
        print(
            json.dumps(
                {
                    "output": str(output_metadata.resolve()),
                    "exists": output_metadata.is_file(),
                    "completed": completed,
                    "metadata": metadata,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.resume and output_metadata.is_file():
        metadata = json.loads(output_metadata.read_text(encoding="utf-8"))
        if bool(metadata.get("completed", False)):
            print(
                f"[REAL] completed output exists; resume skipped: "
                f"{output_metadata.resolve()}"
            )
            return
    if args.dry_run:
        print(json.dumps(dry_run_spec(args), indent=2, sort_keys=True))
        return
    run_real_execution(args)


if __name__ == "__main__":
    main()
