#!/usr/bin/env python3
"""Real-robot pushing tracking-error experiment.

This script mirrors the pushing part of ``experiments/error_experiment.py`` on
the UR10.
Unlike simulation, the object is not reset between rollouts. The experiment
therefore starts from the currently detected object pose and runs one physical
sequence against the nominal trajectory computed from that measured start.

Typical usage:
  /usr/bin/python3 experiments/real_error_experiment.py --num-controls 10 --policy both
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.spatial.transform import Rotation as R

from AURA import AURA
from geometry.object_model import get_obj_shape
from geometry.pose import Pose, flat_to_matrix, matrix_to_quat
from geometry.random_push import generate_path_form_params
from optimization import runOptimizer
from pushing_dynamics import get_pushing_model
from real_world.physical_robot import PhysicalUR10
from systems import get_system
from train_model import load_opt_model_2
from utils.utils import arrayDistance


@dataclass
class RealStepResult:
    step: int
    policy: str
    measured_start_x: float
    measured_start_y: float
    measured_start_theta: float
    target_x: float
    target_y: float
    target_theta: float
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
    measured_tracking_error: float


class _PickerSimulatorStub:
    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})


class _PickerPlannerStub:
    obstacle_config = None

    def __init__(self, motion_validation_step_size: float):
        self.motion_validation_step_size = float(motion_validation_step_size)


def _wrap_angle(theta: float) -> float:
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


def normalized_push_to_path_push(control: np.ndarray) -> np.ndarray:
    """Convert learned-model push face id to the real path generator's radians."""
    control = np.asarray(control, dtype=float).reshape(-1).copy()
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
    control[0] = control[0] * 2.0 * np.pi
    return control


def matrix_to_flat(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return np.concatenate([matrix[:3, 3], matrix_to_quat(matrix[:3, :3])])


def project_se3_pose(pose_flat: np.ndarray | list[float], axis=(0, 1, 0)) -> np.ndarray:
    poses = np.asarray(pose_flat, dtype=float)
    single = poses.ndim == 1
    poses = poses.reshape(-1, 7)
    xy = poses[:, :2]
    rotations = R.from_quat(poses[:, [4, 5, 6, 3]])
    axis0 = np.tile(np.asarray(axis, dtype=float), (poses.shape[0], 1))
    axis1 = rotations.apply(axis0)
    dots = np.sum(axis0[:, :2] * axis1[:, :2], axis=1)
    cross_zs = axis0[:, 0] * axis1[:, 1] - axis0[:, 1] * axis1[:, 0]
    yaw = np.arctan2(cross_zs, dots)[:, None]
    out = np.concatenate([xy, yaw], axis=1)
    return out[0] if single else out


def to_se3_matrix(se2_pose: np.ndarray | list[float]) -> np.ndarray:
    se2_pose = np.asarray(se2_pose, dtype=float).reshape(-1)
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_euler("z", se2_pose[2], degrees=False).as_matrix()
    matrix[:2, 3] = se2_pose[:2]
    return matrix


def object_in_bounds(
    obj_pos: np.ndarray, center: np.ndarray, bounds: np.ndarray
) -> bool:
    return bool(
        abs(obj_pos[0] - center[0]) < bounds[0]
        and abs(obj_pos[1] - center[1]) < bounds[1]
    )


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


def get_object_pose(
    robot: PhysicalUR10,
    max_retries: int = 5,
    rough_detect_pose: np.ndarray | None = None,
    debug_img_id: str | int | None = None,
    debug_dir: str = "debug",
) -> tuple[np.ndarray, object]:
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

    obj_pose = pose["pose"]
    bounding_box = pose.get("bounding_box")
    img = pose.get("result_image")
    if img is not None:
        os.makedirs(debug_dir, exist_ok=True)
        img.save(
            os.path.join(debug_dir, f"{debug_img_id or 'real'}_{time.time():.3f}.jpg")
        )

    tcp_ee_pose = np.eye(4)
    tcp_ee_pose[2, 3] = 0.260
    obj_pose_camera = obj_pose.copy()
    obj_pose_base = tcp_base_pose @ np.linalg.inv(tcp_ee_pose) @ obj_pose_camera
    obj_se2 = project_se3_pose(matrix_to_flat(obj_pose_base))
    obj_pose_base_se2 = to_se3_matrix(obj_se2)
    print(
        "[REAL] Object SE2 in robot/global base frame:",
        np.array2string(obj_se2, precision=5, suppress_small=True),
    )
    return obj_pose_base_se2, bounding_box


def reset_object(
    robot: PhysicalUR10,
    obj_pos_xy: np.ndarray,
    push_z: float,
    center: np.ndarray,
    *,
    pause_between_movements: bool = False,
) -> None:
    """Reset object into the safe workspace using the confirmed collection-script push."""
    direction = np.asarray(center, dtype=float) - np.asarray(obj_pos_xy, dtype=float)
    dist = float(np.linalg.norm(direction))
    if dist <= 1e-9:
        return
    dir_unit = direction / dist
    angle = float(np.arctan2(direction[1], direction[0]))
    r = R.from_euler(
        "xyz", [-np.pi, 0.0, angle - np.pi / 2.0], degrees=False
    ).as_rotvec()

    pre_push_dist = 0.15
    pre_push_coord = np.asarray(obj_pos_xy, dtype=float) - dir_unit * pre_push_dist
    post_push_coord = np.asarray(obj_pos_xy, dtype=float) + dir_unit * dist
    x_d, y_d = pre_push_coord
    x_f, y_f = post_push_coord

    print(f"\n[REAL RESET] Push the object from {(x_d, y_d)} to {(x_f, y_f)}")
    adjust_joint_limit(robot)
    _pause_between_movements(
        pause_between_movements, "RESET: about to move above pre-push pose"
    )
    robot.move_tool([x_d, y_d, 0.10, *r], speed=1.0, acceleration=2.0)
    _pause_between_movements(
        pause_between_movements, "RESET: about to move down to push height"
    )
    robot.move_tool([x_d, y_d, push_z, *r])
    _pause_between_movements(
        pause_between_movements, "RESET: about to push object back toward center"
    )
    robot.move_tool([x_f, y_f, push_z, *r])
    _pause_between_movements(
        pause_between_movements, "RESET: about to lift to camera height"
    )
    robot.move_tool([x_f, y_f, 0.35, *r], speed=1.0, acceleration=2.0)


def two_step_detect(
    robot: PhysicalUR10,
    rough_detect_pose: np.ndarray,
    height: float = 0.12,
    *,
    pause_between_movements: bool = False,
    label: str = "object detection",
):
    adjust_joint_limit(robot)
    _pause_between_movements(
        pause_between_movements, f"{label}: about to move to rough detection pose"
    )
    robot.move_tool(rough_detect_pose)
    obj_pose, bounding_box = get_object_pose(
        robot, max_retries=0, rough_detect_pose=None
    )
    print(
        f"[REAL] {label}: rough-detected SE2",
        np.array2string(
            project_se3_pose(matrix_to_flat(obj_pose)), precision=5, suppress_small=True
        ),
    )
    _pause_between_movements(
        pause_between_movements, f"{label}: about to move above detected object"
    )
    robot.move_tool(list(obj_pose[:2, 3]) + [0.35, 0.0, np.pi, 0.0])
    obj_pose, bounding_box = get_object_pose(
        robot, max_retries=0, rough_detect_pose=None
    )
    print(
        f"[REAL] {label}: refined SE2",
        np.array2string(
            project_se3_pose(matrix_to_flat(obj_pose)), precision=5, suppress_small=True
        ),
    )
    _pause_between_movements(
        pause_between_movements, f"{label}: about to move to inspection height"
    )
    robot.move_tool(list(obj_pose[:2, 3]) + [height, 0.0, np.pi, 0.0])
    return obj_pose, bounding_box


def _pause_between_movements(enabled: bool, label: str) -> None:
    if enabled:
        print(f"[PAUSE] {label}. Sleeping 1.0s.")
        time.sleep(1.0)


def execute_push(
    robot: PhysicalUR10,
    ws_path: np.ndarray,
    push_param: np.ndarray,
    *,
    pause_between_movements: bool = False,
) -> None:
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
    print(
        "  pre-push:   ",
        np.array2string(pre_push_pose_flat, precision=5, suppress_small=True),
    )
    print(
        "  waypoint[0]:", np.array2string(ws_path[0], precision=5, suppress_small=True)
    )
    print(
        "  waypoint[-1]:",
        np.array2string(ws_path[-1], precision=5, suppress_small=True),
    )
    print(
        "  post-push:  ",
        np.array2string(post_push_pose_flat, precision=5, suppress_small=True),
    )

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

    print(
        "Push Param:",
        np.array2string(np.asarray(push_param), precision=5, suppress_small=True),
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
            post_push_pose_flat, to_rotvec=True, speed=1.0, acceleration=2.0
        )


def _make_aura_picker(system, duration: float, center: np.ndarray, bounds: np.ndarray):
    aura = AURA.__new__(AURA)
    aura.system = system
    aura.simulator = _PickerSimulatorStub(
        {
            "state_bounds": [
                (float(center[0] - bounds[0]), float(center[0] + bounds[0])),
                (float(center[1] - bounds[1]), float(center[1] + bounds[1])),
            ],
            "execution_safety_radius": 0.0,
        }
    )
    aura.planner = _PickerPlannerStub(max(0.02, min(0.05, float(duration) / 20.0)))
    aura.propagation_step_size = float(duration)
    aura.last_control_decision = {}
    return aura


def sample_real_push_sequence(
    system,
    start_state: np.ndarray,
    num_controls: int,
    rng: np.random.Generator,
    duration: float,
    center: np.ndarray,
    bounds: np.ndarray,
) -> list[np.ndarray]:
    controls: list[np.ndarray] = []
    current = np.asarray(start_state, dtype=float).copy()
    faces = np.array([0.0, 0.25, 0.5, 0.75], dtype=float)
    previous_face: float | None = None

    for _ in range(num_controls):
        valid: list[tuple[float, np.ndarray, np.ndarray]] = []
        for _candidate_idx in range(200):
            face = float(rng.choice(faces))
            control = np.asarray(
                [
                    face,
                    rng.uniform(-0.32, 0.32),
                    rng.uniform(0.035, 0.080),
                ],
                dtype=float,
            )
            next_state = np.asarray(
                system.propagate(current, control, duration), dtype=float
            )
            if not object_in_bounds(next_state[:2], center, bounds):
                continue
            displacement = float(np.linalg.norm(next_state[:2] - current[:2]))
            repeat_penalty = (
                0.02
                if previous_face is not None and abs(face - previous_face) < 1e-9
                else 0.0
            )
            center_penalty = 0.15 * float(np.linalg.norm(next_state[:2] - center))
            score = (
                displacement
                - center_penalty
                - repeat_penalty
                + float(rng.normal(0.0, 0.005))
            )
            valid.append((score, control, next_state))

        if not valid:
            raise RuntimeError(
                "Could not sample a valid bounded push. Move the object closer to the workspace center."
            )
        valid.sort(key=lambda item: item[0], reverse=True)
        top = valid[: min(8, len(valid))]
        _, control, next_state = top[int(rng.integers(0, len(top)))]
        controls.append(control)
        current = next_state.copy()
        current[2] = _wrap_angle(current[2])
        previous_face = float(control[0])

    return controls


def optimize_control(
    system,
    aura_picker,
    opt_model,
    current_state: np.ndarray,
    target_state: np.ndarray,
    original_control: np.ndarray,
    duration: float,
    pos_std: float,
    rot_std: float,
    num_states: int,
) -> tuple[np.ndarray, dict | None]:
    result = runOptimizer(
        system=system.name,
        nextState=current_state,
        childrenStatesArray=[target_state],
        childrenControlsArray=[np.asarray(original_control, dtype=float)],
        optModel=opt_model,
        numStates=int(num_states),
        posSTD=float(pos_std),
        rotSTD=float(rot_std),
        velSTD=0.003,
        originalControl=np.asarray(original_control, dtype=float),
        controlDuration=float(duration),
    )
    best_control = aura_picker.pick_next_control(
        system=system,
        optimization_result=result,
        current_state=current_state,
        next_state=target_state,
        children_states=[target_state],
        children_controls=[np.asarray(original_control, dtype=float)],
        control_duration=float(duration),
        fallback_control=np.asarray(original_control, dtype=float),
    )
    return np.asarray(best_control, dtype=float), result


def save_results(out_dir: Path, rows: list[RealStepResult], metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out_dir / "real_error_experiment_steps.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    np.savez(
        out_dir / "real_error_experiment_results.npz",
        rows=np.array([asdict(row) for row in rows], dtype=object),
        metadata=np.array(metadata, dtype=object),
    )


def run_real_error_experiment(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    system = get_system("pushing_object")
    obj_shape = get_obj_shape(f"assets/{args.obj_name}/textured.obj")
    opt_model = load_opt_model_2(
        get_pushing_model(system.object_shape),
        lr=float(args.optimizer_learning_rate),
        epochs=int(args.optimizer_epochs),
    )

    center = np.asarray(args.center, dtype=float)
    bounds = np.asarray(args.bounds, dtype=float)
    rough_detect_pose = np.asarray(args.rough_detect_pose, dtype=float)
    tool_offset = Pose([0.0, 0.0, -float(args.push_offset)], [1.0, 0.0, 0.0, 0.0])
    out_dir = Path(args.output_dir)
    rows: list[RealStepResult] = []

    print("[REAL] Connecting to PhysicalUR10...")
    robot = PhysicalUR10()
    robot_home = np.asarray(args.robot_home, dtype=float)
    robot.move_joint(robot_home)

    print(
        "[REAL] Detecting current object pose. This measured pose is the experiment start."
    )
    obj_pose, _ = two_step_detect(
        robot,
        rough_detect_pose,
        height=0.15,
        pause_between_movements=bool(args.pause_between_movements),
        label=f"{args.policy} initial two-step detection",
    )
    start_state = project_se3_pose(matrix_to_flat(obj_pose))
    print(
        "[REAL] Latest object state before planning/execution:",
        np.array2string(start_state, precision=5, suppress_small=True),
    )
    while not object_in_bounds(start_state[:2], center, bounds):
        print(
            "[REAL] Initial object pose is outside bounds; using confirmed reset push.",
            f"pose={np.array2string(start_state, precision=5, suppress_small=True)}",
        )
        reset_object(
            robot,
            start_state[:2],
            max(
                float(args.reset_push_height), float(obj_pose[2, 3] + args.push_offset)
            ),
            center,
            pause_between_movements=bool(args.pause_between_movements),
        )
        time.sleep(float(args.detection_wait))
        obj_pose, _ = two_step_detect(
            robot,
            rough_detect_pose,
            height=0.35,
            pause_between_movements=bool(args.pause_between_movements),
            label=f"{args.policy} initial reset verification",
        )
        start_state = project_se3_pose(matrix_to_flat(obj_pose))
        print(
            "[REAL] Reset object state before planning/execution:",
            np.array2string(start_state, precision=5, suppress_small=True),
        )

    control_sequence = getattr(args, "control_sequence", None)
    if control_sequence is None:
        controls = sample_real_push_sequence(
            system,
            start_state,
            int(args.num_controls),
            rng,
            float(args.duration),
            center,
            bounds,
        )
        print("[REAL] Sampled a new control sequence for this rollout.")
    else:
        controls = [np.asarray(control, dtype=float).copy() for control in control_sequence]
        print("[REAL] Reusing provided control sequence for this rollout.")

    nominal = [start_state.copy()]
    current_nominal = start_state.copy()
    for control in controls:
        current_nominal = np.asarray(
            system.propagate(current_nominal, control, args.duration), dtype=float
        )
        current_nominal[2] = _wrap_angle(current_nominal[2])
        nominal.append(current_nominal.copy())

    print(
        "[REAL] Initial state:",
        np.array2string(start_state, precision=5, suppress_small=True),
    )
    print("[REAL] Planned controls:")
    for i, control in enumerate(controls):
        print(
            f"  {i:02d}: {np.array2string(control, precision=5, suppress_small=True)}"
        )
    if args.require_confirm:
        print("[REAL] Starting real robot execution in 1.0s. Press Ctrl+C to abort.")
        time.sleep(1.0)

    aura_picker = _make_aura_picker(system, args.duration, center, bounds)
    metadata = {
        "obj_name": args.obj_name,
        "num_controls": int(args.num_controls),
        "duration": float(args.duration),
        "policy": args.policy,
        "seed": int(args.seed),
        "start_state": start_state.tolist(),
        "center": center.tolist(),
        "bounds": bounds.tolist(),
    }

    for step_idx, original_control in enumerate(controls):
        print(f"\n[REAL] Step {step_idx + 1}/{len(controls)}")
        while not object_in_bounds(obj_pose[:2, 3], center, bounds):
            print(
                "[REAL] Object is outside bounds before push; using confirmed reset push.",
                "pose=",
                np.array2string(
                    project_se3_pose(matrix_to_flat(obj_pose)),
                    precision=5,
                    suppress_small=True,
                ),
            )
            reset_object(
                robot,
                obj_pose[:2, 3],
                max(
                    float(args.reset_push_height),
                    float(obj_pose[2, 3] + args.push_offset),
                ),
                center,
                pause_between_movements=bool(args.pause_between_movements),
            )
            time.sleep(float(args.detection_wait))
            obj_pose, _ = two_step_detect(
                robot,
                rough_detect_pose,
                height=0.35,
                pause_between_movements=bool(args.pause_between_movements),
                label=f"{args.policy} step {step_idx + 1} reset verification",
            )

        measured_start = project_se3_pose(matrix_to_flat(obj_pose))
        target = nominal[step_idx + 1]
        print(
            f"[AUDIT] step={step_idx} optimizer_target_is_nominal[{step_idx + 1}]="
            f"{np.array2string(target, precision=5, suppress_small=True)} "
            f"measured_start={np.array2string(measured_start, precision=5, suppress_small=True)}"
        )
        print(
            "[REAL] Latest object state before moving robot:",
            np.array2string(measured_start, precision=5, suppress_small=True),
        )

        if args.policy == "optimized":
            chosen_control, _ = optimize_control(
                system,
                aura_picker,
                opt_model,
                measured_start,
                target,
                original_control,
                float(args.duration),
                float(args.optimizer_pos_std),
                float(args.optimizer_rot_std),
                int(args.optimizer_num_states),
            )
        else:
            chosen_control = np.asarray(original_control, dtype=float)

        original_pred = np.asarray(
            system.propagate(measured_start, original_control, args.duration),
            dtype=float,
        )
        chosen_pred = np.asarray(
            system.propagate(measured_start, chosen_control, args.duration), dtype=float
        )
        original_pred_error = float(
            arrayDistance(original_pred, target, system=system.name)
        )
        chosen_pred_error = float(
            arrayDistance(chosen_pred, target, system=system.name)
        )

        print(
            "[REAL] measured start:",
            np.array2string(measured_start, precision=5, suppress_small=True),
        )
        print(
            "[REAL] target:", np.array2string(target, precision=5, suppress_small=True)
        )
        print(
            "[REAL] original control:",
            np.array2string(original_control, precision=5, suppress_small=True),
        )
        print(
            "[REAL] chosen control:",
            np.array2string(chosen_control, precision=5, suppress_small=True),
        )
        print(
            "[REAL] real path-generator control:",
            np.array2string(
                normalized_push_to_path_push(chosen_control),
                precision=5,
                suppress_small=True,
            ),
        )
        print(
            f"[REAL] predicted errors: original={original_pred_error:.6f}, chosen={chosen_pred_error:.6f}"
        )

        path_control = normalized_push_to_path_push(chosen_control)
        _, ws_paths = generate_path_form_params(
            Pose(obj_pose[:3, 3], matrix_to_quat(obj_pose[:3, :3])),
            obj_shape,
            path_control,
            tool_offset=tool_offset,
            total_time=float(args.duration),
            pre_push_offset=float(args.pre_push_offset),
            push_height=float(args.push_height),
            dt=float(args.execution_dt),
        )
        execute_push(
            robot,
            ws_paths,
            path_control,
            pause_between_movements=bool(args.pause_between_movements),
        )

        _pause_between_movements(
            bool(args.pause_between_movements),
            "Post-push motion complete; about to run confirmed post-push object detection",
        )

        time.sleep(float(args.detection_wait))
        new_obj_pose, _ = two_step_detect(
            robot,
            rough_detect_pose,
            height=0.25,
            pause_between_movements=bool(args.pause_between_movements),
            label=f"{args.policy} step {step_idx + 1} post-push two-step detection",
        )
        measured_end = project_se3_pose(matrix_to_flat(new_obj_pose))
        measured_error = float(arrayDistance(measured_end, target, system=system.name))
        print(
            "[REAL] measured end:",
            np.array2string(measured_end, precision=5, suppress_small=True),
        )
        print(f"[REAL] measured tracking error: {measured_error:.6f}")

        rows.append(
            RealStepResult(
                step=step_idx,
                policy=args.policy,
                measured_start_x=float(measured_start[0]),
                measured_start_y=float(measured_start[1]),
                measured_start_theta=float(measured_start[2]),
                target_x=float(target[0]),
                target_y=float(target[1]),
                target_theta=float(target[2]),
                original_rotation=float(original_control[0]),
                original_side=float(original_control[1]),
                original_distance=float(original_control[2]),
                chosen_rotation=float(chosen_control[0]),
                chosen_side=float(chosen_control[1]),
                chosen_distance=float(chosen_control[2]),
                original_pred_error=original_pred_error,
                chosen_pred_error=chosen_pred_error,
                measured_end_x=float(measured_end[0]),
                measured_end_y=float(measured_end[1]),
                measured_end_theta=float(measured_end[2]),
                measured_tracking_error=measured_error,
            )
        )
        save_results(out_dir, rows, metadata)
        obj_pose = new_obj_pose.copy()

    print("\n[REAL] Done.")
    error_list = [float(r.measured_tracking_error) for r in rows]
    if rows:
        print(
            f"[REAL] Mean measured tracking error: {np.mean(error_list):.6f}"
        )
    print(f"[REAL] Results saved in {out_dir}")
    return {
        "policy": args.policy,
        "controls": [np.asarray(control, dtype=float).copy() for control in controls],
        "errors": error_list,
        "mean_error": float(np.mean(error_list)) if error_list else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real UR10 pushing error experiment.")
    parser.add_argument("--obj-name", default="cracker_box_flipped")
    parser.add_argument("--num-controls", type=int, default=10)
    parser.add_argument(
        "--policy", choices=["naive", "optimized", "both"], default="optimized"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--execution-dt", type=float, default=0.008)
    parser.add_argument("--detection-wait", type=float, default=0.0)
    parser.add_argument("--push-offset", type=float, default=0.01)
    parser.add_argument(
        "--push-height",
        type=float,
        default=0.035,
        help="Contact waypoint z height above the table for the generated push path.",
    )
    parser.add_argument(
        "--reset-push-height",
        type=float,
        default=0.035,
        help="Reset-push z height above the table.",
    )
    parser.add_argument("--pre-push-offset", type=float, default=0.03)
    parser.add_argument("--optimizer-pos-std", type=float, default=0.035)
    parser.add_argument("--optimizer-rot-std", type=float, default=0.35)
    parser.add_argument("--optimizer-num-states", type=int, default=10000)
    parser.add_argument("--optimizer-learning-rate", type=float, default=5e-4)
    parser.add_argument("--optimizer-epochs", type=int, default=100)
    parser.add_argument("--center", type=float, nargs=2, default=[0.0, -0.7])
    parser.add_argument("--bounds", type=float, nargs=2, default=[0.30, 0.18])
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
    parser.add_argument("--output-dir", default="results/real_error_experiment")
    parser.add_argument(
        "--pause-between-movements",
        dest="pause_between_movements",
        action="store_true",
        default=True,
        help="Pause before each real robot motion phase so the setup can be inspected.",
    )
    parser.add_argument(
        "--no-pause-between-movements",
        dest="pause_between_movements",
        action="store_false",
        help="Run without interactive checkpoints between robot movements.",
    )
    parser.add_argument("--no-confirm", dest="require_confirm", action="store_false")
    parser.set_defaults(require_confirm=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_controls <= 0:
        raise ValueError("--num-controls must be positive")
    if args.policy == "both":
        base_output_dir = Path(args.output_dir)
        print(
            "[REAL] Running paired real-robot comparison: naive first, optimized second."
        )
        print(
            "[REAL] Each policy runs one physical trial from the currently detected object pose."
        )
        print("[REAL] Starting each rollout after a 1.0s pause.")
        shared_controls = None
        paired_results = {}
        for policy in ("naive", "optimized"):
            policy_args = copy.copy(args)
            policy_args.policy = policy
            policy_args.output_dir = str(base_output_dir / policy)
            policy_args.require_confirm = False
            policy_args.control_sequence = shared_controls
            print(
                f"\n[REAL] ===== Starting {policy} rollout ({args.num_controls} controls) ====="
            )
            time.sleep(1.0)
            result = run_real_error_experiment(policy_args)
            if shared_controls is None:
                shared_controls = result["controls"]
                print("[REAL] Stored naive control sequence; optimized rollout will reuse it exactly.")
            paired_results[policy] = result
        print("\n[REAL] ===== Paired tracking-error summary =====")
        for policy in ("naive", "optimized"):
            result = paired_results.get(policy, {})
            errors = np.asarray(result.get("errors", []), dtype=float)
            print(f"[REAL] {policy} errors: {np.array2string(errors, precision=6, suppress_small=True)}")
            print(f"[REAL] {policy} mean: {float(result.get('mean_error', float('nan'))):.6f}")
        return
    run_real_error_experiment(args)


if __name__ == "__main__":
    main()
