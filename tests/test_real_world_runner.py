from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from geometry.object_model import get_obj_shape
from geometry.pose import matrix_to_flat
from geometry.random_push import generate_path_from_params
from real_world.physical_robot import PhysicalUR10
from real_world.rtde import (
    RTDE,
    ROBOTIS_FINAL_POSITION_REGISTER,
    ROBOTIS_START_POSITION_REGISTER,
    ROBOTIS_STATUS_REGISTER,
    ROBOTIS_TORQUE_REGISTER,
    RobotisGripperResult,
    build_robotis_gripper_script,
    robotis_action_to_stroke,
    robotis_stroke_to_position,
)
from scripts import run_real_world


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/systems/pushing_object.yaml"


def real_result(
    config: dict,
    *,
    task_time: float,
    final_x: float,
) -> dict:
    return {
        "schema_version": 1,
        "panel_id": "pushing_real",
        "system": "pushing_object",
        "environment": "real",
        "planner": "mppi",
        "method": "mppi",
        "run_number": 1,
        "seed": 7,
        "config_hash": run_real_world.config_hash(config),
        "initial_planning_seconds": 0.0,
        "initial_plan_hash": None,
        "initial_plan": None,
        "status": "success",
        "failure_reason": "",
        "nominal_execution_seconds": 2.0,
        "actual_execution_seconds": float(task_time),
        "online_replanning_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "blocking_replanning_seconds": 0.0,
        "compute_overrun_seconds": 0.0,
        "wall_time_definition": "test",
        "task_time_seconds": float(task_time),
        "raw_process_wall_seconds": float(task_time),
        "num_controls": 1,
        "num_replanning": 1,
        "cost": 0.1,
        "tracking_error_mean": 0.02,
        "tracking_error_list": [0.02],
        "goal_distance": 0.01,
        "final_state": [float(final_x), -0.5, 0.0],
        "planned_final_state": [float(final_x), -0.5, 0.0],
        "controls": [[0.0, 0.0, 0.08]],
        "control_duration_steps": [1],
        "control_duration_seconds": [1.0],
        "duration_audit_initial_tree": {
            "range_steps": [1, 1],
            "propagation_step_size_seconds": 1.0,
            "duration_step_histogram": {},
        },
    }


class FakeRTDEControl:
    def __init__(self):
        self.calls = []

    def sendCustomScriptFunction(self, name, script):
        self.calls.append((name, script))
        return True


class FakeRTDEReceive:
    def __init__(self, *, start_position=740, final_position=6, status=1):
        self.registers = {
            ROBOTIS_STATUS_REGISTER: status,
            ROBOTIS_TORQUE_REGISTER: 1,
            ROBOTIS_START_POSITION_REGISTER: start_position,
            ROBOTIS_FINAL_POSITION_REGISTER: final_position,
        }

    def getOutputIntRegister(self, register):
        return self.registers[register]


class FakeRTDE:
    def __init__(self):
        self.moves = []

    def control_robotis_gripper(self, action, **settings):
        self.moves.append((action, settings))


class FakePhysicalRobot:
    def __init__(self):
        self.actions = []

    def control_gripper(self, action, **settings):
        self.actions.append((action, settings))
        return RobotisGripperResult(
            requested_stroke_mm=0.0,
            target_position=740,
            start_position=6,
            final_position=320,
            torque_enabled=True,
            reached_target=False,
            contact_grasp=True,
            force_percent=float(settings["force_percent"]),
        )


def test_robotis_urcap_conversion_and_script_match_vendor_contract() -> None:
    assert robotis_stroke_to_position(0.0) == 740
    assert robotis_stroke_to_position(106.0) == 6
    script = build_robotis_gripper_script(
        0.0, velocity_percent=30.0, force_percent=20.0, wait=True
    )
    assert 'rpc_factory("xmlrpc", "http://127.0.0.1:40405/RPC2")' in script
    assert "gripper_daemon.set_torque(1, 1)" in script
    assert "if (torque_enabled == False):" in script
    assert "move_accepted = gripper_daemon.set_move(1, 132, 300, 740)" in script
    assert "at_target = ((final_position >= 725) and (final_position <= 755))" in script
    assert "moving = gripper_daemon.is_moving(1)" in script
    assert "and (moving == True) and (wait_checks < 80)" in script
    assert "elif ((moving == False) and (final_position > 21)):" in script
    assert "write_output_integer_register(16, 4)" in script
    assert "get_position(1)" in script
    maximum_force_script = build_robotis_gripper_script(
        0.0, velocity_percent=30.0, force_percent=100.0, wait=True
    )
    assert "set_move(1, 661, 300, 740)" in maximum_force_script
    with pytest.raises(ValueError, match="stroke"):
        robotis_stroke_to_position(107.0)


def test_robotis_actions_select_full_stroke_endpoints() -> None:
    assert robotis_action_to_stroke("open") == 106.0
    assert robotis_action_to_stroke("close") == 0.0
    with pytest.raises(ValueError, match="open.*close"):
        robotis_action_to_stroke("half")


def test_rtde_sends_one_self_contained_gripper_function() -> None:
    interface = RTDE.__new__(RTDE)
    interface.rtde_c = FakeRTDEControl()
    interface.rtde_r = FakeRTDEReceive()
    result = interface.move_robotis_gripper(
        106.0, velocity_percent=25.0, force_percent=10.0
    )
    assert len(interface.rtde_c.calls) == 1
    name, script = interface.rtde_c.calls[0]
    assert name == "aura_robotis_gripper_move"
    assert "set_move(1, 66, 250, 6)" in script
    assert result.reached_target
    assert result.start_position == 740
    assert result.final_position == 6


def test_rtde_rejects_false_gripper_success() -> None:
    interface = RTDE.__new__(RTDE)
    interface.rtde_c = FakeRTDEControl()
    interface.rtde_r = FakeRTDEReceive(
        start_position=6, final_position=6, status=1
    )
    with pytest.raises(RuntimeError, match="stopped at raw position 6"):
        interface.move_robotis_gripper(0.0)


def test_close_accepts_torque_held_contact_with_a_grasped_object() -> None:
    interface = RTDE.__new__(RTDE)
    interface.rtde_c = FakeRTDEControl()
    interface.rtde_r = FakeRTDEReceive(
        start_position=6, final_position=320, status=4
    )

    result = interface.control_robotis_gripper("close", force_percent=100.0)

    assert not result.reached_target
    assert result.contact_grasp
    assert result.torque_enabled
    assert result.force_percent == 100.0


def test_open_cannot_report_contact_as_success() -> None:
    interface = RTDE.__new__(RTDE)
    interface.rtde_c = FakeRTDEControl()
    interface.rtde_r = FakeRTDEReceive(
        start_position=320, final_position=320, status=4
    )
    with pytest.raises(RuntimeError, match="invalid contact-grasp"):
        interface.control_robotis_gripper("open", force_percent=100.0)


@pytest.mark.parametrize("start_position", [6, 250, 500, 739])
def test_close_reaches_full_endpoint_from_any_start(start_position: int) -> None:
    interface = RTDE.__new__(RTDE)
    interface.rtde_c = FakeRTDEControl()
    interface.rtde_r = FakeRTDEReceive(
        start_position=start_position, final_position=740, status=1
    )
    result = interface.control_robotis_gripper("close")
    assert result.start_position == start_position
    assert result.final_position == 740
    assert result.reached_target


def test_physical_robot_has_no_separate_gripper_connection() -> None:
    rtde = FakeRTDE()
    hand_camera = object()
    robot = PhysicalUR10(rtde=rtde, hand_camera=hand_camera)
    assert not hasattr(robot, "gripper")
    assert not hasattr(robot, "top_cam")
    assert not hasattr(robot, "get_object_pose_top")
    robot.control_gripper("close", velocity_percent=15.0, force_percent=12.0)
    assert rtde.moves == [
        (
            "close",
            {
                "velocity_percent": 15.0,
                "force_percent": 12.0,
                "wait": True,
            },
        )
    ]


def test_physical_robot_connects_only_to_the_in_hand_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_connections = []

    class FakeCamera:
        def __init__(self, host, port):
            camera_connections.append((host, port))

    monkeypatch.setattr("real_world.physical_robot.Camera", FakeCamera)
    robot = PhysicalUR10(rtde=FakeRTDE(), camera_host="camera.local")

    assert camera_connections == [("camera.local", "5000")]
    assert isinstance(robot.hand_cam, FakeCamera)


def test_robot_motion_debug_prints_current_target_and_settings(capsys) -> None:
    class MotionRTDE:
        def __init__(self):
            self.calls = []

        def get_joint_values(self):
            return [1.0, -1.0, 0.5, -0.5, 0.25, -0.25]

        def get_tool_pose(self):
            return [0.1, -0.5, 0.4, 0.0, 3.14, 0.0]

        def move_joint(self, target, **settings):
            self.calls.append(("moveJ", np.asarray(target), settings))

        def move_tool(self, target, **settings):
            self.calls.append(("moveL", np.asarray(target), settings))

    rtde = MotionRTDE()
    robot = PhysicalUR10(rtde=rtde, hand_camera=object(), motion_debug=True)
    joint_target = np.array([1.3, -1.57, 1.2, -1.2, -1.57, -0.27])
    tool_target = np.array([0.0, -0.49, 0.35, 0.0, np.pi, 0.0])

    robot.move_joint(
        joint_target,
        speed=1.0,
        acceleration=0.5,
        label="test home target",
    )
    robot.move_tool(
        tool_target,
        speed=0.25,
        acceleration=1.2,
        label="test overhead target",
    )

    output = capsys.readouterr().out
    assert "[ROBOT MOVE] test home target" in output
    assert "[ROBOT MOVE] test overhead target" in output
    assert "current joints [rad]" in output
    assert "current TCP [m, rotvec]" in output
    assert "target:" in output
    assert "speed: 1.0, acceleration: 0.5" in output
    assert "speed: 0.25, acceleration: 1.2" in output
    assert [call[0] for call in rtde.calls] == ["moveJ", "moveL"]
    assert all("label" not in call[2] for call in rtde.calls)


def test_robot_motion_debug_is_quiet_by_default(capsys) -> None:
    class MotionRTDE:
        @staticmethod
        def move_tool(target, **settings):
            del target, settings

    robot = PhysicalUR10(rtde=MotionRTDE(), hand_camera=object())
    robot.move_tool([0.0, -0.5, 0.35, 0.0, np.pi, 0.0])

    assert capsys.readouterr().out == ""


def test_real_trial_always_closes_gripper_with_torque_holding() -> None:
    config = run_real_world.load_config(CONFIG_PATH)
    robot = FakePhysicalRobot()
    result = run_real_world.close_gripper_for_trial(robot, config)
    assert robot.actions == [
        (
            "close",
            {
                "velocity_percent": config["gripper_velocity_percent"],
                "force_percent": config["gripper_force_percent"],
                "wait": True,
            },
        )
    ]
    assert config["gripper_force_percent"] == 100.0
    assert result.contact_grasp
    assert result.torque_enabled
    assert result.force_percent == 100.0
    args = run_real_world.parse_args(
        ["--method", "mppi", "--trial", "1"]
    )
    assert not hasattr(args, "gripper")
    assert not hasattr(args, "results_root")


def test_test_gripper_executes_immediately_without_execute_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class GripperOnlyRTDE:
        def __init__(self, robot_ip):
            calls.append(("connect", robot_ip))

        def control_robotis_gripper(self, action, **settings):
            calls.append(("move", action, settings))

            return RobotisGripperResult(
                requested_stroke_mm=robotis_action_to_stroke(action),
                target_position=740,
                start_position=6,
                final_position=739,
                torque_enabled=True,
                reached_target=True,
                force_percent=float(settings["force_percent"]),
            )

        def disconnect(self):
            calls.append(("disconnect",))

    monkeypatch.setattr("real_world.rtde.RTDE", GripperOnlyRTDE)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: pytest.fail("gripper-only command must not prompt"),
    )
    run_real_world.main(["--test-gripper", "close", "--config", str(CONFIG_PATH)])
    assert calls == [
        ("connect", "192.168.0.100"),
        (
            "move",
            "close",
            {
                "velocity_percent": 30.0,
                    "force_percent": 100.0,
            },
        ),
        ("disconnect",),
    ]


def test_real_config_defines_a_valid_nominal_task() -> None:
    config = run_real_world.load_config(CONFIG_PATH)
    assert np.asarray(config["start_state"], dtype=float).shape == (3,)
    assert np.asarray(config["goal_state"], dtype=float).shape == (3,)
    assert float(config["goal_threshold"]) > 0.0
    assert 0.0 < float(config["planner_goal_threshold"]) <= float(
        config["goal_threshold"]
    )
    assert np.asarray(config["rough_detect_pose"], dtype=float).shape == (6,)
    run_real_world.validate_config(config, "mppi")
    assert run_real_world.start_distance(
        np.array(config["start_state"], dtype=float), config
    ) == pytest.approx(0.0)
    assert config["object_workspace_bounds"][3] >= (
        config["start_state"][1] + config["start_threshold"]
    )
    assert config["physical_push_duration_seconds"] == pytest.approx(4.0)
    assert config["execution_dt"] == pytest.approx(0.008)
    assert config["task_time_limit_seconds"] == pytest.approx(300.0)
    assert config["mppi"]["zero_anchor_pushing_model"] is False
    assert config["mppi"]["short_push_cost_scale"] == pytest.approx(2000.0)
    assert config["mppi"]["minimum_preferred_push_distance"] == pytest.approx(0.0)
    assert config["replanning_max_distance"] == pytest.approx(0.1)
    assert config["randup_planning_time"] == pytest.approx(10.0)
    assert config["randup_optimize_after_first_solution"] is True
    assert config["randup_control_duration_min"] == 1
    assert config["randup_control_duration_max"] == 1
    assert config["randup_position_std"] == pytest.approx(0.005)
    assert config["randup_rotation_std"] == pytest.approx(0.05)
    assert config["goal_bias"] == pytest.approx(0.30)


def test_real_start_goal_and_thresholds_are_controlled_by_yaml() -> None:
    config = run_real_world.load_config(CONFIG_PATH)
    config["start_state"] = [-0.20, -0.70, 0.25]
    config["goal_state"] = [0.40, -0.45, -0.50]
    config["start_threshold"] = 0.08
    config["goal_threshold"] = 0.12
    config["planner_goal_threshold"] = 0.06

    run_real_world.validate_config(config, "mppi")


def test_randup_plan_is_expanded_and_printed_per_physical_push(capsys) -> None:
    class AdditiveSystem:
        @staticmethod
        def propagate(state, control, duration):
            return np.asarray(state) + np.asarray(control) * float(duration)

    solution = {
        "controls": [[0.1, 0.0, 0.0], [0.0, -0.2, 0.1]],
        "time_steps": [2, 1],
    }
    controls, expected = run_real_world.expand_randup_plan(
        AdditiveSystem(),
        np.zeros(3),
        solution,
        1.0,
    )

    assert len(controls) == 3
    np.testing.assert_allclose(
        expected,
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.2, -0.2, 0.1],
        ],
    )

    run_real_world.print_randup_plan(2, 2, controls, expected)
    output = capsys.readouterr().out
    assert "RANDUP plan 2: 2 planner controls expanded to 3 physical pushes" in output
    assert "RANDUP plan 2 expected trajectory" in output
    assert "step 3: [ 0.2 -0.2  0.1]" in output


def test_randup_replans_when_measured_pose_exceeds_tracking_threshold() -> None:
    error, needs_replan = run_real_world.randup_requires_replanning(
        measured_state=np.array([0.11, -0.5, 0.0]),
        expected_state=np.array([0.0, -0.5, 0.0]),
        threshold=0.1,
    )
    assert error == pytest.approx(0.11)
    assert needs_replan is True

    error, needs_replan = run_real_world.randup_requires_replanning(
        measured_state=np.array([0.05, -0.5, 0.0]),
        expected_state=np.array([0.0, -0.5, 0.0]),
        threshold=0.1,
    )
    assert error == pytest.approx(0.05)
    assert needs_replan is False


def test_real_yaml_still_rejects_a_goal_outside_the_workspace() -> None:
    config = run_real_world.load_config(CONFIG_PATH)
    config["goal_state"] = [0.46, -0.50, 0.0]

    with pytest.raises(ValueError, match="goal_state lies outside"):
        run_real_world.validate_config(config, "mppi")


def test_timeout_task_time_is_saved_as_the_exact_limit() -> None:
    from real_world import real_execution

    assert run_real_world.recorded_task_time("timeout", 304.7, 300.0) == 300.0
    assert real_execution.recorded_task_time("timeout", 299.2, 300.0) == 300.0
    assert run_real_world.recorded_task_time("success", 42.5, 300.0) == 42.5
    assert real_execution.recorded_task_time("failure", 17.0, 300.0) == 17.0


def test_two_step_detection_uses_only_the_requested_two_moves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from real_world import real_execution

    class DetectionRobot:
        def __init__(self):
            self.moves = []

        def get_q_values(self):
            return np.zeros(6)

        def move_tool(self, pose, **settings):
            self.moves.append((np.asarray(pose, dtype=float), settings))

    rough_pose = np.array([-0.05, -0.65, 0.6, 0.0, np.pi, 0.0])
    rough_object_pose = np.eye(4)
    rough_object_pose[:2, 3] = [0.02, -0.51]
    refined_object_pose = np.eye(4)
    refined_object_pose[:2, 3] = [0.01, -0.50]
    detections = iter(
        [(rough_object_pose, "rough-box"), (refined_object_pose, "refined-box")]
    )
    monkeypatch.setattr(
        real_execution,
        "get_object_pose",
        lambda *args, **kwargs: next(detections),
    )
    sleeps = []
    monkeypatch.setattr(real_execution.time, "sleep", sleeps.append)

    robot = DetectionRobot()
    pose, bounding_box = real_execution.two_step_detection(
        robot, rough_pose, height=0.12
    )

    assert len(robot.moves) == 2
    np.testing.assert_array_equal(robot.moves[0][0], rough_pose)
    np.testing.assert_allclose(
        robot.moves[1][0], [0.02, -0.51, 0.35, 0.0, np.pi, 0.0]
    )
    assert robot.moves[0][1] == {
        "speed": real_execution.DETECTION_MOVE_SPEED,
        "acceleration": real_execution.DETECTION_MOVE_ACCELERATION,
        "label": "object detection: rough detection pose",
    }
    assert robot.moves[1][1] == {
        "speed": real_execution.DETECTION_MOVE_SPEED,
        "acceleration": real_execution.DETECTION_MOVE_ACCELERATION,
        "label": "object detection: refined overhead detection pose",
    }
    assert sleeps == [5.0]
    np.testing.assert_array_equal(pose, refined_object_pose)
    assert bounding_box == "refined-box"


def test_camera_pose_is_rotated_into_the_pushing_model_frame() -> None:
    from real_world import real_execution

    class CameraRobot:
        @staticmethod
        def get_ee_transform():
            return np.eye(4)

        @staticmethod
        def get_object_pose_hand():
            pose = np.eye(4)
            pose[:3, :3] = real_execution.R.from_euler(
                "z", -np.pi / 2.0
            ).as_matrix()
            pose[:3, 3] = [0.02, -0.51, 0.26]
            return {
                "pose": pose,
                "bounding_box": "box",
                "result_image": None,
            }

    pose, bounding_box = real_execution.get_object_pose(
        CameraRobot(), max_retries=0
    )
    calibrated = real_execution.project_se3_to_se2(matrix_to_flat(pose))

    np.testing.assert_allclose(calibrated, [0.02, -0.51, 0.0], atol=1e-12)
    assert bounding_box == "box"


def test_every_post_push_detection_repeats_the_full_two_step_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from real_world import real_execution

    robot = object()
    rough_pose = np.array([-0.05, -0.5, 0.5, 0.0, np.pi, 0.0])
    first_pose = np.eye(4)
    first_pose[0, 3] = 0.12
    later_pose = np.eye(4)
    later_pose[0, 3] = 0.21
    calls = []

    def fake_two_step(robot_arg, rough_pose_arg, **settings):
        calls.append(("two-step", robot_arg, np.asarray(rough_pose_arg), settings))
        if len(calls) == 1:
            return first_pose, "first-box"
        return later_pose, "later-box"

    monkeypatch.setattr(real_execution, "two_step_detection", fake_two_step)

    detected_first, first_box = real_execution.detect_after_push(
        robot,
        rough_detect_pose=rough_pose,
        completed_pushes=1,
        height=0.12,
        debug_img_id="mppi_1_1",
    )
    detected_later, later_box = real_execution.detect_after_push(
        robot,
        rough_detect_pose=rough_pose,
        completed_pushes=2,
        height=0.12,
        debug_img_id="mppi_1_2",
    )

    assert detected_first is first_pose
    assert first_box == "first-box"
    assert detected_later is later_pose
    assert later_box == "later-box"
    assert [call[0] for call in calls] == ["two-step", "two-step"]
    assert calls[0][1] is robot
    np.testing.assert_array_equal(calls[0][2], rough_pose)
    assert calls[0][3] == {
        "height": 0.12,
        "label": "post-push 1 two-step detection",
    }
    assert calls[1][1] is robot
    np.testing.assert_array_equal(calls[1][2], rough_pose)
    assert calls[1][3] == {
        "height": 0.12,
        "label": "post-push 2 two-step detection",
    }


def test_real_push_uses_the_slower_motion_profile() -> None:
    from real_world import real_execution

    class PushRobot:
        def __init__(self):
            self.calls = []

        def get_q_values(self):
            return np.zeros(6)

        def move_tool(self, pose, **settings):
            self.calls.append(("move", np.asarray(pose, dtype=float), settings))

        def execute_ee_waypoints(self, waypoints, **settings):
            self.calls.append(
                ("trajectory", np.asarray(waypoints, dtype=float), settings)
            )

    robot = PushRobot()
    path = np.array(
        [
            [0.10, -0.50, 0.30, 1.0, 0.0, 0.0, 0.0],
            [0.18, -0.50, 0.30, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    real_execution.execute_push(robot, path, np.array([0.0, 0.0, 0.08]))

    assert [call[0] for call in robot.calls] == [
        "move",
        "move",
        "trajectory",
        "move",
    ]
    assert robot.calls[0][2] == {
        "to_rotvec": True,
        "speed": real_execution.PUSH_TRANSIT_SPEED,
        "acceleration": real_execution.PUSH_TRANSIT_ACCELERATION,
        "label": "push: pre-push clearance pose",
    }
    assert robot.calls[1][2] == {
        "to_rotvec": True,
        "speed": real_execution.PUSH_CONTACT_SPEED,
        "acceleration": real_execution.PUSH_CONTACT_ACCELERATION,
        "label": "push: first contact waypoint",
    }
    assert robot.calls[2][2] == {
        "to_rotvec": True,
        "label": "push: contact trajectory",
    }
    assert robot.calls[3][2] == {
        "to_rotvec": True,
        "speed": real_execution.PUSH_TRANSIT_SPEED,
        "acceleration": real_execution.PUSH_TRANSIT_ACCELERATION,
        "label": "push: post-push clearance pose",
    }


def test_real_push_path_stretches_the_collection_profile_smoothly() -> None:
    config = run_real_world.load_config(CONFIG_PATH)
    pose = np.eye(4)
    pose[:2, 3] = [0.1, -0.7]
    shape = get_obj_shape(
        REPO_ROOT / "simulation/assets/cracker_box_flipped/textured.obj"
    )
    control = np.array([0.25, 0.1, 0.08])
    actual = run_real_world.make_push_path(pose, shape, control, config)
    _, expected = generate_path_from_params(
        matrix_to_flat(pose)[None, :],
        shape,
        control[None, :],
        tool_offset=np.array([0.0, 0.0, -0.01, 1.0, 0.0, 0.0, 0.0]),
        pre_push_offset=0.03,
        duration=float(config["physical_push_duration_seconds"]),
        dt=0.008,
        push_height=config["push_height"],
        relative_push_offset=True,
    )
    _, original_speed_path = generate_path_from_params(
        matrix_to_flat(pose)[None, :],
        shape,
        control[None, :],
        tool_offset=np.array([0.0, 0.0, -0.01, 1.0, 0.0, 0.0, 0.0]),
        pre_push_offset=0.03,
        duration=2.0,
        dt=0.008,
        push_height=config["push_height"],
        relative_push_offset=True,
    )
    np.testing.assert_allclose(actual, expected[0], rtol=0.0, atol=0.0)
    assert actual.shape == (500, 7)
    np.testing.assert_allclose(actual[[0, -1]], original_speed_path[0][[0, -1]])
    slow_peak_step = np.linalg.norm(np.diff(actual[:, :3], axis=0), axis=1).max()
    original_peak_step = np.linalg.norm(
        np.diff(original_speed_path[0, :, :3], axis=0), axis=1
    ).max()
    assert slow_peak_step < 0.51 * original_peak_step


def test_push_frame_summary_composes_relative_face_with_global_object_yaw() -> None:
    from real_world.real_execution import to_se3_matrix

    config = run_real_world.load_config(CONFIG_PATH)
    object_state = np.array([0.0, -0.5, np.deg2rad(-90.66)])
    control = np.array([0.75, 0.4, 0.13])
    shape = get_obj_shape(
        REPO_ROOT / "simulation/assets/cracker_box_flipped/textured.obj"
    )
    path = run_real_world.make_push_path(
        to_se3_matrix(object_state),
        shape,
        control,
        config,
    )

    summary = run_real_world.push_frame_summary(object_state, control, path)

    assert summary["object_yaw_degrees"] == pytest.approx(-90.66)
    assert summary["relative_face_degrees"] == pytest.approx(270.0)
    assert summary["global_contact_bearing_degrees"] == pytest.approx(179.34)
    assert summary["global_push_heading_degrees"] == pytest.approx(-0.66)


def test_existing_trial_is_replaced_only_when_save_trial_is_called(
    tmp_path: Path,
) -> None:
    config = run_real_world.load_config(CONFIG_PATH)
    csv_path, artifact_path = run_real_world.result_locations(
        tmp_path, "mppi", 1, config
    )
    old = real_result(config, task_time=10.0, final_x=0.30)
    new = real_result(config, task_time=20.0, final_x=0.35)
    run_real_world.save_trial(old, tmp_path, "mppi", config)
    spec = run_real_world.dry_run_spec(
        "mppi", 1, 7, CONFIG_PATH, tmp_path, config
    )
    assert spec["task_time_limit_seconds"] == 300.0
    assert spec["existing_csv_will_be_replaced"] is True
    assert spec["existing_artifact_will_be_replaced"] is True
    assert json.loads(artifact_path.read_text())["task_time_seconds"] == 10.0

    run_real_world.save_trial(new, tmp_path, "mppi", config)

    assert json.loads(artifact_path.read_text())["task_time_seconds"] == 20.0
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]["overall_time"]) == pytest.approx(20.0)


@pytest.mark.parametrize("method", ["mppi", "randup"])
def test_dry_run_never_calls_physical_execution(
    method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry-run attempted physical execution")

    monkeypatch.setattr(run_real_world, "run_physical_trial", fail_if_called)
    run_real_world.main(
        [
            "--method",
            method,
            "--trial",
            "1",
            "--config",
            str(CONFIG_PATH),
        ]
    )
    assert list(tmp_path.rglob("*")) == []


def test_execute_has_no_typed_authorization_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fail_if_prompted(*args, **kwargs):
        raise AssertionError("live execution requested a typed authorization phrase")

    monkeypatch.setattr("builtins.input", fail_if_prompted)
    monkeypatch.setenv(run_real_world.RUNTIME_READY_ENV, "1")
    monkeypatch.setattr(run_real_world.time, "sleep", calls.append)
    monkeypatch.setattr(
        run_real_world,
        "run_physical_trial",
        lambda args, config, seed: calls.append((args.method, seed))
        or {"status": "success", "task_time_seconds": 1.0},
    )
    monkeypatch.setattr(
        run_real_world,
        "save_trial",
        lambda *args: (tmp_path / "result.csv", tmp_path / "trial.json"),
    )
    monkeypatch.setattr(
        run_real_world,
        "summarize_results",
        lambda results_root: {"results_root": str(results_root), "real_world": []},
    )

    run_real_world.main(
        [
            "--method",
            "mppi",
            "--trial",
            "1",
            "--config",
            str(CONFIG_PATH),
            "--execute",
        ]
    )

    assert calls[0][0] == "mppi"


@pytest.mark.parametrize("method", ["mppi", "randup"])
def test_real_pushes_have_no_input_gate(method: str) -> None:
    args = run_real_world.parse_args(
        [
            "--method",
            method,
            "--trial",
            "1",
            "--execute",
        ]
    )

    assert not hasattr(args, "confirm_each_push")
    assert not hasattr(run_real_world, "confirm_physical_push")


def test_live_runtime_relaunches_with_project_torch_and_cudnn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch_environment = tmp_path / "pytorch-gpu"
    torch_packages = torch_environment / "lib/python3.10/site-packages"
    cudnn_library = torch_packages / "nvidia/cudnn/lib"
    cudnn_library.mkdir(parents=True)
    (cudnn_library / "libcudnn.so.9").touch()
    captured = {}

    class Relaunched(Exception):
        pass

    def capture_relaunch(executable, command, environment):
        captured.update(
            executable=executable,
            command=command,
            environment=environment,
        )
        raise Relaunched

    monkeypatch.delenv(run_real_world.RUNTIME_READY_ENV, raising=False)
    monkeypatch.setenv("AURA_TORCH_ENV", str(torch_environment))
    monkeypatch.setattr(run_real_world.os, "execvpe", capture_relaunch)
    monkeypatch.setattr(
        run_real_world.sys,
        "argv",
        ["run_real_world.py", "--method", "mppi", "--trial", "1", "--execute"],
    )

    with pytest.raises(Relaunched):
        run_real_world.ensure_runtime_environment()

    assert captured["executable"] == sys.executable
    assert captured["command"][0] == sys.executable
    assert str(torch_packages) in captured["environment"]["PYTHONPATH"]
    assert str(cudnn_library) in captured["environment"]["LD_LIBRARY_PATH"]
    assert captured["environment"][run_real_world.RUNTIME_READY_ENV] == "1"


def test_result_paths_and_empty_summary(tmp_path: Path) -> None:
    config = run_real_world.load_config(CONFIG_PATH)
    mppi_csv, mppi_artifact = run_real_world.result_locations(
        tmp_path, "mppi", 4, config
    )
    randup_csv, randup_artifact = run_real_world.result_locations(
        tmp_path, "randup", 4, config
    )
    assert mppi_csv.name == "mppi_004.csv"
    assert randup_csv.name == "randup_rrt_m50_004.csv"
    assert mppi_artifact.name == "mppi_trial-004.json"
    assert randup_artifact.name == "randup_rrt_m50_trial-004.json"
    summary = run_real_world.summarize_results(tmp_path)
    assert [row["completed_trials"] for row in summary["real_world"]] == [0, 0]


def test_plot_failure_does_not_discard_saved_trial(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from scripts import plot_task_time

    config = run_real_world.load_config(CONFIG_PATH)
    result = real_result(config, task_time=20.0, final_x=0.35)

    def fail_plot(_results_root):
        raise RuntimeError("synthetic plot failure")

    monkeypatch.setattr(plot_task_time, "make_figure", fail_plot)
    csv_path, artifact_path = run_real_world.save_trial(
        result, tmp_path, "mppi", config
    )

    assert csv_path.is_file()
    assert artifact_path.is_file()
    assert "Trial results were saved" in capsys.readouterr().err


def test_save_trial_updates_summary_and_figure7(tmp_path: Path) -> None:
    config = run_real_world.load_config(CONFIG_PATH)
    result = {
        "schema_version": 1,
        "panel_id": "pushing_real",
        "system": "pushing_object",
        "environment": "real",
        "planner": "mppi",
        "method": "mppi",
        "run_number": 1,
        "seed": 7,
        "config_hash": run_real_world.config_hash(config),
        "initial_planning_seconds": 0.0,
        "initial_plan_hash": None,
        "initial_plan": None,
        "status": "success",
        "failure_reason": "",
        "nominal_execution_seconds": 2.0,
        "actual_execution_seconds": 3.0,
        "online_replanning_seconds": 0.0,
        "optimizer_seconds": 1.0,
        "blocking_replanning_seconds": 0.0,
        "compute_overrun_seconds": 0.0,
        "wall_time_definition": "test",
        "task_time_seconds": 4.0,
        "raw_process_wall_seconds": 4.5,
        "num_controls": 1,
        "num_replanning": 1,
        "cost": 0.1,
        "tracking_error_mean": 0.02,
        "tracking_error_list": [0.02],
        "goal_distance": 0.01,
        "final_state": [0.35, -0.5, 0.0],
        "planned_final_state": [0.34, -0.5, 0.0],
        "controls": [[0.0, 0.0, 0.08]],
        "control_duration_steps": [1],
        "control_duration_seconds": [1.0],
        "duration_audit_initial_tree": {
            "range_steps": [1, 1],
            "propagation_step_size_seconds": 1.0,
            "duration_step_histogram": {},
        },
    }
    csv_path, artifact_path = run_real_world.save_trial(
        result, tmp_path, "mppi", config
    )
    assert csv_path.is_file()
    assert artifact_path.is_file()
    for suffix in ("png", "pdf", "svg"):
        assert (tmp_path / f"task_time_comparison.{suffix}").is_file()
    summary = run_real_world.summarize_results(tmp_path)["real_world"][0]
    assert summary["completed_trials"] == 1
    assert summary["successful_mean_seconds"] == pytest.approx(4.0)
    assert summary["figure7_penalized_mean_seconds"] == pytest.approx(4.0)
