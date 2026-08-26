import math
from dataclasses import dataclass
from typing import List

import rtde_control
import rtde_receive


ROBOTIS_GRIPPER_ID = 1
ROBOTIS_GRIPPER_MIN_STROKE_MM = 0.0
ROBOTIS_GRIPPER_MAX_STROKE_MM = 106.0
ROBOTIS_GRIPPER_CLOSED_POSITION = 740
ROBOTIS_GRIPPER_DAEMON_URL = "http://127.0.0.1:40405/RPC2"
ROBOTIS_STATUS_REGISTER = 16
ROBOTIS_TORQUE_REGISTER = 17
ROBOTIS_START_POSITION_REGISTER = 18
ROBOTIS_FINAL_POSITION_REGISTER = 19
ROBOTIS_POSITION_TOLERANCE = 15


@dataclass(frozen=True)
class RobotisGripperResult:
    requested_stroke_mm: float
    target_position: int
    start_position: int
    final_position: int
    torque_enabled: bool
    reached_target: bool
    contact_grasp: bool = False
    force_percent: float = 0.0


def robotis_action_to_stroke(action: str) -> float:
    """Return the full-stroke endpoint for an open or close command."""

    command = str(action).strip().lower()
    if command == "open":
        return ROBOTIS_GRIPPER_MAX_STROKE_MM
    if command == "close":
        return ROBOTIS_GRIPPER_MIN_STROKE_MM
    raise ValueError("gripper action must be 'open' or 'close'")


def robotis_stroke_to_position(stroke_mm: float) -> int:
    """Apply the RH-P12-RN URCap's 0--106 mm stroke conversion."""

    stroke = float(stroke_mm)
    if not math.isfinite(stroke) or not 0.0 <= stroke <= ROBOTIS_GRIPPER_MAX_STROKE_MM:
        raise ValueError("RH-P12-RN stroke must be within [0, 106] mm")
    position = int(
        (math.acos((stroke - 10.0) / 111.5) - math.pi / 6.0)
        * 2048.0
        / math.pi
    )
    return min(ROBOTIS_GRIPPER_CLOSED_POSITION, max(0, position))


def build_robotis_gripper_script(
    stroke_mm: float,
    *,
    velocity_percent: float,
    force_percent: float,
    wait: bool,
) -> str:
    """Build the same XML-RPC calls emitted by ROBOTIS URCap 1.1.0."""

    velocity = float(velocity_percent)
    force = float(force_percent)
    if not math.isfinite(velocity) or not 1.0 <= velocity <= 100.0:
        raise ValueError("RH-P12-RN velocity must be within [1, 100] percent")
    if not math.isfinite(force) or not 0.0 <= force <= 100.0:
        raise ValueError("RH-P12-RN force must be within [0, 100] percent")

    current_value = int(force * 6.61)
    velocity_value = int(velocity * 10.0)
    position_value = robotis_stroke_to_position(stroke_mm)
    position_lower = max(0, position_value - ROBOTIS_POSITION_TOLERANCE)
    position_upper = min(1150, position_value + ROBOTIS_POSITION_TOLERANCE)
    contact_position_min = (
        robotis_stroke_to_position(ROBOTIS_GRIPPER_MAX_STROKE_MM)
        + ROBOTIS_POSITION_TOLERANCE
    )
    allow_contact_grasp = position_value == ROBOTIS_GRIPPER_CLOSED_POSITION
    lines = [
        f"write_output_integer_register({ROBOTIS_STATUS_REGISTER}, 0)",
        f"write_output_integer_register({ROBOTIS_TORQUE_REGISTER}, 0)",
        f"write_output_integer_register({ROBOTIS_START_POSITION_REGISTER}, -1)",
        f"write_output_integer_register({ROBOTIS_FINAL_POSITION_REGISTER}, -1)",
        f'gripper_daemon = rpc_factory("xmlrpc", "{ROBOTIS_GRIPPER_DAEMON_URL}")',
        f"is_reachable = gripper_daemon.is_reachable({ROBOTIS_GRIPPER_ID})",
        "if (is_reachable == False):",
        f"  write_output_integer_register({ROBOTIS_STATUS_REGISTER}, -1)",
        "else:",
        f"  torque_enabled = gripper_daemon.get_torque({ROBOTIS_GRIPPER_ID})",
        "  if (torque_enabled == False):",
        f"    gripper_daemon.set_torque({ROBOTIS_GRIPPER_ID}, 1)",
        "    sleep(0.5)",
        f"    torque_enabled = gripper_daemon.get_torque({ROBOTIS_GRIPPER_ID})",
        "  end",
        "  if (torque_enabled == False):",
        f"    write_output_integer_register({ROBOTIS_STATUS_REGISTER}, -2)",
        "  else:",
        f"    write_output_integer_register({ROBOTIS_TORQUE_REGISTER}, 1)",
        f"    start_position = gripper_daemon.get_position({ROBOTIS_GRIPPER_ID})",
        (
            f"    write_output_integer_register({ROBOTIS_START_POSITION_REGISTER}, "
            "start_position)"
        ),
        (
            f"    move_accepted = gripper_daemon.set_move({ROBOTIS_GRIPPER_ID}, "
            f"{current_value}, {velocity_value}, {position_value})"
        ),
        "    if (move_accepted == False):",
        f"      write_output_integer_register({ROBOTIS_STATUS_REGISTER}, -3)",
        "    else:",
    ]
    if wait:
        lines.extend(
            [
                "      sleep(0.5)",
                f"      final_position = gripper_daemon.get_position({ROBOTIS_GRIPPER_ID})",
                (
                    f"      at_target = ((final_position >= {position_lower}) and "
                    f"(final_position <= {position_upper}))"
                ),
                f"      moving = gripper_daemon.is_moving({ROBOTIS_GRIPPER_ID})",
                "      wait_checks = 0",
                (
                    "      while ((at_target == False) and (moving == True) "
                    "and (wait_checks < 80)):"
                ),
                "        sleep(0.25)",
                f"        final_position = gripper_daemon.get_position({ROBOTIS_GRIPPER_ID})",
                (
                    f"        at_target = ((final_position >= {position_lower}) and "
                    f"(final_position <= {position_upper}))"
                ),
                f"        moving = gripper_daemon.is_moving({ROBOTIS_GRIPPER_ID})",
                "        wait_checks = wait_checks + 1",
                "      end",
                (
                    f"      write_output_integer_register({ROBOTIS_FINAL_POSITION_REGISTER}, "
                    "final_position)"
                ),
                "      if (at_target == True):",
                f"        write_output_integer_register({ROBOTIS_STATUS_REGISTER}, 1)",
            ]
        )
        if allow_contact_grasp:
            lines.extend(
                [
                    (
                        "      elif ((moving == False) and "
                        f"(final_position > {contact_position_min})):"
                    ),
                    f"        write_output_integer_register({ROBOTIS_STATUS_REGISTER}, 4)",
                ]
            )
        lines.extend(
            [
                "      else:",
                f"        write_output_integer_register({ROBOTIS_STATUS_REGISTER}, 2)",
                "      end",
            ]
        )
    else:
        lines.extend(
            [
                "      sleep(0.1)",
                f"      final_position = gripper_daemon.get_position({ROBOTIS_GRIPPER_ID})",
                (
                    f"      write_output_integer_register({ROBOTIS_FINAL_POSITION_REGISTER}, "
                    "final_position)"
                ),
                f"      write_output_integer_register({ROBOTIS_STATUS_REGISTER}, 3)",
            ]
        )
    lines.extend(["    end", "  end", "end"])
    return "\n".join(lines) + "\n"


class RTDE:
    def __init__(self, robot_ip: str = "192.168.0.100"):
        """Initialize with the robot IP address"""
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    # Getters
    def get_joint_values(self) -> List[float]:
        """Get the joint positions in radians."""
        return self.rtde_r.getActualQ()

    def get_joint_speed(self) -> List[float]:
        """Get the joint speed in radians per second."""
        return self.rtde_r.getActualQd()

    def get_tool_pose(self) -> List[float]:
        """
        Get the pose of the Tool Center Point (TCP) in Cartesian space.

        Return [x, y, z, rx, ry, rz] position + rotation vector
        """
        return self.rtde_r.getActualTCPPose()

    def get_tool_speed(self) -> List[float]:
        """
        Get the speed of the Tool Center Point (TCP) in Cartesian space.

        Return [vx, vy, vz, wx, wy, wz]
        """
        return self.rtde_r.getActualTCPSpeed()

    def set_tool_pose(self, tcp: List[float]):
        """
        Set the Tool Center Point (TCP) in Cartesian space.

        The pose is defined as [x, y, z, rx, ry, rz] position + rotation vector
        """
        self.rtde_c.setTcp(tcp)

    def move_robotis_gripper(
        self,
        stroke_mm: float,
        *,
        velocity_percent: float = 30.0,
        force_percent: float = 20.0,
        wait: bool = True,
    ) -> RobotisGripperResult:
        """Move an RH-P12-RN through its robot-local ROBOTIS URCap daemon.

        ``sendCustomScriptFunction`` temporarily stops the regular ur_rtde
        control script, executes this self-contained command, and re-uploads
        the control script before returning.
        """

        script = build_robotis_gripper_script(
            stroke_mm,
            velocity_percent=velocity_percent,
            force_percent=force_percent,
            wait=wait,
        )
        completed = self.rtde_c.sendCustomScriptFunction(
            "aura_robotis_gripper_move", script
        )
        if not completed:
            raise RuntimeError("RH-P12-RN URCap command did not complete")

        status = self.rtde_r.getOutputIntRegister(ROBOTIS_STATUS_REGISTER)
        start_position = self.rtde_r.getOutputIntRegister(
            ROBOTIS_START_POSITION_REGISTER
        )
        final_position = self.rtde_r.getOutputIntRegister(
            ROBOTIS_FINAL_POSITION_REGISTER
        )
        torque_enabled = bool(
            self.rtde_r.getOutputIntRegister(ROBOTIS_TORQUE_REGISTER)
        )
        target_position = robotis_stroke_to_position(stroke_mm)
        allows_contact_grasp = target_position == ROBOTIS_GRIPPER_CLOSED_POSITION
        errors = {
            -1: "RH-P12-RN is not reachable through the URCap daemon",
            -2: "RH-P12-RN torque did not enable",
            -3: "RH-P12-RN move request was rejected",
            0: "RH-P12-RN command ended without a status",
            2: (
                "RH-P12-RN neither reached the closed endpoint nor established "
                "a stationary contact grasp"
                if allows_contact_grasp
                else "RH-P12-RN did not reach the requested open position"
            ),
        }
        if status in errors:
            raise RuntimeError(errors[status])
        if status not in (1, 3, 4):
            raise RuntimeError(f"Unknown RH-P12-RN status code: {status}")

        reached_target = (
            abs(final_position - target_position) <= ROBOTIS_POSITION_TOLERANCE
        )
        contact_grasp = (
            status == 4
            and allows_contact_grasp
            and torque_enabled
            and final_position
            > robotis_stroke_to_position(ROBOTIS_GRIPPER_MAX_STROKE_MM)
            + ROBOTIS_POSITION_TOLERANCE
        )
        if status == 4 and not contact_grasp:
            raise RuntimeError("RH-P12-RN reported an invalid contact-grasp status")
        if wait and not (reached_target or contact_grasp):
            raise RuntimeError(
                "RH-P12-RN stopped at raw position "
                f"{final_position}, expected {target_position}"
            )
        return RobotisGripperResult(
            requested_stroke_mm=float(stroke_mm),
            target_position=target_position,
            start_position=start_position,
            final_position=final_position,
            torque_enabled=torque_enabled,
            reached_target=reached_target,
            contact_grasp=contact_grasp,
            force_percent=float(force_percent),
        )

    def control_robotis_gripper(
        self,
        action: str,
        *,
        velocity_percent: float = 30.0,
        force_percent: float = 20.0,
        wait: bool = True,
    ) -> RobotisGripperResult:
        """Open fully, or close until the endpoint or a loaded contact grasp."""

        return self.move_robotis_gripper(
            robotis_action_to_stroke(action),
            velocity_percent=velocity_percent,
            force_percent=force_percent,
            wait=wait,
        )

    def disconnect(self) -> None:
        """Close both RTDE connections."""

        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

    # Joint control
    def move_joint(
        self,
        joint_values: List[float],
        speed: float = 1.05,
        acceleration: float = 1.4,
        # a bool specifying if the move command should be asynchronous
        asynchronous: bool = False,
    ):
        """Move the robot to the target joint positions."""
        self.rtde_c.moveJ(joint_values, speed, acceleration, asynchronous)

    def move_joint_trajectory(
        self,
        path: List[List[float]],
        # a bool specifying if the move command should be asynchronous
        asynchronous: bool = False,
    ):
        """
        Move the robot to follow a given path/trajectory,
        with each waypoint defined as
        [q1, q2, q3, q4, q5, q6, speed, acceleration, blend]
        (angles + others)
        """
        self.rtde_c.moveJ(path, asynchronous)

    def speed_joint(
        self,
        speeds: List[float],
        acceleration: float = 0.5,
        time: float = 0.0,
    ):
        """Accelerate linearly and continue with constant joint speed."""
        self.rtde_c.speedJ(speeds, acceleration, time)

    # Tool control
    def move_tool(
        self,
        pose: List[float],
        speed: float = 0.25,
        acceleration: float = 1.2,
        # a bool specifying if the move command should be asynchronous
        asynchronous: bool = False,
    ):
        """Move the robot to the tool position."""
        self.rtde_c.moveL(pose, speed, acceleration, asynchronous)

    def move_tool_trajectory(
        self,
        path: List[List[float]],
        # a bool specifying if the move command should be asynchronous
        asynchronous: bool = False,
    ):
        """
        Move the robot to follow a given tool path/trajectory,
        with each waypoint defined as
        [x, y, z, rx, ry, rz, speed, acceleration, blend]
        (position + rotation vector + others)
        """
        self.rtde_c.moveL(path, asynchronous)

    def speed_tool(
        self,
        speeds: List[float],
        acceleration: float = 0.25,
        time: float = 0.0,
    ):
        """Accelerate linearly and continue with constant joint speed."""
        self.rtde_c.speedL(speeds, acceleration, time)

    # Servo control
    def servo_joint(
        self,
        joint_values: List[float],  # Target joint positions
        speed: float = 0,  # joint velocity
        acceleration: float = 0,  # joint acceleration
        time: float = 0.008,  # time to control the robot
        lookahead_time: float = 0.1,  # project the current position forward
        gain: int = 300,  # P-term as in PID controller
    ):
        """
        Used to perform online realtime joint control

        The gain parameter works the same way as the P-term of a PID controller
        where it adjusts the current position towards the desired (q).
        The higher the gain, the faster reaction the robot will have.

        The parameter lookahead_time is used to project the current position
        forward in time with the current velocity. A low value gives fast
        reaction, a high value prevents overshoot.

        Note: A high gain or a short lookahead time may cause instability and
        vibrations. Especially if the target positions are noisy or updated
        at a low frequency It is preferred to call this function
        with a new setpoint (q) in each time step
        """
        self.rtde_c.servoJ(
            joint_values, speed, acceleration, time, lookahead_time, gain
        )

    def servo_tool(
        self,
        pose: List[float],  # Target joint positions
        speed: float = 0,  # joint velocity
        acceleration: float = 0,  # joint acceleration
        time: float = 0.008,  # time to control the robot
        lookahead_time: float = 0.1,  # project the current position forward
        gain: int = 300,  # P-term as in PID controller
    ):
        """
        Used to perform online realtime tool control

        The gain parameter works the same way as the P-term of a PID controller
        where it adjusts the current position towards the desired (q).
        The higher the gain, the faster reaction the robot will have.

        The parameter lookahead_time is used to project the current position
        forward in time with the current velocity. A low value gives fast
        reaction, a high value prevents overshoot.

        Note: A high gain or a short lookahead time may cause instability and
        vibrations. Especially if the target positions are noisy or updated
        at a low frequency It is preferred to call this function
        with a new setpoint (q) in each time step
        """
        # Tool pose [x, y, z, rx, ry, rz] position + rotation vector
        self.rtde_c.servoL(
            pose, speed, acceleration, time, lookahead_time, gain
        )

    # Stop
    def stop(
        self,
        a: float = 2.0,  # joint acceleration
        # a bool specifying if the move command should be asynchronous
        asynchronous: bool = False,
    ):
        """Stop the robot."""
        self.rtde_c.stopJ(a, asynchronous)

    def stop_script(self):
        """Terminate the script on controller"""
        self.rtde_c.stopScript()


if __name__ == "__main__":
    # Test the RTDE class
    rtde = RTDE()
    print("Joint values:", rtde.get_joint_values())
    print("Joint speed:", rtde.get_joint_speed())
    print("Tool pose:", rtde.get_tool_pose())
    print("Tool speed:", rtde.get_tool_speed())
    rtde.stop_script()
