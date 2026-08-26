import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from real_world.rtde import RTDE
from real_world.camera import Camera


class PhysicalUR10:
    def __init__(
        self,
        robot_ip: str = "192.168.0.100",
        camera_host: str = "192.168.0.101",
        *,
        rtde: RTDE | None = None,
        hand_camera: Camera | None = None,
        motion_debug: bool = False,
    ):
        """Connect to the UR controller and the in-hand camera service.

        The RH-P12-RN has no separate network connection here. It is wired to
        the UR wrist and controlled by the installed ROBOTIS URCap through the
        existing robot RTDE connection.
        """

        self.rtde = rtde if rtde is not None else RTDE(robot_ip)
        self.hand_cam = (
            hand_camera if hand_camera is not None else Camera(camera_host, "5000")
        )
        self.motion_debug = bool(motion_debug)

    @staticmethod
    def format_motion_values(values) -> str:
        return np.array2string(
            np.asarray(values, dtype=float), precision=5, suppress_small=True
        )

    def print_motion_debug(
        self,
        label: str,
        motion_type: str,
        target,
        *,
        speed,
        acceleration,
        first_target=None,
        waypoint_count: int | None = None,
        dt: float | None = None,
    ) -> None:
        """Print the measured robot state and next commanded motion."""

        if not self.motion_debug:
            return

        print(f"\n[ROBOT MOVE] {label}")
        print(f"  type: {motion_type}")
        print(
            "  current joints [rad]:",
            self.format_motion_values(self.get_q_values()),
        )
        print(
            "  current TCP [m, rotvec]:",
            self.format_motion_values(self.get_ee_pose()),
        )
        if first_target is not None:
            print("  first target:", self.format_motion_values(first_target))
            print("  final target:", self.format_motion_values(target))
            print(f"  waypoints: {waypoint_count}, dt: {float(dt):.6f}s")
        else:
            print("  target:", self.format_motion_values(target))
        print(f"  speed: {speed}, acceleration: {acceleration}", flush=True)

    # Joint control
    def execute_trajectory(
        self,
        waypoints,
        d_t: float = 0.008,
        *,
        label: str = "joint trajectory",
        **kwargs,
    ):
        """Execute a trajectory"""
        waypoints = list(waypoints)
        if not waypoints:
            raise ValueError("joint trajectory requires at least one waypoint")
        self.print_motion_debug(
            label,
            "servoJ trajectory",
            waypoints[-1],
            speed=kwargs.get("speed", "servoJ"),
            acceleration=kwargs.get("acceleration", "servoJ"),
            first_target=waypoints[0],
            waypoint_count=len(waypoints),
            dt=d_t,
        )
        # speed_list = []

        # Execute each waypoint
        for waypoint in waypoints:
            start_t = self.rtde.rtde_c.initPeriod()
            self.rtde.servo_joint(waypoint, time=d_t, **kwargs)
            self.rtde.rtde_c.waitPeriod(start_t)
            # speed_list.append(self.get_ee_speed())

        # Stop servo
        self.rtde.rtde_c.servoStop()
        time.sleep(0.2)
        # # Debug: Plot the speed
        # plt.plot(speed_list)
        # plt.show()

    def execute_ee_waypoints(
        self,
        waypoints: list[list[float]],
        d_t: float = 0.008,
        to_rotvec: bool = False,
        *,
        label: str = "TCP trajectory",
        **kwargs,
    ):
        """Execute a trajectory"""
        waypoints = list(waypoints)
        # Convert waypoints to rotation vector pose
        if to_rotvec:
            waypoints = [self._quat_to_rotvec_pose(p) for p in waypoints]
        if not waypoints:
            raise ValueError("TCP trajectory requires at least one waypoint")
        self.print_motion_debug(
            label,
            "servoL trajectory",
            waypoints[-1],
            speed=kwargs.get("speed", "servoL"),
            acceleration=kwargs.get("acceleration", "servoL"),
            first_target=waypoints[0],
            waypoint_count=len(waypoints),
            dt=d_t,
        )
        # speed_list = []

        # Execute each waypoint
        for waypoint in waypoints:
            start_t = self.rtde.rtde_c.initPeriod()
            self.rtde.servo_tool(waypoint, time=d_t, **kwargs)
            self.rtde.rtde_c.waitPeriod(start_t)
            # speed_list.append(self.get_ee_speed())

        # Stop servo
        self.rtde.rtde_c.servoStop()
        time.sleep(0.2)
        # # Debug: Plot the speed
        # plt.plot(speed_list)
        # plt.show()

    def move_joint(
        self,
        joint_angles: list[float],
        *,
        label: str = "joint move",
        **kwargs,
    ):
        """Move the robot to a joint configuration"""
        self.print_motion_debug(
            label,
            "moveJ",
            joint_angles,
            speed=kwargs.get("speed", 1.05),
            acceleration=kwargs.get("acceleration", 1.4),
        )
        self.rtde.move_joint(joint_angles, **kwargs)

    def move_tool(
        self,
        tool_pose: list[float],
        to_rotvec: bool = False,
        *,
        label: str = "TCP move",
        **kwargs,
    ):
        """Move the robot to a tool pose"""
        if to_rotvec:
            tool_pose = self._quat_to_rotvec_pose(tool_pose)
        self.print_motion_debug(
            label,
            "moveL",
            tool_pose,
            speed=kwargs.get("speed", 0.25),
            acceleration=kwargs.get("acceleration", 1.2),
        )
        self.rtde.move_tool(tool_pose, **kwargs)

    def _quat_to_rotvec_pose(self, quat_pose: list[float]):
        """Convert a quaternion pose to a rotation vector pose"""
        # Quaternion in wxyz format, convert to xyzw format
        quat = np.roll(quat_pose[3:], -1)  # [w, x, y, z] → [x, y, z, w]
        rotvec = R.from_quat(quat).as_rotvec()
        return list(quat_pose[:3]) + list(rotvec)

    # Gripper
    def control_gripper(
        self,
        action: str,
        *,
        velocity_percent: float = 30.0,
        force_percent: float = 20.0,
        wait: bool = True,
    ):
        """Open or close the wrist-connected ROBOTIS RH-P12-RN."""

        return self.rtde.control_robotis_gripper(
            action,
            velocity_percent=velocity_percent,
            force_percent=force_percent,
            wait=wait,
        )

    # Camera
    def get_object_pose_hand(self):
        """Get the pose of the object from the in-hand camera."""
        return self.hand_cam.get_object_pose()

    # Getters
    def get_ee_pose(self):
        """Get the tool pose of the robot [x, y, z, rx, ry, rz]"""
        return self.rtde.get_tool_pose()

    def get_ee_transform(self):
        """Get the tool transform of the robot"""
        x, y, z, rx, ry, rz = self.get_ee_pose()
        transform = np.eye(4)
        transform[:3, 3] = [x, y, z]
        transform[:3, :3] = R.from_rotvec([rx, ry, rz]).as_matrix()
        return transform

    def get_ee_speed(self):
        """Get the speed of the robot [vx, vy, vz, wx, wy, wz]"""
        return self.rtde.get_tool_speed()

    def get_q_values(self):
        """Get the joint values of the robot"""
        return self.rtde.get_joint_values()
