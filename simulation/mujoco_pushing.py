import os
import numpy as np
import time
import threading
from pathlib import Path
import mujoco
import mujoco.viewer
from concurrent.futures import ThreadPoolExecutor, wait

from geometry.pose import Pose, euler_to_quat
from geometry.random_push import generate_path_form_params
from simulation.pushing_object_specs import CRACKER_BOX_FLIPPED_SHAPE
from simulation.mink_ik import UR10IK as IK

SIMULATION_ASSET_DIR = Path(__file__).resolve().parent / "assets"


class Sim:
    def __init__(
        self,
        xml_path=str(SIMULATION_ASSET_DIR / "mujoco_sim.xml"),
        n_envs=1,
        robot_joint_dof=6,
        robot_ee_dof=0,
        dt=0.02,  # Set to 0.02 to match Puna (was 0.002 in old sim)
        visualize=True,
        realtime_sync=True,  # Added for bridge compatibility
        viewer_sync_rate=1,  # Added for bridge compatibility
    ):
        """
        Mujoco Simulation Environment (Matched with Puna)
        """
        # Resolve path
        if not os.path.exists(xml_path) and os.path.exists(
            os.path.join(os.path.dirname(__file__), xml_path)
        ):
            xml_path = os.path.join(os.path.dirname(__file__), xml_path)

        self.xml_path = xml_path

        # Initialize Mujoco
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.vis.global_.offwidth = max(int(self.mj_model.vis.global_.offwidth), 1920)
        self.mj_model.vis.global_.offheight = max(int(self.mj_model.vis.global_.offheight), 1080)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.visualize = visualize
        self.realtime_sync = realtime_sync
        self.viewer_sync_rate = viewer_sync_rate

        # Simulation parameters
        self.robot_joint_dof = robot_joint_dof
        self.robot_ee_dof = robot_ee_dof
        self.robot_joint_idx = np.arange(robot_joint_dof)
        self.robot_ee_idx = np.arange(robot_joint_dof, robot_joint_dof + robot_ee_dof)
        obj_idx = np.arange(robot_joint_dof + robot_ee_dof, self.mj_model.nq)

        if obj_idx.size % 7 == 0:
            self.obj_idxs = obj_idx.reshape(-1, 7)
        else:
            print(
                "Warning: The first object has more than 7DOF. "
                + "The other objects will be ignored in computation."
            )
            self.obj_idxs = obj_idx[None, :7]

        # time
        self.dt = dt
        time_step = self.mj_model.opt.timestep
        self.n_substeps = int(self.dt // time_step)

        # Prepare parallelization (even if n_envs=1)
        self.n_envs = n_envs
        self.mj_datas = [mujoco.MjData(self.mj_model) for _ in range(n_envs)]
        self.mj_datas_qpos = [mj_data.qpos for mj_data in self.mj_datas]
        self.executor = ThreadPoolExecutor(max_workers=self.n_envs)

        # Store the initial state for reset
        self.init_qpos = np.array(self.mj_datas_qpos)

        # Setup IK
        ik_xml = str(SIMULATION_ASSET_DIR / "ur10_rod_ik.xml")
        self.ik = IK(ik_xml)

        # Bridge state
        self.reset_requested = False
        self.sim_lock = threading.Lock()  # For thread safety with network
        self.sim_step_count = 0
        self.params = []
        self.pos_waypoints = []
        self.durations = []
        self.ws_paths = []
        self.control_idx = 0
        self.stop_requested = False
        self.close_requested = False
        self.goal_region = None
        self.plan_path = []
        self.record_path = None
        self.record_fps = 24.0
        self.record_width = 1920
        self.record_height = 1080
        self.recorded_frames = []
        self._record_renderer = None
        self._record_last_time = -np.inf
        self.executed_controls = []
        self.plan_path_history = []
        self.fixed_camera_lookat = None
        self.fixed_camera_distance = None
        self.fixed_camera_azimuth = 180.0
        self.fixed_camera_elevation = -35.0
        self.pre_push_offset = 0.02
        self.push_height = None
        self.relative_push_offset = True

        # Parameters for bridge compatibility
        self.table_height = 0.0
        self.obj_shape = CRACKER_BOX_FLIPPED_SHAPE.copy()
        self.tool_offset = np.array([0, 0, 0.0, 1, 0, 0, 0])  # Pose as flat array [x,y,z,w,x,y,z]

    def get_sim_info(self):
        """Return simulation infomation"""
        return (self.n_envs, self.dt)

    ########## Parallel Simulation Core ##########
    def run_sim(self, duration, ctrl=None, thread_fn=None):
        """Run the simulation for a given duration with optional function"""
        if duration <= 0:
            return
        n_steps = int(duration // self.dt)
        self.step_n(n_steps, ctrl, thread_fn)

    def _step_n_thread(self, thread_i, n_steps, mj_model, mj_data, ctrl=None, thread_fn=None):
        """Step the simulation for one thread"""
        for j in range(n_steps):
            if ctrl is not None:
                if ctrl.ndim == 3:  # (steps, n_envs, dof)
                    # Check if thread_i is within bounds of ctrl
                    if thread_i < ctrl.shape[1]:
                        mj_data.ctrl[:] = ctrl[j, thread_i]
                    else:
                        # Fallback or error? Assuming broadcast if 1
                        mj_data.ctrl[:] = ctrl[j, 0]
                else:
                    mj_data.ctrl[:] = ctrl[j]
            if thread_fn is not None:
                thread_fn(thread_i, j, mj_model, mj_data)
            for _ in range(self.n_substeps):
                mujoco.mj_step(mj_model, mj_data)

    def step_n(self, n_steps, ctrl=None, thread_fn=None):
        """Step the simulation n times with optional function"""

        def vis_thread_fn(thread_i, j, mj_model, mj_data):
            if thread_fn is not None:
                thread_fn(thread_i, j, mj_model, mj_data)

        fn = vis_thread_fn
        futures = [
            self.executor.submit(
                self._step_n_thread,
                i,
                n_steps,
                self.mj_model,
                self.mj_datas[i],
                ctrl,
                fn,
            )
            for i in range(self.n_envs)
        ]
        wait(futures)

    def reset(self, wait_time=0.5):
        """Reset the simulation to the initial state"""
        self.stop()
        self.reset_requested = True

        zero_vel = np.zeros(self.mj_model.nv)
        with self.sim_lock:
            for i, mj_data in enumerate(self.mj_datas):
                mujoco.mj_resetData(self.mj_model, mj_data)
                mj_data.qpos[:] = self.init_qpos[i]
                mj_data.qvel[:] = zero_vel
                mj_data.ctrl[:] = self.init_qpos[i, : self.robot_joint_dof + self.robot_ee_dof]

            # Reset control tracking
            self.control_idx = 0
            self.params = []
            self.pos_waypoints = []
            self.durations = []
            self.ws_paths = []
            self.plan_path_history = []
            self.stop_requested = False

        # Stabilize
        # Don't call run_sim because it creates a thread conflict with the viewer loop
        # Instead, simply sleep and let the viewer loop (which is running mj_step) stabilize the system
        time.sleep(wait_time)
        self.reset_requested = False
        return True

    def stop(self):
        """Stop current execution"""
        self.stop_requested = True
        with self.sim_lock:
            self.params = []
            self.pos_waypoints = []
            self.durations = []
            self.ws_paths = []
            self.control_idx = 0
        return True

    def close(self):
        """Close the simulation"""
        self.close_requested = True
        self.stop()
        self.executor.shutdown(wait=True)

    ########## Object-related functions ##########
    def set_obj_init_poses(self, init_pose, obj_idx=0, env_idx=None):
        """Set the initial poses of the objects"""
        init_pose, env_idx = self._preprocess_values(init_pose, env_idx)
        self.init_qpos[np.ix_(env_idx, self.obj_idxs[obj_idx])] = init_pose

    def get_obj_pose(self, obj_idx=0, env_idx=None):
        """Return object information"""
        _, env_idx = self._preprocess_values(np.zeros((self.n_envs, 1)), env_idx)
        return np.array(self.mj_datas_qpos)[np.ix_(env_idx, self.obj_idxs[obj_idx])]

    # Bridge compatibility aliases
    def set_obj_init_pose(self, pose):
        # pose can be list/array [x, y, theta] or Pose object
        if hasattr(pose, "position"):  # Pose object
            flat = pose.flat()
        elif len(pose) == 3:  # [x, y, theta]
            x, y, theta = pose
            pos = np.array([x, y, self.table_height + self.obj_shape[2] / 2])
            quat = euler_to_quat([0, 0, theta])
            flat = np.concatenate([pos, quat])
        else:
            flat = np.array(pose)

        self.set_obj_init_poses(flat, env_idx=0)

        # Apply immediately
        with self.sim_lock:
            self.mj_datas[0].qpos[self.obj_idxs[0]] = flat
            self.mj_datas[0].qvel[:] = 0
            mujoco.mj_forward(self.mj_model, self.mj_datas[0])

    def get_state(self) -> np.ndarray:
        # Returns [x, y, theta] for compatibility
        pose_7d = self.get_obj_pose(env_idx=0)[0]  # (7,)
        pos = pose_7d[:3]
        quat = pose_7d[3:]
        # Convert quat to yaw
        from geometry.pose import quat_to_euler

        euler = quat_to_euler(quat)
        return np.array([pos[0], pos[1], euler[2]])

    def _draw_goal_region(self, viewer):
        if self.goal_region is None or not hasattr(viewer, "user_scn"):
            return
        self._add_goal_region_to_scene(viewer.user_scn)

    def set_fixed_camera(
        self,
        lookat,
        *,
        distance: float = 1.3,
        azimuth: float = 180.0,
        elevation: float = -35.0,
    ):
        self.fixed_camera_lookat = np.asarray(lookat, dtype=float).reshape(-1)[:3]
        self.fixed_camera_distance = float(distance)
        self.fixed_camera_azimuth = float(azimuth)
        self.fixed_camera_elevation = float(elevation)

    def _sync_fixed_camera(self, viewer):
        if self.fixed_camera_lookat is None:
            return
        viewer.cam.lookat[:] = np.asarray(self.fixed_camera_lookat, dtype=float)
        viewer.cam.distance = float(self.fixed_camera_distance or 1.3)
        viewer.cam.azimuth = float(self.fixed_camera_azimuth)
        viewer.cam.elevation = float(self.fixed_camera_elevation)

    def set_plan_path(self, states):
        path = []
        if states is not None:
            for state in states:
                arr = np.asarray(state, dtype=float).reshape(-1)
                if arr.size >= 2:
                    path.append(arr[:3].copy())
        with self.sim_lock:
            self.plan_path = path
            if self.record_path:
                stage = len(self.executed_controls)
                snapshot = [np.asarray(s, dtype=float).copy() for s in path]
                if self.plan_path_history and self.plan_path_history[-1][0] == stage:
                    self.plan_path_history[-1] = (stage, snapshot)
                else:
                    self.plan_path_history.append((stage, snapshot))

    def _draw_plan_path(self, viewer):
        if not hasattr(viewer, "user_scn"):
            return
        self._add_plan_path_to_scene(viewer.user_scn)

    def _add_plan_path_to_scene(self, scene, plan_path=None):
        if plan_path is None:
            with self.sim_lock:
                path = [np.asarray(s, dtype=float).copy() for s in self.plan_path]
        else:
            path = plan_path
        if path is None or len(path) < 2:
            return
        color = np.array([0.02, 0.25, 0.78, 0.92], dtype=float)
        z = float(self.table_height) + 0.035
        for start, end in zip(path[:-1], path[1:]):
            if scene.ngeom >= scene.maxgeom:
                return
            a = np.asarray(start, dtype=float).reshape(-1)
            b = np.asarray(end, dtype=float).reshape(-1)
            if a.size < 2 or b.size < 2:
                continue
            from_pt = np.array([a[0], a[1], z], dtype=float)
            to_pt = np.array([b[0], b[1], z], dtype=float)
            if np.linalg.norm(to_pt - from_pt) < 1e-9:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.012,
                from_pt,
                to_pt,
            )
            geom.rgba[:] = color
            scene.ngeom += 1

    def _add_goal_region_to_scene(self, scene):
        if self.goal_region is None:
            return
        x, y, radius = self.goal_region
        segments = 48
        half_arc = np.pi * float(radius) / float(segments)
        for i in range(segments):
            if scene.ngeom >= scene.maxgeom:
                return
            angle = 2.0 * np.pi * float(i) / float(segments)
            geom = scene.geoms[scene.ngeom]
            mat = np.array(
                [
                    [np.cos(angle + np.pi / 2.0), -np.sin(angle + np.pi / 2.0), 0.0],
                    [np.sin(angle + np.pi / 2.0), np.cos(angle + np.pi / 2.0), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_BOX,
                np.array([half_arc, 0.008, 0.003], dtype=float),
                np.array([x + radius * np.cos(angle), y + radius * np.sin(angle), self.table_height + 0.008], dtype=float),
                mat.reshape(-1),
                np.array([0.05, 0.85, 0.12, 0.85], dtype=float),
            )
            scene.ngeom += 1

    def start_recording(self, path, fps=24.0, width=1600, height=1000):
        self.record_path = path
        self.record_fps = float(fps)
        self.record_width = min(int(width), int(self.mj_model.vis.global_.offwidth))
        self.record_height = min(int(height), int(self.mj_model.vis.global_.offheight))
        self.recorded_frames = []
        self._record_renderer = None
        self._record_last_time = -np.inf
        self.executed_controls = []
        self.plan_path_history = []

    def _snapshot_from_data(self, mj_data, plan_path=None):
        obj_qpos = mj_data.qpos[self.obj_idxs[0]]
        if self.fixed_camera_lookat is None:
            lookat = np.array(
                [float(obj_qpos[0]), float(obj_qpos[1]), 0.18], dtype=float
            )
        else:
            lookat = np.asarray(self.fixed_camera_lookat, dtype=float).copy()
        if plan_path is None:
            plan_path = self.plan_path
        return (
            mj_data.qpos.copy(),
            mj_data.qvel.copy(),
            mj_data.ctrl.copy(),
            lookat,
            [np.asarray(s, dtype=float).copy() for s in plan_path],
        )

    def _record_frame(self):
        if not self.record_path:
            return
        period = 1.0 / max(float(self.record_fps), 1e-6)
        if float(self.mj_data.time) - self._record_last_time < period:
            return
        with self.sim_lock:
            snapshot = self._snapshot_from_data(self.mj_data)
            sim_time = float(self.mj_data.time)
        self.recorded_frames.append(snapshot)
        self._record_last_time = sim_time

    def _build_replay_recording(self):
        if not self.executed_controls:
            return []

        history = [
            (int(stage), [np.asarray(s, dtype=float).copy() for s in path])
            for stage, path in self.plan_path_history
        ]

        def path_for_stage(stage):
            selected = None
            for hist_stage, hist_path in history:
                if hist_stage <= stage:
                    selected = hist_path
                else:
                    break
            if selected is None and history:
                selected = history[0][1]
            if selected is None:
                selected = self.plan_path
            return [np.asarray(s, dtype=float).copy() for s in selected]

        replay_data = mujoco.MjData(self.mj_model)
        replay_data.qpos[:] = np.asarray(self.init_qpos[0], dtype=float)
        replay_data.qvel[:] = 0.0
        replay_data.ctrl[: self.robot_joint_dof + self.robot_ee_dof] = replay_data.qpos[
            : self.robot_joint_dof + self.robot_ee_dof
        ]
        mujoco.mj_forward(self.mj_model, replay_data)

        snapshots = [self._snapshot_from_data(replay_data, path_for_stage(0))]
        period = 1.0 / max(float(self.record_fps), 1e-6)
        last_time = float(replay_data.time)

        for control_stage, (push_params, total_time) in enumerate(self.executed_controls):
            replay_plan_path = path_for_stage(control_stage)
            obj_state = replay_data.qpos[self.obj_idxs[0]].copy()
            ws_path = self.generate_ws_path(
                push_params,
                total_time,
                obj_state=obj_state,
            )
            waypoints = self.compute_waypoints_for_path(total_time, ws_path)
            if len(waypoints) == 0:
                continue

            replay_data.qpos[self.robot_joint_idx] = waypoints[0]
            replay_data.ctrl[self.robot_joint_idx] = waypoints[0]
            mujoco.mj_forward(self.mj_model, replay_data)

            for waypoint in waypoints:
                replay_data.ctrl[self.robot_joint_idx] = waypoint
                for _ in range(self.n_substeps):
                    mujoco.mj_step(self.mj_model, replay_data)
                if float(replay_data.time) - last_time >= period:
                    snapshots.append(
                        self._snapshot_from_data(replay_data, replay_plan_path)
                    )
                    last_time = float(replay_data.time)

        if len(snapshots) == 1:
            snapshots.append(
                self._snapshot_from_data(
                    replay_data,
                    path_for_stage(len(self.executed_controls)),
                )
            )
        return snapshots

    def save_recording(self):
        if not self.record_path:
            return
        if not self.recorded_frames and not self.executed_controls:
            return
        from simulation.mujoco_video_renderer import save_recording_in_subprocess

        snapshots = list(self.recorded_frames)
        expected_min_frames = 2
        if self.executed_controls:
            expected_duration = sum(float(duration) for _, duration in self.executed_controls)
            expected_min_frames = max(
                expected_min_frames,
                int(0.5 * float(self.record_fps) * expected_duration),
            )
        if len(snapshots) < expected_min_frames and self.executed_controls:
            print(
                "[mujoco video] live pushing capture had too few frames; "
                "replaying controls offscreen",
                flush=True,
            )
            replay_snapshots = self._build_replay_recording()
            if len(replay_snapshots) > len(snapshots):
                snapshots = replay_snapshots
        print(
            f"[mujoco video] rendering {len(snapshots)} frames -> {self.record_path}",
            flush=True,
        )
        camera_config = None
        if self.fixed_camera_lookat is not None:
            camera_config = self._recording_camera_config(snapshots)
        saved = save_recording_in_subprocess(
            kind="pushing",
            xml_path=self.xml_path,
            snapshots=snapshots,
            output_path=self.record_path,
            fps=float(self.record_fps),
            width=int(self.record_width),
            height=int(self.record_height),
            goal_region=self.goal_region,
            table_height=float(self.table_height),
            camera_config=camera_config,
        )
        if saved:
            print(f"[mujoco video] saved {self.record_path}")
        else:
            print(
                "[WARNING] MuJoCo video renderer subprocess failed; "
                "AURA execution results are still valid."
            )
        self.recorded_frames = []
        self.record_path = None
        self._record_renderer = None
        self.executed_controls = []
        self.plan_path_history = []

    def _recording_camera_config(self, snapshots):
        points = []
        for snapshot in snapshots:
            if len(snapshot) < 5:
                continue
            qpos, _qvel, _ctrl, _lookat, plan_path = snapshot
            qpos = np.asarray(qpos, dtype=float).reshape(-1)
            if qpos.size >= self.obj_idxs[0][-1] + 1:
                obj_qpos = qpos[self.obj_idxs[0]]
                if obj_qpos.size >= 2:
                    points.append(obj_qpos[:2])
            for state in plan_path or []:
                arr = np.asarray(state, dtype=float).reshape(-1)
                if arr.size >= 2:
                    points.append(arr[:2])

        if self.goal_region is not None:
            gx, gy, radius = self.goal_region
            points.extend(
                [
                    np.array([gx - radius, gy - radius], dtype=float),
                    np.array([gx + radius, gy + radius], dtype=float),
                ]
            )

        if points:
            points_np = np.asarray(points, dtype=float).reshape(-1, 2)
            xy_min = points_np.min(axis=0)
            xy_max = points_np.max(axis=0)
            center_xy = 0.5 * (xy_min + xy_max)
            span = np.maximum(xy_max - xy_min, 1e-6)
            fit_distance = max(
                1.95,
                2.70 * float(np.max(span)),
                2.00 * float(np.linalg.norm(span)),
            )
            lookat = np.array([center_xy[0], center_xy[1], 0.12], dtype=float)
        else:
            fit_distance = 1.95
            lookat = np.asarray(self.fixed_camera_lookat, dtype=float).reshape(-1)[:3]

        return {
            "lookat": lookat.tolist(),
            "distance": max(float(self.fixed_camera_distance or 1.3), fit_distance),
            "azimuth": float(self.fixed_camera_azimuth),
            "elevation": float(self.fixed_camera_elevation),
        }

    def get_object_pose(self) -> Pose:
        # Returns Pose object
        pose_7d = self.get_obj_pose(env_idx=0)[0]
        return Pose(pose_7d[:3], pose_7d[3:])

    ########## Robot-related functions ##########
    def set_robot_init_joints(self, joints, ee_joints=None, env_idx=None):
        """Set the initial joint positions"""
        joints, env_idx = self._preprocess_values(joints, env_idx)
        self.init_qpos[np.ix_(env_idx, self.robot_joint_idx)] = joints
        if ee_joints is not None:
            ee_joints, env_idx = self._preprocess_values(ee_joints, env_idx)
            self.init_qpos[np.ix_(env_idx, self.robot_ee_idx)] = ee_joints

    def get_robot_joints(self, env_idx=None):
        """Get the robot joint positions"""
        _, env_idx = self._preprocess_values(np.zeros((self.n_envs, 1)), env_idx)
        return np.array(self.mj_datas_qpos)[np.ix_(env_idx, self.robot_joint_idx)]

    def set_arm_qpos(self, joint_angles):
        # Bridge compatibility
        self.set_robot_init_joints(joint_angles, env_idx=0)
        # Apply immediately to current state too
        with self.sim_lock:
            self.mj_datas[0].qpos[self.robot_joint_idx] = joint_angles
            mujoco.mj_forward(self.mj_model, self.mj_datas[0])

    def set_arm_position(self, joint_angles):
        # Bridge compatibility - set control
        with self.sim_lock:
            self.mj_datas[0].ctrl[self.robot_joint_idx] = joint_angles

    ########## Helper functions ##########
    def _preprocess_values(self, values, env_idx):
        """Preprocess the values and env_idx to match"""
        values = np.asarray(values)
        if env_idx is None:
            size = 1 if values.ndim == 1 else len(values)
            env_idx = np.arange(size)
        if isinstance(env_idx, int):
            env_idx = np.array([env_idx])
        env_idx = np.array(env_idx)
        if values.ndim == 1:
            values = np.tile(values, (len(env_idx), 1))
        return values, env_idx

    ########## Path Generation & Execution (from Puna) ##########
    def _canonical_push_params(self, push_params):
        push_params = np.asarray(push_params, dtype=float).reshape(-1).copy()
        if push_params.size < 3:
            raise ValueError(f"Push params must have 3 values, got {push_params}")

        face_raw = float(push_params[0])
        rad_faces = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
        if 0.0 <= face_raw <= 0.75:
            face_idx = int(np.round(face_raw * 4.0)) % 4
        elif abs(face_raw - round(face_raw)) < 1e-9 and 0 <= round(face_raw) <= 3:
            face_idx = int(round(face_raw)) % 4
        elif np.min(np.abs(face_raw - rad_faces)) < 1e-6:
            face_idx = int(np.argmin(np.abs(face_raw - rad_faces)))
        elif 0.0 <= face_raw < 4.0:
            face_idx = int(face_raw) % 4
        else:
            face_idx = int(face_raw / (np.pi / 2.0)) % 4

        rotation = face_idx / 4.0
        side = float(np.clip(push_params[1], -0.4, 0.4))
        distance = float(np.clip(push_params[2], 0.0, 0.30))
        return np.asarray([rotation, side, distance], dtype=float)

    def generate_ws_path(
        self,
        push_params,
        total_time=2.0,
        dt=None,
        max_speed=0.5,
        max_acc=1.0,
        relative_push_offset=True,
        obj_state=None,
    ):
        if dt is None:
            dt = self.dt

        if obj_state is None:
            obj_state = self.get_obj_pose(env_idx=0)[0]
        obj_state = np.asarray(obj_state, dtype=float).reshape(-1)
        obj_pose = Pose(obj_state[:3], obj_state[3:])
        tool_offset = Pose(self.tool_offset[:3], self.tool_offset[3:])
        path_params = self._canonical_push_params(push_params)
        _times, ws_path = generate_path_form_params(
            obj_pose,
            self.obj_shape,
            path_params,
            tool_offset=tool_offset,
            total_time=total_time,
            dt=dt,
            max_speed=max_speed,
            max_acc=max_acc,
            pre_push_offset=float(self.pre_push_offset),
            push_height=self.push_height,
            relative_push_offset=relative_push_offset,
        )
        return ws_path

    def ws_path_to_traj(self, robot_base_pose, t_path, ws_path):
        traj = self.ik.ws_path_to_traj(t_path, ws_path)
        return traj

    def compute_waypoints_for_path(self, total_time, ws_path):
        n_points = len(ws_path)
        times = np.linspace(0, total_time, n_points)

        traj = self.ws_path_to_traj(None, times, ws_path)
        # `to_step_waypoints` returns (position, velocity, acceleration) with
        # shape (3, steps, dof). The viewer expects joint positions only.
        traj_waypoints = traj.to_step_waypoints(self.dt)
        return traj_waypoints[0]

    def get_final_waypoints(self, push_params, total_time, ws_path):
        waypoints = self.compute_waypoints_for_path(total_time, ws_path)

        self.params.append(push_params)
        self.pos_waypoints.append(waypoints[None, :, :])

        self.durations.append(total_time)
        self.ws_paths.append(ws_path)

        target_idx = len(self.params)
        timeout = total_time + 5.0
        start_time = time.time()
        while self.control_idx < target_idx:
            if time.time() - start_time > timeout:
                print(f"Timeout waiting for execution")
                break
            time.sleep(0.05)

        return waypoints

    def execute_waypoints(self, waypoints):
        # waypoints: (steps, n_envs, dof)
        waypoints = np.array(waypoints)
        if waypoints.ndim == 2:
            # (steps, dof) -> (steps, 1, dof)
            waypoints = waypoints[:, None, :]

        n_steps = len(waypoints)

        # We assume n_envs matches or we tile?
        # If waypoints has 1 env but we have N, we might want to broadcast?
        # But _step_n_thread logic below handles it.

        # Execute
        self.step_n(n_steps, ctrl=waypoints)

        # Return final SE2 states for all envs
        # Returns (n_envs, 3)
        states = []
        for i in range(self.n_envs):
            pose_7d = self.get_obj_pose(env_idx=i)[0]
            pos = pose_7d[:3]
            quat = pose_7d[3:]
            # Convert quat to yaw
            from geometry.pose import quat_to_euler

            euler = quat_to_euler(quat)
            states.append([pos[0], pos[1], euler[2]])

        return np.array(states)

    def execute_segment(self, push_params, total_time):
        self.executed_controls.append(
            (np.asarray(push_params, dtype=float).reshape(-1).copy(), float(total_time))
        )
        ws_path = self.generate_ws_path(push_params, total_time)
        self.get_final_waypoints(push_params, total_time, ws_path)
        return self.get_state()

    def run_viewer(self):
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running() and not self.close_requested:
                if self.reset_requested:
                    pass

                if self.params and self.control_idx < len(self.params):
                    current_waypoints = self.pos_waypoints[self.control_idx][0]

                    with self.sim_lock:
                        self.mj_datas[0].qpos[self.robot_joint_idx] = current_waypoints[0]
                        self.mj_datas[0].ctrl[self.robot_joint_idx] = current_waypoints[0]
                        mujoco.mj_forward(self.mj_model, self.mj_datas[0])

                    for waypoint in current_waypoints:
                        if not viewer.is_running() or self.close_requested:
                            break

                        if self.stop_requested:
                            break

                        step_start = time.time()
                        with self.sim_lock:
                            self.mj_datas[0].ctrl[self.robot_joint_idx] = waypoint

                        for _ in range(self.n_substeps):
                            mujoco.mj_step(self.mj_model, self.mj_datas[0])

                        self.mj_data.qpos[:] = self.mj_datas[0].qpos
                        self.mj_data.ctrl[:] = self.mj_datas[0].ctrl
                        mujoco.mj_forward(self.mj_model, self.mj_data)
                        if hasattr(viewer, "user_scn"):
                            viewer.user_scn.ngeom = 0
                        self._draw_goal_region(viewer)
                        self._draw_plan_path(viewer)
                        self._record_frame()
                        self._sync_fixed_camera(viewer)
                        viewer.sync()

                        if self.realtime_sync:
                            elapsed = time.time() - step_start
                            if elapsed < self.dt:
                                time.sleep(self.dt - elapsed)

                    if self.stop_requested:
                        self.stop_requested = False
                        continue

                    self.control_idx += 1
                else:
                    step_start = time.time()
                    with self.sim_lock:
                        mujoco.mj_step(self.mj_model, self.mj_datas[0])

                    self.mj_data.qpos[:] = self.mj_datas[0].qpos
                    mujoco.mj_forward(self.mj_model, self.mj_data)
                    if hasattr(viewer, "user_scn"):
                        viewer.user_scn.ngeom = 0
                    self._draw_goal_region(viewer)
                    self._draw_plan_path(viewer)
                    self._record_frame()
                    self._sync_fixed_camera(viewer)
                    viewer.sync()

                    if self.realtime_sync:
                        elapsed = time.time() - step_start
                        if elapsed < self.dt:
                            time.sleep(self.dt - elapsed)
