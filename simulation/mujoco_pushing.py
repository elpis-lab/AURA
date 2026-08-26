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
from simulation.inverse_kinematics import UR10InverseKinematics
from simulation.pushing_model import CRACKER_BOX_FLIPPED_SHAPE

SIMULATION_ASSET_DIR = Path(__file__).resolve().parent / "assets"


class MujocoPushingSimulator:
    """Low-level MuJoCo execution backend for UR10 object pushing."""

    def __init__(
        self,
        xml_path=str(SIMULATION_ASSET_DIR / "mujoco_sim.xml"),
        n_envs=1,
        robot_joint_dof=6,
        robot_ee_dof=0,
        dt=0.02,  # Set to 0.02 to match Puna (was 0.002 in old sim)
        realtime_sync=True,
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

        self.realtime_sync = realtime_sync

        # Simulation parameters
        self.robot_joint_dof = robot_joint_dof
        self.robot_ee_dof = robot_ee_dof
        self.robot_joint_idx = np.arange(robot_joint_dof)
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
        self.ik = UR10InverseKinematics(ik_xml)

        # Bridge state
        self.reset_requested = False
        self.sim_lock = threading.Lock()  # For thread safety with network
        self.params = []
        self.pos_waypoints = []
        self.control_idx = 0
        self.stop_requested = False
        self.close_requested = False
        self.pre_push_offset = 0.02
        self.push_height = None
        self.table_height = 0.0
        self.obj_shape = CRACKER_BOX_FLIPPED_SHAPE.copy()
        self.tool_offset = np.array([0, 0, 0.0, 1, 0, 0, 0])  # Pose as flat array [x,y,z,w,x,y,z]

    ########## Parallel Simulation Core ##########
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
            self.control_idx = 0
        return True

    def close(self):
        """Close the simulation"""
        self.close_requested = True
        self.stop()
        self.executor.shutdown(wait=True)

    ########## Object-related functions ##########
    def set_initial_object_poses(self, init_pose, obj_idx=0, env_idx=None):
        """Set the initial poses of the objects"""
        init_pose, env_idx = self._preprocess_values(init_pose, env_idx)
        self.init_qpos[np.ix_(env_idx, self.obj_idxs[obj_idx])] = init_pose

    def get_object_poses(self, obj_idx=0, env_idx=None):
        """Return object information"""
        _, env_idx = self._preprocess_values(np.zeros((self.n_envs, 1)), env_idx)
        return np.array(self.mj_datas_qpos)[np.ix_(env_idx, self.obj_idxs[obj_idx])]

    def set_state(self, pose):
        """Set the object's SE(2) state or full seven-value MuJoCo pose."""
        if hasattr(pose, "position"):  # Pose object
            flat = pose.flat()
        elif len(pose) == 3:  # [x, y, theta]
            x, y, theta = pose
            pos = np.array([x, y, self.table_height + self.obj_shape[2] / 2])
            quat = euler_to_quat([0, 0, theta])
            flat = np.concatenate([pos, quat])
        else:
            flat = np.array(pose)

        self.set_initial_object_poses(flat, env_idx=0)

        # Apply immediately
        with self.sim_lock:
            self.mj_datas[0].qpos[self.obj_idxs[0]] = flat
            self.mj_datas[0].qvel[:] = 0
            mujoco.mj_forward(self.mj_model, self.mj_datas[0])
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Return the object's current ``[x, y, yaw]`` state."""
        pose_7d = self.get_object_poses(env_idx=0)[0]  # (7,)
        pos = pose_7d[:3]
        quat = pose_7d[3:]
        # Convert quat to yaw
        from geometry.pose import quat_to_euler

        euler = quat_to_euler(quat)
        return np.array([pos[0], pos[1], euler[2]])

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
            obj_state = self.get_object_poses(env_idx=0)[0]
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

    def workspace_path_to_trajectory(self, t_path, ws_path):
        return self.ik.workspace_path_to_trajectory(t_path, ws_path)

    def compute_waypoints_for_path(self, total_time, ws_path):
        n_points = len(ws_path)
        times = np.linspace(0, total_time, n_points)

        traj = self.workspace_path_to_trajectory(times, ws_path)
        # `to_step_waypoints` returns (position, velocity, acceleration) with
        # shape (3, steps, dof). The viewer expects joint positions only.
        traj_waypoints = traj.to_step_waypoints(self.dt)
        return traj_waypoints[0]

    def get_final_waypoints(self, push_params, total_time, ws_path):
        waypoints = self.compute_waypoints_for_path(total_time, ws_path)

        self.params.append(push_params)
        self.pos_waypoints.append(waypoints[None, :, :])

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
        if n_steps == 0:
            return np.asarray([self.get_state()], dtype=float)

        # Match the interactive viewer path: reposition the arm at the first
        # pre-push waypoint before advancing the push trajectory. Without this
        # initialization, a headless run applies a large joint-space transient
        # from reset and can launch the object before contact is established.
        with self.sim_lock:
            for environment_index, mj_data in enumerate(self.mj_datas):
                waypoint_environment = min(
                    environment_index, waypoints.shape[1] - 1
                )
                first = waypoints[0, waypoint_environment]
                mj_data.qpos[self.robot_joint_idx] = first
                mj_data.qvel[self.robot_joint_idx] = 0.0
                mj_data.ctrl[self.robot_joint_idx] = first
                mujoco.mj_forward(self.mj_model, mj_data)

        # Execute
        self.step_n(n_steps, ctrl=waypoints)

        # Return final SE2 states for all envs
        # Returns (n_envs, 3)
        states = []
        for i in range(self.n_envs):
            pose_7d = self.get_object_poses(env_idx=i)[0]
            pos = pose_7d[:3]
            quat = pose_7d[3:]
            # Convert quat to yaw
            from geometry.pose import quat_to_euler

            euler = quat_to_euler(quat)
            states.append([pos[0], pos[1], euler[2]])

        return np.array(states)

    def execute_segment(self, push_params, total_time):
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
                    viewer.sync()

                    if self.realtime_sync:
                        elapsed = time.time() - step_start
                        if elapsed < self.dt:
                            time.sleep(self.dt - elapsed)
