import os, sys
import numpy as np
import time
import threading
import mujoco
import mujoco.viewer
from concurrent.futures import ThreadPoolExecutor, wait

from geometry.pose import Pose, matrix_to_quat, flat_to_matrix, euler_to_quat, quat_to_matrix
from ik import UR10IK as IK


# Define local helper for euler_to_matrix since it's not in pose.py
def euler_to_matrix(seq: str, euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles to rotation matrix"""
    quat = euler_to_quat(euler, seq)
    matrix = quat_to_matrix(quat)
    return matrix


class Sim:
    def __init__(
        self,
        xml_path="assets/mujoco_sim.xml",
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

        # Initialize Mujoco
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
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
        ik_xml = "assets/ur10_rod_ik.xml"
        if not os.path.exists(ik_xml) and os.path.exists(
            os.path.join(os.path.dirname(__file__), ik_xml)
        ):
            ik_xml = os.path.join(os.path.dirname(__file__), ik_xml)
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

        # Parameters for bridge compatibility
        self.table_height = 0.0
        self.obj_shape = np.array([0.1628, 0.2139, 0.0676])  # Cracker box
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
    def generate_ws_path(
        self,
        push_params,
        total_time=2.0,
        dt=None,
        max_speed=0.5,
        max_acc=1.0,
        relative_push_offset=False,
    ):
        if dt is None:
            dt = self.dt

        push_params = np.array(push_params)
        if push_params.ndim == 1:
            push_params = push_params[None, :]

        obj_states = self.get_obj_pose(env_idx=0)  # Returns (1, 7)
        n_data = 1

        rotations, sides, distances = push_params.T

        # Convert the normalized rotation to the absolute value
        # Input rotations are [0.0, 0.25, 0.5, 0.75]
        push_sides = np.round(rotations * 4)
        rotations = push_sides * (np.pi / 2)

        w, l, h = self.obj_shape
        mask_odd = push_sides % 2 == 1
        sizes = np.where(mask_odd, l, w)
        # sides is ALREADY absolute offset, so no need to multiply by side length if not relative
        # Puna's random_push.py generate_push_params returns RELATIVE offset if not multiplied
        # BUT Puna's generate_path_from_params takes absolute or relative?
        # Puna's generate_path_from_params:
        # sides = np.where(mask_odd, w * sides, l * sides)
        # This implies 'sides' input to that function is RELATIVE (-0.5 to 0.5)

        # However, check Aura's usage.
        # Aura's pushing/geometry/random_push.py:
        # side = np.random.uniform(*side_range) * side_size  <-- ABSOLUTE!
        # Aura's generate_path_form_params takes absolute side unless relative_push_offset=True
        # BUT `generate_path_from_params` in `pushing/sim.py` was copied from Puna which expects RELATIVE!

        # FIX: Check if sides are absolute or relative.
        # Aura generates ABSOLUTE offsets (e.g. 0.02m).
        # Puna generates RELATIVE offsets (e.g. 0.1 * width).

        # If we use Puna's logic: sides = np.where(mask_odd, w * sides, l * sides)
        # We are multiplying an absolute offset (e.g. 0.05) by width (e.g. 0.2) -> 0.01 (tiny!)

        # If input 'sides' is already absolute (from Aura planner), we should NOT multiply by size.
        # We should just assign sides = sides (maybe check bounds)

        # Correct logic for ABSOLUTE side offset input:
        # sides = sides (already absolute)
        # pass  # Do nothing to sides if they are absolute

        # However, to match Puna structure we need to see what `generate_path_from_params` does.
        # In Puna it does multiply.
        # In `pushing/sim.py` I copied Puna's logic.
        # BUT the input `push_params` comes from Aura planner which uses `pushing/geometry/random_push.py`.
        # `pushing/geometry/random_push.py` line 56: `side = np.random.uniform(*side_range) * side_size` -> ABSOLUTE.

        # So `push_params` has ABSOLUTE side offset.
        # `pushing/sim.py` `generate_ws_path` was doing `sides = np.where(mask_odd, w * sides, l * sides)`
        # This is WRONG if sides is absolute. It treats 0.05m as 5% if width is 1.0, or 0.05 * 0.2 = 0.01m.

        # FIX: Remove the multiplication if input is absolute.
        # sizes = np.where(mask_odd, l, w)
        # sides = sides # Assume absolute

        # Let's verify `relative_push_offset` flag usage.
        # If relative_push_offset is False (default), we should treat sides as absolute?
        # Puna code:
        # sides = np.where(mask_odd, w * sides, l * sides)
        # This UNCONDITIONALLY multiplies. It assumes input is relative.

        # In `pushing/sim.py` I should change this to handle absolute input correctly.
        # if relative_push_offset:
        #     sides = np.where(mask_odd, w * sides, l * sides)
        # else:
        #     sides = sides  # Absolute

        # Get local path (x, y) w.r.t. the object
        dir_vecs = np.stack([np.cos(rotations), np.sin(rotations)], axis=1)
        side_vecs = np.stack([-dir_vecs[:, 1], dir_vecs[:, 0]], axis=1)

        # pre_push_offset = 0.02 # Puna
        # pre_push_offset = 0.06 # Old sim / Aura
        # Match Aura/Old sim for consistency with planner?
        # Planner uses 0.06 (line 98 in pushing/geometry/random_push.py).
        # We MUST match the planner.
        pre_push_offset = 0.06

        distances = distances + pre_push_offset
        starts = dir_vecs * (sizes / 2 + pre_push_offset)[:, None] + sides[:, None] * side_vecs

        # Velocity profile
        peak_speed = 2 * distances / total_time
        peak_acc = peak_speed * np.pi / total_time

        n_steps = int(total_time / dt)
        t_paths = np.tile(np.linspace(0, total_time, n_steps), (n_data, 1))

        scale = -peak_speed * total_time / 4 / np.pi
        dists = scale[:, None] * np.sin(2 * np.pi * t_paths / total_time) + (
            peak_speed[:, None] * t_paths / 2
        )

        local_xy = starts[:, None, :] - dists[:, :, None] * dir_vecs[:, None, :]
        local_z = np.zeros((n_data, n_steps, 1))
        local_pos = np.concatenate([local_xy, local_z], axis=2)

        # Rotation
        t_rotate_z = np.tile(np.eye(4)[None, None, :, :], (n_data, 1, 1, 1))
        t_rotate_z[:, 0, :3, :3] = euler_to_matrix("z", rotations + np.pi)
        t_reflect_z = np.eye(4)[None, None, :, :]
        t_reflect_z[0, 0, :3, :3] = euler_to_matrix("x", np.pi)

        # Tool offset
        t_tool_offset = flat_to_matrix(self.tool_offset)[None, None, :, :]
        t_delta = t_rotate_z @ t_reflect_z @ t_tool_offset

        t_local_pos = np.tile(np.eye(4)[None, None, :, :], (n_data, n_steps, 1, 1))
        t_local_pos[:, :, :3, 3] = local_pos

        t_local = t_local_pos @ t_delta

        t_obj = flat_to_matrix(obj_states)
        t_global = t_obj[:, None, :, :] @ t_local

        ws_pos = t_global[:, :, :3, 3]
        ws_quat = matrix_to_quat(t_global[:, :, :3, :3].reshape(-1, 3, 3))
        ws_quat = ws_quat.reshape(n_data, n_steps, 4)
        ws_paths = np.concatenate([ws_pos, ws_quat], axis=-1)

        return ws_paths[0]

    def ws_path_to_traj(self, robot_base_pose, t_path, ws_path):
        traj = self.ik.ws_path_to_traj(t_path, ws_path)
        return traj

    def get_final_waypoints(self, push_params, total_time, ws_path):
        n_points = len(ws_path)
        times = np.linspace(0, total_time, n_points)

        traj = self.ws_path_to_traj(None, times, ws_path)
        # `to_step_waypoints` returns (position, velocity, acceleration) with
        # shape (3, steps, dof). The viewer expects joint positions only.
        traj_waypoints = traj.to_step_waypoints(self.dt)
        waypoints = traj_waypoints[0]

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
        ws_path = self.generate_ws_path(push_params, total_time)
        self.get_final_waypoints(push_params, total_time, ws_path)
        return self.get_state()

    def run_viewer(self):
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running():
                if self.reset_requested:
                    pass

                if self.params and self.control_idx < len(self.params):
                    current_waypoints = self.pos_waypoints[self.control_idx][0]

                    with self.sim_lock:
                        self.mj_datas[0].qpos[self.robot_joint_idx] = current_waypoints[0]
                        self.mj_datas[0].ctrl[self.robot_joint_idx] = current_waypoints[0]
                        mujoco.mj_forward(self.mj_model, self.mj_datas[0])

                    for waypoint in current_waypoints:
                        if not viewer.is_running():
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
