import time
import numpy as np
import threading
from pathlib import Path
import mujoco
import mujoco.viewer
from mujoco import mj_forward

DEFAULT_CAR_XML_PATH = str(Path(__file__).resolve().parent / "assets" / "one_car.xml")


class MujocoCarSimulator:
    """Low-level MuJoCo execution backend for the kinematic car."""

    def __init__(
        self,
        xml_path=DEFAULT_CAR_XML_PATH,
        realtime_sync=True,
        viewer_sync_rate=10,
    ):

        self.realtime_sync = realtime_sync
        self.viewer_sync_rate = viewer_sync_rate

        self.r = 0.0488  # Wheel radius (m)
        self.rear_offset = 0.158
        self.throttle_ctrl_scale = 0.04
        self.steering_ctrl_scale = 1.0

        if not Path(xml_path).exists():
            candidate = Path(__file__).resolve().parent / "assets" / str(xml_path)
            if candidate.exists():
                xml_path = str(candidate)

        self.xml_path = xml_path
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        # MuJoCo 3.8 made multi-contact convex collision (MultiCCD) the
        # default.  This legacy car model contains ellipsoidal wheel/floor
        # pairs whose collision function is declared with a one-contact
        # capacity, so the new default can abort with:
        #   "returned 3 contacts ... expected at most 1 from mj_maxContact"
        # Keep the contact model under which the asset was authored.
        if hasattr(mujoco.mjtDisableBit, "mjDSBL_MULTICCD"):
            self.m.opt.disableflags |= int(
                mujoco.mjtDisableBit.mjDSBL_MULTICCD
            )
        self.m.vis.global_.offwidth = max(int(self.m.vis.global_.offwidth), 1920)
        self.m.vis.global_.offheight = max(int(self.m.vis.global_.offheight), 1080)
        self.d = mujoco.MjData(self.m)

        self.steering_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "buddy_steering_pos"
        )
        self.throttle_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "buddy_throttle_velocity"
        )

        self.sim_step = 0
        self.control_idx = 0

        self.controls = []
        self.durations = []
        # Thread safety for reset
        self.reset_requested = False
        self.reset_completed = False
        self.close_requested = False
        self.sim_lock = threading.Lock()

        self.reset_immediately()
    
    def quaternion_to_yaw(self, q):
        qw, qx, qy, qz = q
        siny = 2 * (qw * qz + qx * qy)
        cosy = 1 - 2 * (qy * qy + qz * qz)
        return np.arctan2(siny, cosy)
        
    def reset(self):
        self.reset_requested = True
        self.reset_completed = False
        
        # Wait for reset to complete
        timeout = 5.0
        start_time = time.time()
        while not self.reset_completed and (time.time() - start_time) < timeout:
            time.sleep(0.05)

        if self.reset_completed:
            print("Reset completed")
            return True
        else:
            print("Reset timed out")
            return False
    
    def reset_immediately(self):
        with self.sim_lock:
            self.d.qpos[:] = 0.0
            self.d.qpos[2] = 0.1             # z height
            self.d.qpos[3:7] = [1, 0, 0, 0]  # quaternion (w, x, y, z)
            self.d.qvel[:] = 0.0
            self.d.ctrl[:] = 0.0
            self.d.act[:] = 0.0
            self.d.qacc[:] = 0.0

            mj_forward(self.m, self.d)

            # Do not pre-step the mesh-based car at the origin. Newer MuJoCo
            # versions can create an over-complete transient floor/mesh contact
            # there before the experiment wrapper installs the requested
            # rear-axle pose. ``set_state`` performs mj_forward at that pose.
            
            # Reset control tracking
            self.control_idx = 0
            self.controls = []
            self.durations = []
        
        self.reset_requested = False
        self.reset_completed = True
    
    def step(self):
        with self.sim_lock:
            mujoco.mj_step(self.m, self.d)
            self.sim_step += 1

    def stop(self):
        with self.sim_lock:
            self.d.ctrl[self.steering_id] = 0.0
            self.d.ctrl[self.throttle_id] = 0.0
            self.d.qvel[:] = 0.0
            self.d.qacc[:] = 0.0

    def close(self):
        self.close_requested = True
        self.stop()

    def get_state(self):
        with self.sim_lock:
            chassis_x = self.d.qpos[0]
            chassis_y = self.d.qpos[1]
            theta = self.quaternion_to_yaw(self.d.qpos[3:7])

            chassis_vx = self.d.qvel[0]
            chassis_vy = self.d.qvel[1]

            x_rear = chassis_x - self.rear_offset * np.cos(theta)
            y_rear = chassis_y - self.rear_offset * np.sin(theta)
            v_forward = chassis_vx * np.cos(theta) + chassis_vy * np.sin(theta)

            return np.array([x_rear, y_rear, theta, v_forward])

    def set_state(self, state):
        """Set the rear-axle SE(2) pose used by the planner interface."""
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size < 3:
            raise ValueError(f"car state must contain x, y, theta; got {state}")
        rear_x, rear_y, theta = state[:3]
        chassis_x = rear_x + self.rear_offset * np.cos(theta)
        chassis_y = rear_y + self.rear_offset * np.sin(theta)
        with self.sim_lock:
            self.d.qpos[0] = chassis_x
            self.d.qpos[1] = chassis_y
            self.d.qpos[2] = 0.1
            self.d.qpos[3:7] = [
                np.cos(0.5 * theta),
                0.0,
                0.0,
                np.sin(0.5 * theta),
            ]
            self.d.qvel[:] = 0.0
            self.d.ctrl[:] = 0.0
            self.d.qacc[:] = 0.0
            mujoco.mj_forward(self.m, self.d)
        return self.get_state()

    def execute_segment(self, control, duration):
        # Add new segment to the end of the controls list
        self.controls.append(control)
        self.durations.append(duration)

        # Expected control_idx to reach
        target_idx = len(self.controls)

        # Wait until the run_viewer loop increments control_idx past this segment
        timeout = float(duration) + 5.0
        start_time = time.time()
        while self.control_idx < target_idx:
            if time.time() - start_time > timeout:
                print("Timeout waiting for car segment execution")
                break
            time.sleep(0.05)

        return self.get_state()
    
    def run_viewer(self):
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running() and not self.close_requested:
                # Check reset request at the start of each iteration
                if self.reset_requested:
                    self.reset_immediately()
                
                if self.controls and self.control_idx < len(self.controls):
                    # Get current segment duration
                    segment_duration = (
                        self.durations[self.control_idx]
                        if self.durations
                        else 1.0
                    )
                    segment_end_time = self.d.time + segment_duration

                    # Get desired control from plan
                    u_vel_desired, u_phi_desired = self.controls[self.control_idx]

                    while (
                        viewer.is_running()
                        and not self.close_requested
                        and self.d.time < segment_end_time
                    ):
                        # Check reset request during control execution
                        if self.reset_requested:
                            break
                        
                        step_start = time.time()

                        with self.sim_lock:
                            # Set steering control
                            self.d.ctrl[self.steering_id] = self.steering_ctrl_scale * u_phi_desired

                            # Convert linear velocity to wheel angular velocity
                            wheel_omega_des = u_vel_desired / self.r

                            # Convert to torque command
                            throttle_cmd = wheel_omega_des * self.throttle_ctrl_scale

                            self.d.ctrl[self.throttle_id] = throttle_cmd
                        
                        # Step simulation
                        self.step()

                        # Update viewer
                        if self.sim_step % self.viewer_sync_rate == 0:
                            viewer.sync()

                        # Realtime synchronization
                        if self.realtime_sync:
                            elapsed = time.time() - step_start
                            sleep_time = self.m.opt.timestep - elapsed
                            if sleep_time > 0:
                                time.sleep(sleep_time)

                    # Move to next control segment
                    if not self.reset_requested:
                        with self.sim_lock:
                            self.d.ctrl[self.steering_id] = 0.0
                            self.d.ctrl[self.throttle_id] = 0.0
                            self.d.qvel[:] = 0.0
                            self.d.qacc[:] = 0.0
                        self.control_idx += 1

                else:
                    # No plan or plan completed - just step simulation
                    step_start = time.time()
                    self.step()

                    # Update viewer
                    if self.sim_step % self.viewer_sync_rate == 0:
                        viewer.sync()

                    # Realtime synchronization
                    if self.realtime_sync:
                        elapsed = time.time() - step_start
                        sleep_time = self.m.opt.timestep - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
