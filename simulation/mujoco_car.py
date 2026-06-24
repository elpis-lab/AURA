import time
import numpy as np
import threading
from pathlib import Path
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mujoco import mj_forward

DEFAULT_CAR_XML_PATH = str(Path(__file__).resolve().parent / "assets" / "one_car.xml")


class Sim:

    def __init__(self, xml_path=DEFAULT_CAR_XML_PATH, realtime_sync=True, viewer_sync_rate=10):

        self.realtime_sync = realtime_sync
        self.viewer_sync_rate = viewer_sync_rate

        self.L = 0.1385 + 0.158  # Wheelbase (m)
        self.r = 0.0488          # Wheel radius (m)
        self.W = 0.2300          # Track width (m)
        self.rear_offset = 0.158
        self.throttle_ctrl_scale = 0.04
        self.steering_ctrl_scale = 1.0

        if not Path(xml_path).exists():
            candidate = Path(__file__).resolve().parent / "assets" / str(xml_path)
            if candidate.exists():
                xml_path = str(candidate)

        self.xml_path = xml_path
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.m.vis.global_.offwidth = max(int(self.m.vis.global_.offwidth), 1920)
        self.m.vis.global_.offheight = max(int(self.m.vis.global_.offheight), 1080)
        self.d = mujoco.MjData(self.m)

        self.steering_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "buddy_steering_pos")
        self.throttle_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "buddy_throttle_velocity")

        self.sim_step = 0
        self.control_idx = 0

        self.controls = []
        self.durations = []
        self.states = []
        self.goal_region = None
        self.plan_path = []
        self.record_path = None
        self.record_fps = 24.0
        self.record_width = 1920
        self.record_height = 1080
        self.recorded_frames = []
        self._record_renderer = None
        self._record_last_time = -np.inf
        self.fixed_camera_lookat = None
        self.fixed_camera_distance = None
        self.fixed_camera_azimuth = 90.0
        self.fixed_camera_elevation = -90.0

        # Thread safety for reset
        self.reset_requested = False
        self.reset_completed = False
        self.close_requested = False
        self.sim_lock = threading.Lock()

        self._do_reset()

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
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
    
    def _do_reset(self):
        with self.sim_lock:
            self.d.qpos[:] = 0.0
            self.d.qpos[2] = 0.1             # z height
            self.d.qpos[3:7] = [1, 0, 0, 0]  # quaternion (w, x, y, z)
            self.d.qvel[:] = 0.0
            self.d.ctrl[:] = 0.0
            self.d.act[:] = 0.0
            self.d.qacc[:] = 0.0

            mj_forward(self.m, self.d)

            # Settle simulation
            for _ in range(100):
                mujoco.mj_step(self.m, self.d)
            
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

    def _sync_follow_camera(self, viewer):
        if self.fixed_camera_lookat is not None:
            viewer.cam.lookat[:] = np.asarray(self.fixed_camera_lookat, dtype=float)
            viewer.cam.distance = float(self.fixed_camera_distance or 6.0)
            viewer.cam.azimuth = float(self.fixed_camera_azimuth)
            viewer.cam.elevation = float(self.fixed_camera_elevation)
            return

        with self.sim_lock:
            chassis_x = float(self.d.qpos[0])
            chassis_y = float(self.d.qpos[1])
        viewer.cam.lookat[:] = [chassis_x, chassis_y, 0.12]
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -45

    def set_fixed_camera(
        self,
        lookat,
        distance: float,
        azimuth: float = 90.0,
        elevation: float = -90.0,
    ):
        self.fixed_camera_lookat = np.asarray(lookat, dtype=float).reshape(-1)[:3]
        self.fixed_camera_distance = float(distance)
        self.fixed_camera_azimuth = float(azimuth)
        self.fixed_camera_elevation = float(elevation)

    def _draw_goal_region(self, viewer):
        if self.goal_region is None or not hasattr(viewer, "user_scn"):
            return
        self._add_goal_region_to_scene(viewer.user_scn)

    def set_plan_path(self, states):
        path = []
        if states is not None:
            for state in states:
                arr = np.asarray(state, dtype=float).reshape(-1)
                if arr.size >= 2:
                    path.append(arr[:3].copy())
        with self.sim_lock:
            self.plan_path = path

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
        for start, end in zip(path[:-1], path[1:]):
            if scene.ngeom >= scene.maxgeom:
                return
            a = np.asarray(start, dtype=float).reshape(-1)
            b = np.asarray(end, dtype=float).reshape(-1)
            if a.size < 2 or b.size < 2:
                continue
            from_pt = np.array([a[0], a[1], 0.035], dtype=float)
            to_pt = np.array([b[0], b[1], 0.035], dtype=float)
            if np.linalg.norm(to_pt - from_pt) < 1e-9:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.018,
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
                np.array([half_arc, 0.012, 0.004], dtype=float),
                np.array([x + radius * np.cos(angle), y + radius * np.sin(angle), 0.010], dtype=float),
                mat.reshape(-1),
                np.array([0.05, 0.85, 0.12, 0.85], dtype=float),
            )
            scene.ngeom += 1

    def start_recording(self, path, fps=24.0, width=1600, height=1000):
        self.record_path = path
        self.record_fps = float(fps)
        self.record_width = min(int(width), int(self.m.vis.global_.offwidth))
        self.record_height = min(int(height), int(self.m.vis.global_.offheight))
        self.recorded_frames = []
        self._record_last_time = -np.inf
        self._record_renderer = None

    def _record_frame(self):
        if not self.record_path:
            return
        period = 1.0 / max(float(self.record_fps), 1e-6)
        if float(self.d.time) - self._record_last_time < period:
            return
        with self.sim_lock:
            if self.fixed_camera_lookat is None:
                lookat = np.array(
                    [float(self.d.qpos[0]), float(self.d.qpos[1]), 0.10],
                    dtype=float,
                )
            else:
                lookat = np.asarray(self.fixed_camera_lookat, dtype=float).copy()
            snapshot = (
                self.d.qpos.copy(),
                self.d.qvel.copy(),
                self.d.ctrl.copy(),
                lookat,
                [np.asarray(s, dtype=float).copy() for s in self.plan_path],
            )
            sim_time = float(self.d.time)
        self.recorded_frames.append(snapshot)
        self._record_last_time = sim_time

    def save_recording(self):
        if not self.record_path or not self.recorded_frames:
            return
        from simulation.mujoco_video_renderer import save_recording_in_subprocess

        snapshots = list(self.recorded_frames)
        print(
            f"[mujoco video] rendering {len(snapshots)} frames -> {self.record_path}",
            flush=True,
        )
        saved = save_recording_in_subprocess(
            kind="car",
            xml_path=self.xml_path,
            snapshots=snapshots,
            output_path=self.record_path,
            fps=float(self.record_fps),
            width=int(self.record_width),
            height=int(self.record_height),
            goal_region=self.goal_region,
            camera_config=(
                {
                    "distance": float(self.fixed_camera_distance or 3.0),
                    "azimuth": float(self.fixed_camera_azimuth),
                    "elevation": float(self.fixed_camera_elevation),
                }
                if self.fixed_camera_lookat is not None
                else None
            ),
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
        
    def receive_states(self, states):
        self.states = states
        return True
    
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
    
    def plot_trajectory(self):
        actual_trajectory_states = []

        # Wait until states are received from the planner
        while not self.states:
            time.sleep(0.5)

            # Check if simulation is still running
            if not hasattr(self, 'm'): 
                 return
            
        planned_trajectory_states = [(state[0], state[1]) for state in self.states]

        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Planned vs Actual Trajectory')
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')

        # Plot planned path
        planned_x = [p[0] for p in planned_trajectory_states]
        planned_y = [p[1] for p in planned_trajectory_states]
        ax.plot(planned_x, planned_y, 'b--', linewidth=2, label='Planned', alpha=0.7)
        ax.plot(planned_x[0], planned_y[0], 'go', markersize=10, label='Start')
        ax.plot(planned_x[-1], planned_y[-1], 'ro', markersize=10, label='Goal')

        # Initialize actual trajectory plots
        actual_trajectory, = ax.plot([], [], 'r-', linewidth=2, label='Actual')
        current_pos, = ax.plot([], [], 'r*', markersize=10)

        ax.legend()
        plt.show(block=False)

        try:
            # Poll for actual trajectory updates
            while True:
                state = self.get_state()
                if state is None:
                    break

                x_rear, y_rear, theta, v_actual = state
                actual_trajectory_states.append((x_rear, y_rear))

                actual_x = [a[0] for a in actual_trajectory_states]
                actual_y = [a[1] for a in actual_trajectory_states]
                actual_trajectory.set_data(actual_x, actual_y)
                current_pos.set_data([x_rear], [y_rear])

                # Auto-scale
                all_x = planned_x + actual_x
                all_y = planned_y + actual_y
                margin = 0.5
                if all_x and all_y: # Ensure lists are not empty
                    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
                    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

                plt.pause(0.1)

                # Check if figure is still open
                if not plt.fignum_exists(fig.number):
                    break

        except KeyboardInterrupt:
            print("Trajectory plotting interrupted")
        except Exception as e:
            if 'Figure' not in str(e) and 'Tcl' not in str(e):
                print(f"Plotting error: {e}")
        finally:
            plt.ioff()
            plt.close(fig)

    def run_viewer(self):
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running() and not self.close_requested:
                # Check reset request at the start of each iteration
                if self.reset_requested:
                    self._do_reset()
                
                if self.controls and self.control_idx < len(self.controls):
                    # Get current segment duration
                    segment_duration = self.durations[self.control_idx] if self.durations else 1.0
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
                            if hasattr(viewer, "user_scn"):
                                viewer.user_scn.ngeom = 0
                            self._draw_goal_region(viewer)
                            self._draw_plan_path(viewer)
                            self._record_frame()
                            self._sync_follow_camera(viewer)
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
                        if hasattr(viewer, "user_scn"):
                            viewer.user_scn.ngeom = 0
                        self._draw_goal_region(viewer)
                        self._draw_plan_path(viewer)
                        self._record_frame()
                        self._sync_follow_camera(viewer)
                        viewer.sync()

                    # Realtime synchronization
                    if self.realtime_sync:
                        elapsed = time.time() - step_start
                        sleep_time = self.m.opt.timestep - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
