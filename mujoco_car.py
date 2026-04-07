import time
import numpy as np
import threading
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mujoco import mj_forward


class Sim:

    def __init__(self, xml_path='one_car.xml', realtime_sync=True, viewer_sync_rate=10):

        self.realtime_sync = realtime_sync
        self.viewer_sync_rate = viewer_sync_rate

        self.L = 0.1385 + 0.158  # Wheelbase (m)
        self.r = 0.0488          # Wheel radius (m)
        self.W = 0.2300          # Track width (m)
        self.rear_offset = 0.158

        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)

        self.steering_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "buddy_steering_pos")
        self.throttle_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "buddy_throttle_velocity")

        self.sim_step = 0
        self.control_idx = 0

        self.controls = []
        self.durations = []
        self.states = []

        # Thread safety for reset
        self.reset_requested = False
        self.reset_completed = False
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
        while self.control_idx < target_idx:
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
            while viewer.is_running():
                # Check reset request at the start of each iteration
                if self.reset_requested:
                    self._do_reset()
                
                if self.controls and self.control_idx < len(self.controls):
                    # Get current segment duration
                    segment_duration = self.durations[self.control_idx] if self.durations else 1.0
                    segment_end_time = self.d.time + segment_duration

                    # Get desired control from plan
                    u_vel_desired, u_phi_desired = self.controls[self.control_idx]

                    while viewer.is_running() and self.d.time < segment_end_time:
                        # Check reset request during control execution
                        if self.reset_requested:
                            break
                        
                        step_start = time.time()

                        with self.sim_lock:
                            # Set steering control
                            self.d.ctrl[self.steering_id] = u_phi_desired

                            # Convert linear velocity to wheel angular velocity
                            wheel_omega_des = u_vel_desired / self.r

                            # Convert to torque command
                            throttle_cmd = wheel_omega_des * 0.04

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