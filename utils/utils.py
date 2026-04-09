import sys
import torch
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


try:
    from ompl import base as ob
    from ompl import control as oc
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "py-bindings"))
    from ompl import base as ob
    from ompl import control as oc


def parse_command_line_args():
    """Parse command line arguments in key=value format"""
    args = {}

    # Default values
    defaults = {
        "plan_time": 10.0,
        "replan_time": 3.0,
        "dynamics_type": "model",
        "planner": "fusion",
    }

    # Parse command line arguments
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Convert to appropriate type
            if key in ["plan_time", "replan_time"]:
                try:
                    args[key] = float(value)
                except ValueError:
                    print(f"Warning: Invalid float value for {key}: {value}. Using default.")
                    args[key] = defaults[key]
            else:
                args[key] = value
        else:
            print(f"Warning: Ignoring invalid argument format: {arg}")

    # Fill in defaults for missing arguments
    for key, default_value in defaults.items():
        if key not in args:
            args[key] = default_value

    return args


def visualize_tree_3d(planner, filename="fusion_tree_3d.png", show_plot=True):
    """Visualize the tree structure in 3D (x, y, theta) using matplotlib

    Args:
        planner: The OMPL planner instance
        filename: Name of the file to save the plot (default: "fusion_tree_3d.png")
        show_plot: Whether to display the plot interactively (default: True)
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Get planner data
    planner_data = ob.PlannerData(planner.getSpaceInformation())
    print("Getting planner data for 3D visualization...")
    planner.getPlannerData(planner_data)
    print("Planner data obtained")

    # Extract all vertices (x, y, theta)
    all_vertices = []
    print("Collecting vertices...")
    for i in range(planner_data.numVertices()):
        vertex = planner_data.getVertex(i)
        state = vertex.getState()
        # Extract SE2 state components
        x = state.getX()
        y = state.getY()
        theta = state.getYaw()
        all_vertices.append((x, y, theta))

    # Get solution path states
    solution_states = []
    try:
        solution_path = planner.getProblemDefinition().getSolutionPath()
        if solution_path:
            print("Extracting solution path...")
            for i in range(solution_path.getStateCount()):
                state = solution_path.getState(i)
                x = state.getX()
                y = state.getY()
                theta = state.getYaw()
                solution_states.append((x, y, theta))
    except:
        print("No solution path available")
        pass

    print("Creating 3D plot...")
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if all_vertices:
        # Convert to numpy arrays for easier plotting
        all_vertices_array = np.array(all_vertices)

        # Plot all tree vertices as small blue dots
        ax.scatter(
            all_vertices_array[:, 0],  # x
            all_vertices_array[:, 1],  # y
            all_vertices_array[:, 2],  # theta
            c="steelblue",
            s=20,
            alpha=0.3,  # More transparent to show density patterns
            label="Tree nodes",
        )

    # Plot solution path if available
    if solution_states:
        solution_array = np.array(solution_states)

        # Plot solution path vertices as larger red dots
        ax.scatter(
            solution_array[:, 0],  # x
            solution_array[:, 1],  # y
            solution_array[:, 2],  # theta
            c="red",
            s=60,
            alpha=0.9,
            label="Solution path nodes",
            marker="o",
            edgecolors="darkred",
            linewidth=2,
        )

        # Connect solution path states with lines
        ax.plot(
            solution_array[:, 0],  # x
            solution_array[:, 1],  # y
            solution_array[:, 2],  # theta
            color="orange",
            linewidth=3,
            alpha=0.8,
            label="Solution path",
        )

        # Mark start and goal specially
        if len(solution_states) > 0:
            # Start state (green square)
            start = solution_array[0]
            ax.scatter(
                start[0],
                start[1],
                start[2],
                c="green",
                s=100,
                marker="s",
                edgecolors="darkgreen",
                linewidth=2,
                label="Start state",
            )

            # Goal state (red star)
            goal = solution_array[-1]
            ax.scatter(
                goal[0],
                goal[1],
                goal[2],
                c="red",
                s=120,
                marker="*",
                edgecolors="darkred",
                linewidth=2,
                label="Goal state",
            )

    # Set labels and title
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Theta (radians)", fontsize=12)
    ax.set_title(
        "Fusion Planner Tree Visualization (3D: x, y, θ)",
        fontsize=14,
        fontweight="bold",
    )

    # Add legend
    ax.legend(loc="upper right", fontsize=10)

    # Set fixed axis limits for x, y, and theta to match state space bounds
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-np.pi, np.pi)  # Theta typically ranges from -π to π

    # Set equal aspect ratio for x and y
    ax.set_box_aspect([1, 1, 0.5])  # Make theta axis shorter for better view

    plt.tight_layout()

    print(f"Saving 3D tree to {filename}")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"3D tree saved to {filename}")

    if show_plot:
        plt.show(block=False)
    else:
        plt.close()


def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_args(args):
    """
    A simple wrapper for argument parser
    args is a list of arguments, each argument is
    a tuple of (name, default(optional), type(optional))
    """
    parser = argparse.ArgumentParser()
    for arg in args:
        kwargs = {"nargs": "?"}
        if len(arg) > 1:
            kwargs["default"] = arg[1]
        if len(arg) > 2:
            kwargs["type"] = arg[2]
        parser.add_argument(arg[0], **kwargs)

    args = parser.parse_args()
    return args


def get_names(object_name):
    """A simple function to extract where the model and data name should be"""
    if "real" in object_name:
        model_name = object_name[5:]
        data_name = object_name + "_2000"  # real world data has 2000 samples
        rep_data_name = object_name + "_100x10"
    else:
        model_name = object_name
        data_name = object_name + "_10000"  # sim data has 10000 samples
        rep_data_name = object_name + "_1000x10"
    return model_name, data_name, rep_data_name


class DataLoader:
    """Class for loading data and splitting them"""

    def __init__(
        self,
        object_name: str,
        folder: str = "data",
        val_size: int = 1000,
        invert_xy: bool = False,
        shuffle: bool = False,
    ):
        """Initialize with data files, split sizes, and some options"""
        self.data_x_file = folder + "/x_" + object_name + ".npy"
        self.data_y_file = folder + "/y_" + object_name + ".npy"
        self.val_size = val_size

        # Load data
        x = np.load(self.data_x_file).astype(np.float32)
        y = np.load(self.data_y_file).astype(np.float32)
        # Pre-process data
        if invert_xy:
            x, y = y, x
        if shuffle:
            idx = np.random.permutation(len(x))
            x, y = x[idx], y[idx]
        self.x = x
        self.y = y

        # Check
        self.pool_size = len(self.x) - self.val_size
        if self.pool_size <= 0:
            raise ValueError("Pool size is 0")

    def load_data(self, verbose=1):
        """Load all data as a dictionary"""
        # Split data

        x_pool = self.x[: self.pool_size]
        y_pool = self.y[: self.pool_size]
        x_val = self.x[self.pool_size : self.pool_size + self.val_size]
        y_val = self.y[self.pool_size : self.pool_size + self.val_size]

        if verbose:
            print("Loading data")
            print(f"Pool data points: {x_pool.shape[0]}")
            print(f"Validation data points: {x_val.shape[0]}")

        datasets = dict()
        datasets["x_pool"] = x_pool
        datasets["y_pool"] = y_pool
        datasets["x_val"] = x_val
        datasets["y_val"] = y_val
        return datasets


def plot_states(states, planned_states=None, obj_shape=None):
    """Plot the states of the object."""
    states = np.array(states)
    if planned_states is not None:
        planned_states = np.array(planned_states)

    plt.figure(figsize=(8, 6))
    # Plot the table as the background
    draw_rectangle(0, -0.505, 1.524, 1.524, 0, "k", alpha=0.1)
    # Plot a robot
    draw_rectangle(0, 0, 0.2, 0.2, 0, "gray", alpha=1.0, label="Robot")

    # Plot the states path
    if planned_states is not None:
        plt.plot(
            planned_states[:, 0],
            planned_states[:, 1],
            "o-",
            color="b",
            label="Planned Path",
        )
    plt.plot(
        states[:, 0],
        states[:, 1],
        "o-",
        color="g",
        label="Actual Path",
    )
    # If object shape is provided, draw rectangles for start and goal
    if obj_shape is not None:
        w, l = obj_shape[0], obj_shape[1]

        # Draw rectangles
        if planned_states is not None:
            for state in planned_states:
                state_x, state_y, state_theta = state
                draw_rectangle(state_x, state_y, w, l, state_theta, "b", alpha=0.3)
        for state in states:
            state_x, state_y, state_theta = state
            draw_rectangle(state_x, state_y, w, l, state_theta, "g", alpha=0.3)

    # Plot start positions
    plt.plot(states[0, 0], states[0, 1], "ro", label="Start")

    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Push Path")
    plt.legend()
    plt.show()


def draw_rectangle(x, y, width, length, theta, color="b", alpha=0.5, label=None):
    """Draw a rectangle at the given position with the given orientation."""
    # Calculate the four corners of the rectangle
    corners = np.array(
        [
            [-width / 2, -length / 2],
            [width / 2, -length / 2],
            [width / 2, length / 2],
            [-width / 2, length / 2],
            [-width / 2, -length / 2],  # Close the rectangle
        ]
    )

    # Rotate the corners
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_corners = np.dot(corners, rot_matrix.T)
    # Translate the corners
    translated_corners = rotated_corners + np.array([x, y])

    # Plot the rectangle
    plt.plot(
        translated_corners[:, 0],
        translated_corners[:, 1],
        color=color,
        alpha=alpha,
    )
    plt.fill(
        translated_corners[:, 0],
        translated_corners[:, 1],
        color=color,
        alpha=alpha,
    )

    # Add text label
    if label is not None:
        plt.text(x, y, label, ha="center", va="center", color="black")


def isStateValid(spaceInformation, state, system=None, config=None, obstacle_config=None):
    """
    Check if state is valid (within bounds and optionally collision-free with obstacles).

    Args:
        spaceInformation: OMPL SpaceInformation object
        state: OMPL state to check
        system: System name (e.g., "simple_car", "pushing", "dublin_airplane")
        config: Configuration dictionary (optional)
        obstacle_config: Obstacle configuration dictionary with keys:
            - "enabled": bool, whether to check obstacles
            - "safety_radius": float, safety radius around obstacles
            - "circles": list of (cx, cy, r) tuples for circular obstacles
            - "aabbs": list of (xmin, ymin, xmax, ymax) tuples for axis-aligned boxes
            - "boxes": list of (cx, cy, hx, hy, yaw) tuples for oriented boxes

    Returns:
        bool: True if state is valid, False otherwise
    """
    # First check bounds
    if not spaceInformation.satisfiesBounds(state):
        return False

    # If no obstacle checking is requested, return True
    if obstacle_config is None or not obstacle_config.get("enabled", False):
        return True

    # Extract state position based on system type
    if system in ["simple_car", "kinematic_car", "pushing", "pushing_object"]:
        # SE2 state: x, y, theta
        try:
            x = state.getX()
            y = state.getY()
            pos = np.array([x, y])
        except AttributeError:
            # Fallback: state might be a compound state wrapper
            try:
                if callable(state):
                    state_obj = state()
                else:
                    state_obj = state
                x = state_obj.getX()
                y = state_obj.getY()
                pos = np.array([x, y])
            except (AttributeError, TypeError):
                # If we can't extract position, skip obstacle check
                return True

    elif system == "dublin_airplane":
        # SE3 state: x, y, z (we check obstacles in 2D, ignoring z)
        try:
            if callable(state):
                compound_state = state()
            else:
                compound_state = state
            x = compound_state[0][0]  # x position
            y = compound_state[0][1]  # y position
            pos = np.array([x, y])
        except (AttributeError, TypeError, IndexError):
            # If we can't extract position, skip obstacle check
            return True
    else:
        # Unknown system, skip obstacle check
        return True

    # Check obstacles
    safety_radius = obstacle_config.get("safety_radius", 0.10)

    # Check circular obstacles
    circles = obstacle_config.get("circles", [])
    for cx, cy, r in circles:
        dist = np.hypot(pos[0] - cx, pos[1] - cy)
        if dist < (r + safety_radius):
            return False

    # Check axis-aligned bounding boxes (AABBs)
    aabbs = obstacle_config.get("aabbs", [])
    for xmin, ymin, xmax, ymax in aabbs:
        # Inflate AABB by safety_radius
        xmin_inflated = xmin - safety_radius
        ymin_inflated = ymin - safety_radius
        xmax_inflated = xmax + safety_radius
        ymax_inflated = ymax + safety_radius
        if xmin_inflated <= pos[0] <= xmax_inflated and ymin_inflated <= pos[1] <= ymax_inflated:
            return False

    # Check oriented boxes
    boxes = obstacle_config.get("boxes", [])
    for cx, cy, hx, hy, yaw in boxes:
        # Transform point to box-local coordinates
        c, s = np.cos(-yaw), np.sin(-yaw)
        px = pos[0] - cx
        py = pos[1] - cy
        plx = c * px - s * py
        ply = s * px + c * py
        # Check if point is inside box (with safety_radius inflation)
        hx_inflated = hx + safety_radius
        hy_inflated = hy + safety_radius
        if abs(plx) < hx_inflated and abs(ply) < hy_inflated:
            return False

    return True


def normalize_obstacle_config(obstacle_config):
    """
    Normalize obstacle config by auto-enabling obstacle checks when
    obstacle geometry entries are present.
    """
    if isinstance(obstacle_config, dict):
        if (
            obstacle_config.get("circles")
            or obstacle_config.get("aabbs")
            or obstacle_config.get("boxes")
        ):
            obstacle_config["enabled"] = obstacle_config.get("enabled", True)
    return obstacle_config


def state2list(state, state_type: str) -> list:
    # If state is already a list, tuple, or numpy array, return it as a list
    if isinstance(state, (list, tuple, np.ndarray)):
        return list(state) if not isinstance(state, np.ndarray) else state.tolist()

    if state_type in ("simple_car", "kinematic_car", "pushing", "pushing_object"):
        # SE2 state: x, y, theta
        return [state.getX(), state.getY(), state.getYaw()]

    elif state_type == "dublin_airplane":
        # SE3 state: x, y, z, qw, qx, qy, qz (position + quaternion)
        # For compound states, the state object itself is the compound state (don't call state())
        # Check if state is callable (wrapper) or direct compound state
        try:
            if callable(state) and not isinstance(state, (list, tuple, np.ndarray)):
                # State is a wrapper, call it to get the compound state
                compound_state = state()
            else:
                # State is already the compound state
                compound_state = state

            # Access compound state components: [0] is R^3 (position), [1] is SO(3) (quaternion)
            return [
                compound_state[0][0],  # x
                compound_state[0][1],  # y
                compound_state[0][2],  # z
                compound_state[1].w,  # quaternion w component
                compound_state[1].x,  # quaternion x component
                compound_state[1].y,  # quaternion y component
                compound_state[1].z,  # quaternion z component
            ]
        except (AttributeError, TypeError, IndexError) as e:
            print(
                f"Warning: Could not access SE3 state components for type {type(state)}, error: {e}"
            )
            return []

    else:
        print(f"Warning: Unknown state type '{state_type}'. Returning empty list.")
        return []


def isSE2Equal(state1, state2, tolerance=1e-6):
    diff_x = abs(state1[0] - state2[0])
    diff_y = abs(state1[1] - state2[1])
    diff_yaw = abs(state1[2] - state2[2])

    return diff_x < tolerance and diff_y < tolerance and diff_yaw < tolerance


def isSE3Equal(state1, state2, tolerance=1e-6):
    """Compare two SE3 states for equality within tolerance.
    Handles both OMPL state objects and lists (from state2list)."""
    # Check if inputs are OMPL state objects (callable) or lists
    is_state1_callable = callable(state1) and not isinstance(state1, (list, tuple, np.ndarray))
    is_state2_callable = callable(state2) and not isinstance(state2, (list, tuple, np.ndarray))

    # Convert OMPL state objects to compound states if needed
    if is_state1_callable:
        compound_state1 = state1()
    else:
        compound_state1 = state1

    if is_state2_callable:
        compound_state2 = state2()
    else:
        compound_state2 = state2

    # Handle list format: [x, y, z, qw, qx, qy, qz]
    if isinstance(compound_state1, (list, tuple, np.ndarray)) and isinstance(
        compound_state2, (list, tuple, np.ndarray)
    ):
        # Compare position components
        pos_diff = np.sqrt(
            (compound_state1[0] - compound_state2[0]) ** 2
            + (compound_state1[1] - compound_state2[1]) ** 2
            + (compound_state1[2] - compound_state2[2]) ** 2
        )

        # Compare quaternion components (normalize first to handle sign ambiguity)
        # Format: [x, y, z, qw, qx, qy, qz]
        quat1_raw = [compound_state1[3], compound_state1[4], compound_state1[5], compound_state1[6]]
        quat2_raw = [compound_state2[3], compound_state2[4], compound_state2[5], compound_state2[6]]

        quat1_norm = normalize_quaternion(quat1_raw)
        quat2_norm = normalize_quaternion(quat2_raw)

        # Handle quaternion sign ambiguity: q and -q represent the same rotation
        # Check both q and -q and take the minimum distance
        quat_diff1 = np.sqrt(
            (quat1_norm[0] - quat2_norm[0]) ** 2
            + (quat1_norm[1] - quat2_norm[1]) ** 2
            + (quat1_norm[2] - quat2_norm[2]) ** 2
            + (quat1_norm[3] - quat2_norm[3]) ** 2
        )
        quat_diff2 = np.sqrt(
            (quat1_norm[0] + quat2_norm[0]) ** 2
            + (quat1_norm[1] + quat2_norm[1]) ** 2
            + (quat1_norm[2] + quat2_norm[2]) ** 2
            + (quat1_norm[3] + quat2_norm[3]) ** 2
        )
        quat_diff = min(quat_diff1, quat_diff2)
    else:
        # OMPL compound state format: compound_state[0] is R^3, compound_state[1] is SO(3)
        # Compare position components
        pos_diff = np.sqrt(
            (compound_state1[0][0] - compound_state2[0][0]) ** 2
            + (compound_state1[0][1] - compound_state2[0][1]) ** 2
            + (compound_state1[0][2] - compound_state2[0][2]) ** 2
        )

        # Compare quaternion components (normalize first to handle sign ambiguity)
        quat1_raw = [
            compound_state1[1].w,
            compound_state1[1].x,
            compound_state1[1].y,
            compound_state1[1].z,
        ]
        quat2_raw = [
            compound_state2[1].w,
            compound_state2[1].x,
            compound_state2[1].y,
            compound_state2[1].z,
        ]

        quat1_norm = normalize_quaternion(quat1_raw)
        quat2_norm = normalize_quaternion(quat2_raw)

        # Handle quaternion sign ambiguity: q and -q represent the same rotation
        quat_diff1 = np.sqrt(
            (quat1_norm[0] - quat2_norm[0]) ** 2
            + (quat1_norm[1] - quat2_norm[1]) ** 2
            + (quat1_norm[2] - quat2_norm[2]) ** 2
            + (quat1_norm[3] - quat2_norm[3]) ** 2
        )
        quat_diff2 = np.sqrt(
            (quat1_norm[0] + quat2_norm[0]) ** 2
            + (quat1_norm[1] + quat2_norm[1]) ** 2
            + (quat1_norm[2] + quat2_norm[2]) ** 2
            + (quat1_norm[3] + quat2_norm[3]) ** 2
        )
        quat_diff = min(quat_diff1, quat_diff2)

    return pos_diff < tolerance and quat_diff < tolerance


def isStateEqual(state1, state2, system, tolerance=1e-6):
    """Generic state comparison function that handles different systems."""
    if system in ("simple_car", "kinematic_car"):
        return isSE2Equal(state1, state2, tolerance)
    elif system == "dublin_airplane":
        return isSE3Equal(state1, state2, tolerance)
    else:
        # Fallback to simple element-wise comparison
        if len(state1) != len(state2):
            return False
        return all(abs(a - b) < tolerance for a, b in zip(state1, state2))


def normalize_quaternion(quat):
    """Normalize quaternion to handle sign ambiguity."""
    quat = np.array(quat)
    # Normalize to unit length
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat = quat / norm
    # Ensure consistent sign (make first non-zero component positive)
    for i in range(4):
        if abs(quat[i]) > 1e-10:
            if quat[i] < 0:
                quat = -quat
            break
    return quat


def arrayDistance(array1, array2, system: str):
    from ompl import base as ob

    # Normalize inputs to flat numpy arrays.
    array1 = np.asarray(array1, dtype=float).reshape(-1)
    array2 = np.asarray(array2, dtype=float).reshape(-1)

    system_alias = {
        "simple_car": "kinematic_car",
        "kinematic_car": "kinematic_car",
        "pushing": "pushing_object",
        "pushing_object": "pushing_object",
        "double_integrator": "double_integrator",
        "position": "position",
    }
    system_key = system_alias.get(system, system)

    if system_key in ("kinematic_car", "pushing_object"):
        # Check if arrays have enough elements for SE2
        if len(array1) < 3 or len(array2) < 3:
            raise ValueError(
                f"SE2 states need at least 3 elements, got {len(array1)} and {len(array2)}"
            )

        # Use OMPL's SE2StateSpace distance function to match what addNoise uses
        se2_space = ob.SE2StateSpace()
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-10.0)
        bounds.setHigh(10.0)
        se2_space.setBounds(bounds)

        # Create OMPL states
        ompl_state1 = se2_space.allocState()
        ompl_state2 = se2_space.allocState()

        # Set state components
        ompl_state1.setX(array1[0])
        ompl_state1.setY(array1[1])
        ompl_state1.setYaw(array1[2])
        ompl_state2.setX(array2[0])
        ompl_state2.setY(array2[1])
        ompl_state2.setYaw(array2[2])

        # Compute distance using OMPL's SE2 state space distance function
        return se2_space.distance(ompl_state1, ompl_state2)

    if system_key == "double_integrator":
        if len(array1) < 6 or len(array2) < 6:
            raise ValueError(
                f"double_integrator states need at least 6 elements, got {len(array1)} and {len(array2)}"
            )
        # Euclidean distance in R^6.
        return float(np.linalg.norm(array1[:6] - array2[:6]))

    if system_key == "position":
        # Check if arrays have enough elements for SE2Position
        if len(array1) < 2 or len(array2) < 2:
            raise ValueError(
                f"SE2Position states need at least 2 elements, got {len(array1)} and {len(array2)}"
            )

        posDistance = np.sqrt((array1[0] - array2[0]) ** 2 + (array1[1] - array2[1]) ** 2)
        return posDistance

    raise ValueError(f"Invalid system: {system}")


def log(message, log_type="info"):
    colors = {
        "error": "\033[91m",  # Red
        "warning": "\033[93m",  # Yellow
        "info": "\033[0m",  # Default
        "success": "\033[92m",  # Green
    }

    reset = "\033[0m"
    color_code = colors.get(log_type.lower(), colors["info"])

    if log_type.lower() == "info":
        print(message)
    else:
        print(f"{color_code}{message}{reset}")


def printState(state, system, situation):
    """Print the state in a readable format."""
    if system in ("simple_car", "kinematic_car", "pushing", "pushing_object"):
        print(
            f"       - {situation} State: x={state[0]:.3f}, y={state[1]:.3f}, theta={state[2]:.3f}"
        )

    elif system == "dublin_airplane":
        # Check if state is an OMPL state object (callable) or numpy array/list
        if callable(state) and not isinstance(state, (list, tuple, np.ndarray)):
            # OMPL state object - need to call it to get compound state
            compound_state = state()
            print(
                f"       - {situation} State: x={compound_state[0][0]:.3f}, y={compound_state[0][1]:.3f}, z={compound_state[0][2]:.3f}, "
                f"quat=[{compound_state[1].w:.3f}, {compound_state[1].x:.3f}, {compound_state[1].y:.3f}, {compound_state[1].z:.3f}]"
            )
        else:
            # Numpy array or list format: [x, y, z, qw, qx, qy, qz]
            state_array = np.array(state) if not isinstance(state, np.ndarray) else state
            if len(state_array) >= 7:
                print(
                    f"       - {situation} State: x={state_array[0]:.3f}, y={state_array[1]:.3f}, z={state_array[2]:.3f}, "
                    f"quat=[{state_array[3]:.3f}, {state_array[4]:.3f}, {state_array[5]:.3f}, {state_array[6]:.3f}]"
                )
            else:
                print(f"       - {situation} State: {state}")

    elif system == "control":
        print(f"       - {situation} Control: {state}")

    else:
        print(f"       - {situation} State: {state}")


def addNoise(system, state, pos_std, rot_std):
    """Add Gaussian noise to a state. Handles both OMPL state objects and numpy arrays."""
    # Check if state is a numpy array/list or OMPL state object
    is_array = isinstance(state, (np.ndarray, list, tuple))

    # Save original state for comparison
    if is_array:
        original_state = (
            np.array(state).copy() if isinstance(state, np.ndarray) else np.array(state)
        )
    else:
        # For OMPL state objects, convert to list for comparison
        original_state = state2list(state, system)

    if system in ("simple_car", "kinematic_car", "pushing", "pushing_object"):
        # SE2 state: [x, y, theta]
        if is_array:
            # Handle numpy array or list
            state = np.array(state) if not isinstance(state, np.ndarray) else state
            state[0] = state[0] + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std)
            state[1] = state[1] + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std)
            state[2] = state[2] + np.clip(np.random.normal(0, rot_std), -rot_std, rot_std)
            noisy_state = state.copy()
        else:
            # Handle OMPL SE2 state object
            if callable(state):
                state = state()
            state.setX(state.getX() + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std))
            state.setY(state.getY() + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std))
            state.setYaw(state.getYaw() + np.clip(np.random.normal(0, rot_std), -rot_std, rot_std))
            # Convert to list for comparison
            noisy_state = state2list(state, system)

    elif system == "dublin_airplane":
        # SE3 state: [x, y, z, qw, qx, qy, qz] or compound state
        if is_array:
            # Handle numpy array or list format: [x, y, z, qw, qx, qy, qz]
            state = np.array(state) if not isinstance(state, np.ndarray) else state.copy()

            # Add noise to position components
            state[0] = state[0] + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std)
            state[1] = state[1] + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std)
            state[2] = state[2] + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std)

            # Add noise to quaternion using small rotation approach
            # Quaternions are unit vectors, so we need to add noise properly
            # Use a variable rotation std for quaternions to create variation
            # Add randomness so noise is sometimes > 0.05 and sometimes smaller
            # Use a random multiplier between 0.5 and 2.0 to create variation
            noise_multiplier = np.random.uniform(0.5, 2.0)
            quat_std = rot_std * noise_multiplier

            # Normalize the input quaternion first
            quat = normalize_quaternion([state[3], state[4], state[5], state[6]])

            # Generate small rotation angles (in radians)
            # Use axis-angle representation for small rotations
            angle = np.random.normal(0, quat_std)
            # Increase clipping range to allow for larger variations
            angle = np.clip(angle, -quat_std * 4, quat_std * 4)  # Clip to 4 std devs

            # Random axis for rotation
            axis = np.random.normal(0, 1, 3)
            axis = axis / np.linalg.norm(axis)  # Normalize axis

            # Create small rotation quaternion: q = [cos(θ/2), sin(θ/2) * axis]
            half_angle = angle / 2.0
            dq_w = np.cos(half_angle)
            dq_xyz = np.sin(half_angle) * axis

            # Multiply quaternions: q_new = q * dq
            # Quaternion multiplication: q1 * q2 = [w1*w2 - x1*x2 - y1*y2 - z1*z2,
            #                                      w1*x2 + x1*w2 + y1*z2 - z1*y2,
            #                                      w1*y2 - x1*z2 + y1*w2 + z1*x2,
            #                                      w1*z2 + x1*y2 - y1*x2 + z1*w2]
            q_new = np.array(
                [
                    quat[0] * dq_w
                    - quat[1] * dq_xyz[0]
                    - quat[2] * dq_xyz[1]
                    - quat[3] * dq_xyz[2],
                    quat[0] * dq_xyz[0]
                    + quat[1] * dq_w
                    + quat[2] * dq_xyz[2]
                    - quat[3] * dq_xyz[1],
                    quat[0] * dq_xyz[1]
                    - quat[1] * dq_xyz[2]
                    + quat[2] * dq_w
                    + quat[3] * dq_xyz[0],
                    quat[0] * dq_xyz[2]
                    + quat[1] * dq_xyz[1]
                    - quat[2] * dq_xyz[0]
                    + quat[3] * dq_w,
                ]
            )

            # Normalize the result
            q_new = normalize_quaternion(q_new)
            state[3] = q_new[0]  # qw
            state[4] = q_new[1]  # qx
            state[5] = q_new[2]  # qy
            state[6] = q_new[3]  # qz
        else:
            # Handle OMPL compound state object
            if callable(state):
                compound_state = state()
            else:
                compound_state = state

            # Add noise to position components
            compound_state[0][0] = compound_state[0][0] + np.clip(
                np.random.normal(0, pos_std), -pos_std, pos_std
            )
            compound_state[0][1] = compound_state[0][1] + np.clip(
                np.random.normal(0, pos_std), -pos_std, pos_std
            )
            compound_state[0][2] = compound_state[0][2] + np.clip(
                np.random.normal(0, pos_std), -pos_std, pos_std
            )

            # Add noise to quaternion using small rotation approach
            # Use a variable rotation std for quaternions to create variation
            # Add randomness so noise is sometimes > 0.05 and sometimes smaller
            noise_multiplier = np.random.uniform(0.5, 2.0)
            quat_std = rot_std * noise_multiplier

            # Get current quaternion
            quat = [
                compound_state[1].w,
                compound_state[1].x,
                compound_state[1].y,
                compound_state[1].z,
            ]
            quat = normalize_quaternion(quat)

            # Generate small rotation angles (in radians)
            angle = np.random.normal(0, quat_std)
            # Increase clipping range to allow for larger variations
            angle = np.clip(angle, -quat_std * 4, quat_std * 4)  # Clip to 4 std devs

            # Random axis for rotation
            axis = np.random.normal(0, 1, 3)
            axis = axis / np.linalg.norm(axis)

            # Create small rotation quaternion
            half_angle = angle / 2.0
            dq_w = np.cos(half_angle)
            dq_xyz = np.sin(half_angle) * axis

            # Multiply quaternions
            q_new = np.array(
                [
                    quat[0] * dq_w
                    - quat[1] * dq_xyz[0]
                    - quat[2] * dq_xyz[1]
                    - quat[3] * dq_xyz[2],
                    quat[0] * dq_xyz[0]
                    + quat[1] * dq_w
                    + quat[2] * dq_xyz[2]
                    - quat[3] * dq_xyz[1],
                    quat[0] * dq_xyz[1]
                    - quat[1] * dq_xyz[2]
                    + quat[2] * dq_w
                    + quat[3] * dq_xyz[0],
                    quat[0] * dq_xyz[2]
                    + quat[1] * dq_xyz[1]
                    - quat[2] * dq_xyz[0]
                    + quat[3] * dq_w,
                ]
            )

            # Normalize and set
            q_new = normalize_quaternion(q_new)
            compound_state[1].w = q_new[0]
            compound_state[1].x = q_new[1]
            compound_state[1].y = q_new[2]
            compound_state[1].z = q_new[3]
            # Convert to list for comparison
            noisy_state = state2list(compound_state, system)

    # Get noisy state for comparison (if not already set)
    if system == "dublin_airplane" and is_array:
        noisy_state = state.copy()
    # For simple_car/pushing with arrays, noisy_state already set above
    # For OMPL objects, noisy_state already converted to list above

    # Calculate distance using our custom function
    distance_custom = arrayDistance(original_state, noisy_state, system)

    # Calculate distance using OMPL's state space distance function
    from ompl import base as ob

    try:
        if system == "dublin_airplane":
            # Create SE3 state space
            r3_space = ob.RealVectorStateSpace(3)
            so3_space = ob.SO3StateSpace()
            se3_space = ob.CompoundStateSpace()
            se3_space.addSubspace(r3_space, 1.0)
            se3_space.addSubspace(so3_space, 0.5)
            se3_space.lock()

            # Create OMPL states
            ompl_state1 = se3_space.allocState()
            ompl_state2 = se3_space.allocState()

            # Set position components
            ompl_state1[0][0] = original_state[0]
            ompl_state1[0][1] = original_state[1]
            ompl_state1[0][2] = original_state[2]
            ompl_state2[0][0] = noisy_state[0]
            ompl_state2[0][1] = noisy_state[1]
            ompl_state2[0][2] = noisy_state[2]

            # Set quaternion components (normalize first)
            quat1_norm = normalize_quaternion(
                [original_state[3], original_state[4], original_state[5], original_state[6]]
            )
            quat2_norm = normalize_quaternion(
                [noisy_state[3], noisy_state[4], noisy_state[5], noisy_state[6]]
            )

            ompl_state1[1].w = quat1_norm[0]
            ompl_state1[1].x = quat1_norm[1]
            ompl_state1[1].y = quat1_norm[2]
            ompl_state1[1].z = quat1_norm[3]
            ompl_state2[1].w = quat2_norm[0]
            ompl_state2[1].x = quat2_norm[1]
            ompl_state2[1].y = quat2_norm[2]
            ompl_state2[1].z = quat2_norm[3]

            # Compute distance using OMPL
            distance_ompl = se3_space.distance(ompl_state1, ompl_state2)

        elif system in ("simple_car", "kinematic_car", "pushing", "pushing_object"):
            # Create SE2 state space
            se2_space = ob.SE2StateSpace()
            bounds = ob.RealVectorBounds(2)
            bounds.setLow(-10.0)
            bounds.setHigh(10.0)
            se2_space.setBounds(bounds)

            # Create OMPL states
            ompl_state1 = se2_space.allocState()
            ompl_state2 = se2_space.allocState()

            # Set state components (allocState returns state object directly, not callable)
            ompl_state1.setX(original_state[0])
            ompl_state1.setY(original_state[1])
            ompl_state1.setYaw(original_state[2])
            ompl_state2.setX(noisy_state[0])
            ompl_state2.setY(noisy_state[1])
            ompl_state2.setYaw(noisy_state[2])

            # Compute distance using OMPL
            distance_ompl = se2_space.distance(ompl_state1, ompl_state2)
        else:
            distance_ompl = None
    except Exception as e:
        print(f"[DEBUG] [addNoise] Error computing OMPL distance: {e}")
        distance_ompl = None

    # Print states and distances
    # if system == "dublin_airplane":
    #     print(
    #         f"[DEBUG] [addNoise] Before noise: pos=[{original_state[0]:.6f}, {original_state[1]:.6f}, {original_state[2]:.6f}] quat=[{original_state[3]:.6f}, {original_state[4]:.6f}, {original_state[5]:.6f}, {original_state[6]:.6f}]"
    #     )
    #     print(
    #         f"[DEBUG] [addNoise] After noise:  pos=[{noisy_state[0]:.6f}, {noisy_state[1]:.6f}, {noisy_state[2]:.6f}] quat=[{noisy_state[3]:.6f}, {noisy_state[4]:.6f}, {noisy_state[5]:.6f}, {noisy_state[6]:.6f}]"
    #     )
    # else:
    #     print(f"[DEBUG] [addNoise] Before noise: {original_state}")
    #     print(f"[DEBUG] [addNoise] After noise:  {noisy_state}")

    # print(f"[DEBUG] [addNoise] Distance (custom): {distance_custom:.6f}")
    # if distance_ompl is not None:
    #     print(f"[DEBUG] [addNoise] Distance (OMPL):  {distance_ompl:.6f}")
    #     diff = abs(distance_custom - distance_ompl)
    #     if diff > 1e-6:
    #         print(f"[DEBUG] [addNoise] ⚠️  Distance mismatch! Difference: {diff:.6f}")
    #     else:
    #         print(f"[DEBUG] [addNoise] ✅ Distances match (diff: {diff:.9f})")
    # print(f"[DEBUG] [addNoise] Noise parameters: pos_std={pos_std:.6f}, rot_std={rot_std:.6f}")
    # # input("Press Enter to continue...")

    return state
