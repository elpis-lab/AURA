import numpy as np
from utils.utils import state2list, isStateEqual, arrayDistance, log, normalize_quaternion

from ompl import base as ob
from ompl import util as ou
from ompl import control as oc


def getChildrenStates(ss, targetState, system="simple_car", tolerance=1e-6):
    """
    Extract children states and their corresponding controls from the OMPL planner tree.

    Args:
        ss: OMPL SimpleSetup object
        targetState: The target state to find children for
        system: The system type ("SE2" or "SE3") to determine state format
        tolerance: Tolerance for state comparison

    Returns:
        tuple: (children_states, children_controls)
    """
    print(f"[INFO] Getting children states for system: {system}")

    # Get planner data
    planner_data = oc.PlannerData(ss.getSpaceInformation())
    planner = ss.getPlanner()
    planner.getPlannerData(planner_data)

    num_vertices = planner_data.numVertices()
    print(f"[INFO] Planner tree has {num_vertices} vertices")

    if num_vertices == 0:
        log("[WARNING] Planner tree is empty", "warning")
        return [], []

    # Search for the target state
    targetVertexIdx = None
    print(f"[DEBUG] Searching for targetState: {targetState}")
    print(f"[DEBUG] Target state type: {type(targetState)}")

    # Enhanced debug for dublin_airplane system
    if system == "dublin_airplane":
        print(f"[DEBUG] [DUBLIN_AIRPLANE] Target state format: [x, y, z, qw, qx, qy, qz]")
        if isinstance(targetState, (list, tuple, np.ndarray)) and len(targetState) >= 7:
            print(
                f"[DEBUG] [DUBLIN_AIRPLANE] Target pos: [{targetState[0]:.6f}, {targetState[1]:.6f}, {targetState[2]:.6f}]"
            )
            print(
                f"[DEBUG] [DUBLIN_AIRPLANE] Target quat: [{targetState[3]:.6f}, {targetState[4]:.6f}, {targetState[5]:.6f}, {targetState[6]:.6f}]"
            )

    print(f"[DEBUG] First 10 planner vertices:")
    # for i in range(min(10, num_vertices)):
    #     state = planner_data.getVertex(i).getState()
    #     state_list = state2list(state, system)
    #     if system == "dublin_airplane":
    #         print(
    #             f"[DEBUG]   Vertex {i}: pos=[{state_list[0]:.6f}, {state_list[1]:.6f}, {state_list[2]:.6f}] quat=[{state_list[3]:.6f}, {state_list[4]:.6f}, {state_list[5]:.6f}, {state_list[6]:.6f}]"
    #         )
    #     else:
    #         print(f"[DEBUG]   Vertex {i}: {state_list}")

    print(f"[DEBUG] Searching through {num_vertices} vertices with tolerance: {tolerance}")
    for i in range(num_vertices):
        state = planner_data.getVertex(i).getState()
        state_list = state2list(state, system)

        if isStateEqual(state_list, targetState, system, tolerance):
            targetVertexIdx = i
            print(f"[INFO] Found target state at vertex index: {i}")
            if system == "dublin_airplane":
                print(f"[DEBUG] [DUBLIN_AIRPLANE] Match found:")
                print(
                    f"[DEBUG]   Target pos: [{targetState[0]:.6f}, {targetState[1]:.6f}, {targetState[2]:.6f}]"
                )
                print(
                    f"[DEBUG]   Found pos:  [{state_list[0]:.6f}, {state_list[1]:.6f}, {state_list[2]:.6f}]"
                )
                print(
                    f"[DEBUG]   Target quat: [{targetState[3]:.6f}, {targetState[4]:.6f}, {targetState[5]:.6f}, {targetState[6]:.6f}]"
                )
                print(
                    f"[DEBUG]   Found quat:  [{state_list[3]:.6f}, {state_list[4]:.6f}, {state_list[5]:.6f}, {state_list[6]:.6f}]"
                )
            break

    if targetVertexIdx is None:
        log(
            f"[WARNING] State {targetState} not found in planner tree",
            "warning",
        )

        # Enhanced debug output for dublin_airplane
        if system == "dublin_airplane":
            print(f"[DEBUG] [DUBLIN_AIRPLANE] Detailed comparison with first 10 vertices:")
            for i in range(min(10, num_vertices)):
                state = planner_data.getVertex(i).getState()
                state_list = state2list(state, system)

                # Calculate position difference
                pos_diff = np.sqrt(
                    (targetState[0] - state_list[0]) ** 2
                    + (targetState[1] - state_list[1]) ** 2
                    + (targetState[2] - state_list[2]) ** 2
                )

                # Calculate quaternion difference (handle sign ambiguity)
                from utils.utils import normalize_quaternion

                quat1_norm = normalize_quaternion(
                    [targetState[3], targetState[4], targetState[5], targetState[6]]
                )
                quat2_norm = normalize_quaternion(
                    [state_list[3], state_list[4], state_list[5], state_list[6]]
                )

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

                distance = arrayDistance(targetState, state_list, system)
                print(
                    f"[DEBUG]   Vertex {i}: pos_diff={pos_diff:.6f}, quat_diff={quat_diff:.6f}, total_dist={distance:.6f}"
                )
                print(
                    f"[DEBUG]     Target: pos=[{targetState[0]:.6f}, {targetState[1]:.6f}, {targetState[2]:.6f}] quat=[{targetState[3]:.6f}, {targetState[4]:.6f}, {targetState[5]:.6f}, {targetState[6]:.6f}]"
                )
                print(
                    f"[DEBUG]     Vertex: pos=[{state_list[0]:.6f}, {state_list[1]:.6f}, {state_list[2]:.6f}] quat=[{state_list[3]:.6f}, {state_list[4]:.6f}, {state_list[5]:.6f}, {state_list[6]:.6f}]"
                )

        # Also check if any vertex is close to the target
        print(f"[DEBUG] Checking for close matches (tolerance: {tolerance}):")
        min_distance = float("inf")
        closest_vertex = None
        closest_vertices = []  # Store top 5 closest

        for i in range(num_vertices):
            state = planner_data.getVertex(i).getState()
            state_list = state2list(state, system)
            distance = arrayDistance(targetState, state_list, system)
            if distance < min_distance:
                min_distance = distance
                closest_vertex = (i, state_list)

            # Keep top 5 closest
            closest_vertices.append((i, state_list, distance))
            if len(closest_vertices) > 5:
                closest_vertices.sort(key=lambda x: x[2])
                closest_vertices = closest_vertices[:5]

        closest_vertices.sort(key=lambda x: x[2])

        if closest_vertex:
            print(f"[INFO] Closest vertex: {closest_vertex[1]} (distance: {min_distance:.6f})")
            if system == "dublin_airplane":
                print(f"[DEBUG] [DUBLIN_AIRPLANE] Closest vertex breakdown:")
                print(f"[DEBUG]   Index: {closest_vertex[0]}")
                print(
                    f"[DEBUG]   Target pos: [{targetState[0]:.6f}, {targetState[1]:.6f}, {targetState[2]:.6f}]"
                )
                print(
                    f"[DEBUG]   Closest pos: [{closest_vertex[1][0]:.6f}, {closest_vertex[1][1]:.6f}, {closest_vertex[1][2]:.6f}]"
                )
                print(
                    f"[DEBUG]   Target quat: [{targetState[3]:.6f}, {targetState[4]:.6f}, {targetState[5]:.6f}, {targetState[6]:.6f}]"
                )
                print(
                    f"[DEBUG]   Closest quat: [{closest_vertex[1][3]:.6f}, {closest_vertex[1][4]:.6f}, {closest_vertex[1][5]:.6f}, {closest_vertex[1][6]:.6f}]"
                )

        if system == "dublin_airplane" and len(closest_vertices) > 0:
            print(f"[DEBUG] [DUBLIN_AIRPLANE] Top 5 closest vertices:")
            for idx, (i, state_list, dist) in enumerate(closest_vertices):
                print(f"[DEBUG]   #{idx+1} Vertex {i}: distance={dist:.6f}")
                print(
                    f"[DEBUG]     pos=[{state_list[0]:.6f}, {state_list[1]:.6f}, {state_list[2]:.6f}] quat=[{state_list[3]:.6f}, {state_list[4]:.6f}, {state_list[5]:.6f}, {state_list[6]:.6f}]"
                )

        return [], []

    # print(f"🔍 Getting edges for vertex {targetVertexIdx}...")
    childVertexIndices = ou.vectorUint()
    planner_data.getEdges(targetVertexIdx, childVertexIndices)

    print(f"[INFO] Found {len(childVertexIndices)} child vertices")

    children_states = []
    children_controls = []
    control_space = ss.getControlSpace()
    control_dimension = control_space.getDimension()

    for childVertexIdx in childVertexIndices:
        childState = planner_data.getVertex(childVertexIdx).getState()
        child_state_list = state2list(childState, system)
        children_states.append(child_state_list)

        # Get the control that takes us from parent (targetState) to this child
        try:
            # Get the edge from parent to child using indices
            edge = planner_data.getEdge(targetVertexIdx, childVertexIdx)

            # Get control directly from the edge
            control = edge.getControl()
            control_values = [control[j] for j in range(control_dimension)]
            children_controls.append(control_values)
            # print(f"   Child {childVertexIdx}: Control: {control_values}")

        except Exception as e:
            print(f"   [WARNING] Could not get control for edge to child {childVertexIdx}: {e}")
            # Use a fallback control if the direct method fails
            fallback_control = [1.0, 0.0, 0.1]
            children_controls.append(fallback_control)

    print(
        f"[INFO] Returning {len(children_states)} children states and {len(children_controls)} controls"
    )
    return children_states, children_controls


def sampleRandomState(system, state, numStates=1000, posSTD=0.003, rotSTD=0.05):
    sampledStates = []

    if system in ("simple_car", "kinematic_car", "pushing", "pushing_object"):
        # Convert state to list if it's not already
        if hasattr(state, "getX"):  # It's an OMPL state object
            stateList = state2list(state, "SE2")
        else:  # It's already a list
            stateList = state
        for _ in range(numStates):
            noisyX = stateList[0] + np.random.normal(0, posSTD)
            noisyY = stateList[1] + np.random.normal(0, posSTD)
            noisyYaw = stateList[2] + np.random.normal(0, rotSTD)
            while noisyYaw > np.pi:
                noisyYaw -= 2 * np.pi
            while noisyYaw < -np.pi:
                noisyYaw += 2 * np.pi
            sampledStates.append([noisyX, noisyY, noisyYaw])

    elif system == "dublin_airplane":
        # Convert state to list if it's not already
        if hasattr(state, "getX") or (
            callable(state) and not isinstance(state, (list, tuple, np.ndarray))
        ):
            # It's an OMPL state object
            stateList = state2list(state, "dublin_airplane")
        else:  # It's already a list or numpy array
            stateList = (
                state
                if isinstance(state, list)
                else state.tolist() if isinstance(state, np.ndarray) else list(state)
            )

        for _ in range(numStates):
            # For dublin_airplane, stateList format is [x, y, z, qw, qx, qy, qz]
            # We'll sample position and apply a small random rotation to the quaternion
            noisyX = stateList[0] + np.random.normal(0, posSTD)
            noisyY = stateList[1] + np.random.normal(0, posSTD)
            noisyZ = stateList[2] + np.random.normal(0, posSTD)

            # Get current quaternion components
            qw, qx, qy, qz = stateList[3], stateList[4], stateList[5], stateList[6]

            # Generate a small random rotation vector (axis-angle)
            # Sample a random direction on the sphere
            axis = np.random.normal(0, 1, 3)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-10:
                axis = axis / axis_norm
            else:
                axis = np.array([0.0, 0.0, 1.0])  # Default to Z axis if 0 vector

            # Sample angle from Gaussian
            angle = np.random.normal(0, rotSTD)

            # Create perturbation quaternion dq = [cos(angle/2), sin(angle/2)*axis]
            dq_w = np.cos(angle / 2.0)
            dq_xyz = np.sin(angle / 2.0) * axis

            # Apply perturbation: q_new = q_current * dq
            # q1 * q2 = (w1w2 - v1.v2, w1v2 + w2v1 + v1 x v2)
            w = qw * dq_w - qx * dq_xyz[0] - qy * dq_xyz[1] - qz * dq_xyz[2]
            x = qw * dq_xyz[0] + qx * dq_w + qy * dq_xyz[2] - qz * dq_xyz[1]
            y = qw * dq_xyz[1] - qx * dq_xyz[2] + qy * dq_w + qz * dq_xyz[0]
            z = qw * dq_xyz[2] + qx * dq_xyz[1] - qy * dq_xyz[0] + qz * dq_w

            # Normalize to be safe
            norm = np.sqrt(w * w + x * x + y * y + z * z)
            if norm > 1e-10:
                sampledStates.append(
                    [noisyX, noisyY, noisyZ, w / norm, x / norm, y / norm, z / norm]
                )
            else:
                sampledStates.append([noisyX, noisyY, noisyZ, 1.0, 0.0, 0.0, 0.0])

    return sampledStates
