import numpy as np
from geometry.pose import Pose


def get_random_push(
    obj_pose,
    obj_shape,
    tool_offset=Pose(),
    rotation_range=(0, 4),  # rotation to push from
    side_range=(-0.4, 0.4),  # side relative offset of the push
    distance_range=(0, 0.3),  # distance of the push
    total_time=3,  # total time to complete the push
    dt=0.1,  # time step of the path
    max_speed=0.5,  # assume it will never exceed this speed
    max_acc=1,  # assume it will never exceed this acceleration
):
    """Get a random push parameter and the corresponding path"""
    push_params = generate_push_params(
        obj_shape,
        rotation_range=rotation_range,
        side_range=side_range,
        distance_range=distance_range,
    )

    times, ws_path = generate_path_form_params(
        obj_pose,
        obj_shape,
        push_params,
        tool_offset=tool_offset,
        total_time=total_time,
        dt=dt,
        max_speed=max_speed,
        max_acc=max_acc,
    )

    return push_params, times, ws_path


def generate_push_params(
    obj_shape,
    rotation_range=(0, 4),  # rotation to push from
    side_range=(-0.4, 0.4),  # absolute side offset of the push
    distance_range=(0, 0.3),  # distance of the push
):
    """Generate a random push parameter"""
    edge = np.random.randint(*rotation_range)
    rotation = edge * np.pi / 2

    # Use absolute offset directly, no need to multiply by edge length
    side = np.random.uniform(*side_range)
    distance = np.random.uniform(*distance_range)

    return (rotation, side, distance)


def generate_path_form_params(
    obj_pose,
    obj_shape,
    push_params,
    tool_offset=Pose(),
    total_time=3,
    dt=0.1,
    max_speed=0.5,
    max_acc=1,
    relative_push_offset=False,
):
    """Generate a workspace path from the push parameters"""
    rotation, side, distance = push_params
    # make sure rotation is discrete to pi/2
    rotations = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    diff = np.abs(rotation - rotations)
    rotation = rotations[np.argmin(diff)]

    # The object is approximated as an AABB
    w, l, h = obj_shape
    if (rotation / np.pi * 2) % 2 == 1:
        size = l
        side = w * side if relative_push_offset else side
    else:
        size = w
        side = l * side if relative_push_offset else side
    # Assume that the center of the object is the center of the AABB
    center_offset = np.zeros(2)

    # Get local path (x, y) w.r.t. the object
    # direction vectors
    dir_vector = np.array([np.cos(rotation), np.sin(rotation)])
    side_offset_vector = np.array([-dir_vector[1], dir_vector[0]])
    # small offset to avoid hitting object at the beginning
    pre_push_offset = 0.04
    # start point
    start = (
        (dir_vector * (size / 2 + pre_push_offset))
        + (side * side_offset_vector)
        + center_offset
    )
    distance += pre_push_offset

    # TODO - Move all these stuff to physics.py
    # Check constraints before generating path
    peak_speed = 2 * distance / total_time
    peak_acc = peak_speed * np.pi / total_time
    peak_speed = np.clip(peak_speed, 0, max_speed)
    peak_acc = np.clip(peak_acc, 0, max_acc)

    # Generate path
    # Sin velocity to complete this path from start to end
    # v(t) = peak_speed / 2 * (sin(2 * pi * t / T - pi / 2) + 1)
    def dist(t, peak_speed):
        return (-peak_speed * total_time / 4 / np.pi) * np.sin(
            2 * np.pi * t / total_time
        ) + (peak_speed * t / 2)

    # Generate path in local frame at 10 Hz
    n_points = int(total_time / dt)
    times = np.linspace(0, total_time, n_points)
    local_positions = [start - dir_vector * dist(t, peak_speed) for t in times]

    # Convert to local pose
    # Add z to path - assume adding z to be 0 in global frame
    local_poses = [
        Pose(position.tolist() + [-h / 2]) for position in local_positions
    ]
    # Add rotation and offset
    # ensure ee pointing down
    reflect_z = Pose([0, 0, 0], [np.pi, 0, 0])
    rotate_z = Pose([0, 0, 0], [0, 0, rotation])
    local_poses = [
        pose @ rotate_z @ reflect_z @ tool_offset for pose in local_poses
    ]

    # Convert to world frame
    ws_path = np.array([(obj_pose @ pose).flat for pose in local_poses])
    return times, ws_path
