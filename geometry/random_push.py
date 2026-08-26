import numpy as np

from geometry.pose import Pose, euler_to_quat, flat_to_matrix, matrix_to_quat, quat_to_matrix


def euler_to_matrix(seq: str, euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles to a rotation matrix."""
    return quat_to_matrix(euler_to_quat(euler, seq))


def get_sin_velocity_profile_peak(dist, duration):
    """Get peak speed and acceleration for a sinusoidal velocity profile."""
    peak_speed = 2 * dist / duration
    peak_acc = peak_speed * np.pi / duration
    return peak_speed, peak_acc


def sin_velocity_profile(t, dist, duration):
    """Get traveled distance at time t for a sinusoidal velocity profile."""
    v_max, _ = get_sin_velocity_profile_peak(dist, duration)
    scale = -v_max * duration / 4 / np.pi
    return scale * np.sin(2 * np.pi * t / duration) + (v_max * t / 2)


def generate_push_params(
    n_params: int,
    rotation_range=(0, 2 * np.pi),
    side_range=(-0.4, 0.4),
    distance_range=(0, 0.3),
):
    """Generate normalized push params: [face_id, relative_side, distance]."""
    rotations = np.random.uniform(*rotation_range, int(n_params))
    rotations = (rotations / (np.pi / 2)).astype(int) / 4
    sides = np.random.uniform(*side_range, int(n_params))
    distances = np.random.uniform(*distance_range, int(n_params))
    return np.stack([rotations, sides, distances], axis=-1)


def _as_tool_offset_flat(tool_offset) -> np.ndarray:
    if isinstance(tool_offset, Pose):
        return tool_offset.flat
    return np.asarray(tool_offset, dtype=float).reshape(7)


def _as_pose_flat(obj_pose) -> np.ndarray:
    if isinstance(obj_pose, Pose):
        return obj_pose.flat
    return np.asarray(obj_pose, dtype=float).reshape(7)


def _normalize_push_params(push_params: np.ndarray) -> np.ndarray:
    params = np.asarray(push_params, dtype=float).copy()
    params = params.reshape(-1, 3)
    normalized_faces = np.array([0.0, 0.25, 0.5, 0.75], dtype=float)
    rad_faces = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    for i, face_raw in enumerate(params[:, 0].copy()):
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
        params[i, 0] = normalized_faces[face_idx]
    params[:, 0] = np.mod(np.round(params[:, 0] * 4.0), 4.0) / 4.0
    params[:, 1] = np.clip(params[:, 1], -0.4, 0.4)
    params[:, 2] = np.clip(params[:, 2], 0.0, 0.3)
    return params


def generate_path_from_params(
    obj_states: np.ndarray,
    obj_shape: tuple[float, float, float],
    push_params: np.ndarray,
    tool_offset=np.array([0, 0, 0, 1, 0, 0, 0]),
    pre_push_offset: float = 0.02,
    duration: float = 2,
    dt: float = 0.1,
    max_speed: float = 0.5,
    max_acc: float = 1,
    total_time: float | None = None,
    push_height: float | None = None,
    relative_push_offset: bool = True,
):
    """Generate workspace paths used by the MuJoCo pushing simulator.

    push_params are [normalized_face, relative_side_offset, absolute_distance].
    The side offset is converted to meters by multiplying by the contacted edge.
    """
    if total_time is not None:
        duration = float(total_time)

    push_params = _normalize_push_params(push_params)
    obj_states = np.asarray(obj_states, dtype=float).reshape(-1, 7)
    assert push_params.ndim == 2 and push_params.shape[1] == 3
    assert obj_states.shape[0] == push_params.shape[0]
    n_data = push_params.shape[0]
    rotations, sides, distances = push_params.T

    push_sides = np.round(rotations * 4)
    rotations = push_sides * (np.pi / 2)

    w, l, _h = obj_shape
    mask_odd = push_sides % 2 == 1
    sizes = np.where(mask_odd, l, w)
    if relative_push_offset:
        sides = np.where(mask_odd, w * sides, l * sides)

    dir_vecs = np.stack([np.cos(rotations), np.sin(rotations)], axis=1)
    side_vecs = np.stack([-dir_vecs[:, 1], dir_vecs[:, 0]], axis=1)
    distances = distances + float(pre_push_offset)
    starts = (
        dir_vecs * (sizes / 2 + float(pre_push_offset))[:, None]
        + sides[:, None] * side_vecs
    )

    peak_speed, peak_acc = get_sin_velocity_profile_peak(distances, duration)
    if np.any(peak_speed > max_speed) or np.any(peak_acc > max_acc):
        raise ValueError("Push path exceeds max speed/acceleration constraints.")

    n_steps = int(duration / dt)
    t_paths = np.tile(np.linspace(0, duration, n_steps), (n_data, 1))
    dists = sin_velocity_profile(t_paths, distances[:, None], duration)

    local_xy = starts[:, None, :] - dists[:, :, None] * dir_vecs[:, None, :]
    local_z_value = 0.0 if push_height is None else float(push_height)
    local_z = np.full((n_data, n_steps, 1), local_z_value)
    local_pos = np.concatenate([local_xy, local_z], axis=2)

    t_rotate_z = np.tile(np.eye(4)[None, None, :, :], (n_data, 1, 1, 1))
    t_rotate_z[:, 0, :3, :3] = euler_to_matrix("z", rotations + np.pi)
    t_reflect_z = np.eye(4)[None, None, :, :]
    t_reflect_z[0, 0, :3, :3] = euler_to_matrix("x", np.pi)
    t_tool_offset = flat_to_matrix(_as_tool_offset_flat(tool_offset))[None, None, :, :]
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
    return t_paths, ws_paths


def generate_path_form_params(
    obj_pose,
    obj_shape,
    push_params,
    tool_offset=Pose(),
    total_time=3,
    dt=0.1,
    max_speed=0.5,
    max_acc=1,
    pre_push_offset=0.02,
    push_height=None,
    relative_push_offset=True,
):
    """Single-push wrapper around generate_path_from_params."""
    times, ws_paths = generate_path_from_params(
        _as_pose_flat(obj_pose)[None, :],
        obj_shape,
        np.asarray(push_params, dtype=float).reshape(1, 3),
        tool_offset=_as_tool_offset_flat(tool_offset),
        pre_push_offset=pre_push_offset,
        duration=total_time,
        dt=dt,
        max_speed=max_speed,
        max_acc=max_acc,
        push_height=push_height,
        relative_push_offset=relative_push_offset,
    )
    return times[0], ws_paths[0]


def get_random_push(
    n_params: int,
    obj_states: np.ndarray,
    obj_shape: tuple[float, float, float],
    tool_offset=np.array([0, 0, 0, 1, 0, 0, 0]),
    rotation_range=(0, 2 * np.pi),
    side_range=(-0.4, 0.4),
    distance_range=(0, 0.3),
    pre_push_offset: float = 0.02,
    duration: float = 2,
    dt: float = 0.1,
    max_speed: float = 0.5,
    max_acc: float = 1,
):
    """Get random relative push params and corresponding workspace paths."""
    push_params = generate_push_params(
        n_params, rotation_range, side_range, distance_range
    )
    times, ws_paths = generate_path_from_params(
        obj_states,
        obj_shape,
        push_params,
        tool_offset,
        pre_push_offset,
        duration,
        dt,
        max_speed,
        max_acc,
    )
    return push_params, times, ws_paths
