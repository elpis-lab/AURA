from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class Pose:
    """
    A class representing a 3D pose with a position and an rotation.
    The rotation is a quaternion in (w, x, y, z) order as in Genesis.
    """

    def __init__(
        self,
        position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        rotation: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0]),
    ):
        """
        Initialize with position and rotation,
        rotation can be euler or quat
        """
        # convert to numpy arrays first
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        if not isinstance(rotation, np.ndarray):
            rotation = np.array(rotation)

        # assume euler if the input length is 3
        if len(rotation) == 3:
            euler = rotation
            rotation = euler_to_quat(euler)
        elif len(rotation) == 4:
            euler = quat_to_euler(rotation)
        else:
            raise ValueError(f"Invalid rotation: {rotation}")

        self.position = position
        self.rotation = rotation
        self.euler = euler

    @property
    def pr(self) -> tuple[np.ndarray, np.ndarray]:
        """Return position and rotation in a tuple"""
        return self.position, self.rotation

    @property
    def flat(self) -> np.ndarray:
        """Return position and rotation in a flat array"""
        return np.concatenate([*self.pr])

    @property
    def invert(self) -> Pose:
        """
        Invert a transform with `position` and `rotation` (w,x,y,z).
        For T(x) = R * x + p, its inverse is T_inv(x) = R_inv * (x - p).
        """
        r = R.from_quat(wxyz_to_xyzw(self.rotation))
        r_inv = r.inv()
        p_inv = -r_inv.apply(self.position)
        q_inv = xyzw_to_wxyz(r_inv.as_quat())
        return Pose(p_inv, q_inv)

    @property
    def homogenous(self) -> np.ndarray:
        """Return the 4x4 homogeneous matrix representation of the pose"""
        r = R.from_quat(wxyz_to_xyzw(self.rotation))
        return np.block([[r.as_matrix(), self.position[:, None]], [0, 0, 0, 1]])

    def __matmul__(self, other: Pose) -> Pose:
        """
        Compose two transforms T1 = (p1, q1) and T2 = (p2, q2).
        The resulting transform T = T1 * T2 is given by:
        p = p1 + R(q1).apply(p2)
        q = q1 * q2
        """
        p1, q1 = self.position, self.rotation
        p2, q2 = other.position, other.rotation

        r1 = R.from_quat(wxyz_to_xyzw(q1))
        r2 = R.from_quat(wxyz_to_xyzw(q2))
        p_new = p1 + r1.apply(p2)
        r_new = r1 * r2
        q_new = xyzw_to_wxyz(r_new.as_quat())
        return Pose(p_new, q_new)

    def __mul__(self, other: Pose) -> Pose:
        """Same as __matmul__"""
        return self.__matmul__(other)

    def same(self, other: Pose, threshold: float = 1e-3) -> float:
        """Check if two poses are the same"""
        ap, ao = self.pr
        bp, bo = other.pr
        # position should be close
        # rotation should be close to other's quat or -quat
        pos_close = np.allclose(ap, bp, atol=threshold)
        ori_close = np.allclose(ao, bo, atol=threshold) or np.allclose(ao, -bo, atol=threshold)
        return pos_close and ori_close

    def distance(self, other: Pose) -> tuple[float, float]:
        """
        Compute the distance between two poses.
        Return position and rotation distance.
        """
        ap, ao = self.pr
        bp, bo = other.pr
        # position distance (m)
        dp = np.linalg.norm(ap - bp)
        # rotation distance (rad)
        dist = np.min([np.abs(np.dot(ao, bo)), 1.0])  # avoid numerical errors
        do = 2 * np.arccos(dist)
        return dp, do

    def interpolate(self, other: Pose, alpha: float) -> Pose:
        """
        Interpolate between two poses.
        Return a pose that is alpha% between self and other.
        """
        # Interpolate position
        p_new = self.position + alpha * (other.position - self.position)

        # Interpolate rotation
        r1 = R.from_quat(wxyz_to_xyzw(self.rotation))
        r2 = R.from_quat(wxyz_to_xyzw(other.rotation))
        rotations = R.from_quat([r1.as_quat(), r2.as_quat()])
        r_interp = Slerp([0, 1], rotations)(alpha)
        q_new = xyzw_to_wxyz(r_interp.as_quat())

        return Pose(p_new, q_new)

    def copy(self) -> Pose:
        """Copy a pose"""
        return Pose(self.position.copy(), self.rotation.copy())

    def __repr__(self) -> str:
        """Return a string representation of the pose"""
        pos = [round(n, 3) for n in self.position.tolist()]
        quat = [round(n, 3) for n in self.rotation.tolist()]
        return f"Pose({pos}, {quat})"


class SE2Pose(Pose):
    """
    A class representing a SE2 pose with a position and an rotation,
    defined as (x, y, theta).
    Under the hood the pose operation is the same as 3D pose class Pose.
    """

    def __init__(
        self,
        position: np.ndarray = np.array([0.0, 0.0]),
        rotation: float = 0,
    ):
        """Initialize with position (x, y) and rotation theta"""
        position, euler = self.to_se3(position, rotation)
        super().__init__(position, euler)

    def to_se2(self, position: np.ndarray, euler: np.ndarray) -> tuple[np.ndarray, float]:
        """Return SE2 position and euler from a SE3 position and euler"""
        position = position[:2]
        euler = euler[2]
        return position, euler

    def to_se3(self, position: np.ndarray, euler: float) -> tuple[np.ndarray, np.ndarray]:
        """Return SE3 position and euler from a SE2 position and euler"""
        position = np.array([position[0], position[1], 0])
        euler = np.array([0.0, 0.0, euler])
        return position, euler

    @property
    def pr(self) -> tuple[np.ndarray, np.ndarray]:
        """Return position and rotation in a tuple"""
        position, euler = self.to_se2(self.position, self.euler)
        return position, euler

    @property
    def flat(self) -> np.ndarray:
        """Return position and rotation in a flat array"""
        position, euler = self.pr
        return np.array([position[0], position[1], euler])

    @property
    def invert(self) -> SE2Pose:
        """Invert a SE2 pose"""
        inverted = super().invert
        position, euler = self.to_se2(inverted.position, inverted.euler)
        return SE2Pose(position, euler)

    @property
    def homogenous(self) -> np.ndarray:
        """Return the 3x3 homogeneous matrix representation of the pose"""
        position, euler = self.to_se2(self.position, self.euler)
        r = R.from_euler("z", euler)
        return np.block([[r.as_matrix()[:2, :2], position[:, None]], [0, 0, 1]])

    def __matmul__(self, other: SE2Pose) -> SE2Pose:
        """SE2Pose multiplication"""
        pose = super().__matmul__(other)
        position, euler = self.to_se2(pose.position, pose.euler)
        return SE2Pose(position, euler)

    def interpolate(self, other: SE2Pose, alpha: float) -> SE2Pose:
        """Interpolate between two SE2 poses"""
        pose = super().interpolate(other, alpha)
        position, euler = self.to_se2(pose.position, pose.euler)
        return SE2Pose(position, euler)

    def copy(self) -> SE2Pose:
        """Copy a pose"""
        return SE2Pose(*self.pr)

    def distance(self, other: "SE2Pose", angular_weight: float = 0.5) -> float:
        """
        Distance to another pose (weighted Euclidean in SE(2))

        Args:
            other: Another SE2Pose
            angular_weight: Weight for angular component (default 0.5)

        Returns:
            Combined distance incorporating position and orientation
        """
        dist_pos = self.position_distance(other)
        dist_theta = self.angular_distance(other)
        return np.sqrt(dist_pos**2 + angular_weight * dist_theta**2)

    def position_distance(self, other: "SE2Pose") -> float:
        """
        Position-only distance (ignoring orientation)

        Args:
            other: Another SE2Pose

        Returns:
            Euclidean distance between positions
        """
        pos_self, _ = self.pr
        pos_other, _ = other.pr
        return np.linalg.norm(pos_self - pos_other)

    def angular_distance(self, other: "SE2Pose") -> float:
        """
        Angular distance only (in radians, normalized to [0, π])

        Args:
            other: Another SE2Pose

        Returns:
            Absolute angular difference in radians
        """
        _, theta_self = self.pr
        _, theta_other = other.pr

        # Calculate angular difference in [-π, π]
        dtheta = theta_self - theta_other
        while dtheta > np.pi:
            dtheta -= 2.0 * np.pi
        while dtheta < -np.pi:
            dtheta += 2.0 * np.pi

        return abs(dtheta)

    def __repr__(self) -> str:
        """Return a string representation of the pose"""
        position, euler = self.to_se2(self.position, self.euler)
        position = [round(n, 3) for n in position.tolist()]
        euler = round(euler, 3)
        return f"SE2 Pose({position[0]}, {position[1]}, {euler})"


# Helper quaternion functions
def euler_to_quat(
    angles: np.ndarray | list[float], seq: str | bool = "xyz", degrees: bool = False
) -> np.ndarray:
    """
    Convert Euler angles to Quaternion (w, x, y, z).
    Backward-compatible with older call sites that passed (angles, degrees).
    """
    if isinstance(seq, bool):
        degrees = seq
        seq = "xyz"
    r = R.from_euler(seq, angles, degrees=degrees)
    return xyzw_to_wxyz(r.as_quat())


def quat_to_euler(quat: np.ndarray | list[float], degrees: bool = False) -> np.ndarray:
    """Convert Quaternion (w, x, y, z) to Euler angles"""
    r = R.from_quat(wxyz_to_xyzw(quat))
    return r.as_euler("xyz", degrees=degrees)


# Utils for quaternion format conversion between scipy and genesis
def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) to (x, y, z, w)"""
    return np.roll(quat, -1)


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) to (w, x, y, z)"""
    return np.roll(quat, 1)


# Some other utils
def angle_diff(a: float, b: float) -> float:
    """Compute the signed angle difference between two angles"""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to the range [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quat_to_matrix(quat: np.ndarray | list[float]) -> np.ndarray:
    """Quaternion (w, x, y, z) to rotation matrix, supports batch shapes (..., 4)."""
    quat = np.asarray(quat)
    r = R.from_quat(wxyz_to_xyzw(quat))
    return r.as_matrix()


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion (w, x, y, z), supports batch shapes (..., 3, 3)."""
    matrix = np.asarray(matrix)
    r = R.from_matrix(matrix)
    return xyzw_to_wxyz(r.as_quat())


def matrix_to_flat(matrix: np.ndarray) -> np.ndarray:
    """Convert one homogeneous transform to ``[x, y, z, qw, qx, qy, qz]``."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got {matrix.shape}")
    return np.concatenate([matrix[:3, 3], matrix_to_quat(matrix[:3, :3])])


def project_se3_to_se2(
    poses: np.ndarray | list[float], axis=(0.0, 1.0, 0.0)
) -> np.ndarray:
    """Project flat SE(3) pose vectors to ``[x, y, yaw]``."""
    pose_array = np.asarray(poses, dtype=float)
    single = pose_array.ndim == 1
    pose_array = pose_array.reshape(-1, 7)
    rotations = R.from_quat(pose_array[:, [4, 5, 6, 3]])
    reference = np.broadcast_to(np.asarray(axis, dtype=float), (len(pose_array), 3))
    rotated = rotations.apply(reference)
    dots = np.sum(reference[:, :2] * rotated[:, :2], axis=1)
    cross = reference[:, 0] * rotated[:, 1] - reference[:, 1] * rotated[:, 0]
    result = np.column_stack((pose_array[:, :2], np.arctan2(cross, dots)))
    return result[0] if single else result


def flat_to_matrix(pose_flat: np.ndarray | list[float]) -> np.ndarray:
    """
    Convert flat pose(s) [x, y, z, qw, qx, qy, qz] to homogeneous matrix/matrices.
    Input shape: (7,) or (..., 7) -> output shape: (4,4) or (..., 4,4).
    """
    pose_flat = np.asarray(pose_flat, dtype=float)
    single = pose_flat.ndim == 1
    poses = pose_flat.reshape(-1, 7)

    t = np.broadcast_to(np.eye(4), (poses.shape[0], 4, 4)).copy()
    t[:, :3, :3] = quat_to_matrix(poses[:, 3:7])
    t[:, :3, 3] = poses[:, :3]

    if single:
        return t[0]
    return t.reshape(*pose_flat.shape[:-1], 4, 4)
