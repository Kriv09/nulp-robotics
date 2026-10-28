import numpy as np


def rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def rot_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


AXIS_ROTATIONS = {"X": rot_x, "Y": rot_y, "Z": rot_z}


def axis_rotation(axis, angle_rad):
    return AXIS_ROTATIONS[axis.upper()](angle_rad)


def to_radians(rotations_deg):
    result = []
    for axis, angle_deg in rotations_deg:
        if axis.upper() not in AXIS_ROTATIONS:
            raise ValueError(f"Unknown axis '{axis}', expected X, Y or Z")
        if not -180 <= angle_deg <= 180:
            raise ValueError(f"Angle {angle_deg} deg around {axis} is out of range [-180, 180]")
        result.append((axis.upper(), np.radians(angle_deg)))
    return result


def intrinsic_rotation_steps(rotations_rad, R_start=np.eye(3)):
    steps = [R_start]
    for axis, angle_rad in rotations_rad:
        steps.append(steps[-1] @ axis_rotation(axis, angle_rad))
    return steps


def intrinsic_rotation(rotations_rad, R_start=np.eye(3)):
    return intrinsic_rotation_steps(rotations_rad, R_start)[-1]


def are_same_rotation(R1, R2, tol=1e-6):
    return np.allclose(R1.T @ R2, np.eye(3), atol=tol)


def relative_rotation_angle_deg(R1, R2):
    cos_angle = (np.trace(R1.T @ R2) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def extrinsic_rotation_steps(rotations_rad, R_start=np.eye(3)):
    steps = [R_start]
    for axis, angle_rad in rotations_rad:
        steps.append(axis_rotation(axis, angle_rad) @ steps[-1])
    return steps


def extrinsic_rotation(rotations_rad, R_start=np.eye(3)):
    return extrinsic_rotation_steps(rotations_rad, R_start)[-1]


def rpy_to_rotations(rpy_deg):
    roll, pitch, yaw = rpy_deg
    return to_radians([("X", roll), ("Y", pitch), ("Z", yaw)])


def axis_angle_rotation(axis, angle_rad):
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -z, y],
                  [z, 0, -x],
                  [-y, x, 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


def rotation_to_axis_angle(R, eps=1e-9):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if angle < eps:
        return np.array([0.0, 0.0, 1.0]), 0.0  # no rotation, any axis works
    if np.pi - angle < 1e-6:
        # sin(angle) ~ 0, axis from R = 2 * a * a^T - I
        A = (R + np.eye(3)) / 2
        i = np.argmax(np.diag(A))
        axis = A[:, i] / np.sqrt(A[i, i])
        return axis / np.linalg.norm(axis), angle
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis, angle


def shortest_rotation_World(R_from, R_to):
    return rotation_to_axis_angle(R_to @ R_from.T)
