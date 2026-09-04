"""Forward kinematics for the four BigDog legs."""

import numpy as np


def _leg_position(q_leg, x_offset, side):
    q0, q1, q2 = q_leg
    q12 = q1 + q2

    x = (
        0.082 * np.sin(q1)
        - 0.175826 * np.cos(q1)
        + 0.192981 * np.cos(q12)
        - 0.00268261 * np.sin(q12)
        + x_offset
    )
    y = (
        side * 0.085 * np.cos(q0)
        + 0.082 * np.cos(q1) * np.sin(q0)
        + 0.175826 * np.sin(q0) * np.sin(q1)
        - 0.00268261 * np.cos(q12) * np.sin(q0)
        - 0.192981 * np.sin(q0) * np.sin(q12)
    )
    z = (
        0.00268261 * np.cos(q0) * np.cos(q12)
        - 0.082 * np.cos(q0) * np.cos(q1)
        - 0.175826 * np.cos(q0) * np.sin(q1)
        + side * 0.085 * np.sin(q0)
        + 0.192981 * np.cos(q0) * np.sin(q12)
    )
    return np.array([x, y, z])


def forwardKinematics(q):
    """Return foot positions relative to each hip and to the body frame.

    The input joint order is FR, FL, BR, BL with three joints per leg.
    Both returned arrays have shape ``(3, 4)``.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 12:
        raise ValueError("forwardKinematics expects 12 joint angles")

    x_offsets = (0.065, 0.065, -0.065, -0.065)
    side_signs = (-1.0, 1.0, -1.0, 1.0)
    rsf = np.column_stack(
        [
            _leg_position(q[3 * i : 3 * i + 3], x_offsets[i], side_signs[i])
            for i in range(4)
        ]
    )
    hip_offsets = np.array(
        [
            [0.139, 0.139, -0.139, -0.139],
            [-0.061, 0.061, -0.061, 0.061],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    return rsf, rsf + hip_offsets


forward_kinematics = forwardKinematics
