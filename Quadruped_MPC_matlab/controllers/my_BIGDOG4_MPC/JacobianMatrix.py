"""Analytical foot Jacobians for the four BigDog legs."""

import numpy as np


def _leg_jacobian(q_leg, side):
    q0, q1, q2 = q_leg
    q12 = q1 + q2
    s0, c0 = np.sin(q0), np.cos(q0)
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q12), np.cos(q12)
    proximal = 0.0819823

    return np.array(
        [
            [
                0.0,
                proximal * c1 - 0.192981 * s12 - 0.00268261 * c12 + 0.175826 * s1,
                -0.00268261 * c12 - 0.192981 * s12,
            ],
            [
                -side * 0.085 * s0
                + proximal * c0 * c1
                + 0.175826 * c0 * s1
                - 0.00268261 * c12 * c0
                - 0.192981 * s12 * c0,
                0.175826 * c1 * s0
                - proximal * s0 * s1
                - 0.192981 * c12 * s0
                + 0.00268261 * s12 * s0,
                0.00268261 * s12 * s0 - 0.192981 * c12 * s0,
            ],
            [
                side * 0.085 * c0
                + proximal * c1 * s0
                + 0.175826 * s0 * s1
                - 0.00268261 * c12 * s0
                - 0.192981 * s12 * s0,
                proximal * c0 * s1
                - 0.175826 * c0 * c1
                + 0.192981 * c12 * c0
                - 0.00268261 * s12 * c0,
                0.192981 * c12 * c0 - 0.00268261 * s12 * c0,
            ],
        ]
    )


def JacobianMatrix(q):
    """Return the 3x12 block Jacobian in FR, FL, BR, BL order."""
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 12:
        raise ValueError("JacobianMatrix expects 12 joint angles")
    side_signs = (-1.0, 1.0, -1.0, 1.0)
    return np.hstack(
        [_leg_jacobian(q[3 * i : 3 * i + 3], side_signs[i]) for i in range(4)]
    )


jacobian_matrix = JacobianMatrix
