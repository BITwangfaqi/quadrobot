"""Static stance-force quadratic program."""

import numpy as np

from Skew import Skew
from qp_solver import solve_qp


def qp(rbf, b_control, flag):
    rbf = np.asarray(rbf, dtype=float).reshape(3, 4)
    b_control = np.asarray(b_control, dtype=float).reshape(6)
    flag = np.asarray(flag, dtype=float).reshape(4)

    control_matrix = np.vstack(
        (
            np.hstack([flag[i] * np.eye(3) for i in range(4)]),
            np.hstack([flag[i] * Skew(rbf[:, i]) for i in range(4)]),
        )
    )
    state_weight = np.diag([1.0, 1.0, 10.0, 50.0, 30.0, 10.0])
    hessian = 2.0 * control_matrix.T @ state_weight @ control_matrix
    hessian += 2.0 * 0.01 * 0.001 * np.eye(12)
    gradient = -2.0 * control_matrix.T @ state_weight @ b_control

    friction = 0.5
    force_block = np.array(
        [
            [1.0, 0.0, -friction],
            [1.0, 0.0, friction],
            [0.0, 1.0, -friction],
            [0.0, 1.0, friction],
            [0.0, 0.0, 1.0],
        ]
    )
    constraint_matrix = np.zeros((20, 12))
    variable_lower = np.zeros(12)
    variable_upper = np.zeros(12)
    constraint_lower = np.zeros(20)
    constraint_upper = np.zeros(20)

    for leg in range(4):
        force_slice = slice(3 * leg, 3 * leg + 3)
        constraint_slice = slice(5 * leg, 5 * leg + 5)
        constraint_matrix[constraint_slice, force_slice] = flag[leg] * force_block
        variable_lower[force_slice] = -flag[leg] * 100000.0
        variable_upper[force_slice] = flag[leg] * 100000.0
        constraint_lower[constraint_slice] = [
            -flag[leg] * 100000.0,
            0.0,
            -flag[leg] * 100000.0,
            0.0,
            flag[leg] * 10.0,
        ]
        constraint_upper[constraint_slice] = [
            0.0,
            flag[leg] * 100000.0,
            0.0,
            flag[leg] * 100000.0,
            flag[leg] * 160.0,
        ]

    return solve_qp(
        hessian,
        gradient,
        constraint_matrix,
        variable_lower,
        variable_upper,
        constraint_lower,
        constraint_upper,
    )
