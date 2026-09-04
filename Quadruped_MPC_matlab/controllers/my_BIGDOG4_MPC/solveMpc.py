"""Convex model-predictive contact-force controller."""

import numpy as np
from scipy.linalg import expm

from Skew import Skew
from qp_solver import solve_qp


def c2qp(A, B, dt, horizon):
    """Discretize the continuous model and build stacked MPC matrices."""
    A = np.asarray(A, dtype=float).reshape(13, 13)
    B = np.asarray(B, dtype=float).reshape(13, 12)
    horizon = int(horizon)

    augmented = np.zeros((25, 25))
    augmented[:13, :13] = A
    augmented[:13, 13:] = B
    discrete = expm(float(dt) * augmented)
    Adt = discrete[:13, :13]
    Bdt = discrete[:13, 13:]

    powers = [np.eye(13)]
    for _ in range(horizon):
        powers.append(Adt @ powers[-1])

    A_qp = np.zeros((13 * horizon, 13))
    B_qp = np.zeros((13 * horizon, 12 * horizon))
    for row in range(horizon):
        A_qp[13 * row : 13 * (row + 1), :] = powers[row + 1]
        for column in range(row + 1):
            B_qp[
                13 * row : 13 * (row + 1),
                12 * column : 12 * (column + 1),
            ] = (
                powers[row - column] @ Bdt
            )
    return A_qp, B_qp


def solveMpc(R_yaw, I_inv, rbf, x0, xd, dt, horizon, gait, mass):
    horizon = int(horizon)
    R_yaw = np.asarray(R_yaw, dtype=float).reshape(3, 3)
    I_inv = np.asarray(I_inv, dtype=float).reshape(3, 3)
    rbf = np.asarray(rbf, dtype=float).reshape(3, 4)
    x0 = np.asarray(x0, dtype=float).reshape(13)
    xd = np.asarray(xd, dtype=float).reshape(13 * horizon)
    gait = np.asarray(gait, dtype=float).reshape(-1)
    if gait.size < 4 * horizon:
        raise ValueError("gait table is shorter than the MPC horizon")

    f_max = 140.0
    inverse_friction = 1.0 / 0.4
    alpha = 0.00002

    continuous_a = np.zeros((13, 13))
    continuous_a[:3, 6:9] = R_yaw.T
    continuous_a[3:6, 9:12] = np.eye(3)
    continuous_a[11, 12] = 1.0

    continuous_b = np.zeros((13, 12))
    for leg in range(4):
        force_slice = slice(3 * leg, 3 * leg + 3)
        continuous_b[6:9, force_slice] = I_inv @ Skew(rbf[:, leg])
        continuous_b[9:12, force_slice] = np.eye(3) / float(mass)

    A_qp, B_qp = c2qp(continuous_a, continuous_b, dt, horizon)
    full_weight = np.array(
        [25, 25, 10, 2, 2, 100, 0, 0, 0.3, 10, 10, 20, 0], dtype=float
    )
    state_weight = np.diag(np.tile(full_weight, horizon))
    hessian = 2.0 * (B_qp.T @ state_weight @ B_qp + alpha * np.eye(12 * horizon))
    gradient = 2.0 * B_qp.T @ state_weight @ (A_qp @ x0 - xd)

    friction_block = np.array(
        [
            [inverse_friction, 0.0, 1.0],
            [-inverse_friction, 0.0, 1.0],
            [0.0, inverse_friction, 1.0],
            [0.0, -inverse_friction, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    constraint_matrix = np.zeros((20 * horizon, 12 * horizon))
    constraint_lower = np.zeros(20 * horizon)
    constraint_upper = np.full(20 * horizon, 100000.0)
    for contact in range(4 * horizon):
        row_slice = slice(5 * contact, 5 * contact + 5)
        column_slice = slice(3 * contact, 3 * contact + 3)
        constraint_matrix[row_slice, column_slice] = friction_block
        constraint_upper[5 * contact + 4] = gait[contact] * f_max

    return solve_qp(
        hessian,
        gradient,
        A=constraint_matrix,
        lbA=constraint_lower,
        ubA=constraint_upper,
    )


solve_mpc = solveMpc
