"""Exact nonlinear single-cart/single-pole dynamics for forward-dynamics MPC.

Coordinate convention
---------------------
q = [x, theta]

x      : cart translation along +x [m]
theta  : pole angle about Webots +y axis [rad], theta=0 is upright.
         Positive theta tilts the pole toward +x.

The cart is actuated by horizontal force F and the pole hinge is passive:

    generalized input = [F, 0].

The rigid-body dynamics are

    M(q) a + h(q, v) = [F, 0].

Unlike inverse-dynamics MPC, acceleration is NOT an optimization variable.
The MPC computes it inside the state-transition function:

    a = M(q)^(-1) ([F, 0] - h(q, v)).

This is the 2-DoF cart-pole analogue of a whole-body forward-dynamics / ABA MPC.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from config import (
    CART_MASS,
    POLE_MASS,
    POLE_COM_LENGTH,
    POLE_I_PIVOT_Y,
    GRAVITY,
)


def mass_matrix_ca(theta):
    """CasADi 2x2 generalized mass matrix M(q)."""
    m = POLE_MASS
    M = CART_MASS
    l = POLE_COM_LENGTH
    J = POLE_I_PIVOT_Y
    c = ca.cos(theta)
    return ca.vertcat(
        ca.horzcat(M + m, m * l * c),
        ca.horzcat(m * l * c, J),
    )


def bias_ca(theta, theta_dot):
    """CasADi bias term h(q,v): centrifugal + gravity."""
    m = POLE_MASS
    l = POLE_COM_LENGTH
    g = GRAVITY
    s = ca.sin(theta)
    return ca.vertcat(
        -m * l * s * theta_dot**2,
        -m * g * l * s,
    )


def forward_acceleration_ca(q, v, force):
    """CasADi forward dynamics a(q,v,F).

    Solves
        M(q) a + h(q,v) = [F, 0]
    for acceleration a.

    For this 2x2 teaching model the linear solve is written analytically.
    This keeps CasADi's expression graph fully expandable and plays well with
    IPOPT's exact derivatives. A high-DoF robot would normally use an ABA
    implementation rather than explicitly forming this formula.
    """
    theta = q[1]
    theta_dot = v[1]

    m = POLE_MASS
    M = CART_MASS
    l = POLE_COM_LENGTH
    J = POLE_I_PIVOT_Y
    g = GRAVITY

    c = ca.cos(theta)
    s = ca.sin(theta)

    # M(q) = [[A, B], [B, D]]
    A = M + m
    B = m * l * c
    D = J

    # rhs = [F, 0] - h(q,v)
    rhs_1 = force + m * l * s * theta_dot**2
    rhs_2 = m * g * l * s

    det = A * D - B**2
    x_ddot = (D * rhs_1 - B * rhs_2) / det
    theta_ddot = (-B * rhs_1 + A * rhs_2) / det

    return ca.vertcat(x_ddot, theta_ddot)


def state_transition_ca(state, force, dt):
    """Explicit-Euler discrete forward-dynamics transition.

    state = [x, theta, x_dot, theta_dot]

    q_{k+1} = q_k + v_k dt
    v_{k+1} = v_k + a(q_k,v_k,F_k) dt
    """
    q = state[0:2]
    v = state[2:4]
    a = forward_acceleration_ca(q, v, force)
    q_next = q + v * dt
    v_next = v + a * dt
    return ca.vertcat(q_next, v_next)


def dynamics_equation_residual_ca(q, v, force, acceleration):
    """Return M(q)a+h(q,v)-[F,0] for diagnostics only."""
    theta = q[1]
    theta_dot = v[1]
    return (
        mass_matrix_ca(theta) @ acceleration
        + bias_ca(theta, theta_dot)
        - ca.vertcat(force, 0.0)
    )


def mass_matrix_np(theta: float) -> np.ndarray:
    """NumPy version of M(q), used by offline plant validation."""
    m = POLE_MASS
    M = CART_MASS
    l = POLE_COM_LENGTH
    J = POLE_I_PIVOT_Y
    c = np.cos(theta)
    return np.array(
        [[M + m, m * l * c], [m * l * c, J]],
        dtype=float,
    )


def bias_np(theta: float, theta_dot: float) -> np.ndarray:
    """NumPy version of h(q,v), used by offline plant validation."""
    m = POLE_MASS
    l = POLE_COM_LENGTH
    g = GRAVITY
    s = np.sin(theta)
    return np.array(
        [-m * l * s * theta_dot**2, -m * g * l * s],
        dtype=float,
    )


def forward_acceleration_np(state: np.ndarray, force: float) -> np.ndarray:
    """Continuous-time forward acceleration for the external test plant."""
    theta = float(state[1])
    theta_dot = float(state[3])
    Mq = mass_matrix_np(theta)
    h = bias_np(theta, theta_dot)
    generalized_input = np.array([float(force), 0.0], dtype=float)
    return np.linalg.solve(Mq, generalized_input - h)


def dynamics_equation_residual_np(state: np.ndarray, force: float, acceleration: np.ndarray) -> np.ndarray:
    """NumPy residual M a + h - [F,0], used for diagnostics."""
    theta = float(state[1])
    theta_dot = float(state[3])
    return (
        mass_matrix_np(theta) @ np.asarray(acceleration, dtype=float).reshape(2)
        + bias_np(theta, theta_dot)
        - np.array([float(force), 0.0], dtype=float)
    )


def state_derivative_np(state: np.ndarray, force: float) -> np.ndarray:
    """Continuous-time state derivative for offline validation."""
    a = forward_acceleration_np(state, force)
    return np.array([state[2], state[3], a[0], a[1]], dtype=float)
