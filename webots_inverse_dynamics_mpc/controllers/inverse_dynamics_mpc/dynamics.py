"""Exact nonlinear single-cart/single-pole dynamics used by the MPC model.

Coordinate convention
---------------------
q = [x, theta]

x      : cart translation along +x [m]
theta  : pole angle about Webots +y axis [rad], theta=0 is upright.
         Positive theta tilts the pole toward +x.

The cart is actuated by horizontal force F. The hinge is passive.
Therefore the generalized actuator vector is [F, 0].

The inverse-dynamics equation used by the MPC is

    tau_id(q, v, a) = M(q) a + h(q, v) = [F, 0].

This is the cart-pole analogue of the paper's whole-body inverse-dynamics
path constraint. Here the second generalized coordinate (the pole hinge)
is unactuated, so its inverse-dynamics row must be zero.
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


def inverse_dynamics_ca(q, v, a):
    """Return generalized inverse-dynamics effort [F_required, tau_hinge_required]."""
    theta = q[1]
    theta_dot = v[1]
    return mass_matrix_ca(theta) @ a + bias_ca(theta, theta_dot)


def mass_matrix_np(theta: float) -> np.ndarray:
    """NumPy version of M(q), used only by offline plant validation."""
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
    """NumPy version of h(q,v), used only by offline plant validation."""
    m = POLE_MASS
    l = POLE_COM_LENGTH
    g = GRAVITY
    s = np.sin(theta)
    return np.array(
        [-m * l * s * theta_dot**2, -m * g * l * s],
        dtype=float,
    )


def forward_acceleration_np(state: np.ndarray, force: float) -> np.ndarray:
    """Forward dynamics for an external plant simulator, not for the MPC.

    Solves
        M(q) a + h(q,v) = [F, 0]
    for a.
    """
    theta = float(state[1])
    theta_dot = float(state[3])
    M = mass_matrix_np(theta)
    h = bias_np(theta, theta_dot)
    rhs = np.array([float(force), 0.0]) - h
    return np.linalg.solve(M, rhs)


def state_derivative_np(state: np.ndarray, force: float) -> np.ndarray:
    """Continuous-time state derivative for offline validation."""
    a = forward_acceleration_np(state, force)
    return np.array([state[2], state[3], a[0], a[1]], dtype=float)
