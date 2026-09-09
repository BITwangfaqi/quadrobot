"""Nonlinear inverse-dynamics MPC for a single-stage inverted pendulum.

Decision variables at every stage k:
    q_k = [x_k, theta_k]
    v_k = [x_dot_k, theta_dot_k]
    a_k = [x_ddot_k, theta_ddot_k]
    F_k = cart motor force

State propagation is purely kinematic (explicit Euler):
    q_{k+1} = q_k + v_k dt_k
    v_{k+1} = v_k + a_k dt_k

Dynamic feasibility is imposed separately as an inverse-dynamics path constraint:
    tau_id = M(q_k) a_k + h(q_k, v_k)
    tau_id[0] = F_k     # actuated cart coordinate
    tau_id[1] = 0       # passive pole hinge

This mirrors the core structure of the whole-body RNEA MPC in the paper, with a
2-DoF cart-pole replacing the floating-base legged robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np

from config import (
    N,
    DT_MIN,
    DT_MAX,
    FORCE_MAX,
    CART_POS_LIMIT,
    CART_VEL_LIMIT,
    POLE_RATE_LIMIT,
    Q_X,
    Q_THETA,
    Q_X_DOT,
    Q_THETA_DOT,
    QN_X,
    QN_THETA,
    QN_X_DOT,
    QN_THETA_DOT,
    R_FORCE,
    R_ACC_CART,
    R_ACC_POLE,
    IPOPT_MAX_ITER,
    IPOPT_TOL,
    IPOPT_ACCEPTABLE_TOL,
)
from dynamics import inverse_dynamics_ca


@dataclass
class MPCResult:
    force: float
    solve_time_ms: float
    success: bool
    status: str
    first_acceleration: np.ndarray
    unactuated_residual: float


class InverseDynamicsMPC:
    def __init__(self):
        self.N = N
        self.dts = self._build_geometric_time_grid(N, DT_MIN, DT_MAX)
        self.opti = ca.Opti()

        # State layout X = [x, theta, x_dot, theta_dot].
        self.X = self.opti.variable(4, N + 1)
        self.A = self.opti.variable(2, N)
        self.U = self.opti.variable(1, N)

        # Parameters updated at every MPC solve.
        self.x0_p = self.opti.parameter(4)
        self.x_ref_p = self.opti.parameter(2)  # [cart reference, pole-angle reference]

        self._build_problem()
        self._configure_solver()

        self.prev_X = None
        self.prev_A = None
        self.prev_U = None
        self.last_force = 0.0

    @staticmethod
    def _build_geometric_time_grid(n: int, dt_min: float, dt_max: float) -> np.ndarray:
        if n <= 1:
            return np.array([dt_min], dtype=float)
        gamma = (dt_max / dt_min) ** (1.0 / (n - 1))
        return np.array([dt_min * gamma**k for k in range(n)], dtype=float)

    def _build_problem(self):
        opti = self.opti

        # Initial-state equality constraint.
        opti.subject_to(self.X[:, 0] == self.x0_p)

        J = 0

        for k in range(self.N):
            dt = float(self.dts[k])
            xk = self.X[:, k]
            qk = xk[0:2]
            vk = xk[2:4]
            ak = self.A[:, k]
            uk = self.U[0, k]

            # --------------------------------------------------------------
            # Inverse-dynamics path constraint (the core of this demo).
            # --------------------------------------------------------------
            tau_id = inverse_dynamics_ca(qk, vk, ak)
            opti.subject_to(tau_id[0] == uk)   # cart is actuated
            opti.subject_to(tau_id[1] == 0.0)  # pole hinge is unactuated

            # Direct actuator limit, matching Webots LinearMotor.maxForce.
            opti.subject_to(opti.bounded(-FORCE_MAX, uk, FORCE_MAX))

            # --------------------------------------------------------------
            # Explicit-Euler state propagation (kinematic transition).
            # --------------------------------------------------------------
            q_next = qk + vk * dt
            v_next = vk + ak * dt
            opti.subject_to(self.X[:, k + 1] == ca.vertcat(q_next, v_next))

            # Soft task objective: center cart and keep pole upright.
            ex = xk[0] - self.x_ref_p[0]
            etheta = xk[1] - self.x_ref_p[1]
            J += Q_X * ex**2
            J += Q_THETA * etheta**2
            J += Q_X_DOT * xk[2]**2
            J += Q_THETA_DOT * xk[3]**2
            J += R_FORCE * uk**2
            J += R_ACC_CART * ak[0]**2
            J += R_ACC_POLE * ak[1]**2

        # Terminal objective.
        xN = self.X[:, self.N]
        exN = xN[0] - self.x_ref_p[0]
        ethetaN = xN[1] - self.x_ref_p[1]
        J += QN_X * exN**2
        J += QN_THETA * ethetaN**2
        J += QN_X_DOT * xN[2]**2
        J += QN_THETA_DOT * xN[3]**2

        # State bounds over the full horizon.
        opti.subject_to(opti.bounded(-CART_POS_LIMIT, self.X[0, :], CART_POS_LIMIT))
        opti.subject_to(opti.bounded(-CART_VEL_LIMIT, self.X[2, :], CART_VEL_LIMIT))
        opti.subject_to(opti.bounded(-POLE_RATE_LIMIT, self.X[3, :], POLE_RATE_LIMIT))

        opti.minimize(J)

    def _configure_solver(self):
        opts = {
            "expand": True,
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": IPOPT_MAX_ITER,
            "ipopt.tol": IPOPT_TOL,
            "ipopt.acceptable_tol": IPOPT_ACCEPTABLE_TOL,
            "ipopt.warm_start_init_point": "yes",
        }
        self.opti.solver("ipopt", opts)

    def _set_cold_start(self, state: np.ndarray):
        X_guess = np.tile(state.reshape(4, 1), (1, self.N + 1))
        A_guess = np.zeros((2, self.N), dtype=float)
        U_guess = np.zeros((1, self.N), dtype=float)
        self.opti.set_initial(self.X, X_guess)
        self.opti.set_initial(self.A, A_guess)
        self.opti.set_initial(self.U, U_guess)

    def _set_warm_start(self, state: np.ndarray):
        # Shift last trajectory by one node and overwrite node 0 with measurement.
        X_guess = np.hstack((self.prev_X[:, 1:], self.prev_X[:, -1:]))
        X_guess[:, 0] = state
        A_guess = np.hstack((self.prev_A[:, 1:], self.prev_A[:, -1:]))
        U_guess = np.hstack((self.prev_U[:, 1:], self.prev_U[:, -1:]))
        self.opti.set_initial(self.X, X_guess)
        self.opti.set_initial(self.A, A_guess)
        self.opti.set_initial(self.U, U_guess)

    def solve(self, state, cart_reference=0.0, pole_reference=0.0) -> MPCResult:
        state = np.asarray(state, dtype=float).reshape(4)
        self.opti.set_value(self.x0_p, state)
        self.opti.set_value(self.x_ref_p, [float(cart_reference), float(pole_reference)])

        if self.prev_X is None:
            self._set_cold_start(state)
        else:
            self._set_warm_start(state)

        tic = perf_counter()
        try:
            sol = self.opti.solve()
            solve_ms = (perf_counter() - tic) * 1000.0

            X_sol = np.asarray(sol.value(self.X), dtype=float)
            A_sol = np.asarray(sol.value(self.A), dtype=float)
            U_sol = np.asarray(sol.value(self.U), dtype=float).reshape(1, self.N)

            self.prev_X = X_sol
            self.prev_A = A_sol
            self.prev_U = U_sol

            force = float(U_sol[0, 0])
            first_a = A_sol[:, 0].copy()

            # Evaluate the passive-hinge inverse-dynamics residual at node 0.
            q0 = ca.DM(state[0:2])
            v0 = ca.DM(state[2:4])
            a0 = ca.DM(first_a)
            tau0 = np.asarray(inverse_dynamics_ca(q0, v0, a0), dtype=float).reshape(2)
            passive_residual = float(tau0[1])

            self.last_force = force
            return MPCResult(
                force=force,
                solve_time_ms=solve_ms,
                success=True,
                status=str(self.opti.stats().get("return_status", "Solve_Succeeded")),
                first_acceleration=first_a,
                unactuated_residual=passive_residual,
            )

        except RuntimeError:
            solve_ms = (perf_counter() - tic) * 1000.0
            status = str(self.opti.stats().get("return_status", "solver_failed"))
            # Keep the last valid MPC command rather than introducing another controller.
            return MPCResult(
                force=float(self.last_force),
                solve_time_ms=solve_ms,
                success=False,
                status=status,
                first_acceleration=np.zeros(2),
                unactuated_residual=np.nan,
            )
