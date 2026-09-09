"""Nonlinear forward-dynamics MPC for a single-stage inverted pendulum.

Decision variables at stage k:
    X_k = [x_k, theta_k, x_dot_k, theta_dot_k]
    F_k = cart motor force

Acceleration is NOT a decision variable. It is computed from forward dynamics:

    a_k = M(q_k)^(-1) ([F_k, 0] - h(q_k, v_k)).

The nonlinear dynamics are embedded directly in the state-transition constraint:

    q_{k+1} = q_k + v_k dt_k
    v_{k+1} = v_k + a_k dt_k

This is the cart-pole analogue of the forward-dynamics / ABA MPC formulation.
It is deliberately structured as the direct counterpart to the inverse-dynamics
MPC demo: same plant, same state, same force input, same horizon and weights.
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
    IPOPT_MAX_ITER,
    IPOPT_TOL,
    IPOPT_ACCEPTABLE_TOL,
)
from dynamics import (
    forward_acceleration_ca,
    state_transition_ca,
    dynamics_equation_residual_ca,
)


@dataclass
class MPCResult:
    force: float
    solve_time_ms: float
    success: bool
    status: str
    first_acceleration: np.ndarray
    dynamics_residual_norm: float


class ForwardDynamicsMPC:
    def __init__(self):
        self.N = N
        self.dts = self._build_geometric_time_grid(N, DT_MIN, DT_MAX)
        self.opti = ca.Opti()

        # State layout X = [x, theta, x_dot, theta_dot].
        self.X = self.opti.variable(4, N + 1)

        # The only control variable is cart force F.
        self.U = self.opti.variable(1, N)

        # Parameters updated at each receding-horizon solve.
        self.x0_p = self.opti.parameter(4)
        self.x_ref_p = self.opti.parameter(2)  # [cart reference, pole-angle reference]

        self._build_problem()
        self._configure_solver()

        self.prev_X = None
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
            uk = self.U[0, k]

            # --------------------------------------------------------------
            # Forward dynamics embedded in the state transition.
            # There is no acceleration decision variable and no separate
            # inverse-dynamics path constraint.
            # --------------------------------------------------------------
            x_next = state_transition_ca(xk, uk, dt)
            opti.subject_to(self.X[:, k + 1] == x_next)

            # Direct actuator limit, matching Webots LinearMotor.maxForce.
            opti.subject_to(opti.bounded(-FORCE_MAX, uk, FORCE_MAX))

            # Soft task objective: center cart and keep pole upright.
            ex = xk[0] - self.x_ref_p[0]
            etheta = xk[1] - self.x_ref_p[1]
            J += Q_X * ex**2
            J += Q_THETA * etheta**2
            J += Q_X_DOT * xk[2]**2
            J += Q_THETA_DOT * xk[3]**2
            J += R_FORCE * uk**2

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
        U_guess = np.zeros((1, self.N), dtype=float)
        self.opti.set_initial(self.X, X_guess)
        self.opti.set_initial(self.U, U_guess)

    def _set_warm_start(self, state: np.ndarray):
        # Shift last trajectory by one node and overwrite node 0 with measurement.
        X_guess = np.hstack((self.prev_X[:, 1:], self.prev_X[:, -1:]))
        X_guess[:, 0] = state
        U_guess = np.hstack((self.prev_U[:, 1:], self.prev_U[:, -1:]))
        self.opti.set_initial(self.X, X_guess)
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
            U_sol = np.asarray(sol.value(self.U), dtype=float).reshape(1, self.N)

            self.prev_X = X_sol
            self.prev_U = U_sol

            force = float(U_sol[0, 0])

            # Acceleration is reconstructed from forward dynamics; it was not
            # optimized independently.
            q0 = ca.DM(state[0:2])
            v0 = ca.DM(state[2:4])
            a0_dm = forward_acceleration_ca(q0, v0, force)
            a0 = np.asarray(a0_dm, dtype=float).reshape(2)

            residual_dm = dynamics_equation_residual_ca(q0, v0, force, a0_dm)
            residual = np.asarray(residual_dm, dtype=float).reshape(2)
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))

            self.last_force = force
            return MPCResult(
                force=force,
                solve_time_ms=solve_ms,
                success=True,
                status=str(self.opti.stats().get("return_status", "Solve_Succeeded")),
                first_acceleration=a0,
                dynamics_residual_norm=residual_norm,
            )

        except RuntimeError:
            solve_ms = (perf_counter() - tic) * 1000.0
            try:
                status = str(self.opti.stats().get("return_status", "solver_failed"))
            except RuntimeError:
                status = "solver_failed_before_stats"
            return MPCResult(
                force=float(self.last_force),
                solve_time_ms=solve_ms,
                success=False,
                status=status,
                first_acceleration=np.zeros(2),
                dynamics_residual_norm=np.nan,
            )
