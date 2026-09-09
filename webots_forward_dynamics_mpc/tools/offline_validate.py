"""Offline closed-loop validation of the forward-dynamics MPC.

The MPC embeds forward dynamics in its state-transition constraints. The test
plant is integrated independently with RK4 using the same continuous nonlinear
rigid-body model. In Webots, this external plant is replaced by Webots/ODE.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CTRL_DIR = ROOT / "controllers" / "forward_dynamics_mpc"
sys.path.insert(0, str(CTRL_DIR))

from config import (  # noqa: E402
    INITIAL_POLE_ANGLE,
    MPC_PERIOD_S,
    CART_REFERENCE,
    POLE_REFERENCE,
    FORCE_MAX,
)
from dynamics import (  # noqa: E402
    state_derivative_np,
    forward_acceleration_np,
    dynamics_equation_residual_np,
)
from mpc import ForwardDynamicsMPC  # noqa: E402


def rk4_step(state: np.ndarray, force: float, dt: float) -> np.ndarray:
    k1 = state_derivative_np(state, force)
    k2 = state_derivative_np(state + 0.5 * dt * k1, force)
    k3 = state_derivative_np(state + 0.5 * dt * k2, force)
    k4 = state_derivative_np(state + dt * k3, force)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def main():
    mpc = ForwardDynamicsMPC()

    state = np.array([0.0, INITIAL_POLE_ANGLE, 0.0, 0.0], dtype=float)
    sim_dt = 0.002
    total_time = 15.0
    next_mpc_t = 0.0
    force = 0.0

    out = ROOT / "logs" / "offline_validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    max_abs_theta = abs(state[1])
    max_abs_x = abs(state[0])
    solve_times = []
    max_fd_res = 0.0

    print("\n=== Offline forward-dynamics MPC validation ===")
    print(f"initial theta: {state[1]:+.4f} rad")

    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "x",
                "theta",
                "x_dot",
                "theta_dot",
                "force",
                "solve_ms",
                "fd_equation_residual_inf",
            ]
        )

        steps = int(total_time / sim_dt)
        last_solve_ms = np.nan

        for i in range(steps + 1):
            t = i * sim_dt

            if t + 1e-12 >= next_mpc_t:
                result = mpc.solve(state, CART_REFERENCE, POLE_REFERENCE)
                force = float(np.clip(result.force, -FORCE_MAX, FORCE_MAX))
                last_solve_ms = result.solve_time_ms
                solve_times.append(result.solve_time_ms)
                if np.isfinite(result.dynamics_residual_norm):
                    max_fd_res = max(max_fd_res, abs(result.dynamics_residual_norm))
                next_mpc_t += MPC_PERIOD_S

            # Independently check M a + h = [F,0] for the plant acceleration.
            plant_a = forward_acceleration_np(state, force)
            plant_res = dynamics_equation_residual_np(state, force, plant_a)
            max_fd_res = max(max_fd_res, float(np.linalg.norm(plant_res, ord=np.inf)))

            writer.writerow([t, *state, force, last_solve_ms, max_fd_res])

            max_abs_theta = max(max_abs_theta, abs(state[1]))
            max_abs_x = max(max_abs_x, abs(state[0]))

            state = rk4_step(state, force, sim_dt)

    print(f"final state       : {state}")
    print(f"max |theta|       : {max_abs_theta:.6f} rad")
    print(f"max |x|           : {max_abs_x:.6f} m")
    print(f"mean MPC solve    : {np.mean(solve_times):.3f} ms")
    print(f"max FD eq residual: {max_fd_res:.3e}")
    print(f"CSV               : {out}")

    balanced = (
        abs(state[1]) < 0.01
        and abs(state[3]) < 0.05
        and abs(state[0]) < 0.10
        and abs(state[2]) < 0.10
    )
    if not balanced:
        raise SystemExit("Validation failed: pendulum did not converge sufficiently close to upright.")
    print("PASS: forward-dynamics MPC stabilized the nonlinear plant.\n")


if __name__ == "__main__":
    main()
