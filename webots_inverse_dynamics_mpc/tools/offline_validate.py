"""Offline validation of the exact same MPC without Webots.

The MPC still uses inverse dynamics as a path constraint. Only the simulated
plant uses forward dynamics, replacing Webots/ODE for this test.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CTRL_DIR = ROOT / "controllers" / "inverse_dynamics_mpc"
sys.path.insert(0, str(CTRL_DIR))

from config import (  # noqa: E402
    INITIAL_POLE_ANGLE,
    MPC_PERIOD_S,
    CART_REFERENCE,
    POLE_REFERENCE,
    FORCE_MAX,
)
from dynamics import state_derivative_np  # noqa: E402
from mpc import InverseDynamicsMPC  # noqa: E402


def rk4_step(state: np.ndarray, force: float, dt: float) -> np.ndarray:
    k1 = state_derivative_np(state, force)
    k2 = state_derivative_np(state + 0.5 * dt * k1, force)
    k3 = state_derivative_np(state + 0.5 * dt * k2, force)
    k4 = state_derivative_np(state + dt * k3, force)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def main():
    mpc = InverseDynamicsMPC()

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
    max_id_res = 0.0

    print("\n=== Offline inverse-dynamics MPC validation ===")
    print(f"initial theta: {state[1]:+.4f} rad")

    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "theta", "x_dot", "theta_dot", "force", "solve_ms", "id_residual"])

        steps = int(total_time / sim_dt)
        for i in range(steps + 1):
            t = i * sim_dt

            if t + 1e-12 >= next_mpc_t:
                result = mpc.solve(state, CART_REFERENCE, POLE_REFERENCE)
                force = float(np.clip(result.force, -FORCE_MAX, FORCE_MAX))
                solve_times.append(result.solve_time_ms)
                if np.isfinite(result.unactuated_residual):
                    max_id_res = max(max_id_res, abs(result.unactuated_residual))
                next_mpc_t += MPC_PERIOD_S

            writer.writerow([t, *state, force, solve_times[-1] if solve_times else np.nan, max_id_res])

            max_abs_theta = max(max_abs_theta, abs(state[1]))
            max_abs_x = max(max_abs_x, abs(state[0]))

            state = rk4_step(state, force, sim_dt)

    print(f"final state       : {state}")
    print(f"max |theta|       : {max_abs_theta:.6f} rad")
    print(f"max |x|           : {max_abs_x:.6f} m")
    print(f"mean MPC solve    : {np.mean(solve_times):.3f} ms")
    print(f"max ID residual   : {max_id_res:.3e}")
    print(f"CSV               : {out}")

    balanced = (
        abs(state[1]) < 0.01
        and abs(state[3]) < 0.05
        and abs(state[0]) < 0.10
        and abs(state[2]) < 0.10
    )
    if not balanced:
        raise SystemExit("Validation failed: pendulum did not converge sufficiently close to upright.")
    print("PASS: inverse-dynamics MPC stabilized the nonlinear plant.\n")


if __name__ == "__main__":
    main()
