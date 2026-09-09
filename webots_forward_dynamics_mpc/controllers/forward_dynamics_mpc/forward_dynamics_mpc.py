"""Webots controller: nonlinear forward-dynamics MPC for a single inverted pendulum."""

from __future__ import annotations

import csv
from pathlib import Path

import sys
import os

# 将工作空间根目录加入到sys.path中以便能够找到mypackage和acados_test
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

# Webots控制器导入
try:
    import controller
except ImportError:
    sys.path.append(os.path.join(os.environ.get("WEBOTS_HOME", "/usr/local/webots"), "lib", "controller", "python"))
    import controller


import numpy as np
from controller import Robot

from config import (
    FORCE_MAX,
    MPC_PERIOD_S,
    CART_REFERENCE,
    POLE_REFERENCE,
    VELOCITY_FILTER_ALPHA,
    PRINT_PERIOD_S,
    LOG_FILENAME,
)
from mpc import ForwardDynamicsMPC


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def main():
    robot = Robot()
    time_step_ms = int(robot.getBasicTimeStep())
    dt = time_step_ms / 1000.0

    cart_motor = robot.getDevice("cart_motor")
    cart_sensor = robot.getDevice("cart_position")
    pole_sensor = robot.getDevice("pole_angle")

    cart_sensor.enable(time_step_ms)
    pole_sensor.enable(time_step_ms)

    # Direct force control: Webots is the external nonlinear plant.
    cart_motor.setAvailableForce(FORCE_MAX)
    cart_motor.setForce(0.0)

    mpc = ForwardDynamicsMPC()

    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILENAME

    prev_x = None
    prev_theta = None
    x_dot_f = 0.0
    theta_dot_f = 0.0
    current_force = 0.0
    last_solve_time = -1e9
    last_print_time = -1e9

    print("\n=== Webots forward-dynamics MPC: single inverted pendulum ===")
    print(f"basic step : {time_step_ms} ms")
    print(f"MPC period : {MPC_PERIOD_S * 1000:.1f} ms")
    print(f"log file   : {log_path}\n")

    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "cart_x_m",
                "theta_rad",
                "cart_v_mps",
                "theta_dot_radps",
                "force_N",
                "solve_time_ms",
                "solver_success",
                "solver_status",
                "model_x_ddot_mps2",
                "model_theta_ddot_radps2",
                "forward_dynamics_equation_residual_inf",
            ]
        )

        solve_time_ms = np.nan
        solver_success = False
        solver_status = "not_solved_yet"
        first_a = np.zeros(2)
        dynamics_residual = np.nan

        while robot.step(time_step_ms) != -1:
            t = robot.getTime()
            x = float(cart_sensor.getValue())
            theta = wrap_to_pi(float(pole_sensor.getValue()))

            if prev_x is None:
                x_dot = 0.0
                theta_dot = 0.0
            else:
                x_dot_raw = (x - prev_x) / dt
                dtheta = wrap_to_pi(theta - prev_theta)
                theta_dot_raw = dtheta / dt
                alpha = VELOCITY_FILTER_ALPHA
                x_dot_f = alpha * x_dot_raw + (1.0 - alpha) * x_dot_f
                theta_dot_f = alpha * theta_dot_raw + (1.0 - alpha) * theta_dot_f
                x_dot = x_dot_f
                theta_dot = theta_dot_f

            prev_x = x
            prev_theta = theta

            # Solve at 50 Hz; hold the first optimized force between updates.
            if t - last_solve_time >= MPC_PERIOD_S - 0.5 * dt:
                state = np.array([x, theta, x_dot, theta_dot], dtype=float)
                result = mpc.solve(
                    state,
                    cart_reference=CART_REFERENCE,
                    pole_reference=POLE_REFERENCE,
                )
                current_force = float(np.clip(result.force, -FORCE_MAX, FORCE_MAX))
                cart_motor.setForce(current_force)

                solve_time_ms = result.solve_time_ms
                solver_success = result.success
                solver_status = result.status
                first_a = result.first_acceleration
                dynamics_residual = result.dynamics_residual_norm
                last_solve_time = t

            writer.writerow(
                [
                    f"{t:.6f}",
                    f"{x:.9f}",
                    f"{theta:.9f}",
                    f"{x_dot:.9f}",
                    f"{theta_dot:.9f}",
                    f"{current_force:.9f}",
                    f"{solve_time_ms:.6f}" if np.isfinite(solve_time_ms) else "",
                    int(bool(solver_success)),
                    solver_status,
                    f"{first_a[0]:.9f}",
                    f"{first_a[1]:.9f}",
                    f"{dynamics_residual:.12e}" if np.isfinite(dynamics_residual) else "",
                ]
            )
            f.flush()

            if t - last_print_time >= PRINT_PERIOD_S:
                print(
                    f"t={t:6.2f}s | x={x:+.3f} m | theta={theta:+.4f} rad "
                    f"| F={current_force:+6.2f} N | solve={solve_time_ms:6.2f} ms "
                    f"| fd_eq_res={dynamics_residual:+.2e}"
                )
                last_print_time = t


if __name__ == "__main__":
    main()
