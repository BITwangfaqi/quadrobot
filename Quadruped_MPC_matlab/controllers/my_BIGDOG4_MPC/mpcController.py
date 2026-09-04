"""Stateful whole-body MPC controller translated from MATLAB."""

import numpy as np

from JacobianMatrix import JacobianMatrix
from SwingTrajectoryBezier import SwingTrajectoryBezier
from forwardKinematics import forwardKinematics
from getMpcTable import getMpcTable
from getSwingState import getSwingState
from matrixLogRot import matrixLogRot
from qp import qp
from setIterations import setIterations
from solveMpc import solveMpc


class MpcController:
    """Hold the state represented by MATLAB ``persistent`` variables."""

    def __init__(self):
        self.state = 0
        self.timer = 0
        self.iteration_counter = 0
        self.position_reference = np.zeros(3)
        self.position_reference_initialized = False
        self.first_swing = np.zeros(4)
        self.swing_time_remaining = np.zeros(4)
        self.pf_init = np.zeros((3, 4))
        self.pf_final = np.zeros((3, 4))
        self.foothold_velocity_offset = np.zeros(3)
        self.force_solution = None

    def compute(
        self,
        R,
        w,
        x,
        y,
        z,
        v,
        q1,
        q2,
        q3,
        q4,
        w1,
        w2,
        w3,
        w4,
        t,
        foot_sensor,
        vx,
        vy,
        offsets,
        durations,
        body_height,
        desired_roll,
        desired_pitch,
        desired_yaw,
        rbs,
        mass,
        inertia,
        nominal_foot_offsets,
        kp_com,
        kd_com,
        kp_base,
        kd_base,
        dt,
        iterations_between_mpc,
        stance_time,
        swing_time,
        step_height,
        horizon,
        kp_cartesian,
        kd_cartesian,
        vz,
        n_iterations,
    ):
        R = np.asarray(R, dtype=float).reshape(3, 3)
        w = np.asarray(w, dtype=float).reshape(3)
        v = np.asarray(v, dtype=float).reshape(3)
        q_legs = [
            np.asarray(q_leg, dtype=float).reshape(3) for q_leg in (q1, q2, q3, q4)
        ]
        w_legs = [
            np.asarray(w_leg, dtype=float).reshape(3) for w_leg in (w1, w2, w3, w4)
        ]
        foot_sensor = np.asarray(foot_sensor, dtype=float).reshape(4)
        offsets = np.asarray(offsets, dtype=float).reshape(4)
        durations = np.asarray(durations, dtype=float).reshape(4)
        rbs = np.asarray(rbs, dtype=float).reshape(3, 4)
        inertia = np.asarray(inertia, dtype=float).reshape(3, 3)
        nominal_foot_offsets = np.asarray(nominal_foot_offsets, dtype=float).reshape(
            3, 4
        )
        kp_com = np.asarray(kp_com, dtype=float).reshape(3, 3)
        kd_com = np.asarray(kd_com, dtype=float).reshape(3, 3)
        kp_base = np.asarray(kp_base, dtype=float).reshape(3, 3)
        kd_base = np.asarray(kd_base, dtype=float).reshape(3, 3)
        kp_cartesian = np.asarray(kp_cartesian, dtype=float).reshape(3, 3)
        kd_cartesian = np.asarray(kd_cartesian, dtype=float).reshape(3, 3)
        horizon = int(horizon)
        n_iterations = int(n_iterations)

        torques = np.zeros((3, 4))
        position = np.array([x, y, z], dtype=float)
        joint_angles = np.concatenate(q_legs)
        joint_velocities = np.column_stack(w_legs)

        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
        yaw = np.arctan2(R[1, 0], R[0, 0])
        R_yaw = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        inertia_world = R @ inertia @ R.T
        inertia_inverse = np.linalg.inv(inertia_world)

        if 3.0 < t < 5.0:
            velocity_reference = np.array([0.0, 0.0, vz])
            position_reference = np.array([0.0, 0.0, 0.1 + vz * (t - 3.0)])
        else:
            if t >= 5.0 and not self.position_reference_initialized:
                self.position_reference[:] = position
                self.position_reference_initialized = True
            velocity_reference = np.array([vx, vy, 0.0])
            position_reference = np.array(
                [self.position_reference[0], self.position_reference[1], body_height]
            )

        orientation_reference = np.array([desired_roll, desired_pitch, desired_yaw])
        angular_velocity_reference = np.zeros(3)
        linear_acceleration = kp_com @ (position_reference - position) + kd_com @ (
            velocity_reference - v
        )
        orientation_error = matrixLogRot(R.T)
        angular_acceleration = kp_base @ orientation_error + kd_base @ (
            angular_velocity_reference - w
        )
        desired_force = float(mass) * (linear_acceleration + np.array([0.0, 0.0, 9.81]))
        desired_torque = inertia_world @ angular_acceleration
        body_control = np.concatenate((desired_force, desired_torque))

        foot_positions_body, feet_from_body = forwardKinematics(joint_angles)
        feet_from_body_world = R @ feet_from_body
        jacobian = JacobianMatrix(joint_angles)

        if self.state == 0:
            self.timer += 1
            self.force_solution = qp(feet_from_body_world, body_control, np.ones(4))
            for leg in range(4):
                block = jacobian[:, 3 * leg : 3 * leg + 3]
                force = self.force_solution[3 * leg : 3 * leg + 3]
                torques[:, leg] = -block.T @ R.T @ force

            if self.timer > 1000:
                self.state = 1
                self.timer = 0
                self.first_swing[:] = [0.0, 1.0, 1.0, 0.0]

        elif self.state == 1:
            for leg in range(4):
                if self.first_swing[leg] == 1.0:
                    self.swing_time_remaining[leg] = swing_time
                    self.pf_init[:, leg] = position + feet_from_body_world[:, leg]
                    self.pf_init[2, leg] = 0.015
                    self.foothold_velocity_offset = (
                        0.5 * stance_time * np.array([v[0], v[1], 0.0])
                    )
                    self.foothold_velocity_offset[0] = min(
                        self.foothold_velocity_offset[0], 0.35
                    )
                else:
                    self.swing_time_remaining[leg] -= dt

                self.swing_time_remaining[leg] = max(
                    self.swing_time_remaining[leg], 0.0
                )
                self.pf_final[:, leg] = (
                    position
                    + velocity_reference * self.swing_time_remaining[leg]
                    + R @ nominal_foot_offsets[:, leg]
                    + self.foothold_velocity_offset
                )
                self.pf_final[2, leg] = 0.015

            iteration, phase = setIterations(
                n_iterations, self.iteration_counter, iterations_between_mpc
            )
            swing_state = getSwingState(
                phase, offsets / n_iterations, durations / n_iterations
            )
            mpc_table = getMpcTable(iteration, n_iterations, offsets, durations)

            if self.iteration_counter % int(iterations_between_mpc) == 0:
                dt_mpc = dt * iterations_between_mpc
                trajectory = np.zeros(12 * horizon)
                trajectory_initial = np.concatenate(
                    (
                        orientation_reference,
                        [x, y, body_height],
                        angular_velocity_reference,
                        velocity_reference,
                    )
                )
                for step in range(horizon):
                    block = trajectory_initial.copy()
                    block[3:6] += (step + 1) * dt_mpc * velocity_reference
                    trajectory[12 * step : 12 * (step + 1)] = block

                state_vector = np.concatenate(
                    ([roll, pitch, yaw, x, y, z], w, v, [-9.8])
                )
                desired_state = np.empty(13 * horizon)
                for step in range(horizon):
                    desired_state[13 * step : 13 * step + 12] = trajectory[
                        12 * step : 12 * (step + 1)
                    ]
                    desired_state[13 * step + 12] = -9.8

                self.force_solution = solveMpc(
                    R_yaw,
                    inertia_inverse,
                    feet_from_body_world,
                    state_vector,
                    desired_state,
                    dt_mpc,
                    horizon,
                    mpc_table,
                    mass,
                )

            self.iteration_counter += 1
            if self.force_solution is None:
                self.force_solution = np.zeros(12 * horizon)

            for leg in range(4):
                block = jacobian[:, 3 * leg : 3 * leg + 3]
                if swing_state[leg] > 0.0:
                    self.first_swing[leg] = 0.0
                    swing_position, swing_velocity, _ = SwingTrajectoryBezier(
                        self.pf_init[:, leg],
                        self.pf_final[:, leg],
                        swing_state[leg],
                        swing_time,
                        step_height,
                    )
                    desired_leg_position = (
                        R.T @ (swing_position - position) - rbs[:, leg]
                    )
                    desired_leg_velocity = R.T @ (swing_velocity - v)
                    leg_velocity = block @ joint_velocities[:, leg]
                    cartesian_force = kp_cartesian @ (
                        desired_leg_position - foot_positions_body[:, leg]
                    ) + kd_cartesian @ (desired_leg_velocity - leg_velocity)
                    torques[:, leg] = block.T @ cartesian_force
                else:
                    # Both MATLAB contact-sensor branches apply the same stance torque.
                    _ = foot_sensor[leg]
                    self.first_swing[leg] = 1.0
                    force = self.force_solution[3 * leg : 3 * leg + 3]
                    torques[:, leg] = -block.T @ R.T @ force

        return tuple(torques[:, leg].copy() for leg in range(4))


_default_controller = MpcController()


def mpcController(*args, **kwargs):
    """MATLAB-compatible functional wrapper using one persistent controller."""
    return _default_controller.compute(*args, **kwargs)


def reset_controller():
    """Reset the module-level controller state, primarily for tests."""
    global _default_controller
    _default_controller = MpcController()
