#!/usr/bin/env python3
"""External Webots controller; run this file from VSCode with <extern> selected.

Webots is loaded lazily at startup. The model, gait, configuration and
optimization modules can therefore be used in offline tests.
"""

import argparse
import importlib
import os
from pathlib import Path
import sys
import time

# Small MPC matrices do not benefit from BLAS thread startup on every call.
# Apply before importing NumPy; explicit user environment settings take priority.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

if __package__:
    from .config import MPCConfig, MPCReference
    from .gait import ContactSchedule
else:
    from config import MPCConfig, MPCReference
    from gait import ContactSchedule


KEYBOARD_HELP = (
    "U: trot; J: stand/resume; hold W/S: forward/back, A/D: left/right, "
    "Q/E: yaw (walking commands require U); I/K: tool up/down; "
    "arrow keys: tool fore/aft/left/right; SPACE: position hold. "
    "Tool forces are disabled unless explicitly configured."
)


def _load_webots_robot():
    """Use the same SDK search path as my_BIGDOG4_MPC.py for external launch."""
    # The SDK's wb.py also needs WEBOTS_HOME to locate libController.so.
    webots_home = Path(os.environ.setdefault("WEBOTS_HOME", "/usr/local/webots"))
    try:
        from controller import Robot
    except ImportError:
        sdk_path = str(webots_home / "lib" / "controller" / "python")
        if sdk_path not in sys.path:
            sys.path.append(sdk_path)
        try:
            from controller import Robot
        except (ImportError, OSError) as error:
            raise RuntimeError(
                f"Cannot load the Webots Python SDK from {sdk_path}. "
                "Set WEBOTS_HOME to your Webots installation, keep the robot "
                "controller as <extern>, and run this script from VSCode."
            ) from error
    return Robot


def _check_dependencies():
    """Report a wrong VSCode interpreter before connecting to the robot."""
    try:
        casadi = importlib.import_module("casadi")
    except ImportError as error:
        raise RuntimeError(
            f"CasADi cannot be imported by {sys.executable}. "
            "In VSCode select the 'inv_dyn_mpc (Webots extern, wewebot)' launch "
            "configuration, or select /home/tdt/anaconda3/envs/wewebot/bin/python "
            "as the Python interpreter."
        ) from error
    return casadi.__version__


def rotation_from_rpy(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def quaternion_from_rpy(rpy):
    """Webots roll/pitch/yaw to the model's xyzw quaternion convention."""
    roll, pitch, yaw = 0.5 * np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    quaternion = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])
    return quaternion / np.linalg.norm(quaternion)


class StateEstimator:
    """Sensor conversion; free-flyer linear/angular velocity is body-local."""

    def __init__(self, dt, filter_hz=50.0):
        self.dt = float(dt)
        self.alpha = 1.0 - np.exp(-2.0 * np.pi * filter_hz * self.dt)
        self.previous_joints = None
        self.joint_velocity = np.zeros(18)

    def update(self, position, rpy, world_velocity, body_gyro, joints):
        values = [np.asarray(value, dtype=float) for value in
                  (position, rpy, world_velocity, body_gyro, joints)]
        if [value.shape for value in values] != [(3,), (3,), (3,), (3,), (18,)]:
            raise ValueError("Sensor dimensions do not match the 18-joint robot")
        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("Non-finite GPS, IMU, gyro or joint sensor reading")
        position, rpy, world_velocity, body_gyro, joints = values
        if self.previous_joints is not None:
            derivative = (joints - self.previous_joints) / self.dt
            self.joint_velocity += self.alpha * (derivative - self.joint_velocity)
        self.previous_joints = joints.copy()
        rotation = rotation_from_rpy(rpy)
        q = np.r_[position, quaternion_from_rpy(rpy), joints]
        v = np.r_[rotation.T @ world_velocity, body_gyro, self.joint_velocity]
        return q, v


def _required_device(robot, name):
    device = robot.getDevice(name)
    if device is None:
        raise RuntimeError(f"Required Webots device is missing: {name}")
    return device


class Actuators:
    """Position startup/fault hold and bounded 18-joint torque commands."""

    def __init__(self, robot, description, locked_wrist=None):
        self.motors = [_required_device(robot, name) for name in description.joint_names]
        self.lower = np.asarray(description.lower_limits, dtype=float)
        self.upper = np.asarray(description.upper_limits, dtype=float)
        self.limits = np.minimum(
            np.asarray(description.effort_limits, dtype=float),
            [min(motor.getMaxTorque(), motor.getAvailableTorque()) for motor in self.motors],
        )
        self.velocity_limits = np.minimum(
            np.asarray(description.velocity_limits, dtype=float),
            [motor.getMaxVelocity() for motor in self.motors],
        )
        if not np.all(np.isfinite(self.limits)) or np.any(self.limits <= 0):
            raise RuntimeError("Motor torque limits must be finite and positive")
        if not np.all(np.isfinite(self.velocity_limits)) or np.any(self.velocity_limits <= 0):
            raise RuntimeError("Motor velocity limits must be finite and positive")
        self.mode = None
        self.locked_wrist = None if locked_wrist is None else np.asarray(locked_wrist, dtype=float)
        self.hold_target = np.clip(np.zeros(18), self.lower, self.upper)
        self.position(self.hold_target)
        # The exact model locks these slider joints at zero and includes their
        # mass/inertia in the gripper. They must not become torque DOFs.
        self.grippers = [_required_device(robot, name) for name in ("joint7", "joint8")]
        for gripper in self.grippers:
            gripper.setPosition(0.0)
            gripper.setVelocity(min(0.02, gripper.getMaxVelocity()))

    def position(self, target):
        target = np.asarray(target, dtype=float).reshape(18)
        if not np.all(np.isfinite(target)):
            target = self.hold_target
        target = np.clip(target, self.lower, self.upper)
        if self.locked_wrist is not None:
            target[-2:] = self.locked_wrist
        for index, motor in enumerate(self.motors):
            motor.setPosition(float(target[index]))
            motor.setVelocity(float(min(0.8, self.velocity_limits[index])))
        self.hold_target = target.copy()
        self.mode = "position"

    def torque(self, torque):
        torque = np.asarray(torque, dtype=float).reshape(18)
        if not np.all(np.isfinite(torque)):
            raise ValueError("Refusing a non-finite motor torque")
        if self.mode != "torque":
            for motor in self.motors[:16] if self.locked_wrist is not None else self.motors:
                motor.setPosition(float("inf"))
                motor.setVelocity(0.0)
        bounded = np.clip(torque, -self.limits, self.limits)
        for index, (motor, value) in enumerate(zip(self.motors, bounded)):
            if self.locked_wrist is not None and index >= 16:
                motor.setPosition(float(self.locked_wrist[index - 16]))
                bounded[index] = 0.0
            else:
                motor.setTorque(float(value))
        self.mode = "torque"
        return bounded


class UserCommand:
    def __init__(self, config):
        self.config = config
        self.mode = "stand"
        self.gait_start = 0.0
        self.stopped = False
        self.base_position = None
        self.yaw = 0.0
        self.nominal_joints = None
        self.tool_origin = None
        self.tool_offset = np.zeros(3)
        self.keys_until = {}

    def reset_reference(self, q, rpy, model, nominal_joints):
        self.base_position = q[:3].copy()
        self.yaw = float(rpy[2])
        self.nominal_joints = np.asarray(nominal_joints, dtype=float).copy()
        rotation = rotation_from_rpy([0, 0, self.yaw])
        self.tool_origin = rotation.T @ (np.asarray(model.tool_position(q)).reshape(3) - q[:3])
        self.tool_offset[:] = 0

    def update(self, keyboard, now, dt, q):
        previous_mode = self.mode
        was_stopped = self.stopped
        key = keyboard.getKey()
        while key != -1:
            key = key & 0xFFFF
            if ord("a") <= key <= ord("z"):
                key -= 32
            self.keys_until[key] = now + 0.15
            if key == ord("U"):
                self.mode = "trot"
                self.stopped = False
            elif key == ord("J"):
                self.mode = "stand"
                self.stopped = False
            elif key == ord(" "):
                self.stopped = True
            key = keyboard.getKey()
        self.keys_until = {key: end for key, end in self.keys_until.items() if end >= now}
        if previous_mode != self.mode:
            self.gait_start = now
        changed = previous_mode != self.mode or was_stopped != self.stopped
        if self.base_position is None:
            return None, changed
        active = lambda key: float(self.keys_until.get(key, -1) >= now)
        pair = lambda positive, negative: active(ord(positive)) - active(ord(negative))
        walking = self.mode == "trot" and not self.stopped
        local_velocity = np.array([
            pair("W", "S") * self.config.maximum_forward_speed,
            pair("A", "D") * self.config.maximum_lateral_speed,
            0.0,
        ]) if walking else np.zeros(3)
        yaw_rate = pair("Q", "E") * self.config.maximum_yaw_rate if walking else 0.0
        self.yaw += yaw_rate * dt
        rotation = rotation_from_rpy([0, 0, self.yaw])
        world_velocity = rotation @ local_velocity
        self.base_position += world_velocity * dt
        error = self.base_position[:2] - q[:2]
        norm = np.linalg.norm(error)
        if norm > self.config.maximum_base_reference_error:
            self.base_position[:2] = q[:2] + error * self.config.maximum_base_reference_error / norm
        tool_command = np.array([
            active(keyboard.UP) - active(keyboard.DOWN),
            active(keyboard.LEFT) - active(keyboard.RIGHT),
            pair("I", "K"),
        ]) * self.config.maximum_tool_speed
        if self.stopped or self.config.tool_force_enabled:
            tool_command[:] = 0.0
        self.tool_offset += tool_command * dt
        self.tool_offset = np.clip(self.tool_offset, -self.config.maximum_tool_displacement,
                                   self.config.maximum_tool_displacement)
        tool_relative = rotation @ (self.tool_origin + self.tool_offset)
        tool_velocity = world_velocity + np.cross([0, 0, yaw_rate], tool_relative) + rotation @ tool_command
        if self.config.tool_force_enabled:
            tool_velocity = np.asarray(self.config.tool_contact_velocity, dtype=float)
        reference = MPCReference(
            base_position=self.base_position.copy(),
            base_quaternion_xyzw=quaternion_from_rpy([0, 0, self.yaw]),
            base_velocity_world=world_velocity,
            yaw_rate=yaw_rate,
            joint_positions=self.nominal_joints.copy(),
            tool_position=self.base_position + tool_relative,
            tool_velocity=tool_velocity,
            tool_force=np.asarray(self.config.tool_force, dtype=float),
        )
        return reference, changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides for MPCConfig")
    parser.add_argument("--world", type=Path, help="Source quadruped_arm.wbt used to build the exact model")
    parser.add_argument("--duration", type=float,
                        help="Stop in position hold after this many simulation seconds (including startup)")
    arguments = parser.parse_args(argv)
    if arguments.duration is not None and (not np.isfinite(arguments.duration) or arguments.duration <= 0):
        parser.error("--duration must be finite and positive")
    config = MPCConfig.from_json(arguments.config)
    casadi_version = _check_dependencies()
    print(f"[inv_dyn_mpc] Python: {sys.executable}; CasADi: {casadi_version}", flush=True)
    Robot = _load_webots_robot()
    if __package__:
        from .robot_description import RobotDescription
        from .rigid_body_model import WholeBodyModel
        from .mpc import InverseDynamicsMPC
    else:
        from robot_description import RobotDescription
        from rigid_body_model import WholeBodyModel
        from mpc import InverseDynamicsMPC

    description = RobotDescription.from_world(arguments.world)
    print("[inv_dyn_mpc] Connecting to Webots; robot controller should be <extern>.", flush=True)
    robot = Robot()
    controller_start_time = float(robot.getTime())
    timestep = int(round(robot.getBasicTimeStep()))
    if timestep <= 0 or abs(timestep - robot.getBasicTimeStep()) > 1e-6:
        raise RuntimeError("This controller requires a positive integer basicTimeStep in milliseconds")
    if timestep not in (1, 2):
        raise RuntimeError("The paper uses a 500 Hz torque/PD loop. Set WorldInfo.basicTimeStep=2 ms "
                           "and reload the world; the 80 Hz MPC period is separate from physics dt.")
    timestep = 2
    dt = timestep / 1000.0
    if config.solve_period < dt:
        raise ValueError("solve_period must be at least the Webots basicTimeStep")
    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)
    gps, imu, gyro = [_required_device(robot, name) for name in ("gps", "imu", "gyro")]
    sensors = [_required_device(robot, name) for name in description.position_sensor_names]
    touches = [_required_device(robot, f"{leg}_TOUCH") for leg in ("FR", "FL", "BR", "BL")]
    for sensor in [gps, imu, gyro, *sensors, *touches]:
        sensor.enable(timestep)
    actuators = Actuators(robot, description, config.nominal_arm[-2:] if config.lock_wrist else None)
    # Build constraints from the stricter of the parsed WBT and live motor
    # limits. Output saturation alone must not hide an infeasible plan.
    description.effort_limits[:] = actuators.limits
    description.velocity_limits[:] = actuators.velocity_limits
    model = WholeBodyModel(description)
    solver = InverseDynamicsMPC(model, config)
    print(f"[inv_dyn_mpc] NLP ready: build={solver.build_time:.2f}s, "
          f"compiled={config.jit}, cache_hit={solver.cache_hit}, "
          f"variables={solver._decision_size}, constraints={len(solver.lbg)}, "
          f"bounded torque nodes={solver.torque_nodes}", flush=True)
    estimator = StateEstimator(dt, config.joint_velocity_filter_hz)
    command = UserCommand(config)
    kp = np.r_[np.full(12, config.leg_kp), np.full(6, config.arm_kp)]
    kd = np.r_[np.full(12, config.leg_kd), np.full(6, config.arm_kd)]
    initial_joints = None
    nominal_joints = None
    startup_time = None
    initialized = False
    solution = None
    solution_time = -np.inf
    next_solve = 0.0
    next_diagnostic = 0.0
    last_finite_joints = np.zeros(18)
    failures = 0
    last_solve_wall = 0.0
    solve_calls, deadline_misses = 0, 0
    # Equation (9) reconstructs commands only from the first two intervals.
    # A configurable timeout cannot authorize extrapolating that polynomial.
    maximum_solution_age = min(config.solution_timeout, float(np.sum(config.time_steps()[:2])))
    print(f"[inv_dyn_mpc] {KEYBOARD_HELP}", flush=True)
    print(f"[inv_dyn_mpc] Model: {solver.model.nv} optimized velocities, {model.total_mass:.3f} kg; "
          f"{model.casadi_backend}, solver={config.solver}; "
          f"simulation dt={dt:.3f}s, solve period={config.solve_period:.4f}s. "
          "Synchronous Python optimization may run slower than real time.", flush=True)

    while robot.step(timestep) != -1:
        now = float(robot.getTime())
        duration_reached = (arguments.duration is not None
                            and now - controller_start_time + 1e-9 >= arguments.duration)
        try:
            rpy = np.asarray(imu.getRollPitchYaw(), dtype=float)
            q, v = estimator.update(gps.getValues(), rpy, gps.getSpeedVector(),
                                    gyro.getValues(), [sensor.getValue() for sensor in sensors])
            last_finite_joints = q[7:].copy()
        except (ValueError, FloatingPointError) as error:
            actuators.position(last_finite_joints)
            estimator = StateEstimator(dt, config.joint_velocity_filter_hz)
            solution = None
            if now >= next_diagnostic:
                print(f"[inv_dyn_mpc] Position hold: invalid sensor data: {error}", flush=True)
                next_diagnostic = now + config.diagnostic_period
            if duration_reached:
                robot.step(timestep)
                print("[inv_dyn_mpc] Requested duration reached; exiting in position hold.", flush=True)
                break
            continue

        if duration_reached:
            actuators.position(last_finite_joints)
            robot.step(timestep)  # Flush hold commands before controller exits.
            print("[inv_dyn_mpc] Requested duration reached; exiting in position hold.", flush=True)
            break

        if initial_joints is None:
            initial_joints = q[7:].copy()
            nominal_joints = np.r_[np.tile(config.nominal_leg, 4), config.nominal_arm]
            if (np.any(nominal_joints < description.lower_limits)
                    or np.any(nominal_joints > description.upper_limits)):
                raise ValueError("Nominal startup posture violates world joint limits")
            startup_time = now
            print("[inv_dyn_mpc] Smooth position startup; stand mode is the default.", flush=True)

        elapsed = now - startup_time
        if elapsed < config.startup_duration + config.settle_duration:
            phase = min(1.0, elapsed / config.startup_duration)
            blend = phase * phase * (3.0 - 2.0 * phase)
            actuators.position(initial_joints + blend * (nominal_joints - initial_joints))
            continue

        if not initialized:
            command.reset_reference(q, rpy, model, nominal_joints)
            command.gait_start = now
            initialized = True
            next_solve = now
        reference, changed = command.update(keyboard, now, dt, q)
        if changed:
            solution = None
            actuators.position(q[7:])
            command.reset_reference(q, rpy, model, nominal_joints)
            reference, _ = command.update(keyboard, now, 0.0, q)
            next_solve = now
            print(f"[inv_dyn_mpc] {'Position hold' if command.stopped else command.mode}", flush=True)
        if command.stopped:
            if actuators.mode != "position":
                actuators.position(q[7:])
            continue

        if now + 1e-9 >= next_solve:
            # Keep the fractional deadline: 12.5 ms over a 2 ms physics grid
            # alternates 12/14 ms intervals instead of drifting to 14 ms always.
            next_solve += config.solve_period
            while next_solve <= now + 1e-9:
                next_solve += config.solve_period
            schedule = ContactSchedule(command.mode, now - command.gait_start, config.gait_period)
            wall_start = time.perf_counter()
            try:
                candidate = solver.solve(q, v, reference, schedule)
                # Validate output before changing motor control mode.
                sample = candidate.sample(0.0)
                if len(sample) != 3 or any(np.asarray(value).shape != (18,) for value in sample):
                    raise ValueError("MPCSolution.sample must return three 18-vectors: qj, vj, tau")
                if not all(np.all(np.isfinite(value)) for value in sample):
                    raise ValueError("MPC returned a non-finite trajectory")
                solution = candidate
                solution_time = now
                failures = 0
            except Exception as error:
                failures += 1
                if failures == 1 or now >= next_diagnostic:
                    print(f"[inv_dyn_mpc] Solve failed ({failures}): {type(error).__name__}: {error}", flush=True)
                    next_diagnostic = now + config.diagnostic_period
            last_solve_wall = time.perf_counter() - wall_start
            solve_calls += 1
            deadline_misses += int(last_solve_wall > config.solve_period)

        age = now - solution_time
        if solution is not None and age <= maximum_solution_age + 1e-12:
            try:
                desired_q, desired_v, feedforward = solution.sample(age)
                torque = feedforward + kp * (desired_q - q[7:]) + kd * (desired_v - v[6:])
                applied = actuators.torque(torque)
            except (ValueError, FloatingPointError, IndexError) as error:
                solution = None
                actuators.position(q[7:])
                print(f"[inv_dyn_mpc] Trajectory unavailable; position hold: {error}. "
                      "Solver retries continue.", flush=True)
                continue
        else:
            if actuators.mode != "position":
                actuators.position(q[7:])
                print(f"[inv_dyn_mpc] Position hold: trajectory age {age:.3f}s exceeds "
                      f"{maximum_solution_age:.3f}s. Solver retries continue.", flush=True)
            applied = np.zeros(18)

        if now >= next_diagnostic:
            contacts = "".join(str(int(sensor.getValue() > 0)) for sensor in touches)
            print(f"[inv_dyn_mpc] t={now:.2f}s {command.mode}/{actuators.mode} "
                  f"z={q[2]:.3f} contacts(FR FL BR BL)={contacts} "
                  f"max|tau|={np.max(np.abs(applied)):.2f} Nm "
                  f"solve={last_solve_wall:.3f}s wall age={age:.3f}s sim "
                  f"iterations={solution.iterations if solution is not None else 0} "
                  f"deadline_misses={deadline_misses}/{solve_calls}", flush=True)
            next_diagnostic = now + config.diagnostic_period


if __name__ == "__main__":
    try:
        main()
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"[inv_dyn_mpc] Cannot start: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
