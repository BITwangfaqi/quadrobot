#!/usr/bin/env python3
"""Webots Python controller for the quadruped MPC example.

Translated from ``my_BIGDOG4_MPC.m`` by retaining the original leg order:
front-right, front-left, back-right, back-left.
"""


import math
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

from mpcController import MpcController


TIME_STEP = 2
LEG_ORDER = ("FR", "FL", "BR", "BL")
JOINT_ORDER = ("hip", "leg", "foot")


def _key_is(key, letter):
    return key in (ord(letter.lower()), ord(letter.upper()))


def _rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rotation_z @ rotation_y @ rotation_x


def _get_required_device(robot, name):
    device = robot.getDevice(name)
    if device is None:
        raise RuntimeError(f"Required Webots device was not found: {name}")
    return device


def main():
    # Imported here so numerical modules remain testable outside Webots.
    from controller import Robot

    robot = Robot()
    keyboard = robot.getKeyboard()
    keyboard.enable(TIME_STEP)

    gyro = _get_required_device(robot, "gyro")
    imu = _get_required_device(robot, "imu")
    gps = _get_required_device(robot, "gps")
    accelerometer = _get_required_device(robot, "accelerometer")
    for sensor in (gyro, imu, gps, accelerometer):
        sensor.enable(TIME_STEP)

    touch_sensors = {
        leg: _get_required_device(robot, f"{leg}_TOUCH")
        for leg in ("FL", "FR", "BL", "BR")
    }
    for sensor in touch_sensors.values():
        sensor.enable(TIME_STEP)

    motors = {
        (leg, joint): _get_required_device(robot, f"{leg}_{joint}_motor")
        for leg in LEG_ORDER
        for joint in JOINT_ORDER
    }
    position_sensors = {
        (leg, joint): _get_required_device(robot, f"{leg}_{joint}_position_sensor")
        for leg in LEG_ORDER
        for joint in JOINT_ORDER
    }

    for motor in motors.values():
        motor.setPosition(float("inf"))
        motor.setVelocity(1.0)
        motor.enableTorqueFeedback(TIME_STEP)
    for sensor in position_sensors.values():
        sensor.enable(TIME_STEP)

    mass = 13.5 + 4.67
    inertia = np.diag([0.06150, 0.1313, 0.1646])
    rbs = np.array(
        [
            [0.139, 0.139, -0.139, -0.139],
            [-0.061, 0.061, -0.061, 0.061],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    nominal_foot_offsets = np.array(
        [
            [0.204, 0.204, -0.204, -0.204],
            [-0.146, 0.146, -0.146, 0.146],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    kp_com = np.diag([400.0, 400.0, 400.0])
    kd_com = np.diag([160.0, 160.0, 160.0])
    kp_base = np.diag([1000.0, 1000.0, 1000.0])
    kd_base = np.diag([40.0, 40.0, 40.0])
    kp_cartesian = np.diag([100.0, 100.0, 100.0])
    kd_cartesian = np.diag([10.0, 20.0, 10.0])

    dt = 0.002
    iterations_between_mpc = 15
    stance_time = 0.15
    swing_time = 0.15
    step_height = 0.07
    horizon = 10
    n_iterations = 10
    offsets = np.array([0, 5, 5, 0], dtype=int)
    durations = np.array([5, 5, 5, 5], dtype=int)

    body_height = 0.3
    vx = 0.0
    vy = 0.0
    vz = 0.1
    time = 0.0
    previous_position = np.zeros(3)
    previous_joint_angles = [np.zeros(3) for _ in LEG_ORDER]
    controller = MpcController()
    debug_enabled = os.environ.get("BIGDOG_DEBUG", "0") == "1"
    next_debug_time = 0.0
    applied_torques = np.zeros((4, 3))

    crouch_targets = {"hip": 0.0, "leg": -0.45, "foot": 1.4}

    while robot.step(TIME_STEP) != -1:
        key = keyboard.getKey()
        desired_roll = 0.0
        desired_pitch = 0.0
        desired_yaw = 0.0

        if _key_is(key, "Z"):
            desired_roll = 0.25
        elif _key_is(key, "X"):
            desired_roll = -0.25
        elif _key_is(key, "C"):
            desired_pitch = 0.25
        elif _key_is(key, "V"):
            desired_pitch = -0.25
        elif _key_is(key, "B"):
            desired_yaw = 0.25
        elif _key_is(key, "N"):
            desired_yaw = -0.25

        if _key_is(key, "T"):
            if time > 5.0:
                vx = min(vx + TIME_STEP / 2000.0, 2.0)
            body_height = 0.25
        else:
            vx = 0.0
            vy = 0.0
            body_height = 0.3

        if _key_is(key, "W"):
            vx = 0.5
        elif _key_is(key, "S"):
            vx = -0.3
        elif _key_is(key, "A"):
            vy = 0.3
        elif _key_is(key, "D"):
            vy = -0.3

        if _key_is(key, "U"):  # Trotting
            offsets, durations = np.array([0, 5, 5, 0]), np.array([5, 5, 5, 5])
        elif _key_is(key, "I"):  # Bounding
            offsets, durations = np.array([5, 5, 0, 0]), np.array([4, 4, 4, 4])
        elif _key_is(key, "O"):  # Pronking
            offsets, durations = np.array([0, 0, 0, 0]), np.array([4, 4, 4, 4])
        elif _key_is(key, "P"):  # Galloping
            offsets, durations = np.array([0, 2, 7, 9]), np.array([4, 4, 4, 4])
        elif _key_is(key, "J"):  # Standing
            offsets, durations = np.zeros(4, dtype=int), np.full(4, 10, dtype=int)
        elif _key_is(key, "K"):  # Trot running
            offsets, durations = np.array([0, 5, 5, 0]), np.array([4, 4, 4, 4])
        elif _key_is(key, "L"):  # Walking
            offsets, durations = np.array([5, 0, 5, 0]), np.array([5, 5, 5, 5])

        angular_velocity = np.asarray(gyro.getValues(), dtype=float)
        roll_pitch_yaw = np.asarray(imu.getRollPitchYaw(), dtype=float)
        position = np.asarray(gps.getValues(), dtype=float)
        rotation = _rotation_matrix(*roll_pitch_yaw)

        joint_angles = []
        joint_velocities = []
        for leg_index, leg in enumerate(LEG_ORDER):
            angles = np.array(
                [
                    position_sensors[(leg, "hip")].getValue(),
                    position_sensors[(leg, "leg")].getValue() + 0.45,
                    position_sensors[(leg, "foot")].getValue() - 1.40,
                ]
            )
            joint_angles.append(angles)
            joint_velocities.append(
                (angles - previous_joint_angles[leg_index]) / (TIME_STEP / 1000.0)
            )

        linear_velocity = (position - previous_position) / (TIME_STEP / 1000.0)
        previous_position = position.copy()
        previous_joint_angles = [angles.copy() for angles in joint_angles]
        foot_sensor = np.array(
            [
                touch_sensors["FR"].getValue(),
                touch_sensors["FL"].getValue(),
                touch_sensors["BR"].getValue(),
                touch_sensors["BL"].getValue(),
            ]
        )

        if time < 3.0:
            applied_torques.fill(0.0)
            for leg in LEG_ORDER:
                for joint in JOINT_ORDER:
                    motors[(leg, joint)].setPosition(crouch_targets[joint])
                    motors[(leg, joint)].setVelocity(1.0)
        else:
            torques = controller.compute(
                rotation,
                angular_velocity,
                position[0],
                position[1],
                position[2],
                linear_velocity,
                *joint_angles,
                *joint_velocities,
                time,
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
            )
            for leg_index, leg in enumerate(LEG_ORDER):
                clipped = np.clip(torques[leg_index], -30.0, 30.0)
                applied_torques[leg_index, :] = clipped
                for joint_index, joint in enumerate(JOINT_ORDER):
                    motors[(leg, joint)].setTorque(float(clipped[joint_index]))

        if debug_enabled and time >= next_debug_time:
            print(
                "BIGDOG_DEBUG"
                f" t={time:.3f} state={controller.state}"
                f" xyz={np.round(position, 4).tolist()}"
                f" rpy={np.round(roll_pitch_yaw, 4).tolist()}"
                f" vel={np.round(linear_velocity, 4).tolist()}"
                f" q={np.round(np.concatenate(joint_angles), 3).tolist()}"
                f" tau_max={np.max(np.abs(applied_torques)):.3f}"
                f" contact={np.round(foot_sensor, 3).tolist()}",
                flush=True,
            )
            next_debug_time += 0.25

        time += TIME_STEP / 1000.0


if __name__ == "__main__":
    main()
