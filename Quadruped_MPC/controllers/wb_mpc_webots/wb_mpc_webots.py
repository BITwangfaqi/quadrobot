#!/usr/bin/env python3
"""Webots feedback controller using the directly ported wb_mpc_locoman OCP."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import pinocchio as pin

from world_model import DEFAULT_WORLD, WebotsModel
from mpc_bridge import MPC, Settings


def load_robot_class():
    home = Path(os.environ.setdefault('WEBOTS_HOME', '/usr/local/webots'))
    sys.path.append(str(home / 'lib/controller/python'))
    from controller import Robot
    return Robot


def require(robot, name):
    device = robot.getDevice(name)
    if device is None:
        raise RuntimeError(f'World is missing device {name}')
    return device


def startup_posture(model, settings):
    """Position-control target before constructing the feedback MPC."""
    # WBT zero puts shoulder/elbow on their hard stops. The first Euler
    # position prediction cannot correct outward measured motion with torque.
    joints = model.q0[7:].copy()
    for name, angle in [('joint2', 0.15), ('joint3', -0.30)]:
        if name in model.names:
            joints[model.names.index(name)] = angle
    if settings.compiled_solver:
        build = json.loads(Path(settings.compiled_solver).with_suffix('.json').read_text())
        nominal = np.asarray(build['nominal'], float)
        if nominal.shape != (model.nq,):
            raise ValueError('Compiled startup reference dimensions do not match the robot')
        joints = nominal[7:].copy()
    if (not np.isfinite(joints).all() or np.any(joints < model.joint_pos_min)
            or np.any(joints > model.joint_pos_max)):
        raise ValueError('Startup posture must be finite and within joint limits')
    return joints


class Interface:
    def __init__(self, robot, model, step, startup_joints=None):
        self.robot, self.model = robot, model
        self.dt = step / 1000.
        self.motors = [require(robot, name) for name in model.names]
        self.encoders = [require(robot, model.devices[name]['sensor']) for name in model.names]
        self.gps, self.imu, self.gyro = [require(robot, name) for name in ('gps', 'imu', 'gyro')]
        self.touches = [require(robot, name+'_TOUCH') for name in ('FR', 'FL', 'BR', 'BL')]
        for device in self.encoders + [self.gps, self.imu, self.gyro] + self.touches:
            device.enable(step)
        self.locked_motors = {name: require(robot, name) for name in model.locked}
        for name, motor in self.locked_motors.items():
            motor.setPosition(model.locked[name])
            motor.setVelocity(min(0.2, motor.getMaxVelocity()))
        self.previous_joints = None
        self.velocity = np.zeros(model.nj)
        self.alpha = 1.0-np.exp(-2*np.pi*40*self.dt)
        self.torque_limits = np.minimum(model.joint_torque_max,
                                       [min(m.getMaxTorque(), m.getAvailableTorque()) for m in self.motors])
        if np.any(self.torque_limits <= 0) or not np.isfinite(self.torque_limits).all():
            raise ValueError('Invalid live motor torque limits')
        model.joint_torque_max[:] = self.torque_limits
        model.joint_vel_max[:] = np.minimum(model.joint_vel_max, [m.getMaxVelocity() for m in self.motors])
        self.last_q = model.q0.copy()
        self.hold(model.q0[7:] if startup_joints is None else startup_joints)

    def measure(self):
        joints = np.array([s.getValue() for s in self.encoders])
        position = np.array(self.gps.getValues())
        rpy = np.array(self.imu.getRollPitchYaw())
        world_velocity = np.array(self.gps.getSpeedVector())
        angular_velocity = np.array(self.gyro.getValues())
        if not np.isfinite(np.r_[joints, position, rpy, world_velocity, angular_velocity]).all():
            raise ValueError('Non-finite sensor data')
        rotation = pin.rpy.rpyToMatrix(rpy)
        if self.previous_joints is not None:
            self.velocity += self.alpha*((joints-self.previous_joints)/self.dt-self.velocity)
        self.previous_joints = joints.copy()
        q = np.r_[position, pin.Quaternion(rotation).coeffs(), joints]
        # GPS/IMU/gyro are mounted at the root with identity transform in this WBT.
        v = np.r_[rotation.T @ world_velocity, angular_velocity, self.velocity]
        self.last_q = q.copy()
        return q, v

    def hold(self, joints):
        for motor, value in zip(self.motors, np.clip(joints, self.model.joint_pos_min, self.model.joint_pos_max)):
            motor.setPosition(float(value))
            motor.setVelocity(min(0.5, motor.getMaxVelocity()))
        self.mode = 'hold'

    def torque(self, values):
        if not np.isfinite(values).all():
            raise ValueError('Non-finite torque command')
        if self.mode != 'torque':
            for motor in self.motors:
                motor.setPosition(float('inf'))
                motor.setVelocity(0.)
        bounded = np.clip(values, -self.torque_limits, self.torque_limits)
        for motor, value in zip(self.motors, bounded):
            motor.setTorque(float(value))
        self.mode = 'torque'
        return bounded


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', type=Path, default=DEFAULT_WORLD)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().with_name('deployment.json'),
                        help='settings JSON; defaults to the compiled deployment')
    parser.add_argument('--arm-joints', type=int, choices=(0, 4, 6), default=4)
    parser.add_argument('--gait', choices=('stand', 'trot', 'walk'), default='stand')
    parser.add_argument('--velocity', type=float, default=0., help='forward speed in m/s')
    parser.add_argument('--duration', type=float, help='simulation seconds, including startup')
    parser.add_argument('--startup', type=float, default=2., help='position settling seconds')
    parser.add_argument('--log', type=Path, help='JSONL measurements and complete solve timings')
    parser.add_argument('--scenario', choices=('manual', 'trot', 'arm'), default='manual', help='scripted validation: switch after one second of standing')
    args = parser.parse_args(argv)
    data = json.loads(args.config.read_text()) if args.config else {}
    unknown = set(data)-{f.name for f in fields(Settings)}
    if unknown:
        parser.error(f'Unknown settings: {sorted(unknown)}')
    settings = Settings(**data)
    if settings.compiled_solver:
        settings.compiled_solver = str((args.config.resolve().parent / settings.compiled_solver).resolve())
    settings.validate()
    if args.startup < 0 or (args.duration is not None and args.duration <= args.startup):
        parser.error('duration must exceed nonnegative startup')
    if not np.isfinite(args.velocity) or abs(args.velocity) > 0.15:
        parser.error('Initial speed must be within +/-0.15 m/s')
    return args, settings


def main(argv=None):
    args, settings = arguments(argv)
    model = WebotsModel(args.world, args.arm_joints)
    robot = load_robot_class()()
    basic = robot.getBasicTimeStep()
    if basic not in (1., 2., 10.):
        raise RuntimeError('Supported WorldInfo.basicTimeStep values: 1, 2, 10 ms')
    step = int(basic)
    io = Interface(robot, model, step, startup_posture(model, settings))
    keyboard = robot.getKeyboard()
    keyboard.enable(step)
    log = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log = args.log.open('w', buffering=1)
    def record(value):
        if log:
            log.write(json.dumps(value, allow_nan=False)+'\n')
    start = robot.getTime()
    mpc = None
    plan = None
    arm_reference = None
    leg_reference = None
    plan_time = -np.inf
    next_solve = start
    next_report = start
    next_record = start
    gait = args.gait
    gait_start = start
    forward = args.velocity
    stopped = False
    fault_latched = False
    consecutive_failures = 0
    keys = {}
    failures = misses = solves = 0
    def arm_gains(value):
        return np.broadcast_to(np.asarray(value, float), (6,))[:model.arm_joints]
    kp = np.r_[np.tile(np.broadcast_to(np.asarray(settings.leg_kp, float), (3,)), 4), arm_gains(settings.arm_kp)]
    kd = np.r_[np.tile(np.broadcast_to(np.asarray(settings.leg_kd, float), (3,)), 4), arm_gains(settings.arm_kd)]
    print(f'[wb_mpc_webots] source=wb_mpc_locoman model={model.mass:.6f}kg nq={model.nq} nv={model.nv}', flush=True)
    print(f'[wb_mpc_webots] feedback={step}ms MPC={settings.solve_period*1000:g}ms ({settings.solve_period/io.dt:g} feedback steps)', flush=True)
    print(f'[wb_mpc_webots] startup arm target: {startup_posture(model, settings)[12:].round(4).tolist()} rad', flush=True)
    print('J stand; U trot; O walk; hold W/S forward/back, A/D lateral, Q/E yaw; I/K arm up/down; SPACE hold. Synchronous solving may slow simulation.', flush=True)
    try:
        while robot.step(step) != -1:
            now = robot.getTime()
            if args.duration is not None and now-start >= args.duration:
                io.hold(io.last_q[7:])
                robot.step(step)
                break
            try:
                q, v = io.measure()
            except ValueError as error:
                io.hold(io.last_q[7:]); plan = None
                if now >= next_report:
                    print(error, flush=True); next_report = now+1
                continue
            if now-start < args.startup:
                continue
            if mpc is None:
                # The physical settled posture replaces the B2 SRDF reference.
                if settings.compiled_solver:
                    metadata = json.loads(Path(settings.compiled_solver).resolve().with_suffix('.json').read_text())
                    model.set_nominal(np.asarray(metadata['nominal']))
                else:
                    model.set_nominal(q)
                model.set_gait_sequence(gait, settings.gait_period)
                mpc = MPC(model, settings)
                gait_start = now
                next_solve = now
                print(f'[wb_mpc_webots] OCP ready: {mpc.ocp.opti.nx} variables, {mpc.ocp.opti.ng} constraints', flush=True)
                print(f'[wb_mpc_webots] backend: {settings.compiled_solver or "CasADi interpreted"}', flush=True)
            key = keyboard.getKey()
            while key != -1:
                if key in (keyboard.UP, keyboard.DOWN, keyboard.LEFT, keyboard.RIGHT):
                    keys[{keyboard.UP:'UP', keyboard.DOWN:'DOWN', keyboard.LEFT:'LEFT', keyboard.RIGHT:'RIGHT'}[key]] = now+0.12
                elif 0 <= key < 128:
                    keys[chr(key).upper()] = now+0.12
                key = keyboard.getKey()
            keys = {k: t for k, t in keys.items() if t >= now}
            desired_gait = gait
            for key, name in [('J', 'stand'), ('U', 'trot'), ('O', 'walk')]:
                if key in keys and (not fault_latched or key == 'J'):
                    desired_gait = name
                    stopped = False
                    fault_latched = False
                    consecutive_failures = 0
            if args.scenario == 'trot' and not fault_latched and now-start >= args.startup+1:
                desired_gait = 'trot'
            if desired_gait != gait:
                gait = desired_gait
                model.gait_sequence.gait_type = gait
                # OCP holds the original GaitSequence instance.
                fresh = type(model.gait_sequence)(gait, settings.gait_period)
                model.gait_sequence.__dict__.update(fresh.__dict__)
                gait_start = now
                # Preserve the previous state/acceleration guess; the original
                # warm start already remasks forces for the new gait.
                plan = None; next_solve = now
            if ' ' in keys:
                if not stopped:
                    io.hold(q[7:])
                stopped = True
            excessive_motion = (np.any(np.abs(v[6:]) > 2*model.joint_vel_max+1)
                                or q[2] < 0.18
                                or np.max(np.abs(pin.rpy.matrixToRpy(pin.Quaternion(q[3:7]).matrix())[:2])) > 0.6)
            if excessive_motion and not fault_latched:
                io.hold(q[7:])
                fault_latched = stopped = True
                record({'event': 'fault', 'time': now, 'reason': 'excessive measured motion'})
                print('[wb_mpc_webots] Excessive motion: HOLD latched; reset the world if fallen, J resumes stand.', flush=True)
            if stopped:
                plan = None
                mpc.reset(); next_solve = now
                if now >= next_record:
                    record({'event': 'state', 'time': now, 'q': q.tolist(), 'v': v.tolist(),
                            'rpy': pin.rpy.matrixToRpy(pin.Quaternion(q[3:7]).matrix()).tolist(),
                            'mode': io.mode, 'gait': gait, 'torque': [0.]*model.nj,
                            'contacts': [sensor.getValue() for sensor in io.touches]})
                    next_record = now+0.02
                continue
            base = np.zeros(6)
            if gait != 'stand':
                base[0] = forward+0.05*(('W' in keys)-('S' in keys))
                base[1] = 0.03*(('A' in keys)-('D' in keys))
                base[5] = 0.1*(('Q' in keys)-('E' in keys))
            arm = np.zeros(3)
            arm[0] = 0.02*(('UP' in keys)-('DOWN' in keys))
            arm[1] = 0.02*(('LEFT' in keys)-('RIGHT' in keys))
            arm[2] = 0.02*(('I' in keys)-('K' in keys))
            if args.scenario == 'arm' and args.startup+1 <= now-start < args.startup+2:
                arm[2] = 0.01
            # Solve only on the MPC clock (default 80 ms / 8 physics steps).
            # All other physics steps reuse the plan and update torque feedback.
            if now+1e-9 >= next_solve:
                while next_solve <= now+1e-9:
                    next_solve += settings.solve_period
                wall = time.perf_counter()
                error = None
                try:
                    plan = mpc.solve(q, v, now-gait_start, base, arm)
                    plan_time = now
                    consecutive_failures = 0
                except (RuntimeError, ValueError) as exc:
                    error = str(exc)
                    failures += 1
                    consecutive_failures += 1
                    plan = None
                    io.hold(q[7:])
                    if consecutive_failures >= 3:
                        fault_latched = stopped = True
                        record({'event': 'fault', 'time': now, 'reason': 'three consecutive rejected solves'})
                        print('[wb_mpc_webots] Repeated solve failure: HOLD latched; J resumes stand.', flush=True)
                    mpc.reset()
                elapsed = time.perf_counter()-wall
                solves += 1
                misses += elapsed > settings.solve_period
                record({'event': 'solve', 'time': now, 'elapsed': elapsed, 'ok': error is None,
                        'cv': plan.cv if error is None else None, 'error': error, 'solver': mpc.last_solver})
                if error and (failures <= 3 or now >= next_report):
                    print('[wb_mpc_webots] '+error[:250], flush=True)
            applied = np.zeros(model.nj)
            if plan is not None and now-plan_time <= plan.dt+1e-9:
                qr, vr, ff = plan.sample(max(0., now-plan_time))
                ff = plan.contact_torque(model, max(0., now-plan_time), [sensor.getValue() > 0 for sensor in io.touches], q, v)
                if leg_reference is None or io.mode != 'torque':
                    leg_reference = q[7:19].copy()
                leg_reference += vr[:12]*io.dt
                leg_reference = np.clip(leg_reference, q[7:19]-0.15, q[7:19]+0.15)
                leg_reference = np.clip(leg_reference, model.joint_pos_min[:12], model.joint_pos_max[:12])
                qr[:12] = leg_reference
                if model.arm_joints:
                    # Keep an arm position reference across MPC updates. Resetting
                    # it to the measurement every MPC update removes position-error
                    # feedback and leaves velocity bias under model mismatch.
                    if arm_reference is None or io.mode != 'torque':
                        arm_reference = q[19:].copy()
                    vr[12:] = plan.v[plan.interval(max(0., now-plan_time))+1, 18:]
                    arm_reference += vr[12:]*io.dt
                    arm_reference = np.clip(arm_reference, q[19:]-0.1, q[19:]+0.1)
                    arm_reference = np.clip(arm_reference, model.joint_pos_min[12:], model.joint_pos_max[12:])
                    qr[12:] = arm_reference
                applied = io.torque(ff+kp*(qr-q[7:])+kd*(vr-v[6:]))
            elif io.mode != 'hold':
                io.hold(q[7:])
            if now >= next_record:
                rotation = pin.Quaternion(q[3:7]).matrix()
                record({'event': 'state', 'time': now, 'q': q.tolist(), 'v': v.tolist(),
                        'rpy': pin.rpy.matrixToRpy(rotation).tolist(), 'mode': io.mode, 'gait': gait,
                        'torque': applied.tolist(), 'contacts': [s.getValue() for s in io.touches]})
                next_record = now+0.02
            if now >= next_report:
                print(f'[wb_mpc_webots] t={now-start:.2f}s {gait}/{io.mode} z={q[2]:.3f} failures={failures}/{solves} deadline_misses={misses}/{solves}', flush=True)
                next_report = now+1.
        record({'event': 'summary', 'solves': solves, 'failures': failures, 'deadline_misses': misses})
    finally:
        io.hold(io.last_q[7:])
        if log:
            log.close()


if __name__ == '__main__':
    main()
