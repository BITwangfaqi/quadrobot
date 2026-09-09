"""Model checks and short solves, independent of the Webots controller SDK."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from world_model import WebotsModel
from mpc_bridge import MPC, Settings
from wb_mpc.args import DYN_ARGS


def model_checks(robot):
    assert (robot.nq, robot.nv, robot.nj, robot.nf) == (23, 22, 16, 15)
    assert abs(robot.mass - 18.134388) < 1e-8
    assert robot.names[:3] == ['FL_hip_motor', 'FL_leg_motor', 'FL_foot_motor']
    # Independently traverse the explicit WBT hinge transforms at nonzero q.
    q = robot.q0.copy()
    q[7:] += np.linspace(0.005, 0.025, robot.nj)
    full = robot.expand_q(q)
    placements = {1: pin.SE3(pin.Quaternion(q[3:7]).matrix(), q[:3])}
    for jid, parent, origin, axis, kind in robot.joint_records:
        angle = full[robot.full_model.joints[jid].idx_q]
        motion = (pin.SE3(pin.exp3(axis*angle), np.zeros(3)) if kind == 'HingeJoint'
                  else pin.SE3(np.eye(3), axis*angle))
        placements[jid] = placements[parent] * origin * motion
    expected_com = sum(inertia.mass*(placements[jid]*offset).act(inertia.lever)
                       for jid, offset, inertia in robot.body_records) / robot.mass
    actual_com = pin.centerOfMass(robot.model, robot.data, q)
    np.testing.assert_allclose(actual_com, expected_com, atol=1e-11)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    for frame in robot.full_model.frames:
        if frame.name in ('FR_foot', 'FL_foot', 'RR_foot', 'RL_foot', 'gripperCenter'):
            expected = placements[frame.parentJoint] * frame.placement
            actual = robot.data.oMf[robot.model.getFrameId(frame.name)]
            np.testing.assert_allclose(actual.homogeneous, expected.homogeneous, atol=1e-11)
    # Reduced inertia must preserve mass/gravity, and independent potential
    # finite differences must agree with inverse dynamics generalized gravity.
    gravity = pin.computeGeneralizedGravity(robot.model, robot.data, q)
    eps = 1e-6
    for j in range(robot.nv):
        delta = np.zeros(robot.nv); delta[j] = eps
        plus = pin.centerOfMass(robot.model, robot.data, pin.integrate(robot.model, q, delta))[2]
        minus = pin.centerOfMass(robot.model, robot.data, pin.integrate(robot.model, q, -delta))[2]
        np.testing.assert_allclose(gravity[j], robot.mass*9.81*(plus-minus)/(2*eps), atol=1e-6)
    print('PASS WBT mass, dimensions, independent tree transforms/COM and gravity finite differences')


def provenance():
    root = Path(__file__).resolve().parent
    source = root.parents[2] / 'wb_mpc_locoman'
    manifest = json.loads((root/'upstream_sha256.json').read_text())
    if source.exists():
        for relative, expected in manifest.items():
            assert hashlib.sha256((source/relative).read_bytes()).hexdigest() == expected, relative
    print('PASS upstream source checksums (when source checkout is available)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solve', action='store_true')
    parser.add_argument('--all-models', action='store_true')
    parser.add_argument('--solver', choices=('fatrop', 'ipopt', 'osqp'), default='fatrop')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'validation/offline.json')
    args = parser.parse_args()
    provenance()
    model_checks(WebotsModel())
    results = []
    if args.solve:
        for name in DYN_ARGS if args.all_models else ['whole_body_rnea']:
            robot = WebotsModel()
            settings = Settings(dynamics=name, solver=args.solver)
            mpc = MPC(robot, settings)
            plan = mpc.solve(robot.q0, np.zeros(robot.nv), 0.)
            # Verify the compiled symbolic RNEA against numerical Pinocchio
            # and world-aligned contact Jacobians, including the sign of f.
            for i in range(len(plan.q)):
                qi, vi, ai = plan.q[i], plan.v[i], plan.a[i]
                tau = pin.rnea(robot.model, robot.data, qi, vi, ai).copy()
                for e, fid in enumerate(mpc.ocp.ee_frames):
                    jac = pin.computeFrameJacobian(robot.model, robot.data, qi, fid, pin.LOCAL_WORLD_ALIGNED)
                    tau -= jac[:3].T @ plan.forces[i, 3*e:3*e+3]
                np.testing.assert_allclose(tau[6:], plan.tau[i], atol=1e-8)
                assert np.max(np.abs(tau[:6])) < settings.max_cv*2
            assert all(x.shape == (16,) for x in plan.sample(settings.dt_min/2))
            no_contact = plan.contact_torque(robot, 0., [False]*4, plan.q[0], plan.v[0])
            expected = pin.rnea(robot.model, robot.data, plan.q[0], plan.v[0], plan.a[0]).copy()[6:]
            np.testing.assert_allclose(no_contact, expected, atol=1e-8)
            all_contact = plan.contact_torque(robot, 0., [True]*4, plan.q[0], plan.v[0])
            np.testing.assert_allclose(all_contact, plan.tau[0], atol=1e-8)
            try:
                plan.sample(settings.solve_period+0.001)
                raise AssertionError('Expired trajectory was accepted')
            except ValueError:
                pass
            # Check every 10 ms execution sample and all nonuniform stage boundaries.
            for age in np.r_[np.arange(0., settings.solve_period, .01), np.cumsum(plan.dts)]:
                if age <= settings.solve_period:
                    i = plan.interval(float(age))
                    _, _, tau = plan.sample(float(age))
                    np.testing.assert_allclose(tau, plan.tau[i])
            assert plan.interval(settings.solve_period-.001) >= 0
            results.append({'model': name, 'settings': asdict(settings), 'variables': mpc.ocp.opti.nx,
                            'constraints': mpc.ocp.opti.ng, 'cv': plan.cv, 'elapsed': plan.elapsed})
            print(f'PASS {name}/{args.solver}: CV={plan.cv:.3g}, complete solve={plan.elapsed:.3f}s')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'model_checks': 'passed', 'solves': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
