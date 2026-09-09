"""Thin runtime adapter around the original make_ocp and solver_function.

The OCP expressions, weights, nonuniform grid and gait remain in wb_mpc/.
This layer supplies measurements and checks/decodes the resulting trajectory.
"""
from copy import deepcopy
from dataclasses import dataclass
import time

import casadi as ca
import numpy as np
import pinocchio as pin

from wb_mpc.args import DYN_ARGS, SOLVER_ARGS
from wb_mpc.optimization import make_ocp


@dataclass
class Settings:
    dynamics: str = 'whole_body_rnea'
    solver: str = 'fatrop'
    nodes: int = 14
    tau_nodes: int = 3
    dt_min: float = 0.02
    dt_max: float = 0.04
    solve_period: float = 0.08
    gait_period: float = 1.12
    swing_height: float = 0.025
    swing_vel_limits: tuple = (0.02, -0.04)
    max_iter: int = 80
    max_cv: float = 0.001
    leg_kp: tuple = (60., 60., 20.)
    leg_kd: tuple = (0.5, 0.5, 0.1)
    arm_kp: tuple = (300., 300., 300., 5., 5., 5.)
    arm_kd: tuple = (12., 12., 12., 0.04, 0.04, 0.04)
    include_base: bool = True
    include_acc: bool = True
    warm_start: bool = True
    sqp_iters: int = 2
    qp_max_iter: int = 20
    arm_force: tuple = (0., 0., 0.)
    compiled_solver: str = ''

    @classmethod
    def from_metadata(cls, values):
        # Older build/test metadata recorded this removed Python-only policy.
        # It never affected the exported Fatrop function; keep those artifacts usable.
        values = dict(values)
        values.pop('ipopt_fallback', None)
        return cls(**values)

    def validate(self):
        if self.dynamics not in DYN_ARGS or self.solver not in SOLVER_ARGS:
            raise ValueError('Unknown model or solver')
        if not 2 <= self.tau_nodes <= self.nodes or self.nodes < 3:
            raise ValueError('Require 2 <= tau_nodes <= nodes and nodes >= 3')
        for key in ('dt_min', 'dt_max', 'solve_period', 'gait_period', 'swing_height', 'max_cv'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.dt_max < self.dt_min or self.max_iter < 1:
            raise ValueError('Invalid time grid or iteration limit')
        if self.solve_period > np.sum(np.geomspace(self.dt_min, self.dt_max, self.nodes)[:-2])+1e-9:
            raise ValueError('Execution period must leave two prediction stages for decoding')
        if self.solver == 'fatrop' and self.dynamics == 'whole_body_rnea' and not self.include_acc:
            raise ValueError('Original Fatrop structure detection requires include_acc=True')
        if np.asarray(self.arm_force).shape != (3,) or not np.isfinite(self.arm_force).all():
            raise ValueError('arm_force must be a finite three-vector in world coordinates')
        limits = np.asarray(self.swing_vel_limits, float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] < 0 or limits[1] > 0:
            raise ValueError('swing_vel_limits must be finite [nonnegative liftoff, nonpositive touchdown]')
        if self.sqp_iters < 1 or self.qp_max_iter < 1:
            raise ValueError('SQP/QP iteration limits must be positive')
        for name in ('leg_kp', 'leg_kd', 'arm_kp', 'arm_kd'):
            values = np.asarray(getattr(self, name))
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f'{name} must contain finite nonnegative gains')
        for name in ('leg_kp', 'leg_kd'):
            values = np.asarray(getattr(self, name))
            if values.ndim > 1 or (values.ndim == 1 and len(values) != 3):
                raise ValueError(f'{name} must be a scalar or hip/thigh/knee gains')
        for name in ('arm_kp', 'arm_kd'):
            values = np.asarray(getattr(self, name))
            if values.ndim > 1 or (values.ndim == 1 and len(values) != 6):
                raise ValueError(f'{name} must be a scalar or six joint gains')


@dataclass
class Plan:
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    tau: np.ndarray
    forces: np.ndarray
    dt: float
    cv: float
    elapsed: float
    dts: np.ndarray

    def interval(self, age):
        if not 0 <= age <= self.dt + 1e-9:
            raise ValueError('Plan has expired; no unbounded-torque extrapolation')
        return min(int(np.searchsorted(np.cumsum(self.dts), min(age, self.dt), side='right')), len(self.dts)-2)

    def sample(self, age):
        if not 0 <= age <= self.dt + 1e-9:
            raise ValueError('Plan has expired; no unbounded-torque extrapolation')
        i = self.interval(age)
        local_age = min(age, self.dt)-np.sum(self.dts[:i])
        return (self.q[i, 7:] + local_age*self.v[i, 6:] + 0.5*local_age**2*self.a[i, 6:],
                self.v[i, 6:] + local_age*self.a[i, 6:], self.tau[i].copy())


    def contact_torque(self, robot, age, contacts, q, v):
        """Measured-state inverse dynamics with contact and base feedback.

        Decoded RNEA uses tau = M*a + h - J.T*f. A scheduled stance
        foot that is still airborne has no physical reaction balancing -J.T*f.
        Keep the acceleration/gravity term, but remove that missing reaction.
        A bounded base-pose correction is distributed over measured support feet.
        """
        i = self.interval(age)
        tau = pin.rnea(robot.model, robot.data, q, v, self.a[i]).copy()[6:]
        active = [foot for foot, contact in enumerate(contacts)
                  if contact and self.forces[i, 3*foot+2] > 1.]
        corrections = {}
        if active:
            pin.framesForwardKinematics(robot.model, robot.data, q)
            rotation = pin.Quaternion(q[3:7]).matrix()
            target_rotation = pin.Quaternion(self.q[i, 3:7]).matrix()
            position_error = self.q[i, :3]-q[:3]
            position_error[2] = robot.q0[2]-q[2]
            velocity_error = target_rotation @ self.v[i, :3]-rotation @ v[:3]
            linear = robot.mass*(80*position_error+12*velocity_error)
            angular = 12*pin.log3(target_rotation @ rotation.T)+2*(target_rotation @ self.v[i, 3:6]-rotation @ v[3:6])
            blocks = [np.vstack((np.eye(3), pin.skew(robot.data.oMf[robot.foot_frames[foot]].translation-q[:3])))
                      for foot in active]
            matrix = np.hstack(blocks)
            delta = np.linalg.lstsq(matrix, np.r_[linear, angular], rcond=1e-5)[0]
            for k, foot in enumerate(active):
                force = self.forces[i, 3*foot:3*foot+3]+delta[3*k:3*k+3]
                force[2] = np.clip(force[2], 0., robot.mass*9.81)
                force[:2] = np.clip(force[:2], -.5*force[2], .5*force[2])
                corrections[foot] = force
        for foot, (fid, contact) in enumerate(zip(robot.foot_frames, contacts)):
            if contact:
                jac = pin.computeFrameJacobian(robot.model, robot.data, q, fid,
                                               pin.LOCAL_WORLD_ALIGNED)
                tau -= jac[:3, 6:].T @ corrections.get(foot, self.forces[i, 3*foot:3*foot+3])
        if robot.arm_ee_frame:
            jac = pin.computeFrameJacobian(robot.model, robot.data, q, robot.arm_ee_frame,
                                           pin.LOCAL_WORLD_ALIGNED)
            tau -= jac[:3, 6:].T @ self.forces[i, 12:15]
        return tau


class MPC:
    def __init__(self, robot, settings):
        settings.validate()
        self.robot, self.settings = robot, settings
        grid = np.geomspace(settings.dt_min, settings.dt_max, settings.nodes)
        self.execution_nodes = int(np.searchsorted(np.cumsum(grid), settings.solve_period-1e-9))+1
        self.execution_dts = grid[:self.execution_nodes+1]
        dyn_args = deepcopy(DYN_ARGS[settings.dynamics])
        if 'include_base' in dyn_args:
            dyn_args['include_base'] = settings.include_base
        if 'include_acc' in dyn_args:
            dyn_args['include_acc'] = settings.include_acc
        self.ocp = make_ocp(dynamics=settings.dynamics, dyn_args=dyn_args,
                            robot=robot, nodes=settings.nodes, tau_nodes=max(settings.tau_nodes, self.execution_nodes), warm_start=settings.warm_start)
        ocp = self.ocp
        ocp.set_time_params(settings.dt_min, settings.dt_max)
        ocp.set_swing_params(settings.swing_height, settings.swing_vel_limits)
        ocp.set_tracking_targets(np.zeros(6), np.zeros(3), np.zeros(3))
        ocp.update_params(ocp.x_nom, 0.)
        options = deepcopy(SOLVER_ARGS[settings.solver])
        if settings.solver == 'osqp':
            options['iters'] = settings.sqp_iters
            options['opts']['max_iter'] = settings.qp_max_iter
            options['opts']['verbose'] = False
        if settings.solver in ('fatrop', 'ipopt'):
            options['opts'][settings.solver + '.max_iter'] = settings.max_iter
            options['opts']['print_time'] = False
            if settings.solver == 'fatrop':
                options['opts']['debug'] = False
        self.last_solver = settings.solver
        if settings.solver in ('fatrop', 'ipopt'):
            options['opts'][settings.solver+'.tol'] = 1e-5
        ocp.init_solver(settings.solver, options)
        self.decoder = self._decoder()
        if settings.compiled_solver:
            self._load_compiled(settings.compiled_solver)

    def _load_compiled(self, filename):
        import hashlib
        import json
        from dataclasses import asdict
        from pathlib import Path
        path = Path(filename).resolve()
        metadata = json.loads(path.with_suffix('.json').read_text())
        expected = asdict(self.settings)
        expected['compiled_solver'] = ''
        # Round trip normalizes tuple/list JSON representations.
        recorded = asdict(Settings.from_metadata(metadata['settings']))
        if json.loads(json.dumps(recorded)) != json.loads(json.dumps(expected)):
            raise ValueError('Compiled solver settings do not match this controller')
        if metadata['model_signature'] != self.robot.signature():
            raise ValueError('Compiled solver model/reference/limits do not match the live robot')
        if metadata.get('casadi') != ca.__version__:
            raise ValueError('Compiled solver CasADi version does not match this environment')
        if not metadata.get('equivalence_passed'):
            raise ValueError('Compiled solver has not passed export equivalence validation')
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get('library_sha256'):
            raise ValueError('Compiled solver checksum mismatch; rebuild the deployment')
        if set(metadata.get('functions', [])) != {'solver_function', 'webots_decode', 'g_data'}:
            raise ValueError('Deployment must contain solver, trajectory decoder and constraints')
        self.ocp.solver_function = ca.external('solver_function', str(path))
        self.decoder = ca.external('webots_decode', str(path))
        self.ocp.g_data = ca.external('g_data', str(path))

    def _decoder(self):
        ocp = self.ocp
        q, v, a, f, tau = [], [], [], [], []
        for i in range(self.execution_nodes+1):
            qi, vi, fi = ocp.get_q(i), ocp.get_v(i), ocp.get_forces(i)
            if self.settings.dynamics == 'centroidal_vel':
                aj = (ocp.get_v(i+1)[6:] - vi[6:]) / ocp.dts[i]
                ai = ca.vertcat(ocp.dyn.base_acc_dynamics()(qi, vi, aj, fi), aj)
            elif self.settings.dynamics == 'whole_body_aba':
                ai = ocp.dyn.aba_dynamics()(qi, vi, ocp.get_tau_j(i), fi)
            else:
                ai = ocp.get_a(i)
            ti = ocp.dyn.rnea_dynamics()(qi, vi, ai, fi)[6:]
            q.append(qi); v.append(vi); a.append(ai); f.append(fi); tau.append(ti)
        return ca.Function('webots_decode', [ocp.opti.x, ocp.opti.p],
                           [ca.horzcat(*xs).T for xs in (q, v, a, tau, f)])

    def reset(self):
        ocp = self.ocp
        ocp.DX_prev = [np.zeros(ocp.ndx_opt) for _ in range(ocp.nodes+1)]
        ocp.U_prev = [np.zeros(n) for n in ocp.nu_opt]
        ocp.lam_g = None
        ocp.opti.set_initial(ocp.opti.x, 0.)
        # The original warm_start_variables will restore gravity support and
        # clear swing forces using the newly updated contact schedule.

    def solve(self, q, v, sim_time, base_velocity=None, arm_velocity=None):
        start = time.perf_counter()
        self.last_solver = self.settings.solver
        ocp = self.ocp
        base = np.zeros(6) if base_velocity is None else np.asarray(base_velocity, float)
        arm = np.zeros(3) if arm_velocity is None else np.asarray(arm_velocity, float)
        if not np.isfinite(np.r_[q, v, base, arm]).all():
            raise ValueError('Non-finite MPC input')
        if self.settings.dynamics == 'centroidal_vel':
            h = pin.computeCentroidalMap(self.robot.model, self.robot.data, q) @ v / self.robot.mass
            initial = np.r_[h, q]
        else:
            initial = np.r_[q, v]
        ocp.set_tracking_targets(base, arm, np.asarray(self.settings.arm_force))
        ocp.update_params(initial, sim_time+1e-9)
        if self.settings.solver == 'osqp':
            ocp.solve(retract_all=False)
            solution = ca.vertcat(*[item for pair in zip(ocp.DX_prev[:-1], ocp.U_prev) for item in pair], ocp.DX_prev[-1])
        else:
            solution = ocp.solver_function(*ocp.get_solver_params())
        parameters = ocp.opti.value(ocp.opti.p)
        g, lo, hi = ocp.g_data(solution, parameters)
        cv = float(ocp.constr_viol_norm_inf(g, lo, hi))
        if not np.isfinite(np.asarray(solution)).all() or not np.isfinite(cv) or cv > self.settings.max_cv:
            raise RuntimeError(f'Rejected MPC result: constraint violation {cv:.6g}')
        decoded = [np.asarray(x, float) for x in self.decoder(solution, parameters)]
        if not all(np.isfinite(x).all() for x in decoded):
            raise RuntimeError('Non-finite decoded trajectory')
        # Models with approximate torque constraints still require physically
        # valid RNEA commands before Webots accepts their results.
        if np.any(np.abs(decoded[3]) > self.robot.joint_torque_max + 0.01):
            raise RuntimeError('Decoded inverse-dynamics torque exceeds motor limits')
        if self.settings.solver != 'osqp':
            ocp.retract_stacked_sol(solution, retract_all=False)
        # Original histories are for offline plots; bound live-controller RAM.
        for name in ('q_sol', 'v_sol', 'a_sol', 'forces_sol', 'tau_sol'):
            if hasattr(ocp, name):
                getattr(ocp, name)[:] = getattr(ocp, name)[-2:]
        return Plan(*decoded, self.settings.solve_period, cv, time.perf_counter()-start, self.execution_dts.copy())
