"""Nonlinear whole-body inverse-dynamics MPC, following Molnar et al. (2026).

Each stage contains tangent state [delta_q, v] and input [a, F_feet, tau]
through k=2; later stages eliminate tau and retain the floating-base RNEA
equations. The prescribed tool force is a parameter, not a redundant variable.
The default locks the last two arm joints, as in the paper: 22 optimized
velocities, with all body inertias retained. There is no SRB planner or WBC QP.
"""

from dataclasses import dataclass, replace
import time

import numpy as np

try:
    from .config import MPCConfig
except ImportError:
    from config import MPCConfig


@dataclass
class MPCSolution:
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    tau: np.ndarray
    forces: np.ndarray
    dt: np.ndarray
    constraint_violation: float
    solve_time: float
    status: str
    iterations: int = 0
    timings: dict = None

    def sample(self, elapsed):
        """Paper Eq. (9), only across the first two bounded torque intervals."""
        elapsed = float(elapsed)
        count = min(2, len(self.tau) - 1)
        times = np.r_[0.0, np.cumsum(self.dt)]
        if not np.isfinite(elapsed) or elapsed < 0 or elapsed > times[count] + 1e-9:
            raise ValueError("MPC trajectory is stale; refusing torque extrapolation")
        k = min(int(np.searchsorted(times, elapsed, side="right") - 1), count - 1)
        local = max(0.0, elapsed - times[k])
        alpha = np.clip(local / self.dt[k], 0.0, 1.0)
        return (self.q[k, 7:] + local * self.v[k, 6:],
                self.v[k, 6:] + local * self.a[k, 6:],
                (1.0 - alpha) * self.tau[k] + alpha * self.tau[k + 1])


def swing_reference(phase, duration, height):
    """C1 piecewise cubic height and vertical speed; zero at swing endpoints."""
    phase = float(np.clip(phase, 0.0, 1.0))
    rising = phase <= 0.5
    u = 2.0 * phase if rising else 2.0 * (1.0 - phase)
    z = height * u * u * (3.0 - 2.0 * u)
    dz = (1.0 if rising else -1.0) * height * 12.0 * u * (1.0 - u) / duration
    return z, dz


class InverseDynamicsMPC:
    def __init__(self, model, config=None):
        try:
            import casadi as ca
        except ImportError as error:
            raise ImportError("CasADi is required; see inv_dyn_mpc/requirements.txt") from error
        self.ca, self.full_model = ca, model
        self.config = (config or MPCConfig()).validate()
        if self.config.lock_wrist:
            try:
                from .reduced_model import ReducedModel
            except ImportError:
                from reduced_model import ReducedModel
            model = ReducedModel(model, self.config.nominal_arm[-2:])
        self.model = model
        self.dt = self.config.time_steps()
        self.times = np.r_[0.0, np.cumsum(self.dt)]
        self.functions = model.casadi_functions()
        self._previous = None
        self._dual = None
        self._previous_contacts = None
        self.torque_nodes = (self.config.horizon if self.config.torque_nodes == 0
                             else min(3, self.config.horizon))
        started = time.perf_counter()
        self._build()
        self.build_time = time.perf_counter() - started

    def _build(self):
        ca, cfg, fn = self.ca, self.config, self.functions
        n, nv, nq = cfg.horizon, self.model.nv, self.model.nq
        self.nx = 2 * nv
        self.nu = [nv + 12 + (self.model.na if k < self.torque_nodes else 0) for k in range(n)]
        # x0 is measured, so substitute it exactly instead of adding 44 fixed
        # optimization variables and 44 initial-state equalities to the KKT.
        self.nx_stage = [0] + [self.nx] * n
        x = [ca.MX.sym(f"x{k}", self.nx_stage[k]) for k in range(n + 1)]
        u = [ca.MX.sym(f"u{k}", self.nu[k]) for k in range(n)]
        qhat, vhat = ca.MX.sym("qhat", nq), ca.MX.sym("vhat", nv)
        states = [ca.vertcat(ca.MX.zeros(nv), vhat)] + x[1:]
        # Per node: absolute q target, local v target, tool p/v, feet z/vz, contact flags.
        np_stage = nq + nv + 6 + 8 + 4
        targets = [ca.MX.sym(f"p{k}", np_stage) for k in range(n + 1)]
        tool_force = ca.MX.sym("tool_force", 3)
        self.parameter_size = nq + nv + (n + 1) * np_stage + 3
        self._stage_parameter_size = np_stage
        self._xs, self._us = [], []
        decision, lower_x, upper_x = [], [], []
        cursor = 0
        self._force_slices = []
        for k in range(n + 1):
            decision.append(x[k])
            self._xs.append(slice(cursor, cursor + self.nx_stage[k]))
            cursor += self.nx_stage[k]
            lower_x.extend([-np.inf] * self.nx_stage[k])
            upper_x.extend([np.inf] * self.nx_stage[k])
            if k < n:
                decision.append(u[k])
                self._us.append(slice(cursor, cursor + self.nu[k]))
                self._force_slices.append(slice(cursor + nv, cursor + nv + 12))
                cursor += self.nu[k]
                lower_x.extend(np.full(nv + 12, -np.inf))
                upper_x.extend(np.full(nv + 12, np.inf))
                if k < self.torque_nodes:
                    lower_x.extend(-self.model.description.effort_limits)
                    upper_x.extend(self.model.description.effort_limits)
        self.lbx, self.ubx = np.asarray(lower_x), np.asarray(upper_x)
        self._decision_size = cursor
        self._state_indices = np.array([np.arange(rows.start, rows.stop) for rows in self._xs[1:]])
        self._acc_indices = np.array([np.arange(rows.start, rows.start + nv) for rows in self._us])
        self._feet_indices = np.array([np.arange(rows.start + nv, rows.start + nv + 12) for rows in self._us])
        self._torque_indices = np.array([np.arange(rows.start + nv + 12, rows.stop)
                                        for rows in self._us[:self.torque_nodes]])
        warm_q = ca.SX.sym('warm_q', nq, n + 1)
        warm_origin = ca.SX.sym('warm_origin', nq)
        self._warm_difference = ca.Function('warm_difference', [warm_origin, warm_q],
            [ca.horzcat(*(fn['difference'](warm_origin, warm_q[:, k]) for k in range(n + 1)))])
        constraints, lower_g, upper_g, equality, ng = [], [], [], [], []

        def constrain(expr, lb=0.0, ub=0.0, is_equality=True):
            expr = ca.vec(expr)
            size = int(expr.numel())
            rows = slice(len(lower_g), len(lower_g) + size)
            constraints.append(expr)
            lower_g.extend(np.broadcast_to(lb, (size,)))
            upper_g.extend(np.broadcast_to(ub, (size,)))
            equality.extend([is_equality] * size)
            return rows

        qweights = ca.DM(np.r_[cfg.base_position_weight, cfg.base_rotation_weight,
                               np.tile([cfg.hip_position_weight, cfg.leg_position_weight, cfg.leg_position_weight], 4),
                               np.full(self.model.na - 12, cfg.arm_position_weight)])
        vweights = ca.DM(np.r_[cfg.base_velocity_weight, np.full(12, cfg.joint_velocity_weight),
                               np.full(self.model.na - 12, cfg.arm_joint_velocity_weight)])
        tauweights = ca.DM(np.r_[np.full(12, cfg.torque_weight),
                                np.full(self.model.na - 12, cfg.arm_torque_weight)])
        cost = 0
        stage_functions = {}
        derivative_functions = {}
        stage_records, transitions = [], []
        for k in range(n + 1):
            # Fatrop expects transition first, then local path constraints.
            if k < n:
                transition = x[k + 1] - ca.vertcat(states[k][:nv] + self.dt[k] * states[k][nv:],
                                                   states[k][nv:] + self.dt[k] * u[k][:nv])
                constrain(transition)
                transitions.append(transition)
            path_start = len(lower_g)
            expression_start = len(constraints)
            q, v = fn["integrate"](qhat, states[k][:nv]), states[k][nv:]
            target = targets[k]
            qref, vref = target[:nq], target[nq:nq + nv]
            pt, vt = target[nq + nv:nq + nv + 3], target[nq + nv + 3:nq + nv + 6]
            heights = target[nq + nv + 6:nq + nv + 10]
            vertical_speeds = target[nq + nv + 10:nq + nv + 14]
            contacts = target[-4:]
            # Official OCP uses a quadratic error in the initial tangent chart.
            # The reference difference depends only on parameters, not decisions.
            qerror = states[k][:nv] - fn["difference"](qhat, qref)
            stage_cost = ca.dot(qweights, qerror ** 2) + ca.dot(vweights, (v - vref) ** 2)
            jac = fn["foot_jacobians"](q)
            if k > 0:
                # q1 = integrate(qhat, dt*vhat) cannot be changed by a0.
                # Allow a small encoder/contact tolerance at this one node;
                # k>=2 retains strict operational joint position bounds.
                margin = cfg.initial_joint_limit_tolerance if k == 1 else 0.0
                constrain(q[7:], self.model.description.lower_limits - margin,
                          self.model.description.upper_limits + margin, False)
                constrain(v[6:], -self.model.description.velocity_limits,
                          self.model.description.velocity_limits, False)
                # x0 is the measured state: imposing Jv=0 there makes any
                # sensor noise or impact infeasible. Apply path targets at k>=1.
                for leg in range(4) if k < n else ():
                    velocity = jac[3 * leg:3 * leg + 3, :] @ v
                    constrain(ca.vertcat(contacts[leg] * velocity[:2],
                                         velocity[2] - vertical_speeds[leg]))
                # In physical contact mode velocity is the specified task.
                # In free space, position feedback creates a velocity command.
                if k < n:
                    constrain(fn["tool_jacobian"](q) @ v - vt)
            if k < n:
                acceleration = u[k][:nv]
                forces = u[k][nv:nv + 12]
                residual = (fn["rnea"](q, v, acceleration) - jac.T @ forces
                            - fn["tool_jacobian"](q).T @ tool_force)
                if k < self.torque_nodes:
                    torque = u[k][nv + 12:]
                    constrain(residual - ca.vertcat(ca.MX.zeros(6), torque))
                    stage_cost += ca.dot(tauweights, torque ** 2)
                else:
                    # Unbounded tail torques are recovered for diagnostics only.
                    # The execution API can never sample this part of the horizon.
                    constrain(residual[:6])
                fref = []
                for leg in range(4):
                    force = forces[3 * leg:3 * leg + 3]
                    # Strict slack in swing avoids a degenerate cone at F=0.
                    constrain(cfg.friction ** 2 * force[2] ** 2 - ca.sumsqr(force[:2])
                              + (1.0 - contacts[leg]), 0, np.inf, False)
                    fref.extend([0, 0, contacts[leg] * self.model.total_mass * 9.81 / ca.sum1(contacts)])
                stage_cost += cfg.acceleration_weight * ca.sumsqr(acceleration)
                stage_cost += cfg.force_weight * ca.sumsqr(forces - ca.vertcat(*fref))
            ng.append(len(lower_g) - path_start)
            # Reuse one symbolic stage function for each constraint layout.
            # Keeping the global graph as MX avoids expanding twelve copies of
            # the same RNEA/Jacobian/Hessian into a giant C translation unit.
            kind = ("terminal" if k == n else "initial" if k == 0 else
                    "first" if k == 1 else "early" if k < self.torque_nodes else "tail")
            local_u = u[k] if k < n else ca.MX.sym("terminal_u", 0)
            local_p = ca.vertcat(qhat, vhat, target, tool_force)
            if kind not in stage_functions:
                stage_functions[kind] = ca.Function(
                    f"stage_{kind}", [x[k], local_u, local_p],
                    [stage_cost, ca.vertcat(*constraints[expression_start:])]
                ).expand(f"stage_{kind}", {"cse": True})
                # Differentiate only the stage decisions, never the measured
                # state/reference parameters. Automatic MX reverse sweeps of a
                # large multi-input function otherwise compute unused parameter
                # derivatives and create enormous generated C functions.
                local_nx = self.nx_stage[k]
                z = ca.SX.sym("z", local_nx + int(local_u.numel()))
                sp = ca.SX.sym("p", int(local_p.numel()))
                sf, sg = stage_functions[kind](z[:local_nx], z[local_nx:], sp)
                lf, lg = ca.SX.sym("lf"), ca.SX.sym("lg", ng[-1])
                lag = lf * sf + ca.dot(lg, sg)
                hess, lag_grad = ca.hessian(lag, z)
                derivative_functions[kind] = (
                    ca.Function(f"grad_{kind}", [z, sp], [ca.gradient(sf, z)], {"cse": True}),
                    ca.Function(f"jac_{kind}", [z, sp], [ca.jacobian(sg, z)], {"cse": True}),
                    ca.Function(f"hess_{kind}", [z, sp, lf, lg], [lag_grad, hess], {"cse": True}),
                )
            stage_f, stage_g = stage_functions[kind](x[k], local_u, local_p)
            cost += stage_f
            constraints[expression_start:] = [stage_g]
            stage_records.append((kind, ca.vertcat(x[k], local_u), local_p, path_start, ng[-1]))
        self.lbg, self.ubg = np.asarray(lower_g), np.asarray(upper_g)
        w, g = ca.vertcat(*decision), ca.vertcat(*constraints)
        p = ca.vertcat(qhat, vhat, *targets, tool_force)
        nlp = {"x": w, "p": p, "f": cost, "g": g}
        full_fn = self.full_model.casadi_functions()
        q_out, v_out, a_out, f_out, tau_out = [], [], [], [], []
        for k in range(n + 1):
            qk, vk = fn["integrate"](qhat, states[k][:nv]), states[k][nv:]
            if cfg.lock_wrist:
                qk, vk = ca.vertcat(qk, cfg.nominal_arm[-2:]), ca.vertcat(vk, 0., 0.)
            q_out.append(qk)
            v_out.append(vk)
            if k < n:
                ak = u[k][:nv]
                if cfg.lock_wrist:
                    ak = ca.vertcat(ak, 0., 0.)
                fk = ca.vertcat(u[k][nv:nv + 12], tool_force)
                residual = (full_fn["rnea"](qk, vk, ak)
                            - full_fn["foot_jacobians"](qk).T @ fk[:12]
                            - full_fn["tool_jacobian"](qk).T @ tool_force)
                tk = (ca.vertcat(u[k][nv + 12:], residual[self.model.nv:])
                      if k < self.torque_nodes else residual[6:])
                a_out.append(ak)
                f_out.append(fk)
                tau_out.append(tk)
        self.retraction_function = ca.Function("retract_solution", [w, p],
            [ca.horzcat(*items) for items in (q_out, v_out, a_out, f_out, tau_out)])
        options = {"print_time": False, "error_on_fail": False, "expand": False,
                   "calc_lam_p": False, "calc_lam_x": False}
        lam_f, lam_g = ca.MX.sym("lam_f"), ca.MX.sym("lam_g", len(lower_g))
        gradients, jacobians, hessians, lag_gradients = [], [], [], []
        transition_gradient = ca.MX.zeros(cursor)
        evaluated = {}
        for kind in derivative_functions:
            records = [(k, record) for k, record in enumerate(stage_records) if record[0] == kind]
            method = 'openmp' if cfg.jit and cfg.derivative_threads > 1 and len(records) > 1 else 'serial'
            zs = ca.horzcat(*(r[1] for _, r in records))
            ps = ca.horzcat(*(r[2] for _, r in records))
            lgs = ca.horzcat(*(lam_g[r[3]:r[3]+r[4]] for _, r in records))
            count = len(records)
            mapped_jac = derivative_functions[kind][1].map(count, method, min(count, cfg.derivative_threads))
            mapped_hess = derivative_functions[kind][2].map(count, method, min(count, cfg.derivative_threads))
            jac_blocks = mapped_jac(zs, ps)
            lag_blocks, hess_blocks = mapped_hess(zs, ps, ca.repmat(lam_f, 1, count), lgs)
            width = int(zs.size1())
            for column, (k, _) in enumerate(records):
                span = slice(column * width, (column + 1) * width)
                evaluated[k] = (jac_blocks[:, span], lag_blocks[:, column], hess_blocks[:, span])
        for k, (kind, zk, pk, start, count) in enumerate(stage_records):
            grad = derivative_functions[kind][0](zk, pk)
            jac, lag_grad, hess = evaluated[k]
            gradients.append(grad)
            if k < n:
                jacobians.append(ca.jacobian(transitions[k], w))
                transition_gradient += jacobians[-1].T @ lam_g[start - self.nx:start]
            left, width = self._xs[k].start, int(zk.numel())
            jacobians.append(ca.horzcat(ca.MX(ca.Sparsity(count, left)), jac,
                                       ca.MX(ca.Sparsity(count, cursor - left - width))))
            lag_gradients.append(lag_grad)
            hessians.append(hess)
        full_gradient = ca.vertcat(*gradients)
        full_jacobian = ca.vertcat(*jacobians)
        full_hessian = ca.diagcat(*hessians)
        options["cache"] = {
            "nlp_grad_f": ca.Function("nlp_grad_f", [w, p],
                                     ([cost] if cfg.solver == "ipopt" else []) + [full_gradient]),
            "nlp_jac_g": ca.Function("nlp_jac_g", [w, p],
                                    [g, full_jacobian]),
            "nlp_hess_l": ca.Function("nlp_hess_l", [w, p, lam_f, lam_g],
                                     [ca.triu(full_hessian)] if cfg.solver == "ipopt" else
                                     [ca.vertcat(*lag_gradients) + transition_gradient, full_hessian]),
        }
        if cfg.solver == "fatrop":
            options.update({"structure_detection": "manual", "N": n,
                            "nx": self.nx_stage, "nu": self.nu + [0],
                            "ng": ng, "equality": equality,
                            "fatrop": {"max_iter": cfg.max_iterations, "tol": cfg.tolerance,
                                       "print_level": 0, "warm_start_init_point": cfg.warm_start,
                                       "mu_init": cfg.barrier_initial,
                                       "warm_start_mult_bound_push": cfg.bound_push,
                                       "bound_push": cfg.bound_push}})
        else:
            options["ipopt"] = {"max_iter": cfg.max_iterations, "tol": cfg.tolerance,
                                 "print_level": 0, "sb": "yes", "bound_relax_factor": 0.0,
                                 "fixed_variable_treatment": "make_constraint",
                                 "warm_start_init_point": "yes" if cfg.warm_start else "no",
                                 "mu_init": cfg.barrier_initial}
        try:
            if cfg.jit:
                try:
                    from .solver_cache import compiled_solver
                except ImportError:
                    from solver_cache import compiled_solver
                self.solver, self.retraction_function, self.cache_hit = compiled_solver(
                    ca, nlp, options, self.model, cfg, self.retraction_function)
            else:
                self.solver = ca.nlpsol("whole_body_inverse_dynamics", cfg.solver, nlp, options)
                self.cache_hit = False
        except RuntimeError as error:
            raise RuntimeError(f"Cannot build {cfg.solver} NLP. Install a CasADi build with this "
                               "solver, or set solver='ipopt' in JSON. " + str(error)) from error
        self.constraint_function = ca.Function("mpc_constraints", [w, p], [g])

    def _parameters(self, q, v, reference, schedule):
        cfg = self.config
        contacts = schedule.horizon(self.times)
        parts = [q, v]
        tool_velocity = reference.tool_velocity.copy()
        if not cfg.tool_force_enabled:
            tool_velocity += cfg.tool_position_gain * (reference.tool_position - self.model.tool_position(q))
        if reference.yaw_rate == 0.0:
            # Vectorized common case; parameters have exactly the same layout
            # as the general yaw path below.
            qref = np.r_[reference.base_position, reference.base_quaternion_xyzw, reference.joint_positions]
            rotation = self._rotation(reference.base_quaternion_xyzw)
            vref = np.r_[rotation.T @ reference.base_velocity_world, np.zeros(self.model.nv - 3)]
            template = np.r_[qref, vref, reference.tool_position, tool_velocity,
                              np.full(4, cfg.ground_height + cfg.foot_radius), np.zeros(4), np.ones(4)]
            targets = np.tile(template, (cfg.horizon + 1, 1))
            targets[:, :3] += self.times[:, None] * reference.base_velocity_world
            offset = self.model.nq + self.model.nv
            targets[:, offset:offset+3] += self.times[:, None] * reference.tool_velocity
            targets[:, -4:] = contacts
            if schedule.mode != 'stand':
                phases = np.mod((schedule.time + self.times[:, None]) / schedule.period
                                - np.array([0., .5, .5, 0.]), 1.)
                phases = np.where(phases >= .5, 2*(phases-.5), 0.)
                u = np.where(phases <= .5, 2*phases, 2*(1-phases))
                heights = cfg.swing_height * u*u*(3-2*u)
                speeds = np.where(phases <= .5, 1., -1.) * cfg.swing_height*12*u*(1-u)/(schedule.period/2)
                targets[:, offset+6:offset+10] += np.where(contacts, 0., heights)
                targets[:, offset+10:offset+14] = np.where(contacts, 0., speeds)
            return np.concatenate((q, v, targets.ravel(), reference.tool_force)), contacts
        for k, elapsed in enumerate(self.times):
            half = 0.5 * reference.yaw_rate * elapsed
            x, y, z, w = reference.base_quaternion_xyzw
            # Left-multiply the reference quaternion by a world-z yaw increment.
            s, c = np.sin(half), np.cos(half)
            quat = np.array([c*x - s*y, c*y + s*x, c*z + s*w, c*w - s*z])
            qref = np.r_[reference.base_position + elapsed * reference.base_velocity_world,
                         quat, reference.joint_positions]
            rot = self._rotation(quat)
            vref = np.r_[rot.T @ reference.base_velocity_world,
                         rot.T @ [0.0, 0.0, reference.yaw_rate], np.zeros(self.model.na)]
            height = np.full(4, cfg.ground_height + cfg.foot_radius)
            vertical = np.zeros(4)
            for leg, phase in enumerate(schedule.swing_phase(elapsed)):
                if not contacts[k, leg]:
                    zref, dzref = swing_reference(phase, 0.5 * schedule.period, cfg.swing_height)
                    height[leg] += zref
                    vertical[leg] = dzref
            parts.append(np.r_[qref, vref, reference.tool_position + elapsed * reference.tool_velocity,
                               tool_velocity, height, vertical, contacts[k].astype(float)])
        parts.append(reference.tool_force)
        return np.concatenate(parts), contacts

    @staticmethod
    def _rotation(quaternion):
        x, y, z, w = quaternion
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

    def _initial_guess(self, q, v, contacts, tool_force):
        guess = np.zeros(self._decision_size)
        previous = self._previous if self.config.warm_start else None
        reuse_forces = (previous is not None
                        and np.array_equal(contacts, self._previous_contacts))
        nv = self.model.nv
        if previous is None:
            guess[self._state_indices[:, :nv]] = self.times[1:, None] * v
            guess[self._state_indices[:, nv:]] = v
        else:
            tangent = np.asarray(self._warm_difference(q, previous.q[:, :self.model.nq].T)).T
            guess[self._state_indices[:, :nv]] = tangent[1:]
            guess[self._state_indices[:, nv:]] = previous.v[1:, :nv]
            guess[self._acc_indices] = previous.a[:, :nv]
            guess[self._torque_indices] = previous.tau[:self.torque_nodes, :self.model.na]
        if reuse_forces:
            guess[self._feet_indices] = previous.forces[:, :12]
        else:
            forces = np.zeros((self.config.horizon, 4, 3))
            forces[:, :, 2] = (self.model.total_mass * 9.81 * contacts[:-1]
                                / contacts[:-1].sum(axis=1, keepdims=True))
            guess[self._feet_indices] = forces.reshape(-1, 12)
        return guess

    def solve(self, q, v, reference, schedule):
        q, v = np.asarray(q, dtype=float).reshape(25), np.asarray(v, dtype=float).reshape(24)
        q, v = q[:self.model.nq], v[:self.model.nv]
        reference = replace(reference, joint_positions=np.asarray(reference.joint_positions)[:self.model.na])
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
            raise ValueError("Non-finite measured state")
        if abs(np.linalg.norm(q[3:7]) - 1.0) > 1e-5:
            raise ValueError("Measured base quaternion is not normalized")
        for field, size in (("base_position", 3), ("base_quaternion_xyzw", 4),
                            ("base_velocity_world", 3), ("joint_positions", self.model.na),
                            ("tool_position", 3), ("tool_velocity", 3), ("tool_force", 3)):
            value = np.asarray(getattr(reference, field), dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid reference {field}")
        if not np.isfinite(reference.yaw_rate) or abs(np.linalg.norm(reference.base_quaternion_xyzw) - 1) > 1e-5:
            raise ValueError("Invalid orientation reference")
        if not self.config.tool_force_enabled and np.linalg.norm(reference.tool_force) > 1e-12:
            raise ValueError("A free-space tool cannot provide external contact force")
        parameters, contacts = self._parameters(q, v, reference, schedule)
        lbx, ubx, lbg, ubg = self.lbx.copy(), self.ubx.copy(), self.lbg.copy(), self.ubg.copy()
        for k, rows in enumerate(self._force_slices):
            lo, hi = np.full(12, -np.inf), np.full(12, np.inf)
            for leg in range(4):
                if contacts[k, leg]:
                    lo[3*leg:3*leg+3] = [-self.config.max_normal_force] * 2 + [0.0]
                    hi[3*leg:3*leg+3] = self.config.max_normal_force
                else:
                    lo[3*leg:3*leg+3] = hi[3*leg:3*leg+3] = 0.0
            lbx[rows], ubx[rows] = lo, hi
        guess = np.clip(self._initial_guess(q, v, contacts, reference.tool_force), lbx, ubx)
        arguments = dict(x0=guess, p=parameters, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        if (self.config.warm_start and self._dual is not None
                and np.array_equal(contacts, self._previous_contacts)):
            arguments.update(lam_x0=self._dual[0], lam_g0=self._dual[1])
        start = time.perf_counter()
        result = self.solver(**arguments)
        duration = time.perf_counter() - start
        decision = np.asarray(result["x"]).reshape(-1)
        values = np.asarray(result["g"]).reshape(-1)
        violation = float(max(0.0, np.max(lbx-decision), np.max(decision-ubx),
                              np.max(lbg-values), np.max(values-ubg)))
        stats = self.solver.stats()
        status = str(stats.get("return_status", "unknown"))
        if (not stats.get("success", False) or not np.all(np.isfinite(decision))
                or not np.all(np.isfinite(values)) or violation > max(1e-5, 10*self.config.tolerance)):
            self._previous = None
            self._dual = None
            self._previous_contacts = None
            raise RuntimeError(f"{self.config.solver}: {status}; max constraint violation={violation:.3g}")
        configurations, velocities, accelerations, forces, torques = (
            np.asarray(value).T for value in self.retraction_function(decision, parameters))
        solution = MPCSolution(configurations, velocities, accelerations, torques,
                               forces, self.dt.copy(), violation, duration, status,
                               int(stats.get("iter_count", 0)), stats.get("fatrop", {}))
        self._previous = solution
        self._dual = (np.asarray(result["lam_x"]).reshape(-1), np.asarray(result["lam_g"]).reshape(-1))
        self._previous_contacts = contacts.copy()
        return solution
