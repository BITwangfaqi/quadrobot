"""Paper's 22-DoF model: lock the last two arm joints, retain their inertia."""
from dataclasses import replace
import numpy as np


class ReducedModel:
    nq, nv, na = 23, 22, 16

    def __init__(self, full, angles=(0., 0.)):
        self.full = full
        self.angles = np.asarray(angles, dtype=float)
        self.total_mass = full.total_mass
        self.description = replace(full.description,
            joint_names=full.description.joint_names[:16],
            position_sensor_names=full.description.position_sensor_names[:16],
            lower_limits=full.description.lower_limits[:16], upper_limits=full.description.upper_limits[:16],
            effort_limits=full.description.effort_limits[:16], velocity_limits=full.description.velocity_limits[:16],
            locked_joints=dict(full.description.locked_joints, joint5=float(self.angles[0]), joint6=float(self.angles[1])))
        self.joint_names = self.description.joint_names
        self.foot_names, self.tool_name = full.foot_names, full.tool_name

    def expand_q(self, q):
        return np.r_[np.asarray(q).reshape(-1), self.angles]

    def expand_v(self, v):
        return np.r_[np.asarray(v).reshape(-1), 0., 0.]

    def integrate(self, q, v):
        return self.full.integrate(self.expand_q(q), self.expand_v(v))[:self.nq]

    def difference(self, q, target):
        return self.full.difference(self.expand_q(q), self.expand_q(target))[:self.nv]

    def neutral(self):
        return self.full.neutral()[:self.nq]

    def rnea(self, q, v, a):
        return self.full.rnea(self.expand_q(q), self.expand_v(v), self.expand_v(a))[:self.nv]

    def foot_jacobians(self, q):
        return self.full.foot_jacobians(self.expand_q(q))[:, :, :self.nv]

    def tool_jacobian(self, q):
        return self.full.tool_jacobian(self.expand_q(q))[:, :self.nv]

    def tool_position(self, q):
        return self.full.tool_position(self.expand_q(q))

    def casadi_functions(self):
        import casadi as ca
        functions = self.full.casadi_functions()
        q, q1 = ca.SX.sym('q', self.nq), ca.SX.sym('q1', self.nq)
        v, a = ca.SX.sym('v', self.nv), ca.SX.sym('a', self.nv)
        qf, q1f = ca.vertcat(q, self.angles), ca.vertcat(q1, self.angles)
        vf, af = ca.vertcat(v, 0., 0.), ca.vertcat(a, 0., 0.)
        expressions = {
            'integrate': ([q, v], functions['integrate'](qf, vf)[:self.nq]),
            'difference': ([q, q1], functions['difference'](qf, q1f)[:self.nv]),
            'rnea': ([q, v, a], functions['rnea'](qf, vf, af)[:self.nv]),
            'feet': ([q], functions['feet'](qf)),
            'foot_jacobians': ([q], functions['foot_jacobians'](qf)[:, :self.nv]),
            'tool': ([q], functions['tool'](qf)),
            'tool_jacobian': ([q], functions['tool_jacobian'](qf)[:, :self.nv]),
        }
        self.casadi_backend = self.full.casadi_backend + '-locked-wrist'
        return {name: ca.Function(name, ins, [out]) for name, (ins, out) in expressions.items()}
