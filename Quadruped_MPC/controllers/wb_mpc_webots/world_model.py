"""Read this project's explicit WBT robot tree into native Pinocchio.

No code or model data is imported from another controller. This reader supports
the explicit nodes used by quadruped_arm.wbt, not arbitrary PROTO expansion.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import re

import numpy as np
import pinocchio as pin

from gait_adapter import GaitSequence

DEFAULT_WORLD = Path(__file__).resolve().parents[2] / 'worlds/quadruped_arm.wbt'


@dataclass
class Node:
    kind: str
    fields: dict = field(default_factory=dict)


class Reader:
    """Token reader preserving strings/comments and resolving local DEF/USE."""
    def __init__(self, text):
        pattern = r'"(?:\\.|[^"\\])*"|#[^\n]*|[{}\[\],]|[^\s{}\[\],]+'
        self.tokens = [t for t in re.findall(pattern, text) if not t.startswith('#') and t != ',']
        self.i = 0
        self.defs = {}

    def pop(self):
        t = self.tokens[self.i]
        self.i += 1
        return t

    def value(self):
        t = self.pop()
        if t == 'DEF':
            name = self.pop()
            value = self.value()
            self.defs[name] = value
            return value
        if t == 'USE':
            return self.defs[self.pop()]
        if t == '[':
            result = []
            while self.tokens[self.i] != ']':
                value = self.value()
                result.extend(value if isinstance(value, list) else [value])
            self.pop()
            return result
        if t.startswith('"'):
            return json.loads(t)
        if t in ('TRUE', 'FALSE', 'NULL'):
            return {'TRUE': True, 'FALSE': False, 'NULL': None}[t]
        try:
            values = [float(t)]
            while self.i < len(self.tokens):
                try:
                    values.append(float(self.tokens[self.i]))
                    self.i += 1
                except ValueError:
                    break
            return values[0] if len(values) == 1 else values
        except ValueError:
            pass
        if self.pop() != '{':
            raise ValueError(f'Expected explicit node after {t}; PROTO/IS is not supported')
        result = Node(t)
        while self.tokens[self.i] != '}':
            key = self.pop()
            result.fields[key] = self.value()
        self.pop()
        return result

    def roots(self):
        nodes = []
        while self.i < len(self.tokens):
            if self.tokens[self.i] == 'EXTERNPROTO':
                self.pop()
                self.value()
            else:
                nodes.append(self.value())
        return nodes


def pose(node):
    axis_angle = np.asarray(node.fields.get('rotation', [0, 0, 1, 0]), float)
    axis = axis_angle[:3]
    rotation = pin.exp3(axis / np.linalg.norm(axis) * axis_angle[3]) if axis_angle[3] else np.eye(3)
    return pin.SE3(rotation, np.asarray(node.fields.get('translation', [0, 0, 0]), float))


class WebotsModel:
    def __init__(self, world=DEFAULT_WORLD, arm_joints=4):
        if arm_joints not in (0, 4, 6):
            raise ValueError('arm_joints must be 0, 4 or 6')
        self.world = Path(world).resolve()
        roots = Reader(self.world.read_text()).roots()
        robots = [n for n in roots if isinstance(n, Node) and n.kind == 'Robot' and n.fields.get('name') == 'Bigdog']
        if len(robots) != 1:
            raise ValueError('Expected one explicit Robot named Bigdog in world')
        self.root = robots[0]
        self.full_model = pin.Model()
        self.full_model.name = 'quadruped_arm_from_wbt'
        root_id = self.full_model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), 'root_joint')
        self.full_model.addJointFrame(root_id)
        self.full_model.addFrame(pin.Frame('base_link', root_id, pin.SE3.Identity(), pin.BODY))
        self.devices = {}
        self.frames = {}
        self.body_records = []
        self.joint_records = []
        self._body(self.root, root_id, pin.SE3.Identity(), root=True)
        self.full_q0 = pin.neutral(self.full_model)
        self.full_q0[:3] = self.root.fields.get('translation', [0, 0, 0.329])
        self.full_q0[3:7] = pin.Quaternion(pose(self.root).rotation).coeffs()
        # WBT encoder zero already describes bent legs. No B2 offsets.
        self.locked = {f'joint{i}': 0.0 for i in range(arm_joints + 1, 7)}
        self.locked.update({'joint7': 0.0, 'joint8': 0.0})
        ids = [self.full_model.getJointId(name) for name in self.locked]
        self.model = pin.buildReducedModel(self.full_model, ids, self.full_q0)
        self.data = self.model.createData()
        self.names = list(self.model.names)[2:]
        self.q0 = pin.neutral(self.model)
        self.q0[:7] = self.full_q0[:7]
        self.nq, self.nv = self.model.nq, self.model.nv
        self.nj, self.nf = self.nv - 6, 12 + (3 if arm_joints else 0)
        self.arm_joints = arm_joints
        self.arm_ee_frame = self.model.getFrameId('gripperCenter') if arm_joints else None
        self.front_force_ratio = 0.5
        self.joint_pos_min = self.model.lowerPositionLimit[7:].copy()
        self.joint_pos_max = self.model.upperPositionLimit[7:].copy()
        self.joint_vel_max = self.model.velocityLimit[6:].copy()
        self.joint_torque_max = self.model.effortLimit[6:].copy()
        pin.computeAllTerms(self.model, self.data, self.q0, np.zeros(self.nv))
        self.mass = pin.computeTotalMass(self.model)
        self.set_gait_sequence('stand', 0.8)

    def _frame(self, name, parent, placement):
        self.frames[name] = self.full_model.addFrame(pin.Frame(name, parent, placement, pin.OP_FRAME))

    def _body(self, node, parent, placement, root=False):
        physics = node.fields.get('physics')
        if isinstance(physics, Node):
            f = physics.fields
            if not all(k in f for k in ('mass', 'centerOfMass', 'inertiaMatrix')):
                raise ValueError('Every dynamic body must have explicit mass, COM and inertia')
            xx, yy, zz, xy, xz, yz = f['inertiaMatrix']
            inertia = pin.Inertia(f['mass'], np.array(f['centerOfMass']), np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]))
            self.full_model.appendBodyToJoint(parent, inertia, placement)
            self.body_records.append((parent, placement, inertia))
        name = node.fields.get('name')
        if name:
            self._frame('wbt/' + name, parent, placement)
        foot_alias = {'FR_4': 'FR_foot', 'FL_4': 'FL_foot', 'BR_4': 'RR_foot', 'BL_4': 'RL_foot'}
        if name in foot_alias:
            self._frame(foot_alias[name], parent, placement)
        if name == 'gripper_base':
            self._frame('gripperCenter', parent, placement * pin.SE3(np.eye(3), np.array([0., 0., 0.1358])))
        children = node.fields.get('children', [])
        if root:
            order = {'FL_hip_motor': 0, 'FR_hip_motor': 1, 'BL_hip_motor': 2, 'BR_hip_motor': 3}
            def rank(child):
                if child.kind == 'HingeJoint':
                    motor = next(d for d in child.fields['device'] if d.kind == 'RotationalMotor')
                    return order[motor.fields['name']]
                return 4
            children = sorted(children, key=rank)
        for child in children:
            if child.kind in ('HingeJoint', 'SliderJoint'):
                self._joint(child, parent, placement)
            elif child.kind in ('Solid', 'Transform', 'Pose', 'Group', 'TouchSensor'):
                self._body(child, parent, placement * pose(child))

    def _joint(self, node, parent, placement):
        params = node.fields['jointParameters'].fields
        if params.get('position', 0) != 0:
            raise ValueError('Nonzero WBT joint position requires a reference-pose conversion')
        axis = np.array(params.get('axis', [1, 0, 0]), float)
        axis /= np.linalg.norm(axis)
        anchor = np.array(params.get('anchor', [0, 0, 0]), float)
        device = node.fields['device']
        motor = next(d for d in device if d.kind in ('RotationalMotor', 'LinearMotor')).fields
        sensor = next(d for d in device if d.kind == 'PositionSensor').fields['name']
        name = motor['name']
        self.devices[name] = dict(motor, sensor=sensor)
        joint_pose = placement * pin.SE3(np.eye(3), anchor)
        jm = pin.JointModelRevoluteUnaligned(axis) if node.kind == 'HingeJoint' else pin.JointModelPrismaticUnaligned(axis)
        jid = self.full_model.addJoint(parent, jm, joint_pose, name)
        self.full_model.addJointFrame(jid)
        self.joint_records.append((jid, parent, joint_pose, axis, node.kind))
        joint = self.full_model.joints[jid]
        lo, hi = motor.get('minPosition', 0.), motor.get('maxPosition', 0.)
        if lo == hi == 0:
            lo, hi = -np.inf, np.inf
        self.full_model.lowerPositionLimit[joint.idx_q] = lo
        self.full_model.upperPositionLimit[joint.idx_q] = hi
        self.full_model.velocityLimit[joint.idx_v] = motor.get('maxVelocity', 10.)
        self.full_model.effortLimit[joint.idx_v] = motor.get('maxTorque', motor.get('maxForce', 10.))
        endpoint = node.fields['endPoint']
        child_pose = pin.SE3(np.eye(3), -anchor) * pose(endpoint)
        self._body(endpoint, jid, child_pose)

    def set_gait_sequence(self, gait_type, gait_period):
        self.gait_sequence = GaitSequence(gait_type, gait_period)
        self.foot_frames = [self.model.getFrameId(name) for name in self.gait_sequence.feet]

    def set_nominal(self, q):
        self.q0 = np.asarray(q, float).copy()
        pin.computeAllTerms(self.model, self.data, self.q0, np.zeros(self.nv))

    def expand_q(self, q):
        full = self.full_q0.copy()
        full[:7] = q[:7]
        for name in self.names:
            full[self.full_model.joints[self.full_model.getJointId(name)].idx_q] = q[self.model.joints[self.model.getJointId(name)].idx_q]
        return full

    def signature(self):
        import hashlib
        payload = self.world.read_bytes() + json.dumps({
            'q0': self.q0.tolist(), 'arm_joints': self.arm_joints,
            'effort': self.joint_torque_max.tolist(), 'velocity': self.joint_vel_max.tolist(),
        }, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()
