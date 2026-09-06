"""Full floating-base leg/arm rigid-body dynamics in Webots encoder coordinates.

q = [world position (3), quaternion xyzw (4), native encoders (18)],
v = [body-local linear velocity (3), body-local angular velocity (3), dq (18)].
Generalized accelerations are derivatives of that body-local tangent velocity.
Forces and translational frame Jacobians are expressed in world axes.

Real Pinocchio (the PyPI ``pin`` distribution) accelerates numerical evaluation.
NumPy supplies an independent recursive Newton--Euler implementation, and an
exact CasADi expression backend works even without ``pinocchio.casadi``.
The spherical feet are point contacts at their centres (radius 0.03 m).
The tool frame is the gripper midpoint, 0.1358 m along gripper_base +z.
"""

import importlib
import os
from pathlib import Path
import numpy as np

try:
    from .robot_description import RobotDescription, axis_rotation
except ImportError:
    from robot_description import RobotDescription, axis_rotation


class _Algebra:
    def __init__(self, casadi=None):
        self.ca = casadi

    def matrix(self, value):
        return self.ca.DM(value) if self.ca else np.asarray(value, dtype=float)

    def zeros(self, rows, cols=1):
        return self.ca.SX.zeros(rows, cols) if self.ca else np.zeros((rows, cols))

    def eye(self, size):
        return self.matrix(np.eye(size))

    def rows(self, *values):
        return self.ca.vertcat(*values) if self.ca else np.vstack(values)

    def cols(self, *values):
        return self.ca.horzcat(*values) if self.ca else np.hstack(values)

    def skew(self, value):
        x, y, z = (value[i, 0] for i in range(3))
        if self.ca:
            return self.ca.vertcat(self.ca.horzcat(0, -z, y), self.ca.horzcat(z, 0, -x),
                                   self.ca.horzcat(-y, x, 0))
        return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])

    def quaternion_rotation(self, value):
        xyz, w = value[:3], value[3, 0]
        cross = self.skew(xyz)
        return self.eye(3) + 2. * w * cross + 2. * cross @ cross

    def rotation(self, axis, angle):
        sin, cos = (self.ca.sin, self.ca.cos) if self.ca else (np.sin, np.cos)
        cross = self.skew(self.matrix(axis).reshape((3, 1)))
        return self.eye(3) + sin(angle) * cross + (1. - cos(angle)) * cross @ cross

    def motion_cross(self, velocity):
        angular = self.skew(velocity[3:6])
        return self.rows(self.cols(angular, self.skew(velocity[:3])),
                         self.cols(self.zeros(3, 3), angular))


class WholeBodyModel:
    nq, nv, na = 25, 24, 18

    def __init__(self, description=None, use_pinocchio=True):
        self.description = description or RobotDescription.from_world()
        self.total_mass = self.description.total_mass
        self.joint_names = self.description.joint_names
        self.foot_names = self.description.foot_names
        self.tool_name = self.description.tool_name
        self.model = self.data = self.pin = None
        self._casadi_cache = None
        self._frame_ids = {}
        self.pinocchio_unavailable_reason = "disabled" if not use_pinocchio else "not installed"
        if use_pinocchio:
            try:
                pin = importlib.import_module("pinocchio")
            except ImportError:
                pin = None
            if pin is not None and hasattr(pin, "Model") and hasattr(pin, "JointModelFreeFlyer"):
                # Some environments mix incompatible NumPy/EigenPy ABIs even
                # though importing Pinocchio succeeds. Probe before building.
                try:
                    pin.Inertia(1., np.zeros(3), np.eye(3))
                except (TypeError, RuntimeError) as error:
                    self.pinocchio_unavailable_reason = f"NumPy/EigenPy interoperability: {error}"
                else:
                    self.pin = pin
                    self._build_pinocchio()
                    self.pinocchio_unavailable_reason = ""

    @property
    def backend(self):
        return "pinocchio" if self.pin is not None else "numpy"

    def _build_pinocchio(self):
        pin = self.pin
        model = pin.Model()
        base = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "free_flyer")
        joints, placements = [base], [pin.SE3.Identity()]
        for index, body in enumerate(self.description.bodies):
            if index:
                parent_joint, parent_pose = joints[body.parent], placements[body.parent]
                if body.joint_index >= 0:
                    reference = axis_rotation(body.axis, -body.reference_position)
                    joint_pose = parent_pose * pin.SE3(reference, body.anchor)
                    joint = model.addJoint(parent_joint, pin.JointModelRevoluteUnaligned(body.axis),
                                           joint_pose, self.joint_names[body.joint_index])
                    placement = pin.SE3(body.rotation, body.translation - body.anchor)
                    expected = body.joint_index + 7
                    if model.joints[joint].idx_q != expected:
                        raise ValueError("Parsed tree does not match required actuator ordering")
                else:
                    joint = parent_joint
                    placement = parent_pose * pin.SE3(body.rotation, body.translation)
                joints.append(joint)
                placements.append(placement)
            if body.mass:
                model.appendBodyToJoint(joints[index], pin.Inertia(body.mass, body.com, body.inertia), placements[index])
        for name, frame in self.description.frames.items():
            placement = placements[frame.body] * pin.SE3(frame.rotation, frame.translation)
            self._frame_ids[name] = model.addFrame(pin.Frame(name, joints[frame.body], placement, pin.FrameType.OP_FRAME))
        model.lowerPositionLimit[7:] = self.description.lower_limits
        model.upperPositionLimit[7:] = self.description.upper_limits
        model.effortLimit[6:] = self.description.effort_limits
        model.velocityLimit[6:] = self.description.velocity_limits
        model.gravity.linear = np.array([0., 0., -9.81])
        self.model, self.data = model, model.createData()
        if model.nq != self.nq or model.nv != self.nv:
            raise ValueError("Expected a 6-DoF base and 18 actuated joints")

    def neutral(self):
        q = np.zeros(self.nq)
        q[6] = 1.
        return q

    def _tree(self, q, v=None, a=None, algebra=None, gravity=False):
        """World placements/Jacobians and local spatial RNEA forward sweep."""
        alg = algebra or _Algebra()
        q = q.reshape((self.nq, 1))
        v = alg.zeros(self.nv) if v is None else v.reshape((self.nv, 1))
        a = alg.zeros(self.nv) if a is None else a.reshape((self.nv, 1))
        rotations, positions, jacobians, velocities, accelerations, transforms = [], [], [], [], [], []
        base_rotation = alg.quaternion_rotation(q[3:7])
        for index, body in enumerate(self.description.bodies):
            if index == 0:
                rotations.append(base_rotation)
                positions.append(q[:3])
                jacobians.append(alg.rows(alg.cols(base_rotation, alg.zeros(3, 21)),
                                          alg.cols(alg.zeros(3, 3), base_rotation, alg.zeros(3, 18))))
                velocities.append(v[:6])
                gravity_accel = base_rotation.T @ alg.matrix([0., 0., 9.81]).reshape((3, 1)) if gravity else alg.zeros(3)
                accelerations.append(a[:6] + alg.rows(gravity_accel, alg.zeros(3)))
                transforms.append(alg.eye(6))
                continue
            zero_rotation = alg.matrix(body.rotation)
            offset = alg.matrix(body.translation - body.anchor).reshape((3, 1))
            joint_rotation = alg.rotation(body.axis, q[7 + body.joint_index, 0] - body.reference_position) if body.joint_index >= 0 else alg.eye(3)
            local_rotation = joint_rotation @ zero_rotation
            local_position = alg.matrix(body.anchor).reshape((3, 1)) + joint_rotation @ offset
            parent = body.parent
            rotation = rotations[parent] @ local_rotation
            world_delta = rotations[parent] @ local_position
            position = positions[parent] + world_delta
            jacobian = alg.rows(jacobians[parent][:3, :] - alg.skew(world_delta) @ jacobians[parent][3:, :],
                                jacobians[parent][3:, :])
            transform = alg.rows(alg.cols(local_rotation.T, -local_rotation.T @ alg.skew(local_position)),
                                 alg.cols(alg.zeros(3, 3), local_rotation.T))
            joint_velocity, joint_acceleration = alg.zeros(6), alg.zeros(6)
            if body.joint_index >= 0:
                subspace = alg.matrix(body.motion_subspace).reshape((6, 1))
                axis_world = rotations[parent] @ alg.matrix(body.axis).reshape((3, 1))
                linear_world = rotations[parent] @ alg.skew(alg.matrix(body.axis).reshape((3, 1))) @ joint_rotation @ offset
                jacobian[:, 6 + body.joint_index] += alg.rows(linear_world, axis_world)[:, 0]
                joint_velocity = subspace * v[6 + body.joint_index, 0]
                joint_acceleration = subspace * a[6 + body.joint_index, 0]
            velocity = transform @ velocities[parent] + joint_velocity
            acceleration = transform @ accelerations[parent] + joint_acceleration + alg.motion_cross(velocity) @ joint_velocity
            rotations.append(rotation)
            positions.append(position)
            jacobians.append(jacobian)
            velocities.append(velocity)
            accelerations.append(acceleration)
            transforms.append(transform)
        return rotations, positions, jacobians, velocities, accelerations, transforms

    def _frame(self, tree, name, algebra=None):
        alg = algebra or _Algebra()
        rotations, positions, jacobians, velocities, accelerations, _ = tree
        frame = self.description.frames[name]
        index = frame.body
        rotation = rotations[index]
        offset = rotation @ alg.matrix(frame.translation).reshape((3, 1))
        position = positions[index] + offset
        jacobian = alg.rows(jacobians[index][:3, :] - alg.skew(offset) @ jacobians[index][3:, :], jacobians[index][3:, :])
        omega = rotation @ velocities[index][3:]
        alpha = rotation @ accelerations[index][3:]
        acceleration = rotation @ (accelerations[index][:3] + alg.skew(velocities[index][3:]) @ velocities[index][:3])
        acceleration += alg.skew(alpha) @ offset + alg.skew(omega) @ alg.skew(omega) @ offset
        return position, jacobian, acceleration, rotation @ alg.matrix(frame.rotation)

    def _rnea_tree(self, tree, algebra=None):
        alg = algebra or _Algebra()
        _, _, _, velocities, accelerations, transforms = tree
        forces = []
        for body, velocity, acceleration in zip(self.description.bodies, velocities, accelerations):
            inertia = alg.matrix(body.spatial_inertia)
            forces.append(inertia @ acceleration - alg.motion_cross(velocity).T @ inertia @ velocity)
        torque = alg.zeros(self.nv)
        for index in range(len(forces) - 1, 0, -1):
            body = self.description.bodies[index]
            if body.joint_index >= 0:
                subspace = alg.matrix(body.motion_subspace).reshape((6, 1))
                torque[6 + body.joint_index, 0] = (subspace.T @ forces[index])[0, 0]
            forces[body.parent] = forces[body.parent] + transforms[index].T @ forces[index]
        torque[:6] = forces[0]
        return torque

    def rnea(self, q, v, a):
        if self.pin:
            return np.asarray(self.pin.rnea(self.model, self.data, np.asarray(q), np.asarray(v), np.asarray(a))).copy()
        if self._casadi_cache is not None:
            return np.asarray(self._casadi_cache["rnea"](q, v, a)).reshape(self.nv)
        return self._rnea_tree(self._tree(np.asarray(q), np.asarray(v), np.asarray(a), gravity=True)).reshape(self.nv)

    def bias(self, q, v):
        return self.rnea(q, v, np.zeros(self.nv))

    def mass_matrix(self, q):
        if self.pin:
            return np.asarray(self.pin.crba(self.model, self.data, np.asarray(q))).copy()
        if self._casadi_cache is not None:
            return np.asarray(self._casadi_cache["mass_matrix"](q))
        tree = self._tree(np.asarray(q))
        mass = np.zeros((self.nv, self.nv))
        for body, rotation, jacobian in zip(self.description.bodies, tree[0], tree[2]):
            local = np.block([[rotation.T, np.zeros((3, 3))], [np.zeros((3, 3)), rotation.T]]) @ jacobian
            mass += local.T @ body.spatial_inertia @ local
        return mass

    def frame_position(self, q, name):
        if self.pin:
            self.pin.framesForwardKinematics(self.model, self.data, np.asarray(q))
            return self.data.oMf[self._frame_ids[name]].translation.copy()
        return self._frame(self._tree(np.asarray(q)), name)[0].reshape(3)

    def frame_jacobian(self, q, name):
        if self.pin:
            return np.asarray(self.pin.computeFrameJacobian(self.model, self.data, np.asarray(q), self._frame_ids[name],
                                                           self.pin.LOCAL_WORLD_ALIGNED)).copy()
        return self._frame(self._tree(np.asarray(q)), name)[1]

    def foot_positions(self, q):
        if self.pin:
            self.pin.framesForwardKinematics(self.model, self.data, np.asarray(q))
            return np.array([self.data.oMf[self._frame_ids[name]].translation for name in self.foot_names])
        if self._casadi_cache is not None:
            return np.asarray(self._casadi_cache["feet"](q)).T
        tree = self._tree(np.asarray(q))
        return np.array([self._frame(tree, name)[0].reshape(3) for name in self.foot_names])

    def foot_jacobians(self, q):
        if self.pin:
            self.pin.computeJointJacobians(self.model, self.data, np.asarray(q))
            self.pin.updateFramePlacements(self.model, self.data)
            return np.array([self.pin.getFrameJacobian(self.model, self.data, self._frame_ids[name],
                                                       self.pin.LOCAL_WORLD_ALIGNED)[:3] for name in self.foot_names])
        if self._casadi_cache is not None:
            return np.asarray(self._casadi_cache["foot_jacobians"](q)).reshape(4, 3, self.nv)
        tree = self._tree(np.asarray(q))
        return np.array([self._frame(tree, name)[1][:3] for name in self.foot_names])

    def foot_bias_accelerations(self, q, v):
        if self.pin:
            self.pin.forwardKinematics(self.model, self.data, np.asarray(q), np.asarray(v), np.zeros(self.nv))
            self.pin.updateFramePlacements(self.model, self.data)
            return np.array([self.pin.getFrameClassicalAcceleration(self.model, self.data, self._frame_ids[name],
                                                                   self.pin.LOCAL_WORLD_ALIGNED).linear for name in self.foot_names])
        if self._casadi_cache is not None:
            return np.asarray(self._casadi_cache["foot_bias"](q, v)).T
        tree = self._tree(np.asarray(q), np.asarray(v))
        return np.array([self._frame(tree, name)[2].reshape(3) for name in self.foot_names])

    def tool_position(self, q):
        if self.pin is None and self._casadi_cache is not None:
            return np.asarray(self._casadi_cache["tool"](q)).reshape(3)
        return self.frame_position(q, self.tool_name)

    def tool_jacobian(self, q):
        if self.pin is None and self._casadi_cache is not None:
            return np.asarray(self._casadi_cache["tool_jacobian"](q))
        return self.frame_jacobian(q, self.tool_name)[:3]

    def tool_rotation(self, q):
        if self.pin:
            self.pin.framesForwardKinematics(self.model, self.data, np.asarray(q))
            return self.data.oMf[self._frame_ids[self.tool_name]].rotation.copy()
        if self._casadi_cache is not None:
            return np.asarray(self._casadi_cache["tool_rotation"](q))
        return self._frame(self._tree(np.asarray(q)), self.tool_name)[3]

    @staticmethod
    def _manifold(q, tangent=None, target=None, casadi=None):
        """SE(3) x R^18 integration/logarithm, matching Pinocchio free flyer."""
        alg = _Algebra(casadi)
        ca = casadi
        sin, cos, sqrt = (ca.sin, ca.cos, ca.sqrt) if ca else (np.sin, np.cos, np.sqrt)
        dot = ca.dot if ca else lambda x, y: float((x.T @ y)[0, 0])
        choose = ca.if_else if ca else lambda condition, yes, no: yes if condition else no
        q = q.reshape((25, 1))
        rotation = alg.quaternion_rotation(q[3:7])

        def product(first, second):
            return alg.rows(first[3, 0] * second[:3] + second[3, 0] * first[:3] + alg.skew(first[:3]) @ second[:3],
                            first[3, 0] * second[3, 0] - dot(first[:3], second[:3]))

        if tangent is not None:
            tangent = tangent.reshape((24, 1))
            angular = tangent[3:6]
            squared = dot(angular, angular)
            theta = sqrt(squared + 1e-24)
            k = alg.skew(angular)
            a = choose(squared < 1e-8, .5 - squared / 24. + squared * squared / 720., (1. - cos(theta)) / theta**2)
            b = choose(squared < 1e-8, 1. / 6. - squared / 120. + squared * squared / 5040., (theta - sin(theta)) / theta**3)
            factor = choose(squared < 1e-8, .5 - squared / 48. + squared * squared / 3840., sin(theta / 2.) / theta)
            increment = alg.rows(factor * angular, cos(theta / 2.))
            return alg.rows(q[:3] + rotation @ (alg.eye(3) + a * k + b * k @ k) @ tangent[:3],
                            product(q[3:7], increment), q[7:] + tangent[6:])
        target = target.reshape((25, 1))
        conjugate = alg.rows(-q[3:6], q[6:7])
        relative = product(conjugate, target[3:7])
        relative = choose(relative[3, 0] < 0, -relative, relative)
        squared = dot(relative[:3], relative[:3])
        norm = sqrt(squared + 1e-24)
        atan2 = ca.atan2 if ca else np.arctan2
        theta = 2. * atan2(norm, relative[3, 0])
        factor = choose(squared < 1e-8, 2. + squared / 3. + 3. * squared * squared / 20., theta / norm)
        angular = factor * relative[:3]
        theta2 = dot(angular, angular)
        theta_safe = sqrt(theta2 + 1e-24)
        c = choose(theta2 < 1e-8, 1. / 12. + theta2 / 720. + theta2**2 / 30240.,
                   (1. - .5 * theta_safe * cos(.5 * theta_safe) / sin(.5 * theta_safe)) / theta_safe**2)
        k = alg.skew(angular)
        linear = (alg.eye(3) - .5 * k + c * k @ k) @ rotation.T @ (target[:3] - q[:3])
        return alg.rows(linear, angular, target[7:] - q[7:])

    def integrate(self, q, tangent):
        if self.pin:
            return np.asarray(self.pin.integrate(self.model, np.asarray(q), np.asarray(tangent)))
        if self._casadi_cache is not None:
            return np.asarray(self._casadi_cache["integrate"](q, tangent)).reshape(self.nq)
        return self._manifold(np.asarray(q), tangent=np.asarray(tangent)).reshape(self.nq)

    def difference(self, q0, q1):
        if self.pin:
            return np.asarray(self.pin.difference(self.model, np.asarray(q0), np.asarray(q1)))
        if self._casadi_cache is not None:
            return np.asarray(self._casadi_cache["difference"](q0, q1)).reshape(self.nv)
        return self._manifold(np.asarray(q0), target=np.asarray(q1)).reshape(self.nv)

    def casadi_functions(self, force_tree=False):
        """Return differentiable Functions with standard positional arguments.

        integrate(q,dq):25; difference(q0,q1):24; rnea(q,v,a):24;
        feet(q):3x4; foot_jacobians(q):12x24 (FR/FL/BR/BL blocks);
        foot_bias(q,v):3x4; tool(q):3; tool_jacobian(q):3x24;
        tool_rotation(q):3x3; tool_jacobian6(q):6x24; mass_matrix(q):24x24.
        Native pinocchio.casadi is used when available. ``force_tree`` selects
        the independent exact symbolic recursive backend for verification.
        """
        if self._casadi_cache is not None and not force_tree:
            return self._casadi_cache
        try:
            import casadi as ca
        except ImportError as error:
            raise ImportError("Whole-body MPC requires CasADi; install the controller requirements.txt") from error
        if self.pin is None and not force_tree and not os.environ.get('INV_DYN_DISABLE_NATIVE_EXPORT'):
            native = self._exported_native_functions(ca)
            if native is not None:
                self._casadi_cache = native
                self.casadi_backend = 'pinocchio.casadi-export'
                return native
        q, q1 = ca.SX.sym("q", self.nq), ca.SX.sym("q1", self.nq)
        v, a, dq = (ca.SX.sym(name, self.nv) for name in ("v", "a", "dq"))
        alg = _Algebra(ca)
        cpin = None
        if self.pin is not None and not force_tree:
            try:
                cpin = importlib.import_module("pinocchio.casadi")
            except ImportError:
                pass
        if cpin is not None:
            cmodel = cpin.Model(self.model)
            cdata = cmodel.createData()
            torque = cpin.rnea(cmodel, cdata, q, v, a)
            mass = cpin.crba(cmodel, cdata, q)
            integrated, difference = cpin.integrate(cmodel, q, dq), cpin.difference(cmodel, q, q1)
            cpin.computeJointJacobians(cmodel, cdata, q)
            cpin.forwardKinematics(cmodel, cdata, q, v, ca.SX.zeros(self.nv))
            cpin.updateFramePlacements(cmodel, cdata)
            def frame(name):
                index = self._frame_ids[name]
                return (cdata.oMf[index].translation,
                        cpin.getFrameJacobian(cmodel, cdata, index, self.pin.LOCAL_WORLD_ALIGNED),
                        cpin.getFrameClassicalAcceleration(cmodel, cdata, index, self.pin.LOCAL_WORLD_ALIGNED).linear,
                        cdata.oMf[index].rotation)
            feet = [frame(name) for name in self.foot_names]
            tool = frame(self.tool_name)
            backend_name = "pinocchio.casadi"
        else:
            dynamic_tree = self._tree(q, v, a, alg, gravity=True)
            torque = self._rnea_tree(dynamic_tree, alg)
            mass = ca.jacobian(torque, a)
            tree = self._tree(q, v, algebra=alg)
            feet = [self._frame(tree, name, alg) for name in self.foot_names]
            tool = self._frame(tree, self.tool_name, alg)
            integrated = self._manifold(q, tangent=dq, casadi=ca)
            difference = self._manifold(q, target=q1, casadi=ca)
            backend_name = "casadi-recursive-newton-euler"
        expressions = {
            "integrate": ([q, dq], integrated), "difference": ([q, q1], difference),
            "rnea": ([q, v, a], torque), "mass_matrix": ([q], mass),
            "feet": ([q], ca.horzcat(*(frame[0] for frame in feet))),
            "foot_jacobians": ([q], ca.vertcat(*(frame[1][:3, :] for frame in feet))),
            "foot_bias": ([q, v], ca.horzcat(*(frame[2] for frame in feet))),
            "tool": ([q], tool[0]), "tool_jacobian": ([q], tool[1][:3, :]),
            "tool_jacobian6": ([q], tool[1]), "tool_rotation": ([q], tool[3]),
        }
        functions = {name: ca.Function(name, arguments, [expression])
                     for name, (arguments, expression) in expressions.items()}
        if not force_tree:
            self._casadi_cache = functions
            self.casadi_backend = backend_name
        return functions

    def _exported_native_functions(self, ca):
        """Use an isolated working Pinocchio installation when EigenPy ABI differs."""
        import fcntl
        import hashlib
        import json
        import subprocess
        python = os.environ.get('INV_DYN_PIN_PYTHON', '/usr/bin/python3')
        pin_path = os.environ.get('INV_DYN_PIN_PATH', '/opt/openrobots/lib/python3.10/site-packages')
        if not Path(python).exists() or not Path(pin_path, 'pinocchio').exists():
            return None
        root = Path(__file__).resolve().parent
        digest = hashlib.sha256(self.description.world_path.read_bytes())
        for name in ('rigid_body_model.py', 'robot_description.py', 'native_model_export.py'):
            digest.update((root / name).read_bytes())
        digest.update((python + pin_path + ca.__version__).encode())
        directory = root / '.cache' / ('native-' + digest.hexdigest()[:20])
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'export.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (directory / 'manifest.json').exists():
                env = dict(os.environ, PYTHONPATH=pin_path, OPENBLAS_NUM_THREADS='1',
                           INV_DYN_DISABLE_NATIVE_EXPORT='1')
                result = subprocess.run([python, str(root / 'native_model_export.py'),
                                         str(self.description.world_path), str(directory)],
                                        env=env, capture_output=True, text=True, timeout=60)
                if result.returncode:
                    print('[inv_dyn_mpc] Native expression export unavailable; using exact tree backend. '
                          + result.stderr[-400:], flush=True)
                    return None
            manifest = json.loads((directory / 'manifest.json').read_text())
            functions = {name: ca.Function.load(str(directory / (name + '.casadi')))
                         for name in manifest['functions']}
        # Guard against a stale or mismatched physical model at this boundary.
        rng = np.random.default_rng(71)
        for _ in range(2):
            q = self.neutral()
            q[7:] = rng.normal(scale=.1, size=self.na)
            v, a = rng.normal(size=(2, self.nv))
            expected = self.rnea(q, v, a)
            if np.max(np.abs(np.asarray(functions['rnea'](q, v, a)).ravel() - expected)) > 1e-8:
                raise RuntimeError('Exported native dynamics disagree with the current WBT model')
        return functions


if __name__ == "__main__":
    robot = WholeBodyModel()
    q = robot.neutral()
    print(f"Backend: {robot.backend}; nq={robot.nq}, nv={robot.nv}, mass={robot.total_mass:.6f} kg")
    print("Feet at raw zero encoders, relative to base:\n", robot.foot_positions(q))
    print("Tool at raw zero encoders, relative to base:", robot.tool_position(q))
    print("Gravity wrench:", robot.bias(q, np.zeros(robot.nv))[:6])
