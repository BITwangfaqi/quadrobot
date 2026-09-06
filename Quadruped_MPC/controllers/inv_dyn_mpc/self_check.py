"""Offline model checks and optional MPC/constrained-plant standing rollout.

Run ``python self_check.py`` after installing requirements.  Add
``--solve --solver ipopt --steps 5`` to solve a small receding horizon problem
and apply its torques to an independently integrated constrained rigid-body
plant.  This exercises physical dynamics, not the optimizer's own rollout.
It does not replace Webots tests of contact, sensing, and motor behavior.
"""

import argparse
from dataclasses import replace
import sys
import time

import numpy as np

try:
    from .config import MPCConfig, MPCReference
    from .gait import ContactSchedule
    from .rigid_body_model import WholeBodyModel
    from .robot_description import RobotDescription
except ImportError:
    from config import MPCConfig, MPCReference
    from gait import ContactSchedule
    from rigid_body_model import WholeBodyModel
    from robot_description import RobotDescription


def _close(actual, expected, tolerance, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: shape {actual.shape} != {expected.shape}")
    if not np.all(np.isfinite(actual)):
        raise AssertionError(f"{label}: nonfinite values")
    error = float(np.max(np.abs(actual - expected)))
    if error > tolerance:
        raise AssertionError(f"{label}: maximum error {error:.3e} > {tolerance:.3e}")
    return error


def check_model():
    """Mandatory dynamics, coordinate, force-sign and derivative invariants."""
    description = RobotDescription.from_world()
    direct = WholeBodyModel(description, use_pinocchio=False)
    model = WholeBodyModel(description)
    q0 = direct.neutral()
    q0[:3] = description.initial_position
    _close(np.array([description.total_mass]), np.array([18.134388]), 1e-9, "WBT total mass")
    if len(description.joint_names) != 18 or len(set(description.joint_names)) != 18:
        raise AssertionError("Expected 18 uniquely named actuators")
    if description.joint_names[:3] != ("FR_hip_motor", "FR_leg_motor", "FR_foot_motor"):
        raise AssertionError("Leg actuator order must be FR, FL, BR, BL")
    if description.joint_names[12:] != tuple(f"joint{i}" for i in range(1, 7)):
        raise AssertionError("Arm actuators must follow the 12 leg joints")
    expected_sensors = tuple(f"{leg}_{joint}_position_sensor" for leg in ("FR", "FL", "BR", "BL")
                             for joint in ("hip", "leg", "foot")) + tuple(f"joint{i}_sensor" for i in range(1, 7))
    if description.position_sensor_names != expected_sensors:
        raise AssertionError("Encoder and actuator orders differ")
    _close(description.lower_limits[12:], np.array([-2.618, 0., -2.967, -1.745, -1.22, -2.0944]),
           1e-12, "Arm lower limits, including Webots omitted-bound defaults")
    _close(description.upper_limits[12:], np.array([2.618, 3.14, 0., 1.745, 1.22, 2.0944]),
           1e-12, "Arm upper limits, including Webots omitted-bound defaults")
    _close(description.effort_limits, np.full(18, 100.), 1e-12, "Motor effort limits")
    _close(description.velocity_limits, np.r_[np.full(12, 50.), np.full(5, 5.), 3.], 1e-12,
           "Motor velocity limits")
    if description.locked_joints != {"joint7": 0., "joint8": 0.}:
        raise AssertionError("This reduced gripper model requires sliders held at zero")
    expected_feet = np.array([[.191758184447, -.145999666873, -.296266054427],
                              [.19198, .146, -.29607], [-.2168, -.146, -.2962],
                              [-.2168, .146, -.29621]])
    _close(direct.foot_positions(q0) - q0[:3], expected_feet, 1e-8, "Native-encoder zero foot FK")
    _close(direct.tool_position(q0) - q0[:3], np.array([.261426439, -1.36568057e-6, .284020593]),
           1e-8, "Native-encoder zero tool FK")
    print(f"PASS world description: 18 joints, {description.total_mass:.6f} kg, exact raw-zero frames")

    rng = np.random.default_rng(731)
    perturbation = rng.normal(0., .2, 24)
    perturbation[18:24] = [0.1, .5, -.7, .1, .15, -.1]
    q = direct.integrate(q0, perturbation)
    v, a = rng.normal(0., .4, 24), rng.normal(0., .7, 24)
    mass = direct.mass_matrix(q)
    _close(mass, mass.T, 1e-11, "Mass symmetry")
    minimum_eigenvalue = float(np.linalg.eigvalsh(mass).min())
    if minimum_eigenvalue <= 1e-8:
        raise AssertionError(f"Mass matrix must be positive definite: min eigenvalue={minimum_eigenvalue}")
    _close(direct.rnea(q, v, a), mass @ a + direct.bias(q, v), 2e-10, "RNEA = M*a + bias")

    # Positive contact force acts ON the robot; upward support cancels gravity.
    zero = np.zeros(24)
    jacobian = direct.foot_jacobians(q0).reshape(12, 24)
    gravity = direct.bias(q0, zero)
    support = np.linalg.lstsq(jacobian.T[:6], gravity[:6], rcond=None)[0]
    torques = gravity[6:] - jacobian.T[6:] @ support
    _close(gravity, np.r_[np.zeros(6), torques] + jacobian.T @ support, 1e-10, "Static contact force sign")
    if np.min(support.reshape(4, 3)[:, 2]) <= 0:
        raise AssertionError("Static support requires positive upward normal forces")
    arbitrary_force = rng.normal(0., 10., 12)
    work_joint = float(v @ jacobian.T @ arbitrary_force)
    work_contact = float((jacobian @ v) @ arbitrary_force)
    _close(np.array([work_joint]), np.array([work_contact]), 1e-11, "Contact virtual work")
    print(f"PASS full dynamics and contact-force sign: min eig(M)={minimum_eigenvalue:.3e}, "
          f"static max |tau|={np.max(np.abs(torques)):.3f} Nm")

    epsilon = 1e-6
    plus, minus = direct.integrate(q, epsilon * v), direct.integrate(q, -epsilon * v)
    numeric_velocity = (direct.foot_positions(plus) - direct.foot_positions(minus)) / (2. * epsilon)
    velocity_error = _close(numeric_velocity, np.einsum("ijk,k->ij", direct.foot_jacobians(q), v),
                            2e-7, "Feet tangent Jacobian")
    numeric_jdot = (direct.foot_jacobians(plus) - direct.foot_jacobians(minus)) / (2. * epsilon)
    bias_error = _close(np.einsum("ijk,k->ij", numeric_jdot, v), direct.foot_bias_accelerations(q, v),
                        2e-7, "Feet Jdot*v")
    _close((direct.tool_position(plus) - direct.tool_position(minus)) / (2. * epsilon),
           direct.tool_jacobian(q) @ v, 2e-7, "Tool tangent Jacobian")
    _close(direct.difference(q, direct.integrate(q, v)), v, 2e-11, "SE(3) integration/logarithm")
    print(f"PASS independent finite differences: feet J error={velocity_error:.3e}, Jdot*v error={bias_error:.3e}")

    try:
        import casadi as ca
    except ImportError as error:
        raise ImportError("CasADi is needed for the remaining mandatory checks; install requirements.txt") from error
    functions = model.casadi_functions()
    tree_functions = model.casadi_functions(force_tree=True)
    numeric_cases = {
        "rnea": ([q, v, a], direct.rnea(q, v, a)),
        "mass_matrix": ([q], direct.mass_matrix(q)),
        "feet": ([q], direct.foot_positions(q).T),
        "foot_jacobians": ([q], direct.foot_jacobians(q).reshape(12, 24)),
        "foot_bias": ([q, v], direct.foot_bias_accelerations(q, v).T),
        "tool": ([q], direct.tool_position(q)),
        "tool_jacobian": ([q], direct.tool_jacobian(q)),
        "tool_rotation": ([q], direct.tool_rotation(q)),
        "integrate": ([q, v], direct.integrate(q, v)),
        "difference": ([q, q0], direct.difference(q, q0)),
    }
    maximum_backend_error = 0.
    for name, (arguments, expected) in numeric_cases.items():
        for label, functions_to_check in (("selected", functions), ("recursive", tree_functions)):
            actual = np.asarray(functions_to_check[name](*arguments)).reshape(expected.shape)
            maximum_backend_error = max(maximum_backend_error, _close(actual, expected, 2e-9,
                                         f"{label} symbolic/NumPy parity: {name}"))
    if model.pin is not None:
        _close(model.rnea(q, v, a), direct.rnea(q, v, a), 2e-9, "Native Pinocchio/NumPy RNEA")
        _close(model.foot_jacobians(q), direct.foot_jacobians(q), 2e-10, "Native Pinocchio/NumPy Jacobian")
    print(f"PASS symbolic/NumPy parity: backend={model.casadi_backend}, max error={maximum_backend_error:.3e}")

    delta = ca.SX.sym("delta", 24)
    for label, function_set in (("selected", functions), ("recursive", tree_functions)):
        roundtrip = function_set["difference"](q0, function_set["integrate"](q0, delta))
        derivative = ca.Function(f"manifold_derivative_{label}", [delta], [ca.jacobian(roundtrip, delta)])
        objective = ca.dot(roundtrip, roundtrip)
        hessian = ca.Function(f"manifold_hessian_{label}", [delta], [ca.hessian(objective, delta)[0]])
        _close(np.asarray(derivative(zero)), np.eye(24), 2e-9, f"{label} tangent Jacobian at zero")
        _close(np.asarray(hessian(zero)), 2. * np.eye(24), 2e-8, f"{label} finite tangent Hessian at zero")
    print("PASS SE(3) symbolic Jacobian/Hessian at zero")
    return model


def _check_solution(model, config, solution):
    """Recompute predicted residuals with the numerical rigid-body model."""
    tolerance = max(5e-3, 30. * config.tolerance)
    if not np.isfinite(solution.constraint_violation) or solution.constraint_violation > tolerance:
        raise AssertionError(f"Solver constraint violation {solution.constraint_violation:.3e}")
    maximum_dynamics_error = 0.
    for stage in range(len(solution.a)):
        q, v, acceleration = solution.q[stage], solution.v[stage], solution.a[stage]
        forces, torques = solution.forces[stage], solution.tau[stage]
        if np.asarray(forces).shape != (15,):
            raise AssertionError("Expected 12 foot-force and 3 tool-force components per stage")
        jacobian = model.foot_jacobians(q).reshape(12, 24)
        external = jacobian.T @ forces[:12] + model.tool_jacobian(q).T @ forces[12:]
        residual = model.rnea(q, v, acceleration) - external - np.r_[np.zeros(6), torques]
        maximum_dynamics_error = max(maximum_dynamics_error, float(np.max(np.abs(residual))))
        if maximum_dynamics_error > tolerance:
            raise AssertionError(f"Predicted inverse-dynamics residual {maximum_dynamics_error:.3e}")
        if (config.torque_nodes == 0 or stage < config.torque_nodes) and np.max(np.abs(torques) - model.description.effort_limits) > tolerance:
            raise AssertionError("Predicted actuator torque exceeds the WBT motor limit")
        foot_forces = forces[:12].reshape(4, 3)
        if np.min(foot_forces[:, 2]) < -tolerance:
            raise AssertionError("Predicted negative unilateral contact normal force")
        if np.max(foot_forces[:, 2]) > config.max_normal_force + tolerance:
            raise AssertionError("Predicted contact normal force exceeds its configured limit")
        if np.max(np.linalg.norm(foot_forces[:, :2], axis=1) - config.friction * foot_forces[:, 2]) > tolerance:
            raise AssertionError("Predicted contact force violates the circular friction cone")
        _close(forces[12:], np.asarray(config.tool_force), tolerance,
               "Tool environment-force equality")
        tangent_now = model.difference(solution.q[0], q)
        tangent_next = model.difference(solution.q[0], solution.q[stage + 1])
        _close(tangent_next, tangent_now + solution.dt[stage] * v, tolerance,
               "Predicted tangent-space Euler transition")
        _close(solution.v[stage + 1], v + solution.dt[stage] * acceleration, tolerance,
               "Predicted velocity Euler transition")
    return maximum_dynamics_error


def check_servo_rate(model):
    """Reproduce the swing-leg sampled-PD instability behind the 10 ms crash."""
    q = model.neutral()
    q[:3] = model.description.initial_position
    mass = model.mass_matrix(q)
    # Diagonal support leaves the other legs free to swing.
    jac = model.foot_jacobians(q)[[0, 3]].reshape(6, 24)
    kkt = np.block([[mass, -jac.T], [jac, np.zeros((6, 6))]])
    response = np.linalg.solve(kkt, np.vstack((np.eye(24)[:, 6:], np.zeros((6, 18)))))[6:24]
    cfg = MPCConfig()
    kp = np.r_[np.full(12, cfg.leg_kp), np.full(6, cfg.arm_kp)]
    kd = np.r_[np.full(12, cfg.leg_kd), np.full(6, cfg.arm_kd)]
    stiffness, damping = response @ np.diag(kp), response @ np.diag(kd)
    radii = []
    for dt in (.002, .010):
        update = np.block([[np.eye(18) - dt**2 * stiffness, dt * (np.eye(18) - dt * damping)],
                           [-dt * stiffness, np.eye(18) - dt * damping]])
        radii.append(float(np.max(np.abs(np.linalg.eigvals(update)))))
    if radii[0] > 1.00001 or radii[1] < 1.1:
        raise AssertionError(f"Unexpected sampled-PD stability radii: {radii}")
    print(f"PASS sampled-PD diagnosis: 2 ms spectral radius={radii[0]:.6f}, 10 ms={radii[1]:.6f}")


def _plant_step(model, q, v, torques, anchors, dt, stabilization):
    """Independent constrained forward dynamics with fixed point contacts.

    Solve M*a + h = S.T*tau + J.T*lambda and stabilized J*a + Jdot*v = 0.
    The plant's contact forces are solved here, never copied from MPC outputs.
    """
    mass = model.mass_matrix(q)
    jacobian = model.foot_jacobians(q).reshape(12, 24)
    drift = (model.foot_positions(q) - anchors).reshape(12)
    contact_bias = model.foot_bias_accelerations(q, v).reshape(12)
    rhs_dynamics = np.r_[np.zeros(6), torques] - model.bias(q, v)
    rhs_contact = -contact_bias - 2. * stabilization * (jacobian @ v) - stabilization**2 * drift
    kkt = np.block([[mass, -jacobian.T], [jacobian, np.zeros((12, 12))]])
    result = np.linalg.solve(kkt, np.r_[rhs_dynamics, rhs_contact])
    acceleration, contact_forces = result[:24], result[24:].reshape(4, 3)
    _close(mass @ acceleration + model.bias(q, v), np.r_[np.zeros(6), torques] + jacobian.T @ result[24:],
           1e-8, "Independent plant dynamics")
    # Lie-group constant-acceleration integration, independent of MPC shooting.
    q_next = model.integrate(q, dt * v + .5 * dt * dt * acceleration)
    v_next = v + dt * acceleration
    return q_next, v_next, contact_forces


def check_solver_derivatives(controller, q, v, reference):
    """Check assembled stage callbacks independently by directional differences."""
    ca = controller.ca
    q, v = q[:controller.model.nq], v[:controller.model.nv]
    reference = replace(reference, joint_positions=reference.joint_positions[:controller.model.na])
    parameters, contacts = controller._parameters(q, v, reference, ContactSchedule())
    rng = np.random.default_rng(731)
    decision = controller._initial_guess(q, v, contacts, reference.tool_force)
    decision += rng.normal(scale=.015, size=decision.size)
    direction = rng.normal(size=decision.size)
    direction /= np.linalg.norm(direction)
    multipliers = rng.normal(scale=.1, size=controller.lbg.size)
    objective_scale, epsilon = .7, 1e-5
    gradient = controller.solver.get_function("nlp_grad_f")
    jacobian = controller.solver.get_function("nlp_jac_g")
    hessian = controller.solver.get_function("nlp_hess_l")

    def grad(value):
        result = gradient(value, parameters)
        return np.asarray(result[-1] if isinstance(result, tuple) else result).reshape(-1)

    def jac(value):
        return np.asarray(jacobian(value, parameters)[1])

    plus, minus = decision + epsilon * direction, decision - epsilon * direction
    g_fd = np.asarray((controller.constraint_function(plus, parameters)
                       - controller.constraint_function(minus, parameters)) / (2 * epsilon)).reshape(-1)
    jac_error = _close(jac(decision) @ direction, g_fd, 2e-5, "Assembled constraint Jacobian")
    h = hessian(decision, parameters, objective_scale, multipliers)
    if isinstance(h, tuple):
        lag_gradient, h = h
        _close(np.asarray(lag_gradient).reshape(-1),
               objective_scale * grad(decision) + jac(decision).T @ multipliers,
               1e-8, "Fatrop Lagrangian gradient")
    elif controller.config.solver == "ipopt":
        h = h + h.T - ca.diag(ca.diag(h))
    lag_fd = (objective_scale * (grad(plus) - grad(minus))
              + (jac(plus) - jac(minus)).T @ multipliers) / (2 * epsilon)
    hess_error = _close(np.asarray(h) @ direction, lag_fd, 2e-4,
                        "Assembled Lagrangian Hessian")
    print(f"PASS stage derivative assembly: J error={jac_error:.3e}, H error={hess_error:.3e}")


def check_force_warm_start(controller, q, v, reference):
    """Preserve solved support loads, and reset them when the mode changes."""
    previous = controller._previous
    q, v = q[:controller.model.nq], v[:controller.model.nv]
    contacts = controller._previous_contacts
    guess = controller._initial_guess(q, v, contacts, reference.tool_force)
    for k, rows in enumerate(controller._force_slices):
        _close(guess[rows], previous.forces[k, :12], 1e-12, "Unchanged contact force warm start")
    changed = ContactSchedule("trot", 0., controller.config.gait_period).horizon(controller.times)
    if np.array_equal(contacts, changed):
        raise AssertionError("Warm-start test needs a changed contact mode")
    guess = controller._initial_guess(q, v, changed, reference.tool_force)
    for k, rows in enumerate(controller._force_slices):
        expected = np.zeros((4, 3))
        expected[changed[k], 2] = controller.model.total_mass * 9.81 / changed[k].sum()
        _close(guess[rows], expected.reshape(12), 1e-12, "Changed contact force reset")
    print("PASS force warm start and reset at contact-mode changes")


def check_solver(model, args):
    try:
        from .mpc import InverseDynamicsMPC
    except ImportError:
        from mpc import InverseDynamicsMPC
    config = replace(MPCConfig(), horizon=args.horizon, max_iterations=args.max_iterations,
                     jit=not args.no_jit)
    config.solver = args.solver
    config.validate()
    q = model.neutral()
    q[:3] = model.description.initial_position
    v = np.zeros(model.nv)
    anchors = model.foot_positions(q)
    reference = MPCReference(q[:3].copy(), q[3:7].copy(), np.zeros(3), 0., q[7:].copy(),
                             model.tool_position(q), np.zeros(3), np.zeros(3))
    controller = InverseDynamicsMPC(model, config)
    print(f"NLP build={controller.build_time:.2f}s, cache_hit={controller.cache_hit}, "
          f"variables={controller.lbx.size}, constraints={controller.lbg.size}")
    check_solver_derivatives(controller, q, v, reference)
    if hasattr(controller.solver, 'symbolic'):
        # Independently audit the generated solver's post-solve f/g callback.
        # Wrong values here could silently bypass the feasibility guard.
        underlying = controller.solver
        class AuditedSolver:
            checked = False
            def __call__(self, **arguments):
                result = underlying(**arguments)
                if not self.checked:
                    expected_f = underlying.symbolic.get_function('nlp_f')(result['x'], arguments['p'])
                    expected_g = controller.constraint_function(result['x'], arguments['p'])
                    _close(result['f'], expected_f, 1e-8, 'Compiled returned objective')
                    _close(result['g'], expected_g, 1e-8, 'Compiled returned constraints')
                    print('PASS compiled post-solve objective and constraints')
                    self.checked = True
                return result
            def stats(self):
                return underlying.stats()
        controller.solver = AuditedSolver()
    initial_q = q.copy()
    total_time, total_wall, largest_residual = 0., 0., 0.
    min_normal, max_friction_excess, max_foot_drift = np.inf, -np.inf, 0.
    substeps = max(1, int(np.ceil(config.solve_period / args.plant_dt)))
    dt = config.solve_period / substeps
    gains_p = np.r_[np.full(12, config.leg_kp), np.full(6, config.arm_kp)]
    gains_d = np.r_[np.full(12, config.leg_kd), np.full(6, config.arm_kd)]
    for index in range(args.steps):
        schedule = ContactSchedule("stand", index * config.solve_period, config.gait_period)
        started = time.perf_counter()
        solution = controller.solve(q, v, reference, schedule)
        wall = time.perf_counter() - started
        total_wall += wall
        if index == 0 and config.warm_start:
            check_force_warm_start(controller, q, v, reference)
        largest_residual = max(largest_residual, _check_solution(model, config, solution))
        total_time += solution.solve_time
        for substep in range(substeps):
            target_q, target_v, feedforward = solution.sample(substep * dt)
            if np.asarray(target_q).shape != (18,) or np.asarray(target_v).shape != (18,):
                raise AssertionError("Solution sample must return 18 native-encoder positions and velocities")
            torques = feedforward + gains_p * (target_q - q[7:]) + gains_d * (target_v - v[6:])
            torques = np.clip(torques, -model.description.effort_limits, model.description.effort_limits)
            q, v, forces = _plant_step(model, q, v, torques, anchors, dt, config.stabilization)
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                raise AssertionError("Physical rollout produced a nonfinite state")
            min_normal = min(min_normal, float(np.min(forces[:, 2])))
            max_friction_excess = max(max_friction_excess,
                                     float(np.max(np.linalg.norm(forces[:, :2], axis=1)
                                                  - config.friction * forces[:, 2])))
            max_foot_drift = max(max_foot_drift, float(np.max(np.linalg.norm(model.foot_positions(q) - anchors, axis=1))))
        print(f"PASS {args.solver} solve {index + 1}/{args.steps}: {solution.status}, "
              f"solver={solution.solve_time:.3f}s, total={wall:.3f}s, iterations={solution.iterations}, "
              f"violation={solution.constraint_violation:.3e}, "
              f"base_z={q[2]:.5f} m")
    if min_normal < -0.1 or max_friction_excess > .1:
        raise AssertionError(f"Standing plant contact invalid: min normal={min_normal:.3f}, "
                             f"friction excess={max_friction_excess:.3f} N")
    if max_foot_drift > 2e-3:
        raise AssertionError(f"Standing plant foot drift exceeds 2 mm: {max_foot_drift:.6f} m")
    if np.linalg.norm(q[:3] - initial_q[:3]) > .03:
        raise AssertionError("Standing plant base drift exceeds 3 cm")
    if abs(np.linalg.norm(q[3:7]) - 1.) > 1e-8:
        raise AssertionError("Physical rollout lost quaternion normalization")
    print(f"PASS independent standing plant ({args.steps * config.solve_period:.3f} s): "
          f"minimum normal={min_normal:.3f} N, maximum foot drift={max_foot_drift:.3e} m, "
          f"predicted dynamics residual={largest_residual:.3e}, average solve={total_time / args.steps:.3f} s, "
          f"average total={total_wall / args.steps:.3f} s")


def check_runtime_helpers():
    """Check controller units, phase changes, interpolation and fault guards."""
    try:
        from .inv_dyn_mpc import StateEstimator, Actuators
        from .mpc import MPCSolution, swing_reference
    except ImportError:
        from inv_dyn_mpc import StateEstimator, Actuators
        from mpc import MPCSolution, swing_reference
    estimator = StateEstimator(.002)
    q, v = estimator.update(np.zeros(3), [0., 0., np.pi / 2], [1., 0., 0.], np.zeros(3), np.zeros(18))
    _close(v[:3], np.array([0., -1., 0.]), 1e-12, "World GPS to body velocity")
    _close(v[6:], np.zeros(18), 0., "First encoder sample has no derivative spike")
    try:
        estimator.update(np.full(3, np.nan), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(18))
    except ValueError:
        pass
    else:
        raise AssertionError("Runtime accepted a nonfinite sensor reading")
    schedule = ContactSchedule("trot", .28, .6)
    table = schedule.horizon([0., .01, .03])
    _close(table.astype(int), np.array([[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]),
           0., "Gait physical-time mode change")
    for phase in [0., .5, 1.]:
        height, speed = swing_reference(phase, .3, .05)
        _close(np.array([height, speed]), np.array([.05 if phase == .5 else 0., 0.]),
               1e-12, "Cubic swing endpoints/apex")
    solution = MPCSolution(np.zeros((4, 25)), np.ones((4, 24)), np.ones((3, 24)) * 2,
                           np.tile(np.arange(3)[:, None], (1, 18)), np.zeros((3, 15)),
                           np.array([.02, .03, .04]), 0., 0., "test")
    qj, vj, tau = solution.sample(.035)
    _close(qj, np.full(18, .015), 1e-12, "Second-interval desired position")
    _close(vj, np.full(18, 1.03), 1e-12, "Second-interval desired velocity")
    _close(tau, np.full(18, 1.5), 1e-12, "Second-interval interpolated torque")
    try:
        solution.sample(.0501)
    except ValueError:
        pass
    else:
        raise AssertionError("Torque interpolation allowed a stale trajectory")

    class Motor:
        def getMaxTorque(self): return 80.
        def getAvailableTorque(self): return 60.
        def getMaxVelocity(self): return 3.
        def setPosition(self, value): self.position = value
        def setVelocity(self, value): self.velocity = value
        def setTorque(self, value): self.torque = value

    class Robot:
        def __init__(self): self.devices = {}
        def getDevice(self, name): return self.devices.setdefault(name, Motor())

    actuators = Actuators(Robot(), RobotDescription.from_world())
    _close(actuators.torque(np.full(18, 1000.)), np.full(18, 60.), 0., "Live motor torque cap")
    try:
        actuators.torque(np.full(18, np.nan))
    except ValueError:
        pass
    else:
        raise AssertionError("Motor interface accepted nonfinite torque")
    actuators.position(np.zeros(18))
    if actuators.mode != "position" or any(motor.position != 0 for motor in actuators.motors):
        raise AssertionError("Fault hold did not restore position mode")
    for config in (MPCConfig(tool_force_enabled="false"), MPCConfig(tool_force=(1., 0., 0.))):
        try:
            config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("Force-mode configuration accepted an invalid setting")
    print("PASS runtime conversion, gait switching, torque interpolation/limits and fault holds")


def check_tasks(model, args):
    """Feasibility checks for moving tool and changing gait, not walking proof."""
    try:
        from .mpc import InverseDynamicsMPC, swing_reference
    except ImportError:
        from mpc import InverseDynamicsMPC, swing_reference
    config = replace(MPCConfig(), horizon=args.horizon, solver=args.solver,
                     max_iterations=args.max_iterations, jit=not args.no_jit)
    solver = InverseDynamicsMPC(model, config)
    q, v = model.neutral(), np.zeros(24)
    q[:3] = model.description.initial_position
    for name, phase in (("tool", 0.), ("trot", 0.), ("trot", .25), ("stand", 0.)):
        position, velocity = model.tool_position(q), np.zeros(3)
        if name == "tool":
            position[2] += .01
        if name == "trot":
            velocity[0] = .06
        reference = MPCReference(q[:3].copy(), q[3:7].copy(), velocity, 0., q[7:].copy(),
                                 position, velocity.copy(), np.zeros(3))
        schedule = ContactSchedule("trot" if name == "trot" else "stand", phase, config.gait_period)
        solution = solver.solve(q, v, reference, schedule)
        _check_solution(model, config, solution)
        contacts = schedule.horizon(solver.times)
        swing_forces = solution.forces[:, :12].reshape(-1, 4, 3)[~contacts[:-1]]
        if swing_forces.size and np.max(np.abs(swing_forces)) > 1e-6:
            raise AssertionError("Swing feet generated contact force")
        for k in range(1, config.horizon):
            feet = model.foot_positions(solution.q[k])
            foot_velocity = np.einsum("ijk,k->ij", model.foot_jacobians(solution.q[k]), solution.v[k])
            for leg in range(4):
                if contacts[k, leg]:
                    _close(foot_velocity[leg], np.zeros(3), .005, "Predicted no-slip stance")
                else:
                    z, dz = swing_reference(schedule.swing_phase(solver.times[k])[leg],
                                            config.gait_period / 2, config.swing_height)
                    target = dz
                    _close(np.array([foot_velocity[leg, 2]]), np.array([target]), .005,
                           "Predicted swing vertical velocity")
            command = (reference.tool_velocity + config.tool_position_gain *
                       (reference.tool_position - model.tool_position(q)))
            _close(model.tool_jacobian(solution.q[k]) @ solution.v[k], command, .005,
                   "Predicted tool velocity target")
        print(f"PASS {args.solver} task {name} phase={phase:.2f}: "
              f"{solution.solve_time:.3f}s, violation={solution.constraint_violation:.3e}")
    force_config = replace(config, tool_force_enabled=True, tool_force=(-5., 0., 0.))
    force_solver = InverseDynamicsMPC(model, force_config)
    reference = MPCReference(q[:3].copy(), q[3:7].copy(), np.zeros(3), 0., q[7:].copy(),
                             model.tool_position(q), np.zeros(3), np.asarray(force_config.tool_force))
    solution = force_solver.solve(q, v, reference, ContactSchedule())
    _check_solution(model, force_config, solution)
    print(f"PASS {args.solver} prescribed -5N tool-force optimization: "
          f"{solution.solve_time:.3f}s, violation={solution.constraint_violation:.3e} "
          "(assumed contact; no physical object in this test)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve", action="store_true", help="also solve MPC and run an independent standing plant")
    parser.add_argument("--tasks", action="store_true", help="also check tool tracking and trot phase feasibility")
    parser.add_argument("--no-jit", action="store_true", help="use interpreted callbacks for comparison without compiling")
    parser.add_argument("--solver", choices=("ipopt", "fatrop"), default=MPCConfig.solver)
    parser.add_argument("--steps", type=int, default=1, help="receding-horizon standing solves (default: 1)")
    parser.add_argument("--horizon", type=int, default=MPCConfig.horizon, help="number of shooting intervals (default: 14)")
    parser.add_argument("--max-iterations", type=int, default=MPCConfig.max_iterations)
    parser.add_argument("--plant-dt", type=float, default=.002, help="maximum independent plant integration step")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.horizon < 2 or args.max_iterations < 1 or not 0 < args.plant_dt <= .02:
        parser.error("steps/max-iterations must be positive, horizon >=2, and 0 < plant-dt <= 0.02")
    started = time.perf_counter()
    model = check_model()
    check_servo_rate(model)
    check_runtime_helpers()
    if args.solve:
        check_solver(model, args)
    if args.tasks:
        check_tasks(model, args)
    print(f"All requested checks passed in {time.perf_counter() - started:.2f} s.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, ImportError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
