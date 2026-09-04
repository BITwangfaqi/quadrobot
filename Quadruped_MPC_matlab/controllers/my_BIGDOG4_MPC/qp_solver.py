"""Quadratic-programming backend replacing the Windows qpOASES MEX file."""

import numpy as np
from scipy import sparse

try:
    import osqp
except ImportError:  # pragma: no cover - exercised only on installations without OSQP
    osqp = None


def _as_vector(value, size, name):
    if value is None:
        return None
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} must have {size} elements")
    return vector


def _scipy_fallback(hessian, gradient, constraint_matrix, lower, upper):
    from scipy.optimize import LinearConstraint, minimize

    objective = lambda x: 0.5 * x @ hessian @ x + gradient @ x
    jacobian = lambda x: hessian @ x + gradient
    constraints = LinearConstraint(constraint_matrix, lower, upper)
    result = minimize(
        objective,
        np.zeros(hessian.shape[0]),
        jac=jacobian,
        hess=lambda _x: hessian,
        constraints=constraints,
        method="trust-constr",
        options={"maxiter": 1000, "verbose": 0},
    )
    if not result.success:
        raise RuntimeError(f"QP solver failed: {result.message}")
    return result.x


def solve_qp(H, g, A=None, lb=None, ub=None, lbA=None, ubA=None):
    """Solve ``0.5*x.T*H*x + g.T*x`` with variable and linear bounds."""
    hessian = np.asarray(H, dtype=float)
    gradient = np.asarray(g, dtype=float).reshape(-1)
    n_variables = gradient.size
    if hessian.shape != (n_variables, n_variables):
        raise ValueError("H has an incompatible shape")
    hessian = 0.5 * (hessian + hessian.T)

    matrices = []
    lower_bounds = []
    upper_bounds = []

    variable_lower = _as_vector(lb, n_variables, "lb")
    variable_upper = _as_vector(ub, n_variables, "ub")
    if variable_lower is not None or variable_upper is not None:
        matrices.append(sparse.eye(n_variables, format="csc"))
        lower_bounds.append(
            variable_lower
            if variable_lower is not None
            else np.full(n_variables, -np.inf)
        )
        upper_bounds.append(
            variable_upper
            if variable_upper is not None
            else np.full(n_variables, np.inf)
        )

    if A is not None:
        linear_matrix = sparse.csc_matrix(np.asarray(A, dtype=float))
        n_constraints = linear_matrix.shape[0]
        if linear_matrix.shape[1] != n_variables:
            raise ValueError("A has an incompatible shape")
        matrices.append(linear_matrix)
        lower_bounds.append(
            _as_vector(lbA, n_constraints, "lbA")
            if lbA is not None
            else np.full(n_constraints, -np.inf)
        )
        upper_bounds.append(
            _as_vector(ubA, n_constraints, "ubA")
            if ubA is not None
            else np.full(n_constraints, np.inf)
        )

    if not matrices:
        return np.linalg.solve(hessian, -gradient)

    constraint_matrix = sparse.vstack(matrices, format="csc")
    lower = np.concatenate(lower_bounds)
    upper = np.concatenate(upper_bounds)

    if osqp is None:
        return _scipy_fallback(hessian, gradient, constraint_matrix, lower, upper)

    solver = osqp.OSQP()
    solver.setup(
        P=sparse.triu(sparse.csc_matrix(hessian), format="csc"),
        q=gradient,
        A=constraint_matrix,
        l=lower,
        u=upper,
        verbose=False,
        eps_abs=1.0e-5,
        eps_rel=1.0e-5,
        max_iter=10000,
        polish=False,
        warm_start=True,
    )
    result = solver.solve()
    if result.info.status_val not in (1, 2):
        raise RuntimeError(f"OSQP failed: {result.info.status}")
    return np.asarray(result.x, dtype=float)
