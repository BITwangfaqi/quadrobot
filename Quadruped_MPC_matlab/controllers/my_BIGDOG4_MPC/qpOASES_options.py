"""Python equivalent of the qpOASES options-structure helper.

The original MATLAB helper is part of qpOASES and is licensed under LGPL-2.1.
These dictionaries are retained for source compatibility; the converted QPs
are solved by OSQP in :mod:`qp_solver`.
"""

import copy
import numbers

import numpy as np


def _default_options():
    eps = np.finfo(float).eps
    return {
        "maxIter": -1,
        "maxCpuTime": -1,
        "printLevel": 1,
        "enableRamping": 1,
        "enableFarBounds": 1,
        "enableFlippingBounds": 1,
        "enableRegularisation": 0,
        "enableFullLITests": 0,
        "enableNZCTests": 1,
        "enableDriftCorrection": 1,
        "enableCholeskyRefactorisation": 0,
        "enableEqualities": 0,
        "terminationTolerance": 5.0e6 * eps,
        "boundTolerance": 1.0e6 * eps,
        "boundRelaxation": 1.0e4,
        "epsNum": -1.0e3 * eps,
        "epsDen": 1.0e3 * eps,
        "maxPrimalJump": 1.0e8,
        "maxDualJump": 1.0e8,
        "initialRamping": 0.5,
        "finalRamping": 1.0,
        "initialFarBounds": 1.0e6,
        "growFarBounds": 1.0e3,
        "initialStatusBounds": -1,
        "epsFlipping": 1.0e3 * eps,
        "numRegularisationSteps": 0,
        "epsRegularisation": 1.0e3 * eps,
        "numRefinementSteps": 1,
        "epsIterRef": 1.0e2 * eps,
        "epsLITests": 1.0e5 * eps,
        "epsNZCTests": 3.1e3 * eps,
    }


def _scheme_options(scheme):
    options = _default_options()
    if scheme == "reliable":
        options.update(
            enableFullLITests=1,
            enableCholeskyRefactorisation=1,
            numRefinementSteps=2,
        )
    elif scheme in ("mpc", "fast"):
        options.update(
            enableRamping=0,
            enableFarBounds=1,
            enableFlippingBounds=0,
            enableRegularisation=1,
            enableNZCTests=0,
            enableDriftCorrection=0,
            enableEqualities=1,
            terminationTolerance=1.0e9 * np.finfo(float).eps,
            initialStatusBounds=0,
            numRegularisationSteps=1,
            numRefinementSteps=0,
        )
    return options


def qpOASES_options(*args):
    start = 0
    if args and isinstance(args[0], dict):
        if len(args) % 2 != 1:
            raise ValueError("Options must be specified in pairs")
        options = copy.deepcopy(args[0])
        start = 1
    elif (
        args
        and isinstance(args[0], str)
        and args[0].lower() in ("default", "reliable", "mpc", "fast")
    ):
        if len(args) % 2 != 1:
            raise ValueError("Options must be specified in pairs")
        options = _scheme_options(args[0].lower())
        start = 1
    else:
        if len(args) % 2 != 0:
            raise ValueError("Options must be specified in pairs")
        options = _default_options()

    for index in range(start, len(args), 2):
        name, value = args[index], args[index + 1]
        if not isinstance(name, str) or not name:
            raise ValueError("Option names must be non-empty strings")
        if name not in options:
            raise ValueError(f"Invalid qpOASES option: {name}")
        if not isinstance(value, numbers.Number) or np.ndim(value) != 0:
            raise TypeError(f"Option {name} must be a scalar number")
        options[name] = value
    return options


qp_oases_options = qpOASES_options
