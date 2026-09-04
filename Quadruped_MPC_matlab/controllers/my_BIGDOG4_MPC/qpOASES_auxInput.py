"""Python equivalent of the qpOASES auxiliary-input helper.

The original MATLAB helper is part of qpOASES and is licensed under LGPL-2.1.
The converted controller uses OSQP, but this compatibility function preserves
the MATLAB helper's public data structure for callers that still need it.
"""

import copy

import numpy as np


_FIELDS = ("hessianType", "x0", "guessedWorkingSetB", "guessedWorkingSetC", "R")


def qpOASES_auxInput(*args):
    if args and isinstance(args[0], dict):
        if len(args) % 2 != 1:
            raise ValueError("Auxiliary inputs must be specified in pairs")
        result = copy.deepcopy(args[0])
        start = 1
    else:
        if len(args) % 2 != 0:
            raise ValueError("Auxiliary inputs must be specified in pairs")
        result = {field: None for field in _FIELDS}
        start = 0

    for index in range(start, len(args), 2):
        name, value = args[index], args[index + 1]
        if not isinstance(name, str) or not name:
            raise ValueError("Auxiliary input names must be non-empty strings")
        if name not in result:
            raise ValueError(f"Invalid qpOASES auxiliary input: {name}")
        if not isinstance(value, (int, float, complex, list, tuple, np.ndarray)):
            raise TypeError(f"Auxiliary input {name} must be numerical")
        result[name] = np.asarray(value) if isinstance(value, (list, tuple)) else value
    return result


qp_oases_aux_input = qpOASES_auxInput
