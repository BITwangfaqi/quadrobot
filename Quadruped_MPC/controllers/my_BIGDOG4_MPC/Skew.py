"""Skew-symmetric matrix helper."""

import numpy as np


def Skew(vector):
    """Return the 3x3 cross-product matrix of a three-element vector."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.size != 3:
        raise ValueError("Skew expects exactly three elements")
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


skew = Skew
