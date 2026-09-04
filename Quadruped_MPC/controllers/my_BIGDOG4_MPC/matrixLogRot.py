"""Logarithm map from SO(3) to a rotation vector."""

import numpy as np


def matrixLogRot(rotation):
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("matrixLogRot expects a 3x3 rotation matrix")

    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cosine))
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )

    if theta < 1.0e-4:
        return 0.5 * vee
    if np.pi - theta < 1.0e-5:
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        axis[0] = np.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = np.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis[2] = np.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
        norm = np.linalg.norm(axis)
        if norm > 0.0:
            return theta * axis / norm
    return vee * theta / (2.0 * np.sin(theta))


matrix_log_rot = matrixLogRot
