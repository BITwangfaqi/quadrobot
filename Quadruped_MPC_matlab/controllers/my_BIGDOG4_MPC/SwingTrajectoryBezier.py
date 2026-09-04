"""Cubic Bezier swing-foot trajectory."""

import numpy as np


def cubicBezier(p0, pf, phase):
    p0 = np.asarray(p0, dtype=float)
    pf = np.asarray(pf, dtype=float)
    blend = phase**3 + 3.0 * phase**2 * (1.0 - phase)
    return p0 + blend * (pf - p0)


def cubicBezier_v(p0, pf, phase):
    return (
        6.0
        * phase
        * (1.0 - phase)
        * (np.asarray(pf, dtype=float) - np.asarray(p0, dtype=float))
    )


def cubicBezier_a(p0, pf, phase):
    return (6.0 - 12.0 * phase) * (
        np.asarray(pf, dtype=float) - np.asarray(p0, dtype=float)
    )


def SwingTrajectoryBezier(pf_init, pf_final, phase, swingtime, height):
    """Return swing-foot position, velocity and acceleration."""
    if swingtime <= 0.0:
        raise ValueError("swingtime must be positive")

    pf_init = np.asarray(pf_init, dtype=float).reshape(3)
    pf_final = np.asarray(pf_final, dtype=float).reshape(3)
    phase = float(np.clip(phase, 0.0, 1.0))

    position = cubicBezier(pf_init, pf_final, phase)
    velocity = cubicBezier_v(pf_init, pf_final, phase) / swingtime
    acceleration = cubicBezier_a(pf_init, pf_final, phase) / (swingtime * swingtime)

    if phase < 0.5:
        z_phase = 2.0 * phase
        z_start, z_end = pf_init[2], pf_init[2] + height
    else:
        z_phase = 2.0 * phase - 1.0
        z_start, z_end = pf_init[2] + height, pf_final[2]

    position[2] = cubicBezier(z_start, z_end, z_phase)
    velocity[2] = cubicBezier_v(z_start, z_end, z_phase) * 2.0 / swingtime
    acceleration[2] = (
        cubicBezier_a(z_start, z_end, z_phase) * 4.0 / (swingtime * swingtime)
    )
    return position, velocity, acceleration


swing_trajectory_bezier = SwingTrajectoryBezier
