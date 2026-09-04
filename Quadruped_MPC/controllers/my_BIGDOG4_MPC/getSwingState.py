"""Gait swing-phase calculation."""

import numpy as np


def getSwingState(phase, offsetsFloat, durationsFloat):
    offsets = np.asarray(offsetsFloat, dtype=float).reshape(4)
    durations = np.asarray(durationsFloat, dtype=float).reshape(4)
    swing_offset = np.mod(offsets + durations, 1.0)
    swing_duration = 1.0 - durations
    state = np.mod(float(phase) - swing_offset, 1.0)

    result = np.zeros(4)
    active = (swing_duration > 0.0) & (state <= swing_duration)
    result[active] = state[active] / swing_duration[active]
    return result


get_swing_state = getSwingState
