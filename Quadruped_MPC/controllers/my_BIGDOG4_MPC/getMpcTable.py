"""Build the horizon contact table for the MPC solver."""

import numpy as np


def getMpcTable(iteration, nIterations, offsets, durations):
    offsets = np.asarray(offsets, dtype=int).reshape(4)
    durations = np.asarray(durations, dtype=int).reshape(4)
    table = np.zeros(4 * int(nIterations))

    for i in range(int(nIterations)):
        gait_iteration = (i + int(iteration)) % int(nIterations)
        progress = np.mod(gait_iteration - offsets, int(nIterations))
        table[4 * i : 4 * i + 4] = progress < durations
    return table


get_mpc_table = getMpcTable
