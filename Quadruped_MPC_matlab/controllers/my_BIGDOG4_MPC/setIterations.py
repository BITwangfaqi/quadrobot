"""Convert the controller counter to gait iteration and phase."""

import numpy as np


def setIterations(nIterations, currentIteration, IterationsBetweenMpc):
    cycle = int(IterationsBetweenMpc) * int(nIterations)
    iteration = int(np.floor((currentIteration / IterationsBetweenMpc) % nIterations))
    phase = (int(currentIteration) % cycle) / cycle
    return iteration, phase


set_iterations = setIterations
