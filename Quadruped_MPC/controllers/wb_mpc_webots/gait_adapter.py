"""Webots trot with a finite all-feet support interval between diagonal swings."""
import numpy as np
from wb_mpc.utils.gait_sequence import GaitSequence as OriginalGaitSequence


class GaitSequence(OriginalGaitSequence):
    def __init__(self, gait_type='trot', gait_period=0.8):
        super().__init__(gait_type, gait_period)
        if gait_type == 'trot':
            self.transfer_period = 0.16
            self.swing_period = gait_period/2-self.transfer_period
            if self.swing_period <= 0:
                raise ValueError('Trot half-period must exceed the 160 ms support transfer')

    def get_gait_schedule(self, t_current, dts, nodes):
        if self.gait_type != 'trot':
            return super().get_gait_schedule(t_current, dts, nodes)
        contacts = np.ones((4, nodes))
        swings = np.zeros((4, nodes))
        times = t_current + np.r_[0., np.cumsum(dts[:nodes-1])]
        for i, time in enumerate(times):
            half = self.gait_period/2
            phase = (time+1e-9) % half
            if phase < self.transfer_period:
                continue
            pair = (0, 3) if (time+1e-9) % self.gait_period < half else (1, 2)
            for foot in pair:
                contacts[foot, i] = 0.
                swings[foot, i] = (phase-self.transfer_period)/self.swing_period
        return contacts, swings
