import numpy as np


class Bezier:
    __p0 = np.array([[0.0], [0.0], [0.0]])
    __pf = np.array([[0.0], [0.0], [0.0]])
    __p = np.array([[0.0], [0.0], [0.0]])
    __v = np.array([[0.0], [0.0], [0.0]])
    __a = np.array([[0.0], [0.0], [0.0]])
    __height = 0.0

    def __init__(self, p0, pf, height):
    # p0是摆动相起点，pf是落地点，height是腿的高度
        self.__p0 = p0
        self.__pf = pf
        self.__height = height


    def calculate_bezier(self, phase, swing_time):
        # phase必须是0到1范围内的浮点数
        self.__p = pos(self.__p0, self.__pf, phase)
        self.__v = vel(self.__p0, self.__pf, phase) / swing_time
        self.__a = acc(self.__p0, self.__pf, phase) / (swing_time * swing_time)
        if phase < 0.5:
            zp = pos(self.__p0[2][0], self.__p0[2][0] + self.__height, phase * 2)
            zv = vel(self.__p0[2][0], self.__p0[2][0] + self.__height, phase * 2)*2/swing_time
            za = acc(self.__p0[2][0], self.__p0[2][0] + self.__height, phase * 2)*4/(swing_time*swing_time)
        else:
            zp = pos(self.__p0[2][0] + self.__height, self.__pf[2][0], phase * 2 - 1)
            zv = vel(self.__p0[2][0] + self.__height, self.__pf[2][0], phase * 2 - 1)*2/swing_time
            za = acc(self.__p0[2][0] + self.__height, self.__pf[2][0], phase * 2 - 1)*4/(swing_time*swing_time)
        self.__p[2][0] = zp
        self.__v[2][0] = zv
        self.__a[2][0] = za

    def get_position(self):
        return self.__p

    def get_velocity(self):
        return self.__v

    def get_acceleration(self):
        return self.__a


def pos(p0, pf, x):
    p_diff = pf - p0
    b = x * x * x + 3 * (x * x * (1 - x))
    return p0 + b * p_diff


def vel(p0, pf, x):
    p_diff = pf - p0
    b = 6 * x * (1 - x)
    return b * p_diff


def acc(p0, pf, x):
    p_diff = pf - p0
    b = 6 - 12 * x
    return b * p_diff
