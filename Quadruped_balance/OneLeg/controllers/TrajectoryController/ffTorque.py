import math
import numpy as np

l1 = 0.03  # 大腿以及hipmotor质心到hip转轴的距离
l2 = 0.09  # 小腿质心到knee转轴的距离
a1 = 0.2  # hip转轴到knee转轴的距离，同L1
a2 = 0.2  # knee转轴到足端的距离，同L2
m1 = 0.8  # 大腿以及hipmotor的总质量
I1 = 0.003448  # 大腿以及hipmotor在切平面的总转动惯量
m2 = 0.2  # 小腿质量
I2 = 0.0004  # 小腿在切平面的转动惯量
g = 9.8  # 重力加速度

# 计算逆动力学需要的参数
# 计算广义质量矩阵
def calculate_B(alpha1, alpha2):
    b11 = I1 + m1 * l1 * l1 + I2 + m2 * (a1 * a1 + l2 * l2 + 2 * a1 * l2 * math.cos(alpha2))
    b12 = I2 + m2 * (l2 * l2 + a1 * l2 * math.cos(alpha2))
    b21 = b12
    b22 = I2 + m2 * l2 * l2
    return np.array([[b11, b12], [b21, b22]])

# 计算科里奥利力矩阵
def calculate_Cqdot(alpha1, alpha2, alpha1_dot, alpha2_dot):
    h = -m2 * a1 * l2 * math.sin(alpha2)
    C = np.array([[h * alpha2_dot, h * (alpha1_dot + alpha2_dot)], [-h * alpha2_dot, 0]])
    return C.dot(np.array([[alpha1_dot], [alpha2_dot]]))

# 计算重力
def calculate_G(alpha1, alpha2):
    g1 = m1 * g * l1 * math.sin(alpha1) + m2 * g * (a1 * math.sin(alpha1) + l2 * math.sin(alpha1 + alpha2))
    g2 = m2 * g * l2 * math.sin(alpha1 + alpha2)
    return np.array([[g1], [g2]])


def calculate_J_simple(alpha1, alpha2):
    return np.array([[-a1 * math.sin(alpha1) - a2 * math.sin(alpha1 + alpha2), -a2 * math.sin(alpha1 + alpha2)],
                     [a1 * math.cos(alpha1) + a2 * math.cos(alpha1 + alpha2), a2 * math.cos(alpha1 + alpha2)]])

