from calculation import *
from parameters import *
import numpy as np

# 比例矩阵
Kp_swing = np.array([[200, 0, 0], [0, 1000, 0], [0, 0, 1000]])
Kd_swing = np.array([[20, 0, 0], [0, 10, 0], [0, 0, 10]])
Kp = np.array([[500, 0, 0], [0, 2000, 0], [0, 0, 2000]])
Kd = np.array([[20, 0, 0], [0, 10, 0], [0, 0, 20]])
torques = [0 for i in range(12)]
old_J_simple = calculate_J_simple(0.0, 0.0)


# 摆动腿力矩控制
def FR_set_swing_torque(phase_swing, swing_time, foot_traj, thetas, omegas, motors, J_simple, J_simple_dot):
    alpha1 = thetas[1]
    alpha2 = thetas[2]
    alpha1_dot = omegas[1]
    alpha2_dot = omegas[2]
    foot_traj.calculate_bezier(phase_swing, swing_time)
    p_ref = foot_traj.get_position()
    v_ref = foot_traj.get_velocity()
    a_ref = foot_traj.get_acceleration()
    # print("desired",p_ref.transpose())
    # print("real",p.transpose())
    a_simple = np.array([[-a_ref[2][0]], [a_ref[1][0]]])
    qdot = np.array([[alpha1_dot], [alpha2_dot]])
    Cqdot = calculate_Cqdot(alpha1, alpha2, alpha1_dot, alpha2_dot)
    G = calculate_G(alpha1, alpha2)
    B = calculate_B(alpha1, alpha2)
    # print(a_simple.transpose())
    tau = B.dot(np.linalg.inv(J_simple).dot((a_simple - J_simple_dot.dot(qdot)))) + Cqdot + G
    # print(tau.transpose())
    # 计算反馈力矩
    delta_p = p_ref - calculate_p(thetas[0], thetas[1], thetas[2], L1, L2)
    delta_v = v_ref - FR_get_foot_vel(omegas, thetas)
    T = calculate_j(thetas[0], thetas[1], thetas[2], L1, L2).transpose().dot(Kp_swing.dot(delta_p) + Kd_swing.dot(delta_v))

    # T = np.array([[0.0], [0.0], [0.0]])
    # print(T.transpose())
    # 执行力矩
    motors[0].setTorque(T[0][0])
    # motor1.setPosition(0)
    motors[1].setTorque(T[1][0] + tau[0][0])
    motors[2].setTorque(T[2][0] + tau[1][0])


def FL_set_swing_torque(phase_swing, swing_time, foot_traj, thetas, omegas, motors, J_simple, J_simple_dot):
    alpha1 = thetas[4]
    alpha2 = thetas[5]
    alpha1_dot = omegas[4]
    alpha2_dot = omegas[5]
    foot_traj.calculate_bezier(phase_swing, swing_time)
    p_ref = foot_traj.get_position()
    v_ref = foot_traj.get_velocity()
    a_ref = foot_traj.get_acceleration()
    # print("desired",p_ref.transpose())
    # print("real",p.transpose())
    a_simple = np.array([[-a_ref[2][0]], [a_ref[1][0]]])
    qdot = np.array([[alpha1_dot], [alpha2_dot]])
    Cqdot = calculate_Cqdot(alpha1, alpha2, alpha1_dot, alpha2_dot)
    G = calculate_G(alpha1, alpha2)
    B = calculate_B(alpha1, alpha2)
    # print(a_simple.transpose())
    tau = B.dot(np.linalg.inv(J_simple).dot((a_simple - J_simple_dot.dot(qdot)))) + Cqdot + G
    # print(tau.transpose())
    # 计算反馈力矩
    delta_p = p_ref - calculate_p(thetas[3], thetas[4], thetas[5], L1, L2)
    delta_v = v_ref - FR_get_foot_vel(omegas, thetas)
    T = calculate_j(thetas[3], thetas[4], thetas[5], L1, L2).transpose().dot(Kp_swing.dot(delta_p) + Kd_swing.dot(delta_v))

    # T = np.array([[0.0], [0.0], [0.0]])
    # print(T.transpose())
    # 执行力矩
    motors[3].setTorque(T[0][0])
    # motor1.setPosition(0)
    motors[4].setTorque(T[1][0] + tau[0][0])
    motors[5].setTorque(T[2][0] + tau[1][0])


def BR_set_swing_torque(phase_swing, swing_time, foot_traj, thetas, omegas, motors, J_simple, J_simple_dot):
    alpha1 = thetas[7]
    alpha2 = thetas[8]
    alpha1_dot = omegas[7]
    alpha2_dot = omegas[8]
    foot_traj.calculate_bezier(phase_swing, swing_time)
    p_ref = foot_traj.get_position()
    v_ref = foot_traj.get_velocity()
    a_ref = foot_traj.get_acceleration()
    # print("desired",p_ref.transpose())
    # print("real",p.transpose())
    a_simple = np.array([[-a_ref[2][0]], [a_ref[1][0]]])
    qdot = np.array([[alpha1_dot], [alpha2_dot]])
    Cqdot = calculate_Cqdot(alpha1, alpha2, alpha1_dot, alpha2_dot)
    G = calculate_G(alpha1, alpha2)
    B = calculate_B(alpha1, alpha2)
    # print(a_simple.transpose())
    tau = B.dot(np.linalg.inv(J_simple).dot((a_simple - J_simple_dot.dot(qdot)))) + Cqdot + G
    # print(tau.transpose())
    # 计算反馈力矩
    delta_p = p_ref - calculate_p(thetas[6], thetas[7], thetas[8], L1, L2)
    delta_v = v_ref - FR_get_foot_vel(omegas, thetas)
    T = calculate_j(thetas[6], thetas[7], thetas[8], L1, L2).transpose().dot(Kp_swing.dot(delta_p) + Kd_swing.dot(delta_v))

    # T = np.array([[0.0], [0.0], [0.0]])
    # print(T.transpose())
    # 执行力矩
    motors[6].setTorque(T[0][0])
    # motor1.setPosition(0)
    motors[7].setTorque(T[1][0] + tau[0][0])
    motors[8].setTorque(T[2][0] + tau[1][0])


def BL_set_swing_torque(phase_swing, swing_time, foot_traj, thetas, omegas, motors, J_simple, J_simple_dot):
    alpha1 = thetas[10]
    alpha2 = thetas[11]
    alpha1_dot = omegas[10]
    alpha2_dot = omegas[11]
    foot_traj.calculate_bezier(phase_swing, swing_time)
    p_ref = foot_traj.get_position()
    v_ref = foot_traj.get_velocity()
    a_ref = foot_traj.get_acceleration()
    # print("desired",p_ref.transpose())
    # print("real",p.transpose())
    a_simple = np.array([[-a_ref[2][0]], [a_ref[1][0]]])
    qdot = np.array([[alpha1_dot], [alpha2_dot]])
    Cqdot = calculate_Cqdot(alpha1, alpha2, alpha1_dot, alpha2_dot)
    G = calculate_G(alpha1, alpha2)
    B = calculate_B(alpha1, alpha2)
    # print(a_simple.transpose())
    tau = B.dot(np.linalg.inv(J_simple).dot((a_simple - J_simple_dot.dot(qdot)))) + Cqdot + G
    # print(tau.transpose())
    # 计算反馈力矩
    delta_p = p_ref - calculate_p(thetas[9], thetas[10], thetas[11], L1, L2)
    delta_v = v_ref - FR_get_foot_vel(omegas, thetas)
    T = calculate_j(thetas[9], thetas[10], thetas[11], L1, L2).transpose().dot(Kp_swing.dot(delta_p) + Kd_swing.dot(delta_v))

    # T = np.array([[0.0], [0.0], [0.0]])
    # print(T.transpose())
    # 执行力矩
    motors[9].setTorque(T[0][0])
    # motor1.setPosition(0)
    motors[10].setTorque(T[1][0] + tau[0][0])
    motors[11].setTorque(T[2][0] + tau[1][0])


# 设置足端力的大小
def FR_set_foot_force(F, thetas, motors):
    T_stand = calculate_j(thetas[0], thetas[1], thetas[2], 0.2, 0.2).transpose().dot(F)
    motors[0].setTorque(T_stand[0][0])
    motors[1].setTorque(T_stand[1][0])
    motors[2].setTorque(T_stand[2][0])


def FL_set_foot_force(F, thetas, motors):
    T_stand = calculate_j(thetas[3], thetas[4], thetas[5], L1, L2).transpose().dot(F)
    motors[3].setTorque(T_stand[0][0])
    motors[4].setTorque(T_stand[1][0])
    motors[5].setTorque(T_stand[2][0])


def BR_set_foot_force(F, thetas, motors):
    T_stand = calculate_j(thetas[6], thetas[7], thetas[8], L1, L2).transpose().dot(F)
    motors[6].setTorque(T_stand[0][0])
    motors[7].setTorque(T_stand[1][0])
    motors[8].setTorque(T_stand[2][0])


def BL_set_foot_force(F, thetas, motors):
    T_stand = calculate_j(thetas[9], thetas[10], thetas[11], L1, L2).transpose().dot(F)
    motors[9].setTorque(T_stand[0][0])
    motors[10].setTorque(T_stand[1][0])
    motors[11].setTorque(T_stand[2][0])


def set_foots_position_and_force(p, f, thetas, omegas, motors):
    FR_set_foot_position_and_force(np.array([[p[0]], [p[1]], [p[2]]]), np.array([[f[0]], [f[1]], [f[2]]]),
                                   thetas, omegas, motors)
    FL_set_foot_position_and_force(np.array([[p[3]], [p[4]], [p[5]]]), np.array([[f[3]], [f[4]], [f[5]]]),
                                   thetas, omegas, motors)
    BR_set_foot_position_and_force(np.array([[p[6]], [p[7]], [p[8]]]), np.array([[f[6]], [f[7]], [f[8]]]),
                                   thetas, omegas, motors)
    BL_set_foot_position_and_force(np.array([[p[9]], [p[10]], [p[11]]]), np.array([[f[9]], [f[10]], [f[11]]]),
                                   thetas, omegas, motors)


# 设置足端位置,通过一个pd控制器
def FR_set_foot_position_and_force(P, F, thetas, omegas, motors):
    # P for desired postiion
    # F for desired forces
    delta_p = P - calculate_p(thetas[0], thetas[1], thetas[2], L1, L2)
    delta_v = -FR_get_foot_vel(omegas, thetas)
    T = calculate_j(thetas[0], thetas[1], thetas[2], L1, L2).transpose().dot(Kp.dot(delta_p) + Kd.dot(delta_v))
    T_stand = calculate_j(thetas[0], thetas[1], thetas[2], L1, L2).transpose().dot(F)
    motors[0].setTorque(T[0][0] + T_stand[0][0])
    motors[1].setTorque(T[1][0] + T_stand[1][0])
    motors[2].setTorque(T[2][0] + T_stand[2][0])
    torques[0] = T[0][0] + T_stand[0][0]
    torques[1] = T[1][0] + T_stand[1][0]
    torques[2] = T[2][0] + T_stand[2][0]


def FL_set_foot_position_and_force(P, F, thetas, omegas, motors):
    delta_p = P - calculate_p(thetas[3], thetas[4], thetas[5], L1, L2)
    delta_v = -FL_get_foot_vel(omegas, thetas)
    T = calculate_j(thetas[3], thetas[4], thetas[5], L1, L2).transpose().dot(Kp.dot(delta_p) + Kd.dot(delta_v))
    T_stand = calculate_j(thetas[3], thetas[4], thetas[5], L1, L2).transpose().dot(F)
    motors[3].setTorque(T[0][0] + T_stand[0][0])
    motors[4].setTorque(T[1][0] + T_stand[1][0])
    motors[5].setTorque(T[2][0] + T_stand[2][0])
    torques[3] = T[0][0] + T_stand[0][0]
    torques[4] = T[1][0] + T_stand[1][0]
    torques[5] = T[2][0] + T_stand[2][0]


def BR_set_foot_position_and_force(P, F, thetas, omegas, motors):
    delta_p = P - calculate_p(thetas[6], thetas[7], thetas[8], L1, L2)
    delta_v = -BR_get_foot_vel(omegas, thetas)
    T = calculate_j(thetas[6], thetas[7], thetas[8], L1, L2).transpose().dot(Kp.dot(delta_p) + Kd.dot(delta_v))
    T_stand = calculate_j(thetas[6], thetas[7], thetas[8], L1, L2).transpose().dot(F)
    motors[6].setTorque(T[0][0] + T_stand[0][0])
    motors[7].setTorque(T[1][0] + T_stand[1][0])
    motors[8].setTorque(T[2][0] + T_stand[2][0])
    torques[6] = T[0][0] + T_stand[0][0]
    torques[7] = T[1][0] + T_stand[1][0]
    torques[8] = T[2][0] + T_stand[2][0]


def BL_set_foot_position_and_force(P, F, thetas, omegas, motors):
    delta_p = P - calculate_p(thetas[9], thetas[10], thetas[11], L1, L2)
    delta_v = -BL_get_foot_vel(omegas, thetas)
    T = calculate_j(thetas[9], thetas[10], thetas[11], L1, L2).transpose().dot(Kp.dot(delta_p) + Kd.dot(delta_v))
    T_stand = calculate_j(thetas[9], thetas[10], thetas[11], L1, L2).transpose().dot(F)
    motors[9].setTorque(T[0][0] + T_stand[0][0])
    motors[10].setTorque(T[1][0] + T_stand[1][0])
    motors[11].setTorque(T[2][0] + T_stand[2][0])
    torques[9] = T[0][0] + T_stand[0][0]
    torques[10] = T[1][0] + T_stand[1][0]
    torques[11] = T[2][0] + T_stand[2][0]


def FR_get_foot_vel(omegas, thetas):
    W = np.array([[omegas[0]], [omegas[1]], [omegas[2]]])
    return calculate_j(thetas[0], thetas[1], thetas[2], L1, L2).dot(W)


def FL_get_foot_vel(omegas, thetas):
    W = np.array([[omegas[3]], [omegas[4]], [omegas[5]]])
    return calculate_j(thetas[3], thetas[4], thetas[5], L1, L2).dot(W)


def BR_get_foot_vel(omegas, thetas):
    W = np.array([[omegas[6]], [omegas[7]], [omegas[8]]])
    return calculate_j(thetas[6], thetas[7], thetas[8], L1, L2).dot(W)


def BL_get_foot_vel(omegas, thetas):
    W = np.array([[omegas[9]], [omegas[10]], [omegas[11]]])
    return calculate_j(thetas[9], thetas[10], thetas[11], L1, L2).dot(W)
