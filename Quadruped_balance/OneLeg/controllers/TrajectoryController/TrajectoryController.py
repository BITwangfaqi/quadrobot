"""TrajectoryController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Motor
from controller import TouchSensor
import math
import numpy as np
import bezier
import ffTorque
import time

# create the Robot instance.机器人对象
robot = Robot()

# get the time step of the current world.控制频率与软件仿真频率相等
timestep = int(robot.getBasicTimeStep())
print("仿真与控制的频率为", 1000 / timestep)

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getMotor('motorname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)
# 得到电机对象
motor1 = robot.getMotor('motor1')
motor2 = robot.getMotor('motor2')
motor3 = robot.getMotor('motor3')
# 得到足端力传感器对象并使能
footSensor = robot.getTouchSensor('FootSensor')
footSensor.enable(1)
# 得到关节编码器并使能
encoderSensor1 = robot.getPositionSensor('encoder1')
encoderSensor2 = robot.getPositionSensor('encoder2')
encoderSensor3 = robot.getPositionSensor('encoder3')
encoderSensor1.enable(1)
encoderSensor2.enable(1)
encoderSensor3.enable(1)

i = 0
# 定义三个角度
# 依次是ab/ad,hip,knee
theta1 = 0.0
theta2 = 0.0
theta3 = 0.0

# alpha系列角度是为了方便前馈力矩的计算
alpha1 = 0.0
alpha2 = 0.0
alpha1_dot = 0.0
alpha2_dot = 0.0
# 定义每条腿的参数
# L1是大腿长度，L2是小腿长度
d = 0.08
L1 = 0.2
L2 = 0.2


# b11 = I1 + m1 * l1 * l1 + I2 + m2 * (a1 * a1 + l2 * l2 + 2 * a1 * l2 * math.cos(alpha2))
# b12 = I2 + m2 * (l2 * l2 + a1 * l2 * math.cos(alpha2))
# b21 = b12
# b22 = I2 + m2 * l2 * l2
# B = np.array([[b11, b12], [b21, b22]])
# h = -m2 * a1 * l2 * math.sin(alpha2)
# C = np.array([[h * alpha2_dot, h * (alpha1_dot + alpha2_dot)], [-h * alpha2_dot, 0]])
# g1 = m1 * g * l1 * math.sin(alpha1) + m2 * g * (a1 * math.sin(alpha1) + l2 * math.sin(alpha1 + alpha2))
# g2 = m2 * g * l2 * math.sin(alpha1 + alpha2)
# J_simple = np.array([[-a1 * math.sin(alpha1) - a2 * math.sin(alpha1 + alpha2), -a2 * math.sin(alpha1 + alpha2)],
#                      [a1 * math.cos(alpha1) + a2 * math.cos(alpha1 + alpha2), a2 * math.cos(alpha1 + alpha2)]])


# 雅可比矩阵，只有线速度部分的映射关系
def calculate_j(theta1, theta2, theta3, L1, L2):
    return np.array([[-math.cos(theta1) * (L2 * math.cos(theta2 + theta3) + L1 * math.cos(theta2)),
                      math.sin(theta1) * (L2 * math.sin(theta2 + theta3) + L1 * math.sin(theta2)),
                      L2 * math.sin(theta2 + theta3) * math.sin(theta1)],
                     [0.0, L2 * math.cos(theta2 + theta3) + L1 * math.cos(theta2), L2 * math.cos(theta2 + theta3)],
                     [math.sin(theta1) * (L2 * math.cos(theta2 + theta3) + L1 * math.cos(theta2)),
                      math.cos(theta1) * (L2 * math.sin(theta2 + theta3) + L1 * math.sin(theta2)),
                      L2 * math.sin(theta2 + theta3) * math.cos(theta1)]])


# 工作空间，足端力
F = np.array([[0.0], [0.0], [-10.0]])

# 关节空间，电机力矩
T = np.array([[0.0], [0.0], [0.0]])


# 获取足端位置，参考坐标系为腿坐标系
def calculate_p(theta1, theta2, theta3, L1, L2):
    return np.array([[-L1 * math.cos(theta2) * math.sin(theta1) - L2 * math.cos(theta2) * math.cos(theta3) * math.sin(
        theta1) + L2 * math.sin(theta1) * math.sin(theta2) * math.sin(theta3)],
                     [L2 * math.sin(theta2 + theta3) + L1 * math.sin(theta2)],
                     [-math.cos(theta1) * (L2 * math.cos(theta2 + theta3) + L1 * math.cos(theta2))]])


# 参考足端位置
p_ref = np.array([[0.0], [0.05], [-0.3]])
v_ref = np.array([[0.0], [0.0], [0.0]])

# 比例矩阵
Kp = np.array([[1000, 0, 0], [0, 100, 0], [0, 0, 100]])
Kd = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

initialized = 0
delay = 2

# time.sleep(3)
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    # 获取仿真时间
    sim_time = robot.getTime()
    if sim_time <= delay:
        if initialized == 0:
            theta1 = encoderSensor1.getValue()
            theta2 = encoderSensor2.getValue()
            theta3 = encoderSensor3.getValue()
            old_theta1 = theta1
            old_theta2 = theta2
            old_theta3 = theta3
            omega1 = theta1 - old_theta1
            omega2 = theta2 - old_theta2
            omega3 = theta3 - old_theta3
            alpha1 = theta2
            alpha2 = theta3
            alpha1_dot = omega2
            alpha2_dot = omega3
            B = ffTorque.calculate_B(alpha1, alpha2)
            Cqdot = ffTorque.calculate_Cqdot(alpha1, alpha2, alpha1_dot, alpha2_dot)
            G = ffTorque.calculate_G(alpha1, alpha2)
            J_simple = ffTorque.calculate_J_simple(alpha1, alpha2)
            old_J_simple = J_simple
            J_simple_dot = J_simple - old_J_simple
            # 贝塞尔轨迹
            # p0,pf都是列向量
            p0 = calculate_p(theta1, theta2, theta3, L1, L2)
            pf = p0 + np.array([[0.0], [0.2], [-0.03]])
            height = 0.15
            print("起脚点坐标:", p0.transpose())
            print("落地点坐标", pf.transpose())
            print("抬脚高度", height)
            foot_traj = bezier.Bezier(p0, pf, height)  # 空中摆动轨迹对象
            foot_stand = bezier.Bezier(pf, p0, 0)  # 无支撑时摆动轨迹
            period_time = 0.6
            swing_time = period_time / 2
            stand_time = period_time / 2
            phase_total = 0
            last_phase_total = 0
            touched = 0
            touch_lock = 0

            initialized = 1

        else:
            pass

    else:
        theta1 = encoderSensor1.getValue()
        theta2 = encoderSensor2.getValue()
        theta3 = encoderSensor3.getValue()
        omega1 = theta1 - old_theta1
        omega2 = theta2 - old_theta2
        omega3 = theta3 - old_theta3
        # 获得足端延小腿方向的力
        foot_force = robot.getTouchSensor('FootSensor').getValues()[2]
        if foot_force < -1.0:
            # print("touched")
            touched = 1
        else:
            touched = 0

        W = np.array([[omega1], [omega2], [omega3]])

        alpha1 = theta2
        alpha2 = theta3
        alpha1_dot = omega2
        alpha2_dot = omega3
        qdot = np.array([[alpha1_dot], [alpha2_dot]])
        B = ffTorque.calculate_B(alpha1, alpha2)
        Cqdot = ffTorque.calculate_Cqdot(alpha1, alpha2, alpha1_dot, alpha2_dot)
        G = ffTorque.calculate_G(alpha1, alpha2)
        J_simple = ffTorque.calculate_J_simple(alpha1, alpha2)
        J_simple_dot = J_simple - old_J_simple
        # 根据末端参考力计算关节力矩
        # T = calculate_j(theta1, theta2, theta3, L1, L2).transpose().dot(F)
        # motor1.setTorque(T[0][0])
        # motor2.setTorque(T[1][0])
        # motor3.setTorque(T[2][0])
        # 计算足端速度
        v = calculate_j(theta1, theta2, theta3, L1, L2).dot(W)
        # 计算足端位置
        p = calculate_p(theta1, theta2, theta3, L1, L2)

        old_theta1 = theta1
        old_theta2 = theta2
        old_theta3 = theta3
        old_J_simple = J_simple

        # 进行五个全周期的计算
        if sim_time < (delay + 5 * period_time):
            phase_total = math.modf((sim_time - delay) / period_time)[0]
            # 每次抬腿的时候要重新设置起脚点
            if (last_phase_total - phase_total) > 0.5:
                p0 = calculate_p(theta1, theta2, theta3, L1, L2)
                foot_traj = bezier.Bezier(p0, pf, height)
                touch_lock = 0
                # print("new foot_traj")
                # print("起脚点坐标:", p0.transpose())
                # print("落地点坐标", pf.transpose())

            last_phase_total = phase_total
            if phase_total < (swing_time / period_time):  # 摆动相位时
                phase_swing = period_time * phase_total / swing_time
                if (phase_swing > 0.5) and ((touched == 1) or (touch_lock == 1)):  # 只有在腿下落且触地时才支撑
                    touch_lock = 1
                    T_stand = calculate_j(theta1, theta2, theta3, L1, L2).transpose().dot(F)
                    # print("stand force")
                    motor1.setTorque(T_stand[0][0])
                    motor2.setTorque(T_stand[1][0])
                    motor3.setTorque(T_stand[2][0])

                else:  # 其他情况不管有没有触地，都继续摆动
                    foot_traj.calculate_bezier(phase_swing, swing_time)
                    p_ref = foot_traj.get_position()
                    a_ref = foot_traj.get_acceleration()
                    # print("desired",p_ref.transpose())
                    # print("real",p.transpose())
                    a_simple = np.array([[-a_ref[2][0]], [a_ref[1][0]]])
                    # print(a_simple.transpose())
                    tau = B.dot(np.linalg.inv(J_simple).dot((a_simple - J_simple_dot.dot(qdot)))) + Cqdot + G
                    # print(tau.transpose())
                    # 计算反馈力矩
                    T = calculate_j(theta1, theta2, theta3, L1, L2).transpose().dot(
                        Kp.dot(p_ref - p) + Kd.dot(v_ref - v))
                    # T = np.array([[0.0], [0.0], [0.0]])
                    # print(T.transpose())
                    # 执行力矩
                    motor1.setTorque(T[0][0])
                    # motor1.setPosition(0)
                    motor2.setTorque(T[1][0] + tau[0][0])
                    motor3.setTorque(T[2][0] + tau[1][0])
                    # print(T[0][0], T[1][0] + tau[0][0], T[2][0] + tau[1][0])
            else:  # 支撑相位时
                if touched == 0:  # 如果没触地，就继续摆动
                    phase_stand = (period_time * phase_total - swing_time) / stand_time
                    foot_stand.calculate_bezier(phase_stand, stand_time)
                    p_ref = foot_stand.get_position()
                    a_ref = foot_stand.get_acceleration()
                    # print("desired",p_ref.transpose())
                    # print("real",p.transpose())
                    a_simple = np.array([[-a_ref[2][0]], [a_ref[1][0]]])
                    # print(a_simple.transpose())
                    tau = B.dot(np.linalg.inv(J_simple).dot((a_simple - J_simple_dot.dot(qdot)))) + Cqdot + G
                    # print(tau.transpose())
                    # 计算反馈力矩
                    T = calculate_j(theta1, theta2, theta3, L1, L2).transpose().dot(
                        Kp.dot(p_ref - p) + Kd.dot(v_ref - v))
                    # T = np.array([[0.0], [0.0], [0.0]])
                    # print(T.transpose())
                    # 执行力矩
                    motor1.setTorque(T[0][0])
                    # motor1.setPosition(0)
                    motor2.setTorque(T[1][0] + tau[0][0])
                    motor3.setTorque(T[2][0] + tau[1][0])
                    # print(T[0][0], T[1][0] + tau[0][0], T[2][0] + tau[1][0])
                elif touched == 1 or touch_lock == 1:  # 如果触地，就支撑
                    touch_lock = 1
                    T_stand = calculate_j(theta1, theta2, theta3, L1, L2).transpose().dot(F)
                    motor1.setTorque(T_stand[0][0])
                    motor2.setTorque(T_stand[1][0])
                    motor3.setTorque(T_stand[2][0])
                    # print(T_stand[0][0],T_stand[1][0],T_stand[2][0])


        else:
            tau = np.array([[0.0], [0.0]])
            motor1.setPosition(theta1)
            motor2.setPosition(theta2)
            motor3.setPosition(theta3)

        if math.modf(sim_time / 0.1)[0] == 0:
            # print("simulation time:", sim_time)
            # print("phase:", phase)
            # forceXYZ = robot.getTouchSensor('FootSensor').getValues()
            #
            # totalForce = math.sqrt(forceXYZ[0] * forceXYZ[0] + forceXYZ[1] * forceXYZ[1] + forceXYZ[2] * forceXYZ[2])
            # print("Angle vector:", theta1, theta2, theta3)
            # print("foot position", p.transpose())
            # print("Torque vector:", T[0][0], T[1][0], T[2][0])
            # print("Force Vector:", forceXYZ)
            # print("Total force:", totalForce)
            pass

# Enter here exit cleanup code.
