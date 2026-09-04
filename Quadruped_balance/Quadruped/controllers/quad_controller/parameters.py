import math
import sys
import os

# 将工作空间根目录加入到sys.path中以便能够找到mypackage和acados_test
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

# Webots控制器导入
try:
    import controller
except ImportError:
    sys.path.append(os.path.join(os.environ.get("WEBOTS_HOME", "/usr/local/webots"), "lib", "controller", "python"))
    import controller


from controller import Robot, Motor
import numpy as np
from calculation import *


# 获得所以的电机对象，在一个4*3的矩阵中
def init_motors(robot):
    # 获得电机对象
    FR_abad_motor = robot.getMotor('FR_abad_motor')
    FR_hip_motor = robot.getMotor('FR_hip_motor')
    FR_knee_motor = robot.getMotor('FR_knee_motor')
    FL_abad_motor = robot.getMotor('FL_abad_motor')
    FL_hip_motor = robot.getMotor('FL_hip_motor')
    FL_knee_motor = robot.getMotor('FL_knee_motor')
    BR_abad_motor = robot.getMotor('BR_abad_motor')
    BR_hip_motor = robot.getMotor('BR_hip_motor')
    BR_knee_motor = robot.getMotor('BR_knee_motor')
    BL_abad_motor = robot.getMotor('BL_abad_motor')
    BL_hip_motor = robot.getMotor('BL_hip_motor')
    BL_knee_motor = robot.getMotor('BL_knee_motor')
    return [FR_abad_motor, FR_hip_motor, FR_knee_motor,
            FL_abad_motor, FL_hip_motor, FL_knee_motor,
            BR_abad_motor, BR_hip_motor, BR_knee_motor,
            BL_abad_motor, BL_hip_motor, BL_knee_motor]


# 获得所以的编码器对象，在一个4*3的矩阵中
def init_encoders(robot):
    # 得到关节编码器并使能
    FR_abad_encoder = robot.getPositionSensor('FR_abad_encoder')
    FR_hip_encoder = robot.getPositionSensor('FR_hip_encoder')
    FR_knee_encoder = robot.getPositionSensor('FR_knee_encoder')
    FL_abad_encoder = robot.getPositionSensor('FL_abad_encoder')
    FL_hip_encoder = robot.getPositionSensor('FL_hip_encoder')
    FL_knee_encoder = robot.getPositionSensor('FL_knee_encoder')
    BR_abad_encoder = robot.getPositionSensor('BR_abad_encoder')
    BR_hip_encoder = robot.getPositionSensor('BR_hip_encoder')
    BR_knee_encoder = robot.getPositionSensor('BR_knee_encoder')
    BL_abad_encoder = robot.getPositionSensor('BL_abad_encoder')
    BL_hip_encoder = robot.getPositionSensor('BL_hip_encoder')
    BL_knee_encoder = robot.getPositionSensor('BL_knee_encoder')
    FR_abad_encoder.enable(1)
    FR_hip_encoder.enable(1)
    FR_knee_encoder.enable(1)
    FL_abad_encoder.enable(1)
    FL_hip_encoder.enable(1)
    FL_knee_encoder.enable(1)
    BR_abad_encoder.enable(1)
    BR_hip_encoder.enable(1)
    BR_knee_encoder.enable(1)
    BL_abad_encoder.enable(1)
    BL_hip_encoder.enable(1)
    BL_knee_encoder.enable(1)
    return [FR_abad_encoder, FR_hip_encoder, FR_knee_encoder,
            FL_abad_encoder, FL_hip_encoder, FL_knee_encoder,
            BR_abad_encoder, BR_hip_encoder, BR_knee_encoder,
            BL_abad_encoder, BL_hip_encoder, BL_knee_encoder]


# 获得所以的力传感器对象，在一个1*4的矩阵中
def init_forcers(robot):
    # 得到足端力传感器对象并使能
    FR_foot_sensor = robot.getTouchSensor('FR_foot_sensor')
    FL_foot_sensor = robot.getTouchSensor('FL_foot_sensor')
    BR_foot_sensor = robot.getTouchSensor('BR_foot_sensor')
    BL_foot_sensor = robot.getTouchSensor('BL_foot_sensor')
    FR_foot_sensor.enable(1)
    FL_foot_sensor.enable(1)
    BR_foot_sensor.enable(1)
    BL_foot_sensor.enable(1)
    return [FR_foot_sensor, FL_foot_sensor, BR_foot_sensor, BL_foot_sensor]

def enablue_motors_feedback(motors):
    for i in motors:
        i.enableTorqueFeedback(1)

def get_encoders_value(encoders):
    thetas = [0 for i in range(12)]
    for i in range(len(encoders)):
        thetas[i] = encoders[i].getValue()
    return thetas


def get_forcers_value(forcers):
    forces = [0 for i in range(4)]
    for i in range(4):
        forces[i] = forcers[i].getValues()
    return forces


def get_motors_torque(motors):
    torques = [0 for i in range(12)]
    for i in range(12):
        torques[i] = motors[i].getTorqueFeedback()
    return torques

def get_omegas(thetas,old_thetas):
    omegas = [0 for i in range(12)]
    for i in range(12):
        omegas[i]=(thetas[i]-old_thetas[i])/0.001
    return omegas

def get_simple_jabocians(thetas):
    return [calculate_J_simple(thetas[1], thetas[2]),
            calculate_J_simple(thetas[4],thetas[5]),
            calculate_J_simple(thetas[7],thetas[8]),
            calculate_J_simple(thetas[10],thetas[11])]

def get_simple_jabocians_dot(jabocians,old_jabocians):
    return [jabocians[0]-old_jabocians[0],
            jabocians[1]-old_jabocians[1],
            jabocians[2]-old_jabocians[2],
            jabocians[3]-old_jabocians[3],]
# 定义每条腿的参数
# L1是大腿长度，L2是小腿长度
d = 0.08
L1 = 0.2
L2 = 0.2

# alpha系列角度是为了方便前馈力矩的计算
FR_alpha1 = 0.0
FR_alpha2 = 0.0
FR_alpha1_dot = 0.0
FR_alpha2_dot = 0.0
FL_alpha1 = 0.0
FL_alpha2 = 0.0
FL_alpha1_dot = 0.0
FL_alpha2_dot = 0.0
BR_alpha1 = 0.0
BR_alpha2 = 0.0
BR_alpha1_dot = 0.0
BR_alpha2_dot = 0.0
BL_alpha1 = 0.0
BL_alpha2 = 0.0
BL_alpha1_dot = 0.0
BL_alpha2_dot = 0.0

# 比例矩阵
Kp = np.array([[1000, 0, 0], [0, 100, 0], [0, 0, 100]])
Kd = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
