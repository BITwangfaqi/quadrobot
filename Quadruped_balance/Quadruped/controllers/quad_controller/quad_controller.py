"""quad_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import numpy as np
from leg_controller import *
from FSM import *
import bezier

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

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
print("仿真与控制的频率为", 1000 / timestep)

# 获得所以的电机对象，在一个12的列表
motors = init_motors(robot)
# 获得所以的编码器对象，在一个12的列表
encoders = init_encoders(robot)
# 获得所以的力传感器对象，在一个4的列表
forcers = init_forcers(robot)
# 使能电机返回力矩
enablue_motors_feedback(motors)

jabocians = [calculate_J_simple(0.0, 0.0), calculate_J_simple(0.0, 0.0), calculate_J_simple(0.0, 0.0),
             calculate_J_simple(0.0, 0.0)]
old_jabocians = [calculate_J_simple(0.0, 0.0), calculate_J_simple(0.0, 0.0), calculate_J_simple(0.0, 0.0),
                 calculate_J_simple(0.0, 0.0)]

old_thetas = [0 for i in range(12)]

initialized = 0



# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # 下面的计算得到的第一个数值是突变的
    sim_time = robot.getTime()  # 获取仿真时间
    thetas = get_encoders_value(encoders)
    omegas = get_omegas(thetas, old_thetas)
    old_thetas = thetas
    jabocians = get_simple_jabocians(thetas)
    jabocians_dot = get_simple_jabocians_dot(jabocians, old_jabocians)
    old_jabocians = jabocians
    #------------------------------------------------------------
    if initialized == 0:

        FR_p0 = calculate_p(thetas[0], thetas[1], thetas[2], L1, L2)
        FR_pf = FR_p0 + np.array([[0.0], [0.05], [0.0]])
        FR_foot_traj = bezier.Bezier(FR_p0, FR_pf, 0.15)
        FL_p0 = calculate_p(thetas[3], thetas[4], thetas[5], L1, L2)
        FL_pf = FL_p0 + np.array([[0.0], [0.05], [0.0]])
        FL_foot_traj = bezier.Bezier(FL_p0, FL_pf, 0.15)
        BR_p0 = calculate_p(thetas[6], thetas[7], thetas[8], L1, L2)
        BR_pf = BR_p0 + np.array([[0.0], [0.05], [0.0]])
        BR_foot_traj = bezier.Bezier(BR_p0, BR_pf, 0.15)
        BL_p0 = calculate_p(thetas[9], thetas[10], thetas[11], L1, L2)
        BL_pf = BL_p0 + np.array([[0.0], [0.05], [0.0]])
        BL_foot_traj = bezier.Bezier(BL_p0, BL_pf, 0.15)
        foot_trajs = [FR_foot_traj, FL_foot_traj, BR_foot_traj, BL_foot_traj]
        initialized = 1

    if sim_time < 3:
        FSM_run('stand', sim_time, thetas, omegas, motors, foot_trajs, jabocians, jabocians_dot)
    else:
        FSM_run('trot', sim_time, thetas, omegas, motors, foot_trajs, jabocians, jabocians_dot)

    # -------------------------测试摆动的代码--------------------------
    # if sim_time > 3 and initialized == 0:
    #     p0 = calculate_p(thetas[0], thetas[1], thetas[2], L1, L2)
    #     pf = p0 + np.array([[0.0], [0.1], [-0.03]])
    #     FR_foot_traj = bezier.Bezier(p0, pf, 0.2)
    #     FL_foot_traj = bezier.Bezier(p0, pf, 0.2)
    #     BR_foot_traj = bezier.Bezier(p0, pf, 0.2)
    #     BL_foot_traj = bezier.Bezier(p0, pf, 0.2)
    #     initialized = 1
    #
    # # set_foots_position_and_force(def_pos, def_force, thetas, omegas, motors) # 类似弹簧的力位控制实现站立
    # if sim_time > 3 and sim_time < 3.3:
    #     phase_swing = (sim_time - 3) / 0.3
    #     swing_time = 0.3
    #     FR_set_swing_torque(phase_swing, swing_time, FR_foot_traj, thetas, omegas, motors, jabocians[0],
    #                         jabocians_dot[0])
    #     FL_set_swing_torque(phase_swing, swing_time, FL_foot_traj, thetas, omegas, motors, jabocians[1],
    #                         jabocians_dot[1])
    #     BR_set_swing_torque(phase_swing, swing_time, BR_foot_traj, thetas, omegas, motors, jabocians[2],
    #                         jabocians_dot[2])
    #     BL_set_swing_torque(phase_swing, swing_time, BL_foot_traj, thetas, omegas, motors, jabocians[3],
    #                         jabocians_dot[3])
    # elif sim_time >= 3.3:
    #     FR_set_foot_position_and_force(pf, np.array([[0.0], [0.0], [0.0]]), thetas, omegas, motors)

    # ------------------------------打印必要的信息------------------------------------------------
    if math.modf(sim_time / 0.01)[0] == 0:
        # x1 = forcers[0].getValues()
        # x2 = forcers[1].getValues()
        # x3 = forcers[2].getValues()
        # x4 = forcers[3].getValues()
        # print("forcer", x1[0], x1[1],x1[2])
        # print(FR_height)
        # print(torques)

        pass

# Enter here exit cleanup code.
