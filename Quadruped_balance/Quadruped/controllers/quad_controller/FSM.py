from leg_controller import *

# 工作空间，足端力
FR_def_force = np.array([[0.0], [0.0], [-64.0]])
FL_def_force = np.array([[0.0], [0.0], [-64.0]])
BR_def_force = np.array([[0.0], [0.0], [-64.0]])
BL_def_force = np.array([[0.0], [0.0], [-64.0]])
def_force = [0.0, 0.0, -30.0, 0.0, 0.0, -30.0, 0.0, 0.0, -30.0, 0.0, 0.0, -30.0]

# 足端位置
FR_def_pos = np.array([[0.0], [0.0], [-0.25]])
FL_def_pos = np.array([[-0.0], [0.0], [-0.25]])
BR_def_pos = np.array([[0.0], [0.0], [-0.25]])
BL_def_pos = np.array([[-0.0], [0.0], [-0.25]])
def_pos = [0.0, 0.0, -0.25, -0.0, 0.0, -0.25, 0.0, 0.0, -0.25, -0.0, 0.0, -0.25]

switch_flag = 0
switch_time = 0.0
last_state = 'stand'  # default to be stand
swing_time = 0.3
period_time = 0.6



def FSM_run(state, time, thetas, omegas, motors, foot_trajs, jabocians, jabocians_dot):
    global switch_flag, switch_time, last_state, swing_time, period_time,p0
    if state != last_state:
        last_state = state
        switch_time = time
        switch_flag = 1

    if state == 'stand':  #
        set_foots_position_and_force(def_pos, def_force, thetas, omegas, motors)  # 类似弹簧的力位控制实现站立
        pass

    elif state == 'trot':
        phase_total = math.modf((time - switch_time) / period_time)[0]
        FL_phase_total = phase_total
        FR_phase_total = math.modf((time+period_time/2 - switch_time) / period_time)[0]
        BR_phase_total = phase_total
        BL_phase_total = math.modf((time+period_time/2 - switch_time) / period_time)[0]

        if FR_phase_total < (swing_time / period_time):  # 摆动相位时
            FR_phase_swing = period_time * FR_phase_total / swing_time
            FR_set_swing_torque(FR_phase_swing, swing_time, foot_trajs[0], thetas, omegas, motors, jabocians[0],
                                jabocians_dot[0])
        else:
            # FR_set_foot_force(FR_def_force, thetas, motors)
            FR_set_foot_position_and_force(FR_def_pos, FR_def_force, thetas, omegas, motors)
            # FR_set_foot_position_and_force(np.array([[0.0],[0.0],[-0.3]]), np.array([[0.0], [0.0], [0.0]]), thetas, omegas, motors)


        if FL_phase_total < (swing_time / period_time):  # 摆动相位时
            FL_phase_swing = period_time * FL_phase_total / swing_time
            FL_set_swing_torque(FL_phase_swing, swing_time, foot_trajs[1], thetas, omegas, motors, jabocians[1],
                                jabocians_dot[1])
        else:
            # FL_set_foot_force(FL_def_force, thetas, motors)
            FL_set_foot_position_and_force(FL_def_pos, FL_def_force, thetas, omegas, motors)


        if BR_phase_total < (swing_time / period_time):  # 摆动相位时
            BR_phase_swing = period_time * BR_phase_total / swing_time
            BR_set_swing_torque(BR_phase_swing, swing_time, foot_trajs[2], thetas, omegas, motors, jabocians[2],
                                jabocians_dot[2])
        else:
            # BR_set_foot_force(BR_def_force, thetas, motors)
            BR_set_foot_position_and_force(BR_def_pos, BR_def_force, thetas, omegas, motors)


        if BL_phase_total < (swing_time / period_time):  # 摆动相位时
            BL_phase_swing = period_time * BL_phase_total / swing_time
            BL_set_swing_torque(BL_phase_swing, swing_time, foot_trajs[3], thetas, omegas, motors, jabocians[3],
                                jabocians_dot[3])
        else:
            # BL_set_foot_force(BL_def_force, thetas, motors)
            BL_set_foot_position_and_force(BL_def_pos, BL_def_force, thetas, omegas, motors)
        pass
    else:
        pass
