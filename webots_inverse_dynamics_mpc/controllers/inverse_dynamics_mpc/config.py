"""Configuration for the single-stage inverted-pendulum inverse-dynamics MPC demo."""

# -----------------------------
# Webots / plant parameters
# -----------------------------
BASIC_TIME_STEP_MS = 10
MPC_PERIOD_S = 0.02  # 50 Hz MPC update

# Cart-pole parameters. These values must match the .wbt world.
CART_MASS = 1.0               # kg
POLE_MASS = 0.20              # kg
POLE_LENGTH = 0.60            # m, full pole length
POLE_WIDTH = 0.04             # m, x/y box width in the Webots model
GRAVITY = 9.81                 # m/s^2
POLE_COM_LENGTH = POLE_LENGTH / 2.0

# For the Webots pole bounding box (0.04 x 0.04 x 0.60), the moment of inertia
# about the hinge axis y through the pole COM is m * (width_x^2 + length_z^2) / 12.
POLE_I_COM_Y = POLE_MASS * (POLE_WIDTH**2 + POLE_LENGTH**2) / 12.0
# Parallel-axis theorem, inertia about the hinge pivot.
POLE_I_PIVOT_Y = POLE_I_COM_Y + POLE_MASS * POLE_COM_LENGTH**2

FORCE_MAX = 40.0              # N, matches LinearMotor.maxForce
CART_POS_LIMIT = 1.80         # m, kept inside SliderJoint stops +/- 2 m
CART_VEL_LIMIT = 6.0          # m/s
POLE_RATE_LIMIT = 15.0        # rad/s

# Initial pole angle used in the .wbt world.
INITIAL_POLE_ANGLE = 0.08      # rad

# -----------------------------
# MPC discretization
# -----------------------------
N = 20
DT_MIN = 0.020                 # s
DT_MAX = 0.045                 # s

# -----------------------------
# MPC objective weights
# state = [cart position, pole angle, cart velocity, pole angular velocity]
# -----------------------------
Q_X = 30.0
Q_THETA = 650.0
Q_X_DOT = 3.0
Q_THETA_DOT = 35.0

# Terminal weights
QN_X = 300.0
QN_THETA = 2500.0
QN_X_DOT = 8.0
QN_THETA_DOT = 120.0

# Input / acceleration regularization
R_FORCE = 0.008
R_ACC_CART = 0.001
R_ACC_POLE = 0.0005

# Desired cart position / upright pole angle.
CART_REFERENCE = 0.0
POLE_REFERENCE = 0.0

# -----------------------------
# Solver settings
# -----------------------------
IPOPT_MAX_ITER = 40
IPOPT_TOL = 1e-5
IPOPT_ACCEPTABLE_TOL = 1e-4

# -----------------------------
# State-estimation settings
# -----------------------------
# Webots PositionSensor provides position only; velocity is finite-differenced.
# 1.0 = no filtering. In simulation this is normally fine; reduce if desired.
VELOCITY_FILTER_ALPHA = 0.85

# -----------------------------
# Logging / diagnostics
# -----------------------------
PRINT_PERIOD_S = 0.25
LOG_FILENAME = "webots_log.csv"
