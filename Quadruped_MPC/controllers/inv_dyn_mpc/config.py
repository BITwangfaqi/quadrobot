"""Explicit tuning and task settings for the whole-body Webots controller.

Positions, velocities and forces are SI units. Joint positions are the raw
Webots encoder angles; they have no offsets from the legacy leg controller.
"""

from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np


@dataclass
class MPCConfig:
    horizon: int = 14
    dt: float = 0.01
    growth: float = 1.193776641714434
    # The author's demo uses 10; Webots contact transients require more.
    # Never accept an unconverged iterate merely to meet a timing target.
    max_iterations: int = 80
    tolerance: float = 1.0e-3
    solver: str = "fatrop"
    jit: bool = True
    # Paper Eq. (8): bound and retain torques at k=0,1,2 only.
    # Setting this to 0 retains bounded torques throughout for comparison.
    torque_nodes: int = 3
    warm_start: bool = True
    barrier_initial: float = 1.0e-4
    lock_wrist: bool = True
    bound_push: float = 1.0e-7
    derivative_threads: int = 1
    tool_position_gain: float = 3.0
    ground_height: float = 0.0
    ground_tolerance: float = 0.01
    initial_joint_limit_tolerance: float = 0.002
    base_position_weight: tuple = (0.0, 0.0, 1000.0)
    base_rotation_weight: tuple = (10000.0, 10000.0, 0.0)
    base_velocity_weight: tuple = (2000.0, 2000.0, 1000.0, 1000.0, 1000.0, 2000.0)
    leg_position_weight: float = 500.0
    hip_position_weight: float = 1000.0
    arm_position_weight: float = 100.0
    joint_velocity_weight: float = 2.0
    arm_joint_velocity_weight: float = 10.0
    acceleration_weight: float = 0.001
    torque_weight: float = 0.0001
    arm_torque_weight: float = 0.01
    force_weight: float = 0.0005
    foot_height_weight: float = 0.0
    friction: float = 0.6
    max_normal_force: float = 400.0
    foot_radius: float = 0.03
    swing_height: float = 0.05
    stabilization: float = 20.0

    solve_period: float = 0.0125
    startup_duration: float = 3.0
    settle_duration: float = 1.0
    solution_timeout: float = 0.10
    gait_period: float = 0.80
    joint_velocity_filter_hz: float = 50.0
    leg_kp: float = 12.0
    leg_kd: float = 1.2
    # The light wrist has effective inertia around 1e-4 kg*m^2. Large generic
    # damping gains are unstable with 2 ms sampled torque control.
    arm_kp: float = 1.0
    arm_kd: float = 0.02
    maximum_forward_speed: float = 0.12
    maximum_lateral_speed: float = 0.08
    maximum_yaw_rate: float = 0.25
    maximum_tool_speed: float = 0.025
    maximum_base_reference_error: float = 0.10
    maximum_tool_displacement: float = 0.12
    diagnostic_period: float = 1.0
    # This world already encodes the bent leg geometry at raw encoder zero.
    # Legacy controller offsets [0, -0.45, 1.4] do not apply to its full model.
    nominal_leg: tuple = (0.0, 0.0, 0.0)
    nominal_arm: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # The force task assumes a physically established external contact.
    tool_force_enabled: bool = False
    tool_force: tuple = (0.0, 0.0, 0.0)
    tool_contact_velocity: tuple = (0.0, 0.0, 0.0)

    def time_steps(self):
        """Intervals; gait samples use their cumulative sum, not stage index."""
        return self.dt * self.growth ** np.arange(self.horizon, dtype=float)

    def validate(self):
        if self.solver not in ("fatrop", "ipopt"):
            raise ValueError("solver must be 'fatrop' or 'ipopt'")
        for name in ("tool_force_enabled", "jit", "warm_start", "lock_wrist"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a JSON boolean")
        if not isinstance(self.horizon, int) or self.horizon < 2:
            raise ValueError("horizon must be an integer >= 2")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if type(self.torque_nodes) is not int or self.torque_nodes not in (0, 3):
            raise ValueError("torque_nodes must be 3 (paper) or 0 (all nodes)")
        if type(self.derivative_threads) is not int or not 1 <= self.derivative_threads <= 16:
            raise ValueError('derivative_threads must be an integer between 1 and 16')
        positive = (
            "dt", "growth", "tolerance", "friction", "max_normal_force",
            "foot_radius", "stabilization", "solve_period", "startup_duration",
            "gait_period", "joint_velocity_filter_hz", "diagnostic_period",
            "maximum_base_reference_error", "maximum_tool_displacement",
            "tool_position_gain",
            "barrier_initial",
            "bound_push",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "settle_duration", "solution_timeout", "swing_height", "leg_kp",
            "leg_kd", "arm_kp", "arm_kd", "maximum_forward_speed",
            "maximum_lateral_speed", "maximum_yaw_rate", "maximum_tool_speed",
            "ground_tolerance", "leg_position_weight", "arm_position_weight",
            "joint_velocity_weight", "acceleration_weight", "torque_weight",
            "force_weight", "foot_height_weight",
            "initial_joint_limit_tolerance",
            "hip_position_weight", "arm_joint_velocity_weight", "arm_torque_weight",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.growth < 1:
            raise ValueError("growth must be >= 1 for the nonuniform horizon")
        if self.solution_timeout < self.solve_period:
            raise ValueError("solution_timeout must be >= solve_period")
        if not np.isfinite(self.ground_height):
            raise ValueError("ground_height must be finite")
        for name, size in (("base_position_weight", 3), ("base_rotation_weight", 3),
                           ("base_velocity_weight", 6)):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must contain {size} finite nonnegative weights")
        for name in ("tool_force", "tool_contact_velocity"):
            vector = np.asarray(getattr(self, name), dtype=float)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must contain three finite values")
        for name, size in (("nominal_leg", 3), ("nominal_arm", 6)):
            vector = np.asarray(getattr(self, name), dtype=float)
            if vector.shape != (size,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must contain {size} finite raw joint angles")
        if not self.tool_force_enabled and np.linalg.norm(self.tool_force) > 0:
            raise ValueError("Nonzero tool_force requires tool_force_enabled=true")
        return self

    @classmethod
    def from_json(cls, path=None):
        if path is None:
            return cls().validate()
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Controller configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values).validate()


@dataclass
class MPCReference:
    """Targets at the current solve time, in the world frame unless named.

    The solver advances base/tool positions with their desired velocities over
    its nonuniform horizon. Quaternion ordering is x, y, z, w.
    """

    base_position: np.ndarray
    base_quaternion_xyzw: np.ndarray
    base_velocity_world: np.ndarray
    yaw_rate: float
    joint_positions: np.ndarray
    tool_position: np.ndarray
    tool_velocity: np.ndarray
    tool_force: np.ndarray
