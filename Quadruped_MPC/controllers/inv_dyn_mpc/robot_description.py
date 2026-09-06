"""Read the rigid-body tree actually instantiated in ``quadruped_arm.wbt``.

No mesh or URDF installation is needed.  Webots endpoint transforms describe
the pose at ``jointParameters.position``; they are NOT DH transforms.  Encoder
angles here use those native Webots conventions, including the small nonzero
FR knee endpoint rotation.  All explicitly specified masses, COMs and inertia
tensors are retained.  The two gripper sliders are fixed at zero and their
inertias remain in the tree; the runtime must hold those positions.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import numpy as np


LEG_NAMES = ("FR", "FL", "BR", "BL")
JOINT_NAMES = tuple(f"{leg}_{joint}_motor" for leg in LEG_NAMES
                    for joint in ("hip", "leg", "foot")) + tuple(
                        f"joint{i}" for i in range(1, 7))
DEFAULT_WORLD = Path(__file__).resolve().parents[2] / "worlds" / "quadruped_arm.wbt"


@dataclass
class _Node:
    kind: str
    fields: dict = field(default_factory=dict)


class _WbtParser:
    """Small structural VRML reader; ignores comments, preserves quoted URLs."""

    _tokens = re.compile(r'"(?:\\.|[^"\\])*"|#[^\n]*|[{}\[\]]|[^\s,{}\[\]"]+')
    _number = re.compile(r"^[+-]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][+-]?\d+)?$")

    def __init__(self, text):
        self.tokens = [t for t in self._tokens.findall(text) if not t.startswith("#")]
        self.index = 0
        self.definitions = {}

    def take(self):
        value = self.tokens[self.index]
        self.index += 1
        return value

    def value(self):
        value = self.take()
        if value == "DEF":
            name = self.take()
            node = self.value()
            self.definitions[name] = node
            return node
        if value == "USE":
            return self.definitions[self.take()]
        if value == "[":
            values = []
            while self.tokens[self.index] != "]":
                values.append(self.value())
            self.take()
            return values
        if value.startswith('"'):
            return json.loads(value)
        if self._number.match(value):
            values = [float(value)]
            while self.index < len(self.tokens) and self._number.match(self.tokens[self.index]):
                values.append(float(self.take()))
            return values[0] if len(values) == 1 else values
        if self.index < len(self.tokens) and self.tokens[self.index] == "{":
            self.take()
            node = _Node(value)
            while self.tokens[self.index] != "}":
                key = self.take()
                node.fields[key] = self.value()
            self.take()
            return node
        return value


def _vector(value, default):
    return np.asarray(default if value is None else value, dtype=float).reshape(-1)


def skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = skew(axis)
    return np.eye(3) + np.sin(angle) * cross + (1. - np.cos(angle)) * cross @ cross


def _pose(node):
    t = _vector(node.fields.get("translation"), [0., 0., 0.])
    rotation = _vector(node.fields.get("rotation"), [0., 0., 1., 0.])
    return axis_rotation(rotation[:3], rotation[3]), t


@dataclass
class Body:
    name: str
    parent: int
    rotation: np.ndarray
    translation: np.ndarray
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    joint_index: int = -1
    axis: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0.]))
    anchor: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reference_position: float = 0.

    @property
    def motion_subspace(self):
        linear = np.cross(self.axis, self.translation - self.anchor)
        return np.r_[self.rotation.T @ linear, self.rotation.T @ self.axis]

    @property
    def spatial_inertia(self):
        cross = skew(self.com)
        return np.block([[self.mass * np.eye(3), -self.mass * cross],
                         [self.mass * cross, self.inertia - self.mass * cross @ cross]])


@dataclass
class Frame:
    name: str
    body: int
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass
class RobotDescription:
    world_path: Path
    bodies: list
    frames: dict
    joint_names: tuple
    position_sensor_names: tuple
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    effort_limits: np.ndarray
    velocity_limits: np.ndarray
    initial_position: np.ndarray
    initial_rotation: np.ndarray
    foot_radius: float = .03
    foot_names: tuple = tuple(f"{leg}_foot" for leg in LEG_NAMES)
    tool_name: str = "tool"
    locked_joints: dict = field(default_factory=lambda: {"joint7": 0., "joint8": 0.})

    @property
    def total_mass(self):
        return float(sum(body.mass for body in self.bodies))

    @property
    def joint_limits(self):
        return np.column_stack((self.lower_limits, self.upper_limits))

    @classmethod
    def from_world(cls, path=None):
        path = DEFAULT_WORLD if path is None else Path(path).expanduser().resolve()
        text = path.read_text(encoding="utf-8")
        match = re.search(r"(?m)^Robot\s*\{", text)
        if match is None:
            raise ValueError(f"No explicit Robot node found in {path}")
        root = _WbtParser(text[match.start():]).value()
        bodies, frames = [], {}
        sensors = [None] * len(JOINT_NAMES)
        lower, upper, effort, velocity = (np.zeros(18) for _ in range(4))

        def add_body(node, parent, joint=None):
            rotation, translation = _pose(node)
            physics = node.fields.get("physics")
            mass, com, inertia = 0., np.zeros(3), np.zeros((3, 3))
            if isinstance(physics, _Node):
                if "mass" not in physics.fields or "inertiaMatrix" not in physics.fields:
                    raise ValueError(f"Explicit mass/inertia required for {node.fields.get('name', node.kind)}")
                mass = float(physics.fields["mass"])
                com = _vector(physics.fields.get("centerOfMass"), [0., 0., 0.])
                entries = _vector(physics.fields["inertiaMatrix"], [])
                if entries.size != 6:
                    raise ValueError("Expected one six-element Webots inertiaMatrix")
                xx, yy, zz, xy, xz, yz = entries
                inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
                if mass <= 0 or np.linalg.eigvalsh(inertia).min() <= 0:
                    raise ValueError(f"Invalid mass/inertia in {node.fields.get('name', node.kind)}")
            name = node.fields.get("name", f"{node.kind}_{len(bodies)}")
            body = Body(name, parent, rotation, translation, mass, com, inertia)
            if joint is not None:
                parameters = joint.fields.get("jointParameters", _Node("parameters")).fields
                motors = [n for n in joint.fields.get("device", [])
                          if isinstance(n, _Node) and n.kind in ("RotationalMotor", "LinearMotor")]
                if len(motors) != 1:
                    raise ValueError("Each articulated joint must have exactly one motor")
                motor = motors[0].fields
                motor_name = motor["name"]
                body.axis = _vector(parameters.get("axis"), [1., 0., 0.])
                body.axis /= np.linalg.norm(body.axis)
                body.anchor = _vector(parameters.get("anchor"), [0., 0., 0.])
                body.reference_position = float(parameters.get("position", 0.))
                if joint.kind == "SliderJoint":
                    if motor_name not in ("joint7", "joint8"):
                        raise ValueError(f"Unexpected slider {motor_name}")
                    body.translation = translation - body.reference_position * body.axis
                    body.reference_position = 0.
                else:
                    if motor_name not in JOINT_NAMES:
                        raise ValueError(f"Unexpected revolute joint {motor_name}")
                    index = JOINT_NAMES.index(motor_name)
                    body.joint_index = index
                    sensor_nodes = [n for n in joint.fields.get("device", [])
                                    if isinstance(n, _Node) and n.kind == "PositionSensor"]
                    if len(sensor_nodes) != 1:
                        raise ValueError(f"Missing encoder for {motor_name}")
                    sensors[index] = sensor_nodes[0].fields["name"]
                    # Webots defaults both bounds to zero (both zero means unbounded).
                    lo, hi = float(motor.get("minPosition", 0.)), float(motor.get("maxPosition", 0.))
                    lower[index], upper[index] = (-np.inf, np.inf) if lo == hi == 0. else (lo, hi)
                    effort[index] = float(motor.get("maxTorque", 10.))
                    velocity[index] = float(motor.get("maxVelocity", 10.))
            index = len(bodies)
            bodies.append(body)
            if name in frames:
                raise ValueError(f"Duplicate physical frame {name}")
            frames[name] = Frame(name, index)
            children = node.fields.get("children", [])
            # Keep Pinocchio configuration ordering identical to the Webots device ordering.
            if parent == -1:
                def child_order(child):
                    if not isinstance(child, _Node) or child.kind != "HingeJoint":
                        return 18
                    for device in child.fields.get("device", []):
                        if isinstance(device, _Node) and device.kind == "RotationalMotor":
                            return JOINT_NAMES.index(device.fields["name"])
                    return 18
                children = sorted(children, key=child_order)
            for child in children:
                walk(child, index)
            return index

        def walk(node, parent):
            if not isinstance(node, _Node):
                return
            if node.kind in ("HingeJoint", "SliderJoint"):
                endpoint = node.fields.get("endPoint")
                if not isinstance(endpoint, _Node) or endpoint.kind != "Solid":
                    raise ValueError("Only explicit Solid joint endpoints are supported")
                add_body(endpoint, parent, node)
            elif node.kind in ("Solid", "TouchSensor"):
                add_body(node, parent)
            elif node.kind in ("Transform", "Pose", "Group"):
                # Preserve fixed transforms when a physical descendant is present.
                def physical(n):
                    return isinstance(n, _Node) and (n.kind in ("Solid", "TouchSensor", "HingeJoint", "SliderJoint")
                           or any(physical(c) for c in n.fields.get("children", [])))
                if physical(node):
                    add_body(node, parent)

        add_body(root, -1)
        if any(sensor is None for sensor in sensors):
            raise ValueError("World is missing one or more expected quadruped/arm joints")
        for leg in LEG_NAMES:
            frame = frames[f"{leg}_4"]
            frames[f"{leg}_foot"] = Frame(f"{leg}_foot", frame.body)
        frames["tool"] = Frame("tool", frames["gripper_base"].body, np.array([0., 0., .1358]))
        rotation, position = _pose(root)
        # The root pose belongs to the free flyer, not a fixed parent transform.
        bodies[0].rotation, bodies[0].translation = np.eye(3), np.zeros(3)
        return cls(path, bodies, frames, JOINT_NAMES, tuple(sensors), lower, upper,
                   effort, velocity, position, rotation)


if __name__ == "__main__":
    description = RobotDescription.from_world()
    print(f"{description.world_path}: {len(description.bodies)} bodies, "
          f"{len(description.joint_names)} actuators, mass={description.total_mass:.6f} kg")
    for name, limits in zip(description.joint_names, description.joint_limits):
        print(f"{name:20s} [{limits[0]: .5f}, {limits[1]: .5f}]")
