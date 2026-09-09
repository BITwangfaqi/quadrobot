import numpy as np
import pinocchio as pin
from meshcat.geometry import Cylinder, MeshLambertMaterial


def visualize_forces(viewer, robot, model, data, q, forces):
    # Update frames
    pin.framesForwardKinematics(model, data, q)
    
    # Forces on each foot
    for foot_idx, frame_id in enumerate(robot.foot_frames):
        force = forces[foot_idx * 3 : (foot_idx + 1) * 3].flatten()
        pos = data.oMf[frame_id].translation
        add_force_to_viewer(viewer, force, pos, foot_idx)
    
    # Arm end-effector force
    if robot.arm_ee_frame is not None:
        ext_idx = len(robot.foot_frames)
        force = forces[ext_idx * 3 :].flatten()
        pos = data.oMf[robot.arm_ee_frame].translation
        add_force_to_viewer(viewer, force, pos, ext_idx, scale=0.002)

def add_force_to_viewer(viewer, force, pos, idx, scale=0.001):
    force_magnitude = np.linalg.norm(force)
    force_direction = force / force_magnitude
    arrow_length = force_magnitude * scale
    
    # Create arrow geometry (cylinder)
    arrow_radius = 0.01  # 1cm radius
    cylinder = Cylinder(arrow_length, arrow_radius)
    material = MeshLambertMaterial(color=0xff0000, opacity=0.8)
    arrow_center = pos + force_direction * arrow_length / 2
    
    # Calculate rotation matrix to align cylinder
    z_axis = np.array([0, 1, 0])  # pinocchio-meshcat conversion
    if np.allclose(force_direction, z_axis):
        rotation_matrix = np.eye(3)
    elif np.allclose(force_direction, -z_axis):
        rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        # Rodrigues' rotation formula
        v = np.cross(z_axis, force_direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, force_direction)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    
    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = arrow_center
    
    # Add arrow to viewer
    viewer[f'force_arrow_{idx}'].set_object(cylinder, material)
    viewer[f'force_arrow_{idx}'].set_transform(transform)
