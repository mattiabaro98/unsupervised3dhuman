import numpy as np
import trimesh


def rotate_3d_array(array, angles):
    """
    Rotate a 3D NumPy array around x, y, and z axes by specified angles.

    Parameters:
        array (numpy.ndarray): 3D NumPy array to be rotated.
        angles (tuple): Three angles for rotation around x, y, and z axes in radians.

    Returns:
        numpy.ndarray: Rotated 3D NumPy array.
    """
    # Define rotation matrices for x, y, and z axes
    rotation_x = np.array(
        [[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]]
    )

    rotation_y = np.array(
        [[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]]
    )

    rotation_z = np.array(
        [[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]]
    )

    rotated_array = np.dot(rotation_z, np.dot(rotation_y, np.dot(rotation_x, array.T))).T

    return rotated_array


def center_rotate(input_ply_path, output_ply_path, direction):

    mesh = trimesh.load(input_ply_path)
    points = mesh.vertices

    center = np.mean(points, axis=0)
    centered_points = points - center

    if direction == "front":
        angles = (0, 0, np.pi / 2)  # RealSense front pic
    if direction == "back":
        angles = (0, np.pi, np.pi / 2)  # RealSense back pic

    rotated_points = rotate_3d_array(centered_points, angles)

    mesh.vertices = rotated_points
    mesh.export(output_ply_path)
