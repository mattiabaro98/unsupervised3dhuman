import numpy as np
import trimesh


def rotate_3d_array(array: np.array, angles: tuple[float]):
    """
    Rotate a 3D NumPy array around x, y, and z axes by specified angles.

    Parameters:
        array (numpy.ndarray): 3D NumPy array to be rotated.
        angles (tuple): Three angles for rotation around x, y, and z axes in radians.

    Returns:
        numpy.ndarray: Rotated 3D NumPy array.
    """
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


def rotate(mesh: trimesh.Trimesh, direction: str) -> trimesh.Trimesh:
    points = mesh.vertices

    if direction == "front":
        angles = (0, 0, np.pi / 2)  # RealSense front pic
    if direction == "back":
        angles = (0, np.pi, -np.pi / 2)  # RealSense back pic

    mesh.vertices = rotate_3d_array(points, angles)

    return mesh


def remove_floor(mesh: trimesh.Trimesh, delta: float) -> trimesh.Trimesh:
    points = mesh.vertices

    mesh = trimesh.Trimesh(vertices=points[points[:, 1] > np.min(points[:, 1] + delta)])

    return mesh


def scale(mesh: trimesh.Trimesh, scale: float) -> trimesh.Trimesh:
    points = mesh.vertices

    mesh.vertices = np.dot(points, scale)

    return mesh


def center(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    points = mesh.vertices

    mesh.vertices = points - np.mean(points, axis=0)

    return mesh


def preprocess(input_ply_path: str, output_ply_path: str, direction: str) -> None:
    mesh = trimesh.load(input_ply_path)

    mesh = center(mesh)
    mesh = rotate(mesh, direction)
    mesh = remove_floor(mesh, 0.1)
    mesh = scale(mesh, 1.15)

    mesh.export(output_ply_path)

    return None
