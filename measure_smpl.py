import json

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from skspatial.objects import Plane, Points


def compute_convex_hull(points):
    """
    Compute the convex hull from a 2D numpy array of points.

    Parameters:
        points (numpy.ndarray): Array of shape (n, 2) representing 2D points.

    Returns:
        numpy.ndarray: Array of shape (m, 2) representing the vertices of the convex hull.
    """
    # Ensure the input is a numpy array
    points = np.array(points)

    # Compute the convex hull
    hull = ConvexHull(points)

    # Get the vertices of the convex hull
    hull_vertices = points[hull.vertices]

    return hull_vertices


def compute_length(xyz):
    points = Points(xyz)
    plane = Plane.best_fit(points)
    normal = np.array(plane.normal)

    y = np.array([0, 1, 0])
    phi = -np.arccos(np.dot(normal, y) / (np.linalg.norm(normal) * np.linalg.norm(y)))
    theta = -np.arctan2(normal[0], normal[2])

    rotX = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    rotY = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

    for i in range(xyz.shape[0]):
        xyz[i] = np.array(plane.project_point(xyz[i]))

    for i in range(xyz.shape[0]):
        xyz[i] = np.dot(rotX, np.dot(rotY, xyz[i]))

    xz = xyz[:, [0, 2]]

    convex_hull = compute_convex_hull(xz)

    length = 0

    for i in range(1, convex_hull.shape[0]):
        length += np.linalg.norm(convex_hull[i] - convex_hull[i - 1])

    length += np.linalg.norm(convex_hull[-1] - convex_hull[0])

    return length


with open("./SMPL_index_measure.json") as json_file:
    indexes = json.load(json_file)

file = "./results/ex1front_pred.ply"
real_height = 1.73
# real_height = 1.69
# real_height = 1.83

pcd = o3d.io.read_point_cloud(file)
xyz = np.array(pcd.points)

upper_y = np.mean(xyz[indexes["head_tip"]][:, 1])
lower_y = np.mean(xyz[indexes["feet_soles"]][:, 1])
smpl_height = upper_y - lower_y

left_tight = compute_length(xyz[indexes["left_tight"]]) * real_height / smpl_height
right_tight = compute_length(xyz[indexes["right_tight"]]) * real_height / smpl_height
left_calf = compute_length(xyz[indexes["left_calf"]]) * real_height / smpl_height
right_calf = compute_length(xyz[indexes["right_calf"]]) * real_height / smpl_height
chest = compute_length(xyz[indexes["chest"]]) * real_height / smpl_height
waist = compute_length(xyz[indexes["waist"]]) * real_height / smpl_height
hip = compute_length(xyz[indexes["hip"]]) * real_height / smpl_height
left_arm = compute_length(xyz[indexes["left_arm"]]) * real_height / smpl_height
right_arm = compute_length(xyz[indexes["right_arm"]]) * real_height / smpl_height


print("tights:", np.round(np.mean([left_tight, right_tight]), 2))
print("calfs:", np.round(np.mean([left_calf, right_calf]), 2))
print("chest:", np.round(chest, 2))
print("waist:", np.round(waist, 2))
print("hip:", np.round(hip, 2))
print("arms:", np.round(np.mean([left_arm, right_arm]), 2))
