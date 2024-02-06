import numpy as np
import open3d as o3d
from skspatial.objects import Plane, Points
from scipy.spatial import ConvexHull


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


left_tight = [898, 899, 901, 903, 904, 905, 906, 907, 909, 910, 934, 935, 957, 962, 964, 1365, 1453]
right_tight = [4385, 4386, 4387, 4388, 4390, 4391, 4392, 4393, 4396, 4397, 4421, 4422, 4443, 4448, 4449, 4838, 4926]

left_calf = [1087, 1088, 1091, 1092, 1096, 1097, 1099, 1100, 1103, 1155, 1371, 1464, 1467, 1469, 1528]
right_calf = [4574, 4575, 4578, 4579, 4581, 4582, 4586, 4587, 4589, 4641, 4844, 4937, 4939, 4941, 4999]

chest = [
    646,
    647,
    652,
    653,
    660,
    661,
    688,
    689,
    690,
    692,
    722,
    724,
    725,
    766,
    767,
    795,
    891,
    893,
    894,
    929,
    940,
    1191,
    1192,
    1193,
    1212,
    1268,
    1270,
    1348,
    1756,
    2843,
    3017,
    3079,
    4087,
    4130,
    4131,
    4134,
    4135,
    4140,
    4141,
    4157,
    4158,
    4176,
    4177,
    4178,
    4180,
    4210,
    4211,
    4212,
    4225,
    4255,
    4256,
    4377,
    4378,
    4379,
    4414,
    4676,
    4677,
    4678,
    4679,
    4680,
    4695,
    4753,
    4824,
    4895,
    5223,
    6305,
]

waist = [
    678,
    679,
    705,
    830,
    846,
    855,
    916,
    917,
    918,
    919,
    920,
    939,
    1336,
    1337,
    1449,
    1768,
    1780,
    1781,
    1784,
    1792,
    2910,
    2911,
    2915,
    2916,
    2917,
    2918,
    2919,
    3100,
    3122,
    3500,
    4166,
    4167,
    4193,
    4332,
    4341,
    4402,
    4403,
    4404,
    4405,
    4406,
    4425,
    4812,
    4813,
    4921,
    5246,
    5247,
    5253,
    5254,
    5257,
    6368,
    6369,
    6370,
    6371,
    6374,
    6375,
    6376,
    6377,
    6378,
    6383,
    6524,
    6543,
    6544,
]

hip = [
    823,
    863,
    864,
    915,
    932,
    1204,
    1205,
    1446,
    1447,
    1511,
    1513,
    1807,
    3084,
    3116,
    3117,
    3118,
    3119,
    3136,
    3137,
    3138,
    4350,
    4351,
    4399,
    4418,
    4690,
    4692,
    4919,
    4920,
    4927,
    4983,
    4984,
    6509,
    6539,
    6540,
    6541,
    6557,
    6558,
    6559,
]

left_arm = [627, 628, 789, 1232, 1233, 1311, 1315, 1378, 1379, 1381, 1382, 1385, 1388, 1389, 1393, 1394, 1396, 1397]
right_arm = [4116, 4117, 4277, 4716, 4717, 4791, 4794, 4850, 4851, 4855, 4856, 4859, 4862, 4863, 4865, 4866, 4870, 4871]

file = "./results/ex5rot_predicted.ply"

pcd = o3d.io.read_point_cloud(file)
xyz = np.array(pcd.points)

real_height = 1.73
smpl_height = 0.84 + 0.95

left_tight = compute_length(xyz[left_tight]) * real_height / smpl_height
right_tight = compute_length(xyz[right_tight]) * real_height / smpl_height
left_calf = compute_length(xyz[left_calf]) * real_height / smpl_height
right_calf = compute_length(xyz[right_calf]) * real_height / smpl_height
chest = compute_length(xyz[chest]) * real_height / smpl_height
waist = compute_length(xyz[waist]) * real_height / smpl_height
hip = compute_length(xyz[hip]) * real_height / smpl_height
left_arm = compute_length(xyz[left_arm]) * real_height / smpl_height
right_arm = compute_length(xyz[right_arm]) * real_height / smpl_height


print("tights:", np.round(np.mean([left_tight, right_tight]), 2))
print("calfs:", np.round(np.mean([left_calf, right_calf]), 2))
print("chest:", np.round(chest, 2))
print("waist:", np.round(waist, 2))
print("hip:", np.round(hip, 2))
print("arms:", np.round(np.mean([left_arm, right_arm]), 2))
