import numpy as np
import open3d as o3d


file = "./results/ex5rot_predicted.ply"


pcd = o3d.io.read_point_cloud(file)
xyz = np.array(pcd.points)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.paint_uniform_color([0, 0, 0])

# minimum = -0.22
# maximum =-0.18
left_tight = [898, 899, 901, 903, 904, 905, 906, 907, 909, 910, 934, 935, 957, 962, 964, 1365, 1453]
right_tight = [4385, 4386, 4387, 4388, 4390, 4391, 4392, 4393, 4396, 4397, 4421, 4422, 4443, 4448, 4449, 4838, 4926]

left_tight_pcd = o3d.geometry.PointCloud()
left_tight_pcd.points = o3d.utility.Vector3dVector(xyz[left_tight])
left_tight_pcd.paint_uniform_color([1, 0, 0])

right_tight_pcd = o3d.geometry.PointCloud()
right_tight_pcd.points = o3d.utility.Vector3dVector(xyz[right_tight])
right_tight_pcd.paint_uniform_color([0, 1, 0])


# minimum = -0.61
# maximum = -0.575
left_calf = [1087, 1088, 1091, 1092, 1096, 1097, 1099, 1100, 1103, 1155, 1371, 1464, 1467, 1469, 1528]
right_calf = [4574, 4575, 4578, 4579, 4581, 4582, 4586, 4587, 4589, 4641, 4844, 4937, 4939, 4941, 4999]

left_calf_pcd = o3d.geometry.PointCloud()
left_calf_pcd.points = o3d.utility.Vector3dVector(xyz[left_calf])
left_calf_pcd.paint_uniform_color([1, 0, 0])

right_calf_pcd = o3d.geometry.PointCloud()
right_calf_pcd.points = o3d.utility.Vector3dVector(xyz[right_calf])
right_calf_pcd.paint_uniform_color([0, 1, 0])


# minimum = 0.3
# maximum = 0.33
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
chest_pcd = o3d.geometry.PointCloud()
chest_pcd.points = o3d.utility.Vector3dVector(xyz[chest])
chest_pcd.paint_uniform_color([0, 0, 1])


# minimum = 0.1
# maximum = 0.13
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
waist_pcd = o3d.geometry.PointCloud()
waist_pcd.points = o3d.utility.Vector3dVector(xyz[waist])
waist_pcd.paint_uniform_color([0, 1, 1])

# minimum = -0.03
# maximum = 0.0
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
hip_pcd = o3d.geometry.PointCloud()
hip_pcd.points = o3d.utility.Vector3dVector(xyz[hip])
hip_pcd.paint_uniform_color([1, 1, 0])


# minimum = -0.35
# maximum = -0.33
left_arm = [627, 628, 789, 1232, 1233, 1311, 1315, 1378, 1379, 1381, 1382, 1385, 1388, 1389, 1393, 1394, 1396, 1397]
right_arm = [4116, 4117, 4277, 4716, 4717, 4791, 4794, 4850, 4851, 4855, 4856, 4859, 4862, 4863, 4865, 4866, 4870, 4871]

left_arm_pcd = o3d.geometry.PointCloud()
left_arm_pcd.points = o3d.utility.Vector3dVector(xyz[left_arm])
left_arm_pcd.paint_uniform_color([1, 0, 0])

right_arm_pcd = o3d.geometry.PointCloud()
right_arm_pcd.points = o3d.utility.Vector3dVector(xyz[right_arm])
right_arm_pcd.paint_uniform_color([0, 1, 0])


# indexes = np.where((xyz[:, 1] > -1) & (xyz[:, 1] < 1) & (xyz[:, 0] > -1) & (xyz[:, 0] < 1))
# print(indexes)
# xyz = xyz[indexes]
# partial_pcd = o3d.geometry.PointCloud()
# partial_pcd.points = o3d.utility.Vector3dVector(xyz)
# partial_pcd.paint_uniform_color([1, 0, 0])

upper_y = 0.84
lower_y = -0.95

upper_plane = o3d.geometry.TriangleMesh.create_box(width=1, height=0.01, depth=1)
upper_plane.translate([-0.5, upper_y, -0.5])
upper_plane.paint_uniform_color([1, 0, 0])

lower_plane = o3d.geometry.TriangleMesh.create_box(width=1, height=0.01, depth=1)
lower_plane.translate([-0.5, lower_y, -0.5])
lower_plane.paint_uniform_color([1, 0, 0])

combined_pcd = (
    left_arm_pcd
    + right_arm_pcd
    + hip_pcd
    + waist_pcd
    + chest_pcd
    + left_tight_pcd
    + right_tight_pcd
    + left_calf_pcd
    + right_calf_pcd
    + pcd
)

# Visualize the combined point clouds and the coordinate frame
o3d.visualization.draw_geometries([combined_pcd, upper_plane, lower_plane])
