import json

import numpy as np
import open3d as o3d

file = "./results/ex5rot_predicted.ply"

with open("./SMPL_index_measure.json") as json_file:
    indexes = json.load(json_file)

pcd = o3d.io.read_point_cloud(file)
xyz = np.array(pcd.points)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.paint_uniform_color([0, 0, 0])

# minimum = -0.22
# maximum =-0.18
left_tight_pcd = o3d.geometry.PointCloud()
left_tight_pcd.points = o3d.utility.Vector3dVector(xyz[indexes["left_tight"]])
left_tight_pcd.paint_uniform_color([1, 0, 0])

right_tight_pcd = o3d.geometry.PointCloud()
right_tight_pcd.points = o3d.utility.Vector3dVector(xyz[indexes["right_tight"]])
right_tight_pcd.paint_uniform_color([0, 1, 0])


# minimum = -0.61
# maximum = -0.575
left_calf_pcd = o3d.geometry.PointCloud()
left_calf_pcd.points = o3d.utility.Vector3dVector(xyz[indexes["left_calf"]])
left_calf_pcd.paint_uniform_color([1, 0, 0])

right_calf_pcd = o3d.geometry.PointCloud()
right_calf_pcd.points = o3d.utility.Vector3dVector(xyz[indexes["right_calf"]])
right_calf_pcd.paint_uniform_color([0, 1, 0])


# minimum = 0.3
# maximum = 0.33
chest_pcd = o3d.geometry.PointCloud()
chest_pcd.points = o3d.utility.Vector3dVector(xyz[indexes["chest"]])
chest_pcd.paint_uniform_color([0, 0, 1])


# minimum = 0.1
# maximum = 0.13
waist_pcd = o3d.geometry.PointCloud()
waist_pcd.points = o3d.utility.Vector3dVector(xyz[indexes["waist"]])
waist_pcd.paint_uniform_color([0, 1, 1])

# minimum = -0.03
# maximum = 0.0
hip_pcd = o3d.geometry.PointCloud()
hip_pcd.points = o3d.utility.Vector3dVector(xyz[indexes["hip"]])
hip_pcd.paint_uniform_color([1, 1, 0])


# minimum = -0.35
# maximum = -0.33
left_arm_pcd = o3d.geometry.PointCloud()
left_arm_pcd.points = o3d.utility.Vector3dVector(xyz[indexes["left_arm"]])
left_arm_pcd.paint_uniform_color([1, 0, 0])

right_arm_pcd = o3d.geometry.PointCloud()
right_arm_pcd.points = o3d.utility.Vector3dVector(xyz[indexes["right_arm"]])
right_arm_pcd.paint_uniform_color([0, 1, 0])


# indexes = np.where((xyz[:, 1] > -1) & (xyz[:, 1] < 1) & (xyz[:, 0] > -1) & (xyz[:, 0] < 1))
# print(indexes)
# xyz = xyz[indexes]
# partial_pcd = o3d.geometry.PointCloud()
# partial_pcd.points = o3d.utility.Vector3dVector(xyz)
# partial_pcd.paint_uniform_color([1, 0, 0])

upper_y = np.mean(xyz[indexes["head_tip"]][:, 1])
lower_y = np.mean(xyz[indexes["feet_soles"]][:, 1])

upper_plane = o3d.geometry.TriangleMesh.create_box(width=1, height=0.01, depth=1)
upper_plane.translate([-0.5, upper_y, -0.5])
upper_plane.paint_uniform_color([1, 0, 0])

lower_plane = o3d.geometry.TriangleMesh.create_box(width=1, height=0.01, depth=1)
lower_plane.translate([-0.5, lower_y, -0.5])
lower_plane.paint_uniform_color([1, 0, 0])

combined_pcd = left_arm_pcd + right_arm_pcd + hip_pcd + waist_pcd + chest_pcd + left_tight_pcd + right_tight_pcd + left_calf_pcd + right_calf_pcd + pcd

# Visualize the combined point clouds and the coordinate frame
o3d.visualization.draw_geometries([combined_pcd, upper_plane, lower_plane])
