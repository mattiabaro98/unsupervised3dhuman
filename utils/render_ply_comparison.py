import numpy as np
import open3d as o3d

n = 1
direction = "front"
file1 = f"./results/ex{n}{direction}_rot.ply"
file2 = f"./results/ex{n}{direction}_pred.ply"

pcd1 = o3d.io.read_point_cloud(file1)
xyz1 = np.array(pcd1.points)
red_pcd = o3d.geometry.PointCloud()
red_pcd.points = o3d.utility.Vector3dVector(xyz1)
red_pcd.paint_uniform_color([1, 0, 0])  # Red color

pcd2 = o3d.io.read_point_cloud(file2)
xyz2 = np.array(pcd2.points)
blue_pcd = o3d.geometry.PointCloud()
blue_pcd.points = o3d.utility.Vector3dVector(xyz2)
blue_pcd.paint_uniform_color([0, 0, 1])  # Blue color

o3d.visualization.draw_geometries([red_pcd + blue_pcd])
