import numpy as np
import open3d as o3d

n = 5
# file1 = f"./examples/ex{n}rot.ply"
file1 = f"./results/ex{n}rot_scaled.ply"
# file2 = f"./results/ex{n}rot_initialized.ply"
file2 = f"./results/ex{n}rot_predicted.ply"


pcd1 = o3d.io.read_point_cloud(file1)
pcd2 = o3d.io.read_point_cloud(file2)
xyz1 = np.array(pcd1.points)
xyz2 = np.array(pcd2.points)

# Create a red point cloud
red_pcd = o3d.geometry.PointCloud()
red_pcd.points = o3d.utility.Vector3dVector(xyz1)
red_pcd.paint_uniform_color([1, 0, 0])  # Red color

# Create a blue point cloud
blue_pcd = o3d.geometry.PointCloud()
blue_pcd.points = o3d.utility.Vector3dVector(xyz2)
blue_pcd.paint_uniform_color([0, 0, 1])  # Blue color

# Create the coordinate frame mesh
axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

# Combine both point clouds
combined_pcd = red_pcd + blue_pcd

# Visualize the combined point clouds and the coordinate frame
o3d.visualization.draw_geometries([combined_pcd, axis_mesh])
