import numpy as np
import open3d as o3d

n = 1
direction = "back"
ply_path = f"./examples/ex{n}{direction}.ply"
output_ply_path = f"./examples/ex{n}{direction}_rot.ply"

pcd = o3d.io.read_point_cloud(ply_path)

points = np.asarray(pcd.points)
center = np.mean(points, axis=0)
centered_points = points - center
pcd.points = o3d.utility.Vector3dVector(centered_points)

# angles = (0, 0, 0)  # ex1
# angles = (0, np.pi / 2, 0) # ex2
angles = (0, 0, np.pi / 2)  # RealSense pics front
# angles = (0, np.pi, np.pi / 2)  # RealSense pics back
R = pcd.get_rotation_matrix_from_xyz(angles)
pcd.rotate(R, center=(0, 0, 0))

o3d.io.write_point_cloud(output_ply_path, pcd)
