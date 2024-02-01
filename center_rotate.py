import numpy as np
import open3d as o3d

# ply_path = "./ex1.ply"
# output_ply_path = "./ex1rot.ply"

ply_path = "./ex2.ply"
output_ply_path = "./ex2rot.ply"

# ply_path = "./ex3.ply"
# output_ply_path = "./ex3rot.ply"

pcd = o3d.io.read_point_cloud(ply_path)

points = np.asarray(pcd.points)
center = np.mean(points, axis=0)
centered_points = points - center
pcd.points = o3d.utility.Vector3dVector(centered_points)

R = pcd.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
pcd.rotate(R, center=(0, 0, 0))

o3d.io.write_point_cloud(output_ply_path, pcd)
