import argparse

import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser(description="Read the contents of a file.")
parser.add_argument("filename1", help="Path to the first file to be read.")
parser.add_argument("filename2", help="Path to the second file to be read.")

args = parser.parse_args()

pcd1 = o3d.io.read_point_cloud(args.filename1)
xyz1 = np.array(pcd1.points)
red_pcd = o3d.geometry.PointCloud()
red_pcd.points = o3d.utility.Vector3dVector(xyz1)
red_pcd.paint_uniform_color([1, 0, 0])  # Red color

pcd2 = o3d.io.read_point_cloud(args.filename2)
xyz2 = np.array(pcd2.points)
blue_pcd = o3d.geometry.PointCloud()
blue_pcd.points = o3d.utility.Vector3dVector(xyz2)
blue_pcd.paint_uniform_color([0, 0, 1])  # Blue color

o3d.visualization.draw_geometries([red_pcd + blue_pcd])
