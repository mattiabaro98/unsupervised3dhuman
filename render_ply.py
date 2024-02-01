import argparse

import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser(description="Read the contents of a file.")
parser.add_argument("--filename", help="Path to the file to be read.")

args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.filename)

xyz = np.array(pcd.points)

axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
o3d.visualization.draw([pcd, axis_mesh])
