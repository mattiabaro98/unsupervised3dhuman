import argparse

import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser(description="Read the contents of a file.")
parser.add_argument("filename", help="Path to the file to be read.")

args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.filename)

xyz = np.array(pcd.points)

o3d.visualization.draw([pcd])
