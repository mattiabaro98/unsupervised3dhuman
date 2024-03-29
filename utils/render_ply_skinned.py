import argparse

import trimesh

parser = argparse.ArgumentParser(description="Read the contents of a file.")
parser.add_argument("filename", help="Path to the file to be read.")

args = parser.parse_args()

# Load the PLY file
mesh = trimesh.load_mesh(args.filename)

# Create a scene with the mesh
scene = trimesh.Scene(mesh)
scene.add_geometry(mesh)

# Plot the scene
scene.show()
