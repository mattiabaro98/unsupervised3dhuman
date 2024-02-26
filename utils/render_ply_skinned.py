import trimesh

n = 5
file = f"./results/ex{n}rot_predicted.ply"

# Load the PLY file
mesh = trimesh.load_mesh(file)

# Create a scene with the mesh and set its color
scene = trimesh.Scene(mesh)
scene.add_geometry(mesh)

# Plot the scene
scene.show()
