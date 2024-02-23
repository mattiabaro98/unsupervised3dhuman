import trimesh

n = 1
file = f"./results/ex{n}front_pred.ply"

# Load the PLY file
mesh = trimesh.load_mesh(file)

# Create a scene with the mesh and set its color
scene = trimesh.Scene(mesh)
scene.add_geometry(mesh)

# Plot the scene
scene.show()
