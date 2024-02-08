import os

from SMPLfitter import SMPLfitter

# import trimesh


file_pc = "ex5rot.ply"
name, _ = os.path.splitext(file_pc)
input_file = "./examples/" + file_pc

fitter = SMPLfitter.SMPLfitter(smpl_gender="male")

points = fitter.load_pc(input_file)
sampled_points = fitter.sample_pc(points)
centered_points, center_trans = fitter.center_pc(sampled_points)

init_pose, init_betas, init_scale, init_cam_trans, center_trans = fitter.initialize_params(center_trans)

pred_pose, pred_betas, pred_scale, pred_cam_trans = fitter.smpl_fit(
    centered_points, init_pose, init_betas, init_scale, init_cam_trans
)

# Store results
# mesh = trimesh.Trimesh()
# mesh.vertices = sampled_points
# mesh.export(f"./results/{name}_sampled.ply")

fitter.save_smpl_ply(pred_pose, pred_betas, pred_scale, pred_cam_trans, center_trans, f"./results/{name}_predicted.ply")
