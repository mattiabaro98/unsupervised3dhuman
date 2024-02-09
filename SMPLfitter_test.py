import os

import trimesh
import torch

from SMPLfitter import SMPLfitter

front_file_pc = "ano2_front_rot.ply"
front_name, _ = os.path.splitext(front_file_pc)
front_input_file = "./examples/" + front_file_pc

back_file_pc = "ano2_back_rot.ply"
back_name, _ = os.path.splitext(back_file_pc)
back_input_file = "./examples/" + back_file_pc

fitter = SMPLfitter.SMPLfitter(smpl_gender="male")

front_points = fitter.load_pc(front_input_file)
front_sampled_points = fitter.sample_pc(front_points)
front_centered_points, front_center_trans = fitter.center_pc(front_sampled_points)

print(front_sampled_points.shape)

back_points = fitter.load_pc(back_input_file)
back_sampled_points = fitter.sample_pc(back_points)
back_centered_points, back_center_trans = fitter.center_pc(back_sampled_points)

init_pose, init_betas, init_scale, init_cam_trans, init_back_trans = fitter.initialize_params()

pred_pose, pred_betas, pred_scale, pred_cam_trans, pred_back_trans, new_front_points, new_back_points = fitter.smpl_fit(front_centered_points, back_centered_points, init_pose, init_betas, init_scale, init_cam_trans, init_back_trans)

# Store results
# mesh = trimesh.Trimesh()
# mesh.vertices = sampled_points
# mesh.export(f"./results/{name}_sampled.ply")

centered_points = torch.cat([new_front_points, new_back_points], dim=1)

mesh = trimesh.Trimesh()
mesh.vertices = centered_points.squeeze().cpu()
mesh.export(f"./results/{back_name}_frontback.ply")

fitter.save_smpl_ply(pred_pose, pred_betas, pred_scale, pred_cam_trans, f"./results/{front_name}_predicted.ply")
