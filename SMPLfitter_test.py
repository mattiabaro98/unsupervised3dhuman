import os

import numpy as np
import trimesh

from SMPLfitter import SMPLfitter

file_pc = "ex3rot.ply"
name, _ = os.path.splitext(file_pc)
input_file = "./examples/" + file_pc


fitter = SMPLfitter.SMPLfitter(smpl_gender="male")

points = fitter.load_pc(input_file)
sampled_points = fitter.sample_pc(points)
centered_points, trans = fitter.center_pc(sampled_points)
pred_pose, pred_betas, pred_cam_t, trans_back = fitter.pose_initializer(centered_points, trans)
new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t = fitter.smpl_fit(
    points, pred_pose, pred_betas, pred_cam_t
)

# Store results

mesh = trimesh.Trimesh()
mesh.vertices = centered_points + trans
mesh.export("./results/%s_sampled.ply" % name)

fitter.save_smpl_ply(pred_betas, pred_pose, pred_cam_t, trans_back, "./results/%s_initialized.ply" % name)
fitter.save_smpl_ply(new_opt_betas, new_opt_pose, new_opt_cam_t, trans_back, "./results/%s_predicted.ply" % name)

np.savez(
    "./results/%s_params.npz" % name,
    beta=new_opt_betas.detach().cpu().numpy().reshape(10),
    pose=new_opt_pose.detach().cpu().numpy().reshape(72),
)
