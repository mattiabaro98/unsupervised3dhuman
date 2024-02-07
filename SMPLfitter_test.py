import os

import trimesh

from SMPLfitter import SMPLfitter

file_pc = "ex5rot.ply"
name, _ = os.path.splitext(file_pc)
input_file = "./examples/" + file_pc

fitter = SMPLfitter.SMPLfitter(smpl_gender="male")

points = fitter.load_pc(input_file)
sampled_points = fitter.sample_pc(points)
centered_points, trans = fitter.center_pc(sampled_points)

init_pose, init_betas, init_cam_t, trans, init_alpha = fitter.pose_default(trans)

pred_pose, pred_betas, pred_cam_t, pred_alpha = fitter.smpl_fit(
    centered_points, init_pose, init_betas, init_cam_t, init_alpha
)

# Store results
# mesh = trimesh.Trimesh()
# mesh.vertices = sampled_points
# mesh.export("./results/%s_sampled.ply" % name)

fitter.save_smpl_ply(pred_betas, pred_pose, pred_cam_t, trans, pred_alpha, "./results/%s_predicted.ply" % name)
