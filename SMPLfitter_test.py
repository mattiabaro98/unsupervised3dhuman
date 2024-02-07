import os

import torch
import trimesh

from SMPLfitter import SMPLfitter

file_pc = "ex7rot.ply"
name, _ = os.path.splitext(file_pc)
input_file = "./examples/" + file_pc

# ex3
# alpha = 1.4
# cam_t = [0, 0.3, 0]

# ex4
# alpha = 1.15
# cam_t = [0, 0.1, 0]

# ex5
alpha = 1.15
cam_t = [0, 0.3, 0]


fitter = SMPLfitter.SMPLfitter(smpl_gender="male")

points = fitter.load_pc(input_file)
sampled_points = fitter.sample_pc(points)
centered_points, trans = fitter.center_pc(sampled_points)

scaled_points = fitter.scale_pc(centered_points, alpha)

pred_pose, pred_betas, pred_cam_t, trans_back = fitter.pose_default(trans)
pred_cam_t[:, :] = torch.Tensor(cam_t).unsqueeze(0).float()

new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t = fitter.smpl_fit(
    scaled_points, pred_pose, pred_betas, pred_cam_t
)

# Store results
mesh = trimesh.Trimesh()
mesh.vertices = scaled_points + trans
mesh.export("./results/%s_scaled.ply" % name)

fitter.save_smpl_ply(pred_betas, pred_pose, pred_cam_t, trans_back, "./results/%s_initialized.ply" % name)
fitter.save_smpl_ply(new_opt_betas, new_opt_pose, new_opt_cam_t, trans_back, "./results/%s_predicted.ply" % name)

# np.savez(
#     "./results/%s_params.npz" % name,
#     beta=new_opt_betas.detach().cpu().numpy().reshape(10),
#     pose=new_opt_pose.detach().cpu().numpy().reshape(72),
# )
