from __future__ import division, print_function

import argparse
import os  # , shutil

import joblib
import numpy as np
import smplx
import torch
import torch.optim as optim
import trimesh
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_axis_angle,
    rotation_6d_to_matrix,
)

from src.Network import point_net_ssg
from src.surfaceem import surface_EM_depth
from src.utils import farthest_point_sample

# # parsing argmument
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, help="PLY file for processing")
args = parser.parse_args()
input_file = args.filename
output_file = os.path.splitext(input_file)[0] + "_smpl_params.npz"

# # Load all Training settings
# if torch.cuda.is_available():
#     device = torch.device("cuda:" + str(opt.gpu_ids))
# else:
device = torch.device("cpu")

# --------pytorch model and optimizer is the key
model = point_net_ssg(device=device).to(device).eval()
model.load_state_dict(torch.load("./pretrained/model_best_depth.pth", map_location=device))

optimizer = optim.Adam(model.parameters())
smplmodel = smplx.create("./smpl_models/", model_type="smpl", gender="male", ext="pkl").to(device)

# -- intial EM
# --- load predefined ------
pred_pose = torch.zeros(1, 72).to(device)
pred_betas = torch.zeros(1, 10).to(device)
pred_cam_t = torch.zeros(1, 3).to(device)
trans_back = torch.zeros(1, 3).to(device)

# # #-------------initialize EM -------
loaded_index = joblib.load("./smpl_models/SMPL_downsample_index.pkl")
selected_index = loaded_index["downsample_index"]


depthEM = surface_EM_depth(
    smplxmodel=smplmodel,
    batch_size=1,
    num_iters=3,
    selected_index=selected_index,
    device=device,
)


# load mesh and sampling
mesh = trimesh.load(input_file)
point_o = mesh.vertices
pts = torch.from_numpy(point_o).float()
index = farthest_point_sample(pts.unsqueeze(0), npoint=2048).squeeze()
pts = pts[index]


# # move to center
trans = torch.mean(pts, dim=0, keepdim=True)
pts = torch.sub(pts, trans)
point_arr = torch.transpose(pts, 1, 0)
point_arr = point_arr.unsqueeze(0).to(device)

point_arr2 = pts.unsqueeze(0).to(device)

# # do the inference
with torch.no_grad():
    pred_shape, pred_pose_body, pred_trans, pred_R6D = model(point_arr)  #

pred_R6D_3D = quaternion_to_axis_angle(matrix_to_quaternion((rotation_6d_to_matrix(pred_R6D))))

pred_pose[0, 3:] = pred_pose_body.unsqueeze(0).float()
pred_pose[0, :3] = pred_R6D_3D.unsqueeze(0).float()

pred_cam_t[0, :] = pred_trans.unsqueeze(0).float()
trans_back[0, :] = trans.unsqueeze(0).float()

pred_pose[0, 16 * 3 : 18 * 3] = (
    torch.Tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    .unsqueeze(0)
    .float()
)

new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t = depthEM(
    pred_pose.detach(), pred_betas.detach(), pred_cam_t.detach(), point_arr2
)

# Write the arrays to the file
np.savez(
    output_file,
    beta=new_opt_betas.detach().cpu().numpy().reshape(10),
    pose=new_opt_pose.detach().cpu().numpy().reshape(72),
)
