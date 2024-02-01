from __future__ import division, print_function

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

from SMPLfitter.src.Network import point_net_ssg
from SMPLfitter.src.surfaceem import surface_EM_depth
from SMPLfitter.src.utils import farthest_point_sample


class SMPLfitter:
    """Implementation of SMPL fitter."""

    def __init__(
        self,
        smpl_gender="male",
    ):

        # Assets path
        self.point_net_ssg_params_path = "./SMPLfitter/pretrained/model_best_depth.pth"
        self.smpl_params_path = "./SMPLfitter/smpl_models/"
        SMPL_downsample_index_path = "./SMPLfitter/smpl_models/SMPL_downsample_index.pkl"
        neutral_smpl_mean_params_path = "./SMPLfitter/smpl_models/neutral_smpl_mean_params.npz"

        # SMPL gender
        if smpl_gender == "male" or smpl_gender == "female":
            self.smpl_gender = smpl_gender
        else:
            print('Wrong gender parameter, "male" or "female" accepted.')

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print("Selected device:", self.device)

        # Initial pose parameters
        arrays = np.load(neutral_smpl_mean_params_path)
        self.init_pose = arrays["pose"]
        self.init_beta = arrays["beta"]

        # Downsample index
        self.selected_index = joblib.load(SMPL_downsample_index_path)["downsample_index"]

        # Initialize SMPL model
        self.smplmodel = smplx.create(self.smpl_params_path, model_type="smpl", gender=self.smpl_gender, ext="pkl").to(
            self.device
        )

    def load_pc(self, input_file):
        # ---------- Load and preprocess point cloud ----------
        # Load pointcloud
        mesh = trimesh.load(input_file)
        points = mesh.vertices
        points = torch.from_numpy(points).float()
        print("Loaded point cloud with %s points" % points.shape[0])

        return points

    def sample_pc(self, points):

        # Sample point cloud to reduce number of points
        index = farthest_point_sample(
            points.unsqueeze(0), npoint=2048
        ).squeeze()  # Return sampled indexes from farthest_point_sample
        sampled_points = points[index]  # Select sampled indexes
        print("Sampled point cloud with %s points" % sampled_points.shape[0])
        return sampled_points

    def center_pc(self, points):

        # Center point cloud
        trans = torch.mean(points, dim=0, keepdim=True)
        points = torch.sub(points, trans)

        return points, trans

    def pose_initializer(self, points, trans):

        # Prepare transposed points, pass points and transposed points to device
        points_transposed = torch.transpose(points, 1, 0)
        points_transposed = points_transposed.unsqueeze(0).to(self.device)

        # ---------- PointNet Single Scale Grouping Model ----------
        # Initialize model
        model = (
            point_net_ssg(device=self.device, init_pose=self.init_pose, init_shape=self.init_beta)
            .to(self.device)
            .eval()
        )
        model.load_state_dict(torch.load(self.point_net_ssg_params_path, map_location=self.device))
        optimizer = optim.Adam(model.parameters())

        # Inference
        with torch.no_grad():
            pred_shape, pred_pose_body, pred_trans, pred_R6D = model(points_transposed)

        pred_pose = torch.zeros(1, 72).to(self.device)
        pred_betas = torch.zeros(1, 10).to(self.device)
        pred_cam_t = torch.zeros(1, 3).to(self.device)
        trans_back = torch.zeros(1, 3).to(self.device)

        pred_R6D_3D = quaternion_to_axis_angle(matrix_to_quaternion((rotation_6d_to_matrix(pred_R6D))))
        pred_pose[0, 3:] = pred_pose_body.unsqueeze(0).float()
        pred_pose[0, :3] = pred_R6D_3D.unsqueeze(0).float()
        pred_pose[0, 16 * 3 : 18 * 3] = torch.Tensor([0, 0, 0, 0, 0, 0]).unsqueeze(0).float()
        pred_cam_t[0, :] = pred_trans.unsqueeze(0).float()
        trans_back[0, :] = trans.unsqueeze(0).float()

        return pred_pose, pred_betas, pred_cam_t, trans_back

    def smpl_fit(self, points, pred_pose, pred_betas, pred_cam_t):

        # ---------- Fit SMPL model ----------

        points = points.unsqueeze(0).to(self.device)

        depthEM = surface_EM_depth(
            smplxmodel=self.smplmodel,
            batch_size=1,
            num_iters=10,
            selected_index=self.selected_index,
            device=self.device,
        )

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t = depthEM(
            pred_pose.detach(), pred_betas.detach(), pred_cam_t.detach(), points
        )

        return new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t

    def save_smpl_ply(self, betas, pose, cam_t, trans_back, filename):

        # save the final results
        output = self.smplmodel(
            betas=betas, global_orient=pose[:, :3], body_pose=pose[:, 3:], transl=cam_t + trans_back, return_verts=True
        )
        mesh = trimesh.Trimesh(
            vertices=output.vertices.detach().cpu().numpy().squeeze(), faces=self.smplmodel.faces, process=False
        )
        mesh.export(filename)
