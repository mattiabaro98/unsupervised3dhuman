from __future__ import division, print_function

import joblib
import smplx
import torch
import trimesh

from SMPLfitter.src.surfaceem import surface_EM_depth
from SMPLfitter.src.utils import farthest_point_sample


class SMPLfitter:
    """Implementation of SMPL fitter."""

    def __init__(
        self,
        smpl_gender="male",
    ):

        # Assets path
        self.smpl_params_path = "./SMPLfitter/smpl_models/"
        SMPL_downsample_index_path = "./SMPLfitter/smpl_models/SMPL_downsample_index.pkl"

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

        print("Point cloud centered")

        return points, trans

    def scale_pc(self, points, alpha):

        # Scale point cloud
        points = torch.mul(points, alpha)

        print("Point cloud scaled")

        return points

    def pose_default(self, trans):

        pred_pose = torch.zeros(1, 72).to(self.device)
        pred_betas = torch.zeros(1, 10).to(self.device)
        pred_cam_t = torch.zeros(1, 3).to(self.device)
        trans_back = torch.zeros(1, 3).to(self.device)

        trans_back[0, :] = trans.unsqueeze(0).float()

        print("Pose initialized")

        return pred_pose, pred_betas, pred_cam_t, trans_back

    def smpl_fit(self, points, pred_pose, pred_betas, pred_cam_t):

        # ---------- Fit SMPL model ----------

        points = points.unsqueeze(0).to(self.device)

        depthEM = surface_EM_depth(
            smplxmodel=self.smplmodel,
            batch_size=1,
            num_iters=100,
            selected_index=self.selected_index,
            device=self.device,
        )

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t = depthEM(
            pred_pose.detach(), pred_betas.detach(), pred_cam_t.detach(), points
        )

        print("SMPL parameters fitted")

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
