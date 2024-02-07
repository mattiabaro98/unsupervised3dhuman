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

    def pose_default(self, trans):

        init_pose = torch.zeros(1, 72).to(self.device)
        init_betas = torch.zeros(1, 10).to(self.device)
        init_cam_t = torch.zeros(1, 3).to(self.device)
        trans_back = torch.zeros(1, 3).to(self.device)
        init_alpha = torch.tensor(1, dtype=torch.float).to(self.device)

        trans_back[0, :] = trans.unsqueeze(0).float()

        print("Pose initialized")

        return init_pose, init_betas, init_cam_t, trans_back, init_alpha

    def smpl_fit(self, points, init_pose, init_betas, init_cam_t, init_alpha):

        # ---------- Fit SMPL model ----------

        points = points.unsqueeze(0).to(self.device)

        depthEM = surface_EM_depth(
            smplxmodel=self.smplmodel,
            batch_size=1,
            num_iters=100,
            selected_index=self.selected_index,
            device=self.device,
        )

        pred_vertices, pred_joints, pred_pose, pred_betas, pred_cam_t, pred_alpha = depthEM(
            init_pose.detach(), init_betas.detach(), init_cam_t.detach(), init_alpha.detach(), points
        )

        print("SMPL parameters fitted")

        return pred_pose, pred_betas, pred_cam_t, pred_alpha

    def save_smpl_ply(self, betas, pose, cam_t, trans_back, alpha, filename):

        # save the final results
        output = self.smplmodel(
            betas=betas, global_orient=pose[:, :3], body_pose=pose[:, 3:], transl=cam_t + trans_back, return_verts=True
        )
        outputVerts = torch.mul(output.vertices, alpha).detach().cpu().numpy().squeeze()
        mesh = trimesh.Trimesh(vertices=outputVerts, faces=self.smplmodel.faces, process=False)
        mesh.export(filename)
