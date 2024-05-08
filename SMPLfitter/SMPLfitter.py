from __future__ import division, print_function

import joblib
import torch
import trimesh

from SMPLfitter.src.smpl import SMPL


class SMPLfitter:
    """Implementation of SMPL fitter."""

    def __init__(
        self,
        smpl_gender="male",
    ):

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print("Selected device:", self.device)

        # SMPL gender
        if smpl_gender == "male" or smpl_gender == "female":
            self.smpl_gender = smpl_gender
        else:
            print('Wrong gender parameter, "male" or "female" accepted.')

        # Initialize SMPL model
        self.smpl_params_path = "./SMPLfitter/smpl_models/smpl/"
        self.smplmodel = SMPL(self.smpl_params_path, gender=self.smpl_gender, device=self.device)

        # Downsample index
        SMPL_downsample_index_path = "./SMPLfitter/smpl_models/SMPL_downsample_index.pkl"
        self.selected_index = joblib.load(SMPL_downsample_index_path)["downsample_index"]


    def initialize_params(
        self,
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """Initilize Parameters
        Input:
        Output:
        """

        init_pose_front = torch.zeros(1, 72).to(self.device)
        init_pose_back = torch.zeros(1, 72).to(self.device)
        init_betas = torch.zeros(1, 10).to(self.device)
        init_cam_trans_front = torch.zeros(1, 3).to(self.device)
        init_cam_trans_back = torch.zeros(1, 3).to(self.device)

        print("Parameters initialized initialized")
        return init_pose_front, init_pose_back, init_betas, init_cam_trans_front, init_cam_trans_back


    def save_smpl_ply(
        self,
        pose: torch.tensor,
        betas: torch.tensor,
        cam_trans: torch.tensor,
        filename: str,
    ) -> None:
        """Save SMPL point cloud
        Input:
        Output:
        """

        output_vertices = self.smplmodel(betas=betas.squeeze(), pose=pose.squeeze(), trans=cam_trans.squeeze())
        mesh = trimesh.Trimesh(
            vertices=output_vertices.detach().cpu().numpy().squeeze(), faces=self.smplmodel.faces, process=False
        )
        mesh.export(filename)

        print(f"Predicted SMPL saved to {filename}")
        return None
