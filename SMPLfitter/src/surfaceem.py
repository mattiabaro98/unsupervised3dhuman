import json

import numpy as np
import torch
from tqdm import tqdm

from SMPLfitter.src.customloss import body_fitting_loss_em
from SMPLfitter.src.prior import MaxMixturePrior


#  surface EM
class surface_EM_depth:
    """Implementation of SMPLify, use surface."""

    def __init__(
        self,
        smplxmodel,
        learning_rate=1e-1,
        batch_size=1,
        num_iters=100,
        selected_index=np.arange(6890),
        device=torch.device("cuda:0"),
        mu=0.05,
    ):

        # Store options
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder="./SMPLfitter/smpl_models/", num_gaussians=8, dtype=torch.float32).to(device)
        # Load SMPL-X model
        self.smpl = smplxmodel
        self.modelfaces = torch.from_numpy(np.int32(smplxmodel.faces)).to(device)
        self.selected_index = selected_index

        # mu prob
        self.mu = mu

    @torch.no_grad()
    def prob_cal(self, modelVerts_in, meshVerts_in, sigma=0.05**2, mu=0.02):
        modelVerts = torch.squeeze(modelVerts_in)
        meshVerts = torch.squeeze(meshVerts_in)

        model_x, model_y, model_z = torch.split(modelVerts, [1, 1, 1], dim=1)
        mesh_x, mesh_y, mesh_z = torch.split(meshVerts, [1, 1, 1], dim=1)

        M = model_x.shape[0]
        N = mesh_x.shape[0]

        delta_x = torch.repeat_interleave(torch.transpose(mesh_x, 0, 1), M, dim=0) - torch.repeat_interleave(model_x, N, dim=1)
        delta_y = torch.repeat_interleave(torch.transpose(mesh_y, 0, 1), M, dim=0) - torch.repeat_interleave(model_y, N, dim=1)
        delta_z = torch.repeat_interleave(torch.transpose(mesh_z, 0, 1), M, dim=0) - torch.repeat_interleave(model_z, N, dim=1)

        deltaVerts = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z

        mu_c = ((2.0 * torch.asin(torch.tensor(1.0)) * sigma) ** (3.0 / 2.0) * mu * M) / ((1 - mu) * N)
        deltaExp = torch.exp(-deltaVerts / (2 * sigma))
        deltaExpN = torch.repeat_interleave(torch.reshape(torch.sum(deltaExp, dim=0), (1, N)), M, dim=0)
        probArray = deltaExp / (deltaExpN + mu_c)

        Ind = torch.where(probArray > 1e-6)  # 2e-7
        modelInd, meshInd = Ind[0], Ind[1]
        probInput = probArray[Ind]

        return probInput, modelInd, meshInd

    # ---- get the man function hrere
    def __call__(self, init_pose, init_betas, init_scale, init_cam_trans, meshVerts):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose
            init_betas: SMPL betas
            init_cam_trans: Camera translation
            meshVerts: point3d from mesh
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """

        global_orient = init_pose[:, :3].detach().clone()
        body_pose = init_pose[:, 3:].detach().clone()
        scale = init_scale.detach().clone()
        betas = init_betas.detach().clone()
        camera_translation = init_cam_trans.clone()

        preserve_betas = init_betas.detach().clone()
        preserve_pose = init_pose[:, 3:].detach().clone()

        global_orient.requires_grad = True
        body_pose.requires_grad = True
        betas.requires_grad = True
        scale.requires_grad = True
        camera_translation.requires_grad = True
        body_opt_params = [body_pose, global_orient, betas, scale, camera_translation]
        body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=20, lr=self.learning_rate, line_search_fn="strong_wolfe")

        store_epoch = {"loss": [], "betas": [], "global_orient": [], "body_pose": [], "scale": [], "camera_translation": [], "sigma": []}

        for i in tqdm(range(self.num_iters)):

            def closure():
                body_optimizer.zero_grad()
                smpl_output = self.smpl(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas,
                    transl=camera_translation,
                    return_verts=True,
                )

                modelVerts = smpl_output.vertices[:, self.selected_index]
                scaled_modelVerts = torch.mul(modelVerts, scale)

                sigma = (0.1**2) * (self.num_iters - i + 1) / self.num_iters
                probInput, modelInd, meshInd = self.prob_cal(scaled_modelVerts, meshVerts, sigma=sigma, mu=self.mu)

                loss = body_fitting_loss_em(
                    body_pose,
                    preserve_pose,
                    betas,
                    preserve_betas,
                    scaled_modelVerts,
                    meshVerts,
                    modelInd,
                    meshInd,
                    probInput,
                    self.pose_prior,
                    smpl_output,
                    self.modelfaces,
                    pose_prior_weight=4.78 * 3.0,
                    shape_prior_weight=4.0,
                    angle_prior_weight=15.2,
                    betas_preserve_weight=4.0,
                    pose_preserve_weight=3.0,
                    chamfer_weight=100.0,
                    correspond_weight=1000.0,
                    point2mesh_weight=200.0,
                )

                loss.backward()
                return loss

            loss = body_optimizer.step(closure)

            store_epoch["loss"].append(loss.tolist()),
            store_epoch["betas"].append(betas.tolist()[0]),
            store_epoch["global_orient"].append(global_orient.tolist()[0]),
            store_epoch["body_pose"].append(body_pose.tolist()[0]),
            store_epoch["scale"].append(scale.tolist()),
            store_epoch["camera_translation"].append(camera_translation.tolist()[0])
            store_epoch["sigma"].append((0.1**2) * (self.num_iters - i + 1) / self.num_iters)

            with open("./store_epoch.json", "w") as f:
                json.dump(store_epoch, f)

        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()
        scale = scale.detach()
        camera_translation = camera_translation.detach()

        return pose, betas, scale, camera_translation
