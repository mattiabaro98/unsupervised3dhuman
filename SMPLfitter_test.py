import numpy as np
import trimesh

from SMPLfitter import SMPLfitter

# file_pc = "mic5_pro_rot.ply"
file_pc = "00003200.ply"
# file_pc = "shortshort_flying_eagle.000075_depth.ply"

input_file="./examples/" + file_pc

fitter = SMPLfitter.SMPLfitter(
    smpl_gender="male"
)

points = fitter.load_pc(input_file)

sampled_points = fitter.sample_pc(points)

centered_points, trans = fitter.center_pc(sampled_points)

print(trans)

# mesh = trimesh.Trimesh()
# mesh.vertices = sampled_points
# mesh.export("./results/sampled_point_cloud.ply")


# def _save_smpl_ply(self, betas, pose, cam_t, trans_back, filename):

#         # save the final results
#         output = self.smplmodel(
#             betas=betas, global_orient=pose[:, :3], body_pose=pose[:, 3:], transl=cam_t + trans_back, return_verts=True
#         )
#         mesh = trimesh.Trimesh(
#             vertices=output.vertices.detach().cpu().numpy().squeeze(), faces=self.smplmodel.faces, process=False
#         )
#         mesh.export(filename)

# mesh = trimesh.Trimesh()
# mesh.vertices = points
# mesh.export("./results/sampled_point_cloud.ply")

# # Save results
# np.savez(
#     self.output_file,
#     beta=new_opt_betas.detach().cpu().numpy().reshape(10),
#     pose=new_opt_pose.detach().cpu().numpy().reshape(72),
# )

# self._save_smpl_ply(pred_betas, pred_pose, pred_cam_t, trans_back, "./results/pest2.ply")

# np.savez(
#     self.output_file,
#     beta=pred_betas.detach().cpu().numpy().reshape(10),
#     pose=pred_pose.detach().cpu().numpy().reshape(72),
# )