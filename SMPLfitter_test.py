import trimesh

from SMPLfitter import SMPLfitter

n = 2
front_input_file = f"./examples/ex{n}front_rot.ply"
back_input_file = f"./examples/ex{n}back_rot.ply"

fitter = SMPLfitter.SMPLfitter(smpl_gender="male")

front_points = fitter.load_pc(front_input_file)
front_sampled_points = fitter.sample_pc(front_points)
front_centered_points, front_center_trans = fitter.center_pc(front_sampled_points)

back_points = fitter.load_pc(back_input_file)
back_sampled_points = fitter.sample_pc(back_points)
back_centered_points, back_center_trans = fitter.center_pc(back_sampled_points)

init_pose, init_betas, init_scale, init_cam_trans, init_back_trans = fitter.initialize_params()

pred_pose, pred_betas, pred_scale, pred_cam_trans, combo_points = fitter.smpl_fit(front_centered_points, back_centered_points, init_pose, init_betas, init_scale, init_cam_trans, init_back_trans)

# Store results
mesh = trimesh.Trimesh()
mesh.vertices = combo_points.squeeze().cpu()
mesh.export(f"./results/ex{n}_frontback.ply")

fitter.save_smpl_ply(pred_pose, pred_betas, pred_scale, pred_cam_trans, f"./results/ex{n}_predicted.ply")
