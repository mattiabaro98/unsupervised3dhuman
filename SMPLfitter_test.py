from SMPLfitter import SMPLfitter


fitter = SMPLfitter.SMPLfitter(input_file="./examples/mic5_pro_rot.ply",
        output_file="./results/mic5_pro_rot_smpl_params.npz",
        smpl_gender="male")

fitter.fit()