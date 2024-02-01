from SMPLfitter import SMPLfitter

# mic5_pro_rot.ply
# 00003200.ply
# shortshort_flying_eagle.000075_depth.ply

fitter = SMPLfitter.SMPLfitter(
    input_file="./examples/00003200.ply", output_file="./results/test.npz", smpl_gender="male"
)

fitter.fit()
