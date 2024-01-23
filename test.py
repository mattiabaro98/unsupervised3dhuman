import h5py
import numpy as np

smpl_mean_file = "./smpl_models/neutral_smpl_mean_params.h5"

file = h5py.File(smpl_mean_file, "r")
init_pose = file["pose"]
init_shape = file["shape"]

np.savez(
    "init_params.npz",
    beta=file["shape"],
    pose=file["pose"],
)
