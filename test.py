import os

import pandas as pd

from SMPLfitter import SMPLfitter


fitter = SMPLfitter.SMPLfitter(smpl_gender="male")
init_pose_front, init_pose_back, init_betas, init_cam_trans_front, init_cam_trans_back = fitter.initialize_params()

fitter.save_smpl_ply(
    init_pose_front, init_betas, init_cam_trans_front, "test.ply"
)
    