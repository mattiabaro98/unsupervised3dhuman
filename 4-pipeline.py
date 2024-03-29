import json
import os

import pandas as pd

from SMPLfitter import SMPLfitter
from SMPLmeasure import SMPLmeasure
from utils.preprocess import preprocess

df_ground_truth = pd.read_csv("./examples/ground_truth.csv").set_index("info")
ground_truth_dict = df_ground_truth.to_dict()

anonimos = [
    "anonimo1",
    # "anonimo2",
    # "anonimo3",
    # "anonimo4",
    # "anonimo5",
    # "anonimo6",
    # "anonimo7",
    # "anonimo8",
    # "anonimo9",
    # "anonimo10",
    # "anonimo11",
    # "anonimo12",
    # "anonimo13",
    # "anonimo14",
    # "anonimo15",
    # "anonimo16",
    # "anonimo17",
    # "anonimo18",
    # "anonimo19",
    # "anonimo20",
    # "anonimo21",
]


def pipeline(front_path, back_path, gender, height, result_path):

    front_path_preprocessed = result_path + "front_proc.ply"
    back_path_preproceessed = result_path + "back_proc.ply"
    preprocess(front_path, front_path_preprocessed, "front")
    preprocess(back_path, back_path_preproceessed, "back")

    fitter = SMPLfitter.SMPLfitter(smpl_gender=gender)

    front_points = fitter.load_pc(front_path_preprocessed)
    front_sampled_points = fitter.sample_pc(front_points)
    front_centered_points, front_center_trans = fitter.center_pc(front_sampled_points)

    back_points = fitter.load_pc(back_path_preproceessed)
    back_sampled_points = fitter.sample_pc(back_points)
    back_centered_points, back_center_trans = fitter.center_pc(back_sampled_points)

    init_pose_front, init_pose_back, init_betas, init_cam_trans_front, init_cam_trans_back = fitter.initialize_params()

    pred_pose_front, pred_pose_back, pred_betas, pred_cam_trans_front, pred_cam_trans_back = fitter.smpl_fit(
        front_centered_points,
        back_centered_points,
        init_pose_front,
        init_pose_back,
        init_betas,
        init_cam_trans_front,
        init_cam_trans_back,
    )

    fitter.save_smpl_ply(
        pred_pose_front, pred_betas, pred_cam_trans_front + front_center_trans, result_path + "front_pred.ply"
    )
    fitter.save_smpl_ply(
        pred_pose_back, pred_betas, pred_cam_trans_back + back_center_trans, result_path + "back_pred.ply"
    )

    measure_smpl = SMPLmeasure.SMPLmeasure()
    result = measure_smpl.measure_smpl(filename=result_path + "front_pred.ply", height=height)

    return result


for anonimo in anonimos:

    samples = os.listdir(f"./examples/{anonimo}/")
    for sample in samples:
        print(
            "-----------------------------------------------------------------------------------------------------------------------------"
        )
        print(anonimo, sample, ground_truth_dict[anonimo]["gender"], ground_truth_dict[anonimo]["height"])
        front_path = f"./examples/{anonimo}/{sample}/a.ply"
        back_path = f"./examples/{anonimo}/{sample}/c.ply"
        result_path = f"./results/{anonimo}_{sample}/"

        if not os.path.exists(result_path):
            os.mkdir(result_path)

        result = pipeline(
            front_path,
            back_path,
            ground_truth_dict[anonimo]["gender"],
            float(ground_truth_dict[anonimo]["height"]),
            result_path,
        )

        with open(f"./results/{anonimo}_{sample}.json", "w") as json_file:
            json.dump(result, json_file, indent=4)
