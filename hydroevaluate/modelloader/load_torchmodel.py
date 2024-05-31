"""
Author: Wenyu Ouyang
Date: 2024-05-30 09:11:04
LastEditTime: 2024-05-31 14:11:46
LastEditors: Wenyu Ouyang
Description: Load model trained by torchhydro
FilePath: \hydroevaluate\hydroevaluate\modelloader\load_torchmodel.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np

from torchhydro.trainers.deep_hydro import DeepHydro

from hydroevaluate.conf.config import custom_cfg
from hydroevaluate.hydroevaluate import work_dir


def run_normal_dl(cfg_path):
    model = DeepHydro(custom_cfg(cfg_path)[0], custom_cfg(cfg_path)[1])
    eval_log, preds_xr, obss_xr = model.model_evaluate()
    # preds_xr.to_netcdf(os.path.join("results", "v002_test", "preds.nc"))
    # obss_xr.to_netcdf(os.path.join("results", "v002_test", "obss.nc"))
    # print(eval_log)
    return eval_log, preds_xr, obss_xr


def read_history_model(user_model_type="wasted", version="1"):
    history_dict_path = os.path.join(work_dir, "test_data/history_dict.npy")
    # 姑且假设所有模型都被放在test_data/models文件夹下
    if not os.path.exists(history_dict_path):
        history_dict = _read_history_model(user_model_type, version, history_dict_path)
    else:
        history_dict = np.load(history_dict_path, allow_pickle=True).flatten()[0]
        model_file_name = f"{user_model_type}_v{str(version)}.pth"
        if model_file_name not in history_dict.keys():
            history_dict[user_model_type] = version

    return history_dict


def _read_history_model(user_model_type, version, history_dict_path):
    result = {}
    models = os.listdir(os.path.join(work_dir, "test_data/models"))
    # 姑且假设model的名字为wasted_v1.pth，即用途_版本.pth
    current_max = 0
    for model_name in models:
        if user_model_type in model_name:
            model_ver = int(model_name.split(".")[0].split("v")[1])
            if model_ver > current_max:
                current_max = model_ver
    model_file_name = f"{user_model_type}_v{str(version)}.pth"
    if model_file_name in models:
        result[user_model_type] = current_max
        np.save(history_dict_path, result, allow_pickle=True)
    return result
