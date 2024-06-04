"""
Author: Wenyu Ouyang
Date: 2024-05-30 09:11:04
LastEditTime: 2024-06-02 13:42:58
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydroevaluate\tests\test_model.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch
from torchhydro.models.cudnnlstm import CpuLstmModel
from hydroevaluate.modelloader.model import load_hydromodel, load_torchmodel


def test_load_hydromodel():
    tmpdir = "C:\\Users\\wenyu\\Downloads\\test_hydromodel_data"
    p_and_e_file = os.path.join(tmpdir, "21401550.csv")
    p_and_e = pd.read_csv(p_and_e_file, index_col=0)[["rain", "pet"]].values.reshape(
        -1, 1, 2
    )
    calibrated_norm_param_file = os.path.join(tmpdir, "basins_norm_params.csv")
    param_range_file = os.path.join(tmpdir, "21401550.yaml")
    area = 2055.56  # Example area value

    # Call the load_hydromodel function
    result = load_hydromodel(
        p_and_e,
        area,
        calibrated_norm_param_file,
        param_range_file,
    )
    flattened_array = result.flatten()
    df = pd.DataFrame(flattened_array, columns=["qsim"])
    file_path = os.path.join(tmpdir, "simq.csv")
    df.to_csv(file_path, index=False)
    # Perform assertions to validate the result
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)  # Example expected shape of the result

    # Add more assertions as needed to validate the result


def test_load_torchmodel(tmp_path):
    model_name = "CpuLSTM"
    model_hyperparam = {
        "n_input_features": 10,
        "n_output_features": 2,
        "n_hidden_states": 5,
    }
    pth_path = os.path.join(tmp_path, "model_tmp.pth")
    model_ = CpuLstmModel(**model_hyperparam)
    # Save the model state_dict instead of using a non-existent save method
    torch.save(model_.state_dict(), pth_path)
    # Call the load_torchmodel function
    model = load_torchmodel(model_name, model_hyperparam, pth_path)

    # Perform assertions to validate the result
    assert isinstance(model, torch.nn.Module)
    # Add more assertions as needed to validate the result
    # sourcery skip: no-loop-in-tests
    for param_tensor in model_.state_dict():
        assert torch.equal(
            model_.state_dict()[param_tensor], model.state_dict()[param_tensor]
        )
