"""
Author: Wenyu Ouyang
Date: 2024-05-30 09:11:04
LastEditTime: 2024-05-31 15:48:11
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydroevaluate\tests\test_load_hydromodel.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pandas as pd
import pytest
from modelloader.model import load_hydromodel


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


# Run the tests
pytest.main([__file__])
