"""
Author: Wenyu Ouyang
Date: 2024-05-30 09:11:04
LastEditTime: 2024-05-31 15:56:10
LastEditors: Wenyu Ouyang
Description: Load hydromodel
FilePath: \hydroevaluate\hydroevaluate\modelloader\load_hydromodel.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import json
import numpy as np
import pandas as pd
import pint
from hydrodatasource.utils.utils import streamflow_unit_conv
from hydromodel.models.model_dict import MODEL_DICT


def load_hydromodel(
    p_and_e: np.ndarray,
    area,
    calibrated_norm_param_file,
    param_range_file,
    model_info_file: str = None,
    target_unit="m^3/s",
):
    """Directly load the calibrated model with the given parameters
    one-time call for only one basin now

    Parameters
    ----------
    model_info : _type_
        _description_
    p_and_e : _type_
        _description_
    area:
        _description_
    calibrated_norm_param_file : str
        calibrated norm parameters file
    param_range_file : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if model_info_file is None:
        model_info = {
            "name": "xaj",
            "source_book": "HF",
            "source_type": "sources5mm",
            "time_interval_hours": 3,
        }
    else:
        model_info = json.load(open(model_info_file, "r"))
    calibrated_norm_params = pd.read_csv(calibrated_norm_param_file, index_col=0).values
    qsim, _ = MODEL_DICT[model_info["name"]](
        p_and_e,
        calibrated_norm_params,
        # we set the warmup_length=0 but later we get results from warmup_length to the end to evaluate
        warmup_length=0,
        **model_info,
        **{"param_range_file": param_range_file},
    )
    ureg = pint.UnitRegistry()
    ureg.force_ndarray_like = True
    q_sim_with_unit = qsim * ureg.mm / ureg.h / 3
    area_np_with_unit = area * ureg.km**2
    streamflow = streamflow_unit_conv(
        q_sim_with_unit, area_np_with_unit, target_unit, True
    )
    return streamflow
