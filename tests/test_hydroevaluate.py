"""
Author: Wenyu Ouyang
Date: 2024-05-30 09:11:04
LastEditTime: 2024-06-01 11:24:25
LastEditors: Wenyu Ouyang
Description: Test cases for EvalDeepHydro
FilePath: \hydroevaluate\tests\test_hydroevaluate.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import omegaconf
from hydroevaluate.hydroevaluate import EvalDeepHydro


def test_load_config():
    config_path = "conf"
    config_name = "default_config.yml"
    eval_deep_hydro = EvalDeepHydro(config_path, config_name)
    assert isinstance(eval_deep_hydro.cfg, omegaconf.dictconfig.DictConfig)
    assert "data_cfgs" in eval_deep_hydro.cfg
    assert "model_cfgs" in eval_deep_hydro.cfg
