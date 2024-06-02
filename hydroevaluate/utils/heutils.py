"""
Author: Wenyu Ouyang
Date: 2024-05-30 09:11:04
LastEditTime: 2024-05-31 13:35:19
LastEditors: Wenyu Ouyang
Description: some common functions
FilePath: \hydroevaluate\hydroevaluate\utils\heutils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""


from hydroevaluate.hydroevaluate import work_dir


import urllib3 as ur
from yaml import Loader, load

import datetime
import xarray as xr

import os.path


def read_yaml(version):
    config_path = os.path.join(work_dir, 'test_data/aiff_config/aiff_v' + str(version) + '.yml')
    if not os.path.exists(config_path):
        version_url = 'https://raw.githubusercontent.com/iHeadWater/AIFloodForecast/main/scripts/conf/v' + str(
            version) + '.yml'
        yml_str = ur.request('GET', version_url).data.decode('utf8')
    else:
        with open(config_path, 'r') as fp:
            yml_str = fp.read()
    conf_yaml = load(yml_str, Loader=Loader)
    return conf_yaml



def convert_baseDatetime_iso(record, key):
    # 解析 ISO 8601 日期时间字符串
    dt = datetime.datetime.fromisoformat(record[key].replace("Z", "+00:00"))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def to_dataarray(df, dims, coords, name):
    # 将数据转换为 xarray.DataArray
    return xr.DataArray(df[name].values, dims=dims, coords=coords, name=name)
