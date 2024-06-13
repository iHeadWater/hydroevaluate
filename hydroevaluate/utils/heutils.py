"""
Author: Wenyu Ouyang
Date: 2024-05-30 09:11:04
LastEditTime: 2024-05-31 13:35:19
LastEditors: Wenyu Ouyang
Description: some common functions
FilePath: \\hydroevaluate\\hydroevaluate\\utils\\heutils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import urllib3 as ur
from yaml import Loader, load

import datetime
import xarray as xr

import os.path

import pandas as pd


# def read_yaml(version):
#     config_path = os.path.join(
#         work_dir, "test_data/aiff_config/aiff_v" + str(version) + ".yml"
#     )
#     if not os.path.exists(config_path):
#         version_url = (
#             "https://raw.githubusercontent.com/iHeadWater/AIFloodForecast/main/scripts/conf/v"
#             + str(version)
#             + ".yml"
#         )
#         yml_str = ur.request("GET", version_url).data.decode("utf8")
#     else:
#         with open(config_path, "r") as fp:
#             yml_str = fp.read()
#     conf_yaml = load(yml_str, Loader=Loader)
#     return conf_yaml


def convert_baseDatetime_iso(record, key):
    # 解析 ISO 8601 日期时间字符串
    dt = datetime.datetime.fromisoformat(record[key].replace("Z", "+00:00"))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def to_dataarray(df, dims, coords, name):
    # 将数据转换为 xarray.DataArray
    return xr.DataArray(df[name].values, dims=dims, coords=coords, name=name)

def gee_gpm_to_1h_data(csv_path):
    """
    Args:
        csv_path (_type_): gee_gpm_csv, do not generate the shape column in the csv

    Returns:
        final_data : gpm 1h mean data, csv type
    """
    # Load the CSV file, ensuring BASIN_ID is read as a string
    data = pd.read_csv(csv_path, dtype={'BASIN_ID': str})
    
    # Convert 'time_start' to datetime
    data['time_start'] = pd.to_datetime(data['time_start'])
    
    # Set 'time_start' as index for resampling
    data.set_index('time_start', inplace=True)
    
    # Extract the 'BASIN_ID' for the first row of each hour
    basin_id = data['BASIN_ID'].resample('H').first()
    
    # Select only the numeric columns for resampling
    numeric_data = data[['precipitationCal']]
    
    # Resample to hourly frequency, taking the mean for 'precipitationCal'
    resampled_data = numeric_data.resample('H').mean()
    
    # Combine the resampled data with the corresponding 'BASIN_ID'
    resampled_data['BASIN_ID'] = basin_id.values
    
    # Reset index to move 'time_start' back to a column
    resampled_data.reset_index(inplace=True)
    
    # Select and rename columns as required
    final_data = resampled_data[['BASIN_ID', 'precipitationCal', 'time_start']]
    final_data.columns = ['basin', 'precipitationCal', 'time']
    
    return final_data