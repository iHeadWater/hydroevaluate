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
import matplotlib.pyplot as plt
import logging

logging.getLogger().setLevel(logging.WARNING)

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
    final_data.columns = ['basin', 'gpm_tp', 'time']
    
    return final_data


# Define the correct conversion function with accurate parsing
def convert_index_to_correct_date(index):
    base_date = pd.to_datetime(index[:8], format='%Y%m%d')
    hour_increment = int(index[11:14])  # Adjust the slicing to correctly extract hours after 'F'
    return base_date + pd.Timedelta(hours=hour_increment)

def gee_gfs_tp_data_process(csv_path):
    df = pd.read_csv(csv_path)

    # Select relevant columns
    df = df[['basin_id', 'system:index', 'total_precipitation_surface']]

    # Apply the correct conversion function
    df['time'] = df['system:index'].apply(lambda x: pd.to_datetime(x[:10], format='%Y%m%d%H') + pd.Timedelta(hours=int(x[11:14])))

    # Drop the original 'system:index' column
    df = df.drop(columns=['system:index'])

    df = df.rename(columns={
        'basin_id': 'basin',
        'total_precipitation_surface': 'gfs_tp_origin'
    })
    # 排序数据，以确保时间顺序正确
    df = df.sort_values(by='time').reset_index(drop=True)

    # Extracting the 'gfs_tp' column
    gfs_tp = df['gfs_tp_origin'].values

    # Processing the 'gfs_tp' column as per the given instructions
    processed_gfs_tp = []
    for i in range(len(gfs_tp)):
        if i % 6 == 0:
            # First hour of each 6-hour block
            processed_gfs_tp.append(gfs_tp[i])
        else:
            # Subsequent hours of each 6-hour block
            processed_gfs_tp.append(gfs_tp[i] - gfs_tp[i-1])

    # Adding the processed 'gfs_tp' column back to the dataframe
    df['gfs_tp'] = processed_gfs_tp
    df = df.drop(columns=['gfs_tp_origin'])
    # Display the dataframe to the user
    return df

def calculate_nse(observed_csv, simulated_csv, column_name):
    """
    计算两个CSV文件中指定列的NSE指标，先取time维度的交集。

    参数：
    observed_csv (str): 观测值CSV文件路径
    simulated_csv (str): 模拟值CSV文件路径
    column_name (str): 要比较的列名

    返回：
    float: NSE指标
    """
    # 读取CSV文件
    observed_df = pd.read_csv(observed_csv)
    simulated_df = pd.read_csv(simulated_csv)

    # 确保 time 列存在
    if 'time' not in observed_df.columns or 'time' not in simulated_df.columns:
        raise ValueError("Both CSV files must contain a 'time' column")

    # 将 time 列转换为 datetime 类型以确保正确的合并
    observed_df['time'] = pd.to_datetime(observed_df['time'])
    simulated_df['time'] = pd.to_datetime(simulated_df['time'])
    
    # 取 time 维度的交集
    merged_df = pd.merge(observed_df, simulated_df, on='time', suffixes=('_obs', '_sim'))

    # 提取交集后的指定列的数据
    observed = merged_df[f'{column_name}_obs']
    simulated = merged_df[f'{column_name}_sim']

    # 计算观测值的平均值
    observed_mean = observed.mean()

    # 计算NSE
    numerator = ((observed - simulated) ** 2).sum()
    denominator = ((observed - observed_mean) ** 2).sum()
    nse = 1 - (numerator / denominator)

    return nse

def plot_time_series(observed_csv, simulated_csv, column_name):
    """
    绘制两个CSV文件中指定列随时间变化的图。

    参数：
    observed_csv (str): 观测值CSV文件路径
    simulated_csv (str): 模拟值CSV文件路径
    column_name (str): 要比较的列名
    """
    # 读取CSV文件
    observed_df = pd.read_csv(observed_csv)
    simulated_df = pd.read_csv(simulated_csv)

    # 确保 time 列存在
    if 'time' not in observed_df.columns or 'time' not in simulated_df.columns:
        raise ValueError("Both CSV files must contain a 'time' column")

    # 将 time 列转换为 datetime 类型以确保正确的合并
    observed_df['time'] = pd.to_datetime(observed_df['time'])
    simulated_df['time'] = pd.to_datetime(simulated_df['time'])

    # 取 time 维度的交集
    merged_df = pd.merge(observed_df, simulated_df, on='time', suffixes=('_obs', '_sim'))

    # 提取交集后的时间和指定列的数据
    time = merged_df['time']
    observed = merged_df[f'{column_name}_obs']
    simulated = merged_df[f'{column_name}_sim']

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(time, observed, label='Observed', color='blue')
    plt.plot(time, simulated, label='Simulated', color='red', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel(column_name)
    plt.title(f'Time Series of {column_name}')
    plt.legend()
    plt.grid(True)
    plt.show()