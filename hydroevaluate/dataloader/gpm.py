"""
Author: Shuolong Xu
Date: 2023-10-29 17:35:04
LastEditTime: 2024-06-03 15:53:54
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydroevaluate\hydroevaluate\dataloader\gpm.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import pandas as pd
from hydrodatasource.reader.postgres import read_forcing_dataframe
from hydroevaluate.utils.heutils import to_dataarray
import logging
logging.getLogger().setLevel(logging.WARNING)


def process_gpm_data(time_period, stcd, tolerance=0.05):
    # 处理 gpmData
    gpm_df = read_forcing_dataframe("gpm_tp", stcd, time_period)

    gpm_df["predictdate"] = pd.to_datetime(gpm_df["predictdate"])

    # 确保起始时间是半小时的，结束时间是整小时的
    if gpm_df["predictdate"].iloc[0].minute == 0:
        gpm_df = gpm_df.iloc[1:].reset_index(drop=True)

    if gpm_df["predictdate"].iloc[-1].minute == 30:
        gpm_df = gpm_df.iloc[:-1].reset_index(drop=True)

    # 计算 intersection_ratio
    gpm_df["intersection_ratio"] = gpm_df["intersection_area"] / gpm_df["raster_area"]

    # 设置布尔掩码
    valid_mask = gpm_df["intersection_ratio"] >= tolerance

    # 计算新的 gpm_tp 列，仅对有效的行进行计算
    gpm_df.loc[valid_mask, "gpm_tp"] = (
        gpm_df["tp"] * gpm_df["intersection_area"] / gpm_df["raster_area"]
    )

    # 对相同 predictdate 的值计算平均，仅对有效的行进行 groupby 和 mean 运算
    gpm_df = gpm_df[valid_mask].groupby("predictdate")["gpm_tp"].mean().reset_index()

    # 确保时间序列为半小时间隔，插值缺失数据
    full_time_range = pd.date_range(
        start=gpm_df["predictdate"].min(),
        end=gpm_df["predictdate"].max(),
        freq="30T",
    )
    gpm_df = (
        gpm_df.set_index("predictdate")
        .reindex(full_time_range)
        .interpolate(method="time")
        .reset_index()
    )
    gpm_df = gpm_df.rename(columns={"index": "predictdate"})

    # 用第一个非空值填充开始的缺失值，用最后一个非空值填充结束的缺失值
    gpm_df.fillna(method="ffill", inplace=True)
    gpm_df.fillna(method="bfill", inplace=True)

    # 将 predictdate 向上取整到最近的整点
    gpm_df["roundedDatetime"] = gpm_df["predictdate"].dt.ceil("H")
    gpm_df = gpm_df.groupby("roundedDatetime")["gpm_tp"].mean().reset_index()

    # 修改列名
    gpm_df = gpm_df.rename(columns={"roundedDatetime": "time"})

    # 添加 basin 列
    gpm_df["basin"] = stcd

    # 转换为 DataArray
    result_dataarray = to_dataarray(
        gpm_df,
        dims=["time"],
        coords={"time": gpm_df["time"]},
        name="gpm_tp",
    )
    result_dataarray = result_dataarray.expand_dims("basin").assign_coords(basin=[stcd])

    return result_dataarray
