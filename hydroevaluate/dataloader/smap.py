import pandas as pd


def process_smapData_wrong(data, stcd, tolerance=0.05):
    # 处理 smapData
    smap_df = read_forcing_dataframe("smap", stcd, data)

    smap_df["predictdate"] = pd.to_datetime(smap_df["predictdate"])

    smap_df["predictdate"] = smap_df["predictdate"] - pd.Timedelta(minutes=30)

    # 计算 intersection_ratio
    smap_df["intersection_ratio"] = (
        smap_df["intersection_area"] / smap_df["raster_area"]
    )

    # 设置布尔掩码
    valid_mask = smap_df["intersection_ratio"] >= tolerance

    # 计算新的 sm_surface 列，仅对有效的行进行计算
    smap_df.loc[valid_mask, "sm_surface"] = (
        smap_df["sm_surface"] * smap_df["intersection_area"] / smap_df["raster_area"]
    )

    # 对相同 predictdate 的值计算平均，仅对有效的行进行 groupby 和 mean 运算
    smap_df = (
        smap_df[valid_mask].groupby("predictdate")["sm_surface"].mean().reset_index()
    )

    # 确保时间序列为3小时间隔，插值缺失数据
    full_time_range = pd.date_range(
        start=smap_df["predictdate"].min(),
        end=smap_df["predictdate"].max(),
        freq="3h",
    )
    smap_df = (
        smap_df.set_index("predictdate")
        .reindex(full_time_range)
        .interpolate(method="time")
        .reset_index()
    )
    smap_df = smap_df.rename(columns={"index": "time"})

    # 用第一个非空值填充开始的缺失值，用最后一个非空值填充结束的缺失值
    smap_df.fillna(method="ffill", inplace=True)
    smap_df.fillna(method="bfill", inplace=True)

    # 添加 basin 列
    smap_df["basin"] = stcd

    # 转换为 DataArray
    result_dataarray = to_dataarray(
        smap_df,
        dims=["time"],
        coords={"time": smap_df["time"]},
        name="sm_surface",
    )
    result_dataarray = result_dataarray.rename("sm_surface")
    result_dataarray = result_dataarray.expand_dims("basin").assign_coords(basin=[stcd])

    return result_dataarray


def process_smapData(data, stcd):
    # 处理 smapData
    smap_df = read_forcing_dataframe("smap", stcd, data)

    smap_df["predictdate"] = pd.to_datetime(smap_df["predictdate"])

    # 将所有时间提前半小时
    smap_df["predictdate"] = smap_df["predictdate"] - pd.Timedelta(minutes=30)

    # 计算 intersection_area 和 sm_surface 的乘积
    smap_df["intersection_area_sm_surface"] = (
        smap_df["intersection_area"] * smap_df["sm_surface"]
    )

    # 计算每个 predictdate 下的 intersection_area 和 intersection_area_sm_surface 的和
    grouped = (
        smap_df.groupby("predictdate")
        .agg({"intersection_area_sm_surface": "sum", "intersection_area": "sum"})
        .reset_index()
    )

    # 计算新的 sm_surface_cal_from_origin 列
    grouped["smap"] = (
        grouped["intersection_area_sm_surface"] / grouped["intersection_area"]
    )

    # 确保时间序列为3小时间隔，插值缺失数据
    full_time_range = pd.date_range(
        start=grouped["predictdate"].min(),
        end=grouped["predictdate"].max(),
        freq="3H",
    )
    grouped = (
        grouped.set_index("predictdate")
        .reindex(full_time_range)
        .interpolate(method="time")
        .reset_index()
    )
    grouped = grouped.rename(columns={"index": "predictdate"})

    # 用第一个非空值填充开始的缺失值，用最后一个非空值填充结束的缺失值
    grouped.fillna(method="ffill", inplace=True)
    grouped.fillna(method="bfill", inplace=True)

    # 修改列名
    grouped = grouped.rename(columns={"predictdate": "time"})

    # 添加 basin 列
    grouped["basin"] = stcd

    # 转换为 DataArray
    result_dataarray = to_dataarray(
        grouped,
        dims=["time"],
        coords={"time": grouped["time"]},
        name="smap",
    )
    result_dataarray = result_dataarray.rename("sm_surface")
    result_dataarray = result_dataarray.expand_dims("basin").assign_coords(basin=[stcd])

    return result_dataarray
