import pandas as pd

from hydroevaluate.utils.heutils import convert_baseDatetime_iso


def process_gfsTp(time, stcd, tolerance=0.05):
    # 处理 gfsTp
    gfstp_df = read_forcing_dataframe("gfs_tp", stcd, time)

    gfstp_df["forecastdatetime"] = pd.to_datetime(gfstp_df["forecastdatetime"])

    # 计算 intersection_ratio
    gfstp_df["intersection_ratio"] = (
        gfstp_df["intersection_area"] / gfstp_df["raster_area"]
    )

    # 设置布尔掩码
    valid_mask = gfstp_df["intersection_ratio"] >= tolerance

    # 计算新的 gfs_tp 列，仅对有效的行进行计算
    gfstp_df.loc[valid_mask, "gfs_tp"] = (
        gfstp_df["tp"] * gfstp_df["intersection_area"] / gfstp_df["raster_area"]
    )

    # 对相同 predictdate 的值计算平均，仅对有效的行进行 groupby 和 mean 运算
    gfstp_df = (
        gfstp_df[valid_mask].groupby("forecastdatetime")["gfs_tp"].mean().reset_index()
    )

    # 确保时间序列为一小时间隔，插值缺失数据
    full_time_range = pd.date_range(
        start=gfstp_df["forecastdatetime"].min(),
        end=gfstp_df["forecastdatetime"].max(),
        freq="H",
    )
    gfstp_df = (
        gfstp_df.set_index("forecastdatetime")
        .reindex(full_time_range)
        .interpolate(method="time")
        .reset_index()
    )
    gfstp_df = gfstp_df.rename(columns={"index": "forecastdatetime"})

    # 用第一个非空值填充开始的缺失值，用最后一个非空值填充结束的缺失值
    gfstp_df.fillna(method="ffill", inplace=True)
    gfstp_df.fillna(method="bfill", inplace=True)

    # 修改列名
    result_df = gfstp_df.rename(columns={"forecastdatetime": "time"})

    # 添加 basin 列
    result_df["basin"] = stcd

    # 转换为 DataArray
    result_dataarray = to_dataarray(
        result_df,
        dims=["time"],
        coords={"time": result_df["time"]},
        name="gfs_tp",
    )
    result_dataarray = result_dataarray.rename("gpm_tp")
    result_dataarray = result_dataarray.expand_dims("basin").assign_coords(basin=[stcd])

    return result_dataarray


# TODO: 暂时用不到，就不改了
def process_gfsData(data, stcd):
    # 处理 gfsData
    for record in data.get("gfsData", []):
        record["baseDatetime"] = convert_baseDatetime_iso(record, "baseDatetime")
        record["forecastdatetime"] = convert_baseDatetime_iso(
            record, "forecastdatetime"
        )

    gfs_header = [
        "baseDatetime",
        "forecastdatetime",
        "intersection_area",
        "latitude",
        "longitude",
        "raster_area",
        "d2m",
        "t2m",
        "dswrf",
    ]
    gfs_df = pd.DataFrame(data.get("gfsData", []), columns=gfs_header)

    gfs_df["forecastdatetime"] = pd.to_datetime(gfs_df["forecastdatetime"])

    # 计算新的列 d2m, t2m, dswrf
    gfs_df["d2m"] = gfs_df["d2m"] * gfs_df["intersection_area"] / gfs_df["raster_area"]
    gfs_df["t2m"] = gfs_df["t2m"] * gfs_df["intersection_area"] / gfs_df["raster_area"]
    gfs_df["dswrf"] = (
        gfs_df["dswrf"] * gfs_df["intersection_area"] / gfs_df["raster_area"]
    )

    # 对相同 forecastdatetime 的值计算平均
    d2m_result_df = gfs_df.groupby("forecastdatetime")["d2m"].mean().reset_index()
    t2m_result_df = gfs_df.groupby("forecastdatetime")["t2m"].mean().reset_index()
    dswrf_result_df = gfs_df.groupby("forecastdatetime")["dswrf"].mean().reset_index()

    # 确保时间序列为一小时间隔，插值缺失数据
    d2m_full_time_range = pd.date_range(
        start=d2m_result_df["forecastdatetime"].min(),
        end=d2m_result_df["forecastdatetime"].max(),
        freq="H",
    )
    t2m_full_time_range = pd.date_range(
        start=d2m_result_df["forecastdatetime"].min(),
        end=d2m_result_df["forecastdatetime"].max(),
        freq="H",
    )
    dswrf_full_time_range = pd.date_range(
        start=d2m_result_df["forecastdatetime"].min(),
        end=d2m_result_df["forecastdatetime"].max(),
        freq="H",
    )
    d2m_result_df = (
        d2m_result_df.set_index("forecastdatetime")
        .reindex(d2m_full_time_range)
        .interpolate(method="time")
        .reset_index()
    )
    t2m_result_df = (
        t2m_result_df.set_index("forecastdatetime")
        .reindex(t2m_full_time_range)
        .interpolate(method="time")
        .reset_index()
    )
    dswrf_result_df = (
        dswrf_result_df.set_index("forecastdatetime")
        .reindex(dswrf_full_time_range)
        .interpolate(method="time")
        .reset_index()
    )
    d2m_result_df = d2m_result_df.rename(columns={"index": "forecastdatetime"})
    t2m_result_df = t2m_result_df.rename(columns={"index": "forecastdatetime"})
    dswrf_result_df = dswrf_result_df.rename(columns={"index": "forecastdatetime"})

    # 用第一个非空值填充开始的缺失值，用最后一个非空值填充结束的缺失值
    d2m_result_df.fillna(method="ffill", inplace=True)
    d2m_result_df.fillna(method="bfill", inplace=True)
    t2m_result_df.fillna(method="ffill", inplace=True)
    t2m_result_df.fillna(method="bfill", inplace=True)
    dswrf_result_df.fillna(method="ffill", inplace=True)
    dswrf_result_df.fillna(method="bfill", inplace=True)

    # 修改列名
    d2m_result_df = d2m_result_df.rename(columns={"forecastdatetime": "time"})
    t2m_result_df = t2m_result_df.rename(columns={"forecastdatetime": "time"})
    dswrf_result_df = dswrf_result_df.rename(columns={"forecastdatetime": "time"})

    # 添加 basin 列
    d2m_result_df["basin"] = stcd
    t2m_result_df["basin"] = stcd
    dswrf_result_df["basin"] = stcd

    # 转换为 DataArray
    d2m_dataarray = to_dataarray(
        d2m_result_df,
        dims=["time"],
        coords={"time": d2m_result_df["time"]},
        name="d2m",
    )
    d2m_dataarray = d2m_dataarray.expand_dims("basin").assign_coords(basin=[stcd])

    t2m_dataarray = to_dataarray(
        t2m_result_df,
        dims=["time"],
        coords={"time": t2m_result_df["time"]},
        name="t2m",
    )
    t2m_dataarray = t2m_dataarray.expand_dims("basin").assign_coords(basin=[stcd])

    dswrf_dataarray = to_dataarray(
        dswrf_result_df,
        dims=["time"],
        coords={"time": dswrf_result_df["time"]},
        name="dswrf",
    )
    dswrf_dataarray = dswrf_dataarray.expand_dims("basin").assign_coords(basin=[stcd])

    return d2m_dataarray, t2m_dataarray, dswrf_dataarray


def process_gfsSoilData(time, stcd):
    # 处理 gfsSoilData
    gfsSoil_df = read_forcing_dataframe("gfs_soil", stcd, time)

    gfsSoil_df["forecastdatetime"] = pd.to_datetime(gfsSoil_df["forecastdatetime"])

    # 计算 intersection_area 和 soilw 的乘积
    gfsSoil_df["intersection_area_soilw"] = (
        gfsSoil_df["intersection_area"] * gfsSoil_df["soilw"]
    )

    # 计算每个 forecastdatetime 下的 intersection_area 和 intersection_area_soilw 的和
    grouped = (
        gfsSoil_df.groupby("forecastdatetime")
        .agg({"intersection_area_soilw": "sum", "intersection_area": "sum"})
        .reset_index()
    )

    # 计算新的 soilw_cal_from_origin 列
    grouped["gfs_soil"] = (
        grouped["intersection_area_soilw"] / grouped["intersection_area"]
    )

    # 确保时间序列为一小时间隔，插值缺失数据
    full_time_range = pd.date_range(
        start=grouped["forecastdatetime"].min(),
        end=grouped["forecastdatetime"].max(),
        freq="H",
    )
    grouped = (
        grouped.set_index("forecastdatetime")
        .reindex(full_time_range)
        .interpolate(method="time")
        .reset_index()
    )
    grouped = grouped.rename(columns={"index": "forecastdatetime"})

    # 用第一个非空值填充开始的缺失值，用最后一个非空值填充结束的缺失值
    grouped.fillna(method="ffill", inplace=True)
    grouped.fillna(method="bfill", inplace=True)

    # 修改列名
    grouped = grouped.rename(columns={"forecastdatetime": "time"})

    # 添加 basin 列
    grouped["basin"] = stcd

    # 转换为 DataArray
    result_dataarray = to_dataarray(
        grouped, dims=["time"], coords={"time": grouped["time"]}, name="gfs_soil"
    )
    result_dataarray = result_dataarray.rename("sm_surface")
    result_dataarray = result_dataarray.expand_dims("basin").assign_coords(basin=[stcd])

    return result_dataarray
