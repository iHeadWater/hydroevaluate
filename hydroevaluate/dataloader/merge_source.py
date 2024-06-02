"""
Author: Wenyu Ouyang
Date: 2024-02-12 09:52:49
LastEditTime: 2024-06-02 16:16:47
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydroevaluate\hydroevaluate\dataloader\merge_source.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""


def get_fake_forcing_dataset(basins, start_time):
    """
    创建一个随机值的xr.Dataset，具有指定的basin和time维度，变量值为['gpm_tp', 'sm_surface', 'sm_rootzone']。

    参数:
    - basins: list, 包含basin的列表
    - end_time: str or pd.Timestamp, time的结束时间

    返回:
    - xr.Dataset, 具有指定维度和随机值的Dataset
    """
    # 确保end_time为pandas.Timestamp类型
    start_time = pd.Timestamp(start_time) + pd.Timedelta(hours=1)

    # 生成时间序列
    end_time = start_time + pd.Timedelta(hours=887)
    time = pd.date_range(end=end_time, periods=297, freq="3h")

    # 创建随机数据
    gpm_tp = np.random.rand(len(basins), len(time))
    sm_surface = np.random.rand(len(basins), len(time))
    sm_rootzone = np.random.rand(len(basins), len(time))

    # 创建xr.Dataset
    data = xr.Dataset(
        {
            "gpm_tp": (["basin", "time"], gpm_tp),
            "sm_surface": (["basin", "time"], sm_surface),
            "sm_rootzone": (["basin", "time"], sm_rootzone),
        },
        coords={"basin": basins, "time": time},
    )

    return data


# 检查在gpm_tp中是否存在该时间点及前720小时的数据
def has_720_hours_before(time_array, start_time):
    start_time = start_time - pd.Timedelta(hours=3)
    time_array = pd.to_datetime(time_array)
    return any(time_array >= start_time)


# 检查在sm_smap中是否存在该时间点及前720小时的数据
def has_720_hours_before_sm_smap(time_array, start_time):
    start_time = start_time - pd.Timedelta(hours=3)
    time_array = pd.to_datetime(time_array)
    return any(time_array >= start_time)


def aggregate_dataset(ds: xr.Dataset, basin_id: str, gap: str = "3h") -> xr.Dataset:
    if gap == "3h":
        gap_hours = 3
        start_times = [2, 5, 8, 11, 14, 17, 20, 23]
        end_times = [1, 4, 7, 10, 13, 16, 19, 22]
        time_index = ds.indexes["time"]

        # 修剪开始时间
        while time_index[0].hour not in start_times:
            ds = ds.isel(time=slice(1, None))
            time_index = ds.indexes["time"]

        # 修剪结束时间
        while time_index[-1].hour not in end_times:
            ds = ds.isel(time=slice(None, -1))
            time_index = ds.indexes["time"]

    df_res = ds.to_dataframe().reset_index()
    df_res.set_index("time", inplace=True)

    numeric_cols = df_res.select_dtypes(include=[np.number]).columns
    aggregated_data = {}

    for col in numeric_cols:
        data = df_res[col].values
        aggregated_values = []
        for start in range(0, len(data), gap_hours):
            chunk = data[start : start + gap_hours]
            if np.isnan(chunk).any():
                aggregated_values.append(np.nan)
            else:
                aggregated_values.append(np.sum(chunk))

        aggregated_times = df_res.index[gap_hours - 1 :: gap_hours][
            : len(aggregated_values)
        ]
        aggregated_data[col] = (
            ("time", "basin"),
            np.array(aggregated_values).reshape(-1, 1),
        )

    result_ds = xr.Dataset(
        aggregated_data,
        coords={"time": aggregated_times, "basin": [basin_id]},
    )

    result_ds = result_ds.transpose("basin", "time")

    return result_ds


def prepare_forcing_now(init_time, stcd):
    start_time = pd.Timestamp(init_time)
    data_start_time = start_time - pd.Timedelta(hours=72)
    end_time = start_time + pd.Timedelta(hours=888)
    gpm_tp = process_gpmData(data_start_time, stcd)
    gpm_tp["time"] = gpm_tp["time"] + pd.Timedelta(hours=8)
    sm_smap = process_smapData(data_start_time, stcd)
    sm_smap["time"] = sm_smap["time"] + pd.Timedelta(hours=8)

    # 选取gpm_tp和smap_tp最后的时间点
    time_gpm_tp_last = gpm_tp["time"].values[-1]
    time_sm_smap_last = sm_smap["time"].values[-1]

    # 计算 gfs_tp 和 gfs_soil 的起始时间点
    time_gfs_tp_start = (
        pd.Timestamp(time_gpm_tp_last) + pd.Timedelta(hours=1) - pd.Timedelta(hours=8)
    )
    time_gfs_soil_start = (
        pd.Timestamp(time_sm_smap_last) + pd.Timedelta(hours=3) - pd.Timedelta(hours=8)
    )

    gfs_soil = process_gfsSoilData(time_gfs_soil_start, stcd)
    gfs_soil["time"] = gfs_soil["time"] + pd.Timedelta(hours=8)
    gfs_tp = process_gfsTp(time_gfs_tp_start, stcd)
    gfs_tp["time"] = gfs_tp["time"] + pd.Timedelta(hours=8)

    if not has_720_hours_before(gpm_tp["time"].values, data_start_time):
        raise ValueError("gpm data not found")

    if not has_720_hours_before_sm_smap(sm_smap["time"].values, data_start_time):
        raise ValueError("smap data not found")

    # 筛选数据
    sm_smap_filtered = sm_smap.sel(time=slice(start_time, time_sm_smap_last))
    gfs_soil_filtered = gfs_soil.sel(time=slice(time_gfs_soil_start, end_time))
    gfs_soil_filtered_every_3_hours = gfs_soil_filtered.isel(time=slice(0, None, 3))

    gpm_tp_filtered = gpm_tp.sel(time=slice(None, time_gpm_tp_last))
    gfs_tp_filtered = gfs_tp.sel(time=slice(time_gfs_tp_start, None))

    # 合并数据
    gpm_tp_input = xr.concat([gpm_tp_filtered, gfs_tp_filtered], dim="time")
    gpm_tp_input = aggregate_dataset(gpm_tp_input, stcd)
    gpm_tp_input = gpm_tp_input.sel(time=slice(start_time, end_time))
    sm_input = xr.concat(
        [sm_smap_filtered, gfs_soil_filtered_every_3_hours], dim="time"
    )

    data = xr.merge([gpm_tp_input, sm_input])
    time_range = pd.date_range(start=start_time, end=end_time, freq="3H")
    data_full = xr.Dataset(
        {
            "gpm_tp": (("basin", "time"), np.zeros((1, len(time_range)))),
            "sm_surface": (("basin", "time"), np.zeros((1, len(time_range)))),
        },
        coords={"time": time_range, "basin": [str(stcd)]},
    )
    data_full["gpm_tp"].loc[{"time": data.time}] = data["gpm_tp"]
    data_full["sm_surface"].loc[{"time": data.time}] = data["sm_surface"]
    logger.warning(data)
    logger.warning(data_full)

    return data_full


def test_format(data, stcd):
    # 处理 gpmdata
    result_dataarray = process_gpmData(data, stcd)
    print("gpm_result_dataarray:")
    print(result_dataarray)

    # 处理 smapdata
    result_dataarray = process_smapData(data, stcd)
    print("smap_result_dataarray:")
    print(result_dataarray)

    # 处理 gfsSoilData
    result_dataarray = process_gfsSoilData(data, stcd)
    print("gfssoil_result_dataarray:")
    print(result_dataarray)

    # 处理 gfsTp
    result_dataarray = process_gfsTp(data, stcd)
    print("gfstp_result_dataarray:")
    print(result_dataarray)

    # 处理 gfsData
    d2m_result_dataarray, t2m_result_dataarray, dswrf_result_dataarray = (
        process_gfsData(data, stcd)
    )
    print("gfsother_result_dataarray:")
    print(d2m_result_dataarray, t2m_result_dataarray, dswrf_result_dataarray)


def fetch_all_data(basin_code, base_url):
    page_index = 0
    all_data = {
        "basinCode": basin_code,
        "gfsTp": [],
        "gfsData": [],
        "gfsSoilData": [],
        "gpmData": [],
        "smapData": [],
    }

    while True:
        url = f"{base_url}?pageIndex={page_index}&basinCode={basin_code}"
        response = requests.get(url)
        data = response.json()

        # 检查是否所有数据字段都为空
        if not any(
            data[field]
            for field in ["gfsTp", "gfsData", "gfsSoilData", "gpmData", "smapData"]
        ):
            break

        # 累加数据
        for key in all_data.keys():
            if key != "basinCode":
                all_data[key].extend(data[key])

        page_index += 1

    return all_data
