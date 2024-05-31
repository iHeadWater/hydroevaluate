import numpy as np
import pandas as pd
from hydro_opendata.reader import minio
import xarray as xr

gfs_reader = minio.GFSReader()
mask_file_path = "/home/xushuolong1/biliuhe_test/gfs_mask/mask-1-gfs.nc"
mask = xr.open_dataset(mask_file_path)
box = (mask.coords["lon"][0] - 0.2, mask.coords["lat"][0] - 0.2, mask.coords["lon"][-1] + 0.2, mask.coords["lat"][-1] + 0.2)

def check_and_convert_types(data):
    for var_name in list(data.variables):
        var = data[var_name]
        if isinstance(var.data, np.ndarray) and var.data.dtype.kind not in ['i', 'f', 'u']:
            # Convert non-compatible types to strings (you might need a different conversion)
            data[var_name] = var.astype(str)
    return data

def batch_process_data_in_range(start_datetime, end_datetime, save_directory):
    """
    This function processes data for every hour in the given range,
    each time fetching data for the 25 hours following the current hour,
    and saves each result with a filename based on the current time.
    
    :param gfs_reader: The GFS reader object with the open_dataset method.
    :param start_datetime: The starting datetime as a string in the format 'YYYY-MM-DD HH:MM:SS'.
    :param end_datetime: The ending datetime as a string in the format 'YYYY-MM-DD HH:MM:SS'.
    :param save_directory: The directory where the .nc files will be saved.
    """
    # Convert the start and end datetime strings to pandas datetime objects
    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)
    
    # Iterate over each hour in the range
    for current_hour in pd.date_range(start_datetime, end_datetime, freq='H'):
        # Calculate the end datetime by adding 25 hours to the current hour
        current_end_datetime = current_hour + pd.Timedelta(hours=25)
        
        # Open the dataset for the specified creation date and time
        dataset = gfs_reader.open_dataset(creation_date=np.datetime64("2023-01-31"),
                                          creation_time='00',
                                          bbox=box,
                                          dataset="wis",
                                          time_chunks=24,)
        
        # Select the data for the valid time range (current hour to next 25 hours)
        data = dataset.sel(valid_time=slice((current_hour + pd.Timedelta(hours=1)).isoformat(),
                                            current_end_datetime.isoformat()))
        
        # Aggregate the data by taking the maximum over the 'step' and 'time' dimensions
        data = data.max(dim='step')
        data = data.max(dim='time')
        data = data.load()
        w_data = mask["w"]
        w_data_interpolated = w_data.interp(
            lat=data.lat, lon=data.lon, method="nearest"
        ).fillna(0)

        # 将 w 数据广播到与当前数据集相同的时间维度上
        w_data_broadcasted = w_data_interpolated.broadcast_like(
            data["tp"]
        )
        data = data["tp"] * w_data_broadcasted
        data.name = "tp"
        data = data.to_dataset()
        
        # Save the processed data to a .nc file named after the current hour
        filename = f"{save_directory}/{current_hour.strftime('%Y-%m-%d %H:%M:%S')}.nc"
        data_checked = check_and_convert_types(data)
        data_checked = data_checked.rename({'valid_time': 'time', 'tp': '__xarray_dataarray_variable__'})
        data_checked = data_checked.transpose('time','lon','lat')
        time_as_datetime64 = pd.to_datetime(data_checked['time'].values)
        data_checked['time'] = time_as_datetime64
        # print(data_checked)
        data_checked.to_netcdf(filename)

        print(f"Saved data to {filename}")

batch_process_data_in_range(start_datetime = '2023-01-31 06:00:00', end_datetime = '2023-02-01 05:00:00', save_directory = '/home/xushuolong1/biliuhe_test/gfs_final')