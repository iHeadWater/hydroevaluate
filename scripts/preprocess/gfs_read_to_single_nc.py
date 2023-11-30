from hydro_opendata.hydro_opendata.s3api import minio
import xarray as xr
import os
import geopandas
import random
import numpy as np
from datetime import datetime, timedelta

# Redefining the first function (generate_forecast_times_updated)

def generate_forecast_times_updated(date_str, hour_str, num):
    # Parse the given date and hour
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    given_hour = int(hour_str)

    # Define the forecasting hours
    forecast_hours = [0, 6, 12, 18]
    
    # Find the closest forecast hour before the given hour
    closest_forecast_hour = max([hour for hour in forecast_hours if hour <= given_hour])

    # Generate the forecast times
    forecast_times = []
    remaining_num = num
    while remaining_num > 0:
        time_difference = given_hour - closest_forecast_hour
        for i in range(time_difference, 6):
            if remaining_num == 0:
                break
            forecast_times.append([date_obj.strftime("%Y-%m-%d"), str(closest_forecast_hour).zfill(2), str(i).zfill(2)])
            remaining_num -= 1

        # Move to the next forecasting hour
        if closest_forecast_hour == 18:
            date_obj += timedelta(days=1)
            closest_forecast_hour = 0
        else:
            closest_forecast_hour += 6
        given_hour = closest_forecast_hour

    return forecast_times

# Combining both functions to fetch the latest data points

def fetch_latest_data(
    date_np = np.datetime64("2022-09-01"),
    time_str = "00",
    bbbox = (-125, 25, -66, 50),
    num = 3
    ):
    forecast_times = generate_forecast_times_updated(date_np, time_str, num)
    gfs_reader = minio.GFSReader()
    time = forecast_times[0]
    data = gfs_reader.open_dataset(
        # data_variable="tp",
        creation_date=np.datetime64(time[0]),
        creation_time=time[1],
        bbox=bbbox,
        dataset="wis",
        time_chunks=24,
    )
    # data = data.to_dataset()
    data = data['tp'].isel(valid_time=int(time[2]))
    # data = data.squeeze(dim='step', drop=True)
    data = data.max(dim='step')
    data = data.rename({'valid_time': 'time'})
    latest_data = data
    # print(latest_data)
    for time in forecast_times[1:]:
        data = gfs_reader.open_dataset(
            # data_variable="tp",
            creation_date=np.datetime64(time[0]),
            creation_time=time[1],
            bbox=bbbox,
            dataset="wis",
            time_chunks=24,
        )
        # data = data.to_dataset()
        data = data['tp'].isel(valid_time=int(time[2]))
        # data = data.squeeze(dim='step', drop=True)
        data = data.max(dim='step')
        data = data.rename({'valid_time': 'time'})
        latest_data = xr.concat([latest_data, data], dim='time')
        print(latest_data)
    
    latest_data = latest_data.to_dataset()
    latest_data = latest_data.transpose('time', 'lon', 'lat')
    # print(latest_data)
    return latest_data

if __name__ == '__main__':
    # Testing the combined function
    mask_file_path = '/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/sw_gfs/415D0432.nc'
    mask = xr.open_dataset(mask_file_path)
    box = (mask.coords["lon"][0] - 0.2, mask.coords["lat"][0] - 0.2, mask.coords["lon"][-1] + 0.2, mask.coords["lat"][-1] + 0.2)
    w_dataset = xr.open_dataset(mask_file_path)
    w_data = w_dataset['w']
    # 初始日期和时间
    start_date = datetime(2023, 9, 25)

    end_date = datetime(2023, 10, 15)

    current_date_time = start_date
    
    # days_to_loop = 1
    
    # end_date = start_date + timedelta(days=days_to_loop)

    # current_date_time = start_date

    while current_date_time < end_date:
        date_str = current_date_time.strftime('%Y-%m-%d')
        time_str = current_date_time.strftime('%H')
        
        # print(f"Date: {date_str}, Time: {time_str}")
        merge_gfs_data = fetch_latest_data(date_np = date_str, time_str = time_str, bbbox = box, num = 25)
        print(merge_gfs_data)
        w_data_interpolated = w_data.interp(
            lat=merge_gfs_data.lat,
            lon=merge_gfs_data.lon,
            method='nearest'
        ).fillna(0)
        
        
        # 将 w 数据广播到与当前数据集相同的时间维度上
        w_data_broadcasted = w_data_interpolated.broadcast_like(merge_gfs_data['tp'])
        merge_gfs_data = merge_gfs_data['tp'] * w_data_broadcasted
        merge_gfs_data.to_netcdf('/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/sw_gfs_data_re2/' + date_str + ' ' + time_str + ':00:00.nc')        

        # 增加一个小时
        current_date_time += timedelta(hours=1)
        
