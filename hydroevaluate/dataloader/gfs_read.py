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
        date_np=np.datetime64("2017-01-01"),
        time_str="00",
        bbbox=(-125, 25, -66, 50),
        num=3
):
    forecast_times = generate_forecast_times_updated(date_np, time_str, num)
    gfs_reader = minio.GFSReader()
    time = forecast_times[0]
    data = gfs_reader.open_dataset(
        # data_variable="tp",
        creation_date=np.datetime64(time[0]),
        creation_time=time[1],
        bbox=bbbox,
        dataset="camels",
        time_chunks=24,
    )
    data = data.to_dataset()
    data = data['total_precipitation_surface'].isel(valid_time=int(time[2]))
    data = data.squeeze(dim='time', drop=True)
    data = data.rename({'valid_time': 'time'})
    latest_data = data
    for time in forecast_times[1:]:
        data = gfs_reader.open_dataset(
            # data_variable="tp",
            creation_date=np.datetime64(time[0]),
            creation_time=time[1],
            bbox=bbbox,
            dataset="camels",
            time_chunks=24,
        )
        data = data.to_dataset()
        data = data['total_precipitation_surface'].isel(valid_time=int(time[2]))
        data = data.squeeze(dim='time', drop=True)
        data = data.rename({'valid_time': 'time'})
        latest_data = xr.concat([latest_data, data], dim='time')
        # print(latest_data)

    latest_data = latest_data.to_dataset()
    latest_data = latest_data.transpose('time', 'lon', 'lat')
    # print(latest_data)
    return latest_data


# Testing the combined function
# mask = xr.open_dataset('/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/mask_GFS/05584500.nc')
mask = xr.open_dataset(path_to_your_nc_file)
box = (mask.coords["lon"][0], mask.coords["lat"][0], mask.coords["lon"][-1], mask.coords["lat"][-1])
test_data = fetch_latest_data(date_np="2017-01-01", time_str="23", bbbox=box, num=3)
# print(test_data)
test_data.to_netcdf('test_data.nc')
