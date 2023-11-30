import xarray as xr
import numpy as np
import datetime
import os
import pandas as pd

def process_file_with_time_now(input_file, gpm_dataset):
    # 1. 从文件名中解析日期和时间，并计算目标时间范围
    filename_date_str = os.path.basename(input_file).split('.')[0]
    date_time_obj = datetime.datetime.strptime(filename_date_str, '%Y-%m-%d %H:%M:%S')
    
    # 确保仅处理2017-01-01 08:00:00之后的文件
    '''
    if date_time_obj < datetime.datetime(2022, 11, 8, 0):
        return None
    '''
    
    start_time = date_time_obj - datetime.timedelta(hours=168)
    end_time = date_time_obj - datetime.timedelta(hours=0)

    # 2. 从 gpm.nc 文件中获取对应时间范围的数据
    subset_gpm = gpm_dataset.sel(time=slice(start_time, end_time))
    
    # 3. 打开输入文件
    ds_input = xr.open_dataset(input_file)
    # print(ds_input)
    # ds_input = ds_input.sel(time = ds_input.time[1:])
    
    # 4. 将两个文件中的变量名都更改为 "waterlevel"
    subset_gpm = subset_gpm.rename({'precipitationCal': 'tp'})
    ds_input = ds_input.rename({'__xarray_dataarray_variable__': 'tp'})
    
    # 5. 使用 xarray 进行插值，将数据集插值到共同的 lat 和 lon 网格上
    ds_input_interp = ds_input.interp(lat=gpm_dataset.lat, lon=gpm_dataset.lon, method='linear')
    # print(ds_input_interp)
    
    # 6. 沿 time 维度拼接两个数据集
    combined_data = xr.concat([subset_gpm, ds_input_interp], dim='time')
    
    # 7. 添加 time_now 维度，并将文件代表的时间设置为该维度的值
    combined_data['time_now'] = date_time_obj
    combined_data = combined_data.assign_coords({"time_now": date_time_obj})
    combined_data = combined_data.expand_dims('time_now')
    
    return combined_data

# Load the gpm.nc dataset
gpm_dataset = xr.open_dataset('/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/sw_gpm_data/415D0432.nc')

# List all the .nc files in the directory
directory_path = '/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/sw_gfs_data_full/'
# directory_path = '/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/test_gfs_gpm/'

nc_files = [f for f in os.listdir(directory_path) if f.endswith('.nc') and "gpm" not in f]

# Define a function to parse datetime from filename
def parse_datetime_from_filename(filename):
    date_str = os.path.basename(filename).split('.')[0]
    return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

# Sort nc_files by datetime in filename
nc_files = sorted(nc_files, key=parse_datetime_from_filename)


''''
processed_datasets_time_now = [process_file_with_time_now(os.path.join(directory_path, file), gpm_dataset) for file in nc_files]

# 逐个合并数据集并检查哪个步骤出错
for i in range(1230, len(processed_datasets_time_now)):
    print(i)
    try:
        # 尝试合并直到目前为止的所有数据集
        xr.concat(processed_datasets_time_now[:i+1], dim='time_now')
    except ValueError as e:
        print(f"Error encountered when trying to concatenate up to dataset number {i}: {e}")
        break
        

'''
processed_datasets_time_now = []
for file in nc_files:
    filepath = os.path.join(directory_path, file)
    processed_data = process_file_with_time_now(filepath, gpm_dataset)
    if processed_data:
        processed_datasets_time_now.append(processed_data)

# Combine all processed datasets
final_combined_dataset_time_now = xr.concat(processed_datasets_time_now, dim='time_now')
# Save the final dataset to a new .nc file
output_file_path = '/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/gpm_gfs_data_24h_re/415D0432.nc'
final_combined_dataset_time_now.to_netcdf(output_file_path)
