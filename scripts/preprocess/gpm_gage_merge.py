import os
import xarray as xr
import pandas as pd

# 1. 获取所有.nc文件的列表
folder_path = "./mask_stream_data_2017/01544500"
nc_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nc')]

# 2. 读取.txt文件内容
with open("01544500.txt", 'r') as file:
    txt_data = file.readlines()

# 3. 为每个.nc文件执行操作
for nc_file in nc_files:
    ds = xr.open_dataset(nc_file)
    
    # 提取时间
    time_value = ds.time.values[0]
    time_stamp = pd.Timestamp(time_value)

    # 只处理整点数据
    if time_stamp.minute == 0 and time_stamp.second == 0:
        time_str = time_stamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # 在.txt文件中查找相应的时间
        matching_lines = [line for line in txt_data if time_str in line]
        
        # 如果找到匹配的行，提取数据并保存到新的.nc文件
        if matching_lines:
            line = matching_lines[0]
            parts = line.split()

            value = float(parts[2])  # 从分割的行中获取值
            id_ = parts[4]  # 获取ID

            # 将数值和ID添加到数据集中
            ds['gage'] = value
            ds['ID'] = id_
        
            # 保存新的.nc文件到指定文件夹
            output_path = os.path.join("01544500_output", os.path.basename(nc_file))
            ds.to_netcdf(output_path)
