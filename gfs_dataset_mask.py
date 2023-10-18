import numpy as np
import xarray as xr
import rioxarray
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas
import shapely
import netCDF4 as nc
from shapely.geometry import mapping
from mean import gen_mask
import numpy as np
import re
import datetime

def list_all_files_with_root(dir_path):
    """
    遍历指定文件夹及其子文件夹下的所有文件，并返回每个文件的 root 路径和文件名
    :param dir_path: 指定的文件夹路径
    :return: 包含 (root, file_name) 对的列表
    """
    result = []

    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
           if not file_name.endswith("923a8.idx"):
                result.append((root, file_name))

    return result


# 打开文件
watershed = geopandas.read_file('../data/671_shp/671-hyd_na_dir_30s.shp')

# 根据watershed的结果，生成各个流域的mask
# gen_mask(watershed, "STAID", "gfs", save_dir="./mask_GFS")

# 设置文件夹路径
mask_folder_path = './mask_GFS/'
gfs_folder_path = '../data/GFS/09/'

# 获取文件夹中所有的文件
all_mask_files = os.listdir(mask_folder_path)

# 正则表达式模式，用于提取数字，因为那个mask函数生成的文件名太乱了，这里要整理一下
pattern = re.compile(r'mask-(\d{1,8})-.*\.nc')

# 遍历文件，把mask重命名，名字为id

for file_name in all_mask_files:
    match = pattern.match(file_name)
    if match:
        # 获取匹配到的数字部分
        number_str = match.group(1)
        
        # 左侧补0至8位
        number_str_padded = number_str.zfill(8)
        
        # 构造新的文件名
        new_file_name = f'{number_str_padded}.nc'
        
        # 获取完整的文件路径
        old_file_path = os.path.join(mask_folder_path, file_name)
        new_file_path = os.path.join(mask_folder_path, new_file_name)
        
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        
        # 输出或其他操作
        print(f'Renamed: {file_name} -> {new_file_name}')


# 正则表达式模式，用于提取日期和时间信息
pattern = re.compile(r'gfs(\d+).t(\d+)z.pgrb2.0p25.f(\d+)')

# 获取文件夹中所有的文件
all_GFS_files = list_all_files_with_root(gfs_folder_path)

all_GFS_files.sort()

all_mask_files = os.listdir(mask_folder_path)

# 依次读取文件
for mask_file_name in all_mask_files:
    
    try:
        mask = xr.open_dataset(mask_folder_path + mask_file_name)
        lon_min = float(format(mask.coords["lon"][0].values))
        lat_min = float(format(mask.coords["lat"][0].values))
        lon_max = float(format(mask.coords["lon"][-1].values))
        lat_max = float(format(mask.coords["lat"][-1].values))

        for GFS_file_root, GFS_file_name in all_GFS_files:
            
            GFS_file_path = GFS_file_root + "/" + GFS_file_name
            
            print(GFS_file_path)
            data = xr.open_dataset(GFS_file_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'stepType': 'instant'}})
            
            # data.rename({'longitude': 'lon', 'latitude': 'lat', 'pwat': 'precipitationCal'})
            
            data_process = data.sel(
                longitude=slice(lon_min, lon_max + 0.01),
                latitude=slice(lat_min, lat_max + 0.01)
            )
            
            match = pattern.match(GFS_file_name)
            
            if match:
                date = match.group(1)
                current_hour = match.group(2)
                forecast_hour = match.group(3)
                data_process_name = f"{date}{current_hour}_{forecast_hour.zfill(3)}"

            data_process_path = "mask_gfs_stream_data/" + mask_file_name.replace(".nc", "") + "/"

            data_process_full_path = data_process_path + str(data_process_name)

            # 检查路径是否存在，若不存在，则创建该路径
            if not os.path.exists(data_process_path):
                os.makedirs(data_process_path)
                print(f"Path {data_process_path} created")
            else:
                print(f"Path {data_process_path} already exists")

            data_process.to_netcdf(data_process_full_path + ".nc")
            
    except:
        for GFS_file_root, GFS_file_name in all_GFS_files:
            
            GFS_file_path = GFS_file_root + "/" + GFS_file_name
            
            data = xr.open_dataset(GFS_file_path, engine='cfgrib')
            
            data_process = data
            
            match = pattern.match(GFS_file_name)
            
            if match:
                date = match.group(1)
                current_hour = match.group(2)
                forecast_hour = match.group(3)
                data_process_name = f"{date}{current_hour}_{forecast_hour.zfill(3)}"

            data_process_path = "mask_gfs_stream_data/" + mask_file_name.replace(".nc", "") + "_error/"

            data_process_full_path = data_process_path + str(data_process_name)

            # 检查路径是否存在，若不存在，则创建该路径
            if not os.path.exists(data_process_path):
                os.makedirs(data_process_path)
                print(f"Path {data_process_path} created")
            else:
                print(f"Path {data_process_path} already exists")

            data_process.to_netcdf(data_process_full_path + ".nc")