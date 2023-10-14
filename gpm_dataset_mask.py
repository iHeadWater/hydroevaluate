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

# 打开文件
watershed = geopandas.read_file('../data/671_shp/671-hyd_na_dir_30s.shp')

# 根据watershed的结果，生成各个流域的mask
gen_mask(watershed, "STAID", "gpm", save_dir="./mask")

# 设置文件夹路径
mask_folder_path = './mask/'
gpm_folder_path = '../data/GPM/'

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
pattern = re.compile(r'3B-HHR-E\.MS\.MRG\.3IMERG\.(\d{8})-S(\d{6})-E(\d{6})\.\d{4}\.V06C\.HDF5\.SUB\.nc4')

# 获取文件夹中所有的文件
all_GPM_files = os.listdir(gpm_folder_path)

# 空列表，用于存储文件名和对应的时间信息
file_times = []

# 遍历文件，提取日期和时间信息
for gpm_file_name in all_GPM_files:
    match = pattern.match(gpm_file_name)
    if match:
        date_str, start_time_str, end_time_str = match.groups()
        dt = datetime.datetime.strptime(date_str + start_time_str, '%Y%m%d%H%M%S')
        file_times.append((gpm_file_name, dt))

# 按时间排序
file_times.sort(key=lambda x: x[1])

all_mask_files = os.listdir(mask_folder_path)
# 依次读取文件
for mask_file_name in all_mask_files:
    
    mask = xr.open_dataset("mask/" + mask_file_name)
    lon_min = float(format(mask.coords["lon"][0].values))
    lat_min = float(format(mask.coords["lat"][0].values))
    lon_max = float(format(mask.coords["lon"][-1].values))
    lat_max = float(format(mask.coords["lat"][-1].values))
    
    for gpm_time_file_name, gpm_time_file_time in file_times:
        gpm_time_file_path = os.path.join(gpm_folder_path, gpm_time_file_name)
        
        data = xr.open_dataset(gpm_time_file_path)

        data_process = data.sel(
            lon=slice(lon_min, lon_max + 0.01),
            lat=slice(lat_min, lat_max + 0.01)
        )
        
        data_process_path = "mask_stream_data/" + mask_file_name.replace(".nc", "") + "/"
        
        data_process_name = gpm_time_file_time
        
        data_process_full_path = data_process_path + str(data_process_name)
        
        # 检查路径是否存在，若不存在，则创建该路径
        if not os.path.exists(data_process_path):
            os.makedirs(data_process_path)
            print(f"Path {data_process_path} created")
        else:
            print(f"Path {data_process_path} already exists")
        
        data_process.to_netcdf(data_process_full_path)