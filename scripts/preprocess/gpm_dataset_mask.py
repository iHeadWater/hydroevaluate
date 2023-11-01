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
# gen_mask(watershed, "STAID", "gpm", save_dir="./mask")

# 设置文件夹路径
mask_folder_path = '/path_to_mask/'
gpm_folder_path = '/path_to_gpm/'
output_base_path = '/path_to_output_folder/'

# 获取文件夹中所有的文件
all_mask_files = os.listdir(mask_folder_path)

# 定义寻找最近索引的函数
def find_nearest_index(array, value):
    idx = np.abs(array - value).argmin()
    return idx

# 获取.nc和.nc4文件列表
nc_files = sorted([os.path.join(mask_folder_path, f) for f in os.listdir(mask_folder_path) if f.endswith('.nc')])
nc4_files = sorted([os.path.join(gpm_folder_path, f) for f in os.listdir(gpm_folder_path) if f.endswith('.nc4')])

# 处理每个.nc文件
for nc_file in nc_files:
    # 读取.nc文件数据
    nc_data = xr.open_dataset(nc_file)
    lon_nc = nc_data.lon.values
    lat_nc = nc_data.lat.values

    # 创建以.nc文件名命名的输出文件夹
    nc_file_name = os.path.splitext(os.path.basename(nc_file))[0]
    nc_output_path = os.path.join(output_base_path, nc_file_name)
    os.makedirs(nc_output_path, exist_ok=True)

    # 对每个.nc4文件进行操作
    for nc4_file in nc4_files:
        # 读取.nc4文件数据
        nc4_data = xr.open_dataset(nc4_file)
        precipitation_cal = nc4_data['precipitationCal']

        # 初始化一个空的DataArray用于存储乘积结果
        multiplied_values = np.empty_like(nc_data['w'].values)

        # 对于每个经纬度点，找到.nc4中最近的点并乘以.nc中的w值
        for i, lon in enumerate(lon_nc):
            for j, lat in enumerate(lat_nc):
                lon_idx = find_nearest_index(nc4_data.lon.values, lon)
                lat_idx = find_nearest_index(nc4_data.lat.values, lat)
                w_value = nc_data['w'].isel(lon=i, lat=j).values
                precipitation_value = precipitation_cal.isel(lon=lon_idx, lat=lat_idx).values
                multiplied_values[i, j] = w_value * precipitation_value

        # 将乘积结果转换成DataArray
        multiplied_data = xr.DataArray(multiplied_values, coords=[('lon', lon_nc), ('lat', lat_nc)])

        # 提取时间戳，用于文件命名
        # 假设时间数据是numpy.datetime64类型的数组
        if isinstance(nc4_data.time.values[0], np.datetime64):
            time_str = np.datetime_as_string(nc4_data.time.values[0], unit='s')
        else:
            # 如果是其他类型，可能需要进一步的处理
            time_str = str(nc4_data.time.values[0])
        
        # 替换T为一个空格
        time_str = time_str.replace('T', ' ')

        # 创建新的文件名
        new_file_name = f"{time_str}.nc"
        output_base_path = os.path.join(nc_output_path, new_file_name)

        # 保存乘积数据到新文件
        multiplied_data.to_netcdf(output_base_path)
