import xarray as xr
import os
from tqdm import tqdm

# 定义文件夹路径
folder_path = './mask_stream_data_2017_1h/01544500/'

# 列出文件夹中的所有NetCDF文件并按文件名排序
nc_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.nc')])

# 使用tqdm显示进度，并逐个加载NetCDF文件并添加到一个列表中
datasets = [xr.open_dataset(os.path.join(folder_path, f)) for f in tqdm(nc_files, desc="Loading files")]

# 按时间维度合并所有数据集
merged_ds = xr.concat(datasets, dim='time')

# 保存合并后的数据集到新的NetCDF文件
output_path = './mask_gpm_stream_data_full/01544500_2017_gpm.nc'
merged_ds.to_netcdf(output_path)