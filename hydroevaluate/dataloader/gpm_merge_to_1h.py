import os
import pandas as pd
import xarray as xr
from tqdm import tqdm

# 源文件夹路径
source_folder_path = '/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/mask_stream_re_data_2017/01423000'

# 目标文件夹路径
target_folder_path = '/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/mask_stream_re_data_2017_1h/01423000'
os.makedirs(target_folder_path, exist_ok=True)  # 如果不存在，创建目标文件夹

# 列出所有的 NetCDF 文件并排序
nc_files = sorted([f for f in os.listdir(source_folder_path) if f.endswith('.nc')])

# 分组文件：每小时的文件为一组（每组两个）
file_pairs = [nc_files[i:i + 2] for i in range(0, len(nc_files), 2)]

# 为每一对文件进行处理
for files in tqdm(file_pairs, desc='Adding files', unit='pair'):
    # 检查确保每组确实有两个文件
    if len(files) != 2:
        print(f"Skipping incomplete hour: {files}")
        continue
    
    # 读取第一个文件
    ds1 = xr.open_dataset(os.path.join(source_folder_path, files[0]))
    # 读取第二个文件
    ds2 = xr.open_dataset(os.path.join(source_folder_path, files[1]))
    
    # 检查坐标是否一致（改进）
    if not all(ds1.coords[dim].equals(ds2.coords[dim]) for dim in ds1.coords):
        print(f"Coordinates do not match for files: {files}")
        continue
    
    # 相加数据集
    combined_ds = ds1 + ds2
    
    original_variable_name = '__xarray_dataarray_variable__'  # 替换为你的原始变量名
    combined_ds = combined_ds.rename({original_variable_name: 'precipitationCal'})

    
    # 构建新的文件名，使用第一个文件的时间
    timestamp = pd.to_datetime(files[0][:-3], format='%Y-%m-%d %H:%M:%S')
    hour_str = timestamp.strftime('%Y-%m-%d %H')
    combined_filename = f"{hour_str}:00:00.nc"
    
    # 保存合并后的数据集
    combined_ds.to_netcdf(os.path.join(target_folder_path, combined_filename))
    
    # 关闭数据集，释放资源
    ds1.close()
    ds2.close()
