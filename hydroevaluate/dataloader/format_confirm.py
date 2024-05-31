import os
from datetime import datetime, timedelta

def check_files_in_order(directory_path):
    # 获取文件夹中的所有文件并排序
    files = sorted(os.listdir(directory_path))
    
    # 从第一个文件名中提取初始时间
    initial_time = datetime.strptime(files[0], '%Y-%m-%d %H:%M:%S.nc')
    
    for file in files:
        # 从文件名中提取当前时间
        current_time = datetime.strptime(file, '%Y-%m-%d %H:%M:%S.nc')
        
        # 检查当前时间是否与预期的初始时间相符
        if current_time != initial_time:
            return False
        
        # 更新初始时间，使其增加30分钟
        initial_time += timedelta(minutes=60)
    
    return True

directory_path = '/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/sw_gfs_data_full' # 替换为您的文件夹路径
if check_files_in_order(directory_path):
    print("Files are in the correct order.")
else:
    print("Files are NOT in the correct order.")

