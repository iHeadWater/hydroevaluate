import os
import pandas as pd

# 定义目录路径
dir_A = '/home/xushuolong1/camels/camel_us/basin_dataset_public_v1p2/usgs_streamflow'
dir_B = '/home/xushuolong1/camels/camel_us/basin_dataset_public_v1p2/usgs_streamflow_extended'
dir_C = '/home/xushuolong1/camels/camel_us/basin_dataset_public_v1p2/usgs_streamflow_merged'
 
# 遍历目录A的子文件夹
for folder in os.listdir(dir_A):
    folder_path_A = os.path.join(dir_A, folder)
    folder_path_B = os.path.join(dir_B, folder)
    folder_path_C = os.path.join(dir_C, folder)

    # 确保目录C的子文件夹存在
    if not os.path.exists(folder_path_C):
        os.makedirs(folder_path_C)

    # 遍历目录A的子文件夹中的txt文件
    for file in os.listdir(folder_path_A):
        if file.endswith("_streamflow_qc.txt"):
            # 获取8位数字
            file_id = file.split('_')[0]

            # 构建目录B中的文件名
            file_path_B = os.path.join(folder_path_B, file_id + '.txt')

            # 如果在目录B中找到匹配的文件
            if os.path.exists(file_path_B):
                # 列名
                column_names = ["id", "year", "month", "day", "value", "status"]

                # 读取文件
                # 使用正则表达式 "\s+" 作为分隔符
                # 并确保 "id" 列被读取为字符串，以保留前导 0
                df1 = pd.read_csv(os.path.join(folder_path_A, file), sep="\s+", header=None, names=column_names, dtype={"id": str, "year": str, "month": str, "day" : str, "value": float})
                df2 = pd.read_csv(file_path_B, sep="\s+", header=None, names=column_names, dtype={"id": str, "year": str, "month": str, "day" : str, "value": float})

                # 删除 df1 中 year 为 2013 或 2014 的行
                df1 = df1[~df1['year'].isin(['2013', '2014'])]

                # 将 df2 拼接到 df1 后面
                final_df = pd.concat([df1, df2], ignore_index=True)
                final_df['value'] = final_df['value'].astype(float)

                # 显示拼接后数据的前几行，以便检查
                final_df.to_csv(os.path.join(folder_path_C, file_id + '_merged.txt'), sep=' ', header=False, index=False, na_rep='', float_format="%8.2f")  # 使用制表符作为分隔符
                
                with open(os.path.join(folder_path_C, file_id + '_merged.txt'), 'r', encoding='utf-8') as file:
                    content = file.read()

                # 使用replace方法去除所有的双引号
                content_no_quotes = content.replace('"', '')

                # 写入新的文件
                with open(os.path.join(folder_path_C, file_id + '_merged.txt'), 'w', encoding='utf-8') as file:
                    file.write(content_no_quotes)