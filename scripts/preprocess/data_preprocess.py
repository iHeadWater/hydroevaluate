import pandas as pd
import matplotlib.pyplot as plt
import os

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input DataFrame by:
    1. Removing rows where 'status' is 'M'.
    2. Sorting based on 'value' in ascending order.
    3. Adding a 'ratings' column with values from 1 to length of DataFrame.
    4. Adding a 'prec_exceed' column based on the formula: 1 - (rating / total_rows)
    5. Labelling 'prec_exceed' values and storing in 'label' column.
 
    Parameters:
    - df: Input DataFrame to be processed.

    Returns:
    - Processed DataFrame.
    """
    
    # 1. 删除status列下所有'M'的数据
    df = df[df['status'] != 'M']

    # 2. 根据value标签进行升序排序
    df = df.sort_values(by='value')

    # 3. 添加一个新的列ratings，并为这个列赋值为从1开始的序列
    df['ratings'] = range(1, len(df) + 1)

    # 4. 计算prec_exceed的值
    total_rows = len(df)
    df['prec_exceed'] = 1 - (df['ratings'] / total_rows)

    # 5. 根据prec_exceed的值进行标记并保存到label标签里
    def label_row(row):
        if row > 0.4:
            return 'N'
        elif 0.2 <= row <= 0.4:
            return 'S'
        elif 0.05 <= row <= 0.2:
            return 'M'
        elif 0.02 <= row <= 0.05:
            return 'L'
        elif row < 0.02:
            return 'X'

    df['label'] = df['prec_exceed'].apply(label_row)
    df = df.sort_index()
    
    return df

def data_process(input_path, rel_path, file_name, output_dir):
    data = pd.read_csv(input_path,sep='\s+',header=None,names=['id','year','month','day','value','status'])
    data["date"] = pd.to_datetime(data.year.map(str) + "/" + data.month.map(str) + "/" + data.day.map(str), format="%Y/%m/%d")
    new_df = process_dataframe(data)
    process_dir = output_dir + '/' + rel_path
    if not os.path.exists(process_dir):
        os.makedirs(process_dir)
    new_df.to_csv(output_dir + '/' + rel_path  + '/' + file_name,sep='\t',index=False)
    
def process_files(src_dir, dst_dir):
    # 遍历指定目录及其所有子目录
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # 获取文件路径和文件名
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, src_dir)
            data_process(file_path, rel_path, file, dst_dir)
            
src_dir = '/home/xushuolong1/camels/camel_us/basin_dataset_public_v1p2/usgs_streamflow_merged'
dst_dir = '/home/xushuolong1/camels/camel_us/basin_dataset_public_v1p2/usgs_streamflow_dataset'
process_files(src_dir, dst_dir)