# -*- coding: utf-8 -*-
import os
from dataretrieval import nwis
import csv
import pandas as pd
 
with open("/home/xushuolong1/camels/camel_us/basin_dataset_public_v1p2/basin_metadata/gauge_information.txt", "r", encoding="utf-8") as file:
    content = file.readlines()

gage_cate = [line.split('\t')[0].strip() for line in content[1:] if line.strip()]
gage_ids = [line.split('\t')[1].strip() for line in content[1:] if line.strip()]

combined_2d = [list(item) for item in zip(gage_cate, gage_ids)]


for folder, filename in combined_2d:
    # 创建文件夹（如果尚不存在）
    if not os.path.exists("data_camels_add/" + folder):
        os.makedirs("data_camels_add/" + folder)
    
    dailyMultiSites = nwis.get_dv(sites=filename, parameterCd=["00010", "00060"],
                                  start="2013-01-01", end="2014-12-30", statCd=["00001","00003"])
    df = dailyMultiSites[0]
    # 在文件夹中创建 txt 文件并保存 DataFrame
    file_path = os.path.join(folder, filename + ".txt")
    try:
        df['year'] = df.index.year.astype(int)
        df['month'] = df.index.month.astype(int).map(lambda x: f'{x:02}')
        df['day'] = df.index.day.astype(int).map(lambda x: f'{x:02}')
        
        # 重命名列
        df.rename(columns={'00060_Mean': 'value', '00060_Mean_cd': 'status', 'site_no': 'id'}, inplace=True)
        df['value'] = df['value'].astype(float)
        df['status'] = df['status'].str.replace(', ', ':')

        # 重新排列列的顺序
        df = df[['id', 'year', 'month', 'day', 'value', 'status']]
        
        # 保存为txt文件，设置适当的分隔符
        df.to_csv("data_camels_add/" + file_path, sep=' ', header=False, index=False, na_rep='', float_format="%8.2f")  # 使用制表符作为分隔符
        # 打开原文件，并读取所有内容
        with open("data_camels_add/" + file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 使用replace方法去除所有的双引号
        content_no_quotes = content.replace('"', '')

        # 写入新的文件
        with open("data_camels_add/" + file_path, 'w', encoding='utf-8') as file:
            file.write(content_no_quotes)
        
    except:
        df.to_csv("data_camels_add/" + file_path + '_error', sep=' ', header=False, index=False, na_rep='')  # 使用制表符作为分隔符