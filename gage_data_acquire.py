# -*- coding: utf-8 -*-
import os
from dataretrieval import nwis
import csv
import pandas as pd
 
with open("/home/xushuolong1/flood_data_preprocess/data/gauge_information.txt", "r", encoding="utf-8") as file:
    content = file.readlines()
    
gage_ids = [line.split('\t')[1].strip() for line in content[1:] if line.strip()]
# sites = gage_ids

for gage_id in gage_ids:
    # get instantaneous values (iv)
    df, md = nwis.get_iv(sites=gage_id, start='2016-01-01', end='2023-06-30', parameterCd='00065')

    df.to_csv(str(gage_id) + '.txt', sep=' ', header=False, na_rep='', float_format="%6.2f")  # 使用制表符作为分隔符
    
    # 打开原文件，并读取所有内容
    with open(str(gage_id) + '.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用replace方法去除所有的双引号
    content_no_quotes = content.replace('"', '')

    # 写入新的文件
    with open(str(gage_id) + '.txt', 'w', encoding='utf-8') as file:
        file.write(content_no_quotes)