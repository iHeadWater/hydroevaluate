"""
    读取nwis数据
    方法两种（目前只有水位和流量，其余以后再补充）
    第一种是没有数据源
    第二种是先下载好了，只是想读取
"""

# -*- coding: utf-8 -*-
import os
from dataretrieval import nwis
import csv
import pandas as pd

def open_local_dataset(
    file_dir = '.',
):
    # 使用read_csv函数从.txt文件中读取数据
    df = pd.read_csv(file_dir, sep=r'\s+')
    
    return df

def open_remote_dataset(
    sites = None,
    start = None,
    end = None,
    multi_index = True,
    ssl_check = True,
    parameterCd = 'All',
    standardlize = True,
    ):
    
    """
    Args:
        sites:站点id
        start:起始时间
        end:结束时间
        multi_index:多索引
        ssl_check:检查ssl
        parameterCd:用于获取部分数据，如果不指定这个参数，则获取全部数据
            00060代表流量
            00065代表水位
            出现其余数据类型，请参考USGS官网或nwis github
        standardlize:用于规范化存储txt
        
    Return:
        (type:DataFrame)
        datetime:时间
        site_no:站点ID
        00060:同上
        00065：同上
        00060_cd：测量情况，A代表实测，A,e代表经过处理，M代表无数据
        00065_cd: 同上
    """
    
    df, md = nwis.get_iv(sites=sites, start=start, end=end, parameterCd=parameterCd)
    df.to_csv(save_dir + sites + ".txt", sep=' ', header=False, na_rep='', float_format="%6.2f")
    
    return [df, md]
