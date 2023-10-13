# -*- coding: utf-8 -*-
import os
from dataretrieval import nwis
import csv
import pandas as pd

"""
用来获取usgs相关流域的数据

"""
def get_nwis_stream_data(
    save_dir = '.',
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
    df.to_csv(save_dir + sites + ".txt", sep=' ', header=True, na_rep='', float_format="%6.2f")
    
    if standardlize == True:
        # 打开原文件，并读取所有内容
        with open(save_dir + sites + ".txt", 'r', encoding='utf-8') as file:
            content = file.read()

        # 使用replace方法去除所有的双引号
        content_no_quotes = content.replace('"', '')

        # 写入新的文件
        with open(save_dir + sites + ".txt", 'w', encoding='utf-8') as file:
            file.write(content_no_quotes)
    return [df, md]


# 等项目忙完再补充，这个是获得所有数据的
def get_nwis_full_data(
    save_dir = '',
    sites = None,
    start = None,
    end = None,
    multi_index = True,
    wide_format = True,
    datetime_index= True,
    state = None,
    service = 'iv',
    ssl_check = True,
    **kwargs):
    """
    完整的获取各种nwis数据的方法
    
    参考：https://github.com/DOI-USGS/dataretrieval-python/blob/master/dataretrieval/nwis.py#L1289
    
    Args:
        save_dir = '':保存目录
        sites:站点的ID，例如:'01013500'
        start:开始时间，如start='2016-01-01'
        end:结束时间，如end='2016-01-02'
        multi_index:多标签,If False, a dataframe with a single-level index (datetime) is returned
        wide_format:If True, return data in wide format with multiple samples per row and one row per time
        service:索取数据类型
            - 'iv' : instantaneous data，即时数据，15分钟尺度
            - 'dv' : daily mean data，日尺度
            - 'qwdata' : discrete samples
            - 'site' : 站点描述
            - 'measurements' : 流量测量值
            - 'peaks': 流量峰值
            - 'gwlevels': 地表水等级
            - 'pmcodes': 获取参数代码
            - 'water_use': 使用水数据
            - 'ratings': get rating table
            - 'stat': 获取数据，A代表实测，A.e代表经过处理，M代表无
        ssl_check:检查ssl
        **kwargs:可调用的其余参数
    """
    pass