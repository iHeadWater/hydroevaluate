# pytest model_stream.py::test_auto_stream
import json
import os.path
import pathlib as pl
import smtplib
from email.mime.text import MIMEText

import geopandas as gpd
import intake as itk
import numpy as np
import pandas as pd
import s3fs
import urllib3 as ur
import xarray as xr
from xarray import Dataset
from yaml import load, Loader

from test import run_normal_dl

work_dir = pl.Path(os.path.abspath(os.curdir)).parent.parent


# 把test和model_stream合并，然后在主方法中update_cfg
def test_auto_stream():
    remote_obj_array = ['1_02051500.nc', '86_21401550.nc', 'camelsus_attributes.nc', 'merge_streamflow.nc']
    test_data = test_read_valid_data(remote_obj_array)
    test_config_path = os.path.join(work_dir, 'scripts/conf/v002.yml')
    # 配置文件中的weight_dir应与模型保存位置相对应
    # test_model_name = test_read_history(user_model_type='model', version='300')
    eval_log, preds_xr, obss_xr = run_normal_dl(test_config_path)
    with open('eval_log.json', mode='a+') as fp:
        last_eval_log = json.load(fp)
        compare_history_report(eval_log, last_eval_log)
        json.dump(eval_log, fp)
    # https://zhuanlan.zhihu.com/p/631317974
    test_email_config = os.path.join(work_dir, 'test_data/privacy_config.yml')
    with open(test_email_config, 'r') as fp:
        email_str = fp.read()
    email_yml = load(email_str, Loader)
    send_address = email_yml['email']['send_address']
    password = email_yml['email']['authenticate_code']
    server = smtplib.SMTP_SSL('smtp.qq.com', 465)
    login_result = server.login(send_address, password)
    if login_result == (235, b'Authentication successful'):
        content = str(eval_log)
        # https://service.mail.qq.com/detail/124/995
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['From'] = 'nickname<' + send_address + '>'
        msg['To'] = str(['nickname<' + addr + '>;' for addr in email_yml['email']['to_address']])
        msg['Subject'] = 'model_report'
        server.sendmail(send_address, email_yml['email']['to_address'], msg.as_string())
        print('发送成功')
    else:
        print('发送失败')


def test_read_history(user_model_type='wasted', version='1'):
    history_dict_path = os.path.join(work_dir, 'test_data/history_dict.npy')
    # 姑且假设所有模型都被放在test_data/models文件夹下
    if not os.path.exists(history_dict_path):
        history_dict = {}
        models = os.listdir(os.path.join(work_dir, 'test_data/models'))
        # 姑且假设model的名字为wasted_v1.pth，即用途_版本.pth
        current_max = 0
        for model_name in models:
            if user_model_type in model_name:
                model_ver = int(model_name.split('.')[0].split('v')[1])
                if model_ver > current_max:
                    current_max = model_ver
        model_file_name = user_model_type + '_v' + str(version) + '.pth'
        if model_file_name in models:
            history_dict[user_model_type] = current_max
            np.save(history_dict_path, history_dict, allow_pickle=True)
        return history_dict
    else:
        history_dict = np.load(history_dict_path, allow_pickle=True).flatten()[0]
        model_file_name = user_model_type + '_v' + str(version) + '.pth'
        if model_file_name not in history_dict.keys():
            history_dict[user_model_type] = version
        return history_dict


def test_read_valid_data(remote_obj_array, need_cache=False):
    storage_option = {'key': 'xxx', 'secret': 'yyy',
                      'client_kwargs': {'endpoint_url': 'zzz'}}
    mc_fs = s3fs.S3FileSystem(endpoint_url=storage_option['client_kwargs']['endpoint_url'],
                              key=storage_option['key'], secret=storage_option['secret'])
    # https://intake.readthedocs.io/en/latest/plugin-directory.html
    data_obj_array = []
    for obj in remote_obj_array:
        if '.' not in obj:
            txt_source = itk.open_textfile(obj, storage_options=storage_option)
            data_obj_array.append(txt_source)
            if need_cache is True:
                txt_source.to_file(path=obj)
        else:
            ext_name = obj.split('.')[1]
            if ext_name == 'csv':
                csv_dataset = pd.read_csv(obj, storage_options=storage_option)
                data_obj_array.append(csv_dataset)
                if need_cache is True:
                    csv_dataset.to_csv(obj)
            elif (ext_name == 'nc') | (ext_name == 'nc4'):
                nc_source = itk.open_netcdf(obj, storage_options=storage_option)
                nc_dataset: Dataset = nc_source.read()
                data_obj_array.append(nc_dataset)
                if need_cache is True:
                    nc_dataset.to_netcdf(path=obj)
            elif ext_name == 'json':
                json_source = itk.open_json(obj, storage_options=storage_option)
                json_dict = json_source.read()
                data_obj_array.append(json_dict)
            elif ext_name == 'shp':
                # Can't run directly, see this: https://github.com/geopandas/geopandas/issues/3129
                remote_shp_obj = mc_fs.open(obj)
                shp_gdf = gpd.read_file(remote_shp_obj, engine='pyogrio')
                data_obj_array.append(shp_gdf)
                if need_cache is True:
                    shp_gdf.to_file(path=obj)
            elif 'grb2' in obj:
                # ValueError: unrecognized engine cfgrib must be one of: ['netcdf4', 'h5netcdf', 'scipy', 'store', 'zarr']
                # https://blog.csdn.net/weixin_44052055/article/details/108658464?spm=1001.2014.3001.5501
                # 似乎只能用conda来装eccodes
                remote_grib_obj = mc_fs.open(obj)
                grib_ds = xr.open_dataset(remote_grib_obj)
                data_obj_array.append(grib_ds)
                if need_cache is True:
                    grib_ds.to_netcdf(obj)
            elif ext_name == 'txt':
                txt_source = itk.open_textfiles(obj, storage_options=storage_option)
                data_obj_array.append(txt_source)
                if need_cache is True:
                    txt_source.to_file(path=obj)
    return data_obj_array


def read_yaml(version):
    config_path = os.path.join(work_dir, 'test_data/aiff_config/aiff_v' + str(version) + '.yml')
    if not os.path.exists(config_path):
        version_url = 'https://raw.githubusercontent.com/iHeadWater/AIFloodForecast/main/scripts/conf/v' + str(
            version) + '.yml'
        yml_str = ur.request('GET', version_url).data.decode('utf8')
    else:
        with open(config_path, 'r') as fp:
            yml_str = fp.read()
    conf_yaml = load(yml_str, Loader=Loader)
    return conf_yaml


def compare_history_report(new_eval_log, old_eval_log):
    if old_eval_log is None:
        old_eval_log = {'NSE of streamflow': 0, 'KGE of streamflow': 0}
    # https://doi.org/10.1016/j.envsoft.2019.05.001
    if (new_eval_log['NSE of streamflow'] > old_eval_log['NSE of streamflow']) & (
            new_eval_log['KGE of streamflow'] > old_eval_log['KGE of streamflow']):
        new_eval_log['review'] = '比上次更好些，再接再厉'
    elif (new_eval_log['NSE of streamflow'] > old_eval_log['NSE of streamflow']) & (
            new_eval_log['KGE of streamflow'] < old_eval_log['KGE of streamflow']):
        new_eval_log['review'] = '拟合比以前更好，但KGE下降，对洪峰预报可能有问题'
    elif (new_eval_log['NSE of streamflow'] < old_eval_log['NSE of streamflow']) & (
            new_eval_log['KGE of streamflow'] > old_eval_log['KGE of streamflow']):
        new_eval_log['review'] = '拟合结果更差了，问题在哪里？KGE更好一些，也许并没有那么差'
    elif (new_eval_log['NSE of streamflow'] < old_eval_log['NSE of streamflow']) & (
            new_eval_log['KGE of streamflow'] < old_eval_log['KGE of streamflow']):
        new_eval_log['review'] = '白改了，下次再说吧'
    else:
        new_eval_log['review'] = '和上次相等，还需要再提高'
