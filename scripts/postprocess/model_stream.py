# pytest model_stream.py::test_auto_stream
import os.path
import pathlib as pl
import smtplib
from email.mime.text import MIMEText

import intake as itk
import numpy as np
import pandas as pd
import pytest
import s3fs
import urllib3 as ur
import xarray as xr
import yaml
from xarray import Dataset
from yaml import load, Loader, Dumper

import config
from test import run_normal_dl

work_dir = pl.Path(os.path.abspath(os.curdir)).parent.parent


@pytest.mark.asyncio
async def test_auto_stream():
    # test_data = test_read_valid_data('001')
    test_config_path = os.path.join(work_dir, 'scripts/conf/v002.yml')
    # 配置文件中的weight_dir应与模型保存位置相对应
    # test_model_name = test_read_history(user_model_type='model', version='300')
    eval_log, preds_xr, obss_xr = run_normal_dl(test_config_path)
    yaml.dump(eval_log, 'eval_log.yml', Dumper=Dumper)
    # https://zhuanlan.zhihu.com/p/631317974
    test_email_config = os.path.join(work_dir, 'test_data/email_config.yml')
    with open(test_email_config, 'r') as fp:
        email_str = fp.read()
    email_yml = load(email_str, Loader)
    sendAddress = email_yml['send_address']
    password = email_yml['authenticate_code']
    server = smtplib.SMTP_SSL('smtp.qq.com', 465)
    loginResult = server.login(sendAddress, password)
    if loginResult == (235, b'Authentication successful'):
        content = str(eval_log)
        # https://service.mail.qq.com/detail/124/995
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['From'] = 'nickname<' + sendAddress + '>'
        msg['To'] = str(['nickname<' + addr + '>;' for addr in email_yml['to_address']])
        msg['Subject'] = 'model_report'
        server.sendmail(sendAddress, email_yml['to_address'], msg.as_string())
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


def test_read_valid_data(version='001', need_cache=False):
    client_mc = config.mc
    conf_yaml = read_yaml(version)
    test_period = conf_yaml['test_period']
    start_time = pd.to_datetime(test_period[0], format='%Y-%m-%d %H:%M:%S').tz_localize(tz='Asia/Shanghai')
    end_time = pd.to_datetime(test_period[1], format='%Y-%m-%d %H:%M:%S').tz_localize(tz='Asia/Shanghai')
    obj_time_list = pd.to_datetime([obj.last_modified for obj in client_mc.list_objects(bucket_name='forestbat-private',
                                                                                        recursive='True')]).tz_convert(
        tz='Asia/Shanghai')
    time_indexes = obj_time_list[(obj_time_list > start_time) & (obj_time_list < end_time)]
    # nc、csv、grib2、txt、json、yaml、zip
    obj_down_array = [obj.object_name for obj in
                      client_mc.list_objects(bucket_name='forestbat-private', recursive='True') if
                      obj.last_modified in time_indexes]
    storage_option = {'key': 'xxx', 'secret': 'yyy',
                      'client_kwargs': {'endpoint_url': 'zzz'}}
    # https://intake.readthedocs.io/en/latest/plugin-directory.html
    for obj in obj_down_array:
        ext_name = obj.split('.')[1]
        if ext_name == 'csv':
            csv_dataset = pd.read_csv(obj, storage_options=storage_option)
            if need_cache is True:
                csv_dataset.to_csv(obj)
        elif (ext_name == 'nc') | (ext_name == 'nc4'):
            nc_source = itk.open_netcdf(obj, storage_options=storage_option)
            nc_dataset: Dataset = nc_source.read()
            if need_cache is True:
                nc_dataset.to_netcdf(path=obj)
        elif ext_name == 'json':
            json_source = itk.open_json(obj, storage_options=storage_option)
            json_dict = json_source.read()
        # need intake-geopandas
        elif ext_name == 'shp':
            # itk.open_shapefile will give fiona.errors.DriverError:
            # '/vsimem/6a1a01e3b26340f4ab2f31bac440a436' not recognized as a supported file format
            # https://github.com/geopandas/geopandas/issues/3129
            shp_source = itk.open_shapefile(obj, use_fsspec=True, storage_options=storage_option)
            if need_cache is True:
                shp_source.to_file(path=obj)
        elif 'grb2' in obj:
            # ValueError: unrecognized engine cfgrib must be one of: ['netcdf4', 'h5netcdf', 'scipy', 'store', 'zarr']
            # https://blog.csdn.net/weixin_44052055/article/details/108658464?spm=1001.2014.3001.5501
            fs_grib = s3fs.S3FileSystem(endpoint_url=storage_option['client_kwargs']['endpoint_url'],
                                        key=storage_option['key'], secret=storage_option['secret'])
            remote_grib_obj = fs_grib.open(obj)
            grib_ds = xr.open_dataset(remote_grib_obj)
            if need_cache is True:
                grib_ds.to_netcdf(obj)
        elif ext_name == 'txt':
            shp_source = itk.open_textfile(obj, storage_options=storage_option)
            if need_cache is True:
                shp_source.to_file(path=obj)
    return obj_down_array


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


def compare_history_report():
    # https://doi.org/10.1016/j.envsoft.2019.05.001
    with open('eval_log.yml', 'r') as fp:
        email_str = fp.read()
    email_yml = load(email_str, Loader)
