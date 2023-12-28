# pytest model_stream.py::test_auto_stream
import os.path
import pathlib as pl
import smtplib
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import pytest
import urllib3 as ur
from yaml import load, Loader

import config
import minio_api as ma
from test import run_normal_dl

work_dir = pl.Path(os.path.abspath(os.curdir)).parent.parent


@pytest.mark.asyncio
async def test_auto_stream():
    # test_data = await test_read_valid_data('001')
    test_config_path = os.path.join(work_dir, 'scripts/conf/v002.yml')
    # 配置文件中的weight_dir应与模型保存位置相对应
    # test_model_name = test_read_history(user_model_type='model', version='300')
    eval_log, preds_xr, obss_xr = run_normal_dl(test_config_path)
    # https://zhuanlan.zhihu.com/p/631317974
    # 保证隐私，授权码要保存在yaml配置文件中
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
        msg['From'] = 'nickname<'+sendAddress+'>'
        msg['To'] = str(['nickname<'+addr+'>;' for addr in email_yml['to_address']])
        msg['Subject'] = 'model_report'
        server.sendmail(sendAddress, email_yml['to_address'], msg.as_string())
        print('发送成功')


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
        model_file_name = user_model_type+'_v'+str(version)+'.pth'
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


@pytest.mark.asyncio
async def test_read_valid_data(version='001'):
    client_mc = config.mc
    conf_yaml = read_yaml(version)
    test_period = conf_yaml['test_period']
    # test_period = ["2023-11-01 16:38:00", "2023-11-01 16:40:00"]
    start_time = pd.to_datetime(test_period[0], format='%Y-%m-%d %H:%M:%S').tz_localize(tz='Asia/Shanghai')
    end_time = pd.to_datetime(test_period[1], format='%Y-%m-%d %H:%M:%S').tz_localize(tz='Asia/Shanghai')
    obj_time_list = pd.to_datetime([obj.last_modified for obj in client_mc.list_objects(bucket_name='forestbat-private', recursive='True')]).tz_convert(tz='Asia/Shanghai')
    time_indexes = obj_time_list[(obj_time_list > start_time) & (obj_time_list < end_time)]
    obj_down_array = [obj.object_name for obj in client_mc.list_objects(bucket_name='forestbat-private', recursive='True') if obj.last_modified in time_indexes]
    await ma.minio_batch_download(obj_down_array, client_mc, bucket_name='forestbat-private', local_path=os.path.join(work_dir, 'test_data'))
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

