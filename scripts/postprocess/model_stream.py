# pytest model_stream.py::test_auto_stream
import json
import os.path
import pathlib
import pathlib as pl
import smtplib
from email.mime.text import MIMEText
import hydrodataset as hds
import geopandas as gpd
import intake as itk
import numpy as np
import pandas as pd
import s3fs
import urllib3 as ur
import xarray as xr
import yaml
from torchhydro.configs.config import default_config_file, update_cfg, cmd
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.trainers.trainer import set_random_seed
from xarray import Dataset
from yaml import load, Loader


work_dir = pl.Path(os.path.abspath(os.curdir)).parent.parent
with open(os.path.join(work_dir, 'test_data/privacy_config.yml'), 'r') as fp:
    private_str = fp.read()
private_yml = yaml.load(private_str, Loader)


def test_auto_stream():
    test_config_path = os.path.join(work_dir, 'scripts/conf/v002.yml')
    with open(test_config_path, 'r+') as fp:
        test_conf_yml = yaml.load(fp, Loader)
    # 配置文件中的weight_dir应与模型保存位置相对应
    # test_model_name = test_read_history(user_model_type='model', version='300')
    eval_log, preds_xr, obss_xr = run_normal_dl(test_config_path)
    with open('eval_log.json', mode='a+', encoding='utf-8') as fp:
        fp.seek(0)
        last_eval_log = json.load(fp)
    compare_history_report(eval_log, last_eval_log)
    '''
    fp.write(test_conf_yml['data_cfgs']['sampler'] + '\n')
    fp.write(test_conf_yml['model_cfgs']['model_name'] + '\n')
    fp.write(test_conf_yml['model_cfgs']['model_hyperparam'] + '\n')
    '''
    eval_log['sampler'] = test_conf_yml['data_cfgs']['sampler']
    eval_log['model_name'] = test_conf_yml['model_cfgs']['model_name']
    eval_log['model_hyperparam'] = test_conf_yml['model_cfgs']['model_hyperparam']
    # json.dump(eval_log, fp)
    # https://zhuanlan.zhihu.com/p/631317974
    send_address = private_yml['email']['send_address']
    password = private_yml['email']['authenticate_code']
    server = smtplib.SMTP_SSL('smtp.qq.com', 465)
    login_result = server.login(send_address, password)
    if login_result == (235, b'Authentication successful'):
        content = str(eval_log)
        # https://service.mail.qq.com/detail/124/995
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['From'] = 'nickname<' + send_address + '>'
        msg['To'] = str(['nickname<' + addr + '>;' for addr in private_yml['email']['to_address']])
        msg['Subject'] = 'model_report'
        server.sendmail(send_address, private_yml['email']['to_address'], msg.as_string())
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


def test_read_valid_data(minio_obj_array, need_cache=False):
    storage_option = {'key': private_yml['minio']['access_key'], 'secret': private_yml['minio']['secret'],
                      'client_kwargs': {'endpoint_url': private_yml['minio']['client_endpoint']}}
    mc_fs = s3fs.S3FileSystem(endpoint_url=storage_option['client_kwargs']['endpoint_url'],
                              key=storage_option['key'], secret=storage_option['secret'])
    # https://intake.readthedocs.io/en/latest/plugin-directory.html
    data_obj_array = []
    for obj in minio_obj_array:
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
    # 需要再算一下洪量
    if (list(new_eval_log['NSE of streamflow']) > old_eval_log['NSE of streamflow']) & (
            list(new_eval_log['KGE of streamflow']) > old_eval_log['KGE of streamflow']):
        new_eval_log['review'] = '比上次更好些，再接再厉'
    elif (list(new_eval_log['NSE of streamflow']) > old_eval_log['NSE of streamflow']) & (
            list(new_eval_log['KGE of streamflow']) < old_eval_log['KGE of streamflow']):
        new_eval_log['review'] = '拟合比以前更好，但KGE下降，对洪峰预报可能有问题'
    elif (list(new_eval_log['NSE of streamflow']) < old_eval_log['NSE of streamflow']) & (
            list(new_eval_log['KGE of streamflow']) > old_eval_log['KGE of streamflow']):
        new_eval_log['review'] = '拟合结果更差了，问题在哪里？KGE更好一些，也许并没有那么差'
    elif (list(new_eval_log['NSE of streamflow']) < old_eval_log['NSE of streamflow']) & (
            list(new_eval_log['KGE of streamflow']) < old_eval_log['KGE of streamflow']):
        new_eval_log['review'] = '白改了，下次再说吧'
    else:
        new_eval_log['review'] = '和上次相等，还需要再提高'


def custom_cfg(
        cfgs_path,
):
    f = open(cfgs_path, encoding="utf-8")
    cfgs = yaml.load(f.read(), Loader=yaml.FullLoader)
    config_data = default_config_file()
    remote_obj_array = ['1_02051500.nc', '86_21401550.nc', 'camelsus_attributes.nc', 'merge_streamflow.nc']
    bucket_name = 'forestbat-private'
    folder_prefix = 'predicate_data'
    minio_obj_list = ['s3://'+bucket_name+'/'+folder_prefix+'/'+i for i in remote_obj_array]
    test_data_list = test_read_valid_data(minio_obj_list)
    args = cmd(
        sub=cfgs["data_cfgs"]["sub"],
        source=cfgs["data_cfgs"]["source"],
        source_region=cfgs["data_cfgs"]["source_region"],
        source_path=hds.ROOT_DIR,
        streamflow_source_path=test_data_list[3],
        rainfall_source_path=test_data_list[0:2],
        attributes_path=test_data_list[2],
        gfs_source_path="",
        download=0,
        ctx=cfgs["data_cfgs"]["ctx"],
        model_name=cfgs["model_cfgs"]["model_name"],
        model_hyperparam={
            "seq_length": 168,
            "forecast_length": 24,
            "n_output": 1,
            "n_hidden_states": 60,
            "dropout": 0.25,
            "len_c": 15,
            "in_channels": 1,
            "out_channels": 8
        },
        weight_path=os.path.join(pathlib.Path(os.path.abspath(os.curdir)).parent.parent,
                                 'test_data/models/best_model.pth'),
        loss_func=cfgs["training_cfgs"]["loss_func"],
        sampler=cfgs["data_cfgs"]["sampler"],
        dataset=cfgs["data_cfgs"]["dataset"],
        scaler=cfgs["data_cfgs"]["scaler"],
        batch_size=cfgs["training_cfgs"]["batch_size"],
        var_t=[["tp"]],
        var_c=cfgs['data_cfgs']['constant_cols'],
        var_out=["streamflow"],
        # train_period=train_period,
        test_period=[
            {"start": "2017-07-01", "end": "2017-09-29"},
        ],  # 该范围为降水的时间范围，流量会整体往后推24h
        opt=cfgs["training_cfgs"]["opt"],
        train_epoch=cfgs["training_cfgs"]["train_epoch"],
        save_epoch=cfgs["training_cfgs"]["save_epoch"],
        te=cfgs["training_cfgs"]["te"],
        gage_id=["1_02051500", "86_21401550"],
        which_first_tensor=cfgs["training_cfgs"]["which_first_tensor"],
        continue_train=cfgs["training_cfgs"]["continue_train"],
        rolling=cfgs['data_cfgs']['rolling'],
        metrics=cfgs['test_cfgs']['metrics'],
        endpoint_url=private_yml['minio']['server_url'],
        access_key=private_yml['minio']['access_key'],
        secret_key=private_yml['minio']['secret'],
        bucket_name=bucket_name,
        folder_prefix=folder_prefix,
        # stat_dict_file=os.path.join(train_path, "GPM_GFS_Scaler_2_stat.json"),
        user='yyy'
    )
    update_cfg(config_data, args)
    random_seed = config_data["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    data_cfgs = config_data["data_cfgs"]
    data_source_name = data_cfgs["data_source_name"]
    data_source = data_sources_dict[data_source_name](
        data_cfgs["data_path"], data_cfgs["download"]
    )
    return data_source, config_data #, minio_obj_list


def run_normal_dl(cfg_path):
    model = DeepHydro(custom_cfg(cfg_path)[0], custom_cfg(cfg_path)[1])
    eval_log, preds_xr, obss_xr = model.model_evaluate()
    # preds_xr.to_netcdf(os.path.join("results", "v002_test", "preds.nc"))
    # obss_xr.to_netcdf(os.path.join("results", "v002_test", "obss.nc"))
    # print(eval_log)
    return eval_log, preds_xr, obss_xr