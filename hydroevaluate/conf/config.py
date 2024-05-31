"""
Author: Wenyu Ouyang
Date: 2023-10-25 18:49:02
LastEditTime: 2023-10-31 21:11:12
LastEditors: Wenyu Ouyang
Description: Some configs for minio server
FilePath: \hydro_privatedata\hydroprivatedata\config.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import logging
import pathlib
import os
import boto3
import hydrodataset as hds
from minio import Minio
import s3fs

import json

import yaml

from hydroevaluate.hydroevaluate import private_yml

from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.trainers.trainer import set_random_seed


MINIO_SERVER = "http://minio.waterism.com:9000"
LOCAL_DATA_PATH = None

minio_paras = {
    "endpoint_url": MINIO_SERVER,
    "access_key": "",
    "secret_key": "",
    "bucket_name": "test",
}

home_path = str(pathlib.Path.home())

if os.path.exists(os.path.join(home_path, ".wisminio")):
    for line in open(os.path.join(home_path, ".wisminio")):
        key = line.split("=")[0].strip()
        value = line.split("=")[1].strip()
        # print(key,value)
        if key == "endpoint_url":
            minio_paras["endpoint_url"] = value
        elif key == "access_key":
            minio_paras["access_key"] = value
        elif key == "secret_key":
            minio_paras["secret_key"] = value
        elif key == "bucket_path":
            minio_paras["bucket_name"] = value

if os.path.exists(os.path.join(home_path, ".hydrodataset")):
    settings_path = os.path.join(home_path, ".hydrodataset", "settings.json")
    if not os.path.exists(settings_path):
        with open(settings_path, "w+") as fp:
            json.dump({"local_data_path": None}, fp)
    with open(settings_path, "r+") as fp:
        settings_json = json.load(fp)
    LOCAL_DATA_PATH = settings_json["local_data_path"]


if LOCAL_DATA_PATH is None:
    """
    hydro_warning.no_directory(
        "LOCAL_DATA_PATH",
        "Please set local_data_path in ~/.hydrodataset, otherwise, you can't use the local data.",
    )
    """
    logging.warning(
        msg="Please set local_data_path in ~/.hydrodataset, otherwise, you can't use the local data."
    )

# Set up MinIO client
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_SERVER,
    aws_access_key_id=minio_paras["access_key"],
    aws_secret_access_key=minio_paras["secret_key"],
)
mc = Minio(
    MINIO_SERVER.replace("http://", ""),
    access_key=minio_paras["access_key"],
    secret_key=minio_paras["secret_key"],
    secure=False,
)
site_bucket = "stations"
site_object = "sites.csv"

fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": minio_paras["endpoint_url"]},
    key=minio_paras["access_key"],
    secret=minio_paras["secret_key"],
)

ro = {
    "client_kwargs": {"endpoint_url": minio_paras["endpoint_url"]},
    "key": minio_paras["access_key"],
    "secret": minio_paras["secret_key"],
}


def custom_cfg(
    cfgs_path,
):
    f = open(cfgs_path, encoding="utf-8")
    cfgs = yaml.load(f.read(), Loader=yaml.FullLoader)
    config_data = default_config_file()
    """
    remote_obj_array = ['1_02051500.nc', '86_21401550.nc', 'camelsus_attributes.nc', 'merge_streamflow.nc']
    bucket_name = 'forestbat-private'
    folder_prefix = 'predicate_data'
    minio_obj_list = ['s3://' + bucket_name + '/' + folder_prefix + '/' + i for i in remote_obj_array]
    test_data_list = test_read_valid_data(minio_obj_list)
    """
    args = cmd(
        sub=cfgs["data_cfgs"]["sub"],
        source=cfgs["data_cfgs"]["source"],
        source_region=cfgs["data_cfgs"]["source_region"],
        source_path=hds.ROOT_DIR,
        streamflow_source_path=os.path.join(hds.ROOT_DIR, "merge_streamflow.nc"),
        rainfall_source_path=hds.ROOT_DIR,
        attributes_path=os.path.join(hds.ROOT_DIR, "camelsus_attributes.nc"),
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
            "out_channels": 8,
        },
        weight_path=os.path.join(
            pathlib.Path(os.path.abspath(os.curdir)).parent.parent,
            cfgs["model_cfgs"]["weight_dir"],
        ),
        loss_func=cfgs["training_cfgs"]["loss_func"],
        sampler=cfgs["data_cfgs"]["sampler"],
        dataset=cfgs["data_cfgs"]["dataset"],
        scaler=cfgs["data_cfgs"]["scaler"],
        batch_size=cfgs["training_cfgs"]["batch_size"],
        var_t=[["tp"]],
        var_c=cfgs["data_cfgs"]["constant_cols"],
        var_out=["streamflow"],
        # train_period=train_period,
        # test_period的dict和拼接数据的periods存在一定抵触
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
        rolling=cfgs["data_cfgs"]["rolling"],
        metrics=cfgs["test_cfgs"]["metrics"],
        endpoint_url=private_yml["minio"]["server_url"],
        access_key=private_yml["minio"]["access_key"],
        secret_key=private_yml["minio"]["secret"],
        # bucket_name=bucket_name,
        # folder_prefix=folder_prefix,
        # stat_dict_file=os.path.join(train_path, "GPM_GFS_Scaler_2_stat.json"),
        user="zxw",
    )
    update_cfg(config_data, args)
    random_seed = config_data["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    data_cfgs = config_data["data_cfgs"]
    data_source_name = data_cfgs["data_source_name"]
    data_source = data_sources_dict[data_source_name](
        data_cfgs["data_path"], data_cfgs["download"]
    )
    return data_source, config_data  # , minio_obj_list
