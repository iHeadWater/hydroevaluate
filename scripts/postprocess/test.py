import os
import pathlib
import warnings

import hydrodataset as hds
import yaml
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.datasets.data_dict import data_sources_dict
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.trainers.trainer import set_random_seed
import pathlib as pl

from yaml import Loader

warnings.filterwarnings("ignore")

work_dir = pl.Path(os.path.abspath(os.curdir)).parent.parent
def custom_cfg(
        cfgs_path,
        train_period=None,
        test_period=None,
):
    f = open(cfgs_path, encoding="utf-8")
    cfgs = yaml.load(f.read(), Loader=yaml.FullLoader)

    '''
    if train_period is None:
        train_period = cfgs["train_period"]
    if test_period is None:
        test_period = cfgs["test_period"]
    '''
    with open(os.path.join(work_dir, 'test_data/privacy_config.yml'), 'r') as fp:
        private_str = fp.read()
    private_yml = yaml.load(private_str, Loader)
    config_data = default_config_file()
    args = cmd(
        sub=cfgs["data_cfgs"]["sub"],
        source=cfgs["data_cfgs"]["source"],
        source_region=cfgs["data_cfgs"]["source_region"],
        source_path=hds.ROOT_DIR,
        streamflow_source_path=os.path.join(hds.ROOT_DIR, "merge_streamflow.nc"),
        rainfall_source_path=hds.ROOT_DIR,
        attributes_path=os.path.join(hds.ROOT_DIR, "camelsus_attributes.nc"),
        download=0,
        ctx=cfgs["data_cfgs"]["ctx"],
        model_name=cfgs["model_cfgs"]["model_name"],
        model_hyperparam=cfgs["model_cfgs"]["model_hyperparam"],
        weight_path=os.path.join(pathlib.Path(os.path.abspath(os.curdir)).parent.parent,
                                 'test_data/models/model_v20.pth'),
        loss_func=cfgs["training_cfgs"]["loss_func"],
        sampler=cfgs["data_cfgs"]["sampler"],
        dataset=cfgs["data_cfgs"]["dataset"],
        scaler=cfgs["data_cfgs"]["scaler"],
        batch_size=cfgs["training_cfgs"]["batch_size"],
        var_t=cfgs["var_t"],
        var_out=cfgs["var_out"],
        # train_period=train_period,
        test_period=[
            {"start": "2017-07-01", "end": "2017-09-29"},
        ],  # 该范围为降水的时间范围，流量会整体往后推24h
        opt=cfgs["training_cfgs"]["opt"],
        train_epoch=cfgs["training_cfgs"]["train_epoch"],
        save_epoch=cfgs["training_cfgs"]["save_epoch"],
        te=cfgs["training_cfgs"]["te"],
        gage_id=["86_21401550"],
        which_first_tensor=cfgs["training_cfgs"]["which_first_tensor"],
        continue_train=cfgs["training_cfgs"]["continue_train"],
        rolling=cfgs['data_cfgs']['rolling'],
        metrics=cfgs['test_cfgs']['metrics'],
        endpoint_url=private_yml['minio']['endpoint_url'],
        access_key=private_yml['minio']['access_key'],
        secret_key=private_yml['minio']['secret'],
        bucket_name='forestbat-private',
        folder_prefix=None,
    )

    update_cfg(config_data, args)
    random_seed = config_data["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    data_cfgs = config_data["data_cfgs"]
    data_source_name = data_cfgs["data_source_name"]
    data_source = data_sources_dict[data_source_name](
        data_cfgs["data_path"], data_cfgs["download"]
    )
    return data_source, config_data


def run_normal_dl(cfg_path):
    model = DeepHydro(custom_cfg(cfg_path)[0], custom_cfg(cfg_path)[1])
    eval_log, preds_xr, obss_xr = model.model_evaluate()
    # preds_xr.to_netcdf(os.path.join("results", "v002_test", "preds.nc"))
    # obss_xr.to_netcdf(os.path.join("results", "v002_test", "obss.nc"))
    # print(eval_log)
    return eval_log, preds_xr, obss_xr
