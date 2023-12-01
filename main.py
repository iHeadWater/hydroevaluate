import os
import hydrodataset as hds
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
import warnings
import yaml

warnings.filterwarnings("ignore")

cfg_path_dir = "scripts/conf/"


def run_normal_dl(
    cfgs_path,
    train_period=None,
    valid_period=None,
    test_period=None,
):
    f = open(cfgs_path, encoding="utf-8")
    cfgs = yaml.load(f.read(), Loader=yaml.FullLoader)

    if train_period is None:
        train_period = cfgs["train_period"]
    if valid_period is None:
        valid_period = cfgs["valid_period"]
    if test_period is None:
        test_period = cfgs["test_period"]

    config_data = default_config_file()
    args = cmd(
        sub=cfgs["data_cfgs"]["sub"],
        source=cfgs["data_cfgs"]["source"],
        source_region=cfgs["data_cfgs"]["source_region"],
        source_path=os.path.join(hds.ROOT_DIR, cfgs["data_cfgs"]["source_path"]),
        download=cfgs["data_cfgs"]["download"],
        ctx=cfgs["data_cfgs"]["ctx"],
        model_name=cfgs["model_cfgs"]["model_name"],
        model_hyperparam=cfgs["model_cfgs"]["model_hyperparam"],
        loss_func=cfgs["training_cfgs"]["loss_func"],
        sampler=cfgs["data_cfgs"]["sampler"],
        dataset=cfgs["data_cfgs"]["dataset"],
        scaler=cfgs["data_cfgs"]["scaler"],
        batch_size=cfgs["training_cfgs"]["batch_size"],
        var_t=cfgs["var_t"],
        var_out=cfgs["var_out"],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        opt=cfgs["training_cfgs"]["opt"],
        train_epoch=cfgs["training_cfgs"]["train_epoch"],
        save_epoch=cfgs["training_cfgs"]["save_epoch"],
        te=cfgs["training_cfgs"]["te"],
        gage_id=cfgs["gage_id"],
        which_first_tensor=cfgs["training_cfgs"]["which_first_tensor"],
    )

    update_cfg(config_data, args)
    train_and_evaluate(config_data)
    print("All processes are finished!")


run_normal_dl(cfg_path_dir + "v001.yml")
