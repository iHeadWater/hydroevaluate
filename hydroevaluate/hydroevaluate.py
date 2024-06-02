"""
Author: Wenyu Ouyang
Date: 2024-05-30 09:11:04
LastEditTime: 2024-06-02 16:03:02
LastEditors: Wenyu Ouyang
Description: main function for hydroevaluate
FilePath: \hydroevaluate\hydroevaluate\hydroevaluate.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

# pytest model_stream.py::test_auto_stream
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import torch
import yaml
from scipy import signal
from yaml import Loader, Dumper

from torchhydro.trainers.train_utils import (
    calculate_and_record_metrics,
)

from hydroevaluate import SETTING
from hydroevaluate.dataloader.data import load_dataset
from hydroevaluate.modelloader.model import infer_torchmodel, load_torchmodel


class EvalDeepHydro:
    def __init__(self, conf_file=None):
        self.conf_dir = SETTING["conf_dir"]
        self.conf_name = conf_file
        self.cfg = self._load_config()
        self._check_config()

    def _load_config(self):
        config_name = self.conf_name
        if config_name is None:
            # TODO: we chose the first as the default, later we will handle with multiple config files
            config_name = os.listdir(self.conf_dir)[0]
        with open(os.path.join(self.conf_dir, config_name), "r") as fp:
            cfg = yaml.load(fp, Loader)
        return cfg

    def _check_config(self):
        # TODO: simply check now, more detailed check will be added later
        if "data_cfgs" not in self.cfg:
            raise KeyError("data_cfgs not found in config file")
        if "model_cfgs" not in self.cfg:
            raise KeyError("model_cfgs not found in config file")
        if "evaluation_cfgs" not in self.cfg:
            raise KeyError("evaluation_cfgs not found in config file")

    def load_model(self):
        model_type = self.cfg["model_cfgs"]["model_name"]
        model_hyperparam = self.cfg["model_cfgs"]["model_hyperparam"]
        trained_param_dir = self.cfg["model_cfgs"]["param_dir"]
        return load_torchmodel(model_type, model_hyperparam, trained_param_dir)

    def load_data(self):
        data_cfgs = self.cfgs["data_cfgs"]
        return load_dataset(data_cfgs)

    def run_model(self):
        eval_cfgs = self.cfg["evaluation_cfgs"]
        dataset = self.load_data()
        model = self.load_model()
        # Assume load_model and evaluate are methods defined in this class
        model.eval()
        # here the batch is just an index of lookup table, so any batch size could be chosen
        seq_first = eval_cfgs["which_first_tensor"] == "sequence"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            pred = infer_torchmodel(seq_first, device, model, dataset.x)
            pred = pred.cpu().numpy()
        ngrid = dataset.ngrid
        if not eval_cfgs["long_seq_pred"]:
            target_len = len(eval_cfgs["output_vars"])
            prec_window = eval_cfgs["prec_window"]
            if eval_cfgs["rolling"]:
                forecast_length = eval_cfgs["forecast_length"]
                pred = pred[:, prec_window:, :].reshape(
                    ngrid, batch_size, forecast_length, target_len
                )

                pred = pred[:, ::forecast_length, :, :]
                pred = np.concatenate(pred, axis=0).reshape(ngrid, -1, target_len)
                pred = pred[:, :batch_size, :]
            else:
                pred = pred[:, prec_window, :].reshape(ngrid, batch_size, target_len)
        return dataset.denormalize(pred)

    def evaluate(self, obs_xr):
        eval_cfgs = self.cfg["evaluation_cfgs"]
        pred_xr = self.run_model()
        fill_nan = eval_cfgs["fill_nan"]
        eval_log = {}
        for i, col in enumerate(eval_cfgs["output_vars"]):
            obs = obs_xr[col].to_numpy()
            pred = pred_xr[col].to_numpy()
            eval_log = calculate_and_record_metrics(
                obs,
                pred,
                eval_cfgs["metrics"],
                col,
                fill_nan[i] if isinstance(fill_nan, list) else fill_nan,
                eval_log,
            )
        test_log = f" Best Metric {eval_log}"
        print(test_log)
        return eval_log, pred_xr, obs_xr


def auto_stream():
    test_config_path = os.path.join(work_dir, "scripts/conf/v002.yml")
    with open(test_config_path, "r+") as fp:
        test_conf_yml = yaml.load(fp, Loader)
    # 配置文件中的weight_dir应与模型保存位置相对应，目前模型路径是直接指定，而非选择最新
    # test_model_name = test_read_history(user_model_type='model', version='300')
    eval_log, preds_xr, obss_xr = load_torchmodel(test_config_path)
    preds_xr_sf_np = preds_xr["streamflow"].to_numpy().T
    obss_xr_sf_np = obss_xr["streamflow"].to_numpy().T
    eval_log["Metrics"] = {}
    eval_log["Config"] = {}
    eval_log["Basin"] = obss_xr["basin"].to_numpy().tolist()
    eval_log["Metrics"]["NSE"] = eval_log["NSE of streamflow"].tolist()
    eval_log.pop("NSE of streamflow")
    eval_log["Metrics"]["MAE"] = eval_log["Bias of streamflow"].tolist()
    eval_log.pop("Bias of streamflow")
    eval_log["Metrics"]["KGE"] = eval_log["KGE of streamflow"].tolist()
    eval_log.pop("KGE of streamflow")
    eval_log["Metrics"]["RMSE"] = eval_log["RMSE of streamflow"].tolist()
    eval_log.pop("RMSE of streamflow")
    eval_log["Metrics"]["Bias of peak height(mm/h)"] = {}
    eval_log["Metrics"]["Bias of peak appearance(h)"] = {}
    eval_log["Reports"] = {
        "Total streamflow(mm/h)": {},
        "Peak rainfall(mm)": {},
        "Peak streamflow(mm/h)": {},
        "Streamflow peak appearance": {},
    }
    for i in range(preds_xr_sf_np.shape[0]):
        basin = obss_xr["basin"].to_numpy()[i]
        pred_peaks_index = signal.argrelmax(preds_xr_sf_np[i])
        pred_peaks_time = (preds_xr["time_now"].to_numpy())[pred_peaks_index]
        obs_peaks_index = signal.argrelmax(obss_xr_sf_np[i])
        obss_peaks_time = (obss_xr["time_now"].to_numpy())[obs_peaks_index]
        eval_log["Metrics"]["Bias of peak height(mm/h)"][basin] = np.mean(
            [
                abs(obss_xr_sf_np[i] - preds_xr_sf_np[i])
                for i in range(len(obs_peaks_index))
            ]
        ).tolist()

        eval_log["Metrics"]["Bias of peak appearance(h)"][basin] = (
            np.mean(
                [
                    abs(obss_peaks_time[i] - pred_peaks_time[i])
                    for i in range(len(obss_peaks_time))
                ]
            ).tolist()
            / 3.6e12
        )
        # 在这里是所有预测值在[0，forecast_length]内的总洪量
        eval_log["Reports"]["Total streamflow(mm/h)"][basin] = np.sum(
            preds_xr_sf_np[i][
                0 : test_conf_yml["model_cfgs"]["model_hyperparam"]["forecast_length"]
            ]
        ).tolist()
        # rainfall对于这个模型是输入先验值，地位“微妙”，找不到合适地点插入, 暂且留空
        eval_log["Reports"]["Peak rainfall(mm)"][basin] = 200
        eval_log["Reports"]["Peak streamflow(mm/h)"][basin] = np.max(
            preds_xr_sf_np[i][
                0 : test_conf_yml["model_cfgs"]["model_hyperparam"]["forecast_length"]
            ]
        ).tolist()
        eval_log["Reports"]["Streamflow peak appearance"][basin] = (
            np.datetime_as_string(pred_peaks_time, unit="s").tolist()
        )
    eval_log["Config"]["model_name"] = test_conf_yml["model_cfgs"]["model_name"]
    eval_log["Config"]["model_hyperparam"] = test_conf_yml["model_cfgs"][
        "model_hyperparam"
    ]
    eval_log["Config"]["weight_path"] = test_conf_yml["model_cfgs"]["weight_dir"]
    eval_log["Config"]["t_range_train"] = test_conf_yml["train_period"]
    eval_log["Config"]["t_range_test"] = test_conf_yml["test_period"]
    eval_log["Config"]["dataset"] = test_conf_yml["data_cfgs"]["dataset"]
    eval_log["Config"]["sampler"] = test_conf_yml["data_cfgs"]["sampler"]
    eval_log["Config"]["scaler"] = test_conf_yml["data_cfgs"]["scaler"]
    # https://zhuanlan.zhihu.com/p/631317974
    send_address = private_yml["email"]["send_address"]
    password = private_yml["email"]["authenticate_code"]
    server = smtplib.SMTP_SSL("smtp.qq.com", 465)
    login_result = server.login(send_address, password)
    if login_result == (235, b"Authentication successful"):
        content = yaml.dump(data=eval_log, Dumper=Dumper)
        # https://service.mail.qq.com/detail/124/995
        # https://stackoverflow.com/questions/58223773/send-a-list-of-dictionaries-formatted-with-indents-as-a-string-through-email-u
        msg = MIMEMultipart()
        msg["From"] = "nickname<" + send_address + ">"
        msg["To"] = str(
            ["nickname<" + addr + ">;" for addr in private_yml["email"]["to_address"]]
        )
        msg["Subject"] = "model_report"
        msg.attach(MIMEText(content, "plain"))
        server.sendmail(
            send_address, private_yml["email"]["to_address"], msg.as_string()
        )
        print("发送成功")
    else:
        print("发送失败")
