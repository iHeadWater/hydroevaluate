"""
Author: Wenyu Ouyang
Date: 2024-05-30 09:11:04
LastEditTime: 2024-05-31 13:45:31
LastEditors: Wenyu Ouyang
Description: main function for hydroevaluate
FilePath: \hydroevaluate\hydroevaluate\hydroevaluate.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

# pytest model_stream.py::test_auto_stream
import os.path
import pathlib as pl
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import s3fs
import yaml
from scipy import signal
from yaml import Loader, Dumper

from hydroevaluate.modelloader.load_model import run_normal_dl

work_dir = pl.Path(os.path.abspath(os.curdir)).parent.parent
with open(os.path.join(work_dir, "test_data/privacy_config.yml"), "r") as fp:
    private_str = fp.read()
private_yml = yaml.load(private_str, Loader)
storage_option = {
    "key": private_yml["minio"]["access_key"],
    "secret": private_yml["minio"]["secret"],
    "client_kwargs": {"endpoint_url": private_yml["minio"]["client_endpoint"]},
}
mc_fs = s3fs.S3FileSystem(
    endpoint_url=storage_option["client_kwargs"]["endpoint_url"],
    key=storage_option["key"],
    secret=storage_option["secret"],
)


def auto_stream():
    test_config_path = os.path.join(work_dir, "scripts/conf/v002.yml")
    with open(test_config_path, "r+") as fp:
        test_conf_yml = yaml.load(fp, Loader)
    # 配置文件中的weight_dir应与模型保存位置相对应，目前模型路径是直接指定，而非选择最新
    # test_model_name = test_read_history(user_model_type='model', version='300')
    eval_log, preds_xr, obss_xr = run_normal_dl(test_config_path)
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
    eval_log["Reports"] = {}
    eval_log["Reports"]["Total streamflow(mm/h)"] = {}
    eval_log["Reports"]["Peak rainfall(mm)"] = {}
    eval_log["Reports"]["Peak streamflow(mm/h)"] = {}
    eval_log["Reports"]["Streamflow peak appearance"] = {}
    for i in range(0, preds_xr_sf_np.shape[0]):
        basin = obss_xr["basin"].to_numpy()[i]
        pred_peaks_index = signal.argrelmax(preds_xr_sf_np[i])
        pred_peaks_time = (preds_xr["time_now"].to_numpy())[pred_peaks_index]
        obs_peaks_index = signal.argrelmax(obss_xr_sf_np[i])
        obss_peaks_time = (obss_xr["time_now"].to_numpy())[obs_peaks_index]
        eval_log["Metrics"]["Bias of peak height(mm/h)"][basin] = np.mean(
            [
                abs(obss_xr_sf_np[i] - preds_xr_sf_np[i])
                for i in range(0, len(obs_peaks_index))
            ]
        ).tolist()

        eval_log["Metrics"]["Bias of peak appearance(h)"][basin] = (
            np.mean(
                [
                    abs(obss_peaks_time[i] - pred_peaks_time[i])
                    for i in range(0, len(obss_peaks_time))
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
