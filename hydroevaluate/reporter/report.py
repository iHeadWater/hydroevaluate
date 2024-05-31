
# import grpc
# from grpc4bmi.bmi_grpc_client import BmiClient

# mymodel = BmiClient(grpc.insecure_channel("localhost:5000"))
# print(mymodel.get_component_name())

#不用跑runbmiserver，BmiClientSubProcess函数包含run-bmi-server
from grpc4bmi.bmi_client_subproc import BmiClientSubProcess
mymodel = BmiClientSubProcess(path = "/home/wangjingyi/code/hydro-model-xaj",module_name = "xaj.xaj_bmi.xajBmi")
print(mymodel.get_component_name())


def compare_history_report(new_eval_log, old_eval_log):
    if old_eval_log is None:
        old_eval_log = {"NSE of streamflow": 0, "KGE of streamflow": 0}
    # https://doi.org/10.1016/j.envsoft.2019.05.001
    # 需要再算一下洪量
    if (list(new_eval_log["NSE of streamflow"]) > old_eval_log["NSE of streamflow"]) & (
        list(new_eval_log["KGE of streamflow"]) > old_eval_log["KGE of streamflow"]
    ):
        new_eval_log["review"] = "比上次更好些，再接再厉"
    elif (
        list(new_eval_log["NSE of streamflow"]) > old_eval_log["NSE of streamflow"]
    ) & (list(new_eval_log["KGE of streamflow"]) < old_eval_log["KGE of streamflow"]):
        new_eval_log["review"] = "拟合比以前更好，但KGE下降，对洪峰预报可能有问题"
    elif (
        list(new_eval_log["NSE of streamflow"]) < old_eval_log["NSE of streamflow"]
    ) & (list(new_eval_log["KGE of streamflow"]) > old_eval_log["KGE of streamflow"]):
        new_eval_log["review"] = (
            "拟合结果更差了，问题在哪里？KGE更好一些，也许并没有那么差"
        )
    elif (
        list(new_eval_log["NSE of streamflow"]) < old_eval_log["NSE of streamflow"]
    ) & (list(new_eval_log["KGE of streamflow"]) < old_eval_log["KGE of streamflow"]):
        new_eval_log["review"] = "白改了，下次再说吧"
    else:
        new_eval_log["review"] = "和上次相等，还需要再提高"