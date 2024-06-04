from hydroevaluate.dataloader.common import GPM
import pandas as pd


def test_gpm_process():
    start_time = "2024-06-02 00:00:00"
    end_time = "2024-06-04 00:00:00"
    time_range = [start_time, end_time]
    gpm = GPM(["gpm_tp"])
    gpm_dataarray = gpm.basin_mean_process("21401550", time_range, "gpm_tp")
    print(gpm_dataarray)
