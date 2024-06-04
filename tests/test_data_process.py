from hydroevaluate.dataloader.common import GPM, GFS, SMAP
import pandas as pd


def test_gpm_process():
    start_time = "2024-06-02 00:00:00"
    end_time = "2024-06-04 00:00:00"
    time_range = [start_time, end_time]
    gpm = GPM(["gpm_tp"])
    gpm_dataarray = gpm.basin_mean_process("21401550", time_range, "gpm_tp")
    print(gpm_dataarray)


def test_gpm_process_without_endtime():
    start_time = "2024-06-01 00:00:00"
    end_time = None
    time_range = [start_time, end_time]
    gpm = GPM(["gpm_tp"])
    gpm_dataarray = gpm.basin_mean_process("21401550", time_range, "gpm_tp")
    print(gpm_dataarray)


def test_gfs_tp_process():
    start_time = "2024-06-02 00:00:00"
    end_time = "2024-06-04 00:00:00"
    time_range = [start_time, end_time]
    gfs = GFS(["gfs_tp"])
    gfs_dataarray = gfs.basin_mean_process("21401550", time_range, "gfs_tp")
    print(gfs_dataarray)


def test_gfs_soilw_process():
    start_time = "2024-06-02 00:00:00"
    end_time = "2024-06-04 00:00:00"
    time_range = [start_time, end_time]
    gfs = GFS(["gfs_soilw"])
    gfs_dataarray = gfs.basin_mean_process("21401550", time_range, "gfs_soilw")
    print(gfs_dataarray)


def test_smap_process():
    start_time = "2024-05-15 00:00:00"
    end_time = "2024-06-04 00:00:00"
    time_range = [start_time, end_time]
    smap = SMAP(["smap_sm_surface"])
    smap_dataarray = smap.basin_mean_process("21401550", time_range, "smap_sm_surface")
    print(smap_dataarray)
