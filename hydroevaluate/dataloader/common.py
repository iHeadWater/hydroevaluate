"""
Author: Wenyu Ouyang
Date: 2024-05-31 14:21:54
LastEditTime: 2024-06-03 16:41:04
LastEditors: Wenyu Ouyang
Description: The common class for loading data
FilePath: \hydroevaluate\hydroevaluate\dataloader\common.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

from abc import ABC

from hydroevaluate.dataloader.gfs import (
    process_gfs_tp,
    process_gfs_soil,
    process_gfs_other_forcing,
)
from hydroevaluate.dataloader.gpm import process_gpm_data
from hydroevaluate.dataloader.smap import process_smap_sm_surface


class EvalDataSource(ABC):
    def __init__(self, name, var_lst):
        self.name = name
        self.var_lst = var_lst

    def load_ts_basin_mean(self, basin_lst: list, time_range: list, vars: list):
        if not isinstance(vars, list):
            vars = [vars]
        if any(var not in self.var_lst for var in vars):
            raise ValueError("Variable not supported")
        ds_lst = [[]]
        for var_ in vars:
            da_lst = []
            for basin_id in basin_lst:
                data = self.basin_mean_process(basin_id, time_range, var_)
                da_lst.append(data)
            ds_lst.append(da_lst)
        # TODO: this should be trans to a xr.Dataset
        return ds_lst

    def basin_mean_process(self, basin_id, start_time, var, tolerance=0.005):
        """Read basin-range grid data and calculate the basin mean
        TODO: now only support single basin processing

        Parameters
        ----------
        basin_id : str
            basin id
        start_time: pd.Timestamp
            the start time for the data
        var: str
            the variable name
        tolerance : float, optional
            the tolerance for the intersection ratio, by default 0.005

        Raises
        ------
        NotImplementedError
            _description_

        Returns
        -------
        xr.DataArray
            basin mean data
        """
        raise NotImplementedError


class GPM(EvalDataSource):
    def __init__(self, var_lst):
        super().__init__("GPM", var_lst)

    def basin_mean_process(self, basin_id, time_range, var, tolerance=0.005):
        if var == "gpm_tp":
            return process_gpm_data(time_range, basin_id, tolerance)
        else:
            raise ValueError("Variable not supported")


class GFS(EvalDataSource):
    def __init__(self, var_lst):
        super().__init__("GFS", var_lst)

    def basin_mean_process(self, basin_id, time_range, var, tolerance=0.005):
        if var == "tp":
            return process_gfs_tp(time_range[0], basin_id, tolerance)
        elif var == "soilw":
            return process_gfs_soil(time_range[0], basin_id, tolerance)
        elif var in ["d2m", "t2m", "dswrf"]:
            # TODO: this should be more efficient
            data_ = process_gfs_other_forcing(time_range[0], basin_id, tolerance)
            return data_.sel(variable=var)
        else:
            raise ValueError("Variable not supported")


class SMAP(EvalDataSource):
    def __init__(self, var_lst):
        super().__init__("SMAP", var_lst)

    def basin_mean_process(self, basin_id, time_range, var, tolerance=0.005):
        if var == "sm_surface":
            return process_smap_sm_surface(time_range[0], basin_id, tolerance)
        else:
            raise ValueError("Variable not supported")


class MultiSource:
    def __init__(self, var_lst, impute_setting):
        """_summary_

        Parameters
        ----------
        merge_type : _type_
            impute means fill the missing value with the other source and the observation is the main source
        var_lst : _type_
            _description_
        """
        self._check_source(var_lst)
        # TODO:
        self._obs_read(var_lst, basin_id_lst, time_range)
        self._pred_read(var_lst, basin_id_lst, time_range)
        self._merge()
        self._aggregate()

    def _check_source(self, var_lst):
        for var in var_lst:
            if var not in ["tp", "soilw", "d2m", "t2m", "dswrf", "sm_surface"]:
                raise ValueError("Variable not supported")
