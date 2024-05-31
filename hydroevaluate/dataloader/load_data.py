"""
Author: Wenyu Ouyang
Date: 2024-05-30 09:11:04
LastEditTime: 2024-05-31 13:44:18
LastEditors: Wenyu Ouyang
Description: 
FilePath: \hydroevaluate\hydroevaluate\utils\load_data.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""


from hydroevaluate.hydroevaluate import mc_fs, storage_option


import geopandas as gpd
import pandas as pd
import xarray as xr
from xarray import Dataset


def read_valid_data(minio_obj_array, need_cache=False):
    # https://intake.readthedocs.io/en/latest/plugin-directory.html
    data_obj_array = []
    for obj in minio_obj_array:
        if "." not in obj:
            txt_source = itk.open_textfile(obj, storage_options=storage_option)
            data_obj_array.append(txt_source)
            if need_cache is True:
                txt_source.to_file(path=obj)
        else:
            ext_name = obj.split(".")[1]
            if ext_name == "csv":
                csv_dataset = pd.read_csv(obj, storage_options=storage_option)
                data_obj_array.append(csv_dataset)
                if need_cache is True:
                    csv_dataset.to_csv(obj)
            elif (ext_name == "nc") | (ext_name == "nc4"):
                nc_source = itk.open_netcdf(obj, storage_options=storage_option)
                nc_dataset: Dataset = nc_source.read()
                data_obj_array.append(nc_dataset)
                if need_cache is True:
                    nc_dataset.to_netcdf(path=obj)
            elif ext_name == "json":
                json_source = itk.open_json(obj, storage_options=storage_option)
                json_dict = json_source.read()
                data_obj_array.append(json_dict)
            elif ext_name == "shp":
                # Can't run directly, see this: https://github.com/geopandas/geopandas/issues/3129
                remote_shp_obj = mc_fs.open(obj)
                shp_gdf = gpd.read_file(remote_shp_obj, engine="pyogrio")
                data_obj_array.append(shp_gdf)
                if need_cache is True:
                    shp_gdf.to_file(path=obj)
            elif "grb2" in obj:
                # ValueError: unrecognized engine cfgrib must be one of: ['netcdf4', 'h5netcdf', 'scipy', 'store', 'zarr']
                # https://blog.csdn.net/weixin_44052055/article/details/108658464?spm=1001.2014.3001.5501
                # 似乎只能用conda来装eccodes
                remote_grib_obj = mc_fs.open(obj)
                grib_ds = xr.open_dataset(remote_grib_obj)
                data_obj_array.append(grib_ds)
                if need_cache is True:
                    grib_ds.to_netcdf(obj)
            elif ext_name == "txt":
                txt_source = itk.open_textfiles(obj, storage_options=storage_option)
                data_obj_array.append(txt_source)
                if need_cache is True:
                    txt_source.to_file(path=obj)
    return data_obj_array
