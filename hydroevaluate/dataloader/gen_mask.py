import numpy as np
import xarray as xr
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas
import shapely
import netCDF4 as nc
from shapely.geometry import mapping
from mean import gen_mask
import numpy as np
import re
import datetime

# 打开文件
# watershed = geopandas.read_file('/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/kaishan_shp/kaishan.shp')
watershed = geopandas.read_file('/home/xushuolong1/flood_data_preprocess/data/671_shp/671-hyd_na_dir_30s.shp')

gen_mask(watershed, "STAID", "gpm", save_dir="/home/xushuolong1/flood_data_preprocess/GPM_data_preprocess/sw_gpm")
#gen_mask(watershed, "STAID", "gfs", save_dir="./test_out")