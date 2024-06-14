import warnings
from hydroevaluate.dataloader.common import GPM, GFS, SMAP
import yaml
from hydroevaluate.utils.heutils import gee_gpm_to_1h_data, calculate_nse, plot_time_series, gee_gfs_tp_data_process
import os

warnings.filterwarnings("ignore")

cfg_path_dir = "scripts/conf/"


def sim_eval(cfgs):
    print("All processes are finished!")
    
def gpm_data_eval_with_gee_baseline(gee_file_path, var_type, time_range, basin_id):
    # you need to create a gee file manually first'
    gee_tp_data = gee_gpm_to_1h_data(gee_file_path)
    observed_csv_path = f'data/gee_gpm_1h/{basin_id}.csv'
    if not os.path.exists('data/gee_gpm_1h'):
        os.makedirs('data/gee_gpm_1h')
    gee_tp_data.to_csv(observed_csv_path, index=False)
    
    gpm = GPM([var_type])
    gpm_dataarray = gpm.basin_mean_process(basin_id, time_range, var_type)
    # 将DataArray转换为DataFrame
    df = gpm_dataarray.to_dataframe().reset_index()

    # 保存为CSV文件
    if not os.path.exists('data/postgres_gpm'):
        os.makedirs('data/postgres_gpm')
    df.to_csv(f'data/postgres_gpm/{basin_id}.csv', index=False)
    nse = calculate_nse(observed_csv=observed_csv_path, simulated_csv = f'data/postgres_gpm/{basin_id}.csv', column_name = var_type)
    print("NSE: " + str(nse))
    plot_time_series(observed_csv=observed_csv_path, simulated_csv = f'data/postgres_gpm/{basin_id}.csv', column_name = var_type)

def gfs_data_eval_with_gee_baseline(gee_file_path, var_type, time_range, basin_id):
    # you need to create a gee file manually first'
    gee_tp_data = gee_gfs_tp_data_process(gee_file_path)
    observed_csv_path = f'data/gee_gfs_1h/{basin_id}.csv'
    if not os.path.exists('data/gee_gfs_1h'):
        os.makedirs('data/gee_gfs_1h')
    gee_tp_data.to_csv(observed_csv_path, index=False)
    
    gfs = GFS([var_type])
    gfs_dataarray = gfs.basin_mean_process(basin_id, time_range, var_type)
    # 将DataArray转换为DataFrame
    df = gfs_dataarray.to_dataframe().reset_index()

    # 保存为CSV文件
    if not os.path.exists('data/postgres_gfs'):
        os.makedirs('data/postgres_gfs')
    df.to_csv(f'data/postgres_gfs/{basin_id}.csv', index=False)
    nse = calculate_nse(observed_csv=observed_csv_path, simulated_csv = f'data/postgres_gfs/{basin_id}.csv', column_name = var_type)
    print("NSE: " + str(nse))
    plot_time_series(observed_csv=observed_csv_path, simulated_csv = f'data/postgres_gfs/{basin_id}.csv', column_name = var_type)
