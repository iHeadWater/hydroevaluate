import os
import pytest
from hydroevaluate.evaluator.eval import gpm_data_eval_with_gee_baseline, gfs_data_eval_with_gee_baseline

def test_gpm_data_eval_with_gee_baseline():
    # Create a temporary directory for test files
    gee_file_path = 'data/gee_gpm/21110400.csv'

    # Call the function
    gpm_data_eval_with_gee_baseline(
        gee_file_path=str(gee_file_path),
        var_type="gpm_tp",
        time_range=["2024-05-01 00:00:00", "2024-06-30 00:00:00"],
        basin_id="21110400",
    )
    
def test_gfs_data_eval_with_gee_baseline():
    gee_file_path = 'data/gee_gfs/21110400.csv'
    
    gfs_data_eval_with_gee_baseline(
        gee_file_path=str(gee_file_path),
        var_type="gfs_tp",
        time_range=["2024-05-01 00:00:00", "2024-06-30 00:00:00"],
        basin_id="21110400",
    )