import pandas as pd
from hydroevaluate.utils.heutils import calculate_nse, gee_gpm_to_1h_data
import os
from loguru import logger

def test_gee_single_basin_gpm_to_1h_data(tmp_path = 'data/gee_gpm'):
    # Create a temporary CSV file as if it is generated by gee
    csv_path = os.path.join(tmp_path, "test_data.csv")
    data = pd.DataFrame({
        "time_start": pd.date_range(start="2022-01-01", periods=10, freq="30T"),
        "BASIN_ID": ['01', '01', '01', '01', '01', '01', '01', '01', '01', '01'],
        "precipitationCal": [0.5, 0.8, 1.2, 0.3, 0.6, 0.9, 1.5, 0.7, 0.4, 1.0]
    })
    logger.info(f"Orignal Data: {data}")
    data.to_csv(csv_path, index=False)
    
    # Call the gee_gpm_to_1h_data function
    result = gee_gpm_to_1h_data(csv_path)
    logger.warning(f"Processed GPM: {result}")
    
    # Perform assertions to validate the result
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 3)
    assert result.columns.tolist() == ["basin", "precipitationCal", "time"]
    assert result["basin"].dtype == object
    assert result["precipitationCal"].dtype == float
    assert result["time"].dtype == "datetime64[ns]"
    
def test_nse_cal(
    observed_csv = os.path.join('data','gee_gpm_1h' "observed.csv"),
    simulated_csv = os.path.join('data', 'postgres_gpm' "simulated.csv"),
    column_name = 'tp'
    ):

    # Create sample data for observed and simulated values
    observed_data = pd.DataFrame({
        "time": pd.date_range(start="2022-01-01", periods=10, freq="h"),
        "tp": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    })
    simulated_data = pd.DataFrame({
        "time": pd.date_range(start="2022-01-01", periods=10, freq="h"),
        "tp": [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2]
    })

    # Save the sample data to CSV files
    observed_data.to_csv(observed_csv, index=False)
    simulated_data.to_csv(simulated_csv, index=False)

    # Call the calculate_nse function
    nse = calculate_nse(observed_csv, simulated_csv, column_name)

    # Perform assertion to validate the result
    assert isinstance(nse, float)
    print(nse)