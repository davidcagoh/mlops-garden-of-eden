import pytest
import pandas as pd
from pathlib import Path
from dataclasses import replace

from src.data_loader import load_training_data, DataSource 
from src.config_manager import DataConfig, FeatureConfig 

# --- Fixture for Test Data and Configuration ---

@pytest.fixture
def data_loader_setup(tmp_path):
    """
    Creates a temporary CSV file and a DataConfig instance pointing to it.
    """
    
    # Define dummy data
    data_content = {
        'id': [1, 2, 3, 4],
        'value': [10.1, 20.2, 30.3, 40.4]
    }
    df_expected = pd.DataFrame(data_content)
    
    # Save the dummy data to a temporary file
    test_file_path = tmp_path / "temp_train_data.csv"
    df_expected.to_csv(test_file_path, index=False)
    
    # Create a mock FeatureConfig instance
    mock_feature_config = FeatureConfig(
        numerical=[], # Empty lists are fine since the loader doesn't use these
        categorical=[]
    )
    
    # Create a mock DataConfig object
    mock_config = DataConfig(
        catalog_name="mock_catalog",
        schema_name="mock_schema",
        raw_table_name="mock_raw",
        features_table_name="mock_features",
        local_raw_path="unused/raw.csv",
        local_train_data_path=str(test_file_path),
        intermediate_clean_table="mock_clean",
        features=mock_feature_config
    )
    
    # Return the config and the expected data
    return mock_config, df_expected

# --- Test Functions ---

def test_local_data_loading_success(data_loader_setup, monkeypatch):
    """
    Verifies that the local data loading path successfully reads the expected data 
    and mocks print() to prevent test output clutter.
    """
    mock_config, df_expected = data_loader_setup
    
    # Use monkeypatch to suppress or mock the print statements in your function
    monkeypatch.setattr('builtins.print', lambda *args: None)

    # Call the loader with the 'local' source
    df_loaded = load_training_data(
        data_config=mock_config, 
        source=DataSource.LOCAL.value
    )

    # Check 1: Ensure it returned a pandas DataFrame
    assert isinstance(df_loaded, pd.DataFrame)
    
    # Check 2: Ensure the loaded data matches the expected data
    pd.testing.assert_frame_equal(df_loaded, df_expected)
    
    # Check 3: Ensure correct number of rows/columns
    assert df_loaded.shape == (4, 2)


def test_local_data_loading_file_not_found(data_loader_setup, tmp_path):
    """
    Verifies that the local data loading path raises FileNotFoundError when the 
    specified file does not exist.
    """
    mock_config, _ = data_loader_setup
    
    # Create a path that does not exist
    missing_path = tmp_path / "non_existent_data.csv"
    
    # Create a new config instance with the missing path (using dataclasses.replace)
    mock_config_missing = replace(mock_config, local_train_data_path=str(missing_path))

    # Assert that calling the function raises the expected error
    with pytest.raises(FileNotFoundError):
        load_training_data(
            data_config=mock_config_missing, 
            source=DataSource.LOCAL.value
        )


def test_databricks_loading_not_implemented(data_loader_setup):
    """
    Verifies that the 'databricks' data source path correctly raises a 
    NotImplementedError, as this feature is pending implementation.
    """
    mock_config, _ = data_loader_setup
    
    # Assert that calling the function with 'databricks' raises the specific error
    with pytest.raises(NotImplementedError) as excinfo:
        load_training_data(
            data_config=mock_config, 
            source=DataSource.DATABRICKS.value
        )
        
    # Check the error message contains the expected generic string
    assert "Databricks loading is not yet implemented" in str(excinfo.value)


def test_invalid_source_raises_value_error(data_loader_setup):
    """
    Verifies that providing an invalid source string raises a ValueError.
    """
    mock_config, _ = data_loader_setup
    
    # Use the expected error handling mechanism
    with pytest.raises(ValueError):
        load_training_data(
            data_config=mock_config, 
            source="sftp_server" # An invalid source string
        )