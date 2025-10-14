import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Literal

from .config_manager import DataConfig
from .utils import logger

class DataSource(Enum):
    """Defines the available data sources for the pipeline."""
    LOCAL = "local"
    DATABRICKS = "databricks"

def load_training_data(
    data_config: DataConfig,
    source: Literal["local", "databricks"]
) -> pd.DataFrame:
    """
    Loads the training dataset either from a local file or a specified data store.
    
    Args:
        data_config (DataConfig): The validated configuration object for data.
        source (Literal["local", "databricks"]): The desired data source.
                                    
    Returns:
        pd.DataFrame: The loaded raw data.
    """
    
    if source == DataSource.LOCAL.value:

        local_path = Path(data_config.local_train_data_path)
        print(f"Loading data locally from: {local_path}")
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local training data file not found at: {local_path}")
            
        df = pd.read_csv(local_path)
        print(f"Successfully loaded {len(df)} rows from local path.")
        return df

    elif source == DataSource.DATABRICKS.value:
        # --- Databricks/Delta Lake Loading Logic will go here ---

        
        raise NotImplementedError(
            f"Databricks loading is not yet implemented. "
        )
    else:
        raise ValueError(f"Invalid source '{source}'. Must be 'local' or 'databricks'.")