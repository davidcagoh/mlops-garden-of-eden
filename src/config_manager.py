from dataclasses import dataclass
from typing import Dict, List, Any
from pathlib import Path

# --- Dataclass Schemas ---

@dataclass(frozen=True)
class ModelConfig:
    """Schema for individual model definitions."""
    type: str
    hyperparameters: Dict[str, Any]

@dataclass(frozen=True)
class ModelsConfig:
    """Schema for the entire models block, dynamically holding ModelConfig."""
    pass # Actual fields (RF, XGB, etc.) will be dynamically added/validated

@dataclass(frozen=True)
class DataConfig:
    """Schema for data source and table names."""
    catalog_name: str
    schema_name: str
    raw_table_name: str
    intermediate_clean_table: str
    features_table_name: str
    local_raw_path: str

@dataclass(frozen=True)
class TuningConfig:
    """Schema for pipeline execution settings."""
    models_to_run: List[str]

@dataclass(frozen=True)
class TrackingConfig:
    """Schema for MLflow and tracking settings."""
    experiment_name: str
    mlflow_tracking_uri: str

@dataclass(frozen=True)
class Config:
    """The main, top-level configuration schema."""
    project_name: str
    target_column: str
    random_seed: int
    data: DataConfig
    tuning: TuningConfig
    tracking: TrackingConfig
    models: ModelsConfig # Holds all model definitions

# --- Loading Function ---

def _load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """
    Internal function to securely load YAML content from a given path.
    """
    pass

def _merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal function to recursively merge the override config into the base config.
    """
    pass

def get_config(env: str = 'dev') -> Config:
    """
    The main public function to load and validate the complete configuration.

    It loads the base config, merges the environment-specific override, 
    and validates the result against the Config dataclass structure.
    """
    pass