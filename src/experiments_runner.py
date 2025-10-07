import logging
from typing import Dict, Any, Tuple
from src.config_manager import Config

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Manages the entire model development lifecycle: data preparation, 
    hyperparameter tuning, training, and tracking.
    """
    def __init__(self, config: Config, env: str):
        self.config = config
        self.env = env

    def load_data(self) -> Any:
        """
        Reads the clean, intermediate data from the ingestion step.
        Returns: The loaded dataset object (e.g., Pandas DataFrame).
        """
        pass

    def preprocess_and_feature_engineer(self, data: Any) -> Tuple[Any, Any]:
        """
        Applies necessary transformations (scaling, encoding) and creates features.
        Args:
            data: The loaded dataset.
        Returns: 
            A tuple containing (features, labels).
        """
        pass

    def split_data(self, features: Any, labels: Any) -> Tuple[Any, Any, Any, Any]:
        """
        Splits the data into training and testing sets based on the config.
        Returns: 
            A tuple containing (X_train, X_test, y_train, y_test).
        """
        pass

    def run_tuning_and_training(self, X_train: Any, y_train: Any) -> str:
        """
        Iterates over models defined in config.tuning.models_to_run,
        performs hyperparameter search, trains the best model, and logs results 
        to MLflow.
        Returns: 
            The MLflow run_id of the best performing model.
        """
        pass

    def run_experiment_pipeline(self) -> str:
        """
        Orchestrates the entire experiment process: Load -> Preprocess -> Split -> Train.
        Returns:
            The MLflow run_id of the best model.
        """
        pass

def run_experiment_stage(config: Config, env: str) -> str:
    """
    External entry point called by the main pipeline script.
    """
    runner = ExperimentRunner(config, env)
    return runner.run_experiment_pipeline()


if __name__ == '__main__':
    # Execution for local testing/debugging
    from src.config_manager import get_config
    run_experiment_stage(get_config(env='dev'), env='dev')