import logging
from typing import Dict, Any, Tuple
from src.config_manager import Config

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Handles retrieving the best model, running predictions on the test set,
    calculating performance metrics, and generating the final report.
    """
    def __init__(self, config: Config, env: str):
        self.config = config
        self.env = env

    def load_test_data(self) -> Tuple[Any, Any]:
        """
        Reads the test dataset (features and labels) prepared by the experiment stage.
        Returns:
            A tuple containing (X_test, y_test).
        """
        pass

    def retrieve_best_model(self, run_id: str) -> Any:
        """
        Retrieves the finalized, best model artifact from MLflow using its run_id.
        Args:
            run_id: The MLflow run ID of the best model run.
        Returns:
            The loaded model object ready for inference.
        """
        pass

    def run_predictions(self, model: Any, X_test: Any) -> Any:
        """
        Generates predictions on the test features using the retrieved model.
        Returns:
            The array or series of predictions (y_pred).
        """
        pass

    def calculate_metrics(self, y_test: Any, y_pred: Any) -> Dict[str, float]:
        """
        Calculates all required classification metrics (e.g., accuracy, F1, AUC).
        Args:
            y_test: The true labels for the test set.
            y_pred: The predicted labels.
        Returns:
            A dictionary of metric names and their calculated float values.
        """
        pass

    def generate_report(self, metrics: Dict[str, float], run_id: str) -> None:
        """
        Generates a final, human-readable report (e.g., printing to console, 
        saving to a Markdown or HTML file).
        """
        pass

    def run_evaluation_pipeline(self, run_id: str) -> Dict[str, float]:
        """
        Orchestrates the entire evaluation process.
        Args:
            run_id: The run_id of the best model from the experiment stage.
        Returns:
            The final dictionary of performance metrics.
        """
        pass

def run_evaluation_stage(config: Config, env: str, run_id: str) -> Dict[str, float]:
    """
    External entry point called by the main pipeline script.
    """
    evaluator = ModelEvaluator(config, env)
    return evaluator.run_evaluation_pipeline(run_id)


if __name__ == '__main__':
    # Execution for local testing/debugging
    from src.config_manager import get_config
    # NOTE: In a real environment, you would pass a valid run_id here.
    metrics = run_evaluation_stage(get_config(env='dev'), env='dev', run_id="MOCK_BEST_RUN_ID")