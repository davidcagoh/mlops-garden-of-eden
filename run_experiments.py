import logging
from src.utils import setup_logging, logger
from src.config_manager import get_config
from src.experiments_runner import run_experiment_stage

# Load Configuration
config = get_config(base_path='config/config_base.yaml') 

# Initialize Logging
setup_logging(
    level=getattr(logging, config.tracking.log_settings.level.upper()),
    log_file=config.tracking.log_settings.file_path
)

# Define Environment-Specific Parameters
ENV = 'dev'

# Start Pipeline Execution
logger.info("--- MLOps Pipeline Start ---")

try:
    final_run_id = run_experiment_stage(config, ENV)
    logger.info(f"Pipeline finished successfully. Best run ID: {final_run_id}")
    
except Exception as e:
    logger.critical(f"Pipeline terminated with a critical failure.")
