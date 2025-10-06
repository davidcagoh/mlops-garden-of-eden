import logging
from src.config_manager import Config, get_config 

logger = logging.getLogger(__name__)

def run_ingestion(config: Config, env: str) -> None:
    """
    entry point for the Data Loader stage.
    Handles reading raw data, initial cleaning, and writing to an intermediate table.
    """
    pass


if __name__ == '__main__':
    # This runs the module in 'dev' environment if executed directly
    run_ingestion(get_config(env='dev'), env='dev')
