# src/utils.py

import logging
import sys
from pathlib import Path

# Define the central logger instance here
logger = logging.getLogger('ProjectLogger') 
logger.setLevel(logging.DEBUG) 

def setup_logging(level, log_file=None):
    """Configures the central ProjectLogger instance based on config parameters."""
    
    # Prevents adding handlers multiple times if called more than once
    if logger.hasHandlers():
        return
    
    # Set the configured level for the central logger
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Console Handler (logs to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler (saves logs to file)
    if log_file:
        from logging.handlers import RotatingFileHandler
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=1024 * 1024 * 5, 
            backupCount=5 
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)