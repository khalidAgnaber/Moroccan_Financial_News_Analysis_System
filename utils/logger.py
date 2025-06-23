"""
Logging utilities for the project.
"""
import os
import logging
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name, level=logging.INFO):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Name of the logger
        level: Logging level
    
    Returns:
        logging.Logger: Configured logger object
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Skip if handlers already configured
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = LOGS_DIR / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name):
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Name of the logger
    
    Returns:
        logging.Logger: Logger object
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        return setup_logger(name)
    
    return logger