import os
import logging
from datetime import datetime

def setup_logger(log_file=None):
    """
    Set up and return a logger instance
    
    Parameters:
    log_file (str, optional): Custom log file path. If None, a default path will be used.
    
    Returns:
    logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up log file with date format DD-MM-YYYY_HH-MM-SS
    if log_file is None:
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log")
    
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Force reconfiguration of the logger
    )
    
    return logging.getLogger(__name__)

# Default logger instance
logger = setup_logger() 