import sys
from src.logger import logger

def error_message_detail(error, error_detail:sys):
    """
    Generate a detailed error message with file name, line number, and error message.
    
    Parameters:
    error: The exception that was raised
    error_detail: The sys.exc_info() object containing exception info
    
    Returns:
    str: Detailed error message
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in Python script [{file_name}] at line number [{exc_tb.tb_lineno}]: {str(error)}"
    
    return error_message

class CustomException(Exception):
    """
    Custom exception class that captures and logs detailed error information.
    """
    def __init__(self, error_message, error_detail:sys=sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        logger.error(self.error_message)
    
    def __str__(self):
        return self.error_message 