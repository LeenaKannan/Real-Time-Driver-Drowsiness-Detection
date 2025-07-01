import logging
import os
from datetime import datetime

def setup_logger(name, log_level="INFO"):
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.FileHandler(f"logs/{name}.log")
        handler.setLevel(getattr(logging, log_level.upper()))
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.addHandler(console_handler)
    
    return logger