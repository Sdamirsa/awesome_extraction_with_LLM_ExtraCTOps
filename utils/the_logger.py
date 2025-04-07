"""
Logger utility for inference logging.

Served by: Direct call

Path to venv, if required: "N/A"

Libraries to import:
- os
- logging
- datetime
- xml.etree.ElementTree
- logging.handlers
- tracemalloc
"""

###################
####  imports  ####
###################
import os
import logging
import datetime
import tracemalloc
from logging.handlers import RotatingFileHandler
import xml.etree.ElementTree as ET

# Enable tracemalloc for debugging memory allocations
tracemalloc.start()

###################
####  logging  ####
###################
# Configure logging
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# Define log folder and file path
script_dir = os.path.dirname(os.path.abspath(__file__))
now = datetime.datetime.now()
log_folder = os.path.join(script_dir, "logs")
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(
    log_folder, f"function-calling-inference_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
)

# Configure RotatingFileHandler
file_handler = RotatingFileHandler(log_file_path, maxBytes=0, backupCount=0)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
file_handler.setFormatter(formatter)

# Create logger
the_logger = logging.getLogger("function-calling-inference")
the_logger.addHandler(file_handler)

###################################
####      Main function        ####
###################################
if __name__ == "__main__":
    the_logger.info("Logger utility initialized successfully.")

###################################
####  Example use in terminal  ####
###################################
"""
no example for terminal use
"""

###################################
####  Example use in notebook  ####
###################################
"""
from utils import inference_logger

# Log an informational message
inference_logger.info("This is an informational log message.")

# Log a warning message
inference_logger.warning("This is a warning log message.")

# Log an error message
inference_logger.error("This is an error log message.")
"""