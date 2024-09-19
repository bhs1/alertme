import logging
from datetime import datetime
import yaml
import json
import os

from alertme.utils.path_utils import get_output_path, get_logs_path


def load_response_data(file_path):
    try:
        with open(file_path, 'r') as file:
            response_data = file.read()
        return response_data
    except Exception as e:
        return str(e)
    
def configure_logging(log_file='app.log'):
    # Generate a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logfile_path = get_logs_path() / log_file
    file_handler = logging.FileHandler(logfile_path)
    stream_handler = logging.StreamHandler()

    # Create a custom formatter
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            # Add run_id, filename and line number to the message
            record.msg = f"[Run {run_id}] {record.filename}:{record.lineno} - {record.msg}"
            if record.levelno == logging.ERROR:
                record.msg = f"\033[91m\033[1m{record.msg}\033[0m"  # Bold red
            return super().format(record)

    # Set the custom formatter for the handlers
    formatter = CustomFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Remove existing handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Log the start of a new run
    logging.info(f"Starting new run {run_id}")

    return run_id

def format_to_yaml(test_case):
    # If test_case is already a dict, use it directly
    if isinstance(test_case, dict):
        test_case_dict = test_case
    else:
        # If it's a string, remove outer quotes and parse as JSON
        test_case_dict = json.loads(test_case.strip("'\""))
    
    # Convert the dictionary to YAML
    yaml_string = yaml.dump(test_case_dict, default_flow_style=False)
    
    # Remove the top-level dashes that yaml.dump adds
    yaml_string = yaml_string.replace('---\n', '')
    
    return yaml_string

# Save the images to files in the scratch directory
def save_image_to_file(image_data, filename):
    file_path = get_output_path('scratch') / filename
    with open(file_path, "wb") as file:
        file.write(image_data.encode('utf-8'))