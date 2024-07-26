import logging
from datetime import datetime

def load_response_data(file_path):
    try:
        with open(file_path, 'r') as file:
            response_data = file.read()
        return response_data
    except Exception as e:
        return str(e)
    
def configure_logging(log_file='data/app.log'):
    # Generate a unique run identifier
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(log_file)
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