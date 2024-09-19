import os
from pathlib import Path

from dotenv import find_dotenv


def get_root_path() -> Path:
    return Path(find_dotenv(usecwd=True)).parent

def get_output_path(sub_path: str = '') -> Path:
    output_path = get_root_path() / 'output' / sub_path
    os.makedirs(output_path, exist_ok=True)
    return output_path

def get_src_path() -> Path:
    return get_root_path() / 'src'

def get_logs_path(sub_path: str = '') -> Path:
    logs_path = get_root_path() / 'logs' / sub_path
    os.makedirs(logs_path, exist_ok=True)
    return logs_path

def get_data_path(sub_path: str = '') -> Path:
    data_path = get_root_path() / 'data' / sub_path
    os.makedirs(data_path, exist_ok=True)
    return data_path
