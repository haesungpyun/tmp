import os
import json
from pathlib import Path
from typing import Any, Union, Dict

def validate_path_and_make_abs_path(file_name: Union[str, Path], is_output_dir=False) -> str:
    DATA_PATH = os.environ.get('DATA_PATH')
    try:
        if os.path.isabs(file_name):
            file_abs_path = file_name
        elif DATA_PATH is not None:
            file_abs_path = os.path.join(DATA_PATH, file_name)
        else:
            file_abs_path = os.path.join(os.getcwd(), file_name)
        assert os.path.exists(file_abs_path)
    except AssertionError:
        # assert not os.path.isabs(file_name) and DATA_PATH is None
        if not is_output_dir:

            raise ValueError(
                f'set DATA_PATH in environ or {file_name} to an absolute path or check the relative path of {file_name}.\n\
                {os.path.join(os.getcwd(), file_name)} does not exist.\n')
            
    if not os.path.exists(file_abs_path):
        if not is_output_dir:
            raise FileNotFoundError(f'File not found at {file_abs_path}')
        else:
            output_dir = os.path.dirname(file_abs_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
    
    return file_abs_path

def read_json(file_name: Union[str, Path]) -> Any:
    file_path = validate_path_and_make_abs_path(file_name)

    with open(file_path, 'r') as f:
        return json.load(f)

def save_analyzed_log(output_dir_path, analyzed_log):
    output_dir_path = validate_path_and_make_abs_path(output_dir_path, is_output_dir=True)
    
    output_file = os.path.join(output_dir_path, 'analyzed_log.json')

    with open(output_file, 'w') as f:
        json.dump(analyzed_log, f, indent=2)

def load_analyzed_log(output_dir_path) -> Dict:
    output_file = os.path.join(output_dir_path, 'analyzed_log_new.json')
    if not os.path.exists(output_file):
        raise FileNotFoundError(f'File not found at {output_file}. May be you have not saved the analyzed log yet.')
    return read_json(output_file)

