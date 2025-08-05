import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
import pickle
import dill

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    
    Args:
        file_path (str): Path to the YAML file.
        
    Returns:
        dict: Contents of the YAML file.
        
    Raises:
        NetworkSecurityException: If there is an error reading the file.
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def write_yaml_file(file_path: str, content: dict) -> None:
    """
    Write content to a YAML file
    
    Args:
        file_path: Path where YAML file will be saved
        content: Python dictionary/object to write to YAML
    """
    try:
        with open(file_path, 'w') as file:
            yaml.dump(content, file, default_flow_style=False, indent=2)
    except Exception as e:
        raise Exception(f"Error writing YAML file: {e}")