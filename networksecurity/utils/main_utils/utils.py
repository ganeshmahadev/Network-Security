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
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save a NumPy array to a file.
    
    Args:
        file_path (str): Path to save the NumPy array.
        array (np.ndarray): NumPy array to save.
        
    Raises:
        NetworkSecurityException: If there is an error saving the file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file_path, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def save_object(file_path: str, obj: object):
    """
    Save an object to a file using dill.
    
    Args:
        file_path (str): Path to save the object.
        obj (object): Object to save.
        
    Raises:
        NetworkSecurityException: If there is an error saving the file.
    """
    try:
        logging.info(f"Saving object to {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e