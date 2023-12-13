import os
from os.path import basename, splitext
import random
from PIL import Image
import numpy as np
import tensorflow as tf

from src import *
from src.utils import TensorType, TensorContainer

# Setting global seed
random.seed(42)

class Loader():
    def __init__(self):
        pass

def load_image(input_path):
    "Load a single image from 'input_path'"
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError("Path doesn't exist or is not a file path.")
    if input_path.suffix not in IMAGE_SUPPORTED_EXTENSIONS:
        raise ValueError("File extension not supported.")

    image = Image.open(input_path)
    return image


def load_images_as_list(input_path)->list:
    if not input_path.exists():
        raise ValueError("Path doesn't exist")
    
    print("Loading images...")
    
    try:
        image_list = [load_image(filename) for filename in input_path.iterdir() if filename.suffix in IMAGE_SUPPORTED_EXTENSIONS]
    except Exception as err:
        print(err)
        return []
    print("Loading completed.")
    return image_list

# Load an entire tensors batch from "directory" (.npz or .npy files) and return a list of narrays
def load_tensors_from_directory(directory_path):

    print("Loading tensors...")
    
    n_files = 0
    tensors_list = []
    for filename in directory_path.iterdir():
        if filename.is_dir():
            continue
        
        tensors_list.append(load_tensor(filename))
        n_files += 1
        
    print(f"Load complete. {n_files} files loaded succesfully.")
    return tensors_list

def load_tensors_as_list(input_directory, tensor_type=TensorType.TF_TENSOR):
    tensors = load_tensors_from_directory(input_directory)
    for tensor in tensors:
        TensorContainer.convert(tensor, tensor_type)
    return tensors

def load_tensor(input_path, name=None)->TensorContainer:
    if input_path.is_dir():
        raise ValueError("Error. 'load_tensor' method can't load a tensor from a directory.")

    if not name:
        name = input_path.stem
    suffix = input_path.suffix

    if suffix not in TENSOR_SUPPORTED_EXTENSIONS:
        raise ValueError(f"Extension {suffix} not suppported.")
    
    tensor = None
    try:
        with np.load(input_path) as data:
            if suffix == ".npy":
                tensor = (TensorContainer(data, name, TensorType.NP_TENSOR))
            elif suffix == ".npz":
                for _, item in data.items():
                    tensor = (TensorContainer(item, name, TensorType.NP_TENSOR))
            else:
                raise ValueError("File extension not supported.")
    except Exception as e:
        print(f"Error loading {name} file: {str(e)}.", "\nFile path: ", input_path)
        return

    return tensor