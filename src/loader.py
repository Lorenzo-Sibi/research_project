from multiprocessing import Value
from pathlib import Path 
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
    
    @staticmethod
    def load(input_path):
        if isinstance(input_path, str):
            input_path = Path(input_path)
        
        if input_path.is_file():
            # Single-file mode
            if input_path.suffix in IMAGE_SUPPORTED_EXTENSIONS:
                return load_image(input_path)
            elif input_path.suffix in TENSOR_SUPPORTED_EXTENSIONS:
                return load_tensor(input_path)
            
        elif input_path.is_dir(): # multiple-file mode
            for filename in input_path.iterdir():
                if filename.suffix in IMAGE_SUPPORTED_EXTENSIONS:
                    return load_images_as_list(input_path)
                elif filename.suffix in TENSOR_SUPPORTED_EXTENSIONS:
                    return load_tensors_as_list(input_path)
                else:
                    raise TypeError("Error. Unsupported file type.")
        else:
            raise ValueError(f"Error. {input_path} unknown file-type.")

def load_image(input_path):
    "Load a single image from 'input_path'"
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError("Path doesn't exist or is not a file path.")
    if input_path.suffix not in IMAGE_SUPPORTED_EXTENSIONS:
        raise ValueError("File extension not supported.")

    image = Image.open(input_path)
    return image


def load_images_as_list(input_path:Path)->list:
    if not input_path.exists():
        raise ValueError(f"Path {input_path} doesn't exist")
    
    print("Loading images...")
    
    try:
        if not input_path.is_dir() and input_path.suffix in (".png"):
            image_list = [load_image(input_path)]
        elif input_path.is_dir():
            image_list = [load_image(filename) for filename in input_path.iterdir() if filename.suffix in IMAGE_SUPPORTED_EXTENSIONS]
        else:
            raise RuntimeError(f"{input_path}: path or file not supported.")
    except Exception as err:
        print(err)
        return []
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

def load_tensor(input_path, name=None):
    if input_path.is_dir():
        raise ValueError("Error. 'load_tensor' method can't load a tensor from a directory.")

    if not name:
        name = input_path.stem
    suffix = input_path.suffix

    if suffix not in TENSOR_SUPPORTED_EXTENSIONS:
        raise ValueError(f"Extension {suffix} not suppported.")

    try:
        with np.load(input_path) as data:
            if suffix == ".npy":
                tensor = (TensorContainer(data, name, TensorType.NP_TENSOR))
            elif suffix == ".npz":
                for _, item in data.items():
                    tensor = (TensorContainer(item, name, TensorType.NP_TENSOR))
            else:
                raise ValueError("File extension not supported.")
        return tensor
    except Exception as e:
        print(f"Error loading {name} file: {str(e)}.", "\nFile path: ", input_path)
        raise ValueError(f"Error loading {name} file: {str(e)}.") from e
