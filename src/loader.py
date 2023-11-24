import os
from os.path import basename, splitext
import random
from PIL import Image
import numpy as np
import tensorflow as tf

from src import *
from src import loader
from src import utils
from src.utils import TensorType, TensorContainer

# Setting global seed
random.seed(RANDOM_SEED)

def load_images_as_list(input_path, batch_size=0):
    if not os.path.exists(input_path):
        print("Path not avaiable or doesn't exist")
        return
    print("Loading images...")
    images_set = [os.path.join(input_path, file) 
                  for file in os.listdir(input_path) 
                  if file.endswith((".jpg", ".png"))]
    n_images = batch_size
    if n_images == 0 or n_images is None:
        images_batch = images_set
        n_images = len(images_set)
    else:
        images_batch = random.sample(images_set, n_images)
    image_list = [Image.open(path) for path in images_batch if path.endswith((".jpg", ".jpeg", ".png"))]
    print("Loading completed.")
    return image_list

# Load an entire tensors batch from "directory" (.npz or .npy files) and return a list of narrays
def load_from_directory(directory, batch_size, return_list=True):
    files = [file for file in os.listdir(directory) if file.endswith((".npz", ".npy"))]
    if batch_size is None or batch_size == 0:
        files_batch = files
    else:
        files_batch = random.sample(files, batch_size)

    tensors_list = []
    print("Loading tensors...")
    for filename in files_batch:
        file_path = os.path.join(directory, filename)
        name = splitext(basename(filename))[0]
        try:
            data = np.load(file_path)
            if filename.endswith(".npz"):
                for _, item in data.items():
                    tensors_list.append(TensorContainer(item, name, TensorType.NP_TENSOR)) 
            else:
                tensors_list.append(TensorContainer(item, name, TensorType.NP_TENSOR))
        except Exception as e:
            print(f"Error loading {filename} file: {str(e)}")
            print("file path:", file_path)
            return
        
    print(f"Load complete. {len(files_batch)} files loaded succesfully.")
    return tensors_list

def load_tensors_as_list(input_directory, n=0, tensor_type=TensorType.TF_TENSOR):
    tensors = load_from_directory(input_directory, n)
    if tensor_type == TensorType.TF_TENSOR:
        utils.convert_to_tf(tensors)
    return tensors