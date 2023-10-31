import os
from os.path import basename, splitext
import random
import argparse
import numpy as np
import tensorflow as tf
from utils import TensorType, TensorContainer

# Setting global seed
random.seed(42)

# Load an entire tensors batch from "directory" (.npz or .npy files) and return a list of narrays
def load_from_directory(directory, batch_size, return_list=True):
    files = [file for file in os.listdir(directory) if file.endswith((".npz", ".npy"))]
    if batch_size is None or batch_size == 0:
        files_batch = files
    else:
        files_batch = random.sample(files, batch_size)

    tensors_dict = []

    for filename in files_batch:
        file_path = os.path.join(directory, filename)
        name = splitext(basename(filename))[0]
        print(name)
        try:
            data = np.load(file_path)
            if filename.endswith(".npz"):
                for _, item in data.items():
                    tensors_dict.append(TensorContainer(item, name, TensorType.NP_TENSOR)) 
            else:
                tensors_dict.append(TensorContainer(item, name, TensorType.NP_TENSOR))
        except Exception as e:
            print(f"Error loading {filename} file: {str(e)}")
            print("file path:", file_path)
            return
        
    print(f"Load complete. {len(files_batch)} files loaded succesfully.")

    if return_list:
        return tensors_dict
    else:
        print("No-Dict Returned.")

# Convert a list of narray into a list of tf tensors
def convert_to_tf(tensors:list):
    for tensor_c in tensors:
        tensor_c.tensor = tf.convert_to_tensor(tensor_c.get_tensor())
        tensor_c.tensor_type = TensorType.TF_TENSOR
    return


def load_tensors(input_path, n=0, tensor_type=TensorType.TF_TENSOR):
    tensors = load_from_directory(input_path, n)
    if tensor_type == TensorType.TF_TENSOR:
        convert_to_tf(tensors)
    return tensors

def main(args):
    np_arr_list = load_from_directory(args.input_path, args.n ,return_list=True)
    tf_arr_list = convert_to_tf(np_arr_list)
    print(tf_arr_list)

def parse_args():
    parser = argparse.ArgumentParser(
        prog='Loader Module',
        description='Load numpy file (.npz .npy). If normaly execuded print the results',
        epilog='')
    
    parser.add_argument("input_path", help="The inputh path where are loceted all the tensors")
    parser.add_argument("output_path", help="")
    parser.add_argument("-n", "--n", default=0, type=int ,help="Import size (default: all the files)")

    return parser.parse_args()

if __name__  == "__main__":
    main(parse_args())