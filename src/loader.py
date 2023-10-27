import os
import random
import argparse
import numpy as np
import tensorflow as tf

# Setting global seed
random.seed(42)

# Load an entire tensors batch from "dir" (.npz or .npy files) and return a list of narrays
def load_from_directory(dir, batch_size, return_list=True):
    files = [file for file in os.listdir(dir) if file.endswith((".npz", ".npy"))]
    if batch_size is None or batch_size == 0:
        files_batch = files
    else:
        files_batch = random.sample(files, batch_size)
    tensors_list = []
    for filename in files_batch:
        file_path = os.path.join(dir, filename)
        try:
            data = np.load(file_path)
            if filename.endswith(".npz"):
                for key, item in data.items():
                    tensors_list.append(item) 
            else:
                tensors_list.append(data)
        except Exception as e:
            print(f"Error loading {filename} file: {str(e)}")
            print("file path:", file_path)
            return
        

    print(f"Load complete. {batch_size} files loaded succesfully")

    if return_list:
        return tensors_list
    else:
        print("No-List Returned")

# Convert a list of narray into a list of tf tensors
def convert_to_tf(tensors:list):
    tf_list = []
    for np_tensor in tensors:
        tf_list.append(tf.convert_to_tensor(np_tensor))
    return tf_list


def load_tensors(input_path, n):
    np_tensors = load_from_directory(input_path, n)
    tf_tensors = convert_to_tf(np_tensors)
    return tf_tensors

def main():
    parser = argparse.ArgumentParser(
        prog='Loader',
        description='Load numpy file (.npz .npy). If normaly execuded print the results',
        epilog='')
    
    parser.add_argument("input_path", help="The inputh path where are loceted all the tensors")
    parser.add_argument("output_path", help="")
    parser.add_argument("-n", "--n", default=0, type=int ,help="Import size (default: all the files)")

    args = parser.parse_args()
    
    np_arr_list = load_from_directory(args.input_path, args.n ,return_list=True)
    tf_arr_list = convert_to_tf(np_arr_list)
    print(tf_arr_list)


if __name__  == "__main__":
    main()