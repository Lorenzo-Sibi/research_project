import os
import numpy as np
import tensorflow as tf


def load_from_directory(dir, return_list=True):

    tensors_list = []

    for filename in os.listdir(dir):
        if filename.endswith(".npz"):
            file_path = os.path.join(dir, filename)
            try:
                data = np.load(file_path)
                file_list.append(data)
            except:
                print(f"Error loading {filename} file: {str(e)}")
    print("Load complete. All files loaded succesfully.")
    if return_list:
        return tensors_list
    else:
        return
    
def convert_to_tf_tensor(object):

    return

def load_tensors(dir):
    return

def main():
    
    return


if __name__  == "__main__":
    main()