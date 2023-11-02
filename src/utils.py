import os
import sys
import enum
import tensorflow as tf
import numpy as np

class TensorType(enum.Enum):
    NP_TENSOR = "np_tensor"
    TF_TENSOR = "tf_tensor"

class TensorContainer:
    def __init__(self, tensor, name:str, tensor_type:TensorType):
        self.tensor = tensor
        self.name = name
        self.tensor_type = tensor_type

    def get_tensor(self):
        return self.tensor

    def get_name(self):
        return self.name

    def get_tensor_type(self):
        return self.tensor_type
    
    def __str__(self):
      return f'{self.tensor}, name: {self.name}, tensor_type: {self.tensor_type}'

# Find the correct resolution for a list of images path
def find_resolution(images_batch):
    resolutions = []
    # Compute the resolution for all images in the batch
    for image_path in images_batch:
        image = tf.image.decode_image(tf.io.read_file(image_path))
        resolutions.append(tf.shape(image)[:-1])
    average_resolution = tf.reduce_min(tf.stack(resolutions, axis=0), axis=0)
    return average_resolution

# Ridimensiona tutte le immagini alla risoluzione specificata
def load_and_process_image(images_path, output_path, resolution):
    print("Resizing images...")
    for image_file in images_path:
        image = tf.image.decode_image(tf.io.read_file(image_file), channels=3)
        resized_image = tf.image.resize(image, resolution)
        np_resized_image = resized_image.numpy()
        image_name = os.path.splitext(os.path.split(image_file)[-1])[0]
        tf.keras.utils.save_img(os.path.join(output_path, image_name + ".png"), np_resized_image, file_format="png")
    return

def reshape_all_3D(tensor_list:list):
    for tensor_c in tensor_list:
        tensor = tensor_c.get_tensor()
        if len(tensor.shape) == 3:
            continue
        else:
            if(tensor_c.get_tensor_type() == TensorType.NP_TENSOR):
                tensor_c.tensor = tensor.squeeze()
            else:
                tensor_c.tensor = tf.squeeze(tensor_c.tensor)
    return tensor_list

def convert_to_tensor_list(tensor_container_list:list, tensor_type:TensorType=TensorType.NP_TENSOR):
    return [tensor_c.tensor for tensor_c in tensor_container_list]

# Convert a list of narray into a list of tf tensors
def convert_to_tf(tensors:list):
    for tensor_c in tensors:
        tensor_c.tensor = tf.convert_to_tensor(tensor_c.get_tensor())
        tensor_c.tensor_type = TensorType.TF_TENSOR
    return