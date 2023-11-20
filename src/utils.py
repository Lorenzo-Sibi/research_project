import os
import random
from PIL import Image
import enum
from src import *
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

random.seed(RANDOM_SEED)
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

def load_images_as_list(input_path, resolution=None, batch_size=0):
    if not os.path.exists(input_path):
        print("Path not avaiable or doesn't exist")
        return
    # List of all images inside input_path directory (COMPLETE PATH)
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
    if resolution == None:
        resolution = find_resolution(images_batch)
    image_list = []
    for path in images_batch:
        if path.endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(path).resize(resolution)
            image_list.append(image)
    print("Loading completed.")
    return image_list

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

def create_labeled_dataset(image_folder, label, target_size=(64, 64), batch_size=4, shuffle=True):
    # Crea un generatore di dati utilizzando tf.data

    data_generator = ImageDataGenerator(rescale=1./255)
    data_generator = data_generator.flow_from_directory(
        image_folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle,
        classes=[label]
    )
    print("DATA GENERATOR:", data_generator)
    # Crea un dataset a partire dal generatore di dati
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, target_size[0], target_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32)
        )
    )
    # Espandi le dimensioni del tensore di dati per avere la forma corretta
    #dataset = dataset.map(lambda x, y: (tf.squeeze(x, axis=1), y))
    return dataset

def create_image_dataframe(image_folder, label):
    image_data = []
    
    # Verifica che la directory esista
    if not os.path.exists(image_folder):
        raise ValueError(f"La directory '{image_folder}' non esiste.")
    
    # Elabora tutte le immagini nella directory
    for filename in os.listdir(image_folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)
            
            # Aggiungi i dati dell'immagine al DataFrame
            image_data.append({
                "Image": image,
                "Label": label
            })
    
    # Crea un DataFrame Pandas dai dati delle immagini
    df = pd.DataFrame(image_data)
    
    return df