import os
import sys
import random
from PIL import Image
import enum
from src import *
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "compression-master/models"))
import tfci

random.seed(RANDOM_SEED)

class TensorType(enum.Enum):
    NP_TENSOR = "np_tensor"
    TF_TENSOR = "tf_tensor"

class TensorContainer:
    def __init__(self, tensor, name:str, tensor_type:TensorType):
        self.__tensor = tensor
        self.__name = name
        self.__tensor_type = tensor_type
    
    @property
    def tensor(self):
        return self.__tensor
    
    @property
    def name(self):
        return self.__name
    
    @property
    def dtype(self):
        return self.__tensor.dtype
    
    @property
    def ttype(self):
        return self.__tensor_type

    @property
    def shape(self):
        return self.__tensor.shape

    @property
    def ndim(self):
        return self.__tensor.ndim
    
    @staticmethod
    def convert(tensor_container, ttype):
        if tensor_container.ttype == ttype:
            return
        else:
            if(ttype == TensorType.NP_TENSOR):
                tensor_container.np_tensor()
            elif(ttype == TensorType.TF_TENSOR):
                tensor_container.tf_tensor()
    
    def squeeze(self):
        if self.ttype == TensorType.NP_TENSOR:
            self.__tensor = np.squeeze(self.__tensor)
        elif self.ttype == TensorType.TF_TENSOR:
            self.__tensor = tf.squeeze(self.__tensor)
    
    def get_tensor(self):
        return self.__tensor

    def get_name(self):
        return self.__name

    def get_tensor_type(self):
        return self.__tensor_type
    
    def np_tensor(self):
        if self.__tensor_type == TensorType.NP_TENSOR:
            return
        assert isinstance(self.__tensor, tf.Tensor)
        
        self.__tensor_type = TensorType.NP_TENSOR
        self.__dtype = self.__tensor.dtype
        self.__shape = self.__tensor.shape
        self.__tensor = self.__tensor.numpy() # type: ignore
        
    def tf_tensor(self):
        if self.__tensor_type == TensorType.TF_TENSOR:
            return
        tensor = self.__tensor
        assert isinstance(tensor, np.ndarray)
        
        self.__tensor = tf.convert_to_tensor(tensor)
        self.__tensor_type = TensorType.TF_TENSOR
        self.__dtype = self.__tensor.dtype
        self.__shape = self.__tensor.shape
    

    
    def __str__(self):
      return f'{self.tensor}, name: {self.name}, tensor_type: {self.__tensor_type}'


def convert_to_tensor_list(tensor_container_list:list, tensor_type:TensorType=TensorType.NP_TENSOR):
    return [tensor_c.tensor for tensor_c in tensor_container_list]

# Convert a list of narray into a list of tf tensors
def convert_to_tf(tensors:list):
    for tensor_c in tensors:
        tensor_c.tensor = tf.convert_to_tensor(tensor_c.get_tensor())
        tensor_c.tensor_type = TensorType.TF_TENSOR
    return

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

def list_tensors(model):
  """Lists all internal tensors of a given model."""
  def get_names_dtypes_shapes(function):
    for op in function.graph.get_operations():
      for tensor in op.outputs:
        yield tensor.name, tensor.dtype.name, tensor.shape

  sender = tfci.instantiate_model_signature(model, "sender")
  tensors = sorted(get_names_dtypes_shapes(sender))
  log = "Sender-side tensors:\n"
  for name, dtype, shape in tensors:
    log += f"{name} (dtype={dtype}, shape={shape})\n"
  log += "\n"

  receiver = tfci.instantiate_model_signature(model, "receiver")
  tensors = sorted(get_names_dtypes_shapes(receiver))
  log += "Receiver-side tensors:\n"
  for name, dtype, shape in tensors:
    log += f"{name} (dtype={dtype}, shape={shape})\n"
  return log    


def tensors_log(logdir='tensors_logs'):
    for i, model_class in enumerate(MODELS_DICT):
        for variant in MODELS_DICT[model_class]:
            for model in MODELS_DICT[model_class][variant]:
                path = os.path.join(logdir, model_class, variant)
                filename = os.path.join(path, model + "_tensors" + ".txt")
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(filename, "w") as f:
                   f.write(list_tensors(model))
        print(f"{i}/{len(MODELS_DICT)}")

def tensors_subtraction(tensor1, tensor2, normalize=True):
    if isinstance(tensor1, TensorContainer) or isinstance(tensor2, TensorContainer):
        type1 = tensor1.get_tensor_type()
        type2 = tensor2.get_tensor_type()
        tensor1 = tensor1.get_tensor()
        tensor2 = tensor2.get_tensor()
        if type1 != type2:
            raise ValueError(f"Tensors of different type. \nTensor 1:{type1} (type)\nTensor 2:{type2} (type)")
        assert(type1 == TensorType.TF_TENSOR)
    
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Tensors with different shape. \nTensor 1:{tensor1.shape}\nTensor 2:{tensor2.shape}")
    
    if normalize:
        # Normalizing both 3D tensors
        tensor1, _ = tf.linalg.normalize(tensor1, axis=(0, 1))
        tensor2, _ = tf.linalg.normalize(tensor2, axis=(0, 1))
    
    tensors_diff = tf.subtract(tensor1, tensor2)
    tensors_diff = tf.squeeze(tensors_diff, 0)
    return tf.abs(tensors_diff)

def from_3D_tensor_to_2D(tensor):
    if isinstance(tensor, TensorContainer):
        name = tensor.get_name()
        tensor = tf.squeeze(tensor.get_tensor())
    
    x, y, z = tensor.shape
    slices = []

    for i in range(z):
        slice_tensor = tensor[:, :, i]
        slices.append(slice_tensor)

    # Calcolate all slice's mean"
    avg_slice = np.mean(slices, axis=0)
    return avg_slice