import os
import sys
import random
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow_compression as tfc

from src import *

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "compression-master/models"))
import tfci

TENSOR_NAMES =  ["hyperprior/entropy_model/conditional_entropy_model_3/add:0"]
MODEL_NAME = "hific-lo"

RESIZED_DESTINATION = "/mnt/ssd-data/sibi/resized_images"

# SEED
random.seed(RANDOM_SEED)

def list_tensors(model):
   tfci.list_tensors(model)

TENSORS_DICT = {
    "hific": "hyperprior/entropy_model/conditional_entropy_model_3/add:0",
    "mbt2018": "analysis/layer_3/convolution:0",
    "bmshj2018":"analysis/layer_2/convolution:0",
    "b2018": "analysis/layer_2/convolution:0",
    #"ms2020": "analysis/layer_2/convolution:0",
}


def dump_tensor_all(input_directory, output_directory, models, one_image=False):
    for i, model_class in enumerate(models):
        for variant in models[model_class]:
            for model in models[model_class][variant]:
                output_path = Path(output_directory, model_class, variant, model)
                if not output_path.exists():
                    output_path.mkdir(parents=True, exist_ok=True)
                tensor_name = TENSORS_DICT[model_class]
                print(f"MODEL CLASS: {model_class}\nMODEL: {model}\n\n")
                if one_image:
                    dump_tensor(input_directory, output_path, model, tensor_name)
                else:
                    dump_tensor_images(input_directory, output_path, model, tensor_name)
                print("_"*100, "\n\n")
        print(f"PROCESS {(i+1)/ len(models)* 100}% COMPETED.\n")

def dump_tensor_images(input_directory, output_directory, model, tensor_name):
    image_filenames = [image_filename for image_filename in os.listdir(input_directory)]
    n_images = len(image_filenames)
    print(f"{n_images} founded. Start dumping tensors...")
    
    for i, image_filename in enumerate(image_filenames):
        output_file = Path(output_directory, Path(image_filename).stem + ".npz")
        input_file = os.path.join(input_directory, image_filename)
        print("FILENAME:", image_filename)

        tfci.dump_tensor(model, [tensor_name], input_file, output_file)
        
        print(f"{i+1}/{n_images}")
    print("Dumping completed.")

def dump_tensor(input_filename, output_directory, model, tensor_name):
    """Dumps the given tensors of an image from a model to .npz files."""
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
    output_filename = Path(output_directory, Path(input_filename).stem + ".npz")
    tfci.dump_tensor(model, [tensor_name], input_filename, output_filename)

def compress_all():
    pass

def compress(model, input_image, rd_parameter=None):
  """Compresses a PNG file to a PNG file"""
  bitstring = tfci.compress_image(model, input_image, rd_parameter=rd_parameter)
  packed = tfc.PackedTensors(bitstring)
  receiver = tfci.instantiate_model_signature(packed.model, "receiver")
  tensors = packed.unpack([t.dtype for t in receiver.inputs])

  # Find potential RD parameter and turn it back into a scalar.
  for i, t in enumerate(tensors):
    if t.dtype.is_floating and t.shape == (1,):
      tensors[i] = tf.squeeze(t, 0)

  output_image, = receiver(*tensors)
  return output_image

def compress_images(model, input_directory, output_directory, rd_parameter=None): 
    image_filenames = [image_filename for image_filename in os.listdir(input_directory)] # es. [image1.png, image2.png, image3.png]
    n_images = len(image_filenames)
    print(f"{n_images} founded. Start compressing in images...")
    
    for i, image_filename in enumerate(image_filenames):
        output_file = os.path.join(output_directory, Path(image_filename).stem + ".png")
        input_file = os.path.join(input_directory, image_filename)

        input_image = tfci.read_png(input_file)
        compressed_image = compress(model, input_image, rd_parameter)
        tfci.write_png(output_file, compressed_image)

        print(f"{i+1}/{n_images}")
    print(f"Compression completed. {n_images} compressed.")