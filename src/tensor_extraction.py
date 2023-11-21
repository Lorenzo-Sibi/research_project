import os
import shutil
import sys
import random
import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow_compression as tfc

from src import *
from src import utils, loader

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "compression-master/models"))
import tfci

TENSOR_NAMES =  ["hyperprior/entropy_model/conditional_entropy_model_3/add:0"]
MODEL_NAME = "hific-lo"

RESIZED_DESTINATION = "/mnt/ssd-data/sibi/resized_images"

# SEED
random.seed(RANDOM_SEED)

def dump_tensor_all(input_directory, output_directory, model, tensor_names:list):
    image_filenames = [image_filename for image_filename in os.listdir(input_directory)]
    n_images = len(image_filenames)
    print(f"{n_images} founded. Start extraction...")
    
    for i, image_filename in enumerate(image_filenames):
        output_file = os.path.join(output_directory, Path(image_filename).stem + ".npz")
        input_file = os.path.join(input_directory, image_filename)
        print("FILENAME:", image_filename)

        tfci.dump_tensor(model, tensor_names, input_file, output_file)
        
        print(f"{i+1}/{n_images}")
    print("Extraction completed.")

def dump_tensor(input_directory, output_directory, model, tensor_names:list):
    """Dumps the given tensors of an image from a model to .npz files."""
    if not os.path.exists(output_directory):
        os.mkdir(output_directory, mode=777)
    tfci.dump_tensor(model, tensor_names, input_directory, output_directory)

def compress(model, input_image, output_directory, rd_parameter=None):
  """Compresses a PNG file to a PNG file"""
  bitstring = tfci.compress_image(model, input_image, rd_parameter=rd_parameter)
  packed = tfc.PackedTensors(bitstring)
  receiver = tfci.instantiate_model_signature(packed.model, "receiver")
  tensors = packed.unpack([t.dtype for t in receiver.inputs])

  # Find potential RD parameter and turn it back into a scalar.
  for i, t in enumerate(tensors):
    if t.dtype.is_floating and t.shape == (1,):
      tensors[i] = tf.squeeze(t, 0)

  return receiver(*tensors)

def compress_all(model, input_directory, output_directory, rd_parameter=None): 
    image_filenames = [image_filename for image_filename in os.listdir(input_directory)] # es. [image1.png, image2.png, image3.png]
    n_images = len(image_filenames)
    print(f"{n_images} founded. Start compressing in .tfci files...")
    
    for i, image_filename in enumerate(image_filenames):
        output_file = os.path.join(output_directory, Path(image_filename).stem + ".tfci")
        input_file = os.path.join(input_directory, image_filename)

        input_image = tfci.read_png(input_file)
        compressed_image = compress(model, input_image, output_file, rd_parameter)
        print("PRE SQUEEZED IMAGE TENSOR", compressed_image.shape)
        tf.squeeze(compressed_image)
        print("SQUEEZED IMAGE TENSOR", compressed_image.shape)
        tfci.write_png(output_directory, compressed_image)

        print(f"{i+1}/{n_images}")
    print(f"Compression completed. {n_images} compressed.")

# def decompress(input_directory, output_directory):
#     filenames = [filename for filename in os.listdir(input_directory)] # es. [file1.png, file2.png, file3.png]
#     n_files = len(filenames)
#     print(f"{n_files} founded. Start decompressing in .tfci files...")

#     for i, filename in enumerate(filenames):
#         output_file = os.path.join(output_directory, Path(filename).stem + ".tfci")
#         input_file = os.path.join(input_directory, filename)
#         tfci.decompress(input_file, output_file)
#         print(f"{i+1}/{n_files}")
#     print("Decompression completed.")


# def execute(command):
#     filenames = [filename for filename in os.listdir(input_directory)] # es. [file1.png, file2.png, file3.png]
#     n, n_files = (0, len(filenames))
#     print(f"{n_files} founded. Start decompressing in .tfci files...")
#     for filename in filenames:
#         output_file = os.path.join(output_directory, Path(filename).stem + ".tfci")
#         input_file = os.path.join(input_directory, filename)
#         command(input_file, output_file)
#         n += 1
#         print("{0}/{1}".format(n, n_files))
#     print("Decompression completed.")