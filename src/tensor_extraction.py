import os
import shutil
import sys
import random
import argparse
import numpy as np
from pathlib import Path

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
    n, n_images = (0, len(image_filenames))
    print(f"{n_images} founded. Start extraction...")
    for image_filename in image_filenames:
        output_file = os.path.join(output_directory, Path(image_filename).stem + ".npz")
        input_file = os.path.join(input_directory, image_filename)
        print("FILENAME:", image_filename)
        print("input_file:", input_file)
        print("output_file:", output_file)
        tfci.dump_tensor(model, tensor_names, input_file, output_file)
        n += 1
        print("{0}/{1}".format(n, n_images))
    print("Extraction completed.")

def dump_tensor(input_directory, output_directory, model, tensor_names:list):
    """Dumps the given tensors of an image from a model to .npz files."""
    if not os.path.exists(output_directory):
        os.mkdir(output_directory, mode=777)
    tfci.dump_tensor(model, tensor_names, input_directory, output_directory)

def compress(model, input_directory, output_directory, rd_parameter=None):
    image_filenames = [image_filename for image_filename in os.listdir(input_directory)] # es. [image1.png, image2.png, image3.png]
    n, n_images = (0, len(image_filenames))
    print(f"{n_images} founded. Start compressing in .tfci files...")
    for image_filename in image_filenames:
        output_file = os.path.join(output_directory, Path(image_filename).stem + ".tfci")
        input_file = os.path.join(input_directory, image_filename)
        tfci.compress(model, input_file, output_file, rd_parameter)
        n += 1
        print("{0}/{1}".format(n, n_images))
    print("Compression completed.")

def decompress(input_directory, output_directory):
    filenames = [filename for filename in os.listdir(input_directory)] # es. [file1.png, file2.png, file3.png]
    n, n_files = (0, len(filenames))
    print(f"{n_files} founded. Start decompressing in .tfci files...")
    for filename in filenames:
        output_file = os.path.join(output_directory, Path(filename).stem + ".tfci")
        input_file = os.path.join(input_directory, filename)
        tfci.decompress(input_file, output_file)
        n += 1
        print("{0}/{1}".format(n, n_files))
    print("Decompression completed.")

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