import os
import shutil
import sys
import random
import argparse
import numpy as np
from pathlib import Path
from src import utils, RANDOM_SEED
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
    print(n_images)
    for image_filename in image_filenames:
        output_file = os.path.join(output_directory, Path(image_filename).stem + ".npz")
        input_file = os.path.join(input_directory, image_filename)
        print("FILENAME:", image_filename)
        print("input_file:", input_file)
        print("output_file:", output_file)
        tfci.dump_tensor(model, tensor_names, input_file, output_file)
        n += 1
        print("{0}/{1}".format(n, n_images))
    return

def dump_tensor(input_directory, output_directory, model, tensor_names:list):
    """Dumps the given tensors of an image from a model to .npz files."""
    if not os.path.exists(output_directory):
        os.mkdir(output_directory, mode=777)
    tfci.dump_tensor(model, tensor_names, input_directory, output_directory)
    
    
def extract_tensors(input_path, output_path, tensor_names:list, model_name, batch_size=0):
    if not os.path.exists(input_path):
        print("images path not avaiable or doesn't exist")
        return
    # List of all images inside input_path directory (COMPLETE PATH)
    print("Loading all the .npz/.npy files...")
    images_set = [os.path.join(input_path, file) 
                  for file in os.listdir(input_path) 
                  if file.endswith((".jpg", ".png"))]
    print("Loading completed.")
    n_images = batch_size
    if n_images == 0 or n_images is None:
        images_batch = images_set
        n_images = len(images_set)
    else:
        images_batch = random.sample(images_set, n_images)

    resolution = utils.find_resolution(images_batch) # find the right resolution for all images inside the batch

    resized_images_path = os.path.join(output_path, "resized_images")
    os.makedirs(resized_images_path, exist_ok=True)

    utils.load_and_process_image(images_batch, resized_images_path, resolution)
    print(f"Resizing complete. {len(images_batch)} images resized.")

    dump_tensor(resized_images_path, output_path, tensor_names, model_name,n_images)
    print(f"DONE. {n_images} Images Dumped.\nDumped images directory: {output_path}")

    shutil.rmtree(resized_images_path, ignore_errors=True)

    return

def main(args):
    
    extract_tensors(args.input_path, args.output_path, TENSOR_NAMES, MODEL_NAME, args.b)


def parsing_args():
    parser = argparse.ArgumentParser(description="Script for sampling images and executing the 'dump' command from tfci.py script")
    parser.add_argument("input_path", type=str, help="The directory path containing all the images")
    parser.add_argument("output_path", type=str, help="The directory path of destination, where store all the .npz file dumped")
    parser.add_argument("tensor-name", type=str, help="The name of the specific tensor to extract")
    parser.add_argument("model", type=str, help="the name of the specific model (es 'hific-lo')")
    parser.add_argument("-b", nargs='?', default=0, type=int)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parsing_args()
    main(args)
