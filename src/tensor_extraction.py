import os
import shutil
import sys
import random
import subprocess
import argparse
import tensorflow as tf
import numpy as np
import PIL

import utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "compression-master/models"))
import tfci

TENSOR_NAMES =  ["hyperprior/entropy_model/conditional_entropy_model_3/add:0"]
MODEL_NAME = "hific-lo"
N_IMAGES = 10

RESIZED_DESTINATION = "/mnt/ssd-data/sibi/resized_images"

# SEED
random.seed(42)

def dump_tensor(input, output_path, n_images):
    n = 0
    for image_file in os.listdir(input):
        output_file = os.path.join(output_path, image_file + ".npz")
        input_file = os.path.join(input, image_file)

        tfci.dump_tensor(MODEL_NAME, TENSOR_NAMES, input_file, output_file)
        #parametri_script = ["python", args.percorso_script_tfci, metodo_dump, args.nome_tensore, percorso_immagine]
        n += 1
        print("{0}/{1}".format(n, n_images))
    return


def main():
    args = parsing_args()

    if not os.path.exists(args.input_path):
        print("images path not avaiable or doesn't exist")
        return

    # Ottieni un elenco di tutte le immagini nella cartella COMPLETE PATH
    images_set = [os.path.join(args.input_path, file) for file in os.listdir(args.input_path) if file.endswith((".jpg", ".png"))]
    n_images = args.batch_size
    # Campiona casualmente 100 immagini
    images_batch = random.sample(images_set, n_images)

    resolution = utils.find_resolution(images_batch)

    resized_images_path = os.path.join(args.output_path, "resized_images")
    os.makedirs(resized_images_path, exist_ok=True)

    utils.load_and_process_image(images_batch, resized_images_path, resolution)
    print(f"Resizing complete. {len(images_batch)} images resized.")

    dump_tensor(resized_images_path, args.output_path, n_images)
    print(f"DONE. {n_images} Images Dumped.\nDumped images directory: {args.output_path}")
    shutil.rmtree(resized_images_path, ignore_errors=True)

def parsing_args():
    parser = argparse.ArgumentParser(description="Script per campionare immagini casuali e eseguire tfci.py su di esse.")
    parser.add_argument("input_path", type=str, help="Il percorso della cartella contentente le immagini")
    parser.add_argument("output_path", type=str, help="Il percorso della cartella delle contente i file .npz di ogni immagine")
    parser.add_argument("tensor-name", type=str, help="The name of the specific tensor to extract")
    parser.add_argument("model", type=str, help="the name of the specific model (es 'hific-lo')")
    parser.add_argument("--batch-size", nargs='?', default=N_IMAGES, type=int)
    
    return parser.parse_args()

if __name__ == "__main__":
    main()
