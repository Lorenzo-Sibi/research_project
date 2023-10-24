import os
import sys
import random
import subprocess
import argparse
import tensorflow as tf
import numpy as np

print(os.path.dirname(__file__))

# Importing the tfci.py script
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "compression-master/models"))
#sys.path.append('./../compression-master/models')  # Aggiunge il percorso alla lista dei percorsi di importazione
import tfci

TENSOR_NAMES =  ["hyperprior/entropy_model/conditional_entropy_model_3/add:0"]
MODEL_NAME = "hific-lo"
N_IMAGES = 10

def load_and_process_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)  # Decodifica l'immagine
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalizza l'immagine
    return img

def main():
    parser = argparse.ArgumentParser(description="Script per campionare immagini casuali e eseguire tfci.py su di esse.")
    parser.add_argument("input_path", type=str, help="Il percorso della cartella contentente le immagini")
    parser.add_argument("output_path", type=str, help="Il percorso della cartella delle contente i file .npz di ogni immagine")
    parser.add_argument("tensor-name", type=str, help="The name of the specific tensor to extract")
    parser.add_argument("model", type=str, help="the name of the specific model (es 'hific-lo')")
    parser.add_argument("--batch-size", nargs='?', default=N_IMAGES, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print("images path not avaiable or doesn't exist")
        return

    # SEED
    random.seed(42)

    # Ottieni un elenco di tutte le immagini nella cartella
    images_set = [file for file in os.listdir(args.input_path) if file.endswith((".jpg", ".png"))]
    n_images = args.batch_size
    # Campiona casualmente 100 immagini
    images_batch = random.sample(images_set, n_images)
    n = 0
    for image in images_batch:
        image_path = os.path.join(args.input_path, image)
        output_file = os.path.join(args.output_path, os.path.splitext(image)[0] + ".npz") 
        #img = load_and_process_image(image_path)

        tfci.dump_tensor(MODEL_NAME, TENSOR_NAMES, image_path, output_file)
        # Esegui lo script tfci.py
        #parametri_script = ["python", args.percorso_script_tfci, metodo_dump, args.nome_tensore, percorso_immagine]
        #subprocess.run(parametri_script)
        n += 1
        print("{0}/{1}".format(n, n_images))

    print("DONE", n, "Images Dumped.", "\nDumped images directory:", args.output_path)
if __name__ == "__main__":
    main()
