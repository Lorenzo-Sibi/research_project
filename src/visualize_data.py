from os.path import basename, splitext, join
from os import listdir
import argparse

import tensorflow as tf
import numpy as np
import pandas as pd

from src.utils import TensorType, TensorContainer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from src import loader, utils

# Creazione di un tensore di esempio (sostituisci con il tuo tensore effettivo)
# Il tensore dovrebbe avere la forma (n_campioni, n_dimensioni)

OPERATIONS = ['statistics-all', 'TSNE']
COLORS = ["blue", "green", "red", "orange", "purple"]

def statistics_axis(tensor_list, axis=0, bins=40, output_path="./"):
    measures = ["Standard Deviation", "Variance", "Mean", "Min", "Max"]
    # Inizializza le liste per raccogliere le statistiche
    statistics = {
        "standard deviation": [],
        "variance": [],
        "mean": [],
        "min": [],
        "max": []
    }

    # Calcola le statistiche per ciascun tensore nella lista

    for measure in measures:
        func = None
        if measure == measures[0]:
            func = np.std
        elif measure == measures[1]:
            func = np.var
        elif measure == measures[2]:
            func = np.mean
        elif measure == measures[3]:
            func = np.min
        elif measure == measures[4]:
            func = np.max

        axis_stats = []
        for tensor in tensor_list:
            if not isinstance(tensor, np.ndarray):
                tensor = tensor.numpy()
            axis_stat = func(tensor, axis=axis) # Return an array of ndim elements (1 element if axis == 0)
            axis_stats.append(axis_stat)

        # Compute the mean of specified measure along 0 axis (tensors)
        avg_stat = np.mean(axis_stats, axis=0)
        statistics[measure.lower()] = avg_stat
    # Crea un set di grafici, uno per ciascuna misura statistica
    fig, axs = plt.subplots(len(measures), 1, figsize=(12, 20))

    for i, measure in enumerate(measures):
        axs[i].hist(statistics[measure.lower()].flatten(), bins=40, color=COLORS[i], alpha=0.7)
        axs[i].set_title(f"Distribution of '{measure}' between {len(tensor_list)} tensors on axis {axis}")
        axs[i].set_xlabel(f'Value {measure}')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(join(output_path, 'statistics_test_axis' + str(axis) + '.png'))

def plot_latent_representation(tensor, output_path=None, cmap="grey"):
    name = "unknown"
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
    plt.imshow(avg_slice, cmap=cmap)
    plt.savefig(join(output_path, f"latent_space_image_{name}.png"))

def plot_latent_representation_all(input_directory, output_directory):
    latents_list = loader.load_tensors_as_list(input_directory)
    for laten_space in latents_list:
        plot_latent_representation(laten_space, output_path=output_directory)

def plot_tensor_fft_spectrum(tensor, log_scale=True, save_in="./"):
    name = "unknown"
    if isinstance(tensor, TensorContainer):
        name = tensor.get_name()
        tensor = tensor.get_tensor()
    x, y, z = tensor.shape
    fft_results = []
    for i in range(z):
        fft_results.append(np.fft.fft2(tensor[:, :, i]))
    fft_magnitude = np.fft.fftshift(np.mean(fft_results, axis=0))
    fft_magnitude = np.abs(fft_magnitude)  # Calcola la magnitudine dell'FFT

    # Plotta lo spettro dell'FFT
    plt.imshow(np.log1p(fft_magnitude), cmap='viridis')  # Applica il logaritmo per una migliore visualizzazione
    plt.title(f'Tensor spectrum (FFT) {name}')
    plt.savefig(join(save_in, name))

def plot_pca(images_lists, labels, n_components=2):
    pca_results = []
    for image_list in images_lists:
        matrix = np.squeeze(np.array([np.array(image).flatten().reshape(1, -1) for image in image_list]))
        print("MATRIX SHAPE", matrix.shape)
        pca = PCA(n_components=2, random_state=42)
        pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
        pca_results.append(pipe.fit_transform(matrix))
    for pca_result, label in zip(pca_results, labels):
        plt.scatter(pca_result[:, 0], pca_result[:, 1], label=label)
        plt.xlabel('First Main Component')
        plt.ylabel('Second Main Component')
    plt.legend()
    plt.show()

def plot_tsne(images_lists, labels, n_components=2, perplexity=5):
    tsne_results = []
    for image_list in images_lists:
        matrix = np.squeeze(np.array([np.array(image).flatten().reshape(1, -1) for image in image_list]))
        print("MATRIX SHAPE", matrix.shape)
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        pipe = Pipeline([('scaler', StandardScaler()), ('tsne', tsne)])
        tsne_results.append(tsne.fit_transform(matrix))
    for tsne_result, label in zip(tsne_results, labels):
        print("SHAPE RESULT", tsne_result.shape)
        plt.scatter(tsne_result[:], tsne_result[:], label=label)
        plt.xlabel('First Main Component')
        plt.ylabel('Second Main Component')
    plt.legend()
    plt.show()
def parse_args():
    parser = argparse.ArgumentParser(
        prog='Visualize Data',
        description='Compute data analysis on a tensors batch (file must end with .npz)',
        epilog='')
    
    parser.add_argument("input_path", help="The inputh path where are loceted all the tnesors")
    parser.add_argument("output_path", default="./")
    parser.add_argument("-op", required=True, choices=OPERATIONS)
    parser.add_argument("-n", "--n", default=0, type=int ,help="Tensors batch size (default: all files)")

    return parser.parse_args()

def main(args):

    image_directory = args.input_path
    image_list = loader.load_images_from_directory(image_directory)
    plot_pca(image_list, n_components=3)  # Imposta il numero di componenti desiderato (2 o 3)

    # tensor_container_list = loader.load_from_directory(args.input_path, args.n)
    # tensors_list = utils.convert_to_tensor_list(tensor_container_list)
    # operation = args.op
    # if operation == OPERATIONS[0]:
    #     for i  in range(0, 4):
    #         statistics_axis(tensors_list, axis=i, output_path=args.output_path)
    #     statistics_axis(tensors_list, axis=None, output_path=args.output_path)
    # elif operation == OPERATIONS[1]:
    #     my_TSNE(tensors_list)

if __name__ == "__main__":
    args = parse_args()
    main(args)