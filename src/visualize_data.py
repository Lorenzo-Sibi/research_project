from os.path import basename, splitext, join
from os import listdir
import argparse

import tensorflow as tf
import numpy as np
import pandas as pd

from src.utils import TensorContainer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from src import loader, utils

OPERATIONS = ['statistics-all', 'TSNE']
COLORS = ["blue", "green", "red", "orange", "purple"]

# Script per il plotting dei dati

def statistics_axis(tensor_list, axis=0, bins=40, output_path="./", title=None):
    if not title:
        title = 'statistic_indexes'
    
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

        # Compute the mean of specified measure along 0 axis (along tensors)
        avg_stat = np.mean(axis_stats, axis=0)
        statistics[measure.lower()] = avg_stat
    # Crea un set di grafici, uno per ciascuna misura statistica
    fig, axs = plt.subplots(len(measures), 1, figsize=(12, 20))

    for i, measure in enumerate(measures):
        axs[i].hist(statistics[measure.lower()].flatten(), bins=bins, color=COLORS[i], alpha=0.7)
        axs[i].set_title(f"Distribution of '{measure}' between {len(tensor_list)} tensors on axis {axis}")
        axs[i].set_xlabel(f'Value {measure}')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(join(output_path, title + f'_axis{axis}.png'))

def plot_tensors_subtraction(tensor1, tensor2, output_path=None, cmap="grey", title="unknown", normalize=True, save=True):
    if not output_path:
        output_path = "./"
    tensors_diff = utils.tensors_subtraction(tensor1, tensor2, normalize=normalize)
    tensor_diff_2D = utils.from_3D_tensor_to_2D(tensors_diff)
    plt.title(title)
    plt.imshow(tensor_diff_2D, cmap=cmap)
    if save:
        plt.savefig(join(output_path, title + ".png"))
    else:
        plt.show()
def plot_latent_representation(tensor, output_path=None, cmap="grey", name="unknown", save=True):
    if not output_path:
        output_path = "./"
    latent_rep = utils.from_3D_tensor_to_2D(tensor)
    plt.title(name)
    plt.imshow(latent_rep, cmap=cmap)
    if save:
        plt.savefig(join(output_path, f"latent_space_image_{name}.png"))
    else:
        plt.show()

def plot_latent_representation_all(input_directory, output_directory):
    latents_list = loader.load_tensors_as_list(input_directory)
    for i, laten_space in enumerate(latents_list):
        print(f"Plotting {i}/{len(latents_list)}")
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
    fft_magnitude = np.abs(fft_magnitude)  # Calculate the FFt magnitude

    # Plot FFT's spectrum
    plt.imshow(np.log(fft_magnitude), cmap='viridis')  # logaritmic scale
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