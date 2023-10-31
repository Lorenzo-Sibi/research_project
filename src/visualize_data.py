import tensorflow as tf
import numpy as np
import argparse
from utils import TensorType, TensorContainer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

import loader

# Creazione di un tensore di esempio (sostituisci con il tuo tensore effettivo)
# Il tensore dovrebbe avere la forma (n_campioni, n_dimensioni)

OPERATIONS = ['statistics', 'statistics-all', 'TSNE']
COLORS = ["blue", "green", "red", "orange", "purple"]

def my_TSNE(tensors):

    # Riduci la dimensionalità dei dati con t-SNE
    tsne = TSNE(n_components=3, perplexity=5)  # Riduzione a 3 dimensioni
    reduced_data = tsne.fit_transform(tensors)

    # Visualizzazione dei dati ridotti in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c='b', marker='o', label='Dati ridotti')

    # Personalizza il grafico
    ax.set_xlabel('Dimensione Ridotta 1')
    ax.set_ylabel('Dimensione Ridotta 2')
    ax.set_zlabel('Dimensione Ridotta 3')
    ax.set_title('Visualizzazione dati ridotti in 3D')

    # Mostra il grafico
    plt.show()

def statistics(tensor_list):
    # Inizializza le liste per raccogliere le statistiche
    std_list = []
    var_list = []
    mean_list = []
    min_list = []
    max_list = []
    
    # Calcola le statistiche per ciascun tensore nella lista
    for tensor_c in tensor_list:
        tensor = tensor_c.tensor
        if not isinstance(tensor, np.ndarray):
            tensor = tensor.numpy()
        std_list.append(np.std(tensor))
        var_list.append(np.var(tensor))
        mean_list.append(np.mean(tensor))
        min_list.append(np.min(tensor))
        max_list.append(np.max(tensor))

    # Crea un istogramma per ciascuna statistica
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    axs[0].hist(std_list, bins=20, color='blue', alpha=0.7)
    axs[0].set_title('Standard Deviation')
    axs[:3].set_xlabel('Value')
    axs[:3].set_ylabel('Frequency')

    axs[1].hist(var_list, bins=20, color='green', alpha=0.7)
    axs[1].set_title('Variance')

    axs[2].hist(mean_list, bins=20, color='red', alpha=0.7)
    axs[2].set_title('Mean')

    axs[3].hist(min_list, bins=20, color='orange', alpha=0.7)
    axs[3].set_title('Min.')

    axs[4].hist(max_list, bins=20, color='purple', alpha=0.7)
    axs[4].set_title('Max')

    plt.savefig('statistics_test.png')

def statistics_axis(tensor_list, axis=0):
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
    plt.savefig('statistics_test_axis'+ str(axis) + '.png')

def plot_3d_tensor(tensor, filename="3d_tensor_plot.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = tensor.shape
    x, y, z = range(x), range(y), range(z)

    X, Y, Z = [i for i in x for _ in y for __ in z], [i for _ in y for i in x for __ in z], [i for _ in y for __ in x for i in z]
    data = tensor.flatten()

    ax.scatter(X, Y, Z, c=data, cmap='viridis', marker='o')
    ax.set_xlabel('Asse X')
    ax.set_ylabel('Asse Y')
    ax.set_zlabel('Asse Z')
    ax.set_title('Rappresentazione 3D del Tensore')
    
    plt.savefig(filename)

def plot_slices_average(tensor, cmap="grey"):
    name = "unknown"
    if isinstance(tensor, TensorContainer):
        name = tensor.get_name()
        tensor = tensor.get_tensor()
    x, y, z = tensor.shape
    slices = []

    for i in range(z):
        # Seleziona il sottotensore lungo l'asse X
        slice_tensor = tensor[:, :, i]

        # Aggiungi il sottotensore alla lista delle "slices"
        slices.append(slice_tensor)

    # Calcola la media di tutte le "slices"
    avg_slice = np.mean(slices, axis=0)
    # Crea un grafico con la media delle immagini
    plt.imshow(avg_slice, cmap=cmap)
    plt.title(f'Mean Rappresentation of {name}')
    plt.show()

def my_PCA(tensor, n_components = 2):
    # 1. Riduci il tensore a una forma adatta per la PCA
    # Supponiamo che 'my_tensor' sia il tuo tensore con forma (x, y, z)
    x, y, z = tensor.shape
    tensor_flattened = tensor.reshape(-1, y)

    # 2. Esegui la PCA per ridurre le dimensioni del tensore
    pca = PCA(n_components=n_components)
    reduced_tensor = pca.fit_transform(tensor_flattened)

    # 3. Plotta il risultato ridotto
    if n_components == 2:
        plt.scatter(reduced_tensor[:, 0], reduced_tensor[:, 1], marker='.')
        plt.xlabel('Prima Componente Principale')
        plt.ylabel('Seconda Componente Principale')
    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_tensor[:, 0], reduced_tensor[:, 1], reduced_tensor[:, 2], marker='.')
        ax.set_xlabel('Prima Componente Principale')
        ax.set_ylabel('Seconda Componente Principale')
        ax.set_zlabel('Terza Componente Principale')

    plt.show()

def plot_tensor_fft_spectrum(tensor, log_scale=False):
    if isinstance(tensor, TensorContainer):
        tensor = tensor.get_tensor()
    x, y, z = tensor.shape
    fft_results = []
    for i in range(z):
        fft_results.append(np.fft.fft2(tensor[:, :, i]))
    fft_magnitude = np.fft.fftshift(np.mean(fft_results, axis=0))
    fft_magnitude = np.abs(fft_magnitude)  # Calcola la magnitudine dell'FFT

    # Plotta lo spettro dell'FFT
    plt.imshow(np.log1p(fft_magnitude), cmap='viridis')  # Applica il logaritmo per una migliore visualizzazione
    plt.colorbar()
    plt.title('Spettro dell\'FFT del Tensore')
    plt.show()

def main(args):

    np_tensors_list = loader.load_from_directory(args.input_path, args.n)

    operation = args.op
    if operation == OPERATIONS[0]:
        statistics(np_tensors_list)
    elif operation == OPERATIONS[1]:
        for i  in range(0, 4):
            statistics_axis(np_tensors_list, axis=i)
    elif operation == OPERATIONS[2]:
        my_TSNE(np_tensors_list)

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

if __name__ == "__main__":
    args = parse_args()
    main(args)