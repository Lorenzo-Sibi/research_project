import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

import loader

# Creazione di un tensore di esempio (sostituisci con il tuo tensore effettivo)
# Il tensore dovrebbe avere la forma (n_campioni, n_dimensioni)

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

def plot_statistics(tensor_list):
    # Inizializza le liste per raccogliere le statistiche
    std_list = []
    var_list = []
    mean_list = []
    min_list = []
    max_list = []

    # Calcola le statistiche per ciascun tensore nella lista
    for tensor in tensor_list:
        if not isinstance(tensor, np.ndarray):
            tensor = tensor.numpy()  # Converte il tensore in un array NumPy
        std_list.append(np.std(tensor))
        var_list.append(np.var(tensor))
        mean_list.append(np.mean(tensor))
        min_list.append(np.min(tensor))
        max_list.append(np.max(tensor))

    # Crea un istogramma per ciascuna statistica
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    axs[0].hist(std_list, bins=20, color='blue', alpha=0.7)
    axs[0].set_title('Standard Deviation')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(var_list, bins=20, color='green', alpha=0.7)
    axs[1].set_title('Variance')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(mean_list, bins=20, color='orange', alpha=0.7)
    axs[2].set_title('Mean')
    axs[2].set_xlabel('Value')
    axs[2].set_ylabel('Frequency')

    axs[3].hist(min_list, bins=20, color='red', alpha=0.7)
    axs[3].set_title('Min.')
    axs[3].set_xlabel('Value')
    axs[3].set_ylabel('Frequency')

    axs[4].hist(max_list, bins=20, color='purple', alpha=0.7)
    axs[4].set_title('Max')
    axs[4].set_xlabel('Value')
    axs[4].set_ylabel('Frequency')

    plt.savefig('statistics_test.png')

def main():
    parser = argparse.ArgumentParser(
        prog='Visualize Data',
        description='Compute data analysis on a tensors batch (file must end with .npz)',
        epilog='')
    
    parser.add_argument("input_path", help="The inputh path where are loceted all the tnesors")
    parser.add_argument("output_path", default="./")
    parser.add_argument("-n", "--n", default=1, type=int ,help="Tensors batch size (default: 10)")

    args = parser.parse_args()

    np_tensors_list = loader.load_from_directory(args.input_path, args.n)
    my_TSNE(np_tensors_list[0])

if __name__ == "__main__":
    main()