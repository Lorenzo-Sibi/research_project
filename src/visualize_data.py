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