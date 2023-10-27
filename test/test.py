import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import tensor_extraction
import loader
import utils
import data_analysis
import visualize_data


def plot_latent_space_pca(tensor_list, num_components=2):

    # Appiattisci i tensori in una forma bidimensionale
    flattened_data = [tf.reshape(tensor, [tf.reduce_prod(tensor.shape)]).numpy() for tensor in tensor_list]
    combined_data = np.concatenate(flattened_data, axis=0)
    
    print(flattened_data)

    # Esegui la PCA per ridurre la dimensionalità
    pca = PCA(n_components=num_components)
    reduced_data = pca.fit_transform(flattened_data)

    # Visualizza i dati ridotti in un grafico 2D o 3D
    if num_components == 2:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
        plt.xlabel('Componente Principale 1')
        plt.ylabel('Componente Principale 2')
    elif num_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
        ax.set_xlabel('Componente Principale 1')
        ax.set_ylabel('Componente Principale 2')
        ax.set_zlabel('Componente Principale 3')

    plt.title('Grafico dello Spazio Latente Ridotto')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser("Testing functionalities...", )
    parser.add_argument("-l", "--list", nargs="+", required=True, help="Any args")

    return parser.parse_args()

def main(args):
    args_list = args.list # input_path, batch_size
    print(args_list)
    
    np_tensors = loader.load_from_directory(args_list[0], int(args_list[1]))
    visualize_data.plot_statistics(np_tensors)
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)