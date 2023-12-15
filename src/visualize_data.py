from os.path import basename, splitext, join
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from src.utils import TensorContainer
from PIL import Image, ImageOps

from src import loader, utils, preprocess
from src.loader import Loader

OPERATIONS = ['statistics-all', 'TSNE']
COLORS = ["blue", "green", "red", "orange", "purple"]

PLOT_DIR = "plots"

IMAGE_SUPPORTED_EXTENSIONS = (".png", ".jpeg", ".jpg")
TENSOR_SUPPORTED_EXTENSIONS = (".npz", ".npy")

# Data plots script

"""
Per utilizzare il comando statistics_all le cartelle devono essere strutturate in questo modo:

main-directory
    |- model_class (es. b2028)
        |- variant1/ (es. gdn-128) 
            ...
        |- variant2/
            |- model1/
            |- model2/
                | ... files.nps ...

NON inserire la cartella di output all'interno della input_directory
"""


def statistics_all(input_directory, output_directory, axis=0, bins=40):
    
    # Itera sulle sottocartelle di main-directory
    for model_class in input_directory.iterdir():
        if model_class.is_dir():
            model_class_name = model_class.name

            # Crea la cartella principale per la classe del modello
            class_output_path = Path(output_directory, model_class_name)
            class_output_path.mkdir(parents=True, exist_ok=True)
            print("CLASS OUTPUT PATH", class_output_path)
            for variant in model_class.iterdir():
                if variant.is_dir():
                    variant_name = variant.name

                    # Crea la cartella per la variante
                    variant_output_path = class_output_path / variant_name
                    variant_output_path.mkdir(parents=True, exist_ok=True)

                    for model_folder in variant.iterdir():
                        if model_folder.is_dir():
                            # Ottieni la lista dei tensori dalla cartella "model_folder"
                            tensor_list = load_tensors_from_model(model_folder)
                            model_name = model_folder.name
                            print(model_folder)
                            print(f"Processing {model_class_name}/{variant_name}/{model_name}...")

                            # Calcolo delle statistiche e salvataggio nella sottocartella
                            title = f"{model_name}_stats"
                            statistics(tensor_list, axis=axis, bins=bins, output_path=str(variant_output_path), title=title)
                        else:
                            print("Folder structure not respected!")
                            return
    print("Processing completed.")
    return


"""
Possible refactor implementation for load_from_directory (loader.py module)
"""
def load_tensors_from_model(model_path): 
    # Carica i tensori dalla cartella "model"
    tensor_list = []
    for file_path in model_path.glob("*.npz"):
        with np.load(file_path) as data:
            tensor_list.extend(item for _, item in data.items())
    return tensor_list


def statistics(tensor_list, axis=0, bins=40, output_path="./", title=None):
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
            if isinstance(tensor, TensorContainer):
                tensor = tensor.get_tensor()
            if not isinstance(tensor, np.ndarray):
                tensor = tensor.numpy()
            tensor = np.squeeze(tensor)
            x, y, z = tensor.shape
            for layer in range(z):
                layer_norm = np.linalg.norm(tensor[:, :, layer])
                tensor[:, :, layer] = tensor[:, :, layer] / layer_norm
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
    plt.close("all")

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
    plt.close("all")
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
    plt.close("all")

def plot_latent_representation_all(input_directory, output_directory):
    latents_list = loader.load_tensors_as_list(input_directory)
    for i, laten_space in enumerate(latents_list): # type: ignore
        print(f"Plotting {i}/{len(latents_list)}") # type: ignore
        plot_latent_representation(laten_space, output_path=output_directory)


def plot_spectrum(input_path, output_path):
    """
        Function to pass to data.py. It handle all different cases 
        (single files, or directory containing multiple files).
    """
    if not output_path.is_dir():
        raise TypeError(f"Output path {output_path} is not a directory.")
    
    if input_path.is_file(): # handle the case of single file
        
        if input_path.suffix in IMAGE_SUPPORTED_EXTENSIONS or input_path.suffix in TENSOR_SUPPORTED_EXTENSIONS:
            plot_single_fft_spectrum(input_path, output_path)
        else:
            raise ValueError(f"Given input filename's extension {input_path.suffix} is not compatible. ")
        
    else:                   # handle the case of directory
        for input_filename in input_path.iterdir():
            if input_filename.suffix in IMAGE_SUPPORTED_EXTENSIONS or input_path.suffix in TENSOR_SUPPORTED_EXTENSIONS:
                plot_single_fft_spectrum(input_filename, output_path)

def plot_single_fft_spectrum(input_path, output_path):
    if not input_path.is_file():
        raise TypeError("Input filename given is not a file.")
    if not output_path.is_dir():
        raise TypeError("Output path is not a directory.")
    
    array = Loader.load(input_path) # it automatically load the file as array
    if isinstance(array, TensorContainer):
        array = array.tensor
    array = np.array(array)
    
    fft_spectrum = preprocess.array_fft_spectrum(array)
    
    plt.imshow(fft_spectrum, cmap="viridis")
    plt.title(f'Magnitude Spectrum of {input_path.name}'), plt.xticks([]), plt.yticks([])
    plt.savefig(output_path / f"{input_path.stem}_spectrum.png")
    plt.close("all")

def plot_average_spectrum(input_directory, output_directory, title=None):
    if not output_directory.is_dir():
        raise TypeError(f"Error. Output path: {output_directory} is not a directory.")
    if not input_directory.is_dir():
        raise TypeError(f"Error. Input path: {input_directory} is not a directory.")
    
    if not title:
        title = f"{input_directory.stem}_average_spectrum"
        
    fft_avg_spectrum = preprocess.average_fft(input_directory)
    
    plt.imshow(fft_avg_spectrum, cmap="viridis")
    plt.colorbar()
    plt.savefig(output_directory / f"{title}.png", dpi=600)
    plt.close("all")
    
def plot_all_average_spectrum(input_directory, output_directory):
    if not output_directory.is_dir():
        raise TypeError(f"Error. {output_directory} is not a directory.")
    if not input_directory.is_dir():
        raise TypeError(f"Error. {input_directory} is not a directory.")
        
    for model_class in input_directory.iterdir():
        if model_class.is_dir():
            model_class_name = model_class.name

            class_output_path = Path(output_directory, model_class_name)
            class_output_path.mkdir(parents=True, exist_ok=True)
            
            for variant in model_class.iterdir():
                if variant.is_dir():
                    variant_name = variant.name

                    variant_output_path = class_output_path / variant_name
                    variant_output_path.mkdir(parents=True, exist_ok=True)

                    for model_folder in variant.iterdir():
                        if model_folder.is_dir():
                            print(model_folder)
                            model_name = model_folder.name
                            title = f"{model_name}_average_spectrum"
                            print(f"Processing {model_class_name}/{variant_name}/{model_name}...")
                            plot_average_spectrum(model_folder, variant_output_path, title=title)
                        
                        else:
                            raise ValueError("Folder Structure not respected.")
        else:
            raise ValueError("Folder Structure not respected.")
    print("Processing completed.")
    return