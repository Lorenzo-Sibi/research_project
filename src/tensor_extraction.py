import os
import sys
import random
from pathlib import Path
import tensorflow as tf
import tensorflow_compression as tfc

from src import *

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "compression-master/models"))
import tfci

random.seed(RANDOM_SEED)

def list_tensors(model):
   tfci.list_tensors(model)

TENSORS_DICT = {
    "hific": "hyperprior/entropy_model/conditional_entropy_model_3/add:0",
    "mbt2018": "analysis/layer_3/convolution:0",
    "bmshj2018":"analysis/layer_2/convolution:0",
    "b2018": "analysis/layer_2/convolution:0",
    #"ms2020": "analysis/layer_2/convolution:0",
}

"""
LATENT SPACES EXTRACTON IMPLEMENTATION
"""

def dump_tensor_all(input_directory, output_directory, models, one_image=False):
    for i, model_class in enumerate(models):
        for variant in models[model_class]:
            for model in models[model_class][variant]:
                output_path = Path(output_directory, model_class, variant, model)
                if not output_path.exists():
                    output_path.mkdir(parents=True, exist_ok=True)
                tensor_name = TENSORS_DICT[model_class]
                print(f"MODEL CLASS: {model_class}\nMODEL: {model}\n\n")
                if one_image:
                    dump_tensor(input_directory, output_path, model, tensor_name)
                else:
                    dump_from_dir(Path(input_directory, model_class, variant, model), output_path, model)
                print("\n\n")
        print(f"TOTAL PROCESS {(i+1)/ len(models)* 100}% COMPETED.\n\n")

def dump(input_path, output_path, model):
    try:
        if input_path.is_dir():
            dump_from_dir(input_path, output_path, model)
        else:
            dump_from_file(input_path, output_path, model)
    except Exception as err:
        print(err)
        return
    
def dump_from_dir(input_directory, output_directory, model):
    if not input_directory.is_dir():
        raise ValueError(f"Error. {input_directory} is not a directory.")
    if not output_directory.is_dir():
        raise ValueError(f"Error. {output_directory} is not a directory.")

    for filename in input_directory.iterdir():
        print("FILENAME: ", filename)
        if not filename.is_file() or (filename.is_file() and filename.suffix not in (".png")):
            continue
        dump_from_file(filename, output_directory, model)
    return

def dump_from_file(input_path, output_path, model):
    if not input_path.is_file():
        raise ValueError(f"Error. {input_path} is not a file.")
    if input_path.suffix not in (".png"):
        raise ValueError(f"Error. {input_path.suffix} is not compatible. PNG files are the only compatible.")
    if not output_path.is_dir():
        raise ValueError(f"Error. {output_path} is not a directory.")
    
    output_filename = Path(output_path, f"{input_path.stem}.npz")
    print(MODELS_LATENTS_DICT[model])
    print("OUTOU PATH: ", output_filename)
    tensor_name = MODELS_LATENTS_DICT[model]
    
    tfci.dump_tensor(model, [tensor_name], str(input_path), str(output_filename))

def dump_tensor_images(input_directory, output_directory, model, tensor_name):
    image_filenames = [image_filename for image_filename in os.listdir(input_directory)]
    n_images = len(image_filenames)
    print(f"{n_images} founded. Start dumping tensors...")
    
    for i, image_filename in enumerate(image_filenames):
        output_file = Path(output_directory, Path(image_filename).stem + ".npz")
        input_file = os.path.join(input_directory, image_filename)

        tfci.dump_tensor(model, [tensor_name], input_file, output_file)
        
        sys.stdout.write(f"\r{image_filename}: {i+1}/{n_images}")
        sys.stdout.flush()
    sys.stdout.write(f"\rDumping completed for {model}")
    sys.stdout.flush()

def dump_tensor(input_filename, output_directory, model, tensor_name):
    """Dumps the given tensors of an image from a model to .npz files."""
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
    output_filename = Path(output_directory, Path(input_filename).stem + ".npz")
    tfci.dump_tensor(model, [tensor_name], input_filename, output_filename)


"""
COMPRESSION IMPLEMENTATION
"""

def compress_all(input_directory, output_directory, models, one_image=False):
    for i, model_class in enumerate(models):
        for variant in models[model_class]:
            for model in models[model_class][variant]:
                output_path = os.path.join(output_directory, model_class, variant, model)
                if not Path(output_path).exists():
                    Path(output_path).mkdir(parents=True, exist_ok=True)
                print(f"\nMODEL CLASS: {model_class}\nMODEL: {model}")
                if one_image:
                    output_path = os.path.join(output_path, Path(input_directory).stem + ".png")
                    input_image = tfci.read_png(input_directory)
                    compressed_image = compress(model, input_image)
                    tfci.write_png(output_path, compressed_image)
                else:
                    compress_images(model, input_directory, output_path)
        print(f" PROCESS {(i+1)/ len(models)* 100}% COMPLETE.\n")
    pass

def compress(model, input_image, rd_parameter=None):
  """Compresses a PNG file to a PNG file"""
  bitstring = tfci.compress_image(model, input_image, rd_parameter=rd_parameter)
  packed = tfc.PackedTensors(bitstring)
  receiver = tfci.instantiate_model_signature(packed.model, "receiver")
  tensors = packed.unpack([t.dtype for t in receiver.inputs])

  # Find potential RD parameter and turn it back into a scalar.
  for i, t in enumerate(tensors):
    if t.dtype.is_floating and t.shape == (1,):
      tensors[i] = tf.squeeze(t, 0)

  output_image, = receiver(*tensors) # type: ignore
  return output_image

def compress_images(model, input_directory, output_directory, rd_parameter=None, ): 
    image_filenames = [image_filename for image_filename in os.listdir(input_directory)] # es. [image1.png, image2.png, image3.png]
    n_images = len(image_filenames)
    
    print(f"{n_images} founded. Start compressing images...")
    
    for i, image_filename in enumerate(image_filenames):
        output_file = os.path.join(output_directory, Path(image_filename).stem + ".png")
        input_file = os.path.join(input_directory, image_filename)

        input_image = tfci.read_png(input_file)
        compressed_image = compress(model, input_image, rd_parameter)
        tfci.write_png(output_file, compressed_image)

        sys.stdout.write(f"\r{i+1}/{n_images}")
        sys.stdout.flush()
        
    sys.stdout.write(f"\rCompression completed. {n_images} compressed.\n\n")
    sys.stdout.flush()
    