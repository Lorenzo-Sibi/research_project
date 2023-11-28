import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from src import *
from src import loader

HIGH_PASS_KERNEL = np.array([
    [0.0, -1.0, 0.0], 
    [-1.0, 4.0, -1.0],
    [0.0, -1.0, 0.0]
])

random.seed(RANDOM_SEED)

def check_sizes(image_list, target_width, target_height):
    for image in image_list:
        width, height = image.size
        if(width < target_width or height < target_height):
            raise ValueError(
                f"Wrong sizes: indicated sizes: {target_width}x{target_height}",
                f"Image {image.filename} with sizes {width}x{height}"
            )
def crop_center(image, target_width, target_height):
    width, height = image.size
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    rigth = (width + target_width) // 2
    bottom = (height + target_height) // 2
    
    return image.crop((left, top, rigth, bottom))

def crop_all(input_directory, output_directory, target_width, target_height, format="png"):
    if not os.path.exists(input_directory):
        print("Input directory doesn't exist!")
        return
    image_list = loader.load_images_as_list(input_directory)
    try:
        check_sizes(image_list, target_width, target_height)
        print("cropping...")
        for image in image_list:
            filename = Path(image.filename).stem or "unknown"
            image = crop_center(image, target_width, target_height)
            image.save(os.path.join(output_directory, f"{filename}.{format}"), format=format)
    except Exception as e:
        print(e)
    print("Cropping completed.")

def high_pass_filter_conv2D(image, kernel=HIGH_PASS_KERNEL):
    kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
    if not isinstance(image, (list, np.ndarray, pd.Series)):
        image = np.array(image)
    image = np.array(image)
    # Filter the source image
    image_flt = cv2.filter2D(image, -1, kernel)

    return image_flt

def filter_image_dir(input_filename, output_directory=None):
    image = Image.open(input_filename)
    filename = Path(image.filename).stem or "unknown"
    image_flt = high_pass_filter_conv2D(image) # Is an array-like
    if not output_directory:
        return Image.fromarray(image_flt)
    else:
        cv2.imwrite((os.path.join(output_directory, f"{filename}.png")), image_flt)
        return Image.fromarray(image_flt)

def filter_images(input_directory, output_directory):
    image_list = loader.load_images_as_list(input_directory)

    print(f"Filtering {len(image_list)} images...")
    filtered_images = []
    for i, image in enumerate(image_list):
        filename = Path(image.filename).stem or "unknown"
        image_flt = high_pass_filter_conv2D(image) # An array-like
        filtered_images.append(Image.fromarray(image_flt))
        cv2.imwrite((os.path.join(output_directory, f"{filename}.png")), image_flt)
        print(f"{filename}.png {i+1}/{len(image_list)}")
    print("Filtering completed.")
    return filter_images

def noise_residual(image, filter_fnc):
    if not isinstance(image, (list, np.ndarray, pd.Series)):
        image = np.array(image)
    image_flt = filter_fnc(image)

    residual = np.subtract(image, image_flt)
    return residual

def esitmated_fingerprint(input_directory):
    image_list = loader.load_images_as_list(input_directory)
    N = len(image_list)
    sum = np.zeros(np.array(image_list[0]).shape)
    for image in image_list:
        noise_res = noise_residual(image, high_pass_filter_conv2D)
        sum += noise_res
    
    return (sum / N)