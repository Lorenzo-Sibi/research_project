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
    [-1.0, -1.0, -1.0],
    [-1.0, 8.0, -1.0],
    [-1.0, -1.0, -1.0]
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

def crop_all(input_directory, output_directory, target_width, target_height, check_size=False,format="png"):
    input_directory = Path(input_directory)
    if not input_directory.exists():
        print("Input directory doesn't exist!")
        return
    try:
        image_list = loader.load_images_as_list(input_directory)
        print(f"Total images loaded: {len(image_list)}")
        if check_size:
            check_sizes(image_list, target_width, target_height)
        print("cropping...")
        for image in image_list:
            print(image)
            w, h = image.size
            if w < target_width or  h < target_height:
                continue
            filename = Path(image.filename).stem or "unknown"
            image = crop_center(image, target_width, target_height)
            image.save(os.path.join(output_directory, f"{filename}.{format}"), format=format)
    except Exception as e:
        print(e)
    print("Cropping completed.")

def highpass(image_array, kernel=HIGH_PASS_KERNEL):
    """
    Applies a high-pass filter to an image array using the specified kernel.

    Args:
        image_array (array-like): The input image array to filter.
        kernel (array-like, optional): The kernel to use for the high-pass filter. Defaults to HIGH_PASS_KERNEL.

    Returns:
        array-like: The filtered image array.
    """
    kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
    image_flt = cv2.filter2D(image_array, -1, kernel)

    return image_flt

# def filter_image(input_filename, output_directory=None):
#     """
#     Filters an image and saves the filtered image to an output directory if provided.

#     Args:
#         input_filename (str): The path to the input image file.
#         output_directory (str, optional): The directory to save the filtered image. Defaults to None.

#     Returns:
#         Image: The filtered image as an Image object.
#     """
#     image = Image.open(input_filename)
#     filename = Path(image.filename).stem or "unknown"
#     image_flt = highpass(image) # Is an array-like
#     if output_directory:
#         cv2.imwrite((os.path.join(output_directory, f"{filename}.png")), image_flt)
#     return Image.fromarray(image_flt)

def filter_images(input_directory:Path, output_directory:Path):
    """
    Filters a list of images from the input directory and saves the filtered images to the output directory.

    Args:
        input_directory (Path obj): The directory containing the input images.
        output_directory (Path obj): The directory to save the filtered images.

    Returns:
        list: A list of filtered images as Image objects.
    """
    image_list = loader.load_images_as_list(input_directory)

    print(f"Filtering {len(image_list)} images...")
    filtered_images = []
    for i, image in enumerate(image_list):
        print(image)
        filename = Path(image.filename).stem or "unknown"
        image = np.array(image)
        image_flt = highpass(image) # An array-like
        filtered_images.append(Image.fromarray(image_flt))
        cv2.imwrite((os.path.join(output_directory, f"{filename}.png")), image_flt)
        print(f"{filename}.png {i+1}/{len(image_list)}")
    print("Filtering completed.")
    return filtered_images

def noise_residual(image, filter_fnc):
    if not isinstance(image, (list, np.ndarray, pd.Series)):
        image = np.array(image)
    image_flt = filter_fnc(image)

    residual = np.subtract(image, image_flt)
    return residual

def esitmated_fingerprint(input_directory):
    """
    Estimate fingerprint of a set of images using the Laplacian filter.
    
    This method is used to estimate the fingerprint F of same class images, using
    the same method used by:
    R. Corvi, D. Cozzolino, G. Zingarini, G. Poggi, K. Nagano, L. Verdoliva:
    "On the detection of Synthetic Images Generated by Diffusion Models".
    
    However, a Laplacian filter (3x3 kernel) is used as denoising filter.

    Args:
        input_directory (pathlib.Path): the director containing all the images

    Returns:
        estimated_fingerprint (ndarray)
    """
    image_list = loader.load_images_as_list(input_directory)
    N = len(image_list)
    sum = np.zeros(np.array(image_list[0]).shape)
    for image in image_list:
        noise_res = noise_residual(image, highpass)
        # Compute fft spectrum on noise_res
        fft_spectrum = compute_fft_spectrum(noise_res)
        fft_spectrum_sum += fft_spectrum
    
    esitmated_fingerprint = sum / N
    return esitmated_fingerprint

#################################################################

def fft2d(image_array):
    """_summary_

    Args:
        image_array (ndarray): The input image array. It should have 2 dimensions.

    Returns:
        ndarray: The complex-valued two-dimensional Fourier Transform of the input image array.
    """
    assert image_array.ndim == 2, f"Wrong number of dimensions: {image_array.ndim} instead of 2."
    
    image_array = np.fft.fft2(image_array)
    image_array = np.fft.fftshift(image_array)
    return image_array

def array_fft_spectrum(array, filter_fnc=None, epsilon=1e-12):
    """
    Description: 
        Compute the magnitude spectrum of a SINGLE array and return it as ndarray.
        The array spectrum is calculated as the average over the 3 channels RGB.
        If filter_fnc is None, no filter is applied to the array.
    Args:
        array (ndarray)
        filter_fnc (function)
        epsilon (float)
    Return:
        magnitude spectrum (ndarray)
    """
    array = array / 255.
    assert array.ndim == 3, f"Wrong number of dimensions: {array.ndim} instead of 3."

    for channel in range(array.shape[2]):
        
        img = array[:, :, channel]
        if filter_fnc:
            img = filter_fnc(img)
        
        fft_img = fft2d(img)
        fft_img = np.log(np.abs(fft_img) + epsilon)
        
        fft_min = np.min(fft_img)
        fft_max = np.max(fft_img)
        
        if (fft_max - fft_min) > 0:
            fft_img = np.array((fft_img - fft_min) / (fft_max))
        else:
            print("Unexpected behavior. Maximum value less than minimum value. Potential division by zero!")
            fft_img = np.array((fft_img - fft_min) / ((fft_max - fft_min) + np.finfo(float).eps))

        array[:, :, channel] = fft_img
    return np.average(array, axis=2) # in this method is obtained the specrum of the grayscale image

def average_fft(input_path):
    if input_path.is_file():
        raise ValueError(f"Input path {input_path} is a file, not a directory.")
    
    fft = []
    for filename in input_path.iterdir():
        if filename.is_dir():
            continue
        if filename.suffix in IMAGE_SUPPORTED_EXTENSIONS:
            image = loader.load_image(filename)
        elif filename.suffix in TENSOR_SUPPORTED_EXTENSIONS:
            image = loader.load_tensor(filename)
            image.squeeze()
            image = image.get_tensor()
            image = image * 255
            
        else:
            raise ValueError("Error. File extension not supported")
        
        image = np.array(image)
        fft_img_spectrum = array_fft_spectrum(image, highpass)
        fft.append(fft_img_spectrum)
        
    print(f"{len(fft)} images processed.")
    return np.average(np.array(fft), axis=0)