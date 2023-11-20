import os
import random
from PIL import Image
from pathlib import Path

from src import *
from src import loader

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
