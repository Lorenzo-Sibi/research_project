import os
import random
from PIL import Image
from pathlib import Path

from src import *

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
    image_list = load_images_as_list(input_directory)
    try:
        check_sizes(image_list, target_width, target_height)
        for image in image_list:
            filename = Path(image.filename).stem or "unknown"
            image = crop_center(image, target_width, target_height)
            image.save(os.path.join(output_directory, f"{filename}.{format}"), format=format)
    except Exception as e:
        print(e)
def load_images_as_list(input_path, batch_size=0):
    if not os.path.exists(input_path):
        print("Path not avaiable or doesn't exist")
        return
    # List of all images inside input_path directory (COMPLETE PATH)
    print("Loading images...")
    images_set = [os.path.join(input_path, file) 
                  for file in os.listdir(input_path) 
                  if file.endswith((".jpg", ".png"))]
    n_images = batch_size
    if n_images == 0 or n_images is None:
        images_batch = images_set
        n_images = len(images_set)
    else:
        images_batch = random.sample(images_set, n_images)
    image_list = []
    for path in images_batch:
        if path.endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(path)
            image_list.append(image)
    print("Loading completed.")
    return image_list
