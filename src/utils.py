import os
import sys
import tensorflow as tf

# Find the correct resolution for a list of images path
def find_resolution(images_batch):
    resolutions = []
    # Calcola le risoluzioni di tutte le immagini
    for image_path in images_batch:
        image = tf.image.decode_image(tf.io.read_file(image_path))
        resolutions.append(tf.shape(image)[:-1])
    average_resolution = tf.reduce_min(tf.stack(resolutions, axis=0), axis=0)
    return average_resolution

# Ridimensiona tutte le immagini alla risoluzione specificata
def load_and_process_image(images_path, output_path, resolution):
    print("Resizing images...")
    for image_file in images_path:
        image = tf.image.decode_image(tf.io.read_file(image_file), channels=3)
        resized_image = tf.image.resize(image, resolution)
        np_resized_image = resized_image.numpy()
        image_name = os.path.splitext(os.path.split(image_file)[-1])[0]
        tf.keras.utils.save_img(os.path.join(output_path, image_name + ".png"), np_resized_image, file_format="png")
    return
