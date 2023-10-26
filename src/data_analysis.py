import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

import loader

# compute mean, std deviation, min and max value on a single tensor
def compute_analysis(tensor, show_plot=False):
    mean_value = tf.math.reduce_mean(tensor)
    std_deviation = tf.math.reduce_std(tensor)
    variance = tf.math.reduce_variance(tensor)
    min_value = tf.math.reduce_min(tensor)
    max_value = tf.math.reduce_max(tensor)
    # if show_plot:
    #     # Istogramma della distribuzione dei tensori
    #     plt.hist(tensor, bins=50, color='blue', alpha=0.7)
    #     plt.xlabel('Valore')
    #     plt.ylabel('Frequenza')
    #     plt.title('Istogramma della distribuzione dei tensori')
    #     plt.show()
    return {
        "mean_value": mean_value.numpy(),
        "std_deviation": std_deviation.numpy(),
        "variance" : variance.numpy(),
        "min_value": min_value.numpy(),
        "max_value": max_value.numpy(),
    }

def main():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='Compute data analysis on a tensors batch (file must end with .npz)',
        epilog='')
    
    parser.add_argument("input_path", help="The inputh path where are loceted all the tnesors")
    parser.add_argument("output_path", default="./")
    parser.add_argument("-n", "--n", default=10, type=int ,help="Tensors batch size (default: 10)")

    args = parser.parse_args()

    np_tensor_list = loader.load_from_directory(args.input_path, args.n)
    tf_tensor_list = loader.convert_to_tf(np_tensor_list)

    test_tensor = tf_tensor_list[1]
    tt_size = tf.size(test_tensor).numpy()

    print(test_tensor.shape)

    desired_shape = (tf.reduce_prod(test_tensor[0, 0]) // 3, 3)
    # x = int(np.sqrt(tt_size))
    # y = tt_size // x

    image = tf.reshape(test_tensor[0, 0], desired_shape)
    print(image.shape)
    plt.imshow(image)
    plt.gray()
    plt.savefig(os.path.join(args.output_path, "plot_test2.png"))
    print(compute_analysis(test_tensor, show_plot=True))
    return


if __name__ == "__main__":
    main()