import os
import sys
import argparse
import tensorflow as tf

sys.path.append(os.path.join("/home/lorenzo-sibi/Scrivania/research_project"))
from src import loader, visualize_data, utils

def parse_args():
    parser = argparse.ArgumentParser("Testing functionalities...", )
    parser.add_argument("-l", "--list", nargs="+", required=True, help="Any args")

    return parser.parse_args()

def main(args):
    input_directory, input_directory_cmp, dataset_name, dataset_name_cmp = args.list # input_path, batch_size

    tensors_c_list = loader.load_tensors_as_list(input_directory)
    tensors_c_list_cmp = loader.load_tensors_as_list(input_directory_cmp)

    if(len(tensors_c_list) != len(tensors_c_list_cmp)):
        print('Error: Different number of images in the two directories')
        return
    
    # for i in range(len(tensors_c_list)):
    #     t1 = tensors_c_list[i]
    #     name = t1.get_name()
    #     t2 = list(filter(lambda tc: tc.get_name() == name, tensors_c_list_cmp))[0]
    #     visualize_data.plot_tensors_subtraction(t1, t2, title=f"{t1.get_name()} subtraction", save=False)
    tensors = []
    tensors_cmp =[]
    for i in range(len(tensors_c_list)):
        t1 = tensors_c_list[i]
        name = t1.get_name()
        t2 = list(filter(lambda tc: tc.get_name() == name, tensors_c_list_cmp))[0]
        normalized_tensor1, _ = tf.linalg.normalize(t1.get_tensor(), axis=(0,1))
        normalized_tensor2, _ = tf.linalg.normalize(t2.get_tensor(), axis=(0,1))

        tensors.append(t1.get_tensor())
        tensors_cmp.append(t2.get_tensor())

    for i in range(3):
        visualize_data.statistics_axis(tensors, axis=i, title="statistic_indexes_" + dataset_name, bins=100)
        visualize_data.statistics_axis(tensors_cmp, axis=i, title="statistic_indexes_" + dataset_name_cmp, bins=100)

    for i in range(3):
        visualize_data.statistics_axis(tensors, axis=i, title="statistic_indexes_" + dataset_name, bins=100)
        visualize_data.statistics_axis(tensors_cmp, axis=i, title="statistic_indexes_" + dataset_name_cmp, bins=100)
if __name__ == "__main__":
    args = parse_args()
    main(args)