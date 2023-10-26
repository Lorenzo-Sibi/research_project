import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import tensor_extraction
import loader
import utils

def parse_args():
    parser = argparse.ArgumentParser("Testing functionalities...", )
    parser.add_argument("-l", "--list", nargs="+", required=True, help="Any args")

    return parser.parse_args()

def main(args):
    args_list = args.list # input_path, batch_size
    print(args_list)
    
    np_tensors = loader.load_from_directory(args_list[0], args_list[1])

    for np_tensor in np_tensors:
        print(np_tensor.shape())


if __name__ == "__main__":
    args = parse_args()
    main(args)