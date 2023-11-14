import argparse
import os

from src import preprocess
from src.tensor_extraction import dump_tensor, dump_tensor_all
from src import utils

MODELS_LIST = [
    "hific-lo", "hific-mi", "hific-hi",
    "ms2020-cc10-mse-[1-10]", "ms2020-cc8-msssim-[1-9]",
    "mbt2018-mean-mse-[1-8]", "mbt2018-mean-msssim-[1-8]",
    "bmshj2018-factorized-mse-[1-8]", "bmshj2018-factorized-msssim-[1-8]", "bmshj2018-hyperprior-mse-[1-8]", "bmshj2018-hyperprior-msssim-[1-8]",
    "b2018-leaky_relu-128-[1-4]", "b2018-leaky_relu-192-[1-4]", "b2018-gdn-128-[1-4]", "b2018-gdn-192-[1-4]"
]

TENSOR_NAMES = ["hyperprior/entropy_model/conditional_entropy_model_3/add:0"]
MODEL_NAME = "hific-lo"
def main(args):
    if args.command == "crop":
        target_width, target_height = args.size
        preprocess.crop_all(args.input_directory, args.output_directory, target_width, target_height)
    if args.command == "dump":
        if args.all_images:
            dump_tensor_all(args.input_directory, args.output_directory, args.model, args.tensors)
        else:
            dump_tensor(args.input_directory, args.output_directory, args.model, args.tensors)

def parse_args():
    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(
        title="commands",
        dest="command",
        help="invoke <command> -h for more information"
    )

    # 'dump' subcomand
    dump_cmd = subparser.add_parser(
        "dump",
        description="dump a tensor given an input image. If --all-images flag is enabled, all images in input_directory will be dumped"
    )
    dump_cmd.add_argument(
        "-a", "--all-images",
        action="store_true",
        help="If active dumps all image tensors"
    )
    dump_cmd.add_argument(
        "model",
        choices=MODELS_LIST,
        help="Specify the name of the model"
    )
    dump_cmd.add_argument(
        "tensors",
        nargs="?",
        help="The name of the tensor to extract"
    )

    # 'crop' subcommand.
    crop_cmd = subparser.add_parser(
        "crop",
        description="Crop all the images in input_directory"
    )
    crop_cmd.add_argument(
        "size",
        nargs=2,
        type=int,
        help="Target size in the form (target_width, target_height)"
    )
    

    for cmd in (dump_cmd, crop_cmd):
        cmd.add_argument(
            "input_directory",
            help="input directory"
        )
        cmd.add_argument(
            "output_directory",
            help="output directory"
        )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

