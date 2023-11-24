import argparse
import os

from src import preprocess
from src import tensor_extraction
from src import utils

MODELS_DICT = {
    "hific": ["hific-lo", "hific-mi", "hific-hi"],
    "ms2020": ["ms2020-cc10-mse-[1-10]", "ms2020-cc8-msssim-[1-9]"],
    "mbt218": ["mbt2018-mean-mse-[1-8]", "mbt2018-mean-msssim-[1-8]"],
    "bmshj2018": ["bmshj2018-factorized-mse-[1-8]", "bmshj2018-factorized-msssim-[1-8]", "bmshj2018-hyperprior-mse-[1-8]", "bmshj2018-hyperprior-msssim-[1-8]"],
    "b2018": ["b2018-leaky_relu-128-[1-4]", "b2018-leaky_relu-192-[1-4]", "b2018-gdn-128-[1-4]", "b2018-gdn-192-[1-4]"]
}

MODELS_LIST = sum(MODELS_DICT.values(), [])

TENSOR_NAMES = ["hyperprior/entropy_model/conditional_entropy_model_3/add:0"]
MODEL_NAME = "hific-lo"
def main(args):
    if args.command == "compress":
        if args.one_image:
            tensor_extraction.compress(
                args.model, 
                args.input_directory, 
                args.output_directory, 
                args.rd_parameter
            )
        else:
            tensor_extraction.compress_all(
                args.model, 
                args.input_directory,
                args.output_directory, 
                args.rd_parameter
            )
    
    elif args.command == "crop":
        target_width, target_height = args.size
        preprocess.crop_all(
            args.input_directory, 
            args.output_directory, 
            target_width, 
            target_height)
    
    elif args.command == "tensors":
        tensor_extraction.list_tensors(args.model)

    elif args.command == "dump":
        if args.all_images:
            tensor_extraction.dump_tensor_all(
                args.input_directory, 
                args.output_directory, 
                args.model, 
                args.tensors
            )
        else:
            tensor_extraction.dump_tensor(
                args.input_directory, 
                args.output_directory, 
                args.model, 
                args.tensors
            )

def parse_args():
    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(
        title="commands",
        dest="command",
        help="invoke <command> -h for more information"
    )

    # 'compress' subcomand
    compress_cmd = subparser.add_parser(
        "compress",
        description="Comprime le immagini fornite nella cartella input_directory restituiendole in formao .png in output_directory"
    )

    compress_cmd.add_argument(
        "model",
        help="Specify the name of the model"
    )

    compress_cmd.add_argument(
        "--rd_parameter", "-r", 
        type=float,
        help="Rate-distortion parameter (for some models). Ignored if 'target_bpp' is set."
    )

    compress_cmd.add_argument(
        "-o", "--one_image",
        required=False,
        action='store_true',
        help="Flag for compressing only an image"
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
        "-t", "--tensors",
        nargs="+",
        required=True,
        help="The name of the tensor to extract"
    )

    # 'tensors' subcommand.
    tensors_cmd = subparser.add_parser(
      "tensors",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Lists names of internal tensors of a given model.")
    
    tensors_cmd.add_argument(
        "model",
        choices=MODELS_LIST,
        help="Unique model identifier. See 'models' command for options.")

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
    

    for cmd in (compress_cmd, dump_cmd, crop_cmd):
        cmd.add_argument(
            "input_directory",
            help="input directory"
        )
        cmd.add_argument(
            "output_directory",
            help="output directory"
        )
    print(parser.parse_args())
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

