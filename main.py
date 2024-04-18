import argparse
import os
import sys

from pathlib import Path
from src import preprocess
from src import tensor_extraction
from src.utils import tensors_log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = info, 1 = warning, 2 = error, 3 = fatal

sys.path.append(os.path.join(os.path.dirname(__file__), "compression-master/models"))
import tfci
from src import MODELS_LATENTS_DICT

# 67 models in total
MODELS_DICT = {
    "hific": {
        "variants": ["hific-lo", "hific-mi", "hific-hi"],
    },
    "bmshj2018": {
        "factorized-mse": [f"bmshj2018-factorized-mse-{i}" for i in range(1, 9)],
        "factorized-msssim": [f"bmshj2018-factorized-msssim-{i}" for i in range(1, 9)],
        "hyperprior-mse": [f"bmshj2018-hyperprior-mse-{i}" for i in range(1, 9)],
        "hyperprior-msssim": [f"bmshj2018-hyperprior-msssim-{i}" for i in range(1, 9)]
    },
    "b2018": {
        "leaky_relu-128": [f"b2018-leaky_relu-128-{i}" for i in range(1, 5)],
        "leaky_relu-192": [f"b2018-leaky_relu-192-{i}" for i in range(1, 5)],
        "gdn-128": [f"b2018-gdn-128-{i}" for i in range(1, 5)],
        "gdn-192": [f"b2018-gdn-192-{i}" for i in range(1, 5)]
    },
    "mbt2018": {
        "mean": [f"mbt2018-mean-mse-{i}" for i in range(1, 9)],
        "mean-msssim": [f"mbt2018-mean-msssim-{i}" for i in range(1, 9)]
    },
    # "ms2020": {
    #     "cc10": [f"ms2020-cc10-mse-{i}" for i in range(1, 11)],
    #     "cc8": [f"ms2020-cc8-msssim-{i}" for i in range(1, 10)]
    # },
}

MODELS_DICT_MOMENTANEO = {
    "hific": {
        "variants": ["hific-lo", "hific-hi"],
    },
    "bmshj2018": {
        "factorized-mse": ["bmshj2018-factorized-mse-2", "bmshj2018-factorized-mse-8"],
        "factorized-msssim": ["bmshj2018-factorized-msssim-2", "bmshj2018-factorized-msssim-8"],
        "hyperprior-mse": ["bmshj2018-hyperprior-mse-2", "bmshj2018-hyperprior-mse-8"],
        "hyperprior-msssim": ["bmshj2018-hyperprior-msssim-2", "bmshj2018-hyperprior-msssim-8"]
    },
    "b2018": {
        "leaky_relu-128": ["b2018-leaky_relu-128-2", "b2018-leaky_relu-128-4"],
        "leaky_relu-192": ["b2018-leaky_relu-192-2", "b2018-leaky_relu-192-4"],
        "gdn-128": ["b2018-gdn-128-2", "b2018-gdn-128-4"],
        "gdn-192": ["b2018-gdn-192-2", "b2018-gdn-192-4"]
    },
    "mbt2018": {
        "mean": ["mbt2018-mean-mse-2", "mbt2018-mean-mse-8"],
        "mean-msssim": ["mbt2018-mean-msssim-2", "mbt2018-mean-msssim-8"]
    },
}

TENSORS_DICT = {
    "hific-lo": "hyperprior/entropy_model/conditional_entropy_model_3/add:0",
    "hific-hi": "hyperprior/entropy_model/conditional_entropy_model_3/add:0",
    "hific-lo": "hyperprior/entropy_model/conditional_entropy_model_3/add:0",
    
    "mbt2018": "analysis/layer_3/convolution:0",
    "bmshj2018":"analysis/layer_2/convolution:0",
    "b2018": "analysis/layer_2/convolution:0",
    #"ms2020": "analysis/layer_2/convolution:0",
}

def main(args):
    
    if args.command == "compress-all":
        tensor_extraction.compress_all(
            args.input_directory,
            args.output_directory,
            MODELS_DICT_MOMENTANEO,
        )

    elif args.command == "compress":
        tensor_extraction.compress_images(
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
            target_height, 
            check_size=args.check_size
        )
    
    elif args.command == "filter":
        input_path = Path(args.input_directory)
        output_path = Path(args.output_directory)
        preprocess.filter_images(input_path, output_path)
    
    elif args.command == "tensors":
        tensor_extraction.list_tensors(args.model)

    elif args.command == "tensors-all":
        tensors_log()

    elif args.command == "dump-all":
        tensor_extraction.dump_tensor_all(
            args.input_directory,
            args.output_directory,
            MODELS_DICT_MOMENTANEO
        )

    elif args.command == "dump":
        input_path = Path(args.input_directory)
        output_path = Path(args.output_directory)
        tensor_extraction.dump(input_path, output_path, args.model)

def parse_args():
    parser = argparse.ArgumentParser()
    
    subparser = parser.add_subparsers(
        title="commands",
        dest="command",
        help="invoke <command> -h for more information"
    )

    # 'compress-all' subcomand
    compress_all_cmd = subparser.add_parser(
        "compress-all",
        description="Compress all the images in 'input_directory' (or just one if -i flag is active) for all models available"
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

    # 'dump-all' subcommand.
    dump_all_cmd = subparser.add_parser(
        "dump-all",
    )


    # 'dump' subcomand
    dump_cmd = subparser.add_parser(
        "dump",
        description="dump the latent space tensor given an input image or a directory for a specific model and tensor."
    )

    dump_cmd.add_argument(
        "model",
        help="Specify the name of the model",
        choices=MODELS_LATENTS_DICT.keys()
    )

    # 'tensors-all' subcommand.
    tensors_cmd = subparser.add_parser(
        "tensors-all",
        description="Lists names of internal tensors of a given model."
    )

    # 'tensors' subcommand.
    tensors_cmd = subparser.add_parser(
        "tensors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Lists names of internal tensors of a given model."
    )
    
    tensors_cmd.add_argument(
        "model",
        help="Unique model identifier. See 'models' command for options.")

    # 'filter' subcommand.
    filter_cmd = subparser.add_parser(
       "filter",
        help="filter an image using an high-pass filter"
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
        help="Target size in the form <TARGET_WIDTH TARGET_HEIGHT> (no comas, brackets or quotes)"
    )
    
    crop_cmd.add_argument(
        "--check_size",
        type=bool,
        default=False
    )
    

    for cmd in (compress_all_cmd, compress_cmd, dump_all_cmd, dump_cmd, crop_cmd, filter_cmd):
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

