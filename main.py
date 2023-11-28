import argparse
import os
import sys

from src import preprocess
from src import tensor_extraction

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "compression-master/models"))
import tfci


MODELS_DICT = MODELS_DICT = {
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
TENSORS_DICT = {
    "hific": "hyperprior/entropy_model/conditional_entropy_model_3/add:0",
    "mbt2018": "analysis/layer_3/convolution:0",
    "bmshj2018":"analysis/layer_2/convolution:0",
    "b2018": "analysis/layer_2/convolution:0",
    #"ms2020": "analysis/layer_2/convolution:0",
}

def list_tensors(model):
  """Lists all internal tensors of a given model."""
  def get_names_dtypes_shapes(function):
    for op in function.graph.get_operations():
      for tensor in op.outputs:
        yield tensor.name, tensor.dtype.name, tensor.shape

  sender = tfci.instantiate_model_signature(model, "sender")
  tensors = sorted(get_names_dtypes_shapes(sender))
  log = "Sender-side tensors:\n"
  for name, dtype, shape in tensors:
    log += f"{name} (dtype={dtype}, shape={shape})\n"
  log += "\n"

  receiver = tfci.instantiate_model_signature(model, "receiver")
  tensors = sorted(get_names_dtypes_shapes(receiver))
  log += "Receiver-side tensors:\n"
  for name, dtype, shape in tensors:
    log += f"{name} (dtype={dtype}, shape={shape})\n"
  return log

def tensors_log(logdir='tensors_logs'):
    for i, model_class in enumerate(MODELS_DICT):
        for variant in MODELS_DICT[model_class]:
            for model in MODELS_DICT[model_class][variant]:
                path = os.path.join(logdir, model_class, variant)
                filename = os.path.join(path, model + "_tensors" + ".txt")
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(filename, "w") as f:
                   f.write(list_tensors(model))
        print(f"{i}/{len(MODELS_DICT)}")
                

def main(args):
    
    if args.command == "compress-all":
        tensor_extraction.compress_all(
            args.input_directory,
            args.output_directory,
            MODELS_DICT,
            one_image=args.image
        )

    elif args.command == "compress":
        if args.one_image:
            tensor_extraction.compress(
                args.model, 
                args.input_directory, 
                args.output_directory, 
                args.rd_parameter
            )
        else:
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
            target_height)
    
    elif args.command == "filter":
        if args.one_image:
            preprocess.filter_image_dir(args.input_directory, args.output_directory)
        else:
            preprocess.filter_images(args.input_directory, args.output_directory)
    
    elif args.command == "tensors":
        tensor_extraction.list_tensors(args.model)

    elif args.command == "tensors-all":
        tensors_log()

    elif args.command == "dump-all":
        one_image = args.image
        tensor_extraction.dump_tensor_all(
            args.input_directory,
            args.output_directory,
            MODELS_DICT,
            one_image
        )

    elif args.command == "dump":
        if args.image:
            tensor_extraction.dump_tensor(
                args.input_directory, 
                args.output_directory, 
                args.model, 
                args.tensor[0],
            )
        else:
            tensor_extraction.dump_tensor_all_images(
                args.input_directory, 
                args.output_directory, 
                args.model, 
                args.tensor[0]
            )

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

    compress_all_cmd.add_argument(
        "-i", "--image",
        action="store_true",
        help="If active compress a single image with all models."
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

    # 'dump-all' subcommand.
    dump_all_cmd = subparser.add_parser(
        "dump-all",
    )

    dump_all_cmd.add_argument(
        "-i", "--image",
        action="store_true",
        help="If active dumps a single image tensors for all models."
    )


    # 'dump' subcomand
    dump_cmd = subparser.add_parser(
        "dump",
        description="dump a tensor given an input image or a directory for a specific model and tensor. If --image flag is enabled, only the given image will be dumped"
    )
    dump_cmd.add_argument(
        "-i", "--image",
        action="store_true",
        help="If active dumps a single image tensors"
    )

    dump_cmd.add_argument(
        "model",
        help="Specify the name of the model"
    )
    dump_cmd.add_argument(
        "-t", "--tensor",
        nargs=1,
        required=True,
        help="The name of the tensor to extract"
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

    filter_cmd.add_argument(
       "-o", "--one_image",
       required=False,
       action='store_true',
       help="Flag for filtering only an image"
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

