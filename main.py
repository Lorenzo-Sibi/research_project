import argparse
import os
import sys

from src import preprocess
from src import tensor_extraction

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "compression-master/models"))
import tfci


MODELS_DICT = MODELS_DICT = {
    "hific": {
        "variants": ["hific-lo", "hific-mi", "hific-hi"]
    },
    "ms2020": {
        "cc10": [f"ms2020-cc10-mse-{i}" for i in range(1, 11)],
        "cc8": [f"ms2020-cc8-msssim-{i}" for i in range(1, 10)]
    },
    "mbt218": {
        "mean": [f"mbt2018-mean-mse-{i}" for i in range(1, 9)],
        "mean-msssim": [f"mbt2018-mean-msssim-{i}" for i in range(1, 9)]
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
    }
}

TENSOR_NAMES = ["hyperprior/entropy_model/conditional_entropy_model_3/add:0"]
MODEL_NAME = "hific-lo"

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

    elif args.command == "tensors-all":
        tensors_log()

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
      "tensors-all",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Lists names of internal tensors of a given model.")

    # 'tensors' subcommand.
    tensors_cmd = subparser.add_parser(
      "tensors",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Lists names of internal tensors of a given model.")
    
    tensors_cmd.add_argument(
        "model",
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

