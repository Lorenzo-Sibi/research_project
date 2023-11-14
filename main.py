import argparse
import os

from src import preprocess
from src import utils

def main(args):
    if args.command == "crop":
        target_width, target_height = list(map(int, args.size))
        preprocess.crop_all(args.input_directory, args.output_directory, target_width, target_height)

def parse_args():
    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(
        title="commands",
        dest="command",
        help="invoke <command> -h for more information"
    )
    # 'crop' subcommand.
    crop_cmd = subparser.add_parser(
        "crop",
        description="Crop all the images in input_directory"
    )
    crop_cmd.add_argument(
        "size",
        nargs=2,
        help="Target size in the form (target_width, target_height)"
    )
    crop_cmd.add_argument(
        "input_directory",
        help="input directory"
    )
    crop_cmd.add_argument(
        "output_directory",
        help="output directory"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

