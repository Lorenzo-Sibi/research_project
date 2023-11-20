import argparse
import os

from src import visualize_data

PLOT_TYPE_LIST = ["latent_representation"]

def main(args):
    if args.command == "plot":
        if args.type == "latent_representation":
            visualize_data.plot_latent_representation_all(args.input_directory, args.output_directory)
        if args.type == "statistics":
            pass

def parse_args():
    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(
        title="commands",
        dest="command",
        help="invoke <command> -h for more information"
    )

    # 'plot' subcomand
    plot_cmd = subparser.add_parser(
        "plot",
        description=""
    )
    plot_cmd.add_argument(
        "type",
        choices=PLOT_TYPE_LIST,
        help="Specify which plot compute"
    )

    plot_cmd.add_argument(
        "input_directory",
        help="input directory"
    )
    plot_cmd.add_argument(
        "output_directory",
        help="output directory"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

