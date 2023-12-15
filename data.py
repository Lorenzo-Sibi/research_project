import argparse
import os
from pathlib import Path
from main import MODELS_DICT
from src import visualize_data, preprocess

PLOT_ALL_TYPE_LIST = ["statistics", "spectrum", "average-spectrum"]
PLOT_TYPE_LIST = ["latent_representation", "spectrum", "average-spectrum"]

"""
Per utilizzare il comando statistics_all le cartelle devono essere strutturate in questo modo:

main-directory
    |- model_class (es. b2028)
        |- variant1/ (es. gdn-128) 
            ...
        |- variant2/
            |- model1/
            |- model2/
                | ... files.nps ...

NON inserire la cartella di output all'interno della input_directory
"""
# TO DO: refactor this methid and insert inside appropriate module
def spectrum_all(input_directory, output_directory):
    
    # Itera sulle sottocartelle di main-directory
    for model_class in input_directory.iterdir():
        if model_class.is_dir():
            model_class_name = model_class.name

            # Crea la cartella principale per la classe del modello
            class_output_path = Path(output_directory, model_class_name)
            class_output_path.mkdir(parents=True, exist_ok=True)
            
            for variant in model_class.iterdir():
                if variant.is_dir():
                    variant_name = variant.name

                    # Crea la cartella per la variante
                    variant_output_path = class_output_path / variant_name
                    variant_output_path.mkdir(parents=True, exist_ok=True)

                    for model_folder in variant.iterdir():
                        if model_folder.is_dir():
                            model_name = model_folder.name
                            print(f"Calculating /{model_name} spectrum...")

                            # Calcolo delle statistiche e salvataggio nella sottocartella
                            title = f"{model_name}_spectrum"
                            # fingerprint = preprocess.esitmated_fingerprint(str(model_folder))
                            # visualize_data.plot_tensor_fft_spectrum(fingerprint, save_in=str(variant_output_path), name=model_name)
                            for input_filename in model_folder.iterdir():
                                if input_filename.suffix in (".png", ".jpeg", ".jpg"):
                                    visualize_data.plot_single_fft_spectrum(input_filename, variant_output_path)
                                    break
                        else:
                            print("Folder structure not respected!")
                            return
    print("Processing completed.")
    return
    

def main(args):
    
    input_directory = Path(args.input_directory)
    output_directory = Path(args.output_directory)
    
    if args.command == "plot-all":
        if args.type == "statistics":
            visualize_data.statistics_all(input_directory, output_directory, 2)
        elif args.type == "spectrum":
            spectrum_all(input_directory, output_directory)
        elif args.type == "average-spectrum":
            visualize_data.plot_all_average_spectrum(input_directory, output_directory)
            
    elif args.command == "plot":
        if args.type == "spectrum":
            visualize_data.plot_spectrum(input_directory, output_directory)
        elif args.type == "average-spectrum":
            visualize_data.plot_average_spectrum(input_directory, output_directory, title=args.title)
        elif args.type == "latent_representation":
            visualize_data.plot_latent_representation_all(input_directory, output_directory)
        elif args.type == "statistics":
            pass

def parse_args():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(
        title="commands",
        dest="command",
        help="invoke <command> -h for more information"
    )

    # 'plot-all' subcomand
    plot_all_cmd = subparser.add_parser(
        "plot-all",
        description="execute specified plot type for all models avaiable",
    )
    
    plot_all_cmd.add_argument(
        "type",
        choices=PLOT_ALL_TYPE_LIST,
        help="Specify which plot compute"
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
        "-t", "--title",
        action='store',
        default=None,
        help="Specify the plot figure's title."
    )

    for cmd in (plot_all_cmd, plot_cmd):
        cmd.add_argument(
        "input_directory",
        help="input directory")
        
        cmd.add_argument(
            "output_directory",
            help="output directory")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

