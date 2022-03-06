import os, os.path
import numpy as np
import pandas as pd
import data_structures as ds
import setup_analysis as sa
import support_functions as sf
import sector_models as sm
import argparse



def parse_arguments() -> dict:

    parser = argparse.ArgumentParser(description = "Run SISEPUEDE models from the command line.")
    parser.add_argument("--input", type = str,
                        help = f"Path to an input CSV, long by {sa.model_attributes.dim_time_period}, that contains required input variables.")
    parser.add_argument("--output", type = str,
                        help="Path to output csv file", default = sa.fp_csv_default_single_run_out)
    parser.add_argument(
        "--models",
        type = str,
        help = "Models to run using the input file. Possible values include 'AFOLU', 'CIRCECON', 'ENERGY', 'INDUSTRY'",
        default = "AFOLU",
    )
    parsed_args = parser.parse_args()

    # Since defaults are env vars, still need to checking to make sure its passed
    errors = []
    if parsed_args.input is None:
        errors.append("Missing --input DATA INPUT FILE")
    if errors:
        raise ValueError(f"Missing arguments detected: {sf.format_print_list(errors)}")

    # json args over-write specified args
    parsed_args_as_dict = vars(parsed_args)

    return parsed_args_as_dict


def main(args: dict) -> None:

    print("\n***\n***\n*** Welcome to SISEPUEDE! Hola Edmundo y equipo ITEM—esta mensaje va a cambiar en el futuro, but have fun seeing this message *every time*.\n***\n***\n")

    fp_in = args.get("input")
    fp_out = args.get("output")
    models_run = args.get("models")

    # load data
    if not fp_in:
        raise ValueError("Cannot run: no input data file was specified.")
    else:
        if os.path.exists(args["input"]):
            print(f"Reading input data from {fp_in}...")
            df_input_data = pd.read_csv(fp_in)
            print("Done.")
        else:
            raise ValueError(f"Input file '{fp_in}' not found.")

    # notify of output path
    print(f"\n\n*** STARTING MODELS ***\n\nOutput file will be written to {fp_out}.\n")

    # set up models
    if "AFOLU" in models_run:
        print("\nRunning AFOLU...")
        model_afolu = sm.AFOLU(sa.model_attributes)
        df_output_data = model_afolu.project(df_input_data)

    ## other models
    print("\n*** MODEL RUNS COMPLETE ***\n")

    # write output
    print(f"\nWriting data to {fp_out}...")
    df_output_data.to_csv(fp_out, index = None, encoding = "UTF-8")
    print("\n*** MODEL RUNS SUCCESSFULLY COMPLETED. Q les vayan bien damas y caballeros.")


if __name__ == "__main__":

    args = parse_arguments()

    main(args)
