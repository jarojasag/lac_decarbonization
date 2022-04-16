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
    parser.add_argument(
        "--input",
        type = str,
        help = f"Path to an input CSV, long by {sa.model_attributes.dim_time_period}, that contains required input variables."
    )
    parser.add_argument(
        "--output",
        type = str,
        help = "Path to output csv file",
        default = sa.fp_csv_default_single_run_out
    )
    parser.add_argument(
        "--models",
        type = str,
        help = "Models to run using the input file. Possible values include 'AFOLU', 'CircularEconomy', 'Energy', 'IPPU'",
        default = "AFOLU"
    )
    parser.add_argument(
        "--integrated",
        help = "Include this flag to run included models as integrated sectors. Output from upstream models will be passed as inputs to downstream models.",
        action = "store_true"
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

    init_merge_q = True
    run_integrated_q = bool(args.get("integrated"))
    df_output_data = []

    # run AFOLU and collect output
    if "AFOLU" in models_run:
        print("\n\tRunning AFOLU")
        # get the model, run it using the input data, then update the output data (for integration)
        model_afolu = sm.AFOLU(sa.model_attributes)
        df_output_data.append(model_afolu.project(df_input_data))

    # run CircularEconomy and collect output
    if "CircularEconomy" in models_run:
        print("\n\tRunning CircularEconomy")
        model_circecon = sm.CircularEconomy(sa.model_attributes)
        # integrate AFOLU output?
        if run_integrated_q and set(["AFOLU"]).issubset(set(models_run)):
            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data[0],
                model_circecon.integration_variables
            )
        df_output_data.append(model_circecon.project(df_input_data))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] if run_integrated_q else df_output_data


    # run IPPU and collect output
    if "IPPU" in models_run:
        print("\n\tRunning IPPU")
        model_ippu = sm.IPPU(sa.model_attributes)
        # integrate Circular Economy output?
        if run_integrated_q and set(["CircularEconomy"]).issubset(set(models_run)):
            df_input_data = sa.model_attributes.transfer_df_variables(
                df_input_data,
                df_output_data[0],
                model_ippu.integration_variables
            )
        df_output_data.append(model_ippu.project(df_input_data))
        df_output_data = [sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")] if run_integrated_q else df_output_data


    # run Energy and collect output
    if "Energy" in models_run:
        print("\n\t*** NOTE: Energy INCOMPLETE. IT WILL NOT BE RUN")


    # build output data frame
    df_output_data = sf.merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")


    print("\n*** MODEL RUNS COMPLETE ***\n")

    # write output
    print(f"\nWriting data to {fp_out}...")
    df_output_data.to_csv(fp_out, index = None, encoding = "UTF-8")
    print("\n*** MODEL RUNS SUCCESSFULLY COMPLETED. Q les vayan bien damas y caballeros.")


if __name__ == "__main__":

    args = parse_arguments()

    main(args)
