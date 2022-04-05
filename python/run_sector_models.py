import os, os.path
import numpy as np
import pandas as pd
import data_structures as ds
import setup_analysis as sa
import support_functions as sf
import sector_models as sm
import argparse


# use to merge data frames together into a single output
def merge_output_df_list(
    dfs_output_data: list,
    model_attributes,
    merge_type: str = "concatenate"
) -> pd.DataFrame:

    # check type
    valid_merge_types = ["concatenate", "merge"]
    if merge_type not in valid_merge_types:
        str_valid_types = sf.format_print_list(valid_merge_types)
        raise ValueError(f"Invalid merge_type '{merge_type}': valid types are {str_valid_types}.")

    # start building the output dataframe and retrieve dimensions of analysis for merging/ordering
    df_out = dfs_output_data[0]
    dims_to_order = model_attributes.sort_ordered_dimensions_of_analysis
    dims_in_out = set([x for x in dims_to_order if x in df_out.columns])

    if (len(dfs_output_data) == 0):
        return None
    if len(dfs_output_data) == 1:
        return dfs_output_data[0]
    elif len(dfs_output_data) > 1:
        # loop to merge where applicable
        for i in range(1, len(dfs_output_data)):
            if merge_type == "concatenate":
                # check available dims; if there are ones that aren't already contained, keep them. Otherwise, drop
                fields_dat = [x for x in dfs_output_data[i].columns if (x not in dims_to_order)]
                fields_new_dims = [x for x in dfs_output_data[i].columns if (x in dims_to_order) and (x not in dims_in_out)]
                dims_in_out = dims_in_out | set(fields_new_dims)
                dfs_output_data[i] = dfs_output_data[i][fields_new_dims + fields_dat]
            elif merge_type == "merge":
                df_out = pd.merge(df_out, dfs_output_data[i])

        # clean up - assume merged may need to be re-sorted on rows
        if merge_type == "concatenate":
            fields_dim = [x for x in dims_to_order if x in dims_in_out]
            df_out = pd.concat(dfs_output_data, axis = 1).reset_index(drop = True)
        elif merge_type == "merge":
            fields_dim = [x for x in dims_to_order if x in df_out.columns]
            df_out = pd.concat(df_out, axis = 1).sort_values(by = fields_dim).reset_index(drop = True)

        fields_dat = [x for x in df_out.columns if x not in dims_in_out]
        fields_dat.sort()
        #
        return df_out[fields_dim + fields_dat]


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

    init_merge_q = True
    df_output_data = []

    # set up models
    if "AFOLU" in models_run:
        print("\n\tRunning AFOLU")
        model_afolu = sm.AFOLU(sa.model_attributes)
        df_output_data += [model_afolu.project(df_input_data)]
        init_merge_q = False

    if "CircularEconomy" in models_run:
        print("\n\tRunning CircularEconomy")
        model_circecon = sm.CircularEconomy(sa.model_attributes)
        df_output_data += [model_circecon.project(df_input_data)]

    #########################
    #   other models here   #
    #########################

    # build output data frame
    df_output_data = merge_output_df_list(df_output_data, sa.model_attributes, "concatenate")


    print("\n*** MODEL RUNS COMPLETE ***\n")

    # write output
    print(f"\nWriting data to {fp_out}...")
    df_output_data.to_csv(fp_out, index = None, encoding = "UTF-8")
    print("\n*** MODEL RUNS SUCCESSFULLY COMPLETED. Q les vayan bien damas y caballeros.")


if __name__ == "__main__":

    args = parse_arguments()

    main(args)
