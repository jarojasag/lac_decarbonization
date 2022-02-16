import os, os.path
import pandas as pd
import numpy as np
import support_functions as sf
import data_structures as ds

# TEMP
import importlib
importlib.reload(sf)
importlib.reload(ds)

# setup some
dir_py = os.path.dirname(os.path.realpath(__file__))
print(dir_py)
dir_proj = os.path.dirname(dir_py)

# key subdirectories for the project
dir_jl = sf.check_path(os.path.join(dir_proj, "julia"), False)
dir_out = sf.check_path(os.path.join(dir_proj, "out"), True)
dir_ref = sf.check_path(os.path.join(dir_proj, "ref"), False)

# attribute tables and readthedocs
dir_docs = sf.check_path(os.path.join(os.path.dirname(dir_py), "docs", "source"), False)
dir_attribute_tables = sf.check_path(os.path.join(dir_docs, "csvs"), False)


# get model attributes
model_attributes = ds.ModelAttributes(dir_attribute_tables)


##  some paths
fp_csv_default_single_run_out = os.path.join(dir_out, "single_run_output.csv")
