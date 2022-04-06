import os, os.path
import numpy as np
import pandas as pd


##  function to "projct" backwards waste that was deposited (used only in the absence of historical data)
def back_project_array(
    array_in: np.ndarray,
    n_periods: int = 10,
    bp_gr: float = 0.03,
    use_mean_forward: bool = False,
    n_periods_for_gr: int = 10
) -> np.ndarray:
    """
        array_in: array to use for back projection

        n_periods: number of periods to back project

        bp_gr: float specifying the average growth rate for row entries during the back projection periods

        use_mean_forward: default is False. If True, use the average empirical growth rate in array_in for the first 'n_periods_for_gr' periods

        n_periods_for_gr: if use_mean_forward == True, number of periods to look forward (rows 1:n_periods_for_gr)
    """

    if use_mean_forward:
        # get a mean growth rate
        n_periods_for_gr = max(min(n_periods_for_gr, len(array_in) - 1), 1)
        growth_scalars = array_in[1:(n_periods_for_gr + 1)]/array_in[0:(n_periods_for_gr)]
        vec_mu = np.mean(growth_scalars, axis = 0)
    else:
        vec_mu = (1 + bp_gr)*np.ones(len(array_in[0]))
    # set up an array of exponents
    array_exponent = -np.outer(n_periods - np.arange(n_periods), np.ones(len(vec_mu)))

    return (vec_mu**array_exponent)*array_in[0]


##  build a dictionary from a dataframe
def build_dict(df_in, dims = None):

    if len(df_in.columns) == 2:
        dict_out = dict([x for x in zip(df_in.iloc[:, 0], df_in.iloc[:, 1])])
    else:
        if dims == None:
            dims = (len(df_in.columns) - 1, 1)
        n_key = dims[0]
        n_val = dims[1]
        if n_key + n_val != len(df_in.columns):
            raise ValueError(f"Invalid dictionary dimensions {dims}: the sum of dims should be equal to the number of columns in the input dataframe ({len(df_in.columns)}). They sum to {n_key + n_val}.")

        # keys to zip
        if n_key == 1:
            keys = df_in.iloc[:, 0]
        else:
            keys = [tuple(x) for x in np.array(df_in[list(df_in.columns)[0:n_key]])]
        # values to zip
        if n_val == 1:
            vals = df_in.iloc[:, len(df_in.columns) - 1]
        else:
            vals = [np.array(x) for x in np.array(df_in[list(df_in.columns)[n_key:(n_key + n_val)]])]

        dict_out = dict([x for x in zip(keys, vals)])

    return dict_out


# check that the data frame contains required information
def check_fields(df, fields):
    s_fields_df = set(df.columns)
    s_fields_check = set(fields)
    if s_fields_check.issubset(s_fields_df):
        return True
    else:
        fields_missing = format_print_list(s_fields_check - s_fields_df)
        raise ValueError(f"Required fields {fields_missing} not found in the data frame.")


# check that a dictionary contains the required keys
def check_keys(dict_in, keys):
    s_keys_dict = set(dict_in.keys())
    s_keys_check = set(keys)
    if s_keys_check.issubset(s_keys_dict):
        return True
    else:
        fields_missing = format_print_list(s_keys_check - s_keys_dict)
        raise KeyError(f"Required keys {fields_missing} not found in the dictionary.")


##  check path and create a directory if needed
def check_path(fp, create_q = False):
    if os.path.exists(fp):
        return fp
    elif create_q:
        os.makedirs(fp, exist_ok = True)
        return fp
    else:
        raise ValueError(f"Path '{fp}' not found. It will not be created.")


##  check row sums to ensure they add to 1
def check_row_sums(
    array: np.ndarray,
    sum_restriction: float = 1,
    thresh_correction: float = 0.001,
    msg_pass: str = ""
):
    sums = array.sum(axis = 1)
    max_diff = np.max(np.abs(sums - sum_restriction))
    if max_diff > thresh_correction:
        raise ValueError(f"Invalid row sums in array{msg_pass}. The maximum deviance is {max_diff}, which is greater than the threshold for correction.")
    else:
        return (array.transpose()/sums).transpose()


##  print a set difference; sorts to ensure easy reading for user
def check_set_values(subset: set, superset: set, str_append: str) -> str:
    if not set(subset).issubset(set(superset)):
        invalid_vals = list(set(subset) - set(superset))
        invalid_vals.sort()
        invalid_vals = format_print_list(invalid_vals)
        raise ValueError(f"Invalid values {invalid_vals} found{str_append}.")


##  clean names of an input table to eliminate spaces/unwanted characters
def clean_field_names(nms, dict_repl: dict = {"  ": " ", " ": "_", "$": "", "\\": "", "\$": "", "`": "", "-": "_", ".": "_", "\ufeff": "", ":math:text": "", "{": "", "}": ""}):
    # check return type
    return_df_q =  False
    if type(nms) in [pd.core.frame.DataFrame]:
        df = nms
        nms = list(df.columns)
        return_df_q = True

    # get namses to clean, then loop
    nms = [str_replace(nm.lower(), dict_repl) for nm in nms]

    for i in range(len(nms)):
        nm = nms[i]
        # drop characters in front
        while (nm[0] in ["_", "-", "."]) and (len(nm) > 1):
            nm = nm[1:]
        # drop trailing characters
        while (nm[-1] in ["_", "-", "."]) and (len(nm) > 1):
            nm = nm[0:-1]
        nms[i] = nm

    if return_df_q:
        nms = df.rename(columns = dict(zip(list(df.columns), nms)))

    return nms


##  export a dictionary of data frames to an excel
def dict_to_excel(fp_out: str, dict_out: dict) -> None:
    with pd.ExcelWriter(fp_out) as excel_writer:
        for k in dict_out.keys():
            dict_out[k].to_excel(excel_writer, sheet_name = str(k), index = False, encoding = "UTF-8")


##  function to help fill in fields that are in another dataframe the same number of rows
def df_get_missing_fields_from_source_df(df_target, df_source, side = "right", column_vector = None):

    if df_target.shape[0] != df_source.shape[0]:
        raise RuntimeError(f"Incompatible shape found in data frames; the target number of rows ({df_target.shape[0]}) should be the same as the source ({df_source.shape[0]}).")
    # concatenate
    flds_add = [x for x in df_source.columns if x not in df_target]

    if side.lower() == "right":
        lcat = [df_target.reset_index(drop = True), df_source[flds_add].reset_index(drop = True)]
    elif side.lower() == "left":
        lcat = [df_source[flds_add].reset_index(drop = True), df_target.reset_index(drop = True)]
    else:
        raise ValueError(f"Invalid side specification {side}. Specify a value of 'right' or 'left'.")

    df_out = pd.concat(lcat,  axis = 1)

    if type(column_vector) == list:
        flds_1 = [x for x in column_vector if (x in df_out.columns)]
        flds_2 = [x for x in df_out.columns if (x not in flds_1)]
        df_out = df_out[flds_1 + flds_2]

    return df_out


##  simple but often used function
def format_print_list(list_in, delim = ","):
    return ((f"{delim} ").join(["'%s'" for x in range(len(list_in))]))%tuple(list_in)


##  print a set difference; sorts to ensure easy reading for user
def print_setdiff(superset: set, subset: set) -> str:
    missing_vals = list(superset - subset)
    missing_vals.sort()
    return format_print_list(missing_vals)


##  project a vector of growth scalars from a vector of growth rates and elasticities
def project_growth_scalar_from_elasticity(
    vec_rates: np.ndarray,
    vec_elasticity: np.ndarray,
    rates_are_factors = False,
    elasticity_type = "standard"
):
    """
        - vec_rates: a vector of growth rates, where the ith entry is the growth rate of the driver from i to i + 1. If rates_are_factors = False (default), rates are proportions (e.g., 0.02). If rates_are_factors = True, then rates are scalars (e.g., 1.02)

        - vec_elasticity: a vector of elasticities.

        - rates_are_factors: Default = False. If True, rates are treated as growth factors (e.g., a 2% growth rate is entered as 1.02). If False, rates are growth rates (e.g., 2% growth rate is 0.02).

        - elasticity_type: Default = "standard"; acceptable options are "standard" or "log"

            If standard, the growth in the demand is 1 + r*e, where r = is the growth rate of the driver and e is the elasiticity.

            If log, the growth in the demand is (1 + r)^e
    """
    # CHEKCS
    if vec_rates.shape[0] + 1 != vec_elasticity.shape[0]:
        raise ValueError(f"Invalid vector lengths of vec_rates ('{len(vec_rates)}') and vec_elasticity ('{len(vec_elasticity)}'). Length of vec_elasticity should be equal to the length vec_rates + 1.")
    valid_types = ["standard", "log"]
    if elasticity_type not in valid_types:
        v_types = sf.format_print_list(valid_types)
        raise ValueError(f"Invalid elasticity_type {elasticity_type}: valid options are {v_types}.")
    # check factors
    if rates_are_factors:
        vec_rates = vec_rates - 1 if (elasticity_type == "standard") else vec_rates
    else:
        vec_rates = vec_rates if (elasticity_type == "standard") else vec_rates + 1
    # check if transpose needs to be used
    transpose_q = True if len(vec_rates.shape) != len(vec_elasticity.shape) else False

    # get scalar
    if elasticity_type == "standard":
        rates_adj = (vec_rates.transpose()*vec_elasticity[0:-1].transpose()).transpose() if transpose_q else vec_rates*vec_elasticity[0:-1]
        vec_growth_scalar = np.cumprod(1 + rates_adj, axis = 0)
        ones = np.ones(1) if (len(vec_growth_scalar.shape) == 1) else np.ones((1, vec_growth_scalar.shape[1]))
        vec_growth_scalar = np.concatenate([ones, vec_growth_scalar])
    elif elasticity_type == "log":
        ones = np.ones(1) if (len(vec_rates.shape) == 1) else np.ones((1, vec_rates.shape[1]))
        vec_growth_scalar = np.cumprod(np.concatenate([ones, vec_rates], axis = 0)**vec_elasticity)

    return vec_growth_scalar


##  repeat the first row and prepend
def prepend_first_element(array: np.ndarray, n_rows: int) -> np.ndarray:
    out = np.concatenate([
        np.repeat(array[0:1], n_rows, axis = 0), array
    ])
    return out

##  replace values in a two-dimensional array
def repl_array_val_twodim(array, val_repl, val_new):
    # only for two dimensional arrays
    w = np.where(array == val_repl)
    inds = w[0]*len(array[0]) + w[1]
    np.put(array, inds, val_new)
    return None


##  set a vector to element-wise stay within bounds
def scalar_bounds(scalar, bounds: tuple):
    bounds = np.array(bounds).astype(float)
    return min([max([scalar, min(bounds)]), max(bounds)])


##  multiple string replacements using a dictionary
def str_replace(str_in: str, dict_replace: dict) -> str:
    for k in dict_replace.keys():
        str_in = str_in.replace(k, dict_replace[k])
    return str_in


##  subset a data frame using a dictionary
def subset_df(df, dict_in):
    for k in dict_in.keys():
        if k in df.columns:
            if type(dict_in[k]) != list:
                val = [dict_in[k]]
            else:
                val = dict_in[k]
            df = df[df[k].isin(val)]
    return df


##  set a vector to element-wise stay within bounds
def vec_bounds(vec, bounds: tuple):
    def f(x):
        return scalar_bounds(x, bounds)
    f_z = np.vectorize(f)

    return f_z(vec).astype(float)
