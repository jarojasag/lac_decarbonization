import model_attributes as ma
import model_ippu as mi
import model_socioeconomic as se
import numpy as np
import pandas as pd
import support_functions as sf
from transformations_base_general import *
from typing import *




##########################################
###                                    ###
###    CROSS-SECTOR TRANSFORMATIONS    ###
###                                    ###
##########################################
"""
NOTE: use MLTI for multi-sector transformations. Some are specified in one
    sector but are called from different models
"""

def transformation_mlti_industrial_carbon_capture(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_ippu: Union[mi.IPPU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Capture carbon at point of emission in industry (excludes Fuel Production
        and electricity). Uses IPPU (input variable is associated with IPPU due
        to heavy CO2 emissions in cement and metal production)

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: float specifying capture as proprtion of final value (e.g.,
        a 30% capture is entered as 0.3)  OR  dictionary mapping individual 
        categories to proportions (must be specified for each category)
        * NOTE: overrides `categories` keyword argument if both are specified
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_circecon: optional IPPU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # get attribute table, CircularEconomy model for variables, and check categories
    attr_ippu = model_attributes.get_attribute_table(model_attributes.subsec_name_ippu)
    bounds = (0, 1)
    model_ippu = (
        mi.IPPU(model_attributes) 
        if model_ippu is None
        else model_ippu
    )
    modvar = model_ippu.modvar_ippu_frac_captured_co2
    
    # call from general
    df_out = transformation_general_with_magnitude_differential_by_cat(
        df_input,
        magnitude,
        modvar,
        vec_ramp,
        model_attributes,
        bounds = bounds,
        categories = categories,
        magnitude_type = "final_value_floor",
        **kwargs
    )

    return df_out