import model_attributes as ma
import model_ippu as mi
import model_socioeconomic as se
import numpy as np
import pandas as pd
import support_functions as sf
import transformations_base_general as tbg
from typing import *




###################################
###                             ###
###    AFOLU TRANSFORMATIONS    ###
###                             ###
###################################

##############
#    AGRC    #
##############

def transformation_agrc_reduce_supply_chain_losses(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce Supply Chain Losses" transformation.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: minimum reduction, from final time period, in supply chain 
        losses--e.g., to reduce supply chain losses by 30%, enter 0.3
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for property and method access 
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_afolu = (
        mafl.AFOLU(model_attributes) 
        if model_afolu is None
        else model_afolu
    )
    
    magnitude = (
        float(sf.vec_bounds(1 - magnitude, (0.0, 1.0)))
        if sf.isnumber(magnitude)
        else None
    )

    if magnitude is None:
        # LOGGING
        return df_input
    
    # call general transformation
    df_out = tbg.transformation_general(
        df_input,
        model_attributes,
        {
            model_afolu.modvar_agrc_frac_production_lost: {
                "bounds": (0, 1),
                "magnitude": magnitude,
                "magnitude_type": "baseline_scalar",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out




def transformation_agrc_reduce_supply_chain_losses2(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce Supply Chain Losses" transformation.


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: float specifying decrease as proprtion of final value (e.g.,
        a 30% reduction is entered as 0.3) OR  dictionary mapping individual 
        categories to reductions (must be specified for each category)
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
    modvar = "tmp"

    # convert the magnitude to a reduction as per input instructions
    magnitude = (
        float(sf.vec_bounds(1 - magnitude, bounds))
        if sf.isnumber(magnitude)
        else dict(
            (k, float(sf.vec_bounds(1 - v, bounds)))
            for k, v in magnitude.items()
        )
    )
    
    # call from general
    df_out = tbg.transformation_general_with_magnitude_differential_by_cat(
        df_input,
        magnitude,
        modvar,
        vec_ramp,
        model_attributes,
        bounds = bounds,
        categories = categories,
        magnitude_type = "baseline_scalar",
        **kwargs
    )

    return df_out




def transformation_waso_increase_recycling(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_circecon: Union[mc.CircularEconomy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Recycling" transformation (affects industrial 
        production in integrated environment)

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: proportion of recyclable solid waste that is recycled by 
        final time period
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_circecon: optional CircularEconomy object to pass for variable 
        access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    
    model_circecon = (
        mc.CircularEconomy(model_attributes) 
        if model_circecon is None
        else model_circecon
    )

    # check category specification
    categories = model_attributes.get_valid_categories(
        categories,
        model_attributes.subsec_name_waso
    )
    if categories is None:
        # LOGGING
        return df_input
    
    # call general transformation
    df_out = tbg.transformation_general(
        df_input,
        model_attributes,
        {
            model_circecon.modvar_waso_frac_recycled: {
                "bounds": (0, 1),
                "magnitude": magnitude,
                "magnitude_type": "final_value_floor",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out


##############
#    FRST    #
##############


##############
#    LNDU    #
##############


# USE AFOLU MODEL TRANSITION MATRIX FUNCTIONS TO MODIFY MATRICES



##############
#    LSMM    #
##############


##############
#    LVST    #
##############


