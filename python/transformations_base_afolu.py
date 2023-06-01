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

def transformation_agrc_improve_rice_management(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Improve Rice Management" transformation.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: proportional reduction, by final time period, in methane 
        emitted from rice production--e.g., to reduce methane from rice by 
        30%, enter 0.3
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
            model_afolu.modvar_agrc_ef_ch4: {
                "bounds": (0, np.inf),
                "categories": ["rice"],
                "magnitude": magnitude,
                "magnitude_type": "baseline_scalar",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out



def transformation_agrc_increase_crop_productivity(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Crop Productivity" transformation (increases yield
        factors)


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: float specifying increase as proprtion of final value (e.g.,
        a 30% increase is entered as 0.3) OR  dictionary mapping individual 
        categories to reductions (must be specified for each category)
        * NOTE: overrides `categories` keyword argument if both are specified
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)HEREHERE
    """

    # get attribute table, CircularEconomy model for variables, and check categories
    modvar = model_afolu.modvar_agrc_yf
    bounds = (0, np.inf)

    # convert the magnitude to a reduction as per input instructions
    magnitude = (
        float(sf.vec_bounds(1 + magnitude, bounds))
        if sf.isnumber(magnitude)
        else dict(
            (k, float(sf.vec_bounds(1 + v, bounds)))
            for k, v in magnitude.items()
        )
    )

    # check category specification
    categories = model_attributes.get_valid_categories(
        categories,
        model_attributes.subsec_name_agrc
    )
    if categories is None:
        # LOGGING
        return df_input

    # call from general
    df_out = tbg.transformation_general_with_magnitude_differential_by_cat(
        df_input,
        magnitude,
        modvar,
        vec_ramp,
        model_attributes,
        categories = categories,
        magnitude_type = "baseline_scalar",
        **kwargs
    )

    return df_out



def transformation_agrc_reduce_supply_chain_losses(
    df_input: pd.DataFrame,
    magnitude: float,
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
    - magnitude: proportional minimum reduction, from final time period, in 
        supply chain losses--e.g., to reduce supply chain losses by 30%, enter 
        0.3
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





##############
#    FRST    #
##############


##############
#    LNDU    #
##############

def transformation_lndu_increase_soc_factor_fmg(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Set a new floor for F_MG (as described in in V4 Equation 2.25 (2019R)). Used
        to Implement the "Expand Conservation Agriculture" transformation, which 
        reduces losses of soil organic carbon through no-till. Can be 
        implemented in cropland and grassland. 


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: float specifying target value of F_{MG} (e.g. per Table 5.5 in 
        V4, Chapter 5 [Croplands], no-till can increase F_{MG} to 1.1 under 
        certain conditions) OR  dictionary mapping individual 
        categories to reductions (must be specified for each category)
        * NOTE: overrides `categories` keyword argument if both are specified
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
    - field_region: field in df_input that specifies the region
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)HEREHERE
    """

    # get attribute table, CircularEconomy model for variables, and check categories
    modvar = model_afolu.modvar_lndu_factor_soil_management
    bounds = (0.0, np.inf)

    # convert the magnitude to a reduction as per input instructions
    magnitude = (
        float(sf.vec_bounds(magnitude, bounds))
        if sf.isnumber(magnitude)
        else dict(
            (k, float(sf.vec_bounds(v, bounds)))
            for k, v in magnitude.items()
        )
    )

    # check category specification
    categories = model_attributes.get_valid_categories(
        categories,
        model_attributes.subsec_name_lndu
    )
    if categories is None:
        # LOGGING
        return df_input

    # call from general - set as floor (don't want to make it worse)
    df_out = tbg.transformation_general_with_magnitude_differential_by_cat(
        df_input,
        magnitude,
        modvar,
        vec_ramp,
        model_attributes,
        categories = categories,
        magnitude_type = "final_value_floor",
        **kwargs
    )

    return df_out

# USE AFOLU MODEL TRANSITION MATRIX FUNCTIONS TO MODIFY MATRICES



##############
#    LSMM    #
##############


##############
#    LVST    #
##############

def transformation_lvst_increase_productivity(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Livestock Productivity" transformation.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: proportional increase, by final time period, in livestock
        carrying capacity per area of managed grassland--e.g., to increase 
        productivity by 30%, enter 0.3. 
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
        float(sf.vec_bounds(1 + magnitude, (0.0, np.inf)))
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
            model_afolu.modvar_lvst_carrying_capacity_scalar: {
                "bounds": (0.0, np.inf),
                "magnitude": magnitude,
                "magnitude_type": "baseline_scalar",
                "vec_ramp": vec_ramp
            }
        },
        **kwargs
    )
    return df_out



def transformation_lvst_reduce_enteric_fermentation(
    df_input: pd.DataFrame,
    magnitude: Union[Dict[str, float], float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: Union[List[str], None] = None,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce Enteric Fermentation" transformation.


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
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # get attribute table, CircularEconomy model for variables, and check categories
    modvar = model_afolu.modvar_lvst_ef_ch4_ef
    bounds = (0, 1)

    # convert the magnitude to a reduction as per input instructions
    magnitude = (
        float(sf.vec_bounds(1 - magnitude, bounds))
        if sf.isnumber(magnitude)
        else dict(
            (k, float(sf.vec_bounds(1 - v, bounds)))
            for k, v in magnitude.items()
        )
    )

    # check category specification
    categories = model_attributes.get_valid_categories(
        categories,
        model_attributes.subsec_name_lvst
    )
    if categories is None:
        # LOGGING
        return df_input

    # call from general
    df_out = tbg.transformation_general_with_magnitude_differential_by_cat(
        df_input,
        magnitude,
        modvar,
        vec_ramp,
        model_attributes,
        categories = categories,
        magnitude_type = "baseline_scalar",
        **kwargs
    )

    return df_out



##############
#    SOIL    #
##############

def transformation_soil_reduce_excess_fertilizer(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce Excess Fertilizer" transformation. Can be used to
        reduce excess N from fertilizer or reduce liming. See `magnitude` in 
        function arguments for information on dictionary specification.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: dictionary with keys for "fertilizer_n" and "lime" mapping 
        to proportional reductions in the per unit application of fertilizer N 
        and lime, respectively. If float, applies to fertilizer N and lime 
        uniformly.
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: optional subset of categories to apply to
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

    ##  BUILD TRANSFORMATION FOR BIOGAS/LANDFILL

    dict_key_to_modvar = {
        "fertilizer_n": model_afolu.modvar_soil_demscalar_fertilizer,
        "lime": model_afolu.modvar_soil_demscalar_liming
    }

    dict_transformation = {}
    for key, modvar in dict_key_to_modvar.items():
        # get the current magnitude of gas capture
        mag = (
            magnitude.get(key)
            if isinstance(magnitude, dict)
            else (magnitude if sf.isnumber(magnitude) else None)
        )

        mag = (
            float(sf.vec_bounds(1 - mag, (0, 1)))
            if mag is not None
            else None
        )

        (
            dict_transformation.update(
                {
                    modvar: {
                        "bounds": None,
                        "magnitude": mag,
                        "magnitude_type": "baseline_scalar",
                        "vec_ramp": vec_ramp
                    }
                }
            )
            if mag is not None
            else None
        )


    # call general transformation
    df_out = (
        tbg.transformation_general(
            df_input,
            model_attributes,
            dict_transformation,
            **kwargs
        )
        if len(dict_transformation) > 0
        else df_input
    )

    return df_out