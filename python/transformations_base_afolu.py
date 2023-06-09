import model_attributes as ma
import model_afolu as mafl
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

def transformation_support_lndu_transition_to_category_targets_single_region(
    df_input: pd.DataFrame,
    magnitude: Dict[str, float],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    cats_stable: Union[List[str], None] = None,
    magnitude_type: str = "final_value",
    max_value: float = 0.8,
    model_afolu: Union[mafl.AFOLU, None = None,
    **kwargs
 ) -> pd.DataFrame:
    """
    Modify transition probabilities to ach eieve targets for land use categories.


    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: dictionary mapping land use categories to target fractions.
        NOTE: caution should be taken to not overuse this; transition matrices
            can be chaotic, and modifying too many target categories may cause 
            strange behavior. 
    - model_attributes: ModelAttributes object used to call strategies/
        variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - cats_stable: optional set of categories to preserve with stable transition
        probabilities *out* of the categori
    - field_region: field in df_input that specifies the region
    - magnitude_type: type of magnitude to use. Valid types include
        * "baseline_scalar": multiply baseline value by magnitude
        * "final_value": magnitude is the final value for the variable to take 
            (achieved in accordance with vec_ramp)
        * "final_value_ceiling": magnitude is the lesser of (a) the existing 
            final value for the variable to take (achieved in accordance with 
            vec_ramp) or (b) the existing specified final value, whichever is 
            smaller
        * "final_value_floor": magnitude is the greater of (a) the existing 
            final value for the variable to take (achieved in accordance with 
            vec_ramp) or (b) the existing specified final value, whichever is 
            greater
        * "transfer_value_scalar": transfer value from categories to other 
            categories based on a scalar. Must specify "categories_source" &
            "categories_target" in dict_modvar_specs. See description below in 
            OPTIONAL for information on specifying this.
    - max_value: maximum value in final time period that any land use class can 
        take
    - model_afolu: optional AFOLU object to pass for variable access
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)HEREHERE
    """

    model_afolu = (
        mafl.AFOLU(model_attributes)
        if model_afolu is None
        else model_afolu
    )

    attr_lndu = model_attributes.get_attribute_table(
        model_attributes.subsec_name_lndu
    )

    if not isinstance(magnitude, dict):
        return df_input

    magnitude = dict(
        (k, float(sf.vec_bounds(v, (0.0, 1.0)))) for k, v in magnitude.items()
        if k in attr_lndu.key_values
        and sf.isnumber(v)
    )
    magnitude = (
        None
        if (sum(magnitude.values()) > 1) | (len(magnitude) == 0)
        else magnitude
    )
    n_tp = len(df_input)
    
    # get indices for categories
    cats_to_modify = sorted(list(magnitude.keys()))
    inds_to_modify = [attr_lndu.get_key_value_index(x) for x in cats_to_modify]
    

    ####################################################################################
    #                                                                                  #
    #   REPEAT MODEL PROJECTIONS OF LAND USE UNDER LURF = 0 (specified transitions)    #
    #       - Calculate fractions going forward                                        #
    #       - Determine appropriate target magnitudes (based on magnitude type)        #
    #       - Use model_afolu.adjust_transition_matrix() to scale columns up/down      #
    #                                                                                  #
    ####################################################################################


    ##  1. GET COMPONENTS USED FOR LAND USE PROJECTION

    # get the initial distribution of land
    vec_lndu_initial_frac = model_attributes.get_standard_variables(
        df_input, 
        model_afolu.modvar_lndu_initial_frac, 
        return_type = "array_base"
    )[0]
    
    # determine when to initialize the scaling
    ind_first_nz = np.where(vec_ramp > 0)[0][0]
    ind_last_zero = ind_first_nz - 1

    # get transition matrices and emission factors
    qs, efs = model_afolu.get_markov_matrices(
        df_input, 
        len(df_input)
    )

    ##  2. IF magnitude_type IS RELATIVE TO BASE VALUE, NEED TO KNOW BASELINE VALUES
    
    # project land use over all time periods and get final fractions without intervention
    arr_emissions_conv, arr_land_use, arrs_land_conv = model_afolu.project_land_use(
        vec_lndu_initial_frac,
        qs,
        efs, 
        n_tp = ind_last_zero,
    )
    vec_lndu_final_frac_unadj = arr_land_use[-1, :]
    
    if magnitude_type in ["baseline_scalar", "final_value_ceiling", "final_value_floor"]:
        
        for i, cat in enumerate(cats_to_modify):

            ind = inds_to_modify[i]
            mag_cur = magnitude.get(cat)
            val_unadj = vec_lndu_final_frac_unadj[ind]

            if magnitude_type == "baseline_scalar":
                mag_new = mag_cur*val_unadj
            elif magnitude_type == "final_value_ceiling":
                mag_new = min(mag_cur, val_unadj)
            elif magnitude_type == "final_value_floor":
                mag_new = max(mag_cur, val_unadj)

            magnitude.update({cat: mag_new})


    ##  3. CONTINUE WITH PROJECTION AND ADJUSTMENT OF PROBS DURING vec_implementation_ramp NON-ZERO YEARS

    # run forward to final period before ramp for all associated with no change 
    arr_emissions_conv, arr_land_use, arrs_land_conv = model_afolu.project_land_use(
        vec_lndu_initial_frac,
        qs[0:ind_first_nz],
        efs, 
        n_tp = ind_last_zero,
    )
    vec_lndu_final_virnz_frac_unadj = arr_land_use[-1, :] # at time ind_first_nz - 1


    ##  4. PREPARE TRANSITION MATRIX FOR MODIFICATION

    # initialize adjustment dictionary
    dict_adj = {}

    # check unadjusted final period fractions
    n_tp_scale = n_tp - ind_first_nz - 1
    fracs_unadj_first_effect_tp = np.dot(vec_lndu_final_frac_unadj, qs[ind_first_nz - 1])
    fracs_unadj_first_effect_tp = np.dot(fracs_unadj_first_effect_tp, qs[ind_first_nz])[inds_to_modify]
    fracs_target_final_tp = np.array([magnitude.get(x) for x in cats_to_modify])

    """
    OPTION FOR EXPANSION: SPECIFY 

    df_tp = df_input[[model_attributes.dim_time_period]].copy()
    
    df_tmp = df_tp.iloc[0:ind_first_nz].copy()
    df_tmp = pd.concat(
        [
            df_tmp.reset_index(drop = True), 
            pd.DataFrame(
                arr_land_use[:, inds_to_modify],
                columns = cats_to_modify
            )
        ],
        axis = 1
    )

    df_append = {
        model_attributes.dim_time_period: [int(df_input[model_attributes.dim_time_period].iloc[-1])]
    }
    df_append.update(
        dict(
            (cats_to_modify: [])
        )
    )


    df_tmp = pd.DataFrame({
        time_periods.field_time_period: [0, 1, 2, 3, 4, 5, 35],
        #"val": [0.3, 0.31, 0.32, 0.3275, 0.3325, 0.335, 0.335]
        "val": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.335]
    })
    df_tmp = pd.merge(df_tp, df_tmp, how = "left")
    ?df_tmp.interpolate
    df_tmp.interpolate(method = "linear", order = 2).plot(x = "time_period")
    """
    arr_target_shares = (fracs_target_final_tp - fracs_unadj_first_effect_tp)/n_tp_scale
    arr_target_shares = np.outer(np.arange(1, n_tp_scale + 1), arr_target_shares)
    arr_target_shares += fracs_unadj_first_effect_tp

    # verify and implement stable output transition categories
    cats_ignore = []
    cats_stable = (
        [x for x in attr_lndu.key_values if x in cats_stable]
        if sf.islistlike(cats_stable)
        else []
    )

    
    for cat_stable in cats_stable:
        ind_lndu_stable = attr_lndu.get_key_value_index(cat_stable)
        for cat in attr_lndu.key_values:
            ind = attr_lndu.get_key_value_index(cat)
            if cat not in cats_ignore:
                dict_adj.update({(ind_lndu_stable, ind): 1})
    

    
    ##  5. FINALLY, ADJUST TRANSITION MATRICES AND OVERWRITE

    """
    Process:

    a. Start with land use prevalance at time ind_first_nz 
        i. estimated as prevalence at x_{ind_first_nz - 1}Q_{ind_first_nz - 1}
    b. Next, project forward from time t to t+1 and get adjustment to columnar
        inflows (scalars_to_adj)
    c. Use `model_afolu.get_lndu_scalar_max_out_states` to get true positional 
        scalars. This accounts for states that might "max out" (as determined by
        model_afolu.mask_lndu_max_out_states), or reach 100% or 0% probability 
        during the scaling process.
    d. Then, with the scalars obtained, adjust the matrix using 
        model_afolu.adjust_transition_matrix
    """;

    x = np.dot(vec_lndu_final_virnz_frac_unadj, qs[ind_first_nz - 1]) 
    inds_iter = list(range(ind_first_nz, n_tp))    

    for ind_row, i in enumerate(inds_iter):

        # in first iteation, this is projected prevalence at ind_first_nz + 1
        x_next_unadj = np.dot(x, qs[i]) 
        ind_row = min(ind_row, arr_target_shares.shape[0] - 1)
        scalars_to_adj = arr_target_shares[ind_row, :]/x_next_unadj[inds_to_modify]

        for j, z in enumerate(inds_to_modify):

            scalar = scalars_to_adj[j]

            mask_lndu_max_out_states = model_afolu.get_lndu_scalar_max_out_states(scalar)
            scalar_lndu_cur = model_afolu.get_matrix_column_scalar(
                qs[i][:, z],
                scalar,
                x,
                mask_max_out_states = mask_lndu_max_out_states
            )

            dict_adj.update(
                dict(
                    ((ind, z), s) for ind, s in enumerate(scalar_lndu_cur)
                )
            )

        qs[i] = model_afolu.adjust_transition_matrix(qs[i], dict_adj)
        x = np.dot(x, qs[i])

        #scalar_true = x/x_next_unadj
        #scalar_error = (scalar_true[inds_to_modify] - scalars_to_adj)/scalars_to_adj
        #print(f"scalar_error:\t{scalar_error}")
        #print(f"scalar_true:\t{scalar_true[inds_to_modify]}")
        #print(f"scalars_to_adj (target):\t{scalars_to_adj}")

    # convert to input format and overwrite in output data
    df_out = model_afolu.format_transition_matrix_as_input_dataframe(qs)
    df_out = sf.match_df_to_target_df(
        df_input,
        df_out,
        [model_attributes.dim_time_period]
    )

    return df_out, qs
    


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
        dataframe (only added if integer)
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