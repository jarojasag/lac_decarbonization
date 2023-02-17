import model_attributes as ma
import model_afolu as mafl
import model_ippu as mi
import model_circular_economy as mc
import model_electricity as ml
import model_energy as me
import model_socioeconomic as se
import numpy as np
import pandas as pd
import support_functions as sf
from typing import *



####################################
###                              ###
###    ENERGY TRANSFORMATIONS    ###
###                              ###
####################################

def transformation_test_elec(
    df_input: pd.DataFrame,
    model_attributes: ma.ModelAttributes,
    vec_ramp: np.ndarray,
    all_regions: Union[List[str], None] = None,
    field_region = "nation",
    model_energy: Union[me.NonElectricEnergy, None] = None,
    strategy_id: int = 3001
) -> pd.DataFrame:

    model_energy = me.NonElectricEnergy(model_attributes) if not isinstance(model_energy, me.NonElectricEnergy) else model_energy
    all_regions = sorted(list(set(df_input[field_region]))) if (all_regions is None) else all_regions
    df_out = []

    # model variables to explore
    modvars = [
        model_energy.modvar_trns_fuel_fraction_biofuels,
        model_energy.modvar_trns_fuel_fraction_diesel,
        model_energy.modvar_trns_fuel_fraction_electricity,
        model_energy.modvar_trns_fuel_fraction_gasoline,
        model_energy.modvar_trns_fuel_fraction_hydrogen,
        model_energy.modvar_trns_fuel_fraction_kerosene,
        model_energy.modvar_trns_fuel_fraction_natural_gas
    ]


    #
    all_regions = sorted(list(set(df_input[field_region]))) if (all_regions is None) else all_regions

    for region in all_regions:

        df_in = df_input[df_input[field_region] == region].reset_index(drop = True)
        df_in_new = df_in.copy()
        n_tp = len(df_in)
        fields_fuelmix = [x for x in df_in.columns if x.startswith("frac_trns_fuelmix_")]


        # get electric categories and build dictionary of target values
        cats_elec = model_attributes.get_variable_categories(model_energy.modvar_trns_fuel_fraction_electricity)
        cats_half = ["aviation", "water_borne"]
        dict_targets_final_tp = dict((x, 1.0 if x not in cats_half else 0.5) for x in cats_elec)

        # l
        for cat in dict_targets_final_tp.keys():

            target_value = dict_targets_final_tp.get(cat)
            scale_non_elec = 1 - target_value
            field_elec = model_attributes.build_varlist(
                model_attributes.subsec_name_trns,
                model_energy.modvar_trns_fuel_fraction_electricity,
                restrict_to_category_values = [cat]
            )[0]

            val_final_elec = float(df_in[field_elec].iloc[n_tp - 1])

            # get model variables that need to be adjusted
            modvars_adjust = []
            for modvar in modvars:
                modvars_adjust.append(modvar) if cat in model_attributes.get_variable_categories(modvar) else None

            # loop over adjustment variables to build new trajectories
            for modvar in modvars_adjust:
                field_cur = model_attributes.build_varlist(
                    model_attributes.subsec_name_trns,
                    modvar,
                    restrict_to_category_values = [cat]
                )[0]
                vec_old = np.array(df_in[field_cur])
                val_final = vec_old[n_tp - 1]
                val_new = (val_final/(1 - val_final_elec))*scale_non_elec if (field_cur != field_elec) else target_value
                vec_new = vec_ramp*val_new + (1 - vec_ramp)*vec_old

                df_in_new[field_cur] = vec_new

        df_out.append(df_in_new)

    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)

    df_out = sf.add_data_frame_fields_from_dict(
        df_out,
        {
            model_attributes.dim_strategy_id: strategy_id
        },
        prepend_q = True,
        overwrite_fields = True
    )

    return df_out






###########################
#    SUPPORT FUNCTIONS    #
###########################

def get_time_period(
    model_attributes: ma.ModelAttributes,
    return_type: str = "max"
) -> int:
    """
    Get max or min time period using model_attributes. Set return_type = "max"
        for the maximum time period or return_type = "min" for the minimum time
        period.
    """
    attr_time_period = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_time_period}")
    return_val = min(attr_time_period.key_values) if (return_type == "min") else max(attr_time_period.key_values)

    return return_val



def transformation_general(
    df_input: pd.DataFrame,
    model_attributes: ma.ModelAttributes,
    dict_modvar_specs: Dict[str, Dict[str, str]],
    regions_apply: Union[List[str], None] = None,
    field_region = "nation",
    model_energy: Union[me.NonElectricEnergy, None] = None,
    strategy_id: Union[int, None] = None
) -> pd.DataFrame:
    """
    Generalized function to implement some common transformations. Many other
        transformation functions are wraper for this function.

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - dict_modvar_specs: dictionary mapping model variable to some
        characteristics:

        REQUIRED
        --------
        * "magnitude": magnitude of change to apply by final
        * "magnitude_type": type of magnitude to use. Valid types include
            * "baseline_additive": add the magnitude to the baseline
            * "baseline_scalar": multiply baseline value by magnitude
            * "baseline_scalar_diff_reduction": reduce the difference between
                the value in the baseline time period and the upper bound (NOTE:
                requires specification of bounds to work) by magnitude
            * "final_value": magnitude is the final value for the variable to
                take (achieved in accordance with vec_ramp)
            * "transfer_value": transfer value from categories to other
                categories. Must specify "categories_source" &
                "categories_target" in dict_modvar_specs. See description below
                in OPTIONAL for information on specifying this.
            * "transfer_value_to_acheieve_magnitude": transfer value from
                categories to other categories to acheive a target magnitude.
                Must specify "categories_source" & "categories_target" in
                dict_modvar_specs. See description below in OPTIONAL for
                information on specifying this.
        * "vec_ramp": implementation ramp vector to use for the variable

        OPTIONAL
        --------
        * "bounds": optional specification of bounds to use on final change
        * "categories": optional category restrictions to use
        * "categories_source" & "categories_target": must be specified together
            and only valid with the "transfer_value" or
            "transfer_value_to_acheieve_magnitude" magnitude_types. Transfers
            some quantity from categories specified within "categories_source"
            to categories "categories_target". "categories_target" is a
            dictionary of target categories mapping to proportions of the
            magnitude to receive.

            For example,

                {
                    "magnitude" = 0.8,
                    "categories_source" = ["cat_1", "cat_2", "cat_3"],
                    "categories_target" = {"cat_4": 0.7, "cat_5": 0.3}
                }

            will distribute 0.8 from categories 1, 2, and 3 to 4 and 5, giving
            0.56 to cat_4 and 0.24 to cat_5. In general, the source distribution
            is proportional to the source categories' implied pmf at the final
            time period.

        * "time_period_baseline": time period to use as baseline for change if
            magnitude_type in ["baseline_additive", "baseline_scalar"]

        EXAMPLE
        -------
        * The dictionary should take the following form:

        {
            modvar_0: {
                "magnitude": 0.5,
                "magnitude_type": "final_value",
                "vec_ramp": np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0]),
                "bounds": (0, 1),    # optional
                "time_period_change": 0    # optional
            },
            modvar_1: ...
        }

    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - field_region: field in df_input that specifies the region
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    ##  INITIAlIZATION

    # core vars (ordered)
    model_energy = me.NonElectricEnergy(model_attributes) if not isinstance(model_energy, me.NonElectricEnergy) else model_energy
    all_regions = sorted(list(set(df_input[field_region])))
    # dertivative vars (alphabetical)
    attr_time_period = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_time_period}")
    df_out = []
    regions_apply = all_regions if (regions_apply is None) else [x for x in regions_apply if x in all_regions]
    # valid specifications of magnitude type
    magnitude_types_valid = [
        "baseline_additive",
        "baseline_scalar",
        "baseline_scalar_diff_reduction",
        "final_value",
        "transfer_value",
        "transfer_value_to_acheieve_magnitude"
    ]


    ##  CHECK SPECIFICATION DICTIONARY

    modvars = sorted([x for x in dict_modvar_specs.keys() if x in model_attributes.all_model_variables])
    dict_modvar_specs_clean = {}
    for modvar in modvars:
        # default verified to true; set to false if any condition is not met
        verified_modvar = True
        dict_modvar_specs_cur = dict_modvar_specs.get(modvar)

        # check magnitude
        magnitude = dict_modvar_specs_cur.get("magnitude")
        verified_modvar = (isinstance(magnitude, int) or isinstance(magnitude, float)) & verified_modvar

        # check magnitude type
        magnitude_type = dict_modvar_specs_cur.get("magnitude_type")
        verified_modvar = (magnitude_type in magnitude_types_valid) & verified_modvar

        # check ramp vector
        vec_ramp = dict_modvar_specs_cur.get("vec_ramp")
        verified_modvar = isinstance(vec_ramp, np.ndarray) & verified_modvar

        # check for bounds
        bounds = dict_modvar_specs_cur.get("bounds")
        bounds = None if not (isinstance(bounds, tuple) and len(bounds) == 2) else bounds
        # check special case
        verified_modvar = ((bounds is not None) & verified_modvar) if (magnitude_type == "baseline_scalar_diff_reduction") else verified_modvar

        # check for categories
        categories = dict_modvar_specs_cur.get("categories")
        categories = None if not isinstance(categories, list) else categories

        # check for source/target categories
        categories_source = dict_modvar_specs_cur.get("categories_source")
        categories_source = None if not isinstance(categories_source, list) else categories_source
        categories_target = dict_modvar_specs_cur.get("categories_target")
        categories_target = dict((k, v) for k, v in categories_target.items() if isinstance(v, int) or isinstance(v, float)) if isinstance(categories_target, dict) else None
        # check special case
        verified_modvar = ((categories_source is not None) & verified_modvar) if (magnitude_type in ["transfer_value", "transfer_value_to_acheieve_magnitude"]) else verified_modvar
        verified_modvar = ((categories_target is not None) & verified_modvar) if (magnitude_type in ["transfer_value", "transfer_value_to_acheieve_magnitude"]) else verified_modvar

        # check for time period as baseline
        tp_baseline = dict_modvar_specs_cur.get("time_period_baseline")
        tp_baseline = min(attr_time_period.key_values) if (tp_baseline not in attr_time_period.key_values) else tp_baseline


        ## IF VERIFIED, ADD TO CLEANED DICTIONARY

        if verified_modvar:

            subsector = model_attributes.dict_model_variable_to_subsector.get(modvar)

            # check categories against subsector
            if (categories is not None) or (categories_source is not None) or (categories_target is not None):
                pycat = model_attributes.get_subsector_attribute(subsector, "pycategory_primary")
                attr = model_attributes.dict_attributes.get(pycat)

                if (categories is not None):
                    categories = [x for x in categories if x in attr.key_values]
                    categories = None if (len(categories) == 0) else categories

                if (categories_source is not None):
                    categories_source = [x for x in categories_source if x in attr.key_values]
                    categories_source = None if (len(categories_source) == 0) else categories_source

                if (categories_target is not None):
                    categories_target = dict((k, v) for k, v in categories_target.items() if k in attr.key_values)
                    categories_target = None if (sum(list(categories_target.values())) != 1.0) else categories_target

            vector_targets_ordered = [x for x in attr.key_values if x in categories_target.keys()] if isinstance(categories_target, dict) else None


            dict_modvar_specs_clean.update({
                modvar: {
                    "bounds": bounds,
                    "categories": categories,
                    "categories_source": categories_source,
                    "categories_target": categories_target,
                    "magnitude": magnitude,
                    "magnitude_type": magnitude_type,
                    "subsector": subsector,
                    "tp_baseline": tp_baseline,
                    "vec_ramp": vec_ramp,
                    "vector_targets_ordered": vector_targets_ordered
                }
            })

    modvars = sorted(list(dict_modvar_specs_clean.keys()))


    ##  ITERATE OVER REGIONS AND MODVARS TO BUILD TRANSFORMATION

    for region in all_regions:
        df_in = df_input[df_input[field_region] == region].sort_values(by = [model_attributes.dim_time_period]).reset_index(drop = True)
        df_in_new = df_in.copy()
        vec_tp = list(df_in[model_attributes.dim_time_period])
        n_tp = len(df_in)

        if region in regions_apply:
            for modvar in modvars:

                dict_cur = dict_modvar_specs_clean.get(modvar)

                # get components
                bounds = dict_cur.get("bounds")
                categories = dict_cur.get("categories")
                categories_source = dict_cur.get("categories_source")
                categories_target = dict_cur.get("categories_target")
                magnitude = dict_cur.get("magnitude")
                magnitude_type = dict_cur.get("magnitude_type")
                tp_baseline = dict_cur.get("tp_baseline")
                vec_ramp = dict_cur.get("vec_ramp")
                vector_targets_ordered = dict_cur.get("vector_targets_ordered")
                ind_tp_baseline = vec_tp.index(tp_baseline) if (magnitude_type in ["baseline_scalar", "baseline_additive", "baseline_scalar_diff_reduction"]) else None

                # set fields
                fields_adjust = model_attributes.build_varlist(
                    dict_cur.get("subsector"),
                    modvar,
                    restrict_to_category_values = categories
                )
                fields_adjust_source = None
                fields_adjust_target = None

                if (categories_source is not None) and (categories_target is not None):

                    fields_adjust = None

                    fields_adjust_source = model_attributes.build_varlist(
                        dict_cur.get("subsector"),
                        modvar,
                        restrict_to_category_values = categories_source
                    )

                    fields_adjust_target = model_attributes.build_varlist(
                        dict_cur.get("subsector"),
                        modvar,
                        restrict_to_category_values = sorted(list(categories_target.keys()))
                    )


                ##  DO MIXING

                if magnitude_type in ["transfer_value", "transfer_value_to_acheieve_magnitude"]:

                    # TRANSFER OF MAGNITUDE BETWEEN CATEGORIES

                    # get baseline values
                    arr_base_source = np.array(df_in_new[fields_adjust_source])
                    arr_base_target = np.array(df_in_new[fields_adjust_target])
                    sum_preservation = np.sum(np.array(df_in_new[fields_adjust_source + fields_adjust_target]), axis = 1)

                    # get value of target in baseline and magnitude to transfer
                    vec_target_initial = arr_base_target[tp_baseline, :]
                    total_target_initial = sum(vec_target_initial) if (magnitude_type == "transfer_value_to_acheieve_magnitude") else 0
                    magnitude_transfer = magnitude - total_target_initial

                    # get distribution to transfer--check that it does not violate bounds if specified
                    vec_source_initial = arr_base_source[tp_baseline, :]
                    vec_distribution_transfer = np.nan_to_num(vec_source_initial/sum(vec_source_initial), 0.0)
                    vec_transfer = magnitude_transfer*vec_distribution_transfer
                    vec_source_new = sf.vec_bounds(vec_source_initial - vec_transfer, bounds)
                    vec_transfer = (vec_source_initial - vec_source_new) if (max(np.abs(vec_source_new - vec_transfer)) > 0) else vec_transfer
                    magnitude_transfer = sum(vec_transfer)

                    # new target vector - note that these are both ordered properly according to category
                    vec_target = magnitude_transfer*np.array([categories_target.get(x) for x in vector_targets_ordered])
                    arr_new_source = np.outer(
                        np.ones(arr_base_source.shape[0]),
                        vec_source_new
                    )
                    arr_new_target = arr_base_target + np.outer(
                        np.ones(arr_base_target.shape[0]),
                        vec_target
                    )

                    arr_base = np.concatenate([arr_base_source, arr_base_target], axis = 1)
                    arr_final = np.concatenate([arr_new_source, arr_new_target], axis = 1)
                    fields_adjust = fields_adjust_source + fields_adjust_target

                else:

                    # CASE WITH MOST STANDARD MODIFICATIONS

                    arr_base = np.array(df_in_new[fields_adjust])
                    arr_base = sf.vec_bounds(arr_base, bounds) if (bounds is not None) else arr_base

                    # the final value depends on the magnitude type
                    if magnitude_type == "baseline_scalar":
                        arr_final = np.outer(
                            np.ones(arr_base.shape[0]),
                            magnitude * arr_base[ind_tp_baseline, :]
                        )

                    elif magnitude_type == "baseline_scalar_diff_reduction":
                        arr_final = np.outer(
                            np.ones(arr_base.shape[0]),
                            arr_base[ind_tp_baseline, :] + magnitude * (bounds[1] - arr_base[ind_tp_baseline, :])
                        )

                    elif magnitude_type == "baseline_additive":
                        arr_final = np.outer(
                            np.ones(arr_base.shape[0]),
                            arr_base[ind_tp_baseline, :]
                        ) + magnitude

                    elif magnitude_type == "final_value":
                        arr_final = magnitude * np.ones(arr_base.shape)

                # check if bounds need to be applied
                arr_final = sf.vec_bounds(arr_final, bounds) if (bounds is not None) else arr_final
                vec_ramp = sf.vec_bounds(vec_ramp, (0, 1))
                arr_transform = sf.do_array_mult(1 - vec_ramp, arr_base)
                arr_transform += sf.do_array_mult(vec_ramp, arr_final)

                # update dataframe if needed
                for fld in enumerate(fields_adjust):
                    i, fld = fld
                    df_in_new[fld] = arr_transform[:, i]

        df_out.append(df_in_new)

    # concatenate and add strategy if applicable
    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
    if isinstance(strategy_id, int):
        df_out = sf.add_data_frame_fields_from_dict(
            df_out,
            {
                model_attributes.dim_strategy_id: strategy_id
            },
            prepend_q = True,
            overwrite_fields = True
        )

    return df_out






##########################################
#    CARBON CAPTURE AND SEQUESTRATION    #
##########################################

def transformation_ccsq_increase_direct_air_capture(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase direct air capture" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of direct air capture in final time period
        * IMPORTANT: entered in MT
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """
    # get conversion units
    units = model_attributes.get_variable_characteristic(
        model_energy.modvar_ccsq_total_sequestration,
        model_attributes.varchar_str_unit_mass
    )
    scalar = model_attributes.get_mass_equivalent("mt", units)
    categories = ["direct_air_capture"]

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_ccsq_total_sequestration: {
                "bounds": (0, np.inf),
                "categories": categories,
                "magnitude": magnitude*scalar,
                "magnitude_type": "final_value",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        model_energy = model_energy,
        **kwargs
    )

    return df_out






###########################################
#    ENERGY TECHNOLOGY TRANSFORMATIONS    #
###########################################

def transformation_entc_increase_efficiency_of_electricity_production(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.ElectricEnergy,
    bounds: Tuple = (0, 0.9),
    field_region: str = "nation",
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase efficiency of electricity production" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase to apply, as additive factor, to
        energy technology efficiency factors
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: ElectricEnergy model used to define variables
    - vec_ramp: implementation ramp vector

    Keyword Arguments
    -----------------
    - bounds: optional bounds on the efficiency. Default is maximum of 90%
        efficiency
    - field_region: field in df_input that specifies the region
    - magnitude: final magnitude of generation capacity.
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # only apply to generation techs
    categories = model_attributes.get_categories_from_attribute_characteristic(
        model_attributes.subsec_name_entc,
        {
            "power_plant": 1
        }
    )

    # iterate over categories to modify output data frame -- will use to copy into new variables
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_electricity.modvar_entc_efficiency_factor_technology: {
                "bounds": bounds,
                "categories": categories,
                "magnitude": magnitude,
                "magnitude_type": "baseline_additive",
                "vec_ramp": vec_ramp,
                "time_period_baseline": get_time_period(model_attributes, "max")
            }
        },
        field_region = field_region,
        model_energy = model_energy,
        **kwargs
    )

    return df_out



def transformation_entc_increase_renewables(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.ElectricEnergy,
    field_region: str = "nation",
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase renewables" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase to apply to renewable energy minimum
        installed capacity. Entered as a scalar--for example, to double from
        existing (or planned) residual capacity, enter 2.
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: ElectricEnergy model used to define variables
    - vec_ramp: implementation ramp vector

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - magnitude: final magnitude of generation capacity.
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    attr = model_attributes.get_attribute_table(model_attributes.subsec_name_entc)

    # initialize output and categories
    df_out = df_input.copy()
    categories = model_attributes.get_categories_from_attribute_characteristic(
        model_attributes.subsec_name_entc,
        {
            "renewable_energy_technology": 1
        }
    )

    # source fields to target fields
    fields_source = model_attributes.build_varlist(
        model_attributes.subsec_name_entc,
        model_electricity.modvar_entc_nemomod_residual_capacity,
        restrict_to_category_values = categories
    )
    fields_target = model_attributes.build_varlist(
        model_attributes.subsec_name_entc,
        model_electricity.modvar_entc_nemomod_total_annual_min_capacity,
        restrict_to_category_values = categories
    )
    dict_source_to_target = dict(zip(fields_source, fields_target))

    # iterate over categories to modify output data frame -- will use to copy into new variables
    df_out_source = transformation_general(
        df_input,
        model_attributes,
        {
            model_electricity.modvar_entc_nemomod_residual_capacity: {
                "bounds": (0, np.inf),
                "categories": categories,
                "magnitude": magnitude,
                "magnitude_type": "baseline_scalar",
                "vec_ramp": vec_ramp,
                "time_period_baseline": get_time_period(model_attributes, "max")
            }
        },
        field_region = field_region,
        model_energy = model_energy,
        **kwargs
    )

    # copy into df_out (leave actual fields unchange)
    fields_ind = [field_region, model_attributes.dim_time_period]
    for k in fields_source:
        dict_update = sf.build_dict(
            df_out_source[fields_ind + [k]],
            dims = (2, 1)
        )

        new_col = sf.df_to_tuples(df_out[fields_ind])
        new_col = [dict_update.get(x) for x in new_col]
        df_out[dict_source_to_target.get(k)] = new_col

    return df_out



def transformation_entc_reduce_cost_of_renewables(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.ElectricEnergy,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce cost of renewables" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: fractional scalar of reduction applied to final period. For
        example, to reduce costs by 30% by the final time period, enter 0.3
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: ElectricEnergy model used to define variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # apply to capital, fixed, and variable for now
    modvars = [
        model_electricity.modvar_entc_nemomod_capital_cost,
        model_electricity.modvar_entc_nemomod_fixed_cost,
        model_electricity.modvar_entc_nemomod_variable_cost
    ]

    # get categories
    categories = model_attributes.get_categories_from_attribute_characteristic(
        model_attributes.subsec_name_entc,
        {
            "renewable_energy_technology": 1
        }
    )

    # setup dictionaries
    dict_base = {
        "bounds": (0, np.inf),
        "categories": categories,
        "magnitude": 1 - magnitude,
        "magnitude_type": "baseline_scalar",
        "time_period_baseline": get_time_period(model_attributes, "max"),
        "vec_ramp": vec_ramp
    }
    dict_run = dict((modvar, dict_base) for modvar in modvars)


    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        dict_run,
        model_energy = model_energy,
        **kwargs
    )

    return df_out



def transformation_entc_renewable_target(
    df_input: pd.DataFrame,
    magnitude: Union[float, str],
    cats_renewable: List[str],
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.ElectricEnergy,
    field_region: str = "nation",
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "% of electricity is generated by renewables in 2050" 
        transformation. Applies to both renewable (true renewable) and fossil
        fuel (fake renewable) transformations

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of target to hit by 2050   OR   optional str 
        "VEC_FIRST_RAMP", which will set the magnitude to the mix of renewable
        capacity at the first time period where VEC_FIRST_RAMP != 0
    - cats_renewable: technology categories to use for renewable energy
        generation. Must be present in the $CAT-TECHNOLOGY$ attribute table
        in model_attributes
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: ElectricEnergy model used to define variables
    - vec_ramp: implementation ramp vector

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - magnitude: final magnitude of generation capacity.
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    attr = model_attributes.get_attribute_table(model_attributes.subsec_name_entc)
    cats_renewable = [x for x in attr.key_values if x in cats_renewable]
    inds_renewable = [attr.get_key_value_index(x) for x in cats_renewable]

    dfs = df_input.groupby(field_region)
    df_out = []

    for df in dfs:
        
        tup, df = df
        # setup renewable specification
        for cat in attr.key_values:
            var_names = model_attributes.build_varlist(
                model_attributes.subsec_name_entc, 
                model_electricity.modvar_entc_nemomod_renewable_tag_technology,
                restrict_to_category_values = [cat]
            )

            if var_names is not None:
                if len(var_names) > 0:
                    val = 1 if (cat in cats_renewable) else 0
                    df[var_names[0]] = val
        

        # setup magnitude of change
        if magnitude == "VEC_FIRST_RAMP":
            w = np.where(np.array(vec_ramp) != 0)[0][0]

            arr_entc_residual_capacity = model_attributes.get_standard_variables(
                df,
                model_electricity.modvar_entc_nemomod_residual_capacity,
                expand_to_all_cats = True,
                return_type = "array_base"
            )

            magnitude = arr_entc_residual_capacity[w, inds_renewable].sum()/arr_entc_residual_capacity[w, :].sum()

        # iterate over categories to modify output data frame -- will use to copy into new variables
        df_transformed = transformation_general(
            df,
            model_attributes,#HEREHERER 
            {
                model_electricity.modvar_enfu_nemomod_renewable_production_target: {
                    "bounds": (0, 1),
                    "categories": [model_electricity.cat_enfu_elec],
                    "magnitude": magnitude,
                    "magnitude_type": "final_value",
                    "vec_ramp": vec_ramp,
                    "time_period_baseline": get_time_period(model_attributes, "max")
                }
            },
            field_region = field_region,
            model_energy = model_energy,
            **kwargs
        )

        df_out.append(df_transformed)
    
    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)

    return df_out



def transformation_entc_specify_transmission_losses(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.ElectricEnergy,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce transmission losses" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of transmission loss in final time period. For
        example, if the final value is 5%, enter 0.05.
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: ElectricEnergy model used to define variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_electricity.modvar_enfu_transmission_loss_electricity: {
                "bounds": (0, 1),
                "magnitude": magnitude,
                "magnitude_type": "final_value",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        model_energy = model_energy,
        **kwargs
    )

    return df_out



def transformation_entc_retire_fossil_fuel_early(
    df_input: pd.DataFrame,
    dict_categories_to_vec_ramp: Dict[str, np.ndarray],
    model_attributes: ma.ModelAttributes,
    model_electricity: ml.ElectricEnergy,
    magnitude: float = 0.0,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Maximize Industrial Production Efficiency" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - dict_categories_to_vec_ramp: dictionary mapping categories to ramp vector
        to use for retirement
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_electricity: ElectricEnergy model used to define variables

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - magnitude: final magnitude of generation capacity.
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    attr = model_attributes.get_attribute_table(model_attributes.subsec_name_entc)

    # initialize output
    df_out = df_input.copy()

    # iterate over categories to modify output data frame
    for cat in dict_categories_to_vec_ramp.keys():
        if cat in attr.key_values:
            vec_ramp = dict_categories_to_vec_ramp.get(cat)

            if isinstance(vec_ramp, np.ndarray):
                df_out = transformation_general(
                    df_out,
                    model_attributes,
                    {
                        model_electricity.modvar_entc_nemomod_residual_capacity: {
                            "bounds": (0, np.inf),
                            "categories": [cat],
                            "magnitude": 0.0,
                            "magnitude_type": "final_value",
                            "vec_ramp": vec_ramp,
                            "time_period_baseline": get_time_period(model_attributes, "max")
                        }
                    },
                    model_energy = model_energy,
                    **kwargs
                )
    return df_out




############################################
#    FUGITIVE EMISSIONS TRANSFORMATIONS    #
############################################

def transformation_fgtv_maximize_flaring(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: me.NonElectricEnergy,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Maximize Flaring" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in efficiency relative to time 0
        (interpreted as a magnitude change, not a scalar change).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_fgtv_frac_non_fugitive_flared: {
                "bounds": (0, 1),
                "magnitude": magnitude,
                "magnitude_type": "final_value",
                "vec_ramp": vec_ramp
            }
        },
        model_energy = model_energy,
        **kwargs
    )

    return df_out



def transformation_fgtv_reduce_leaks(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce Leaks" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: fractional magnitude of reduction in leaks relative to time 0
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_fgtv_frac_reduction_fugitive_leaks: {
                "bounds": (0, 1),
                "magnitude": magnitude,
                "magnitude_type": "baseline_scalar_diff_reduction",
                "vec_ramp": vec_ramp
            }
        },
        model_energy = model_energy,
        **kwargs
    )
    return df_out






###########################################
#    INDUSTRIAL ENERGY TRANSFORMATIONS    #
###########################################

def transformation_inen_maximize_energy_efficiency(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Maximize Industrial Energy Efficiency" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in efficiency relative to the final time
        period.
            * Interpreted as a magnitude increase in the industrial efficiency
                factor--e.g., to increase the efficiency factor by 0.3 by the
                final time period, enter 0.3.
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_enfu_efficiency_factor_industrial_energy: {
                "bounds": (0, 1),
                "magnitude": magnitude,
                "magnitude_type": "baseline_additive",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        model_energy = model_energy,
        **kwargs
    )
    return df_out



def transformation_inen_maximize_production_efficiency(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Maximize Industrial Production Efficiency" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: proportional reduction in demand relative to final time period
        (to reduce production energy demand by 30% by the final time period,
        enter magnitude = 0.3).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """


    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_inen_demscalar: {
                "bounds": (0, 1),
                "magnitude": 1 - magnitude,
                "magnitude_type": "final_value",
                "vec_ramp": vec_ramp
            }
        },
        model_energy = model_energy,
        **kwargs
    )
    return df_out



def transformation_inen_shift_modvars(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    dict_modvar_specs: Union[Dict[str, float], None] = None,
    categories: Union[List[str], None] = None,
    regions_apply: Union[List[str], None] = None,
    field_region = "nation",
    model_energy: Union[me.NonElectricEnergy, None] = None,
    strategy_id: Union[int, None] = None
) -> pd.DataFrame:
    """
    Implement fuel switch transformations

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: target magnitude of fuel mixture
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - categories: INEN categories to apply transformation to
    - dict_modvar_specs: dictionary of targets modvars to shift into (assumes
        that will take from others). Maps from modvar to fraction of magnitude.
        Sum of values must == 1.
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # core vars (ordered)
    model_energy = me.NonElectricEnergy(model_attributes) if not isinstance(model_energy, me.NonElectricEnergy) else model_energy
    all_regions = sorted(list(set(df_input[field_region])))
    # dertivative vars (alphabetical)
    attr_inen = model_attributes.get_attribute_table("Industrial Energy")
    attr_time_period = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_time_period}")
    df_out = []
    regions_apply = all_regions if (regions_apply is None) else [x for x in regions_apply if x in all_regions]

    # model variables to explore
    modvars = [
        model_energy.modvar_inen_frac_en_coal,
        model_energy.modvar_inen_frac_en_coke,
        model_energy.modvar_inen_frac_en_diesel,
        model_energy.modvar_inen_frac_en_electricity,
        model_energy.modvar_inen_frac_en_furnace_gas,
        model_energy.modvar_inen_frac_en_gasoline,
        model_energy.modvar_inen_frac_en_hydrogen,
        model_energy.modvar_inen_frac_en_kerosene,
        model_energy.modvar_inen_frac_en_natural_gas,
        model_energy.modvar_inen_frac_en_oil,
        model_energy.modvar_inen_frac_en_pliqgas,
        model_energy.modvar_inen_frac_en_solar,
        model_energy.modvar_inen_frac_en_solid_biomass
    ]

    dict_modvar_specs_def = {model_energy.modvar_inen_frac_en_electricity: 1}
    dict_modvar_specs = dict_modvar_specs_def if not isinstance(dict_modvar_specs, dict) else dict_modvar_specs
    dict_modvar_specs = dict((k, v) for k, v in dict_modvar_specs.items() if (k in modvars) and (isinstance(v, int) or isinstance(v, float)))
    dict_modvar_specs = dict_modvar_specs_def if (sum(list(dict_modvar_specs.values())) != 1.0) else dict_modvar_specs

    modvars_source = [x for x in modvars if x not in dict_modvar_specs.keys()]
    modvars_target = [x for x in modvars if x in dict_modvar_specs.keys()]
    cats_all = [set(model_attributes.get_variable_categories(x)) for x in modvars_target]
    cats_all = set.intersection(*cats_all)

    subsec = model_attributes.subsec_name_inen

    ##  ITERATE OVER REGIONS AND MODVARS TO BUILD TRANSFORMATION

    for region in all_regions:

        df_in = df_input[df_input[field_region] == region].sort_values(by = [model_attributes.dim_time_period]).reset_index(drop = True)
        df_in_new = df_in.copy()
        vec_tp = list(df_in[model_attributes.dim_time_period])
        n_tp = len(df_in)

        if region in regions_apply:
            for cat in cats_all:

                target_value = magnitude#*dict_modvar_specs.get(modvar_target)
                scale_non_elec = 1 - target_value
                fields = [
                    model_attributes.build_varlist(
                        subsec,
                        x,
                        restrict_to_category_values = [cat]
                    )[0] for x in modvars_target
                ]

                target_distribution = magnitude*np.array([dict_modvar_specs.get(x) for x in modvars_target])
                vec_final_vals = np.array(df_in[fields].iloc[n_tp - 1]).astype(float)
                val_final_target = sum(vec_final_vals)

                # get model variables that need to be adjusted
                modvars_adjust = []
                for modvar in modvars:
                    modvars_adjust.append(modvar) if cat in model_attributes.get_variable_categories(modvar) else None

                # loop over adjustment variables to build new trajectories
                for modvar in modvars_adjust:
                    field_cur = model_attributes.build_varlist(
                        subsec,
                        modvar,
                        restrict_to_category_values = [cat]
                    )[0]
                    vec_old = np.array(df_in[field_cur])
                    val_final = vec_old[n_tp - 1]
                    val_new = (val_final/(1 - val_final_target))*scale_non_elec if (modvar not in modvars_target) else magnitude*dict_modvar_specs.get(modvar)
                    vec_new = vec_ramp*val_new + (1 - vec_ramp)*vec_old

                    df_in_new[field_cur] = vec_new

        df_out.append(df_in_new)


    # concatenate and add strategy if applicable
    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
    if isinstance(strategy_id, int):
        df_out = sf.add_data_frame_fields_from_dict(
            df_out,
            {
                model_attributes.dim_strategy_id: strategy_id
            },
            prepend_q = True,
            overwrite_fields = True
        )

    return df_out





##############################
#    SCOE TRANSFORMATIONS    #
##############################

def transformation_scoe_electrify_category_to_target(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    cats_elec: Union[List[str], None] = None,
    regions_apply: Union[List[str], None] = None,
    field_region = "nation",
    model_energy: Union[me.NonElectricEnergy, None] = None,
    strategy_id: Union[int, None] = None
) -> pd.DataFrame:
    """
    Implement the "Switch to electricity for heat" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of final proportion of heat energy that is
        electrified for each category
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # core vars (ordered)
    model_energy = me.NonElectricEnergy(model_attributes) if not isinstance(model_energy, me.NonElectricEnergy) else model_energy
    all_regions = sorted(list(set(df_input[field_region])))
    # dertivative vars (alphabetical)
    attr_time_period = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_time_period}")
    df_out = []
    regions_apply = all_regions if (regions_apply is None) else [x for x in regions_apply if x in all_regions]
    subsec = model_attributes.subsec_name_scoe

    # model variables to explore
    modvars = [
        model_energy.modvar_scoe_frac_heat_en_coal,
        model_energy.modvar_scoe_frac_heat_en_diesel,
        model_energy.modvar_scoe_frac_heat_en_electricity,
        model_energy.modvar_scoe_frac_heat_en_gasoline,
        model_energy.modvar_scoe_frac_heat_en_hydrogen,
        model_energy.modvar_scoe_frac_heat_en_kerosene,
        model_energy.modvar_scoe_frac_heat_en_natural_gas,
        model_energy.modvar_scoe_frac_heat_en_pliqgas,
        model_energy.modvar_scoe_frac_heat_en_solid_biomass
    ]


    ##  ITERATE OVER REGIONS AND MODVARS TO BUILD TRANSFORMATION

    for region in all_regions:

        df_in = df_input[df_input[field_region] == region].sort_values(by = [model_attributes.dim_time_period]).reset_index(drop = True)
        df_in_new = df_in.copy()
        vec_tp = list(df_in[model_attributes.dim_time_period])
        n_tp = len(df_in)

        # get electric categories and build dictionary of target values
        cats_elec_all = model_attributes.get_variable_categories(model_energy.modvar_scoe_frac_heat_en_electricity)
        cats_elec = [x for x in cats_elec_all if x in cats_elec] if isinstance(cats_elec, list) else cats_elec_all
        dict_targets_final_tp = dict((x, magnitude) for x in cats_elec)


        if region in regions_apply:
            for cat in dict_targets_final_tp.keys():

                field_elec = model_attributes.build_varlist(
                    subsec,
                    model_energy.modvar_scoe_frac_heat_en_electricity,
                    restrict_to_category_values = [cat]
                )[0]

                val_final_elec = float(df_in[field_elec].iloc[n_tp - 1])
                target_value = min(max(dict_targets_final_tp.get(cat) + val_final_elec, 0), 1)
                scale_non_elec = 1 - target_value

                # get model variables that need to be adjusted
                modvars_adjust = []
                for modvar in modvars:
                    modvars_adjust.append(modvar) if cat in model_attributes.get_variable_categories(modvar) else None

                # loop over adjustment variables to build new trajectories
                for modvar in modvars_adjust:
                    field_cur = model_attributes.build_varlist(
                        subsec,
                        modvar,
                        restrict_to_category_values = [cat]
                    )[0]
                    vec_old = np.array(df_in[field_cur])
                    val_final = vec_old[n_tp - 1]
                    val_new = (val_final/(1 - val_final_elec))*scale_non_elec if (field_cur != field_elec) else target_value
                    vec_new = vec_ramp*val_new + (1 - vec_ramp)*vec_old

                    df_in_new[field_cur] = vec_new

        df_out.append(df_in_new)


    # concatenate and add strategy if applicable
    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
    if isinstance(strategy_id, int):
        df_out = sf.add_data_frame_fields_from_dict(
            df_out,
            {
                model_attributes.dim_strategy_id: strategy_id
            },
            prepend_q = True,
            overwrite_fields = True
        )

    return df_out



def transformation_scoe_increase_energy_efficiency_heat(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase efficiency of fuel for heat" transformation in SCOE

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in efficiency relative to the final time
        period (interpreted as an additive change to the baseline value--e.g.,
        an increase of 0.25 in the efficiency factor relative to the final time
        period is entered as 0.25).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    dict_base = {
        "bounds": (0, 1),
        "magnitude": magnitude,
        "magnitude_type": "baseline_additive",
        "time_period_baseline": get_time_period(model_attributes, "max"),
        "vec_ramp": vec_ramp
    }

    modvars = [
        model_energy.modvar_scoe_efficiency_fact_heat_en_coal,
        model_energy.modvar_scoe_efficiency_fact_heat_en_diesel,
        #model_energy.modvar_scoe_efficiency_fact_heat_en_electricity,
        model_energy.modvar_scoe_efficiency_fact_heat_en_gasoline,
        model_energy.modvar_scoe_efficiency_fact_heat_en_hydrogen,
        model_energy.modvar_scoe_efficiency_fact_heat_en_kerosene,
        model_energy.modvar_scoe_efficiency_fact_heat_en_natural_gas,
        model_energy.modvar_scoe_efficiency_fact_heat_en_pliqgas,
        model_energy.modvar_scoe_efficiency_fact_heat_en_solid_biomass
    ]

    dict_run = dict((modvar, dict_base) for modvar in modvars)


    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        dict_run,
        model_energy = model_energy,
        **kwargs
    )

    return df_out



def transformation_scoe_reduce_demand_for_appliance_energy(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase appliance efficiency" transformation in SCOE

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of reduction in electric energy demand relative to 
        final time period (interpreted as an proportional scalar--e.g., a 30% 
        retuction in  electric energy demand in the final time period is entered 
        as 0.3).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_scoe_demscalar_elec_energy_demand : {
                "bounds": (0, np.inf),
                "magnitude": float(sf.vec_bounds(1 - magnitude, (0, np.inf))),
                "magnitude_type": "baseline_scalar",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }

        },
        model_energy = model_energy,
        **kwargs
    )

    return df_out



def transformation_scoe_reduce_demand_for_heat_energy(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Reduce demand for heat energy" transformation in SCOE

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of reduction in heat energy demand relative final
        time period (interpreted as an proportional scalar--e.g.,
        an 30% in heat energy demand in the final time period is entered as
        0.3).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_scoe_demscalar_heat_energy_demand : {
                "bounds": (0, np.inf),
                "magnitude": float(sf.vec_bounds(1 - magnitude, (0, np.inf))),
                "magnitude_type": "baseline_scalar",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }

        },
        model_energy = model_energy,
        **kwargs
    )

    return df_out





###############################################
#    TRANSPORTATION DEMAND TRANSFORMATIONS    #
###############################################

def transformation_trde_reduce_demand(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Transportation Non-Electricity Energy Efficiency"
        transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of reduction in demand relative to the final time
        period (interprted as a scalar change; i.e., enter 0.25 to reduce demand
        by 25% by the final time period).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_trde_demand_scalar: {
                "magnitude": 1 - magnitude,
                "magnitude_type": "baseline_scalar",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        model_energy = model_energy,
        **kwargs
    )

    return df_out



########################################
#    TRANSPORTATION TRANSFORMATIONS    #
########################################

def transformation_trns_electrify_category_to_target(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    cats_elec: Union[List[str], None] = None,
    regions_apply: Union[List[str], None] = None,
    field_region = "nation",
    model_energy: Union[me.NonElectricEnergy, None] = None,
    strategy_id: Union[int, None] = None
) -> pd.DataFrame:
    """
    Implement the "Electrify light duty road transport" transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of final proportion of light duty transport that is
        electrified.
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    # core vars (ordered)
    model_energy = me.NonElectricEnergy(model_attributes) if not isinstance(model_energy, me.NonElectricEnergy) else model_energy
    all_regions = sorted(list(set(df_input[field_region])))
    # dertivative vars (alphabetical)
    attr_time_period = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_time_period}")
    df_out = []
    regions_apply = all_regions if (regions_apply is None) else [x for x in regions_apply if x in all_regions]

    # model variables to explore
    modvars = [
        model_energy.modvar_trns_fuel_fraction_biofuels,
        model_energy.modvar_trns_fuel_fraction_diesel,
        model_energy.modvar_trns_fuel_fraction_electricity,
        model_energy.modvar_trns_fuel_fraction_gasoline,
        model_energy.modvar_trns_fuel_fraction_hydrogen,
        model_energy.modvar_trns_fuel_fraction_kerosene,
        model_energy.modvar_trns_fuel_fraction_natural_gas
    ]


    ##  ITERATE OVER REGIONS AND MODVARS TO BUILD TRANSFORMATION

    for region in all_regions:

        df_in = df_input[df_input[field_region] == region].sort_values(by = [model_attributes.dim_time_period]).reset_index(drop = True)
        df_in_new = df_in.copy()
        vec_tp = list(df_in[model_attributes.dim_time_period])
        n_tp = len(df_in)

        # get electric categories and build dictionary of target values
        cats_elec_all = model_attributes.get_variable_categories(model_energy.modvar_trns_fuel_fraction_electricity)
        cats_elec = [x for x in cats_elec_all if x in cats_elec] if isinstance(cats_elec, list) else cats_elec_all
        dict_targets_final_tp = dict((x, magnitude) for x in cats_elec)


        if region in regions_apply:
            for cat in dict_targets_final_tp.keys():

                target_value = dict_targets_final_tp.get(cat)
                scale_non_elec = 1 - target_value
                field_elec = model_attributes.build_varlist(
                    model_attributes.subsec_name_trns,
                    model_energy.modvar_trns_fuel_fraction_electricity,
                    restrict_to_category_values = [cat]
                )[0]

                val_final_elec = float(df_in[field_elec].iloc[n_tp - 1])

                # get model variables that need to be adjusted
                modvars_adjust = []
                for modvar in modvars:
                    modvars_adjust.append(modvar) if cat in model_attributes.get_variable_categories(modvar) else None

                # loop over adjustment variables to build new trajectories
                for modvar in modvars_adjust:
                    field_cur = model_attributes.build_varlist(
                        model_attributes.subsec_name_trns,
                        modvar,
                        restrict_to_category_values = [cat]
                    )[0]
                    vec_old = np.array(df_in[field_cur])
                    val_final = vec_old[n_tp - 1]
                    val_new = (val_final/(1 - val_final_elec))*scale_non_elec if (field_cur != field_elec) else target_value
                    vec_new = vec_ramp*val_new + (1 - vec_ramp)*vec_old

                    df_in_new[field_cur] = vec_new

        df_out.append(df_in_new)


    # concatenate and add strategy if applicable
    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
    if isinstance(strategy_id, int):
        df_out = sf.add_data_frame_fields_from_dict(
            df_out,
            {
                model_attributes.dim_strategy_id: strategy_id
            },
            prepend_q = True,
            overwrite_fields = True
        )

    return df_out



def transformation_trns_increase_energy_efficiency_electric(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Transportation Non-Electricity Energy Efficiency"
        transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in efficiency relative to the final time
        period (interpreted as a scalar change to the baseline value--e.g., a
        25% increase relative to the final time period value is entered as
        0.25).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """


    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_trns_electrical_efficiency: {
                "magnitude": 1 + magnitude,
                "magnitude_type": "baseline_scalar",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        model_energy = model_energy,
        **kwargs
    )
    return df_out



def transformation_trns_increase_energy_efficiency_non_electric(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Transportation Non-Electricity Energy Efficiency"
        transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in efficiency relative to the final time
        period (interpreted as a scalar change to the baseline value--e.g., a
        25% increase relative to the final time period value is entered as
        0.25).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """

    dict_base = {
        "magnitude": 1 + magnitude,
        "magnitude_type": "baseline_scalar",
        "time_period_baseline": get_time_period(model_attributes, "max"),
        "vec_ramp": vec_ramp
    }

    modvars = [
        model_energy.modvar_trns_fuel_efficiency_biofuels,
        model_energy.modvar_trns_fuel_efficiency_diesel,
        model_energy.modvar_trns_fuel_efficiency_gasoline,
        model_energy.modvar_trns_fuel_efficiency_hydrogen,
        model_energy.modvar_trns_fuel_efficiency_kerosene,
        model_energy.modvar_trns_fuel_efficiency_natural_gas
    ]

    dict_run = dict((modvar, dict_base) for modvar in modvars)

    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        dict_run,
        model_energy = model_energy,
        **kwargs
    )

    return df_out



def transformation_trns_increase_vehicle_occupancy(
    df_input: pd.DataFrame,
    magnitude: float,
    vec_ramp: np.ndarray,
    model_attributes: ma.ModelAttributes,
    categories: List[str] = ["road_light"],
    model_energy: Union[me.NonElectricEnergy, None] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Implement the "Increase Transportation Non-Electricity Energy Efficiency"
        transformation

    Function Arguments
    ------------------
    - df_input: input data frame containing baseline trajectories
    - magnitude: magnitude of increase in vehicle occupancy for private light
        vehicles relative to final time period (interpreted as a scalar change
        to the baseline value--e.g., a 25% increase relative to the final time
        period value is entered as 0.25).
    - model_attributes: ModelAttributes object used to call strategies/variables
    - vec_ramp: ramp vec used for implementation

    Keyword Arguments
    -----------------
    - field_region: field in df_input that specifies the region
    - model_energy: optional NonElectricEnergy object to pass to
        transformation_general
    - regions_apply: optional set of regions to use to define strategy. If None,
        applies to all regions.
    - strategy_id: optional specification of strategy id to add to output
        dataframe (only added if integer)
    """


    # call general transformation
    df_out = transformation_general(
        df_input,
        model_attributes,
        {
            model_energy.modvar_trns_average_passenger_occupancy: {
                "categories": categories,
                "magnitude": 1 + magnitude,
                "magnitude_type": "baseline_scalar",
                "time_period_baseline": get_time_period(model_attributes, "max"),
                "vec_ramp": vec_ramp
            }
        },
        model_energy = model_energy,
        **kwargs
    )
    return df_out
