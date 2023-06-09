
def transformation_support_lndu_check_pasture_magnitude(
    magnitude: float,
    pasture_fraction: float, 
    magnitude_type: str,
    model_afolu: mafl.AFOLU,
    max_change_allocated_to_pasture_frac_adjustment: float = 0.0
) -> Tuple[float, float]:
    """
    Some transformations may operate on pastures rather than grasslands as a
        whole--use this function to express pasture fractions in terms of 
        grassland fractions.  

    Returns a tuple of the form

        (magnitude_grassland, pasture_fraction)

        where `magnitude_grassland` is the magnitude applied to grasslands and
        `pasture_fraction` is the pasture fraction. If `pasture_fraction` is 
        None, then no change occurs.

    Function Arguments
    ------------------
    - magnitude: magnitude of tranformation 
    - pasture_fraction: fraction of grassland that is used as pasture
    - magntitude_type: valid type of magnitude, used to modify calculation of 
        fraction
    - model_afolu: AFOLU class used to access model attributes, properties, and
        methods
    - max_change_allocated_to_pasture_frac_adjustment: maximum allowable 
        fraction of changes that can be allocated to the pasture fraction 
        adjustments (e.g., silvopasture might rely on shifting existing 
        pastures to secondary forests rather than grassland as a whole)
    """
    model_attributes = model_afolu.model_attributes
    attr_lndu = model_attributes.get_attribute_table(
        model_attributes.subsec_name_lndu
    )

    # temporar
    area_grassland = 1
    area_pasture = pasture_fraction*area_grassland
    area_grassland_no_pasture = area_grassland - pasture_fraction


    ##  START BY GETTING TARGET AREA

    if magnitude_type == "baseline_scalar":
        area_pasture_new = area_pasture*magnitude
    elif magnitude_type == "final_value":
        area_pasture_new = magnitude
    elif magnitude_type == "final_value_ceiling":
        area_pasture_new = min(magnitude, area_pasture)
    elif magnitude_type == "final_value_floor":
        area_pasture_new = max(magnitude, area_pasture)

    

    ##  MODIFY BASED ON max_increase_in_pasture_fraction

    area_pasture_delta = area_pasture_new - area_pasture
    area_as_change_in_pasture_frac = float(
        sf.vec_bounds(
            area_pasture_delta*max_change_allocated_to_pasture_frac_adjustment,
            (-area_pasture, area_grassland_no_pasture)
        )
    )

    area_grassland_no_pasture + area_pasture_new






    



def transformation_support_lndu_check_ltct_magnitude_dictionary(
    dict_magnitudes: Dict[str, Dict[str, Any]],
    model_attributes: ma.ModelAttributes,
    model_afolu: Union[mafl.AFOLU, None] = None,
) -> Union[None]:
    """
    Support function for 
        transformation_support_lndu_transition_to_category_targets_single_region
    
    Checks to verify that dict_magnitudes is properly specified. 

    Function Arguments
    ------------------
    - dict_magnitudes: dictionary mapping land use categories to fraction 
        information. Should take the following form:

        {
            category: {
                "magnitude_type": magnitude_type,
                "magnitude": value,
                "categories_target": {
                    "cat_target_0": prop_magnitude_0,
                    "cat_target_1": prop_magnitude_1,
                    ...
                } 
                # NOTE: 
                # key "categories_target" REQUIRED only if 
                #   magnitude_type == "transfer_value_scalar"
            }
        }
    - model_attributes: ModelAttributes object used to call strategies/
        variables

    Keyword Arguments
    -----------------
    - model_afolu: optional AFOLU object to pass for variable access
    """

    model_afolu = (
        mafl.AFOLU(model_attributes)
        if model_afolu is None
        else model_afolu
    )

    attr_lndu = model_attributes.get_attribute_table(
        model_attributes.subsec_name_lndu
    )

    # valid specifications of magnitude type
    magnitude_types_valid = [
        "baseline_scalar",
        "final_value",
        "final_value_ceiling",
        "final_value_floor",
        "transfer_value",
        "transfer_value_scalar",
    ]

    if magnitude_types_valid not in magnitude_types_valid:
        return None


    ##  CHECK SPECIFICATION DICTIONARY

    cats_all = attr_lndu.key_values
    dict_magnitude_cleaned = {}

    for cat in cats_all:
        # default verified to true; set to false if any condition is not met
        verified_cat = True
        dict_magnitude_cur = dict_magnitude.get(cat)
        
        # check magnitude type
        magnitude_type = dict_magnitude_cur.get("magnitude_type")
        verified_cat &= (magnitude_type in magnitude_types_valid)
        
        # check magnitude
        magnitude = dict_magnitude_cur.get("magnitude")
        verified_cat &= sf.isnumber(magnitude)

        # check for source/target categories
        categories_source = dict_magnitude_cur.get("categories_source")
        categories_source = None if not isinstance(categories_source, list) else categories_source
        categories_target = dict_magnitude_cur.get("categories_target")
        categories_target = (
            dict((k, v) for k, v in categories_target.items() if sf.isnumber(v)) 
            if isinstance(categories_target, dict) 
            else None
        )
        # check special case
        verified_cat = (
            ((categories_source is not None) & verified_cat) 
            if (magnitude_type in ["transfer_value", "transfer_value_to_acheieve_magnitude"]) 
            else verified_cat
        )
        verified_cat = (
            ((categories_target is not None) & verified_cat) 
            if (magnitude_type in ["transfer_value", "transfer_value_to_acheieve_magnitude"]) 
            else verified_cat
        )

        # check for time period as baseline
        tp_baseline = dict_magnitude_cur.get("time_period_baseline")
        tp_baseline = max(attr_time_period.key_values) if (tp_baseline not in attr_time_period.key_values) else tp_baseline
