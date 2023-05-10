"""
Use this file to build functions and methods that can generate metadata and/or
    tables/other information (including figures) based on the Model Attributes
    module.
"""

import model_attributes as ma
import numpy as np
import os, os.path
import pandas as pd
import support_classes as sc
import support_functions as sf




def build_emissions_information_table(
    model_attributes: ma.ModelAttributes,
    field_out_categories: str = "category_value",
    field_out_category_primary_name: str = "category_name",
    field_out_field_emission: str = "field",
    field_out_field_subsector_total: str = "subsector_total_field",
    field_out_gas: str = "gas",
    field_out_gas_name: str = "gas_name",
    field_out_info: str = "model_variable_information",
    field_out_model_variable: str = "model_variable",
    field_out_sector: str = "sector",
    field_out_subsector: str = "subsector",
) -> pd.DataFrame:
    """
    Build a data frame with rows giving gasses, gas names, model variables, 
        subsector, sector, and subsector field totals.

    Function Arguments
    ------------------
    - model_attributes: model_attributes.ModelAttributes object used to generate
        and manage variables
    """
    attr_gas = model_attributes.dict_attributes.get("emission_gas")
    dict_gas_to_name = attr_gas.field_maps.get(f"{attr_gas.key}_to_name")
    dict_gas_to_emision_modvars = model_attributes.dict_gas_to_total_emission_modvars
    
    
    df_out = []
    
    for gas in dict_gas_to_emision_modvars.keys():
        
        gas_name = dict_gas_to_name.get(gas)
        modvars = dict_gas_to_emision_modvars.get(gas)
        
        # loop over available modvars
        for modvar in modvars:
            
            # get some attributes 
            subsec = model_attributes.get_variable_subsector(modvar)
            field_subsector_total = model_attributes.get_subsector_emission_total_field(subsec)
            sector = model_attributes.get_subsector_attribute(subsec, "sector")
            pycat_primary = model_attributes.get_subsector_attribute(subsec, "pycategory_primary")

            # fields and categories
            fields = model_attributes.build_varlist(None, modvar)
            cats = model_attributes.get_variable_categories(modvar)
            cats = [""] if (cats is None) else cats
            pycats_primary = [
                ("" if (x == "") else pycat_primary)
                for x in cats
            ]
            # attempt a description 
            info = model_attributes.get_variable_attribute(modvar, "information")
            info = "" if not isinstance(info, str) else info

            # build current component
            df_cur = pd.DataFrame({
                field_out_field_emission: fields,
                field_out_categories: cats,
                field_out_category_primary_name: pycats_primary,
            })
            df_cur[field_out_field_subsector_total] = field_subsector_total
            df_cur[field_out_model_variable] = modvar
            df_cur[field_out_gas] = gas
            df_cur[field_out_gas_name] = gas_name
            df_cur[field_out_info] = info
            df_cur[field_out_sector] = sector
            df_cur[field_out_subsector] = subsec
            
            df_out.append(df_cur)
    
    # set ordererd output fields
    fields_ord = [
        field_out_sector,
        field_out_subsector,
        field_out_field_emission,
        field_out_model_variable,
        field_out_categories,
        field_out_category_primary_name,
        field_out_field_subsector_total,
        field_out_gas,
        field_out_gas_name,
        field_out_field_subsector_total,
        field_out_info,
    ]
    
    df_out = (
        pd.concat(df_out, axis = 0)
        .sort_values(by = fields_ord)
        .reset_index(drop = True)
    )
    df_out = df_out[fields_ord]
    
    return df_out