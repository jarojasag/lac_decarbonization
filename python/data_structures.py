import os, os.path
import numpy as np
import pandas as pd
import support_functions as sf
import warnings

##  the AttributeTable class checks existence, keys, key values, and generates field maps
class AttributeTable:

    def __init__(self, fp_table: str, key: str, fields_to_dict: list, clean_table_fields: bool = True):

        # verify table exists and check keys
        table = pd.read_csv(sf.check_path(fp_table, False), skipinitialspace = True)
        fields_to_dict = [x for x in fields_to_dict if x != key]

        # clean the fields in the attribute table?
        dict_fields_clean_to_fields_orig = {}
        if clean_table_fields:
            fields_orig = list(table.columns)
            dict_fields_clean_to_fields_orig = dict(zip(sf.clean_field_names(fields_orig), fields_orig))
            table = sf.clean_field_names(table)
            fields_to_dict = sf.clean_field_names(fields_to_dict)
            key = sf.clean_field_names([key])[0]


        # add a key if not specified
        if not key in table.columns:
            print(f"Key {key} not found in table '{fp_table}''. Adding integer key.")
            table[key] = range(len(table))
        # check all fields
        sf.check_fields(table, [key] + fields_to_dict)
        # check key
        if len(set(table[key])) < len(table):
            raise ValueError(f"Invalid key {key} found in '{fp_table}': the key is not unique. Check the table and specify a unique key.")


        # if no fields for the dictionary are specified, default to all
        if len(fields_to_dict) == 0:
            fields_to_dict = [x for x in table.columns if (x != key)]

        # clear RST formatting in the table if applicable
        if table[key].dtype in [object, str]:
            table[key] = np.array([sf.str_replace(str(x), {"`": "", "\$": ""}) for x in list(table[key])]).astype(str)
        # set all keys
        key_values = list(table[key])
        key_values.sort()

        # next, create dict maps
        field_maps = {}
        for fld in fields_to_dict:
            field_fwd = f"{key}_to_{fld}"
            field_rev = f"{fld}_to_{key}"

            field_maps.update({field_fwd: sf.build_dict(table[[key, fld]])})
            # check for 1:1 correspondence before adding reverse
            vals_unique = set(table[fld])
            if (len(vals_unique) == len(table)):
                field_maps.update({field_rev: sf.build_dict(table[[fld, key]])})

        self.dict_fields_clean_to_fields_orig = dict_fields_clean_to_fields_orig
        self.field_maps = field_maps
        self.fp_table = fp_table
        self.key = key
        self.key_values = key_values
        self.n_key_values = len(key_values)
        self.table = table

    # function for the getting the index of a key value
    def get_key_value_index(self, key_value):
        if key_value not in self.key_values:
            raise KeyError(f"Error: invalid AttributeTable key value {key_value}.")
        return self.key_values.index(key_value)



class ModelAttributes:

    def __init__(self, dir_attributes: str, dict_config: dict = {}):

        # initialize dimensions of analysis - later, check for presence
        self.dim_time_period = "time_period"
        self.dim_design_id = "design_id"
        self.dim_future_id = "future_id"
        self.dim_strategy_id = "strategy_id"
        self.dim_primary_id = "primary_id"
        self.dimensions_of_analysis = [self.dim_time_period, self.dim_design_id, self.dim_future_id, self.dim_strategy_id, self.dim_primary_id]

        # set some basic properties
        self.attribute_file_extension = ".csv"
        self.matchstring_landuse_to_forests = "forests_"
        self.substr_analytical_parameters = "analytical_parameters"
        self.substr_dimensions = "attribute_dim_"
        self.substr_categories = "attribute_"
        self.substr_varreqs = "table_varreqs_by_"
        self.substr_varreqs_allcats = f"{self.substr_varreqs}category_"
        self.substr_varreqs_partialcats = f"{self.substr_varreqs}partial_category_"

        # temporary - but read from config
        self.varchar_str_emission_gas = "$EMISSION-GAS$"
        self.varchar_str_unit_mass = "$UNIT-MASS$"

        # add attributes and dimensional information
        self.attribute_directory = dir_attributes
        self.all_categories, self.all_dims, self.all_attributes, self.configuration_requirements, self.dict_attributes, self.dict_varreqs = self.get_attribute_tables(dir_attributes)
        self.all_sectors, self.all_sectors_abvs, self.all_subsectors, self.all_subsector_abvs = self.get_sector_dims()
        self.all_subsectors_with_primary_category, self.all_subsectors_without_primary_category = self.get_all_subsectors_with_primary_category()
        self.dict_model_variables_by_subsector, self.dict_model_variable_to_subsector, self.dict_model_variable_to_category_restriction = self.get_variables_by_subsector()
        self.all_model_variables, self.dict_model_variables_to_variables = self.get_variable_fields_by_variable()

        # run checks and raise errors if invalid data, are entered
        #self.check_dimensions_of_analysis()
        self.check_land_use_tables()
        self.configuration = self.check_configuration(dict_config)




    ############################################################
    #   FUNCTIONS FOR ATTRIBUTE TABLES, DIMENSIONS, SECTORS    #
    ############################################################

    # some restrictions on the config values
    def check_config_defaults(self, val, param):
        if param == "Global Warming Potential":
            val = int(val)
        elif param == "Discount Rate":
            val = min(max(float(val), 0), 1)
        elif param == "Emissions Mass":
            val = str(val)

        return val

    # check the configuration dictionary
    def check_configuration(self, dict_in):
        # required config keys are set here; defaults can be added
        required_keys = []
        dict_aps = self.configuration_requirements.field_maps["analytical_parameter_to_configuration_file_parameter"]
        dict_def = self.configuration_requirements.field_maps["analytical_parameter_to_default_value"]
        for k in dict_aps.keys():
            val = self.check_config_defaults(dict_def[k], k) if (dict_aps[k] not in dict_in.keys()) else self.check_config_defaults(dict_aps[k], k)
            dict_in.update({dict_aps[k]: val})
        return dict_in

    # ensure dimensions of analysis are properly specified
    def check_dimensions_of_analysis(self):
        if not set(self.dimensions_of_analysis).issubset(set(self.all_dims)):
            missing_vals = sf.print_setdiff(set(self.dimensions_of_analysis), set(self.all_dims))
            raise ValueError(f"Missing specification of required dimensions of analysis: no attribute tables for dimensions {missing_vals} found in directory '{self.attribute_directory}'.")

    # get subsectors that have a primary cateogry; these sectors can leverage the functions below effectively
    def get_all_subsectors_with_primary_category(self):
        l_with = list(self.dict_attributes["abbreviation_subsector"].field_maps["subsector_to_primary_category_py"].keys())
        l_with.sort()
        l_without = list(set(self.all_subsectors) - set(l_with))
        l_without.sort()

        return l_with, l_without

    # retrieve and format attribute tables for use
    def get_attribute_tables(self, dir_att):
        # get available types
        all_types = [x for x in os.listdir(dir_att) if (self.attribute_file_extension in x) and ((self.substr_categories in x) or (self.substr_varreqs_allcats in x) or (self.substr_varreqs_partialcats in x) or (self.substr_analytical_parameters in x))]
        all_categories = []
        all_dims = []
        ##  batch load attributes/variable requirements and turn them into AttributeTable objects
        dict_attributes = {}
        dict_varreqs = {}
        for att in all_types:
            fp = os.path.join(dir_att, att)
            if self.substr_dimensions in att:
                nm = att.replace(self.substr_dimensions, "").replace(self.attribute_file_extension, "")
                k = f"dim_{nm}"
                att_table = AttributeTable(fp, nm, [])
                dict_attributes.update({k: att_table})
                all_dims.append(nm)
            elif self.substr_categories in att:
                nm = sf.clean_field_names([x for x in pd.read_csv(fp, nrows = 0).columns if "$" in x])[0]
                att_table = AttributeTable(fp, nm, [])
                dict_attributes.update({nm: att_table})
                all_categories.append(nm)
            elif (self.substr_varreqs_allcats in att) or (self.substr_varreqs_partialcats in att):
                nm = att.replace(self.substr_varreqs, "").replace(self.attribute_file_extension, "")
                att_table = AttributeTable(fp, "variable", [])
                dict_varreqs.update({nm: att_table})
            elif (att == f"{self.substr_analytical_parameters}{self.attribute_file_extension}"):
                nm = att.replace(self.attribute_file_extension, "")
                configuration_requirements = AttributeTable(fp, "analytical_parameter", [])
            else:
                raise ValueError(f"Invalid attribute '{att}': ensure '{self.substr_categories}', '{self.substr_varreqs_allcats}', or '{self.substr_varreqs_partialcats}' is contained in the attribute file.")

        ##  add some subsector/python specific information into the subsector table
        field_category = "primary_category"
        field_category_py = field_category + "_py"
        # add a new field
        df_tmp = dict_attributes["abbreviation_subsector"].table
        df_tmp[field_category_py] = sf.clean_field_names(df_tmp[field_category])
        df_tmp = df_tmp[df_tmp[field_category_py] != "none"].reset_index(drop = True)
        # set a key and prepare new fields
        key = field_category_py
        fields_to_dict = [x for x in df_tmp.columns if x != key]
        # next, create dict maps to add to the table
        field_maps = {}
        for fld in fields_to_dict:
            field_fwd = f"{key}_to_{fld}"
            field_rev = f"{fld}_to_{key}"
            field_maps.update({field_fwd: sf.build_dict(df_tmp[[key, fld]])})
            # check for 1:1 correspondence before adding reverse
            vals_unique = set(df_tmp[fld])
            if (len(vals_unique) == len(df_tmp)):
                field_maps.update({field_rev: sf.build_dict(df_tmp[[fld, key]])})

        dict_attributes["abbreviation_subsector"].field_maps.update(field_maps)

        return (all_categories, all_dims, all_types, configuration_requirements, dict_attributes, dict_varreqs)


    # get different dimensions
    def get_sector_dims(self):
        # sector info
        all_sectors = list(self.dict_attributes["abbreviation_sector"].table["sector"])
        all_sectors.sort()
        all_sectors_abvs = list(self.dict_attributes["abbreviation_sector"].table["abbreviation_sector"])
        all_sectors_abvs.sort()
        # subsector info
        all_subsectors = list(self.dict_attributes["abbreviation_subsector"].table["subsector"])
        all_subsectors.sort()
        all_subsector_abvs = list(self.dict_attributes["abbreviation_subsector"].table["abbreviation_subsector"])
        all_subsector_abvs.sort()

        return (all_sectors, all_sectors_abvs, all_subsectors, all_subsector_abvs)

    # function for dimensional attributes
    def get_dimensional_attribute(self, dimension, return_type):
        if dimension not in self.all_dims:
            valid_dims = sf.format_print_list(self.all_dims)
            raise ValueError(f"Invalid dimension '{dimension}'. Valid dimensions are {valid_dims}.")
        # add attributes here
        dict_out = {
            "pydim": ("dim_" + dimension)
        }

        if return_type in dict_out.keys():
            return dict_out[return_type]
        else:
            valid_rts = sf.format_print_list(list(dict_out.keys()))
            # warn user, but still allow a return
            warnings.warn(f"Invalid dimensional attribute '{return_type}'. Valid return type values are:{valid_rts}")
            return None

    # function for grabbing an attribute column from an attribute table ordered the same as key values
    def get_ordered_category_attribute(self, subsector: str, attribute: str) -> list:
        pycat = self.get_subsector_attribute(subsector, "pycategory_primary")
        attr_cur = self.dict_attributes[pycat]

        if attribute not in attr_cur.table.columns:
            raise ValueError(f"Missing attribute column '{attribute}': attribute not found in '{subsector}' attribute table.")

        # get the dictionary and order
        dict_map = sf.build_dict(attr_cur.table[[attr_cur.key, attribute]])
        return [dict_map[x] for x in attr_cur.key_values]

    # function for retrieving different attributes associated with a subsector
    def get_subsector_attribute(self, subsector, return_type):
        dict_out = {
            "pycategory_primary": self.dict_attributes["abbreviation_subsector"].field_maps["subsector_to_primary_category_py"][subsector],
            "abv_subsector": self.dict_attributes["abbreviation_subsector"].field_maps["subsector_to_abbreviation_subsector"][subsector]
        }
        dict_out.update({"sector": self.dict_attributes["abbreviation_subsector"].field_maps["abbreviation_subsector_to_sector"][dict_out["abv_subsector"]]})
        dict_out.update({"abv_sector": self.dict_attributes["abbreviation_sector"].field_maps["sector_to_abbreviation_sector"][dict_out["sector"]]})

        # format some strings
        key_allvarreqs = self.substr_varreqs_allcats.replace(self.substr_varreqs, "") + dict_out["abv_sector"] + "_" + dict_out["abv_subsector"]
        key_partialvarreqs = self.substr_varreqs_partialcats.replace(self.substr_varreqs, "") + dict_out["abv_sector"] + "_" + dict_out["abv_subsector"]

        if key_allvarreqs in self.dict_varreqs.keys():
            dict_out.update({"key_varreqs_all": key_allvarreqs})
        if key_partialvarreqs in self.dict_varreqs.keys():
            dict_out.update({"key_varreqs_partial": key_partialvarreqs})

        if return_type in dict_out.keys():
            return dict_out[return_type]
        else:
            valid_rts = sf.format_print_list(list(dict_out.keys()))
            # warn user, but still allow a return
            warnings.warn(f"Invalid subsector attribute '{return_type}'. Valid return type values are:{valid_rts}")
            return None

    # reorganize a bit to create variable fields associated with each variable
    def get_variable_fields_by_variable(self):
        dict_vars_to_fields = {}
        modvars_all = []
        for subsector in self.all_subsectors_with_primary_category:
            modvars = self.dict_model_variables_by_subsector[subsector]
            modvars.sort()
            modvars_all += modvars
            for var in modvars:
                dict_vars_to_fields.update({var: self.build_varlist(subsector, variable_subsec = var)})

        return modvars_all, dict_vars_to_fields


    #########################################################################
    #    QUICK RETRIEVAL OF FUNDAMENTAL TRANSFORMATIONS (GWP, MASS, ETC)    #
    #########################################################################

    # get gwp
    def get_gwp(self, gas):
        gwp = int(self.configuration["global_warming_potential"])
        key_dict = f"emission_gas_to_global_warming_potential_{gwp}"

        if gas in self.dict_attributes["emission_gas"].field_maps[key_dict].keys():
            return self.dict_attributes["emission_gas"].field_maps[key_dict][gas]
        else:
            valid_vals = sf.format_print_list(self.dict_attributes["emission_gas"].key_values)
            raise KeyError(f"Invalid gas '{gas}': defined gasses are {valid_vals}.")

    # get mass
    def get_mass_equivalent(self, mass):
        me = str(self.configuration["emissions_mass"]).lower()
        key_dict = f"unit_mass_to_mass_equivalent_{me}"

        if mass in self.dict_attributes["unit_mass"].field_maps[key_dict].keys():
            return self.dict_attributes["unit_mass"].field_maps[key_dict][mass]
        else:
            valid_vals = sf.format_print_list(self.dict_attributes["unit_mass"].key_values)
            raise KeyError(f"Invalid mass '{mass}': defined gasses are {valid_vals}.")

    # get scalar
    def get_scalar(self, modvar: str, return_type: str = "total"):

        valid_rts = ["total", "gas", "mass"]
        if return_type not in valid_rts:
            tps = sf.format_print_list(valid_rts)
            raise ValueError(f"Invalid return type '{return_type}' in get_scalar: valid types are {tps}.")

        # get scalars
        gas = self.get_variable_characteristic(modvar, self.varchar_str_emission_gas)
        scalar_gas = 1 if not gas else self.get_gwp(gas.lower())
        #
        mass = self.get_variable_characteristic(modvar, self.varchar_str_unit_mass)
        scalar_mass = 1 if not mass else self.get_mass_equivalent(mass.lower())

        if return_type == "gas":
            out = scalar_gas
        elif return_type == "mass":
            out = scalar_mass
        elif return_type == "total":
            out = scalar_gas*scalar_mass

        return out


    ####################################################
    #    SECTOR-SPECIFIC AND CROSS SECTORIAL CHECKS    #
    ####################################################

    # LAND USE checks
    def check_land_use_tables(self):

        # specify some generic variables
        catstr_forest = self.dict_attributes["abbreviation_subsector"].field_maps["subsector_to_primary_category_py"]["Forest"]
        catstr_landuse = self.dict_attributes["abbreviation_subsector"].field_maps["subsector_to_primary_category_py"]["Land Use"]
        attribute_forest = self.dict_attributes[catstr_forest]
        attribute_landuse = self.dict_attributes[catstr_landuse]
        cats_forest = attribute_forest.key_values
        cats_landuse = attribute_landuse.key_values
        matchstr_forest = self.matchstring_landuse_to_forests

        ##  check that all forest categories are in land use and that all categories specified as forest are in the land use table
        set_cats_forest_in_land_use = set([matchstr_forest + x for x in cats_forest])
        set_land_use_forest_cats = set([x.replace(matchstr_forest, "") for x in cats_landuse if (matchstr_forest in x)])

        if not set_cats_forest_in_land_use.issubset(set(cats_landuse)):
            missing_vals = set_cats_forest_in_land_use - set(cats_landuse)
            missing_str = sf.format_print_list(missing_vals)
            raise KeyError(f"Missing key values in land use attribute file '{attribute_landuse.fp_table}': did not find land use categories {missing_str}.")
        elif not set_land_use_forest_cats.issubset(cats_forest):
            extra_vals = set_land_use_forest_cats - set(cats_forest)
            extra_vals = sf.format_print_list(extra_vals)
            raise KeyError(f"Undefined forest categories specified in land use attribute file '{attribute_landuse.fp_table}': did not find forest categories {extra_vals}.")



    #########################################################
    #    VARIABLE REQUIREMENT AND MANIPULATION FUNCTIONS    #
    #########################################################

    ##  add subsector emissions aggregates to an output dataframe
    def add_subsector_emissions_aggregates(self, df_in: pd.DataFrame, list_subsectors: list, stop_on_missing_fields_q: bool = False):
        # loop over base subsectors
        for subsector in list_subsectors:#self.required_base_subsectors:
            vars_subsec = self.dict_model_variables_by_subsector[subsector]
            # add subsector abbreviation
            fld_nam = self.get_subsector_attribute(subsector, "abv_subsector")
            fld_nam = f"emission_co2e_subsector_total_{fld_nam}"

            flds_add = []
            for var in vars_subsec:
                var_type = self.get_variable_attribute(var, "variable_type").lower()
                gas = self.get_variable_characteristic(var, self.varchar_str_emission_gas)
                if (var_type == "output") and gas:
                    flds_add +=  self.dict_model_variables_to_variables[var]

            # check for missing fields; notify
            missing_fields = [x for x in flds_add if x not in df_in.columns]
            if len(missing_fields) > 0:
                str_mf = print_setdiff(set(df_in.columns), set(flds_add))
                str_mf = f"Missing fields {str_mf}.%s"
                if stop_on_missing_fields_q:
                    raise ValueError(str_mf%(" Subsector emission totals will not be added."))
                else:
                    warnings.warn(str_mf%(" Subsector emission totals will exclude these fields."))

            keep_fields = [x for x in flds_add if x in df_in.columns]
            df_in[fld_nam] = df_in[keep_fields].sum(axis = 1)


    ##  function for converting an array to a variable out dataframe (used in sector models)
    def array_to_df(self, arr_in, modvar: str, include_scalars = False) -> pd.DataFrame:
        # get subsector and fields to name based on variable
        subsector = self.dict_model_variable_to_subsector[modvar]
        fields = self.build_varlist(subsector, variable_subsec = modvar)

        scalar_em = 1
        scalar_me = 1
        if include_scalars:
            # get scalars
            gas = self.get_variable_characteristic(modvar, self.varchar_str_emission_gas)
            mass = self.get_variable_characteristic(modvar, self.varchar_str_unit_mass)
            # will conver ch4 to co2e e.g. + kg to MT
            scalar_em = 1 if not gas else self.get_gwp(gas.lower())
            scalar_me = 1 if not mass else self.get_mass_equivalent(mass.lower())

        # raise error if there's a shape mismatch
        if len(fields) != arr_in.shape[1]:
            flds_print = sf.format_print_list(fields)
            raise ValueError(f"Array shape mismatch for fields {flds_print}: the array only has {arr_in.shape[1]} columns.")

        return pd.DataFrame(arr_in*scalar_em*scalar_me, columns = fields)


    ##  function for bulding a basic variable list from the (no complexitiies)
    def build_vars_basic(self, dict_vr_varschema: dict, dict_vars_to_cats: dict, category_to_replace: str) -> list:
        # dict_vars_to_loop has keys that are variables to loop over that map to category values
        vars_out = []
        vars_loop = list(set(dict_vr_varschema.keys()) & set(dict_vars_to_cats.keys()))
        # loop over required variables (exclude transition probability)
        for var in vars_loop:
            error_str = f"Invalid value associated with variable key '{var}'  build_vars_basic/dict_vars_to_cats: the value in the dictionary should be the string 'none' or a list of category values."
            var_schema = clean_schema(dict_vr_varschema[var])
            if type(dict_vars_to_cats[var]) == list:
                for catval in dict_vars_to_cats[var]:
                    vars_out.append(var_schema.replace(category_to_replace, catval))
            elif type(dict_vars_to_cats[var]) == str:
                if dict_vars_to_cats[var].lower() == "none":
                    vars_out.append(var_schema)
                else:
                    raise ValueError(error_str)
            else:
                raise ValueError(error_str)

        return vars_out

    ##  function to build variables that rely on the outer product (e.g., transition probabilities)
    def build_vars_outer(self, dict_vr_varschema: dict, dict_vars_to_cats: dict, category_to_replace: str, appendstr_i: str = "-I", appendstr_j: str = "-J") -> list:
        # build categories for I/J
        cat_i, cat_j = self.format_category_for_outer(category_to_replace, appendstr_i, appendstr_j)

        vars_out = []
        # run some checks and notify of any dropped variables
        set_vr_schema_vars = set(dict_vr_varschema.keys())
        set_vars_to_cats_vars = set(dict_vars_to_cats.keys())
        vars_to_loop = set_vr_schema_vars & set_vars_to_cats_vars
        # variables not in dict_vars_to_cats
        if len(set_vr_schema_vars - vars_to_loop) > 0:
            l_drop = list(set_vr_schema_vars - vars_to_loop)
            l_drop.sort()
            l_drop = sf.format_print_list(l_drop)
            warnings.warn(f"\tVariables {l_drop} not found in set_vars_to_cats_vars.")

        # variables not in dict_vr_varschema
        if len(set_vars_to_cats_vars - vars_to_loop) > 0:
            l_drop = list(set_vars_to_cats_vars - vars_to_loop)
            l_drop.sort()
            l_drop = sf.format_print_list(l_drop)
            warnings.warn(f"\tVariables {l_drop} not found in set_vr_schema_vars.")

        vars_to_loop = list(vars_to_loop)

        # loop over the variables available in both the variable schema dictionary and the dictionary mapping each variable to categories
        for var in vars_to_loop:
            var_schema = clean_schema(dict_vr_varschema[var])
            if (cat_i not in var_schema) or (cat_j not in var_schema):
                fb_tab = dict_attributes[self.get_subsector_attribute(subsector, "pycategory_primary")].fp_table
                raise ValueError(f"Error in {var} variable schema: one of the outer categories '{cat_i}' or '{cat_j}' was not found. Check the attribute file found at '{fp_tab}'.")
            for catval_i in dict_vars_to_cats[var]:
                for catval_j in dict_vars_to_cats[var]:
                    vars_out.append(var_schema.replace(cat_i, catval_i).replace(cat_j, catval_j))

        return vars_out

    # function to check category subsets that are specified
    def check_category_restrictions(self, categories_to_restrict_to, attribute_table: AttributeTable, stop_process_on_error: bool = True) -> list:
        if categories_to_restrict_to != None:
            if type(categories_to_restrict_to) != list:
                raise TypeError(f"Invalid type of categories_to_restrict_to: valid types are 'None' and 'list'.")
            valid_cats = [x for x in categories_to_restrict_to if x in attribute_table.key_values]
            invalid_cats = [x for x in categories_to_restrict_to if (x not in attribute_table.key_values)]
            if len(invalid_cats) > 0:
                missing_cats = sf.format_print_list(invalid_cats)
                msg_err = f"Invalid categories {invalid_cats} found."
                if stop_process_on_error:
                    raise ValueError(msg_err)
                else:
                    warnings.warn(msg_err + " They will be dropped.")
            return valid_cats
        else:
            return attribute_table.key_values


    # function to build a variable using an ordered set of categories associated with another variable
    def build_target_varlist_from_source_varcats(self, modvar_source: str, modvar_target: str):
        # get source categories
        cats_source = self.get_categories(modvar_source)
        # build the target variable list using the source categories
        subsector_target = self.dict_model_variable_to_subsector[modvar_target]
        vars_target = self.build_varlist(subsector_target, variable_subsec = modvar_target, restrict_to_category_values = cats_source)

        return vars_target


    ##  function for building a list of variables (fields) for data tables
    def build_varlist(
        self,
        subsector: str,
        variable_subsec = None,
        restrict_to_category_values = None,
        dict_force_override_vrp_vvs_cats = None,
        variable_type = None
    ) -> list:
        """
            dict_force_override_vrp_vvs_cats can be set do a dictionary of the form
            {MODEL_VAR_NAME: [catval_a, catval_b, catval_c, ... ]}
            where catval_i are not all unique; this is useful for making a variable that maps unique categories to a subset of non-unique categories that represent proxies (e.g., buffalo -> cattle_dairy, )
        """
        # get some subsector info
        category = self.dict_attributes["abbreviation_subsector"].field_maps["abbreviation_subsector_to_primary_category"][self.get_subsector_attribute(subsector, "abv_subsector")].replace("`", "")
        category_ij_tuple = self.format_category_for_outer(category, "-I", "-J")
        attribute_table = self.dict_attributes[self.get_subsector_attribute(subsector, "pycategory_primary")]
        valid_cats = self.check_category_restrictions(restrict_to_category_values, attribute_table)

        # get dictionary of variable to variable schema and id variables that are in the outer (Cartesian) product (i x j)
        dict_vr_vvs, dict_vr_vvs_outer = self.separate_varreq_dict_for_outer(subsector, "key_varreqs_all", category_ij_tuple, variable = variable_subsec, variable_type = variable_type)
        # build variables that apply to all categories
        vars_out = self.build_vars_basic(dict_vr_vvs, dict(zip(list(dict_vr_vvs.keys()), [valid_cats for x in dict_vr_vvs.keys()])), category)
        if len(dict_vr_vvs_outer) > 0:
            vars_out += self.build_vars_outer(dict_vr_vvs_outer, dict(zip(list(dict_vr_vvs_outer.keys()), [valid_cats for x in dict_vr_vvs_outer.keys()])), category)

        # build those that apply to partial categories
        dict_vrp_vvs, dict_vrp_vvs_outer = self.separate_varreq_dict_for_outer(subsector, "key_varreqs_partial", category_ij_tuple, variable = variable_subsec, variable_type = variable_type)
        dict_vrp_vvs_cats, dict_vrp_vvs_cats_outer = self.get_partial_category_dictionaries(subsector, category_ij_tuple, variable_in = variable_subsec, restrict_to_category_values = restrict_to_category_values)

        # check dict_force_override_vrp_vvs_cats - use w/caution if not none. Cannot use w/outer
        if dict_force_override_vrp_vvs_cats != None:
            # check categories
            for k in dict_force_override_vrp_vvs_cats.keys():
                sf.check_set_values(dict_force_override_vrp_vvs_cats[k], attribute_table.key_values, f" in dict_force_override_vrp_vvs_cats at key {k} (subsector {subsector})")
            dict_vrp_vvs_cats = dict_force_override_vrp_vvs_cats

        if len(dict_vrp_vvs) > 0:
            vars_out += self.build_vars_basic(dict_vrp_vvs, dict_vrp_vvs_cats, category)
        if len(dict_vrp_vvs_outer) > 0:
            vl = self.build_vars_outer(dict_vrp_vvs_outer, dict_vrp_vvs_cats_outer, category)
            vars_out += self.build_vars_outer(dict_vrp_vvs_outer, dict_vrp_vvs_cats_outer, category)

        return vars_out


    # clean a partial category dictionary to return either none (no categorization) or a list of applicable cateogries
    def clean_partial_category_dictionary(self, dict_in: dict, all_category_values, delim: str = "|") -> dict:
        for k in dict_in.keys():
            if "none" == dict_in[k].lower().replace(" ", ""):
                dict_in.update({k: "none"})
            else:
                cats = dict_in[k].replace("`", "").split(delim)
                dict_in.update({k: [x for x in cats if x in all_category_values]})
                missing_vals = [x for x in cats if x not in dict_in[k]]
                if len(missing_vals) > 0:
                    missing_vals = sf.format_print_list(missing_vals)
                    warnings.warn(f"clean_partial_category_dictionary: Invalid categories values {missing_vals} dropped when cleaning the dictionary. Category values not found.")
        return dict_in

    # function to retrieve an (ordered) list of categories for a variable
    def get_categories(self, variable: str):
        if variable not in self.all_model_variables:
            raise ValueError(f"Invalid variable '{variable}': variable not found.")
        # initialize as all categories
        subsector = self.dict_model_variable_to_subsector[variable]
        all_cats = self.dict_attributes[self.get_subsector_attribute(subsector, "pycategory_primary")].key_values
        if self.dict_model_variable_to_category_restriction[variable] == "partial":
            cats = self.get_variable_attribute(variable, "categories")
            if "none" not in cats.lower():
                cats = cats.replace("`", "").split("|")
                cats = [x for x in cats if x in all_cats]
            else:
                cats = None
        else:
            cats = all_cats
        return cats

    # function for getting input/output fields for a list of subsectors
    def get_input_output_fields(self, subsectors_required: list, build_df_q = False):
        # initialize output lists
        vars_out = []
        vars_req = []
        subsectors_out = []

        for subsector in subsectors_required:
            vars_subsector_req = self.build_varlist(subsector, variable_type = "input")
            vars_subsector_out = self.build_varlist(subsector, variable_type = "output")
            vars_req += vars_subsector_req
            vars_out += vars_subsector_out
            if build_df_q:
                subsectors_out += [subsector for x in vars_subsector]

        if build_df_q:
            vars_req = pd.DataFrame({"subsector": subsectors_out, "variable": vars_req}).sort_values(by = ["subsector", "variable"]).reset_index(drop = True)
            vars_out = pd.DataFrame({"subsector": subsectors_out, "variable": vars_out}).sort_values(by = ["subsector", "variable"]).reset_index(drop = True)

        return vars_req, vars_out

    # function to build a dictionary of categories applicable to a give variable; split by unidim/outer
    def get_partial_category_dictionaries(
        self,
        subsector: str,
        category_outer_tuple: tuple,
        key_type: str = "key_varreqs_partial",
        delim: str = "|",
        variable_in = None,
        restrict_to_category_values = None,
        var_type = None
    ):
        # key_type = key_varreqs_all, key_varreqs_partial

        key_attribute = self.get_subsector_attribute(subsector, key_type)
        #valid_cats = self.dict_attributes[self.get_subsector_attribute(subsector, "pycategory_primary")].key_values
        valid_cats = self.check_category_restrictions(restrict_to_category_values, self.dict_attributes[self.get_subsector_attribute(subsector, "pycategory_primary")])

        if key_attribute != None:
            dict_vr_vvs_cats_ud, dict_vr_vvs_cats_outer = self.separate_varreq_dict_for_outer(subsector, key_type, category_outer_tuple, target_field = "categories", variable = variable_in, variable_type = var_type)
            dict_vr_vvs_cats_ud = self.clean_partial_category_dictionary(dict_vr_vvs_cats_ud, valid_cats, delim)
            dict_vr_vvs_cats_outer = self.clean_partial_category_dictionary(dict_vr_vvs_cats_outer, valid_cats, delim)

            return dict_vr_vvs_cats_ud, dict_vr_vvs_cats_outer
        else:
            return {}, {}


    ##  function for retrieving the variable schema associated with a variable
    def get_variable_attribute(self, variable: str, attribute: str) -> str:
        # check variable first
        if variable not in self.all_model_variables:
            raise ValueError(f"Invalid model variable '{variable}' found in get_variable_characteristic.")

        subsector = self.dict_model_variable_to_subsector[variable]
        cat_restriction_type = self.dict_model_variable_to_category_restriction[variable]
        key_varreqs = self.get_subsector_attribute(subsector, f"key_varreqs_{cat_restriction_type}")
        key_fm = f"variable_to_{attribute}"

        sf.check_keys(self.dict_varreqs[key_varreqs].field_maps, [key_fm])
        var_attr = self.dict_varreqs[key_varreqs].field_maps[key_fm][variable]

        return var_attr


    ##  function for mapping variable to default characteristic (e.g., gas, units, etc.)
    def get_variable_characteristic(self, variable: str, characteristic: str) -> str:
        var_schema = self.get_variable_attribute(variable, "variable_schema")
        dict_out = clean_schema(var_schema, return_default_dict_q = True)
        return dict_out.get(characteristic)


    ##  function to extract a variable (with applicable categories from an input data frame)
    def get_standard_variables(self,
        df_in: pd.DataFrame,
        modvar: str,
        override_vector_for_single_mv_q: bool = False,
        return_type: str = "data_frame",
        var_bounds = None,
        force_boundary_restriction: bool = True
    ):

        flds = self.dict_model_variables_to_variables[modvar]
        flds = flds[0] if ((len(flds) == 1) and not override_vector_for_single_mv_q) else flds

        valid_rts = ["data_frame", "array_base", "array_units_corrected"]
        if return_type not in valid_rts:
            vrts = sf.format_print_list(valid_rts)
            raise ValueError(f"Invalid return_type in get_standard_variables: valid types are {vrts}.")

        # initialize output, apply various common transformations based on type
        out = df_in[flds]
        if return_type != "data_frame":
            out = np.array(out)
            if return_type == "array_units_corrected":
                out *= self.get_scalar(modvar, "total")

        if type(var_bounds) in [tuple, list, np.ndarray]:
            # get numeric values and check
            var_bounds = [x for x in var_bounds if type(x) in [int, float]]
            if len(var_bounds) <= 1:
                raise ValueError(f"Invalid specification of variable bounds '{var_bounds}': there must be a maximum and a minimum numeric value specified.")

            # ensure array
            out = np.array(out)
            b_0, b_1 = np.min(var_bounds), np.max(var_bounds)
            m_0, m_1 = np.min(out), np.max(out)

            # check bounds
            if m_1 > b_1:
                str_warn = f"Invalid maximum value of '{varname}': specifed value of {m_1} exceeds bound {b_1}."
                if force_restriction:
                    warnings.warn(str_warn + "\nForcing maximum value in trajectory.")
                else:
                    raise ValueError(str_warn)
            # check min
            if m_0 < b_0:
                str_warn = f"Invalid minimum value of '{varname}': specifed value of {m_0} below bound {b_0}."
                if force_restriction:
                    warnings.warn(str_warn + "\nForcing minimum value in trajectory.")
                else:
                    raise ValueError(str_warn)

            if force_boundary_restriction:
                out = sf.vec_bounds(out, var_bounds)
            out = pd.DataFrame(out, flds) if (return_type == "data_frame") else out

        return out


    ##  function to get all variables associated with a subsector (will not function if there is no primary category)
    def get_subsector_variables(self, subsector: str, var_type = None) -> list:
        # get some information used
        category = self.dict_attributes["abbreviation_subsector"].field_maps["abbreviation_subsector_to_primary_category"][self.get_subsector_attribute(subsector, "abv_subsector")].replace("`", "")
        category_ij_tuple = self.format_category_for_outer(category, "-I", "-J")
        # initialize output list, dictionary of variable to categorization (all or partial), and loop
        vars_by_subsector = []
        dict_var_type = {}
        for key_type in ["key_varreqs_all", "key_varreqs_partial"]:
            dicts = self.separate_varreq_dict_for_outer(subsector, key_type, category_ij_tuple, variable_type = var_type)
            for x in dicts:
                l_vars = list(x.keys())
                vars_by_subsector += l_vars
                dict_var_type.update(dict(zip(l_vars, [key_type.replace("key_varreqs_", "") for x in l_vars])))

        return dict_var_type, vars_by_subsector


    # list variables by all valid subsectors (excludes those without a primary category)
    def get_variables_by_subsector(self) -> dict:
        dict_vars_out = {}
        dict_vartypes_out = {}
        dict_vars_to_subsector = {}
        for subsector in self.dict_attributes["abbreviation_subsector"].field_maps["subsector_to_primary_category_py"].keys():
            dict_var_type, vars_by_subsector = self.get_subsector_variables(subsector)
            dict_vars_out.update({subsector: vars_by_subsector})
            dict_vartypes_out.update(dict_var_type)
            dict_vars_to_subsector.update(dict(zip(vars_by_subsector, [subsector for x in vars_by_subsector])))

        return dict_vars_out, dict_vars_to_subsector, dict_vartypes_out


    # use this to avoid changing function in multiple places
    def format_category_for_outer(self, category_to_replace, appendstr_i = "-I", appendstr_j = "-J"):
        cat_i = category_to_replace.replace("$", f"{appendstr_i}$")[len(appendstr_i):]
        cat_j = category_to_replace.replace("$", f"{appendstr_j}$")[len(appendstr_j):]
        return (cat_i, cat_j)


    # separate a variable requirement dictionary into those associated with simple vars and those with outer
    def separate_varreq_dict_for_outer(
        self,
        subsector: str,
        key_type: str,
        category_outer_tuple: tuple,
        target_field: str = "variable_schema",
        field_to_split_on: str = "variable_schema",
        variable = None,
        variable_type = None
    ) -> tuple:
        # field_to_split_on gives the field from the attribute table to use to split between outer and unidim
        # target field is the field to return in the dictionary
        # key_type = key_varreqs_all, key_varreqs_partial
        key_attribute = self.get_subsector_attribute(subsector, key_type)
        if key_attribute != None:
            dict_vr_vvs = self.dict_varreqs[self.get_subsector_attribute(subsector, key_type)].field_maps[f"variable_to_{field_to_split_on}"].copy()
            dict_vr_vtf = self.dict_varreqs[self.get_subsector_attribute(subsector, key_type)].field_maps[f"variable_to_{target_field}"].copy()

            # filter on variable type if specified
            if variable_type != None:
                if variable != None:
                    warnings.warn(f"variable and variable_type both specified in separate_varreq_dict_for_outer: the variable assignment is higher priority, and variable_type will be ignored.")
                else:
                    dict_var_types = self.dict_varreqs[self.get_subsector_attribute(subsector, key_type)].field_maps[f"variable_to_variable_type"]
                    drop_vars = [x for x in dict_var_types.keys() if dict_var_types[x].lower() != variable_type.lower()]
                    [dict_vr_vvs.pop(x) for x in drop_vars]
                    [dict_vr_vtf.pop(x) for x in drop_vars]

            dict_vr_vtf_outer = dict_vr_vtf.copy()

            vars_outer = [x for x in dict_vr_vtf.keys() if (category_outer_tuple[0] in dict_vr_vvs[x]) and (category_outer_tuple[1] in dict_vr_vvs[x])]
            vars_unidim = [x for x in dict_vr_vtf.keys() if (x not in vars_outer)]
            [dict_vr_vtf_outer.pop(x) for x in vars_unidim]
            [dict_vr_vtf.pop(x) for x in vars_outer]

            if variable != None:
                vars_outer = list(dict_vr_vtf_outer.keys())
                vars_unidim = list(dict_vr_vtf.keys())
                [dict_vr_vtf_outer.pop(x) for x in vars_outer if (x != variable)]
                [dict_vr_vtf.pop(x) for x in vars_unidim if (x != variable)]
        else:
            dict_vr_vtf = {}
            dict_vr_vtf_outer = {}

        return dict_vr_vtf, dict_vr_vtf_outer


    # returns ordered variable (by attribute key) with cateogries replaced
    def switch_variable_category(self, source_subsector: str, target_variable: str, attribute_field: str, cats_to_switch = None, dict_force_override = None) -> list:
        """
            attribute_field is the field in the primary category attriubte table to use for the switch;
            if dict_force_override is specified, then this dictionary will be used to switch categories

            cats_to_switch to can be specified to only operate on a subset of source categorical values
        """

        sf.check_keys(self.dict_model_variable_to_subsector, [target_variable])
        target_subsector = self.dict_model_variable_to_subsector[target_variable]
        pycat_primary_source = self.get_subsector_attribute(source_subsector, "pycategory_primary")

        if dict_force_override == None:
            key_dict = f"{pycat_primary_source}_to_{attribute_field}"
            sf.check_keys(self.dict_attributes[pycat_primary_source].field_maps, [key_dict])
            dict_repl = self.dict_attributes[pycat_primary_source].field_maps[key_dict]
        else:
            dict_repl = dict_force_override

        if cats_to_switch == None:
            cats_all = self.dict_attributes[pycat_primary_source].key_values
        else:
            cats_all = self.check_category_restrictions(cats_to_switch, self.dict_attributes[pycat_primary_source])
        cats_target = [dict_repl[x].replace("`", "") for x in cats_all]

        # use the 'dict_force_override_vrp_vvs_cats' override dictionary in build_varlist here
        return self.build_varlist(target_subsector, target_variable, cats_target, {target_variable: cats_target})




    #########################################
    #    INTERNALLY-CALCULATED VARIABLES    #
    #########################################

    # retrives mutually-exclusive fields used to sum to generate internal variables
    def get_mutex_cats_for_internal_variable(self, subsector: str, variable: str, attribute_sum_specification_field: str, return_type: str = "fields"):
        # attribute_sum_specification_field gives the field in the category attribute table that defines what to sum over (e.g., gdp component in the value added)
        # get categories to sum over
        pycat_primary = self.get_subsector_attribute(subsector, "pycategory_primary")
        df_tmp = self.dict_attributes[pycat_primary].table
        sum_cvs = list(df_tmp[df_tmp[attribute_sum_specification_field].isin([1])][pycat_primary])
        # get the variable list, check, and add to output
        fields_sum = self.build_varlist(subsector, variable_subsec = variable, restrict_to_category_values = sum_cvs)
        # check return types
        if return_type == "fields":
            return fields_sum
        elif return_type == "category_values":
            return sum_cvs
        else:
            raise ValueError(f"Invalid return_type '{return_type}'. Please specify 'fields' or 'category_values'.")


    # function to add GDP based on value added
    def manage_internal_variable_to_df(self, df_in:pd.DataFrame, subsector: str, internal_variable: str, component_variable: str, attribute_sum_specification_field: str, action: str = "add"):
        # get the gdp field
        field_check = self.build_varlist(subsector, variable_subsec = internal_variable)[0]
        valid_actions = ["add", "remove", "check"]
        if action not in valid_actions:
            str_valid = sf.format_print_list(valid_actions)
            raise ValueError(f"Invalid actoion '{action}': valid actions are {str_valid}.")
        if action == "check":
            return True if (field_check in df_in.columns) else False
        elif action == "remove":
            if field_check in df_in.columns:
                df_in.drop(labels = field_check, axis = 1, inplace = True)
        elif action == "add":
            if field_check not in df_in.columns:
                # get fields to sum over
                fields_sum = self.get_mutex_cats_for_internal_variable(subsector, component_variable, attribute_sum_specification_field, "fields")
                sf.check_fields(df_in, fields_sum)
                # add to the data frame (inline)
                df_in[field_check] = df_in[fields_sum].sum(axis = 1)


    # manage internal variables in data frames
    def manage_gdp_to_df(self, df_in: pd.DataFrame, action: str = "add"):
        return self.manage_internal_variable_to_df(df_in, "Economy", "GDP", "Value Added", "gdp_component", action)
    def manage_pop_to_df(self, df_in: pd.DataFrame, action: str = "add"):
        return self.manage_internal_variable_to_df(df_in, "General", "Total Population", "Population", "total_population_component", action)



# function for cleaning a variable schema
def clean_schema(var_schema: str, return_default_dict_q: bool = False) -> str:

    var_schema = var_schema.split("(")
    var_schema[0] = var_schema[0].replace("`", "").replace(" ", "")

    dict_repls = {}
    if len(var_schema) > 1:
        repls =  var_schema[1].replace("`", "").split(",")
        for dr in repls:
            dr0 = dr.replace(" ", "").replace(")", "").split("=")
            var_schema[0] = var_schema[0].replace(dr0[0], dr0[1])
            dict_repls.update({dr0[0]: dr0[1]})

    if return_default_dict_q:
        return dict_repls
    else:
        return var_schema[0]
