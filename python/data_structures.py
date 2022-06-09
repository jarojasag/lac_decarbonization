import itertools
import numpy as np
import os, os.path
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




##  CONFIGURATION file
class Configuration:

    def __init__(self,
        fp_config: str,
        attr_energy: AttributeTable,
        attr_gas: AttributeTable,
        attr_length: AttributeTable,
        attr_mass: AttributeTable,
        attr_volume: AttributeTable,
        attr_required_parameters: AttributeTable = None
    ):
        self.fp_config = fp_config
        self.attr_required_parameters = attr_required_parameters

        # set tables
        self.attr_energy = attr_energy
        self.attr_gas = attr_gas
        self.attr_length = attr_length
        self.attr_mass = attr_mass
        self.attr_volume = attr_volume

        # set required parametrs by type
        self.params_string = ["energy_units", "emissions_mass", "historical_solid_waste_method", "length_units", "volume_units"]
        self.params_float = ["days_per_year"]
        self.params_float_fracs = ["discount_rate"]
        self.params_int = ["global_warming_potential", "historical_back_proj_n_periods"]

        self.dict_config = self.get_config_information(attr_energy, attr_gas, attr_length, attr_mass, attr_volume, attr_required_parameters)


    # some restrictions on the config values
    def check_config_defaults(self,
        param,
        val,
        dict_valid_values: dict = dict({})
    ):
        if param in self.params_int:
            val = int(val)
        elif param in self.params_float:
            val = float(val)
        elif param in self.params_float_fracs:
            val = min(max(float(val), 0), 1)
        elif param in self.params_string:
            val = str(val)

        if param in dict_valid_values.keys():
            if val not in dict_valid_values[param]:
                valid_vals = sf.format_print_list(dict_valid_values[param])
                raise ValueError(f"Invalid specification of configuration parameter '{param}': valid values are {valid_vals}")

        return val

    # function to retrieve a configuration value
    def get(self, key: str):
        if key in self.dict_config.keys():
            return self.dict_config[key]
        else:
            raise KeyError(f"Configuration parameter '{key}' not found.")

    # function for retrieving a configuration file and population missing values with defaults
    def get_config_information(self,
        attr_energy: AttributeTable = None,
        attr_gas: AttributeTable = None,
        attr_length: AttributeTable = None,
        attr_mass: AttributeTable = None,
        attr_volume: AttributeTable = None,
        attr_parameters_required: AttributeTable = None,
        field_req_param: str = "configuration_file_parameter",
        field_default_val: str = "default_value"
    ) -> dict:

        # set some variables from defaults
        attr_energy = attr_energy if (attr_energy is not None) else self.attr_energy
        attr_gas = attr_gas if (attr_gas is not None) else self.attr_gas
        attr_length = attr_length if (attr_length is not None) else self.attr_length
        attr_mass = attr_mass if (attr_mass is not None) else self.attr_mass
        attr_volume = attr_volume if (attr_volume is not None) else self.attr_volume

        # check path and parse the config if it exists
        dict_conf = {}
        if self.fp_config != None:
            if os.path.exists(self.fp_config):
                dict_conf = self.parse_config(self.fp_config)

        # update with defaults if a value is missing in the specified configuration
        if attr_parameters_required != None:
            if attr_parameters_required.key != field_req_param:
                # add defaults
                for k in attr_parameters_required.key_values:
                    param_config = attr_parameters_required.field_maps[f"{attr_parameters_required.key}_to_{field_req_param}"][k] if (attr_parameters_required.key != field_req_param) else k
                    if param_config not in dict_conf.keys():
                        val_default = self.infer_types(attr_parameters_required.field_maps[f"{attr_parameters_required.key}_to_{field_default_val}"][k])
                        dict_conf.update({param_config: val_default})

        # check valid configuration values and update where appropriate
        valid_energy = self.get_valid_values_from_attribute_column(attr_energy, "energy_equivalent_", str, "unit_energy_to_energy")
        valid_gwp = self.get_valid_values_from_attribute_column(attr_gas, "global_warming_potential_", int)
        valid_historical_solid_waste_method = ["back_project", "historical"]
        valid_length = self.get_valid_values_from_attribute_column(attr_length, "length_equivalent_", str, "unit_length_to_length")
        valid_mass = self.get_valid_values_from_attribute_column(attr_mass, "mass_equivalent_", str, "unit_mass_to_mass")
        valid_volume = self.get_valid_values_from_attribute_column(attr_volume, "volume_equivalent_", str)

        dict_checks = {
            "energy_units": valid_energy,
            "emissions_mass": valid_mass,
            "global_warming_potential": valid_gwp,
            "historical_solid_waste_method": valid_historical_solid_waste_method,
            "length_units": valid_length,
            "volume_units": valid_volume
        }
        keys_check = list(dict_conf.keys())
        for k in keys_check:
            dict_conf.update({k: self.check_config_defaults(k, dict_conf[k], dict_checks)})


        ###   check some parameters
        # positive integer restriction
        dict_conf["historical_back_proj_n_periods"] = max(dict_conf["historical_back_proj_n_periods"], 1)

        # set some attributes
        self.valid_energy = valid_energy
        self.valid_gwp = valid_gwp
        self.valid_historical_solid_waste_method = valid_historical_solid_waste_method
        self.valid_length = valid_length
        self.valid_mass = valid_mass
        self.valid_volume = valid_volume

        return dict_conf

    # function to retrieve available emission mass specifications
    def get_valid_values_from_attribute_column(self,
        attribute_table: AttributeTable,
        column_match_str: str,
        return_type: type = None,
        field_map_to_val: str = None
    ):
        cols = [x.replace(column_match_str, "") for x in attribute_table.table.columns if (x[0:min(len(column_match_str), len(x))] == column_match_str)]
        if return_type != None:
            cols = [return_type(x) for x in cols]
        # if a dictionary is specified, map the values to a name
        if field_map_to_val != None:
            if field_map_to_val in attribute_table.field_maps.keys():
                cols = [attribute_table.field_maps[field_map_to_val][x] for x in cols]
            else:
                raise KeyError(f"Error in get_valid_values_from_attribute_column: the field map '{field_map_to_val}' is not defined.")

        return cols

    # guess the input type for a configuration file
    def infer_type(self, val):
        if val != None:
            val = str(val)
            if val.replace(".", "").replace(",", "").isnumeric():
                num = float(val)
                val = int(num) if (num == int(num)) else float(num)
        return val

    # apply to a list if necessary
    def infer_types(self, val_in, delim = ","):
        if val_in != None:
            return [self.infer_type(x) for x in val_in.split(delim)] if (delim in val_in) else self.infer_type(val_in)
        else:
            return None

    # function for parsing a configuration file into a dictionary
    def parse_config(self, fp_config: str) -> dict:
        """
            parse_config returns a dictionary of configuration values
        """

        #read in aws initialization
        if os.path.exists(fp_config):
        	with open(fp_config) as fl:
        		lines_config = fl.readlines()
        else:
            raise ValueError(f"Invalid configuation file {fp_config} specified: file not found.")

        dict_out = {}
        #remove unwanted blank characters
        for ln in lines_config:
            ln_new = sf.str_replace(ln.split("#")[0], {"\n": "", "\t": ""})
            if (":" in ln_new):
                ln_new = ln_new.split(":")
                key = str(ln_new[0])
                val = self.infer_types(str(ln_new[1]).strip())
                dict_out.update({key: val})

        return dict_out




class ModelAttributes:

    def __init__(self, dir_attributes: str, fp_config: str = None):

        # initialize dimensions of analysis - later, check for presence
        self.dim_time_period = "time_period"
        self.dim_design_id = "design_id"
        self.dim_future_id = "future_id"
        self.dim_strategy_id = "strategy_id"
        self.dim_primary_id = "primary_id"
        # ordered by sort hierarchy
        self.sort_ordered_dimensions_of_analysis = [self.dim_primary_id, self.dim_design_id, self.dim_strategy_id, self.dim_future_id, self.dim_time_period]

        # set some basic properties
        self.attribute_file_extension = ".csv"
        self.matchstring_landuse_to_forests = "forests_"
        self.substr_analytical_parameters = "analytical_parameters"
        self.substr_dimensions = "attribute_dim_"
        self.substr_categories = "attribute_"
        self.substr_varreqs = "table_varreqs_by_"
        self.substr_varreqs_allcats = f"{self.substr_varreqs}category_"
        self.substr_varreqs_partialcats = f"{self.substr_varreqs}partial_category_"

        # temporary - but read from table at some point
        self.varchar_str_emission_gas = "$EMISSION-GAS$"
        self.varchar_str_unit_length = "$UNIT-LENGTH$"
        self.varchar_str_unit_energy = "$UNIT-ENERGY$"
        self.varchar_str_unit_mass = "$UNIT-MASS$"
        self.varchar_str_unit_volume = "$UNIT-VOLUME$"

        # add attributes and dimensional information
        self.attribute_directory = dir_attributes
        self.all_pycategories, self.all_dims, self.all_attributes, self.configuration_requirements, self.dict_attributes, self.dict_varreqs = self.load_attribute_tables(dir_attributes)
        self.all_sectors, self.all_sectors_abvs, self.all_subsectors, self.all_subsector_abvs = self.get_sector_dims()
        self.all_subsectors_with_primary_category, self.all_subsectors_without_primary_category = self.get_all_subsectors_with_primary_category()
        self.dict_model_variables_by_subsector, self.dict_model_variable_to_subsector, self.dict_model_variable_to_category_restriction = self.get_variables_by_subsector()
        self.all_model_variables, self.dict_variables_to_model_variables, self.dict_model_variables_to_variables = self.get_variable_fields_by_variable()
        self.all_primary_category_flags = self.get_all_primary_category_flags()

        # miscellaneous parameters that need to be checked before running
        self.field_enfu_electricity_demand_category = "electricity_demand_category"

        # run checks and raise errors if invalid data are found in the attribute tables
        self.check_enfu_attribute_table()
        self.check_inen_enfu_crosswalk()
        self.check_lndu_attribute_tables()
        self.check_trde_category_variable_crosswalk()
        self.check_trns_trde_crosswalks()
        self.check_wali_gnrl_crosswalk()
        self.check_wali_trww_crosswalk()
        self.check_waso_attribute_table()


        # get configuration
        self.configuration = Configuration(
            fp_config,
            self.dict_attributes["unit_energy"],
            self.dict_attributes["emission_gas"],
            self.dict_attributes["unit_length"],
            self.dict_attributes["unit_mass"],
            self.dict_attributes["unit_volume"],
            self.configuration_requirements
        )




    ############################################################
    #   FUNCTIONS FOR ATTRIBUTE TABLES, DIMENSIONS, SECTORS    #
    ############################################################

    ##  function to ensure dimensions of analysis are properly specified
    def check_dimensions_of_analysis(self):
        if not set(self.sort_ordered_dimensions_of_analysis).issubset(set(self.all_dims)):
            missing_vals = sf.print_setdiff(set(self.sort_ordered_dimensions_of_analysis), set(self.all_dims))
            raise ValueError(f"Missing specification of required dimensions of analysis: no attribute tables for dimensions {missing_vals} found in directory '{self.attribute_directory}'.")


    ##  function to ensure a sector is properly specified
    def check_sector(self, sector: str):
        # check sectors
        if sector not in self.all_sectors:
            valid_sectors = sf.format_print_list(self.all_sectors)
            raise ValueError(f"Invalid sector specification '{sector}': valid sectors are {valid_sectors}")


    ##  function to ensure a sector is properly specified
    def check_subsector(self, subsector: str):
        # check sectors
        if subsector not in self.all_subsectors:
            valid_subsectors = sf.format_print_list(self.all_subsectors)
            raise ValueError(f"Invalid subsector specification '{subsector}': valid sectors are {valid_subsectors}")


    ##  simple inline function to dimensions in a data frame (if they are converted to floats)
    def clean_dimension_fields(self, df_in: pd.DataFrame):
        fields_clean = [x for x in self.sort_ordered_dimensions_of_analysis if x in df_in.columns]
        for fld in fields_clean:
            df_in[fld] = np.array(df_in[fld]).astype(int)


    ##  commonly used--restrict variable values
    def check_restricted_value_argument(self, arg, valid_values: list, func_arg: str = "", func_name: str = ""):
        if arg not in valid_values:
            vrts = sf.format_print_list(valid_values)
            raise ValueError(f"Invalid {func_arg} in {func_name}: valid values are {vrts}.")

    ##  get subsectors that have a primary cateogry; these sectors can leverage the functions below effectively
    def get_all_subsectors_with_primary_category(self):
        l_with = list(self.dict_attributes["abbreviation_subsector"].field_maps["subsector_to_primary_category_py"].keys())
        l_with.sort()
        l_without = list(set(self.all_subsectors) - set(l_with))
        l_without.sort()

        return l_with, l_without


    ##  function to return all primary category flags
    def get_all_primary_category_flags(self) -> list:
        all_pcflags = sorted(list(set(self.dict_attributes["abbreviation_subsector"].table["primary_category"])))
        all_pcflags = [x.replace("`", "") for x in all_pcflags if sf.clean_field_names([x])[0] in self.all_pycategories]

        return all_pcflags


    ##  function to simplify retrieval of attribute tables within functions
    def get_attribute_table(self, subsector: str, table_type = "pycategory_primary"):


        if table_type == "pycategory_primary":
            key_dict = self.get_subsector_attribute(subsector, table_type)
            return self.dict_attributes.get(key_dict)
        elif table_type in ["key_varreqs_all", "key_varreqs_partial"]:
            key_dict = self.get_subsector_attribute(subsector, table_type)
            return self.dict_varreqs.get(key_dict)
        else:
            raise ValueError(f"Invalid table_type '{table_type}': valid options are 'pycategory_primary', 'key_varreqs_all', 'key_varreqs_partial'.")


    ##  get the baseline scenario associated with a scenario dimension
    def get_baseline_scenario_id(self, dim: str):

        """
            get_baseline_scenario_id returns the scenario id associated with a baseline scenario (as specified in the attribute table)

            - dim: a scenario dimension specified in an attribute table (attribute_dim_####.csv) within the ModelAttributes class

        """
        if dim not in self.all_dims:
            fpl = sf.format_print_list(self.all_dims)
            raise ValueError(f"Invalid dimension '{dim}': valid dimensions are {fpl}.")

        # get field to check
        field_check = f"baseline_{dim}"
        if field_check not in self.dict_attributes[f"dim_{dim}"].table:
            warnings.warn(f"No baseline specified for dimension '{dim}'.")
            return None
        else:
            tab = self.dict_attributes[f"dim_{dim}"].table
            tab_red = list(tab[tab[field_check] == 1][dim])

            if len(tab_red) > 1:
                raise ValueError(f"Multiple baselines specified for dimension {dim}. Ensure that only baseline is set in the attribute table at '{tab.fp_table}'")

            return tab_red[0]


    ##  function to get all dimensions of analysis in a data frame - can be used on two data frames for merges
    def get_df_dimensions_of_analysis(self, df_in: pd.DataFrame, df_in_shared: pd.DataFrame = None) -> list:
        if type(df_in_shared) == pd.DataFrame:
            cols = [x for x in self.sort_ordered_dimensions_of_analysis if (x in df_in.columns) and (x in df_in_shared.columns)]
        else:
            cols = [x for x in self.sort_ordered_dimensions_of_analysis if x in df_in.columns]
        return cols

    ##  function to return categories from an attribute table that match some characteristics (defined in dict_subset)
    def get_categories_from_attribute_characteristic(self,
        subsector: str,
        dict_subset: dict,
        attribute_type: str = "pycategory_primary"
    ) -> list:
        #
        pycat = self.get_subsector_attribute(subsector, attribute_type)
        attr = self.dict_attributes[pycat] if (attribute_type == "pycategory_primary") else self.dict_varreqs[pycat]
        #
        return list(sf.subset_df(attr.table, dict_subset)[pycat])


    ##  function for dimensional attributes
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


    ##  function to get different dimensions
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


    ##  function to retrieve time periods
    def get_time_periods(self):
        pydim_time_period = self.get_dimensional_attribute("time_period", "pydim")
        time_periods = self.dict_attributes[pydim_time_period].key_values
        return time_periods, len(time_periods)


    ##  function for grabbing an attribute column from an attribute table ordered the same as key values
    def get_ordered_category_attribute(self,
        subsector: str,
        attribute: str,
        attr_type: str = "pycategory_primary",
        skip_none_q: bool = False,
        return_type: type = list,
        clean_attribute_schema_q: bool = False,
    ) -> list:

        valid_return_types = [list, np.ndarray, dict]
        if return_type not in valid_return_types:
            str_valid_types = sf.format_print_list(valid_return_types)
            raise ValueError(f"Invalid return_type '{return_type}': valid types are {str_valid_types}.")

        pycat = self.get_subsector_attribute(subsector, attr_type)
        if attr_type == "pycategory_primary":
            attr_cur = self.dict_attributes[pycat]
        elif attr_type in ["key_varreqs_all", "key_varreqs_partial"]:
            attr_cur = self.dict_varreqs[pycat]
        else:
            raise ValueError(f"Invalid attribute type '{attr_type}': select 'pycategory_primary', 'key_varreqs_all', or 'key_varreqs_partial'.")

        if attribute not in attr_cur.table.columns:
            raise ValueError(f"Missing attribute column '{attribute}': attribute not found in '{subsector}' attribute table.")

        # get the dictionary and order
        tab = attr_cur.table[attr_cur.table[attribute] != "none"] if skip_none_q else attr_cur.table
        dict_map = sf.build_dict(tab[[attr_cur.key, attribute]]) if (not clean_attribute_schema_q) else dict(zip(tab[attr_cur.key], list(tab[attribute].apply(clean_schema))))
        kv = [x for x in attr_cur.key_values if x in list(tab[attr_cur.key])]

        if return_type == dict:
            out = dict_map
        else:
            out = [dict_map[x] for x in kv]
            out = np.array(out) if return_type == np.ndarray else out

        return out


    ##  fuction to return a list of variables from one subsector that are ordered according to a primary category (which the variables are mapped to) from another subsector
    def get_ordered_vars_by_nonprimary_category(self,
        subsector_var: str,
        subsector_targ: str,
        varreq_type: str,
        return_type: str = "vars"
    ):

        # get var requirements for the variable subsector + the attribute for the target categories
        varreq_var = self.get_subsector_attribute(subsector_var, varreq_type)
        pycat_targ = self.get_subsector_attribute(subsector_targ, "pycategory_primary")
        attr_vr_var = self.dict_varreqs[varreq_var]
        attr_targ = self.dict_attributes[pycat_targ]

        # use the attribute table to map the category to the original variable
        tab_for_cw = attr_vr_var.table[attr_vr_var.table[pycat_targ] != "none"]
        vec_var_targs = [clean_schema(x) for x in list(tab_for_cw[pycat_targ])]
        inds_varcats_to_cats = [vec_var_targs.index(x) for x in attr_targ.key_values]

        if return_type == "inds":
            return inds_varcats_to_cats
        elif return_type == "vars":
            vars_ordered = list(tab_for_cw["variable"])
            return [vars_ordered[x] for x in inds_varcats_to_cats]
        else:
            raise ValueError(f"Invalid return_type '{return_type}' in order_vars_by_category: valid types are 'inds', 'vars'.")


    ##  function for retrieving different attributes associated with a sector
    def get_sector_attribute(self, sector: str, return_type: str):

        # check sector specification
        self.check_sector(sector)

        # initialize some key vars
        match_str_to = "sector_to_" if (return_type == "abbreviation_sector") else "abbreviation_sector_to_"
        attr_sec = self.dict_attributes["abbreviation_sector"]
        maps = [x for x in attr_sec.field_maps.keys() if (match_str_to in x)]
        map_retrieve = f"{match_str_to}{return_type}"

        if not map_retrieve in maps:
            valid_rts = sf.format_print_list([x.replace(match_str_to, "") for x in maps])
            # warn user, but still allow a return
            warnings.warn(f"Invalid sector attribute '{return_type}'. Valid return type values are:{valid_rts}")
            return None
        else:
            # set the key
            key = sector if (return_type == "abbreviation_sector") else attr_sec.field_maps["sector_to_abbreviation_sector"][sector]
            sf.check_keys(attr_sec.field_maps[map_retrieve], [key])
            return attr_sec.field_maps[map_retrieve][key]


    ##  function to return a list of subsectors by sector
    def get_sector_subsectors(self, sector: str):

        self.check_sector(sector)

        subsectors = list(
            sf.subset_df(
                self.dict_attributes["abbreviation_subsector"].table,
                {"sector": [sector]}
            )["subsector"]
        )

        return subsectors


    ##  function for retrieving different attributes associated with a subsector
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


    ##  retrieve a dictionary that maps variables to each other based on shared categories within a subsector
    def get_var_dicts_by_shared_category(self,
        subsector:str,
        category_pivot:str,
        fields_to_filter_on:list
    ) -> dict:

        dict_out = {}

        # get available dictionaries
        for table_type in ["key_varreqs_all", "key_varreqs_partial"]:
            # check attribute table
            attr_table = self.get_attribute_table(subsector, table_type)
            if attr_table is not None:
                # get columns available in the data
                cols = list(set(attr_table.table.columns & set(fields_to_filter_on)))
                if len(cols) > 0 & (category_pivot in attr_table.table.columns):
                    for field in cols:
                        df_tmp = attr_table.table[attr_table.table[field] == 1][[category_pivot, "variable"]].copy()
                        df_tmp[category_pivot] = df_tmp[category_pivot].apply(clean_schema)
                        dict_out.update({field: sf.build_dict(df_tmp[[category_pivot, "variable"]])})

        # next, loop over available combinations to build cross dictionaries
        dict_mapping = {}
        keys_to_pair = list(dict_out.keys())
        for pair in list(itertools.combinations(keys_to_pair, 2)):
            # get keys from dict and set keys for dict_mapping
            key_1 = pair[0]
            key_2 = pair[1]
            key_new = f"{key_1}_to_{key_2}"
            key_new_rev = f"{key_2}_to_{key_1}"

            # categories available in both dictionaries are used to update the dict_mapping
            shared_cats = list(set(dict_out[key_1]) & set(dict_out[key_2]))
            dict_mapping.update({
                key_new: dict([(dict_out[key_1][x], dict_out[key_2][x]) for x in shared_cats]),
                key_new_rev: dict([(dict_out[key_2][x], dict_out[key_1][x]) for x in shared_cats])
            })

        return dict_mapping


    ##  function to reorganize a bit to create variable fields associated with each variable
    def get_variable_fields_by_variable(self):
        dict_vars_to_fields = {}
        dict_fields_to_vars = {}
        modvars_all = []
        for subsector in self.all_subsectors_with_primary_category:
            modvars = self.dict_model_variables_by_subsector[subsector]
            modvars.sort()
            modvars_all += modvars
            for var in modvars:
                var_lists = self.build_varlist(subsector, variable_subsec = var)
                dict_vars_to_fields.update({var: var_lists})
                dict_fields_to_vars.update(dict(zip(var_lists, [var for x in var_lists])))

        return modvars_all, dict_fields_to_vars, dict_vars_to_fields


    ##  function to merge an array for a variable with partial categories to all categories
    def merge_array_var_partial_cat_to_array_all_cats(self,
        array_vals: np.ndarray,
        modvar: str,
        missing_vals: float = 0.0,
        output_cats: list = None,
        output_subsec: str = None
    ) -> np.ndarray:
        """
            Reformat a partial category array (with partical categories along columns) to place columns appropriately for a full category array. Useful for simplifying matrix operations between variables.

            - array_vals: input array of data with column categories

            - modvar: the variable associated with the *input* array. This is used to identify which categories are represented in the array's columns. If None, then output_cats and output_subsec must be specified.

            - missing_vals: values to set for categories not in array_vals. Default is 0.0.

            - output_cats: vector of categories associated with the output variable. Only used if modvar == None. The combination of output_cats + output_subsec provide a manual override to the modvar option.

            - output_subsec: output subsector. Default is None. Only used if modvar == None. The combination of output_cats + output_subsec provide a manual override to the modvar option.
        """

        # check inputs
        if (type(modvar) == type(None)) and (type(None) in [type(output_cats), type(output_subsec)]):
            raise ValueError(f"Error in input specification. If modvar == None, then output_cats and output_subsec cannot be None.")
        if not type(missing_vals) in [int, float, np.float64, np.int64]:
            raise ValueError(f"Error in input specification of missing_vals: missing_vals should be a floating point number of integer.")

        # get subsector/categories information
        if type(modvar) != type(None):
            # check variable first
            if modvar not in self.all_model_variables:
                raise ValueError(f"Invalid model variable '{modvar}' found in get_variable_characteristic.")

            subsector = self.get_variable_subsector(modvar)
            attr_subsec = self.get_attribute_table(subsector)
            cat_restriction_type = self.dict_model_variable_to_category_restriction[modvar]
        else:
            subsector = output_subsec
            attr_subsec = self.get_attribute_table(subsector)
            cat_restriction_type = None
            # check that all categories are defined
            if not set(output_cats).issubset(set(attr_subsec.key_values)):
                invalid_values = sf.format_print_list(list(set(output_cats) - set(attr_subsec.key_values)))
                raise ValueError(f"Error in merge_array_var_partial_cat_to_array_all_cats: Invalid categories {invalid_values} specified for subsector {subsector} in output_cats.")
            # check that all categories are unique
            if len(set(output_cats)) != len(output_cats):
                raise ValueError(f"Error in merge_array_var_partial_cat_to_array_all_cats: Categories specified in output_cats are not unique. Check that categories are unique.")

        # return the array if all categories are specified
        if cat_restriction_type == "all":
            return array_vals
        else:
            array_default = np.ones((len(array_vals), attr_subsec.n_key_values))*missing_vals
            cats = self.get_variable_categories(modvar) if (type(modvar) != type(None)) else output_cats
            inds_cats = [attr_subsec.get_key_value_index(x) for x in cats]
            inds = np.repeat([inds_cats], len(array_default), axis = 0)
            np.put_along_axis(array_default, inds, array_vals, axis = 1)

            return array_default


    ##  function to merge an array for a variable with partial categories to all categories
    def reduce_all_cats_array_to_partial_cat_array(self, array_vals: np.ndarray, modvar: str) -> np.ndarray:
        """
            Reduce an all category array (with all categories along columns) to columns associated with the variable modvar. Inverse of merge_array_var_partial_cat_to_array_all_cats.

            - array_vals: input array of data with column categories

            - modvar: the variable associated with the desired *output* array. This is used to identify which categories should be selected.
        """

        # check variable first
        if modvar not in self.all_model_variables:
            raise ValueError(f"Invalid model variable '{modvar}' found in get_variable_characteristic.")

        subsector = self.get_variable_subsector(modvar)
        attr_subsec = self.get_attribute_table(subsector)
        cat_restriction_type = self.dict_model_variable_to_category_restriction[modvar]

        if cat_restriction_type == "all":
            return array_vals
        else:
            cats = self.get_variable_categories(modvar)
            inds_cats = [attr_subsec.get_key_value_index(x) for x in cats]
            return array_vals[:, inds_cats]


    ##  function to retrieve and format attribute tables for use
    def load_attribute_tables(self, dir_att):
        # get available types
        all_types = [x for x in os.listdir(dir_att) if (self.attribute_file_extension in x) and ((self.substr_categories in x) or (self.substr_varreqs_allcats in x) or (self.substr_varreqs_partialcats in x) or (self.substr_analytical_parameters in x))]
        all_pycategories = []
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
                all_pycategories.append(nm)
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

        return (all_pycategories, all_dims, all_types, configuration_requirements, dict_attributes, dict_varreqs)




    #########################################################################
    #    QUICK RETRIEVAL OF FUNDAMENTAL TRANSFORMATIONS (GWP, MASS, ETC)    #
    #########################################################################

    ##  function to get energy equivalent scalar
    def get_energy_equivalent(self, energy: str, energy_to_match: str = None):

        """
            for a given energy unit *energy*, get the scalar to convert to units *energy_to_match*
            - energy: a unit of energy defined in the unit_energy attribute table

            - energy_to_match: Default is None. A unit of energy to match. The scalar a that is returned is multiplied by energy, i.e., energy*a = energy_to_match. If None (default), return the configuration default.
        """
        # none checks
        if energy is None:
            return None

        # get the valid values
        valid_vals = sf.format_print_list(self.dict_attributes["unit_energy"].key_values)

        if energy_to_match is None:
            energy_to_match = str(self.configuration.get("energy_units")).lower()
        key_dict = f"unit_energy_to_energy_equivalent_{energy_to_match}"

        # check that the target energy unit is defined
        if not key_dict in self.dict_attributes["unit_energy"].field_maps.keys():
            raise KeyError(f"Invalid energy target '{energy_to_match}': defined energy units are {valid_vals}.")

        # check that the target energy unit is defined
        if energy in self.dict_attributes["unit_energy"].field_maps[key_dict].keys():
            return self.dict_attributes["unit_energy"].field_maps[key_dict][energy]
        else:
            raise KeyError(f"Invalid energy '{energy}': defined energy units are {valid_vals}.")


    ##  function to get gwp multiplier associated with a gas
    def get_gwp(self, gas: str, gwp: int = None):
        """
            for a given gas, get the scalar to convert to CO2e using the specified global warming potential *gwp*
            - gas: a gas defined in the emission_gas attribute table

            - gwp: Default is None. A unit of energy to match. The scalar a that is returned is multiplied by energy, i.e., energy*a = energy_to_match. If None (default), return the configuration default.
        """
        # none checks
        if gas is None:
            return None

        if gwp is None:
            gwp = int(self.configuration.get("global_warming_potential"))
        key_dict = f"emission_gas_to_global_warming_potential_{gwp}"

        # check that the target energy unit is defined
        if not key_dict in self.dict_attributes["emission_gas"].field_maps.keys():
            valid_gwps = sf.format_print_list(self.configuration.valid_gwp)
            raise KeyError(f"Invalid GWP '{gwp}': defined global warming potentials are {valid_gwps}.")
        # check gas and return if valid
        if gas in self.dict_attributes["emission_gas"].field_maps[key_dict].keys():
            return self.dict_attributes["emission_gas"].field_maps[key_dict][gas]
        else:
            valid_gasses = sf.format_print_list(self.dict_attributes["emission_gas"].key_values)
            raise KeyError(f"Invalid gas '{gas}': defined gasses are {valid_gasses}.")


    ##  function to get the length equivalent scalar
    def get_length_equivalent(self, length: str, length_to_match: str = None):
        """
            for a given lanegh unit *length*, get the scalar to convert to units *length_to_match*

            Function Arguments
            ------------------
            length: a unit of length defined in the unit_length attribute table

            length_to_match: Default is None. A unit of length to match. The scalar a that is returned is multiplied by length, i.e., length*a = length_to_match. If None (default), return the configuration default.
        """
        # none checks
        if length is None:
            return None

        if length_to_match is None:
            length_to_match = str(self.configuration.get("length_units")).lower()
        key_dict = f"unit_length_to_length_equivalent_{length_to_match}"

        # check that the target length unit is defined
        if not key_dict in self.dict_attributes["unit_length"].field_maps.keys():
            valid_lengths_to_match = sf.format_print_list(self.configuration.valid_length).lower()
            raise KeyError(f"Invalid length to match '{length_to_match}': defined length units to match are {valid_lengths_to_match}.")

        # check length and return if valid
        if length in self.dict_attributes["unit_length"].field_maps[key_dict].keys():
            return self.dict_attributes["unit_length"].field_maps[key_dict][length]
        else:
            valid_vals = sf.format_print_list(self.dict_attributes["unit_length"].key_values)
            raise KeyError(f"Invalid length '{length}': defined lengths are {valid_vals}.")


    ##  function to get the mass equivalent scalar
    def get_mass_equivalent(self, mass: str, mass_to_match: str = None):
        """
            for a given mass unit *mass*, get the scalar to convert to units *mass_to_match*

            Function Arguments
            ------------------
            mass: a unit of mass defined in the unit_mass attribute table

            mass_to_match: Default is None. A unit of mass to match. The scalar a that is returned is multiplied by mass, i.e., mass*a = mass_to_match. If None (default), return the configuration default.
        """

        # none checks
        if mass is None:
            return None

        if mass_to_match is None:
            mass_to_match = str(self.configuration.get("emissions_mass")).lower()
        key_dict = f"unit_mass_to_mass_equivalent_{mass_to_match}"

        # check that the target mass unit is defined
        if not key_dict in self.dict_attributes["unit_mass"].field_maps.keys():
            valid_masses_to_match = sf.format_print_list(self.configuration.valid_mass).lower()
            raise KeyError(f"Invalid mass to match '{mass_to_match}': defined mass units to match are {valid_masses_to_match}.")

        # check mass and return if valid
        if mass in self.dict_attributes["unit_mass"].field_maps[key_dict].keys():
            return self.dict_attributes["unit_mass"].field_maps[key_dict][mass]
        else:
            valid_vals = sf.format_print_list(self.dict_attributes["unit_mass"].key_values)
            raise KeyError(f"Invalid mass '{mass}': defined masses are {valid_vals}.")


    ##  function to get a volume equivalent scalar
    def get_volume_equivalent(self, volume: str, volume_to_match: str = None):
        """
            for a given volume unit *volume*, get the scalar to convert to units *volume_to_match*
            - volume: a unit of volume defined in the unit_volume attribute table

            - volume_to_match: Default is None. A unit of volume to match. The scalar a that is returned is multiplied by volume, i.e., volume*a = volume_to_match. If None (default), return the configuration default.
        """
        # none checks
        if volume is None:
            return None

        if volume_to_match is None:
            volume_to_match = str(self.configuration.get("volume_units")).lower()
        key_dict = f"unit_volume_to_volume_equivalent_{volume_to_match}"

        # check that the target mass unit is defined
        if not key_dict in self.dict_attributes["unit_volume"].field_maps.keys():
            valid_volume_to_match = sf.format_print_list(self.configuration.valid_volume).lower()
            raise KeyError(f"Invalid volume to match '{volume_to_match}': defined volume units to match are {valid_volume_to_match}.")

        if volume in self.dict_attributes["unit_volume"].field_maps[key_dict].keys():
            return self.dict_attributes["unit_volume"].field_maps[key_dict][volume]
        else:
            valid_vols = sf.format_print_list(self.dict_attributes["unit_volume"].key_values)
            raise KeyError(f"Invalid volume '{volume}': defined volumes are {valid_vols}.")

    # get scalar
    def get_scalar(self, modvar: str, return_type: str = "total"):

        valid_rts = ["total", "gas", "length", "mass", "energy", "volume"]
        if return_type not in valid_rts:
            tps = sf.format_print_list(valid_rts)
            raise ValueError(f"Invalid return type '{return_type}' in get_scalar: valid types are {tps}.")

        # get scalars
        energy = self.get_variable_characteristic(modvar, self.varchar_str_unit_energy)
        scalar_energy = 1 if not energy else self.get_energy_equivalent(energy.lower())
        #
        gas = self.get_variable_characteristic(modvar, self.varchar_str_emission_gas)
        scalar_gas = 1 if not gas else self.get_gwp(gas.lower())
        #
        length = self.get_variable_characteristic(modvar, self.varchar_str_unit_length)
        scalar_length = 1 if not length else self.get_length_equivalent(length.lower())
        #
        mass = self.get_variable_characteristic(modvar, self.varchar_str_unit_mass)
        scalar_mass = 1 if not mass else self.get_mass_equivalent(mass.lower())
        #
        volume = self.get_variable_characteristic(modvar, self.varchar_str_unit_volume)
        scalar_volume = 1 if not volume else self.get_volume_equivalent(volume.lower())


        if return_type == "energy":
            out = scalar_energy
        elif return_type == "gas":
            out = scalar_gas
        elif return_type == "length":
            out = scalar_length
        elif return_type == "mass":
            out = scalar_mass
        elif return_type == "volume":
            out = scalar_volume
        elif return_type == "total":
            # total is used for scaling gas & mass to co2e in proper units
            out = scalar_gas*scalar_mass

        return out


    ####################################################
    #    SECTOR-SPECIFIC AND CROSS SECTORIAL CHECKS    #
    ####################################################

    ##  function to check attribute crosswalks (e.g., one attribute table specifies another category as an element; this function verifies that they are valid)
    def check_subsector_attribute_table_crosswalk(self,
        dict_subsector_primary: dict,
        subsector_target: str,
        type_primary: str = "categories",
        type_target: str = "categories",
        injection_q: bool = True
    ):
        """
            Checks the validity of categories specified as an attribute (subsector_target) of a primary subsctor category (subsector_primary)

            Function Arguments
            ------------------
            dict_subsector_primary: dictionary of form {subsector_primary: field_attribute_target}. The key gives the primary subsector, and 'field_attribute_target' is the field in the attribute table associated with the categories to check.
                NOTE: dict_subsector_primary can also be specified only as a string (subsector_primary) -- if dict_subsector_primary is a string, then field_attribute_target is assumed to be the primary python category of subsector_target (e.g., $CAT-TARGET$)

            subsector_target: target subsector to check values against

            type_primary: default = "categories". Represents the type of attribute table for the primary table; valid values are 'categories', 'varreqs_all', and 'varreqs_partial'

            type_target: default = "categories". Type of the target table. Valid values are the same as those for type_primary.

            injection_q: default = True. If injection_q, then target categories should be associated with a unique primary category (exclding those are specified as 'none').
        """

        ##  RUN CHECKS ON INPUT SPECIFICATIONS

        # check type specifications
        dict_valid_types_to_attribute_keys = {
            "categories": "pycategory_primary",
            "varreqs_all": "key_varreqs_all",
            "varreqs_partial": "key_varreqs_partial"
        }
        valid_types = list(dict_valid_types_to_attribute_keys.keys())
        str_valid_types = sf.format_print_list(valid_types)
        if type_primary not in valid_types:
            raise ValueError(f"Invalid type_primary '{type_primary}' specified. Valid values are '{str_valid_types}'.")
        if type_target not in valid_types:
            raise ValueError(f"Invalid type_target '{type_target}' specified. Valid values are '{str_valid_types}'.")

        # get the primary subsector + field, then run checks
        if type(dict_subsector_primary) == dict:
            if len(dict_subsector_primary) != 1:
                raise KeyError(f"Error in dictionary dict_subsector_primary: only one key (subsector_primary) should be specified.")
            subsector_primary = list(dict_subsector_primary.keys())[0]
        elif type(dict_subsector_primary) == str:
            subsector_primary = dict_subsector_primary
        else:
            t_str = str(type(dict_subsector_primary))
            raise ValueError(f"Invalid type '{t_str}' of dict_subsector_primary: 'dict' and 'str' are acceptable values.")
        # check that the subsectors are valid
        self.check_subsector(subsector_primary)
        self.check_subsector(subsector_target)

        # check primary table type and fetch attribute
        dict_tables_primary = self.dict_attributes if (type_primary == "categories") else self.dict_varreqs
        key_primary = self.get_subsector_attribute(subsector_primary, dict_valid_types_to_attribute_keys[type_primary])
        if not key_primary:
            raise ValueError(f"Invalid type_primary '{type_primary}' specified for primary subsector '{subsector_primary}': type not found.")
        attr_prim = dict_tables_primary[key_primary]

        # check target table type and fetch attribute
        dict_tables_primary = self.dict_attributes if (type_target == "categories") else self.dict_varreqs
        key_target = self.get_subsector_attribute(subsector_target, dict_valid_types_to_attribute_keys[type_target])
        key_target_pycat = self.get_subsector_attribute(subsector_target, "pycategory_primary")
        if not key_primary:
            raise ValueError(f"Invalid type_primary '{type_target}' specified for primary subsector '{subsector_target}': type not found.")
        attr_targ = dict_tables_primary[key_target]

        # check that the field is properly specified in the primary table
        field_subsector_primary = str(dict_subsector_primary[subsector_primary]) if (type(dict_subsector_primary) == dict) else key_target
        if field_subsector_primary not in attr_prim.table.columns:
            raise ValueError(f"Error in check_subsector_attribute_table_crosswalk: field '{field_subsector_primary}' not found in the '{subsector_primary}' attribute table. Check the file at '{attr_prim.fp_table}'.")


        ##  CHECK ATTRIBUTE TABLE CROSSWALKS

        # get categories specified in the
        primary_cats_defined = list(attr_prim.table[field_subsector_primary])
        primary_cats_defined = [clean_schema(x) for x in primary_cats_defined if (x != "none")] if (key_target == key_target_pycat) else [x for x in primary_cats_defined if (x != "none")]

        # ensure that all population categories properly specified
        if not set(primary_cats_defined).issubset(set(attr_targ.key_values)):
            valid_vals = sf.format_print_list(set(attr_targ.key_values))
            invalid_vals = sf.format_print_list(list(set(primary_cats_defined) - set(attr_targ.key_values)))
            raise ValueError(f"Invalid categories {invalid_vals} specified in field '{field_subsector_primary}' of the {subsector_primary} attribute table at '{attr_prim.fp_table}'.\n\nValid categories from {subsector_target} are: {valid_vals}")

        if injection_q:
            # check that domestic wastewater categories are mapped 1:1 to a population category
            if len(set(primary_cats_defined)) != len(primary_cats_defined):
                duplicate_vals = sf.format_print_list(set([x for x in primary_cats_defined if primary_cats_defined.count(x) > 1]))
                raise ValueError(f"Error in {subsector_primary} attribute table at '{attr_prim.fp_table}': duplicate specifications of target categories {duplicate_vals}. There map of {subsector_primary} categories to {subsector_target} categories should be an injection map.")


    ##  function to check the energy fuels table to ensure that an electricity category is specified
    def check_enfu_attribute_table(self):
        # some shared values
        subsec = "Energy Fuels"
        attr = self.get_attribute_table(subsec)

        if not (set(attr.table[self.field_enfu_electricity_demand_category].astype(int)) == set([1 , 0]) and (sum(attr.table[self.field_enfu_electricity_demand_category]) == 1)):
            raise ValueError(f"Error in subsector {subsec}: there must be a unique electricity category specified in fuels. Check the field '{self.field_enfu_electricity_demand_category}' in the table at '{attr.fp_table}'.")


    ##  function to check the liquid waste/population crosswalk in liquid waste
    def check_inen_enfu_crosswalk(self):
        return self.check_subsector_attribute_table_crosswalk("Industrial Energy", "Energy Fuels", type_primary = "varreqs_partial", injection_q = False)


    ##  function to check that the land use attribute tables are specified
    def check_lndu_attribute_tables(self):

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


    ##  function to check the variables specified in the Transportation Demand attribute table
    def check_trde_category_variable_crosswalk(self):
        self.check_subsector_attribute_table_crosswalk(
            "Transportation Demand",
            "Transportation Demand",
            type_primary = "categories",
            type_target = "varreqs_partial",
            injection_q = False
        )


    ##  function to check the transportation/transportation demand crosswalk in both the attribute table and the varreqs table
    def check_trns_trde_crosswalks(self):
        self.check_subsector_attribute_table_crosswalk("Transportation", "Transportation Demand", type_primary = "varreqs_partial", injection_q = True)
        self.check_subsector_attribute_table_crosswalk("Transportation", "Transportation Demand", injection_q = False)


    ##  function to check the liquid waste/population crosswalk in liquid waste
    def check_wali_gnrl_crosswalk(self):
        self.check_subsector_attribute_table_crosswalk("Liquid Waste", "General", injection_q = True)


    ##  liquid waste/wastewater crosswalk
    def check_wali_trww_crosswalk(self):
        self.check_subsector_attribute_table_crosswalk("Liquid Waste", "Wastewater Treatment", type_primary = "varreqs_all")


    ##  function to check if the solid waste attribute table is properly defined
    def check_waso_attribute_table(self):
        # check that only one category is assocaited with sludge
        attr_waso = self.get_attribute_table("Solid Waste")
        cats_sludge = self.get_categories_from_attribute_characteristic("Solid Waste", {"sewage_sludge_category": 1})
        if len(cats_sludge) > 1:
            raise ValueError(f"Error in Solid Waste attribute table at {attr_waso.fp_table}: multiple sludge categories defined in the 'sewage_sludge_category' field. There should be no more than 1 sewage sludge category.")


    ##  function to check the projection input dataframe and (1) return time periods available, (2) a dicitonary of scenario dimenions, and (3) an interpolated data frame if there are missing values.
    def check_projection_input_df(self,
        df_project: pd.DataFrame,
        # options for formatting the input data frame to correct for errors
        interpolate_missing_q: bool = True,
        strip_dims: bool = True,
        drop_invalid_time_periods: bool = True
    ) -> tuple:
        # check for required fields
        sf.check_fields(df_project, [self.dim_time_period])

        # field initialization
        fields_dat = [x for x in df_project.columns if (x not in self.sort_ordered_dimensions_of_analysis)]
        fields_dims_notime = [x for x in self.sort_ordered_dimensions_of_analysis if (x != self.dim_time_period) and (x in df_project.columns)]

        # check that there's only one primary key included (or one dimensional vector)
        if len(fields_dims_notime) > 0:
            df_fields_dims_notime = df_project[fields_dims_notime].drop_duplicates()
            if len(df_fields_dims_notime) > 1:
                raise ValueError(f"Error in project: the input data frame contains multiple dimensions of analysis. The project method is restricted to a single dimension of analysis. The following dimensions were found:\n{df_fields_dims_notime}")
            else:
                dict_dims = dict(zip(fields_dims_notime, list(df_fields_dims_notime.iloc[0])))
        else:
            dict_dims = {}

        # next, check time periods
        df_time = self.dict_attributes["dim_time_period"].table[[self.dim_time_period]]
        set_times_project = set(df_project[self.dim_time_period])
        set_times_defined = set(df_time[self.dim_time_period])
        set_times_keep = set_times_project & set_times_defined

        # raise errors if issues occur
        if (not set_times_project.issubset(set_times_defined)) and (not drop_invalid_time_periods):
            sf.check_set_values(set_times_project, set_times_defined, " in projection dataframe. Set 'drop_invalid_time_periods = True' to drop these time periods and proceed.")

        # intiialize interpolation_q and check for consecutive time steps to determine if a merge + interpolation is needed
        interpolate_q = False

        if (set_times_keep != set(range(min(set_times_keep), max(set_times_keep) + 1))):
            if not interpolate_missing_q:
                raise ValueError(f"Error in specified times: some time periods are missing and interpolate_missing_q = False. Modeling will not proceed. Set interpolate_missing_q = True to interpolate missing values.")
            else:
                set_times_keep = set(range(min(set_times_keep), max(set_times_keep) + 1))
                df_project = pd.merge(
                    df_time[df_time[self.dim_time_period].isin(set_times_keep)],
                    df_project,
                    how = "left",
                    on = [self.dim_time_period]
                )
                interpolate_q = True

        elif len(df_project[fields_dat].dropna()) != len(df_project):
                interpolate_q = True

        # set some information on time series
        projection_time_periods = list(set_times_keep)
        projection_time_periods.sort()
        n_projection_time_periods = len(projection_time_periods)

        # format data frame
        df_project = df_project.interpolate() if interpolate_q else df_project
        df_project = df_project[df_project[self.dim_time_period].isin(set_times_keep)]
        df_project.sort_values(by = [self.dim_time_period], inplace = True)
        df_project = df_project[[self.dim_time_period] + fields_dat] if strip_dims else df_project[fields_dims_notime + [self.dim_time_period] + fields_dat]

        return dict_dims, df_project, n_projection_time_periods, projection_time_periods


    ##  function transfer variables from one data frame (source) to another (target)
    def transfer_df_variables(self,
        df_target: pd.DataFrame,
        df_source: pd.DataFrame,
        variables_transfer: list,
        join_type: str = "concatenate"
    ) -> pd.DataFrame:

        vars_extract = [df_target]
        for var_int in variables_transfer:
            df_ext = self.get_optional_or_integrated_standard_variable(df_source, var_int, None)
            if type(df_ext) != type(None):
                vars_extract.append(df_ext[1])

        return sf.merge_output_df_list(vars_extract, self, join_type)



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
    def array_to_df(self,
        arr_in: np.ndarray,
        modvar: str,
        include_scalars = False,
        reduce_from_all_cats_to_specified_cats = False
    ) -> pd.DataFrame:
        """
            use array_to_df to convert an input np.ndarray into a data frame that has the proper variable labels (ordered by category for the appropriate subsector)

            - arr_in: np.ndarray to convert to data frame. If entered as a vector, it will be converted to a (n x 1) array, where n = len(arr_in)

            - modvar: the name of the model variable to use to name the dataframe

            - include_scalars: default = False. If True, will rescale to reflect emissions mass correction.

            - reduce_from_all_cats_to_specified_cats: default = False. If True, the input data frame is given across all categories and needs to be reduced to the set of categories associated with the model variable (selects subset of columns).

        """

        # get subsector and fields to name based on variable
        subsector = self.dict_model_variable_to_subsector[modvar]
        fields = self.build_varlist(subsector, variable_subsec = modvar)
        # transpose if needed
        arr_in = np.array([arr_in]).transpose() if (len(arr_in.shape) == 1) else arr_in

        # is the array that's being passed column-wise associated with all categories?
        if reduce_from_all_cats_to_specified_cats:
            attr = self.get_attribute_table(subsector)
            cats = self.get_variable_categories(modvar)
            indices = [attr.get_key_value_index(x) for x in cats]
            arr_in = arr_in[:, indices]

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


    ##  function to assign keys (e.g., variables in a variable table) based on collections of attribute fields (e.g., a secondary category)
    def assign_keys_from_attribute_fields(self,
        subsector: str,
        field_attribute: str,
        dict_assignment: dict,
        type_table: str = "categories",
        clean_field_vals: bool = True,
        clean_attr_key: bool = False,
    ) -> tuple:
        """
            use assign_keys_from_categories() to assign key_values that are associated with a secondary category. Use matchstrings defined in dict_assignment to create an output dictionary

            Output Format
            -------------
            tuple: (dict_out, vars_unassigned)
            dict_out -> {key_value: {assigned_dictionary_key: variable_name, ...}}


            Function Arguments
            ------------------
            subsector: the subsector to pull the attribute table from

            field_attribute: field in the attribute table to use to split elements

            dict_assignment: dict. {match_str: assigned_dictionary_key} map a variable match string to an assignment

            type_table: default = "categories". Represents the type of attribute table; valid values are 'categories', 'varreqs_all', and 'varreqs_partial'

            clean_field_vals: default = True. Apply clean_schema() to the values found in attr_subsector[field_attribute]?

            clean_attr_key: default is False. Apply clean_schema() to the keys that are assigned to the output dictionary (e.g., clean_schema(variable_name))

        """

        # check the subsector
        self.check_subsector(subsector)
        # check type specifications
        dict_valid_types_to_attribute_keys = {
            "categories": "pycategory_primary",
            "varreqs_all": "key_varreqs_all",
            "varreqs_partial": "key_varreqs_partial"
        }
        valid_types = list(dict_valid_types_to_attribute_keys.keys())
        str_valid_types = sf.format_print_list(valid_types)
        if type_table not in valid_types:
            raise ValueError(f"Invalid type_primary '{type_primary}' specified. Valid values are '{str_valid_types}'.")

        # retrieve the attribute table and check the field specification
        attr_subsector = self.get_attribute_table(subsector, dict_valid_types_to_attribute_keys[type_table])
        sf.check_fields(attr_subsector.table, [field_attribute])

        # get the unique field values
        all_field_values = list(set(
            self.get_ordered_category_attribute(
                subsector,
                field_attribute,
                skip_none_q = True,
                attr_type = dict_valid_types_to_attribute_keys[type_table]
            )
        ))
        all_field_values.sort()

        # loop to build the output dictionaries
        dict_out = {}
        dict_vals_unassigned = {}

        for val in all_field_values:
            dict_out_key = clean_schema(val) if clean_field_vals else val
            subsec_keys = attr_subsector.table[attr_subsector.table[field_attribute] == val][attr_subsector.key]
            # loop over the keys to assign
            dict_assigned = {}
            for subsec_key in subsec_keys:
                for k in dict_assignment.keys():
                    if k in subsec_key:
                        val_assigned = clean_schema(subsec_key) if clean_attr_key else subsec_key
                        dict_assigned.update({dict_assignment[k]: val_assigned})

            dict_out.update({dict_out_key: dict_assigned})
            dict_vals_unassigned.update({dict_out_key: list(set(dict_assignment.values()) - set(dict_assigned.keys()))})

        return dict_out, dict_vals_unassigned

    ##  support function for assign_keys_from_attribute_fields
    def get_vars_by_assigned_class_from_akaf(self, dict_in: dict, var_class: str) -> list:
         return [x.get(var_class) for x in dict_in.values() if (x.get(var_class) is not None)]


    ##  function to build a sampling range dataframe from defaults
    def build_default_sampling_range_df(self):
        df_out = []
        # set field names
        pd_max = max(self.get_time_periods()[0])
        field_max = f"max_{pd_max}"
        field_min = f"min_{pd_max}"

        for sector in self.all_sectors:
            subsectors_cur = list(sf.subset_df(self.dict_attributes["abbreviation_subsector"].table, {"sector": [sector]})["subsector"])

            for subsector in subsectors_cur:
                for variable in self.dict_model_variables_by_subsector[subsector]:
                    variable_type = self.get_variable_attribute(variable, "variable_type")
                    variable_calculation = self.get_variable_attribute(variable, "internal_model_variable")
                    # check that variables are input/not calculated internally
                    if (variable_type.lower() == "input") & (variable_calculation == 0):
                        max_ftp_scalar = self.get_variable_attribute(variable, "default_lhs_scalar_maximum_at_final_time_period")
                        min_ftp_scalar = self.get_variable_attribute(variable, "default_lhs_scalar_minimum_at_final_time_period")
                        mvs = self.dict_model_variables_to_variables[variable]

                        df_out.append(pd.DataFrame({"variable": mvs, field_max: [max_ftp_scalar for x in mvs], field_min: [min_ftp_scalar for x in mvs]}))

        return pd.concat(df_out, axis = 0).reset_index(drop = True)

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
        cats_source = self.get_variable_categories(modvar_source)
        # build the target variable list using the source categories
        subsector_target = self.dict_model_variable_to_subsector[modvar_target]
        vars_target = self.build_varlist(subsector_target, variable_subsec = modvar_target, restrict_to_category_values = cats_source)

        return vars_target


    ##  function for building a list of variables (fields) for data tables
    def build_varlist(
        self,
        subsector: str,
        variable_subsec: str = None,
        restrict_to_category_values: list = None,
        dict_force_override_vrp_vvs_cats: dict = None,
        variable_type: str = None
    ) -> list:
        """

        Build a list of fields (complete variable schema from a data frame) based on the subsector and variable name.

        Function Arguments
        ------------------
        subsector: str, the subsector to build the variable list for.

        variable_subsec: default is None. If None, then builds varlist of all variables required for this variable.

        restrict_to_category_values: default is None. If None, applies to all categories specified in attribute tables. Otherwise, will restrict to specified categories.

        dict_force_override_vrp_vvs_cats: dict_force_override_vrp_vvs_cats can be set do a dictionary of the form
            {MODEL_VAR_NAME: [catval_a, catval_b, catval_c, ... ]}
            where catval_i are not all unique; this is useful for making a variable that maps unique categories to a subset of non-unique categories that represent proxies (e.g., buffalo -> cattle_dairy, )

        variable_type: input or output. If None, defaults to input.

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


    ##  clean a partial category dictionary to return either none (no categorization) or a list of applicable cateogries
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


    ##  function for getting input/output fields for a list of subsectors
    def get_input_output_fields(self, subsectors_inuired: list, build_df_q = False):
        # initialize output lists
        vars_out = []
        vars_in = []
        subsectors_out = []
        subsectors_in = []

        for subsector in subsectors_inuired:
            vars_subsector_in = self.build_varlist(subsector, variable_type = "input")
            vars_subsector_out = self.build_varlist(subsector, variable_type = "output")
            vars_in += vars_subsector_in
            vars_out += vars_subsector_out
            if build_df_q:
                subsectors_out += [subsector for x in vars_subsector_out]
                subsectors_in += [subsector for x in vars_subsector_in]

        if build_df_q:
            vars_in = pd.DataFrame({"subsector": subsectors_in, "variable": vars_in}).sort_values(by = ["subsector", "variable"]).reset_index(drop = True)
            vars_out = pd.DataFrame({"subsector": subsectors_out, "variable": vars_out}).sort_values(by = ["subsector", "variable"]).reset_index(drop = True)

        return vars_in, vars_out


    ##  function to retrive multiple variables that, across categories, must sum to some value. Gives a correction threshold to allow for small errors
    def get_multivariables_with_bounded_sum_by_category(self,
        df_in: pd.DataFrame,
        modvars: list,
        sum_restriction: float,
        correction_threshold: float = 0.000001,
        force_sum_equality: bool = False,
        msg_append: str = ""
    ) -> dict:

        """
            use get_multivariables_with_bounded_sum_by_category() to retrieve a array or data frame of input variables. If return_type == "array_units_corrected", then the ModelAttributes will re-scale emissions factors to reflect the desired output emissions mass (as defined in the configuration)

            - df_in: data frame containing input variables

            - modvars: variables to sum over and restrict

            - sum_restriction: maximium sum that array may equal

            - correction_threshold: tolerance for correcting categories that

            - force_sum_equality: default is False. If True, will force the sum to equal one (overrides correction_threshold)

            - msg_append: use to passage an additional error message to support troubleshooting

        """
        # retrieve arrays
        arr = 0
        init_q = True
        dict_arrs = {}
        for modvar in modvars:
            if modvar not in self.dict_model_variables_to_variables.keys():
                raise ValueError(f"Invalid variable specified in get_standard_variables: variable '{modvar}' not found.")
            else:
                # some basic info
                subsector_cur = self.get_variable_subsector(modvar)
                cats = self.get_variable_categories(modvar)

                if init_q:
                    subsector = subsector_cur
                    init_q = False
                elif subsector_cur != subsector:
                    raise ValueError(f"Error in get_multivariables_with_bounded_sum_by_category: variables must be from the same subsector.")
                # get current variable, merge to all categories, update dictionary, and check totals
                arr_cur = self.get_standard_variables(df_in, modvar, True, "array_base")
                if cats:
                    arr_cur = self.merge_array_var_partial_cat_to_array_all_cats(arr_cur, modvar)

                dict_arrs.update({modvar: arr_cur})
                arr += arr_cur

        if force_sum_equality:
            for modvar in modvars:
                arr_cur = dict_arrs[modvar]
                arr_cur = np.nan_to_num(arr_cur/arr, 0.0)
                dict_arrs.update({modvar: arr_cur})
        else:
            # correction sums if within correction threshold
            w = np.where(arr > sum_restriction + correction_threshold)[0]
            if len(w) > 0:
                raise ValueError(f"Invalid summations found: some categories exceed the sum threshold.{msg_append}")

            w = np.where((arr <= sum_restriction + correction_threshold) & (arr > sum_restriction))[0]
            if len(w) > 0:
                if np.max(sums - sum_restriction) <= correction_threshold:
                    w = np.where((sums <= sum_restriction + correction_threshold) & (sums > sum_restriction))
                    inds = w[0]*len(arr[0]) + w[1]
                    for modvar in modvars:
                        arr_cur = dict_arrs[modvar]
                        np.put(arr_cur, inds, arr_cur[w[0], w[1]].flatten()/arr_cur[w[0], w[1]].flatten())
                        dict_arrs.update({modvar: arr_cur})

        return dict_arrs


    ##  function to return an optional variable if another (integrated) variable is not passed
    def get_optional_or_integrated_standard_variable(self,
        df_in: pd.DataFrame,
        var_integrated: str,
        var_optional: str,
        override_vector_for_single_mv_q: bool = False,
        return_type: str = "data_frame",
        var_bounds = None,
        force_boundary_restriction: bool = True
    ) -> tuple:
        # get fields needed
        subsector_integrated = self.get_variable_subsector(var_integrated)
        fields_check = self.build_varlist(subsector_integrated, var_integrated)
        # check and return the output variable + which variable was selected
        if set(fields_check).issubset(set(df_in.columns)):
            out = self.get_standard_variables(df_in, var_integrated, override_vector_for_single_mv_q, return_type, var_bounds, force_boundary_restriction)
            return var_integrated, out
        elif type(var_optional) != type(None):
            out = self.get_standard_variables(df_in, var_optional, override_vector_for_single_mv_q, return_type, var_bounds, force_boundary_restriction)
            return var_optional, out
        else:
            return None


    ##  function to build a dictionary of categories applicable to a give variable; split by unidim/outer
    def get_partial_category_dictionaries(self,
        subsector: str,
        category_outer_tuple: tuple,
        key_type: str = "key_varreqs_partial",
        delim: str = "|",
        variable_in = None,
        restrict_to_category_values = None,
        var_type = None
    ) -> tuple:

        key_attribute = self.get_subsector_attribute(subsector, key_type)
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
        """
            use get_variable_attribute to retrieve a variable attribute--any cleaned field available in the variable requirements table--associated with a variable.
        """
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


    ##  function to retrieve an (ordered) list of categories for a variable
    def get_variable_categories(self, variable: str):
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


    ##  function for mapping variable to default characteristic (e.g., gas, units, etc.)
    def get_variable_characteristic(self, variable: str, characteristic: str) -> str:
        """
            use get_variable_characteristic to retrieve a characterisetic--e.g., characteristic = "$UNIT-MASS$" or characteristic = "$EMISSION-GAS$"--associated with a variable.
        """
        var_schema = self.get_variable_attribute(variable, "variable_schema")
        dict_out = clean_schema(var_schema, return_default_dict_q = True)
        return dict_out.get(characteristic)


    ##  function to retrieve a variable that is associated with a category in a file (see Transportation Demand for an example)
    def get_variable_from_category(self, subsector: str, category: str, var_type: str = "all") -> str:

        # run some checks
        self.check_subsector(subsector)
        if var_type not in ["all", "partial"]:
            raise ValueError(f"Invalid var_type '{var_type}' in get_variable_from_category: valid types are 'all', 'partial'")

        # get the value from the dictionary
        pycat_trde = self.get_subsector_attribute("Transportation Demand", "pycategory_primary")
        key_vrp_trde = self.get_subsector_attribute("Transportation Demand", f"key_varreqs_{var_type}")

        # get from the dictionary
        key_dict = f"{pycat_trde}_to_{key_vrp_trde}"
        dict_map = self.dict_attributes[pycat_trde].field_maps.get(key_dict)

        if dict_map is not None:
            return dict_map.get(category)
        else:
            return None


    ##  easy function for getting a variable subsector
    def get_variable_subsector(self, modvar):
        dict_check = self.dict_model_variable_to_subsector
        if modvar not in dict_check.keys():
            raise KeyError(f"Invalid model variable '{modvar}': model variable not found.")
        else:
            return dict_check[modvar]


    ##  function to convert units
    def get_variable_unit_conversion_factor(self, var_to_convert: str, var_to_match: str, units: str) -> float:

        """
        get_variable_conversion_factor gives a conversion factor to scale 'var_to_convert' in the same units 'units' as 'var_to_match'

        Function Arguments
        ------------------

        var_to_convert: string of a model variable to scale units

        var_to_match: string of a model variable to match units

        units: valid values are 'energy', 'length', 'mass', 'volume'

        """
        # return None if no variable passed
        if var_to_convert is None:
            return None

        # check specification
        dict_valid_units = {
            "energy": self.varchar_str_unit_energy,
            "length": self.varchar_str_unit_length,
            "mass": self.varchar_str_unit_mass,
            "volume": self.varchar_str_unit_volume
        }

        # check values
        if units not in dict_valid_units.keys():
            str_valid_units = sf.format_print_list(sorted(list(dict_valid_units.keys())))
            raise ValueError(f"Invalid units '{units}' specified in get_variable_conversion_factor: valid values are {str_valid_units}")

        # get arguments
        args = (
            self.get_variable_characteristic(var_to_convert, dict_valid_units[units]),
            self.get_variable_characteristic(var_to_match, dict_valid_units[units])
        )
        # switch based on input units
        if units == "energy":
            val_return = self.get_energy_equivalent(*args)
        elif units == "length":
            val_return = self.get_length_equivalent(*args)
        elif units == "mass":
            val_return = self.get_mass_equivalent(*args)
        elif units == "volume":
            val_return = self.get_volume_equivalent(*args)

        return val_return


    ##  function to extract a variable (with applicable categories from an input data frame)
    def get_standard_variables(self,
        df_in: pd.DataFrame,
        modvar: str,
        override_vector_for_single_mv_q: bool = False,
        return_type: str = "data_frame",
        var_bounds = None,
        force_boundary_restriction: bool = True,
        expand_to_all_cats: bool = False,
        all_cats_missing_val: float = 0.0,
        return_num_type: type = np.float64
    ):

        """
            use get_standard_variables() to retrieve an array or data frame of input variables. If return_type == "array_units_corrected", then the ModelAttributes will re-scale emissions factors to reflect the desired output emissions mass (as defined in the configuration).

            - df_in: data frame containing input variables

            - modvar: variable name to retrieve

            - override_vector_for_single_mv_q: default is False. Set to True to return a vector if the dimension of the variable is 1; otherwise, an array will be returned (if not a dataframe).

            - return_type: valid values are "data_frame", "array_base" (np.ndarray not corrected for configuration emissions), or "array_units_corrected" (emissions corrected for configuration)

            - var_bounds: Default is None (no bounds). Otherwise, gives boundaries to enforce variables that are retrieved. For example, some variables may be restricted to the range (0, 1). Use a list-like structure to pass a minimum and maximum bound (np.inf can be used to as no bound).

            - force_boundary_restriction: default is True. Set to True to enforce the boundaries on the variable. If False, a variable that is out of bounds will raise an error.

            - expand_to_all_cats: default is False. If True, return the variable in the shape of all categories.

            - all_cats_missing_val: default is 0. If expand_to_all_cats == True, categories not associated with modvar with be filled with this value.
        """

        if (modvar is None) or (df_in is None):
            return None

        if modvar not in self.dict_model_variables_to_variables.keys():
            raise ValueError(f"Invalid variable specified in get_standard_variables: variable '{modvar}' not found.")
        else:
            flds = self.dict_model_variables_to_variables[modvar]
            flds = flds[0] if ((len(flds) == 1) and not override_vector_for_single_mv_q) else flds

        # check some types
        self.check_restricted_value_argument(
            return_type,
            ["data_frame", "array_base", "array_units_corrected", "array_units_corrected_gas"],
            "return_type", "get_standard_variables"
        )
        self.check_restricted_value_argument(
            return_num_type,
            [float, int, np.float64, np.int64],
            "return_num_type", "get_standard_variables"
        )

        # initialize output, apply various common transformations based on type
        out = np.array(df_in[flds]).astype(return_num_type)
        if return_type == "array_units_corrected":
            out *= self.get_scalar(modvar, "total")
        elif return_type == "array_units_corrected_gas":
            out *= self.get_scalar(modvar, "gas")

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
                str_warn = f"Invalid maximum value of '{modvar}': specifed value of {m_1} exceeds bound {b_1}."
                if force_boundary_restriction:
                    warnings.warn(str_warn + "\nForcing maximum value in trajectory.")
                else:
                    raise ValueError(str_warn)
            # check min
            if m_0 < b_0:
                str_warn = f"Invalid minimum value of '{modvar}': specifed value of {m_0} below bound {b_0}."
                if force_boundary_restriction:
                    warnings.warn(str_warn + "\nForcing minimum value in trajectory.")
                else:
                    raise ValueError(str_warn)

            if force_boundary_restriction:
                out = sf.vec_bounds(out, var_bounds)


        # merge output to all categories?
        if expand_to_all_cats:
            out = np.array([out]).transpose() if (len(out.shape) == 1) else out
            out = self.merge_array_var_partial_cat_to_array_all_cats(np.array(out), modvar, missing_vals = all_cats_missing_val)
            if return_type == "data_frame":
                sec = self.get_variable_subsector(modvar)
                flds = self.get_attribute_table(sec).key_values

        # convert back to data frame if necessary
        if (return_type == "data_frame"):
            flds = [flds] if (not type(flds) in [list, np.ndarray]) else flds
            out = pd.DataFrame(out, columns = flds)

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

    # return a list of variables by sector
    def get_variables_by_sector(self, sector: str, return_var_type: str = "input") -> list:
        df_attr_sec = self.dict_attributes["abbreviation_subsector"].table
        #list_out = list(np.concatenate([self.build_varlist(x) for x in list(df_attr_sec[df_attr_sec["sector"] == sector]["subsector"])]))
        sectors = list(df_attr_sec[df_attr_sec["sector"] == sector]["subsector"])
        vars_input, vars_output = self.get_input_output_fields(sectors)

        if return_var_type == "input":
            return vars_input
        elif return_var_type == "output":
            return vars_output
        elif return_var_type == "both":
            vars_both = vars_input + vars_output
            vars_both.sort()
            return vars_both
        else:
            raise ValueError(f"Invalid return_var_type specification '{return_var_type}' in get_variables_by_sector: valid values are 'input', 'output', and 'both'.")


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

    ##  retrives mutually-exclusive fields used to sum to generate internal variables
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

    ##  useful function for calculating simple driver*emission factor emissions
    def get_simple_input_to_output_emission_arrays(
        self,
        df_ef: pd.DataFrame,
        df_driver: pd.DataFrame,
        dict_vars: dict,
        variable_driver: str
    ) -> list:
        """
            NOTE: this only works w/in subsector. Returns a list of dataframes.

            df_ef: data frame that contains the emission factor variables

            df_driver: data frame containing the variables driving emissions


        """
        # check if
        df_out = []
        subsector_driver = self.dict_model_variable_to_subsector[variable_driver]
        for var in dict_vars.keys():
            subsector_var = self.dict_model_variable_to_subsector[var]
            if subsector_driver != subsector_driver:
                warnings.warn(f"In get_simple_input_to_output_emission_arrays, driver variable '{variable_driver}' and emission variable '{var}' are in different sectors. This instance will be skipped.")
            else:
                # get emissions factor fields and apply scalar using get_standard_variables
                arr_ef = np.array(self.get_standard_variables(df_ef, var, True, "array_units_corrected"))
                # get the emissions driver array (driver must h)
                arr_driver = np.array(df_driver[self.build_target_varlist_from_source_varcats(var, variable_driver)])

                df_out.append(self.array_to_df(arr_driver*arr_ef, dict_vars[var]))
        return df_out

    ##  function to add a variable based on components
    def manage_internal_variable_to_df(self,
        df_in:pd.DataFrame,
        subsector: str,
        internal_variable: str,
        component_variable: str,
        attribute_sum_specification_field: str,
        action: str = "add"
    ):
        # get the field to add
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


    ##  manage internal variables in data frames
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
