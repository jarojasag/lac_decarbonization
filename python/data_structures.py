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
        self.table = table



class ModelAttributes:

    def __init__(self, dir_attributes):
        self.attribute_file_extension = ".csv"
        self.substr_categories = "attribute_"
        self.substr_varreqs = "table_varreqs_by_"
        self.substr_varreqs_allcats = f"{self.substr_varreqs}category_"
        self.substr_varreqs_partialcats = f"{self.substr_varreqs}partial_category_"

        self.all_attributes, self.dict_attributes, self.dict_varreqs = self.get_attribute_tables(dir_attributes)

        # run checks and raise errors if there are problems
        self.check_land_use_tables()



    # retrieve and format attribute tables for use
    def get_attribute_tables(self, dir_att):

        # get available types
        all_types = [x for x in os.listdir(dir_att) if (self.attribute_file_extension in x) and ((self.substr_categories in x) or (self.substr_varreqs_allcats in x) or (self.substr_varreqs_partialcats in x))]
        ##  batch load attributes/variable requirements and turn them into AttributeTable objects
        dict_attributes = {}
        dict_varreqs = {}
        for att in all_types:
            fp = os.path.join(dir_att, att)
            if self.substr_categories in att:
                nm = sf.clean_field_names([x for x in pd.read_csv(fp, nrows = 0).columns if "$" in x])[0]
                att_table = AttributeTable(fp, nm, [])
                dict_attributes.update({nm: att_table})
            elif (self.substr_varreqs_allcats in att) or (self.substr_varreqs_partialcats in att):
                nm = att.replace(self.substr_varreqs, "").replace(self.attribute_file_extension, "")
                att_table = AttributeTable(fp, "variable", [])
                dict_varreqs.update({nm: att_table})
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

        return (all_types, dict_attributes, dict_varreqs)

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
        matchstr_forest = "forests_"

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



    ########################################
    #    VARIABLE REQUIREMENT FUNCTIONS    #
    ########################################

    # use this to avoid changing function in multiple places
    def format_category_for_outer(self, category_to_replace, appendstr_i = "-I", appendstr_j = "-J"):
        cat_i = category_to_replace.replace("$", f"{appendstr_i}$")[len(appendstr_i):]
        cat_j = category_to_replace.replace("$", f"{appendstr_j}$")[len(appendstr_j):]
        return (cat_i, cat_j)

    ##  function for bulding a basic variable list from the (no complexitiies)
    def build_vars_basic(self, dict_vr_varschema, dict_vars_to_cats, category_to_replace):
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
    def build_var_outer(self, dict_vr_varschema, dict_vars_to_cats, category_to_replace, appendstr_i = "-I", appendstr_j = "-J"):
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

    # separate a variable requirement dictionary into those associated with simple vars and those with outer
    def separate_varreq_dict_for_outer(self, subsector, key_type, category_outer_tuple, target_field = "variable_schema", field_to_split_on = "variable_schema", variable = None):
        # field_to_split_on gives the field from the attribute table to use to split between outer and unidim
        # target field is the field to return in the dictionary
        # key_type = key_varreqs_all, key_varreqs_partial
        key_attribute = self.get_subsector_attribute(subsector, key_type)
        if key_attribute != None:
            dict_vr_vvs = self.dict_varreqs[self.get_subsector_attribute(subsector, key_type)].field_maps[f"variable_to_{field_to_split_on}"].copy()
            dict_vr_vtf = self.dict_varreqs[self.get_subsector_attribute(subsector, key_type)].field_maps[f"variable_to_{target_field}"].copy()
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

    # clean a partial category dictionary to return either none (no categorization) or a list of applicable cateogries
    def clean_partial_category_dictionary(self, dict_in, all_category_values, delim = "|"):
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

    # function to build a dictionary of categories applicable to a give variable; split by unidim/outer
    def get_partial_category_dictionaries(self, subsector, category_outer_tuple, key_type = "key_varreqs_partial", delim = "|", variable_in = None):
        # key_type = key_varreqs_all, key_varreqs_partial
        key_attribute = self.get_subsector_attribute(subsector, key_type)
        valid_cats = self.dict_attributes[self.get_subsector_attribute(subsector, "pycategory_primary")].key_values
        if key_attribute != None:
            dict_vr_vvs_cats_ud, dict_vr_vvs_cats_outer = self.separate_varreq_dict_for_outer(subsector, key_type, category_outer_tuple, target_field = "categories", variable = variable_in)
            dict_vr_vvs_cats_ud = self.clean_partial_category_dictionary(dict_vr_vvs_cats_ud, valid_cats, delim)
            dict_vr_vvs_cats_outer = self.clean_partial_category_dictionary(dict_vr_vvs_cats_outer, valid_cats, delim)

            return dict_vr_vvs_cats_ud, dict_vr_vvs_cats_outer
        else:
            return {}, {}


    ##  function for land use, which introduces an outer product between categories
    def build_varlist(self, subsector, variable_subsec = None):
        # get some subsector info
        category = self.dict_attributes["abbreviation_subsector"].field_maps["abbreviation_subsector_to_primary_category"][self.get_subsector_attribute(subsector, "abv_subsector")].replace("`", "")
        category_ij_tuple = self.format_category_for_outer(category, "-I", "-J")
        attribute_table = self.dict_attributes[self.get_subsector_attribute(subsector, "pycategory_primary")]

        # get dictionary of variable to variable schema and id variables that are in the outer (Cartesian) product (i x j)
        dict_vr_vvs, dict_vr_vvs_outer = self.separate_varreq_dict_for_outer(subsector, "key_varreqs_all", category_ij_tuple, variable = variable_subsec)
        # build variables that apply to all categories
        vars_out = self.build_vars_basic(dict_vr_vvs, dict(zip(list(dict_vr_vvs.keys()), [attribute_table.key_values for x in dict_vr_vvs.keys()])), category)
        if len(dict_vr_vvs_outer) > 0:
            vars_out += self.build_var_outer(dict_vr_vvs_outer, dict(zip(list(dict_vr_vvs_outer.keys()), [attribute_table.key_values for x in dict_vr_vvs_outer.keys()])), category)

        # build those that apply to partial categories
        dict_vrp_vvs, dict_vrp_vvs_outer = self.separate_varreq_dict_for_outer(subsector, "key_varreqs_partial", category_ij_tuple, variable = variable_subsec)
        dict_vrp_vvs_cats, dict_vrp_vvs_cats_outer = self.get_partial_category_dictionaries(subsector, category_ij_tuple, variable_in = variable_subsec)

        if len(dict_vrp_vvs) > 0:
            vars_out += self.build_vars_basic(dict_vrp_vvs, dict_vrp_vvs_cats, category)
        if len(dict_vrp_vvs_outer) > 0:
            vl = self.build_var_outer(dict_vrp_vvs_outer, dict_vrp_vvs_cats_outer, category)
            vars_out += self.build_var_outer(dict_vrp_vvs_outer, dict_vrp_vvs_cats_outer, category)

        return vars_out


# function for cleaning a variable schema
def clean_schema(var_schema):

    var_schema = var_schema.split("(")
    var_schema[0] = var_schema[0].replace("`", "").replace(" ", "")

    if len(var_schema) > 1:
        repls =  var_schema[1].replace("`", "").split(",")
        dict_repls = {}
        for dr in repls:
            dr0 = dr.replace(" ", "").replace(")", "").split("=")
            var_schema[0] = var_schema[0].replace(dr0[0], dr0[1])

    return var_schema[0]
