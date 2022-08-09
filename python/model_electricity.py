import support_functions as sf
import data_structures as ds
from model_socioeconomic import Socioeconomic
from model_ippu import IPPU
import pandas as pd
import numpy as np
import time


###########################
###                     ###
###     ENERGY MODEL    ###
###                     ###
###########################

class ElectricEnergy:

    def __init__(self, attributes: ds.ModelAttributes):

        # some subector reference variables
        self.subsec_name_ccsq = "Carbon Capture and Sequestration"
        self.subsec_name_econ = "Economy"
        self.subsec_name_enfu = "Energy Fuels"
        self.subsec_name_enst = "Energy Storage"
        self.subsec_name_entc = "Energy Technology"
        self.subsec_name_fgtv = "Fugitive Emissions"
        self.subsec_name_gnrl = "General"
        self.subsec_name_inen = "Industrial Energy"
        self.subsec_name_ippu = "IPPU"
        self.subsec_name_scoe = "Stationary Combustion and Other Energy"
        self.subsec_name_trns = "Transportation"
        self.subsec_name_trde = "Transportation Demand"

        # add some key fields from nemo mod
        self.field_nemomod_description = "desc"
        self.field_nemomod_emission = "e"
        self.field_nemomod_fuel = "f"
        self.field_nemomod_id = "id"
        self.field_nemomod_lorder = "lorder"
        self.field_nemomod_mode = "m"
        self.field_nemomod_multiplier = "multiplier"
        self.field_nemomod_name = "name"
        self.field_nemomod_order = "order"
        self.field_nemomod_region = "r"
        self.field_nemomod_storage = "s"
        self.field_nemomod_technology = "t"
        self.field_nemomod_tg1 = "tg1"
        self.field_nemomod_tg2 = "tg2"
        self.field_nemomod_time_slice = "l"
        self.field_nemomod_value = "val"
        self.field_nemomod_year = "y"

        # sort hierarchy
        self.fields_nemomod_sort_hierarchy = [
            self.field_nemomod_id,
            self.field_nemomod_region,
            self.field_nemomod_technology,
            self.field_nemomod_storage,
            self.field_nemomod_fuel,
            self.field_nemomod_emission,
            self.field_nemomod_mode,
            self.field_nemomod_time_slice,
            self.field_nemomod_year,
            # value and description should always be at the end
            self.field_nemomod_value,
            self.field_nemomod_description
        ]

        # initialize dynamic variables
        self.model_attributes = attributes
        self.required_dimensions = self.get_required_dimensions()
        self.required_subsectors, self.required_base_subsectors = self.get_required_subsectors()
        self.required_variables, self.output_variables = self.get_elec_input_output_fields()


        ##  SET MODEL FIELDS

        # Energy Fuel model variables
        self.modvar_enfu_ef_combustion_co2 = ":math:\\text{CO}_2 Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_mobile_ch4 = ":math:\\text{CH}_4 Mobile Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_mobile_n2o = ":math:\\text{N}_2\\text{O} Mobile Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_stationary_ch4 = ":math:\\text{CH}_4 Stationary Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_stationary_n2o = ":math:\\text{N}_2\\text{O} Stationary Combustion Emission Factor"
        self.modvar_enfu_efficiency_factor_industrial_energy = "Average Industrial Energy Fuel Efficiency Factor"
        self.modvar_enfu_energy_demand_by_fuel_ccsq = "Energy Demand by Fuel in CCSQ"
        self.modvar_enfu_energy_demand_by_fuel_elec = "Energy Demand by Fuel in Electricity"
        self.modvar_enfu_energy_demand_by_fuel_inen = "Energy Demand by Fuel in Industrial Energy"
        self.modvar_enfu_energy_demand_by_fuel_scoe = "Energy Demand by Fuel in SCOE"
        self.modvar_enfu_energy_demand_by_fuel_total = "Total Energy Demand by Fuel"
        self.modvar_enfu_energy_demand_by_fuel_trns = "Energy Demand by Fuel in Transportation"
        self.modvar_enfu_volumetric_energy_density = "Volumetric Energy Density"
        # key categories
        self.cat_enfu_electricity = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_enfu, {self.model_attributes.field_enfu_electricity_demand_category: 1})[0]

        # Energy (Electricity) Mode Fields
        self.cat_enmo_gnrt = self.model_attributes.get_categories_from_attribute_characteristic(
            self.model_attributes.dim_mode,
            {"generation_category": 1}
        )[0]
        self.cat_enmo_stor = self.model_attributes.get_categories_from_attribute_characteristic(
            self.model_attributes.dim_mode,
            {"storage_category": 1}
        )[0]

        # Energy (Electricity) Technology Fields


        # Energy (Electricity) Storage Fields



    ##  FUNCTIONS FOR MODEL ATTRIBUTE DIMENSIONS

    def check_df_fields(self,
        df_elec_trajectories: pd.DataFrame,
        subsector: str = "All",
        var_type: str = "input",
        msg_prepend: str = None
    ):
        if subsector == "All":
            check_fields = self.required_variables
            msg_prepend = "Electricity"
        else:
            self.model_attributes.check_subsector(subsector)
            if var_type == "input":
                check_fields, ignore_fields = self.model_attributes.get_input_output_fields([self.subsec_name_econ, self.subsec_name_gnrl, subsector])
            elif var_type == "output":
                ignore_fields, check_fields = self.model_attributes.get_input_output_fields([subsector])
            else:
                raise ValueError(f"Invalid var_type '{var_type}' in check_df_fields: valid types are 'input', 'output'")
            msg_prepend = msg_prepend if (msg_prepend is not None) else subsector
        sf.check_fields(df_elec_trajectories, check_fields, f"{msg_prepend} projection cannot proceed: fields ")

    def get_required_subsectors(self):
        ## TEMPORARY
        subsectors = [self.subsec_name_enfu, self.subsec_name_enst, self.subsec_name_entc]#self.model_attributes.get_setor_subsectors("Energy")
        subsectors_base = subsectors.copy()
        subsectors += [self.subsec_name_econ, self.subsec_name_gnrl]
        return subsectors, subsectors_base

    def get_required_dimensions(self):
        ## TEMPORARY - derive from attributes later
        required_doa = [self.model_attributes.dim_time_period]
        return required_doa

    def get_elec_input_output_fields(self):
        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(self.required_subsectors)

        return required_vars + self.get_required_dimensions(), output_vars



    ###############################################################
    #    GENERALIZED DATA FRAME CHECK FUNCTIONS FOR FORMATTING    #
    ###############################################################

    ##  add a field by expanding to all possible values if it is missing
    def add_index_field_from_key_values(self,
        df_input: pd.DataFrame,
        index_values: list,
        field_index: str,
        outer_prod: bool = True
    ) -> pd.DataFrame:
        """
            Add a field (if necessary) to input dataframe if it is missing based on input index_values.

            - df_input: input data frame to modify
            - index_values: values to expand the data frame along
            - field_index: new field to add
            - outer_prod: assume data frame is repeated to all regions. If not, assume that the index values are applied as a column only (must be one element or of the same length as df_input)
        """

        field_dummy = "merge_key"

        # add the region field if it is not present
        if field_index not in df_input.columns:
            if outer_prod:
                # initialize the index values
                df_merge = pd.DataFrame({field_index: index_values})
                df_merge[field_dummy] = 0
                df_input[field_dummy] = 0
                # order columns and do outer product
                order_cols = list(df_input.columns)
                df_input = pd.merge(df_input, df_merge, on = field_dummy, how = "outer")
                df_input = df_input[[field_index] + [x for x in order_cols if (x != field_dummy)]]
            else:
                # check shape
                if (len(df_input) == len(index_values)) or (not (isinstance(index_values, list) or isinstance(index_values, np.ndarray))):
                    df_input[field_index] = index_values
                else:
                    raise ValueError(f"Error in add_index_field_from_key_values: invalid input shape in index_values. Set outer_prod = True to use outer product.")

        return df_input


    ##  add a fuel field if it is missing
    def add_index_field_fuel(self,
        df_input: pd.DataFrame,
        field_fuel: str = None,
        outer_prod: bool = True,
        restriction_fuels: list = None
    ):
        """
            Add a fuel field (if necessary) to input dataframe if it is missing. Defaults to all defined fuels, and assumes that the input data frame is repeated across all fuels.

            - df_input: input data frame to add field to
            - field_fuel: the name of the field. Default is set to NemoMod naming convention.
            - outer_prod: product against all fuels
            - restriction_fuels: subset of fuels to restrict addition to
        """

        field_fuel = self.field_nemomod_fuel if (field_fuel is None) else field_fuel

        # get regions
        fuels = self.model_attributes.get_attribute_table(self.subsec_name_enfu).key_values
        fuels = [x for x in fuels if x in restriction_fuels] if (restriction_fuels is not None) else fuels
        # add to output using outer product
        df_input = self.add_index_field_from_key_values(df_input, fuels, field_fuel, outer_prod = outer_prod)

        return df_input


    ##  add an id field if it is missing
    def add_index_field_id(self,
        df_input: pd.DataFrame,
        field_id: str = None
    ):
        """
            Add a the id field (if necessary) to input dataframe if it is missing.
        """
        field_id = self.field_nemomod_id if (field_id is None) else field_id

        # add the id field if it is not present
        if field_id not in df_input.columns:
            df_input[field_id] = range(1, len(df_input) + 1)

        # order columns and return
        order_cols = [field_id] + [x for x in list(df_input.columns) if (x != field_id)]
        df_input = df_input[order_cols]

        return df_input


    ##  add a region field if it is missing
    def add_index_field_region(self,
        df_input: pd.DataFrame,
        field_region: str = None,
        outer_prod: bool = True,
        restriction_regions: list = None
    ):
        """
            Add a region field (if necessary) to input dataframe if it is missing. Defaults to configuration regions, and assumes that the input data frame is repeated across all regions.

            - df_input: input data frame to add field to
            - field_region: the name of the field. Default is set to NemoMod naming convention.
            - outer_prod: product against all regions
            - restriction_regions: subset of regions to restrict addition to
        """

        field_region = self.field_nemomod_region if (field_region is None) else field_region

        # get regions
        regions = self.model_attributes.dict_attributes[self.model_attributes.dim_region].key_values
        regions = [x for x in regions if x in self.model_attributes.configuration.get("region")]
        regions = [x for x in regions if x in restriction_regions] if (restriction_regions is not None) else regions
        # add to output using outer product
        df_input = self.add_index_field_from_key_values(df_input, regions, field_region, outer_prod = outer_prod)

        return df_input


    ##  add a fuel field if it is missing
    def add_index_field_technology(self,
        df_input: pd.DataFrame,
        field_technology: str = None,
        outer_prod: bool = True,
        restriction_technologies: list = None
    ) -> pd.DataFrame:
        """
            Add a technology field (if necessary) to input dataframe if it is missing. Defaults to all defined technology, and assumes that the input data frame is repeated across all technologies.

            - df_input: input data frame to add field to
            - field_technology: the name of the field. Default is set to NemoMod naming convention.
            - outer_prod: product against all technologies
            - restriction_technologies: subset of technologies to restrict addition to
        """

        field_technology = self.field_nemomod_technology if (field_technology is None) else field_technology

        # get regions
        techs = self.model_attributes.get_attribute_table(self.subsec_name_entc).key_values
        techs = [x for x in techs if x in restriction_technologies] if (restriction_technologies is not None) else techs
        # add to output using outer product
        df_input = self.add_index_field_from_key_values(df_input, techs, field_technology, outer_prod = outer_prod)

        return df_input


    ##  add a year field if it is missing (assumes repitition)
    def add_index_field_year(self,
        df_input: pd.DataFrame,
        field_year: str = None,
        outer_prod: bool = True,
        restriction_years: list = None
    ) -> pd.DataFrame:
        """
            Add a year field (if necessary) to input dataframe if it is missing. Defaults to all defined years (if defined in time periods), and assumes that the input data frame is repeated across all years.

            - df_input: input data frame to add field to
            - field_year: the name of the field. Default is set to NemoMod naming convention.
            - outer_prod: product against all years
            - restriction_years: subset of years to restrict addition to
        """

        field_year = self.field_nemomod_year if (field_year is None) else field_year

        # get regions
        years = self.model_attributes.get_time_period_years()
        years = [x for x in years if x in restriction_years] if (restriction_years is not None) else years
        # add to output using outer product
        df_input = self.add_index_field_from_key_values(df_input, years, field_year, outer_prod = outer_prod)

        return df_input


    ##  batch function to add different fields
    def add_multifields_from_key_values(self,
        df_input_base: pd.DataFrame,
        fields_to_add: list
    ) -> pd.DataFrame:
        """
            Add a multiple fields, assuming repitition of the data frame across dimensions. Based on NemoMod defaults.

            - df_input_base: input data frame to add field to
            - fields_to_add: fields to add. Must be entered as NemoMod defaults.
        """

        df_input = df_input_base.copy()
        # if id is in the table and we are adding other fields, rename it
        field_id_rnm = f"subtable_{self.field_nemomod_id}"
        if len([x for x in fields_to_add if (x != self.field_nemomod_id)]) > 0:
            df_input.rename(columns = {self.field_nemomod_id: field_id_rnm}, inplace = True) if (self.field_nemomod_id in df_input.columns) else None
        # ordered additions
        df_input = self.add_index_field_technology(df_input) if (self.field_nemomod_technology in fields_to_add) else df_input
        df_input = self.add_index_field_fuel(df_input) if (self.field_nemomod_fuel in fields_to_add) else df_input
        df_input = self.add_index_field_region(df_input) if (self.field_nemomod_region in fields_to_add) else df_input
        df_input = self.add_index_field_year(df_input) if (self.field_nemomod_year in fields_to_add) else df_input

        # set sorting hierarchy, then drop original id field
        fields_sort_hierarchy = [x for x in self.fields_nemomod_sort_hierarchy if (x in fields_to_add) and (x != self.field_nemomod_id)]
        fields_sort_hierarchy = fields_sort_hierarchy + [field_id_rnm] if (field_id_rnm in df_input.columns) else fields_sort_hierarchy
        df_input = df_input.sort_values(by = fields_sort_hierarchy).reset_index(drop = True)
        df_input.drop([field_id_rnm], axis = 1, inplace = True) if (field_id_rnm in df_input.columns) else None
        # add the final id field if necessary
        df_input = self.add_index_field_id(df_input) if (self.field_nemomod_id in fields_to_add) else df_input
        df_input = df_input[[x for x in self.fields_nemomod_sort_hierarchy if x in df_input.columns]]

        return df_input



    #######################################################################################
    #    ATTRIBUTE TABLE TRANSFORMATION FUNCTIONS TO FORMAT NEMOMOD DIMENSIONS FOR SQL    #
    #######################################################################################

    ##  format EMISSION for NemoMod
    def format_nemomod_attribute_table_emission(self,
        attribute_emission: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the EMISSION dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_emission: Emission Gas AttributeTable
            - dict_rename: dictionary to rename to "val" and "desc" fields for NemoMod
        """

        # set some defaults
        attribute_emission = self.model_attributes.dict_attributes["emission_gas"] if (attribute_emission is None) else attribute_emission
        dict_rename = {"emission_gas": self.field_nemomod_value, "name": self.field_nemomod_description} if (dict_rename is None) else dict_rename

        # set values out
        df_out = attribute_emission.table.copy()
        df_out.rename(columns = dict_rename, inplace = True)
        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]
        df_out = df_out[fields_ord].sort_values(by = fields_ord).reset_index(drop = True)

        return df_out


    ##  format FUEL for NemoMod
    def format_nemomod_attribute_table_fuel(self,
        attribute_fuel: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the FUEL dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_fuel: Fuel AttributeTable
            - dict_rename: dictionary to rename to "val" and "desc" fields for NemoMod
        """

        # set some defaults
        attribute_fuel = self.model_attributes.get_attribute_table(self.subsec_name_enfu) if (attribute_fuel is None) else attribute_fuel
        pycat_fuel = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")
        dict_rename = {pycat_fuel: self.field_nemomod_value, "description": self.field_nemomod_description} if (dict_rename is None) else dict_rename

        # set values out
        df_out = attribute_fuel.table.copy()
        df_out.rename(columns = dict_rename, inplace = True)
        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]
        df_out = df_out[fields_ord].sort_values(by = fields_ord).reset_index(drop = True)

        return df_out


    ##  format MODE_OF_OPERATION for NemoMod
    def format_nemomod_attribute_table_mode_of_operation(self,
        attribute_mode: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the MODE_OF_OPERATION dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_mode: Mode of Operation AttributeTable
            - dict_rename: dictionary to rename to "val" and "desc" fields for NemoMod
        """

        # get the region attribute - reduce only to applicable regions
        attribute_mode = self.model_attributes.dict_attributes[self.model_attributes.dim_mode] if (attribute_mode is None) else attribute_mode
        dict_rename = {self.model_attributes.dim_mode: self.field_nemomod_value, "description": self.field_nemomod_description} if (dict_rename is None) else dict_rename

        # set values out
        df_out = attribute_mode.table.copy().rename(columns = dict_rename)
        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]
        df_out = df_out[fields_ord].sort_values(by = fields_ord).reset_index(drop = True)

        return df_out



    ##  format NODE for NemoMod
    def format_nemomod_attribute_table_node(self,
        attribute_node: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the NODE dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_node: Node AttributeTable
            - dict_rename: dictionary to rename to "val" and "desc" fields for NemoMod

            CURRENTLY UNUSED
        """

        return None


    ##  format REGION for NemoMod
    def format_nemomod_attribute_table_region(self,
        attribute_region: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the REGION dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_region: CAT-REGION AttributeTable
            - dict_rename: dictionary to rename to "val" and "desc" fields for NemoMod
        """

        # get the region attribute - reduce only to applicable regions
        attribute_region = self.model_attributes.dict_attributes[self.model_attributes.dim_region] if (attribute_region is None) else attribute_region
        dict_rename = {self.model_attributes.dim_region: self.field_nemomod_value, "category_name": self.field_nemomod_description} if (dict_rename is None) else dict_rename

        # set values out
        df_out = attribute_region.table.copy().rename(columns = dict_rename)
        df_out = df_out[df_out[self.field_nemomod_value].isin(self.model_attributes.configuration.get("region"))]
        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]
        df_out = df_out[fields_ord].sort_values(by = fields_ord).reset_index(drop = True)

        return df_out


    ##  format STORAGE for NemoMod
    def format_nemomod_attribute_table_storage(self,
        attribute_storage: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the STORAGE dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_storage: CAT-STORAGE AttributeTable
            - dict_rename: dictionary to rename to "val" and "desc" fields for NemoMod
        """

        # set some defaults
        attribute_storage = self.model_attributes.get_attribute_table(self.subsec_name_enst) if (attribute_storage is None) else attribute_storage
        pycat_strg = self.model_attributes.get_subsector_attribute(self.subsec_name_enst, "pycategory_primary")
        dict_rename = {pycat_strg: self.field_nemomod_value, "description": self.field_nemomod_description} if (dict_rename is None) else dict_rename

        # set values out
        df_out = attribute_storage.table.copy()
        df_out.rename(columns = dict_rename, inplace = True)
        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)] + [f"netzero{x}" for x in ["year", "tg1", "tg2"]]
        df_out = df_out[fields_ord].sort_values(by = fields_ord).reset_index(drop = True)

        return df_out


    ##  format TECHNOLOGY for NemoMod
    def format_nemomod_attribute_table_technology(self,
        attribute_technology: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the TECHNOLOGY dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_technology: CAT-TECHNOLOGY AttributeTable
            - dict_rename: dictionary to rename to "val" and "desc" fields for NemoMod
        """

        # set some defaults
        attribute_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc) if (attribute_technology is None) else attribute_technology
        pycat_tech = self.model_attributes.get_subsector_attribute(self.subsec_name_entc, "pycategory_primary")
        dict_rename = {pycat_tech: self.field_nemomod_value, "description": self.field_nemomod_description} if (dict_rename is None) else dict_rename

        # set values out
        df_out = attribute_technology.table.copy()
        df_out.rename(columns = dict_rename, inplace = True)
        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]
        df_out = df_out[fields_ord].sort_values(by = fields_ord).reset_index(drop = True)

        return df_out



    ###############################################################
    #    DATA TABLE TRANSFORMATION FUNCTIONS TO FORMAT FOR SQL    #
    ###############################################################

    ##  format AnnualEmissionLimit for NemoMod
    def format_nemomod_table_annual_emission_limit(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the AnnualEmissionLimit input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """
        return None


    ##  format CapacityFactor for NemoMod
    def format_nemomod_table_capacity_factor(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the CapacityFactor input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format CapacityToActivityUnit for NemoMod
    def format_nemomod_table_capacity_to_activity_unit(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the CapacityToActivityUnit input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format CapitalCost for NemoMod
    def format_nemomod_table_capital_cost(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the CapitalCost input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format DefaultParameters for NemoMod
    def format_nemomod_table_default_parameters(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the DefaultParameters input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format DiscountRate for NemoMod
    def format_nemomod_table_discount_rate(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the DiscountRate input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format EmissionsActivityRatio for NemoMod
    def format_nemomod_table_emissions_activity_ratio(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the EmissionsActivityRatio input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format FixedCost for NemoMod
    def format_nemomod_table_fixed_cost(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the FixedCost input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format InterestRateStorage for NemoMod
    def format_nemomod_table_interest_rate_storage(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the InterestRateStorage input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format InterestRateTechnology for NemoMod
    def format_nemomod_table_interest_rate_technology(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the InterestRateTechnology input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format InputActivityRatio for NemoMod
    def format_nemomod_table_input_activity_ratio(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the InputActivityRatio input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format MinStorageCharge for NemoMod
    def format_nemomod_table_min_storage_charge(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the MinStorageCharge input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format MinimumUtilization for NemoMod
    def format_nemomod_table_minimum_utilization(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the MinimumUtilization input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format MinimumUtilization for NemoMod
    def format_nemomod_table_minimum_utilization(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the MinimumUtilization input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format ModelPeriodEmissionLimit for NemoMod
    def format_nemomod_table_model_period_emission_limit(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the ModelPeriodEmissionLimit input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format ModelPeriodExogenousEmission for NemoMod
    def format_nemomod_table_model_period_exogenous_emission(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the ModelPeriodExogenousEmission input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format OperationalLife for NemoMod
    def format_nemomod_table_operational_life(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the OperationalLife input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format OperationalLifeStorage for NemoMod
    def format_nemomod_table_operational_life_storage(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the OperationalLifeStorage input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format REMinProductionTarget for NemoMod
    def format_nemomod_re_min_production_target(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the REMinProductionTarget (renewable energy minimum production target) input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format RETagTechnology for NemoMod
    def format_nemomod_table_re_tag_technology(self,
        attribute_technology: ds.AttributeTable
    ) -> pd.DataFrame:
        """
            Format the RETagTechnology (renewable energy technology tag) input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_technology: AttributeTable for technology, used to identify tag
        """

        return None


    ##  format ReserveMargin for NemoMod
    def format_nemomod_table_reserve_margin(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the ReserveMargin input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format ReserveMarginTagFuel for NemoMod
    def format_nemomod_table_reserve_margin_tag_fuel(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the ReserveMarginTagFuel input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format ReserveMarginTagTechnology for NemoMod
    def format_nemomod_table_reserve_margin_tag_technology(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the ReserveMarginTagTechnology input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format ResidualCapacity for NemoMod
    def format_nemomod_table_residual_capacity(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the ResidualCapacity input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format ResidualStorageCapacity for NemoMod
    def format_nemomod_table_residual_storage_capacity(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the ResidualStorageCapacity input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format SpecifiedAnnualDemand for NemoMod
    def format_nemomod_table_specified_annual_demand(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the SpecifiedAnnualDemand input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """


        return None


    ##  format SpecifiedDemandProfile for NemoMod
    def format_nemomod_table_specified_demand_profile(self,
        df_reference_demand_profile: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the SpecifiedDemandProfile input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_reference_demand_profile: data frame of reference demand profile for the region
        """

        # check for required fields
        fields_required = ["l", "val"]
        fields_required_can_add = ["id", "f", "r", "y"]
        df_out = df_reference_demand_profile.copy()
        sf.check_fields(df_out, fields_required, msg_prepend = f"Error in format_nemomod_table_specified_demand_profile: required fields ")

        # filter if region is already included
        if (self.field_nemomod_region in df_reference_demand_profile.columns):
            df_out = df_out[
                df_out[self.field_nemomod_region].isin(self.model_attributes.configuration.get("region"))
            ]
            n = len(df_out[self.field_nemomod_region].unique())

        # format by repition if fields are missing
        df_out = self.add_multifields_from_key_values(df_out, fields_required_can_add)

        return df_out


    ##  format StorageMaxChargeRate for NemoMod
    def format_nemomod_table_storage_max_charge_rate(self,
        attribute_storage: ds.AttributeTable
    ) -> pd.DataFrame:
        """
            Format the StorageMaxChargeRate input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_storage: AttributeTable for storage, used to identify the maximum charge rate
        """

        return None


    ##  format StorageMaxDischargeRate for NemoMod
    def format_nemomod_table_storage_max_discharge_rate(self,
        attribute_storage: ds.AttributeTable
    ) -> pd.DataFrame:
        """
            Format the StorageMaxDischargeRate input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_storage: AttributeTable for storage, used to identify the maximum discharge rate
        """

        return None


    ##  format StorageStartLevel for NemoMod
    def format_nemomod_table_storage_start_level(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the StorageStartLevel input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format TechnologyFromStorage and TechnologyToStorage for NemoMod
    def format_nemomod_table_technology_from_and_to_storage(self,
        attribute_storage: ds.AttributeTable = None,
        attribute_technology: ds.AttributeTable = None
    ) -> pd.DataFrame:
        """
            Format the TechnologyFromStorage and TechnologyToStorage input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_storage: AttributeTable for storage, used to identify storage characteristics
            - attribute_technology: AttributeTable for technology, used to identify whether or not a technology can charge a storage
        """


        # set some defaults
        attribute_storage = self.model_attributes.get_attribute_table(self.subsec_name_enst) if (attribute_storage is None) else attribute_storage
        attribute_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc) if (attribute_technology is None) else attribute_technology
        pycat_strg = self.model_attributes.get_subsector_attribute(self.subsec_name_enst, "pycategory_primary")
        pycat_tech = self.model_attributes.get_subsector_attribute(self.subsec_name_entc, "pycategory_primary")


        """
            ##  get dictionaries mapping technology categories to storage categories
                NOTE attribute table crosswalk checks are performed in the ModelAttributes class,
                so if it runs, we know that the categories provided are valid
        """

        # from storage
        dict_from_storage = self.model_attributes.get_ordered_category_attribute(
            self.subsec_name_entc,
            "technology_from_storage",
            return_type = dict,
            skip_none_q = True
        )
        dict_from_storage = self.model_attributes.clean_partial_category_dictionary(
            dict_from_storage,
            attribute_storage.key_values
        )

        # to storage
        dict_to_storage = self.model_attributes.get_ordered_category_attribute(
            self.subsec_name_entc,
            "technology_to_storage",
            return_type = dict,
            skip_none_q = True
        )
        dict_to_storage = self.model_attributes.clean_partial_category_dictionary(
            dict_to_storage,
            attribute_storage.key_values
        )

        # cat storage dictionary
        dict_storage_techs_to_storage = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            pycat_strg,
            return_type = dict,
            skip_none_q = True,
            clean_attribute_schema_q = True
        )


        ##  build tech from storage
        df_tech_from_storage = []
        for k in dict_from_storage.keys():
            df_tech_from_storage += list(zip([k for x in dict_from_storage[k]], dict_from_storage[k]))
        df_tech_from_storage = pd.DataFrame(df_tech_from_storage, columns = [self.field_nemomod_technology, self.field_nemomod_storage])
        # specify that storage can generate from storage
        df_tech_from_storage[self.field_nemomod_mode] = self.cat_enmo_gnrt
        df_tech_from_storage[self.field_nemomod_value] = 1.0
        df_tech_from_storage = self.add_multifields_from_key_values(df_tech_from_storage, [self.field_nemomod_id, self.field_nemomod_region])

        ##  build tech to storage
        df_tech_to_storage = []
        for k in dict_to_storage.keys():
            df_tech_to_storage += list(zip([k for x in dict_to_storage[k]], dict_to_storage[k]))
        df_tech_to_storage = pd.DataFrame(df_tech_to_storage, columns = [self.field_nemomod_technology, self.field_nemomod_storage])
        # specify that tech can generate from storage, while storage only stores
        def storage_mode(tech: str) -> str:
            return self.cat_enmo_stor if (tech in dict_storage_techs_to_storage.keys()) else self.cat_enmo_gnrt
        df_tech_to_storage[self.field_nemomod_mode] = df_tech_to_storage[self.field_nemomod_technology].apply(storage_mode)
        df_tech_to_storage[self.field_nemomod_value] = 1.0
        df_tech_to_storage = self.add_multifields_from_key_values(df_tech_to_storage, [self.field_nemomod_id, self.field_nemomod_region])


        return df_tech_from_storage, df_tech_to_storage


    ##  format LTsGroup, TSGROUP1, TSGROUP2, and YearSplit for NemoMod
    def format_nemomod_table_tsgroup_tables(self,
        attribute_time_slice: ds.AttributeTable = None
    ) -> pd.DataFrame:
        """
            Format the LTsGroup, TSGROUP1, TSGROUP2, and YearSplit input tables for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_time_slice: AttributeTable for time slice, used to identify the maximum discharge rate
        """
        # retrieve the attribute and check fields
        fields_req = [
            "time_slice",
            self.field_nemomod_tg1,
            self.field_nemomod_tg2,
            self.field_nemomod_lorder,
            "weight"
        ]
        attribute_time_slice = self.model_attributes.dict_attributes["time_slice"] if (attribute_time_slice is None) else attribute_time_slice
        sf.check_fields(attribute_time_slice.table, fields_req, msg_prepend = "Missing fields in table 'LTsGroup': ")


        ##  FORMAT THE LTsGroup TABLE

        df_ltsgroup = attribute_time_slice.table.copy().drop_duplicates().reset_index(drop = True)
        df_ltsgroup[self.field_nemomod_id] = range(1, len(df_ltsgroup) + 1)
        df_ltsgroup.rename(columns = {"time_slice": self.field_nemomod_time_slice}, inplace = True)
        fields_ext = [
            self.field_nemomod_id,
            self.field_nemomod_time_slice,
            self.field_nemomod_tg1,
            self.field_nemomod_tg2,
            self.field_nemomod_lorder
        ]
        df_ltsgroup = df_ltsgroup[fields_ext]


        ##  FORMAT THE YearSplit TABLE

        df_year_split = attribute_time_slice.table.copy().drop_duplicates().reset_index(drop = True)
        df_year_split = df_year_split[["time_slice", "weight"]].rename(
            columns = {
                "time_slice": self.field_nemomod_time_slice,
                "weight": self.field_nemomod_value
            }
        )
        df_year_split = pd.merge(df_year_split, df_ltsgroup[[self.field_nemomod_time_slice, self.field_nemomod_id]], how = "left")
        df_year_split = self.add_multifields_from_key_values(df_year_split, [self.field_nemomod_id, self.field_nemomod_year])


        ##  FORMAT TSGROUP1 and TSGROUP2

        # get data used to identify order
        df_tgs = pd.merge(
            df_ltsgroup[[self.field_nemomod_id, self.field_nemomod_time_slice, self.field_nemomod_tg1, self.field_nemomod_tg2]],
            df_year_split[df_year_split[self.field_nemomod_year] == min(df_year_split[self.field_nemomod_year])][[self.field_nemomod_time_slice, self.field_nemomod_value]],
            how = "left"
        ).sort_values(by = [self.field_nemomod_id])
        # some dictionaries
        dict_field_to_attribute = {self.field_nemomod_tg1: "ts_group_1", self.field_nemomod_tg2: "ts_group_2"}
        dict_tg = {}

        # loop over fields
        for fld in [self.field_nemomod_tg1, self.field_nemomod_tg2]:

            # prepare from LTsGroup table
            dict_agg = {
                self.field_nemomod_id: "first",
                fld: "first"
            }

            df_tgs_out = df_tgs[[
                self.field_nemomod_id, fld
            ]].groupby(
                [fld]
            ).agg(dict_agg).sort_values(
                by = [self.field_nemomod_id]
            ).reset_index(
                drop = True
            )

            # get attribute for time slice group
            attr_cur = self.model_attributes.dict_attributes[dict_field_to_attribute[fld]].table.copy()
            attr_cur.rename(
                columns = {
                    dict_field_to_attribute[fld]: self.field_nemomod_name,
                    "description": self.field_nemomod_description,
                    "multiplier": self.field_nemomod_multiplier
                }, inplace = True
            )

            df_tgs_out[self.field_nemomod_order] = range(1, len(df_tgs_out) + 1)
            df_tgs_out = df_tgs_out.drop([self.field_nemomod_id], axis = 1).rename(
                columns = {
                    fld: self.field_nemomod_name
                }
            )
            df_tgs_out = pd.merge(df_tgs_out, attr_cur).sort_values(by = [self.field_nemomod_order]).reset_index(drop = True)

            # order for output
            df_tgs_out = df_tgs_out[[
                self.field_nemomod_name,
                self.field_nemomod_description,
                self.field_nemomod_order,
                self.field_nemomod_multiplier,
            ]]

            dict_tg.update({fld: df_tgs_out})

        return df_ltsgroup, dict_tg[self.field_nemomod_tg1], dict_tg[self.field_nemomod_tg2], df_year_split


    ##  format TotalAnnualMaxCapacity for NemoMod
    def format_nemomod_table_total_annual_max_capacity(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the TotalAnnualMaxCapacity input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format TotalAnnualMaxCapacityInvestment for NemoMod
    def format_nemomod_table_total_annual_max_capacity_investment(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the TotalAnnualMaxCapacityInvestment input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format TotalAnnualMinCapacity for NemoMod
    def format_nemomod_table_total_annual_min_capacity(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the TotalAnnualMinCapacity input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None


    ##  format TotalAnnualMinCapacityInvestment for NemoMod
    def format_nemomod_table_total_annual_min_capacity_investment(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the TotalAnnualMinCapacityInvestment input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        return None
