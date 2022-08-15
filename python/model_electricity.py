import support_functions as sf
import data_structures as ds
from model_socioeconomic import Socioeconomic
from model_circular_economy import CircularEconomy
import pandas as pd
import numpy as np
import time
from typing import Union


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
        self.field_nemomod_table_name = "tablename"
        self.field_nemomod_technology = "t"
        self.field_nemomod_tg1 = "tg1"
        self.field_nemomod_tg2 = "tg2"
        self.field_nemomod_time_slice = "l"
        self.field_nemomod_value = "val"
        self.field_nemomod_year = "y"
        # dictionary to map fields to type
        self.dict_fields_nemomod_to_type = {
            self.field_nemomod_description: str,
            self.field_nemomod_emission: str,
            self.field_nemomod_fuel: str,
            self.field_nemomod_id: int,
            self.field_nemomod_lorder: int,
            self.field_nemomod_mode: str,
            self.field_nemomod_multiplier: float,
            self.field_nemomod_name: str,
            self.field_nemomod_order: int,
            self.field_nemomod_region: str,
            self.field_nemomod_storage: str,
            self.field_nemomod_table_name: str,
            self.field_nemomod_technology: str,
            self.field_nemomod_tg1: str,
            self.field_nemomod_tg2: str,
            self.field_nemomod_time_slice: str,
            #self.field_nemomod_value: float,
            self.field_nemomod_year: str
        }

        # sort hierarchy
        self.fields_nemomod_sort_hierarchy = [
            self.field_nemomod_id,
            self.field_nemomod_region,
            self.field_nemomod_table_name,
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
        self.modvar_enfu_energy_density_gravimetric = "Gravimetric Energy Density"
        self.modvar_enfu_energy_density_volumetric = "Volumetric Energy Density"
        self.modvar_enfu_exports_fuel = "Fuel Exports"
        self.modvar_enfu_frac_fuel_demand_imported = "Fraction of Fuel Demand Imported"
        self.modvar_enfu_transmission_loss_electricity = "Electrical Transmission Loss"
        # key categories
        self.cat_enfu_elec = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_enfu, {self.model_attributes.field_enfu_electricity_demand_category: 1})[0]
        self.cat_enfu_ngas = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_enfu, {self.model_attributes.field_enfu_natural_gas_fuel_category: 1})[0]
        self.cat_enfu_wste = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_enfu, {self.model_attributes.field_enfu_waste_fuel_category: 1})[0]
        # associated indices
        self.ind_enfu_elec = self.model_attributes.get_attribute_table(self.subsec_name_enfu).get_key_value_index(self.cat_enfu_elec)
        self.ind_enfu_ngas = self.model_attributes.get_attribute_table(self.subsec_name_enfu).get_key_value_index(self.cat_enfu_ngas)
        self.ind_enfu_wste = self.model_attributes.get_attribute_table(self.subsec_name_enfu).get_key_value_index(self.cat_enfu_wste)

        # Energy (Electricity) Mode Fields
        self.cat_enmo_gnrt = self.model_attributes.get_categories_from_attribute_characteristic(
            self.model_attributes.dim_mode,
            {"generation_category": 1}
        )[0]
        self.cat_enmo_stor = self.model_attributes.get_categories_from_attribute_characteristic(
            self.model_attributes.dim_mode,
            {"storage_category": 1}
        )[0]

        # Energy (Electricity) Technology Variables
        self.modvar_entc_nemomod_capital_cost = "NemoMod CapitalCost"
        self.modvar_entc_ef_scalar_ch4 = ":math:\\text{CH}_4 NemoMod EmissionsActivityRatio Scalar"
        self.modvar_entc_ef_scalar_co2 = ":math:\\text{CO}_2 NemoMod EmissionsActivityRatio Scalar"
        self.modvar_entc_ef_scalar_n2o = ":math:\\text{N}_2\\text{O} NemoMod EmissionsActivityRatio Scalar"
        self.modvar_entc_efficiency_factor_technology = "Technology Efficiency of Fuel Use"
        self.modvar_entc_nemomod_fixed_cost = "NemoMod FixedCost"
        self.modvar_entc_nemomod_reserve_margin = "NemoMod ReserveMargin"
        self.modvar_entc_nemomod_reserve_margin_tag_technology = "NemoMod ReserveMarginTagTechnology"
        self.modvar_entc_nemomod_residual_capacity = "NemoMod ResidualCapacity"
        self.modvar_entc_nemomod_total_annual_max_capacity = "NemoMod TotalAnnualMaxCapacity"
        self.modvar_entc_nemomod_total_annual_max_capacity_investment = "NemoMod TotalAnnualMaxCapacityInvestment"
        self.modvar_entc_nemomod_total_annual_min_capacity = "NemoMod TotalAnnualMinCapacity"
        self.modvar_entc_nemomod_total_annual_min_capacity_investment = "NemoMod TotalAnnualMinCapacityInvestment"
        self.modvar_entc_nemomod_variable_cost = "NemoMod VariableCost"
        # other key variables
        self.drop_flag_tech_capacities = -999

        # Energy (Electricity) Storage Variables
        self.modvar_enst_nemomod_capital_cost_storage = "NemoMod CapitalCostStorage"
        self.modvar_enst_nemomod_residual_capacity = "NemoMod ResidualStorageCapacity"
        self.modvar_enst_nemomod_storage_start_level = "NemoMod StorageStartLevel"
        self.modvar_enst_nemomod_total_annual_max_capacity_storage = "NemoMod TotalAnnualMaxCapacityStorage"
        self.modvar_enst_nemomod_total_annual_max_capacity_investment_storage = "NemoMod TotalAnnualMaxCapacityInvestmentStorage"
        self.modvar_enst_nemomod_total_annual_min_capacity_storage = "NemoMod TotalAnnualMinCapacityStorage"
        self.modvar_enst_nemomod_total_annual_min_capacity_investment_storage = "NemoMod TotalAnnualMinCapacityInvestmentStorage"
        # Additional Miscellaneous variables and functions (to clean up repetition)
        self.units_energy_nemomod = self.model_attributes.configuration.get("energy_units_nemomod")

        # instantiate a CircularEconomy class for access to variables
        self.model_circecon = CircularEconomy(self.model_attributes)
        self.model_socioeconomic = Socioeconomic(self.model_attributes)


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
            if len(df_input) == 0:
                df_input[field_index] = None
            elif outer_prod:
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
        restriction_regions: list = None,
        restrict_to_config_region: bool = True
    ):
        """
            Add a region field (if necessary) to input dataframe if it is missing. Defaults to configuration regions, and assumes that the input data frame is repeated across all regions.

            - df_input: input data frame to add field to
            - field_region: the name of the field. Default is set to NemoMod naming convention.
            - outer_prod: product against all regions
            - restriction_regions: subset of regions to restrict addition to
            - restrict_to_config_region: only allow regions specified in the configuration? Generally set to true, but can be set to false for data construction
        """

        field_region = self.field_nemomod_region if (field_region is None) else field_region

        # get regions
        regions = self.model_attributes.dict_attributes[self.model_attributes.dim_region].key_values
        regions = [x for x in regions if x in self.model_attributes.configuration.get("region")] if restrict_to_config_region else regions
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


    ##  function used in verify_min_max_constraint_inputs
    def conflict_resolution_func_vmmci(self,
        mm_tuple: tuple,
        approach: str = "swap"
    ) -> float:

        max_true = max(mm_tuple)
        min_true = min(mm_tuple)

        if approach == "swap":
            out = (min_true, max_true)
        elif approach == "max_sup":
            out = (max_true, max_true)
        elif approach == "min_sup":
            out = (min_true, min_true)
        elif approach == "mean":
            mean_true = (max_true + min_true)/2
            out = (mean_true, mean_true)

        return out


    ##  return a scalar - use to reduce clutter in converting energy units to NemoMod energy units
    def get_nemomod_energy_scalar(self, modvar: str) -> float:
        var_energy = self.model_attributes.get_variable_characteristic(modvar, self.model_attributes.varchar_str_unit_energy)
        scalar = self.model_attributes.get_energy_equivalent(var_energy, self.units_energy_nemomod)
        return (scalar if (scalar is not None) else 1)


    ##  get waste outputs needed for inputs as energy
    def get_waste_energy_components(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_technology: ds.AttributeTable = None,
        return_emission_factors: bool = True
    ) -> tuple:

        """
            Retrieve total energy to be obtained from waste incineration (minimum capacity) and implied annual emission factors derived from incineration inputs in the waste sector (NemoMod emission/energy)
            - df_elec_trajectories: data frame of input variables, which must include waste sector outputs used to calcualte emission factors
            - attribute_technology: technology attribute table, used to map fuel to tech. If None, use ModelAttributes default.
            - return_emission_factors: bool--calculate emission factors?
        """
        # some attribute initializations
        attribute_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc) if (attribute_technology is None) else attribute_technology
        pycat_enfu = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")
        dict_techs_to_fuel = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            pycat_enfu,
            return_type = dict,
            skip_none_q = True,
            clean_attribute_schema_q = True
        )
        dict_fuel_to_techs = sf.reverse_dict(dict_techs_to_fuel)
        # output variable initialization
        dict_efs = {}
        vec_enfu_total_energy_waste = None

        # retrieve waste totals incinerated (in waste model only)
        tuple_waso_incineration = self.model_attributes.get_optional_or_integrated_standard_variable(
            df_elec_trajectories,
            self.model_circecon.modvar_waso_waste_total_incineration,
            None,
            override_vector_for_single_mv_q = True,
            return_type = "array_base"
        )

        #
        if tuple_waso_incineration is not None:
            # get mass of waste incinerated,
            modvar_waso_mass_incinerated, array_waso_mass_incinerated = tuple_waso_incineration
            vec_waso_mass_incinerated = np.sum(array_waso_mass_incinerated, axis = 1)
            # convert to energy units using gravimetric density (aka specific energy)
            vec_enfu_energy_density_gravimetric = self.model_attributes.get_standard_variables(
                df_elec_trajectories,
                self.modvar_enfu_energy_density_gravimetric,
                override_vector_for_single_mv_q = True,
                return_type = "array_base",
                expand_to_all_cats = True
            )
            vec_enfu_energy_density_gravimetric = vec_enfu_energy_density_gravimetric[:, self.ind_enfu_wste]
            # convert units -- first, in terms of mass incinerated, then in terms of energy density
            vec_enfu_energy_density_gravimetric /= self.model_attributes.get_variable_unit_conversion_factor(
                self.modvar_enfu_energy_density_gravimetric,
                modvar_waso_mass_incinerated,
                "mass"
            )
            vec_enfu_energy_density_gravimetric *= self.get_nemomod_energy_scalar(self.modvar_enfu_energy_density_gravimetric)
            vec_enfu_total_energy_waste = vec_enfu_energy_density_gravimetric*vec_waso_mass_incinerated

        # get emission factors?
        if (vec_enfu_total_energy_waste is not None) and return_emission_factors:
            # loop over waste emissions, divide by total energy

            list_modvars_enfu_to_tech = [
                (self.model_circecon.modvar_waso_emissions_ch4_incineration, self.modvar_entc_ef_scalar_ch4),
                (self.model_circecon.modvar_waso_emissions_co2_incineration, self.modvar_entc_ef_scalar_co2),
                (self.model_circecon.modvar_waso_emissions_n2o_incineration, self.modvar_entc_ef_scalar_n2o)
            ]

            for modvars in list_modvars_enfu_to_tech:

                modvar, modvar_scalar = modvars

                vec_waso_emissions_incineration = self.model_attributes.get_optional_or_integrated_standard_variable(
                    df_elec_trajectories,
                    modvar,
                    None,
                    override_vector_for_single_mv_q = False,
                    return_type = "array_base"
                )
                # if the data is available, calculate the factor and add it to the dictionary (long by time periods in df_elec_trajectories)
                if (vec_waso_emissions_incineration is not None):
                    # get incineration emissions total and scale units
                    emission = self.model_attributes.get_variable_characteristic(modvar, self.model_attributes.varchar_str_emission_gas)
                    modvar_waso_emissions_emissions, vec_waso_emissions_incineration = vec_waso_emissions_incineration
                    vec_waso_emissions_incineration *= self.model_attributes.get_scalar(modvar, "mass")
                    # get control scalar on reductions
                    vec_entc_ear_scalar = self.model_attributes.get_standard_variables(
                        df_elec_trajectories,
                        modvar_scalar,
                        override_vector_for_single_mv_q = True,
                        return_type = "array_base",
                        expand_to_all_cats = True,
                        var_bounds = (0, 1)
                    )
                    cat_tech = dict_fuel_to_techs[self.cat_enfu_wste]
                    ind_tech = attribute_technology.get_key_value_index(cat_tech)
                    vec_entc_ear_scalar = vec_entc_ear_scalar[:, ind_tech]

                    dict_efs.update({emission: vec_entc_ear_scalar*vec_waso_emissions_incineration/vec_enfu_total_energy_waste})

        return vec_enfu_total_energy_waste, dict_efs


    ##  format model variable from SISEPUEDE as an input table (long) for NemoMod
    def format_model_variable_as_nemomod_table(self,
        df_elec_trajectories: pd.DataFrame,
        modvar: str,
        table_nemomod: str,
        fields_index_nemomod: list,
        field_melt_nemomod: str,
        dict_fields_to_pass: dict = {},
        scalar_to_nemomod_units: float = 1,
        drop_flag: Union[float, int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
            Format a SISEPUEDE variable as a nemo mod input table.
            - df_elec_trajectories: data frame containing input variables to be reformatted
            - modvar: SISEPUEDE model variable to extract and reshape
            - table_nemomod: target NemoMod table
            - fields_index_nemomod: indexing fields to add/preserve in table
            - field_melt_nemomod: name of field to store columns under in long format
            - dict_fields_to_pass: dictionary to pass fields to the output data frame before sorting
                * Dictionary takes the form {field_1: new_col, ...}, where new_col = [x_0, ..., x_{n - 1}] or new_col = obj
            - scalar_to_nemomod_units: scalar applied to the values to convert to proper units
            - drop_flag: values that should be dropped from the table
            **kwargs: passed to ModelAttributes.get_standard_variables()
        """

        # set some defaults
        subsector = self.model_attributes.get_variable_subsector(modvar, throw_error_q = False)
        if subsector is None:
            # logger.log()
            return None
        attr = self.model_attributes.get_attribute_table(subsector)
        pycat = self.model_attributes.get_subsector_attribute(subsector, "pycategory_primary")
        # get the variable
        df_out = self.model_attributes.get_standard_variables(
            df_elec_trajectories,
            modvar,
            override_vector_for_single_mv_q = True,
            return_type = "array_base",
            expand_to_all_cats = False,
            **kwargs
        )
        df_out *= scalar_to_nemomod_units
        cats_ordered_out = self.model_attributes.get_variable_categories(modvar)
        df_out = pd.DataFrame(df_out, columns = cats_ordered_out)
        # add a year (would not be in data frame)
        if (self.field_nemomod_year in fields_index_nemomod) and (self.model_attributes.dim_time_period in df_elec_trajectories.columns):
            df_out = self.model_attributes.add_year_from_time_period(
                df_out,
                self.field_nemomod_year,
                df_elec_trajectories[self.model_attributes.dim_time_period]
            )
        # add any additional fields
        if len(dict_fields_to_pass) > 0:
            for fld in dict_fields_to_pass.keys():
                if (fld not in df_out.columns) :
                    df_out[fld] = dict_fields_to_pass[fld]
                    fields_index_nemomod += [fld] if (fld not in fields_index_nemomod) else []

        # next, melt, drop any values, and add keys/sort/reset index for NemoMod
        df_out = pd.melt(
            df_out,
            [x for x in df_out.columns if x in fields_index_nemomod],
            cats_ordered_out,
            var_name = field_melt_nemomod,
            value_name = self.field_nemomod_value
        )
        df_out = df_out[~df_out[self.field_nemomod_value].isin([drop_flag])] if (drop_flag is not None) else df_out
        df_out = self.add_multifields_from_key_values(df_out, fields_index_nemomod)

        return {table_nemomod: df_out}



    ##  format a description of a dummy tech from the fuel
    def format_dummy_tech_description_from_fuel(self, fuel: str) -> str:
        return f"Dummy technology for fuel {fuel}"

    ##  function to map a fuel to a dummy variable used to extract demands and allow for a solution
    def get_dummy_fuel_tech(self, fuel: str) -> str:
        return f"supply_{fuel}"



    ##  get a dictionary of dummy techs for use in OperationalLifeStorage and OutputActivityRatio
    def get_dummy_techs(self,
        attribute_technology: ds.AttributeTable = None
    ) -> dict:
        """
            Get a dictionary mapping fuels mapped to powerplant generation to associated dummy techs
            - attribute_technology: AttributeTable for technology, used to identify operational lives of generation and storage technologies. If None, use ModelAttributes default.
        """
        # get some defaults
        attribute_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc) if (attribute_technology is None) else attribute_technology
        pycat_enfu = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")

        dict_techs_to_fuel = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            pycat_enfu,
            return_type = dict,
            skip_none_q = True,
            clean_attribute_schema_q = True
        )

        fuels_keep = sorted([x for x in dict_techs_to_fuel.values() if (x != self.cat_enfu_elec)])
        techs_dummy = [self.get_dummy_fuel_tech(x) for x in fuels_keep]

        dict_return = dict(zip(fuels_keep, techs_dummy))

        return dict_return



    # defin a function to compare max/min for related constraints
    def verify_min_max_constraint_inputs(self,
        df_max: pd.DataFrame,
        df_min: pd.DataFrame,
        field_max: str,
        field_min: str,
        conflict_resolution_option: str = "swap",
        comparison: str = "weak",
        drop_invalid_comparisons_on_strong: bool = True,
        field_id: str = None,
        return_passthrough: bool = False
    ) -> Union[None, dict]:
        """
        Verify that a minimum trajectory is less than or equal (weak) or less than (strong) a maximum trajectory. Data frames must have comparable indices.
        - df_max: data frame containing the maximum trajectory
        - df_min: data frame containing the minimum trajectory
        - field_max: field in df_max to use to compare
        - field_min: field in df_min to use to compare
        - conflict_resolution_option: if the minimum trajectory is greater than the maximum trajectory, this parameter is used to define the resolution:
            * "swap": swap instances where the minimum exceeds the maximum
            * "max_sup": set the larger value as the minimum and the maximum
            * "min_sup": set the smaller value as the minimum and the maximum
            * "mean": use the mean of the two as the minimum and the maximum
            * "error": stop and return an error
        - comparison: "weak" allows the minimum <= maximum, while "strong" => minimum < maximum
            * If comparison == "strong", then cases where maximum == minimum cannot be resolved will be dropped if drop_invalid_comparisons_on_strong == True; otherwise, an error will be returned (independent of conflict_resolution_option)
        - drop_invalid_comparisons_on_strong: drop cases where minimum == maximum?
        - field_id: id field contained in both that is used for re-merging
        - return_passthrough: if no changes are required, return original dataframes?
        """

        # check for required field
        field_id = self.field_nemomod_id if (field_id is None) else field_id
        sf.check_fields(df_max, [field_id, field_max])
        sf.check_fields(df_min, [field_id, field_min])

        # merge to facilitate comparison
        fields_shared = list(set(df_max.columns) & set(df_max.columns))
        fields_shared = [x for x in fields_shared if x not in [field_min, field_max]]
        fields_max = fields_shared + [field_max]
        fields_min = fields_shared + [field_min]
        df_compare = pd.merge(df_max[fields_max], df_min[fields_min], on = fields_shared, suffixes = ("_max", "_min"))

        # set fields to use for comparison
        field_maxm = f"{field_max}_max" if (field_max == field_min) else field_max
        field_minm = f"{field_min}_min" if (field_max == field_min) else field_min
        #
        vec_comparison = np.array(df_compare[[field_minm, field_maxm]])
        w_resolve = np.where(vec_comparison[:, 1] < vec_comparison[:, 0]) if (comparison == "weak") else np.where(vec_comparison[:, 1] <= vec_comparison[:, 0])[0]

        if (len(w_resolve) > 0):
            if conflict_resolution_option != "error":
                df_new_vals = df_compare[[field_minm, field_maxm]].apply(
                    self.conflict_resolution_func_vmmci,
                    approach = conflict_resolution_option,
                    axis = 1,
                    raw = True
                )
                # some replacements
                df_max_replace = pd.concat([df_compare[fields_shared], df_new_vals[[field_maxm]]], axis = 1).rename(columns = {field_maxm: field_max})
                df_min_replace = pd.concat([df_compare[fields_shared], df_new_vals[[field_minm]]], axis = 1).rename(columns = {field_minm: field_min})
                df_max_out = sf.replace_numerical_column_from_merge(df_max, df_max_replace, field_max)
                df_min_out = sf.replace_numerical_column_from_merge(df_min, df_min_replace, field_min)
            else:
                raise ValueError(f"Error in verify_min_max_constraint_inputs: minimum trajectory meets or exceeds maximum trajectory in at least one row.")

            return df_max_out, df_min_out
        else:
            return (df_max, df_min) if return_passthrough else None



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
            - attribute_emission: Emission Gas AttributeTable. If None, use ModelAttributes default.
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

        return {self.model_attributes.table_nemomod_emission: df_out}


    ##  format FUEL for NemoMod
    def format_nemomod_attribute_table_fuel(self,
        attribute_fuel: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the FUEL dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_fuel: Fuel AttributeTable. If None, use ModelAttributes default.
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

        return {self.model_attributes.table_nemomod_fuel: df_out}


    ##  format MODE_OF_OPERATION for NemoMod
    def format_nemomod_attribute_table_mode_of_operation(self,
        attribute_mode: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the MODE_OF_OPERATION dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_mode: Mode of Operation AttributeTable. If None, use ModelAttributes default.
            - dict_rename: dictionary to rename to "val" and "desc" fields for NemoMod
        """

        # get the region attribute - reduce only to applicable regions
        attribute_mode = self.model_attributes.dict_attributes[self.model_attributes.dim_mode] if (attribute_mode is None) else attribute_mode
        dict_rename = {self.model_attributes.dim_mode: self.field_nemomod_value, "description": self.field_nemomod_description} if (dict_rename is None) else dict_rename

        # set values out
        df_out = attribute_mode.table.copy().rename(columns = dict_rename)
        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]
        df_out = df_out[fields_ord].sort_values(by = fields_ord).reset_index(drop = True)

        return {self.model_attributes.table_nemomod_mode_of_operation: df_out}



    ##  format NODE for NemoMod
    def format_nemomod_attribute_table_node(self,
        attribute_node: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the NODE dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_node: Node AttributeTable. If None, use ModelAttributes default.
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
            - attribute_region: CAT-REGION AttributeTable. If None, use ModelAttributes default.
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

        return {self.model_attributes.table_nemomod_region: df_out}


    ##  format STORAGE for NemoMod
    def format_nemomod_attribute_table_storage(self,
        attribute_storage: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the STORAGE dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_storage: CAT-STORAGE AttributeTable. If None, use ModelAttributes default.
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

        return {self.model_attributes.table_nemomod_storage: df_out}


    ##  format TECHNOLOGY for NemoMod
    def format_nemomod_attribute_table_technology(self,
        attribute_technology: ds.AttributeTable = None,
        dict_rename: dict = None
    ) -> pd.DataFrame:
        """
            Format the TECHNOLOGY dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_technology: CAT-TECHNOLOGY AttributeTable. If None, use ModelAttributes default.
            - dict_rename: dictionary to rename to "val" and "desc" fields for NemoMod
        """

        # set some defaults
        attribute_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc) if (attribute_technology is None) else attribute_technology
        pychat_entc = self.model_attributes.get_subsector_attribute(self.subsec_name_entc, "pycategory_primary")
        dict_rename = {pychat_entc: self.field_nemomod_value, "description": self.field_nemomod_description} if (dict_rename is None) else dict_rename

        # add dummies
        dict_fuels_to_dummy_techs = self.get_dummy_techs(attribute_technology = attribute_technology)
        df_out_dummies = pd.DataFrame({self.field_nemomod_fuel: list(dict_fuels_to_dummy_techs.keys())})
        df_out_dummies[self.field_nemomod_value] = df_out_dummies[self.field_nemomod_fuel].replace(dict_fuels_to_dummy_techs)
        df_out_dummies[self.field_nemomod_description] = df_out_dummies[self.field_nemomod_fuel].apply(self.format_dummy_tech_description_from_fuel)
        df_out_dummies.drop([self.field_nemomod_fuel], axis = 1, inplace = True)

        # set values out
        df_out = attribute_technology.table.copy()
        df_out.rename(columns = dict_rename, inplace = True)

        fields_ord = [x for x in self.fields_nemomod_sort_hierarchy if (x in df_out.columns)]
        df_out = pd.concat(
            [df_out[fields_ord], df_out_dummies[fields_ord]],
            axis = 0
        ).sort_values(
            by = fields_ord
        ).reset_index(
            drop = True
        )

        return {self.model_attributes.table_nemomod_technology: df_out}


    ##  format TECHNOLOGY for NemoMod
    def format_nemomod_attribute_table_year(self
    ) -> pd.DataFrame:
        """
            Format the YEAR dimension table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            * Based off of years defined in the attribute_time_period.csv attribute table
        """
        years = self.model_attributes.get_time_period_years()
        df_out = pd.DataFrame({
            self.field_nemomod_value: years,
            self.field_nemomod_description: [f"Year {y}" for y in years]
        })
        return {self.model_attributes.table_nemomod_year: df_out}



    ###############################################################
    #    DATA TABLE TRANSFORMATION FUNCTIONS TO FORMAT FOR SQL    #
    ###############################################################

    ##  format AnnualEmissionLimit for NemoMod
    def format_nemomod_table_annual_emission_limit(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_emission: ds.AttributeTable = None,
        attribute_time_period: ds.AttributeTable = None,
        dict_gas_to_emission_fields: dict = None,
        drop_flag: int = -999
    ) -> pd.DataFrame:
        """
            Format the AnnualEmissionLimit input tables for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
            - attribute_emission: AttributeTable table with gasses. If None, use ModelAttribute default.
            - attribute_time_period: AttributeTable table with time periods for year identification. If None, use ModelAttribute default.
            - dict_gas_to_emission_fields: dictionary with gasses (in attribute_gas) as keys that map to fields to use to calculate total exogenous emissions
            - drop_flag: values to drop
        """

        # get some defaults and attribute tables
        dict_gas_to_emission_fields = self.model_attributes.dict_gas_to_total_emission_fields if (dict_gas_to_emission_fields is None) else dict_gas_to_emission_fields
        attribute_emission = self.model_attributes.dict_attributes.get("emission_gas") if (attribute_emission is None) else attribute_emission
        attribute_time_period = self.model_attributes.dict_attributes.get(self.model_attributes.dim_time_period) if (attribute_time_period is None) else attribute_time_period

        modvars_limit = [
            self.model_socioeconomic.modvar_gnrl_emission_limit_ch4,
            self.model_socioeconomic.modvar_gnrl_emission_limit_co2,
            self.model_socioeconomic.modvar_gnrl_emission_limit_n2o
        ]
        df_out = []

        for modvar in enumerate(modvars_limit):
            i, modvar = modvar

            # get emission and global warming potential (divide out for limit)
            emission = self.model_attributes.get_variable_characteristic(modvar, self.model_attributes.varchar_str_emission_gas)
            gwp = self.model_attributes.get_gwp(emission)

            # then, get total exogenous emissions
            fields = list(set(dict_gas_to_emission_fields[emission]) & set(df_elec_trajectories.columns))
            vec_exogenous_emissions = np.sum(np.array(df_elec_trajectories[fields]), axis = 1)
            # retrieve the limit, store the origina (for dropping), and convert units
            vec_emission_limit = self.model_attributes.get_standard_variables(
                df_elec_trajectories,
                modvar,
                return_type = "array_base"
            )
            vec_drop_flag = vec_emission_limit.copy()
            vec_emission_limit *= self.model_attributes.get_scalar(modvar, "mass")

            # force limit to  to prevent infeasibilities
            vec_emission_limit_out = sf.vec_bounds(vec_emission_limit - vec_exogenous_emissions, (0, np.inf))
            df_lim = pd.DataFrame({
                "drop_flag": vec_drop_flag,
                self.field_nemomod_value: vec_emission_limit_out
            })
            df_lim[self.field_nemomod_emission] = emission
            df_lim = self.model_attributes.add_year_from_time_period(
                df_lim,
                self.field_nemomod_year,
                df_elec_trajectories[self.model_attributes.dim_time_period],
                attribute_time_period = attribute_time_period
            )

            if i == 0:
                df_out = [df_lim for x in modvars_limit]
            else:
                df_out[i] = df_lim

        # concatenate and order hierarchically
        df_out = pd.concat(df_out, axis = 0)
        df_out = df_out[~df_out["drop_flag"].isin([drop_flag])].drop(["drop_flag"], axis = 1)
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_emission,
                self.field_nemomod_year,
                self.field_nemomod_value,
                "drop_flag"
            ]
        )
        dict_return = {self.model_attributes.table_nemomod_annual_emission_limit: df_out}

        return dict_return



    ##  format CapacityFactor for NemoMod
    def format_nemomod_table_capacity_factor(self,
        df_reference_capacity_factor: pd.DataFrame,
        attribute_technology: ds.AttributeTable = None,
        attribute_region: ds.AttributeTable = None
    ) -> pd.DataFrame:
        """
            Format the CapacityFactor input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_reference_capacity_factor: data frame of regional capacity factors for technologies that vary (others revert to default)
            - attribute_technology: AttributeTable for technology, used to separate technologies from storage and identify primary fuels. If None, defaults to ModelAttributes attribute table.
            - attribute_region: AttributeTable for regions. If None, defaults to ModelAttributes attribute table.
        """

        # check fields
        fields_req = [self.field_nemomod_region, self.field_nemomod_time_slice]
        sf.check_fields(df_reference_capacity_factor, fields_req)

        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        attribute_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc) if (attribute_technology is None) else attribute_technology
        attribute_region = self.model_attributes.dict_attributes.get(self.model_attributes.dim_region) if (attribute_region is None) else attribute_region
        pycat_entc = self.model_attributes.get_subsector_attribute(self.subsec_name_entc, "pycategory_primary")

        ###############################################
        #    INTEGRATE CLIMATE CHANGE FACTORS HERE    #
        ###############################################

        # regions to keep
        regions_keep = set(attribute_region.key_values) & set(df_reference_capacity_factor[self.field_nemomod_region]) & set(self.model_attributes.configuration.get("region"))
        # reshape to long
        fields_melt = [x for x in df_reference_capacity_factor.columns if (x in attribute_technology.key_values)]
        df_out = pd.melt(
            df_reference_capacity_factor[
                df_reference_capacity_factor[self.field_nemomod_region].isin(regions_keep)
            ],
            [self.field_nemomod_region, self.field_nemomod_time_slice],
            fields_melt,
            self.field_nemomod_technology,
            self.field_nemomod_value
        )
        # add output fields
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_technology,
                self.field_nemomod_time_slice,
                self.field_nemomod_year,
                self.field_nemomod_value
            ]
        )

        # ensure capacity factors are properly specified
        df_out[self.field_nemomod_value] = sf.vec_bounds(np.array(df_out[self.field_nemomod_value]), (0, 1))
        dict_return = {self.model_attributes.table_nemomod_capacity_factor: df_out}

        return dict_return



    ##  format CapacityToActivityUnit for NemoMod
    def format_nemomod_table_capacity_to_activity_unit(self,
        return_type: str = "table"
    ) -> pd.DataFrame:
        """
            Format the CapacityToActivityUnit input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - return_type: "table" or "value". If value, returns only the CapacityToActivityUnit value for all techs (used in DefaultParams)
            * Based on configuration parameters
        """

        # first, get power units, swap to get energy unit equivalent, then get units for the default total energy variable
        units_power = self.model_attributes.dict_attributes["unit_power"].field_maps["power_to_unit_power"].get(
            self.model_attributes.configuration.get("power_units")
        )
        units_energy_power_equivalent = self.model_attributes.get_energy_power_swap(units_power)
        cau = self.model_attributes.get_energy_equivalent(units_energy_power_equivalent, self.units_energy_nemomod)

        if return_type == "table":
            df_out = pd.DataFrame({self.field_nemomod_value: [cau]})
            df_out = self.add_multifields_from_key_values(df_out, [self.field_nemomod_id, self.field_nemomod_region, self.field_nemomod_technology])
        elif return_type == "value":
            df_out = cau

        dict_return = {self.model_attributes.table_nemomod_capacity_to_activity_unit: df_out}

        return dict_return



    ##  format CapitalCost, FixedCost, and VaribleCost for NemoMod
    def format_nemomod_table_costs_technology(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the CapitalCost, FixedCost, and VaribleCost input tables for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        dict_return = {}
        # get some scalars (monetary and power)
        scalar_cost_capital = self.model_attributes.get_scalar(self.modvar_entc_nemomod_capital_cost, "monetary")
        scalar_cost_capital /= self.model_attributes.get_scalar(self.modvar_entc_nemomod_capital_cost, "power")
        scalar_cost_fixed = self.model_attributes.get_scalar(self.modvar_entc_nemomod_fixed_cost, "monetary")
        scalar_cost_fixed /= self.model_attributes.get_scalar(self.modvar_entc_nemomod_fixed_cost, "power")
        scalar_cost_variable = self.model_attributes.get_scalar(self.modvar_entc_nemomod_variable_cost, "monetary")
        scalar_cost_variable /= self.get_nemomod_energy_scalar(self.modvar_entc_nemomod_variable_cost)

        # CapitalCost
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_capital_cost,
                self.model_attributes.table_nemomod_capital_cost,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                scalar_to_nemomod_units = scalar_cost_capital,
                var_bounds = (0, np.inf)
            )
        )
        # FixedCost
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_fixed_cost,
                self.model_attributes.table_nemomod_fixed_cost,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                scalar_to_nemomod_units = scalar_cost_fixed,
                var_bounds = (0, np.inf)
            )
        )
        # VariableCost
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_variable_cost,
                self.model_attributes.table_nemomod_variable_cost,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                dict_fields_to_pass = {self.field_nemomod_mode: self.cat_enmo_gnrt},
                scalar_to_nemomod_units = scalar_cost_variable,
                var_bounds = (0, np.inf)
            )
        )

        return dict_return



    ##  format CapitalCostStorage for NemoMod
    def format_nemomod_table_costs_storage(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the CapitalCostStorage input tables for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        dict_return = {}
        # get some scalars (monetary and power)
        scalar_cost_capital_storage = self.model_attributes.get_scalar(self.modvar_enst_nemomod_capital_cost_storage, "monetary")
        scalar_cost_capital_storage /= self.get_nemomod_energy_scalar(self.modvar_enst_nemomod_capital_cost_storage)

        # CapitalCostStorage
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_capital_cost_storage,
                self.model_attributes.table_nemomod_capital_cost_storage,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                scalar_to_nemomod_units = scalar_cost_capital_storage,
                var_bounds = (0, np.inf)
            )
        )

        return dict_return



    ##  format DefaultParameters for NemoMod
    def format_nemomod_table_default_parameters(self,
        attribute_nemomod_table: ds.AttributeTable = None,
        field_default_values: str = "default_value"
    ) -> pd.DataFrame:
        """
            Format the DefaultParameters input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_nemomod_table: NemoMod tables AttributeTable that includes default values stored in the field 'field_default_values'
        """

        attribute_nemomod_table = self.model_attributes.dict_attributes.get("nemomod_table") if (attribute_nemomod_table is None) else attribute_nemomod_table

        # check fields (key is always contained in an attribute table if it is successfully initialized)
        sf.check_fields(attribute_nemomod_table.table, [field_default_values])

        # get dictionary and update parameters
        dict_repl = attribute_nemomod_table.field_maps[f"{attribute_nemomod_table.key}_to_{field_default_values}"].copy()
        dict_repl.update(self.format_nemomod_table_capacity_to_activity_unit(return_type = "value"))
        dict_repl.update(self.format_nemomod_table_discount_rate(return_type = "value"))

        # build output table
        df_out = attribute_nemomod_table.table[[attribute_nemomod_table.key]].copy().rename(columns = {attribute_nemomod_table.key: self.field_nemomod_table_name})
        df_out[self.field_nemomod_value] = df_out[self.field_nemomod_table_name].replace(dict_repl)
        dict_return = {self.model_attributes.table_nemomod_default_params: df_out}

        return dict_return



    ##  format DiscountRate for NemoMod
    def format_nemomod_table_discount_rate(self,
        return_type: str = "table"
    ) -> pd.DataFrame:
        """
            Format the DiscountRate input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - return_type: "table" or "value". If value, returns only the DiscountRate
            * Based on configuration specification of discount_rate
        """

        discount_rate = self.model_attributes.configuration.get("discount_rate")
        df_out = pd.DataFrame({self.field_nemomod_value: [discount_rate]})

        if return_type == "table":
            df_out = self.add_multifields_from_key_values(df_out, [self.field_nemomod_id, self.field_nemomod_region])
        elif return_type == "value":
            df_out = discount_rate

        return {self.model_attributes.table_nemomod_discount_rate: df_out}



    ##  format EmissionsActivityRatio for NemoMod
    def format_nemomod_table_emissions_activity_ratio(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_time_period: ds.AttributeTable = None
    ) -> pd.DataFrame:
        """
            Format the EmissionsActivityRatio input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_enfu = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")
        pycat_entc = self.model_attributes.get_subsector_attribute(self.subsec_name_entc, "pycategory_primary")
        # attribute tables
        attr_enfu = self.model_attributes.dict_attributes[pycat_enfu]
        attr_entc = self.model_attributes.dict_attributes[pycat_entc]

        # cat to fuel dictionary
        dict_techs_to_fuel = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            pycat_enfu,
            return_type = dict,
            skip_none_q = True,
            clean_attribute_schema_q = True
        )
        dict_fuel_to_techs = sf.reverse_dict(dict_techs_to_fuel)
        # get some ordered indexing to convert
        cats_enfu_fuel_ordered = [x for x in attr_enfu.key_values if (x in dict_fuel_to_techs.keys())]
        inds_enfu_extract = [attr_enfu.get_key_value_index(x) for x in cats_enfu_fuel_ordered]
        cats_entc_ordered_by_fuels = [dict_fuel_to_techs.get(x) for x in cats_enfu_fuel_ordered]
        cats_entc_ordered_by_techs = [x for x in attr_entc.key_values if (x in cats_entc_ordered_by_fuels)]
        # set required variables for emission factors and initialize output dictionary
        list_modvars_enfu_to_tech = [
            (self.modvar_enfu_ef_combustion_stationary_ch4, self.modvar_entc_ef_scalar_ch4),
            (self.modvar_enfu_ef_combustion_co2, self.modvar_entc_ef_scalar_co2),
            (self.modvar_enfu_ef_combustion_stationary_n2o, self.modvar_entc_ef_scalar_n2o)
        ]

        df_out = []
        # loop over fuel emission factors to specify for each technology
        for modvars in enumerate(list_modvars_enfu_to_tech):
            ind, modvars = modvars
            modvar, modvar_scalar = modvars
            # get the fuel factors
            arr_enfu_tmp = self.model_attributes.get_standard_variables(
                df_elec_trajectories, modvar, True, "array_base", expand_to_all_cats = False
            )
            # convert emissions mass (configuration) and energy (self.modvar_enfu_energy_demand_by_fuel_total) to the units for NemoMod
            arr_enfu_tmp *= self.model_attributes.get_scalar(modvar, "mass")
            arr_enfu_tmp /= self.get_nemomod_energy_scalar(modvar)
            arr_enfu_tmp = arr_enfu_tmp[:, inds_enfu_extract]
            # expand to tech
            arr_entc_tmp = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(
                arr_enfu_tmp,
                None,
                missing_vals = 0.0,
                output_cats = cats_entc_ordered_by_fuels,
                output_subsec = self.model_attributes.subsec_name_entc
            )
            # apply scalar
            arr_enfu_scalar = self.model_attributes.get_standard_variables(
                df_elec_trajectories,
                modvar_scalar,
                override_vector_for_single_mv_q = True,
                return_type = "array_base",
                expand_to_all_cats = True,
                var_bounds = (0, 1)
            )
            arr_entc_tmp *= arr_enfu_scalar


            ##  FORMAT AS DATA FRAME

            emission = self.model_attributes.get_variable_characteristic(modvar, self.model_attributes.varchar_str_emission_gas)
            df_entc_tmp = pd.DataFrame(arr_entc_tmp, columns = attr_entc.key_values)
            df_entc_tmp = df_entc_tmp[cats_entc_ordered_by_techs]
            # add some key fields (emission and year)
            df_entc_tmp[self.field_nemomod_emission] = emission
            df_entc_tmp[self.field_nemomod_mode] = self.cat_enmo_gnrt
            df_entc_tmp = self.model_attributes.add_year_from_time_period(
                df_entc_tmp,
                self.field_nemomod_year,
                df_elec_trajectories[self.model_attributes.dim_time_period],
                attribute_time_period = attribute_time_period
            )

            # melt into a long form table
            df_entc_tmp = pd.melt(
                df_entc_tmp,
                [self.field_nemomod_emission, self.field_nemomod_mode, self.field_nemomod_year],
                cats_entc_ordered_by_techs,
                var_name = self.field_nemomod_technology,
                value_name = self.field_nemomod_value
            )

            if len(df_out) == 0:
                df_out = [df_entc_tmp for x in range(len(list_modvars_enfu_to_tech))]
            else:
                df_out[ind] = df_entc_tmp[df_out[0].columns]


        #############################################
        #    GET INTEGRATED WASTE EMISSIONS HERE    #
        #############################################

        # get total waste and emission factors from incineration as derived from waste model - note: ef scalars are applied within get_waste_energy_components
        vec_enfu_total_energy_waste, dict_efs = self.get_waste_energy_components(
            df_elec_trajectories,
            return_emission_factors = True
        )

        # format to a new data frame
        df_enfu_efs_waste = None
        if (vec_enfu_total_energy_waste is not None) and len(dict_efs) > 0:
            # melt a data frame
            df_enfu_efs_waste = pd.DataFrame(dict_efs)
            df_enfu_efs_waste[self.field_nemomod_technology] = dict_fuel_to_techs[self.cat_enfu_wste]
            df_enfu_efs_waste[self.field_nemomod_mode] = self.cat_enmo_gnrt
            df_enfu_efs_waste = self.model_attributes.add_year_from_time_period(
                df_enfu_efs_waste,
                self.field_nemomod_year,
                df_elec_trajectories[self.model_attributes.dim_time_period],
                attribute_time_period = attribute_time_period
            )
            # melt into a long form table
            df_enfu_efs_waste = pd.melt(
                df_enfu_efs_waste,
                [self.field_nemomod_technology, self.field_nemomod_mode, self.field_nemomod_year],
                list(dict_efs.keys()),
                var_name = self.field_nemomod_emission,
                value_name = self.field_nemomod_value
            )

        # concatenate and replace waste if applicable
        df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
        if df_enfu_efs_waste is not None:
            df_out = df_out[~df_out[self.field_nemomod_technology].isin([dict_fuel_to_techs[self.cat_enfu_wste]])]
            df_out = pd.concat([df_out, df_enfu_efs_waste], axis = 0).reset_index(drop = True)

        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_emission,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_technology,
                self.field_nemomod_value,
                self.field_nemomod_year
            ]
        )

        dict_return = {self.model_attributes.table_nemomod_emissions_activity_ratio: df_out}

        return dict_return



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
        df_elec_trajectories: pd.DataFrame,
        attribute_technology: ds.AttributeTable = None,
        max_ratio: float = 1000000.0
    ) -> pd.DataFrame:
        """
            Format the InputActivityRatio input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
            - attribute_technology: AttributeTable for technology, used to separate technologies from storage and identify primary fuels.
            - max_ratio: replacement for any input_activity_ratio values derived from efficiencies of 0
        """

        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        attribute_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc) if (attribute_technology is None) else attribute_technology
        pycat_enfu = self.model_attributes.get_subsector_attribute(self.subsec_name_enfu, "pycategory_primary")
        pycat_enst = self.model_attributes.get_subsector_attribute(self.subsec_name_enst, "pycategory_primary")
        pycat_entc = self.model_attributes.get_subsector_attribute(self.subsec_name_entc, "pycategory_primary")

        # cat to fuel dictionary + reverse
        dict_techs_to_fuel = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            pycat_enfu,
            return_type = dict,
            skip_none_q = True,
            clean_attribute_schema_q = True
        )
        dict_fuel_to_techs = sf.reverse_dict(dict_techs_to_fuel)
        # cat to storage dictionary
        dict_tech_to_storage = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            pycat_enst,
            return_type = dict,
            skip_none_q = True,
            clean_attribute_schema_q = True
        )
        # revise some dictionaries for the output table
        dict_tech_to_mode = {}
        # update generation tech mode
        for k in dict_techs_to_fuel.keys():
            dict_tech_to_mode.update({k: self.cat_enmo_gnrt})

        # update storage mode/fuel (will add keys to dict_techs_to_fuel)
        for k in dict_tech_to_storage.keys():
            dict_techs_to_fuel.update({k: self.cat_enfu_elec})
            dict_tech_to_mode.update({k: self.cat_enmo_stor})

        dict_return = {}

        # Initialize InputActivityRatio
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_efficiency_factor_technology,
                self.model_attributes.table_nemomod_input_activity_ratio,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                var_bounds = (0, np.inf)
            )
        )

        # make some modifications
        df_tmp = dict_return[self.model_attributes.table_nemomod_input_activity_ratio]
        df_tmp[self.field_nemomod_fuel] = df_tmp[self.field_nemomod_technology].replace(dict_techs_to_fuel)
        df_tmp[self.field_nemomod_mode] = df_tmp[self.field_nemomod_technology].replace(dict_tech_to_mode)
        # convert efficiency to input_activity_ratio_ratio
        df_tmp[self.field_nemomod_value] = np.nan_to_num(1/np.array(df_tmp[self.field_nemomod_value]), max_ratio, posinf = max_ratio)
        # re-sort using hierarchy
        df_tmp = self.add_multifields_from_key_values(
            df_tmp,
            [
                self.field_nemomod_id,
                self.field_nemomod_fuel,
                self.field_nemomod_technology,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_year,
                self.field_nemomod_value
            ]
        )
        # ensure changes are made to dict
        dict_return.update({self.model_attributes.table_nemomod_input_activity_ratio: df_tmp})

        return dict_return



    ##  format MinStorageCharge for NemoMod
    def format_nemomod_table_min_storage_charge(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_storage: ds.AttributeTable = None,
        field_attribute_min_charge: str = "minimum_charge_fraction"
    ) -> pd.DataFrame:
        """
            Format the MinStorageCharge input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
            - attribute_storage: AttributeTable used to identify minimum storage charge by storage type. If None, defaults to ModelAttribute cat_storage table
            - field_attribute_min_charge: field in attribute_storage containing the minimum storage charge fraction by storage type
        """

        ##
        # NOTE: ADD A CHECK IN THE StorageStartLevel TABLE TO COMPARE TO MINIMUM STORAGE CHARGE AND SELECT MAX BETWEEN THE TWO

        # set some defaults
        attribute_storage = self.model_attributes.get_attribute_table(self.subsec_name_enst) if (attribute_storage is None) else attribute_storage
        pycat_strg = self.model_attributes.get_subsector_attribute(self.subsec_name_enst, "pycategory_primary")
        # initialize storage info
        dict_strg_to_min_charge = attribute_storage.field_maps.get(f"{attribute_storage.key}_to_{field_attribute_min_charge}")
        all_storage = list(dict_strg_to_min_charge.keys())
        df_out = pd.DataFrame({
            self.field_nemomod_storage: all_storage,
            self.field_nemomod_value: [dict_strg_to_min_charge.get(x) for x in all_storage]
        })

        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_storage,
                self.field_nemomod_year,
                self.field_nemomod_value
            ]
        )
        dict_return = {self.model_attributes.table_nemomod_min_storage_charge: df_out}

        return dict_return



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



    ##  format OperationalLife and OperationalLifeStorage for NemoMod
    def format_nemomod_table_operational_life(self,
        attribute_fuel: ds.AttributeTable = None,
        attribute_storage: ds.AttributeTable = None,
        attribute_technology: ds.AttributeTable = None,
        operational_life_dummies: Union[float, int] = 250
    ) -> pd.DataFrame:
        """
            Format the OperationalLife and OperationalLifeStorage input tables for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_fuel: AttributeTable for fuel, used to set dummy fuel supplies as a technology. If None, use ModelAttributes default.
            - attribute_storage: AttributeTable for storage, used to build OperationalLifeStorage from Technology. If None, use ModelAttributes default.
            - attribute_technology: AttributeTable for technology, used to identify operational lives of generation and storage technologies. If None, use ModelAttributes default.
            - operational_life_dummies: Operational life for dummy technologies that are entered to account for fuel inputs.
            Notes:
            - Validity checks for operational lives are performed on initialization of the ModelAttributes class.
        """




        # set some defaults
        attribute_storage = self.model_attributes.get_attribute_table(self.subsec_name_enst) if (attribute_storage is None) else attribute_storage
        attribute_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc) if (attribute_technology is None) else attribute_technology
        pycat_strg = self.model_attributes.get_subsector_attribute(self.subsec_name_enst, "pycategory_primary")
        pychat_entc = self.model_attributes.get_subsector_attribute(self.subsec_name_entc, "pycategory_primary")

        # cat storage dictionary
        dict_storage_techs_to_storage = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            pycat_strg,
            return_type = dict,
            skip_none_q = True,
            clean_attribute_schema_q = True
        )

        # get the life time
        dict_techs_to_operational_life = self.model_attributes.get_ordered_category_attribute(
            self.model_attributes.subsec_name_entc,
            "operational_life",
            return_type = dict,
            skip_none_q = True
        )

        # get dummy techs
        dict_fuels_to_dummy_techs = self.get_dummy_techs(attribute_technology = attribute_technology)
        all_techs = sorted(list(dict_techs_to_operational_life.keys())) + sorted(list(dict_fuels_to_dummy_techs.values()))
        all_ols = [dict_techs_to_operational_life.get(x, operational_life_dummies) for x in all_techs]

        # initliaze data frame
        df_operational_life = pd.DataFrame({
            self.field_nemomod_technology: all_techs,
            self.field_nemomod_value: all_ols
        })
        # split off and perform some cleaning
        df_operational_life_storage = df_operational_life[df_operational_life[self.field_nemomod_technology].isin(dict_storage_techs_to_storage.keys())].copy()
        df_operational_life_storage[self.field_nemomod_technology] = df_operational_life_storage[self.field_nemomod_technology].replace(dict_storage_techs_to_storage)
        df_operational_life_storage.rename(columns = {self.field_nemomod_technology: self.field_nemomod_storage}, inplace = True)
        df_operational_life = df_operational_life[~df_operational_life[self.field_nemomod_technology].isin(dict_storage_techs_to_storage.keys())].copy()

        # add required fields
        fields_reg = [self.field_nemomod_id, self.field_nemomod_region]
        df_operational_life = self.add_multifields_from_key_values(df_operational_life, fields_reg)
        df_operational_life_storage = self.add_multifields_from_key_values(df_operational_life_storage, fields_reg)

        dict_return = {
            self.model_attributes.table_nemomod_operational_life: df_operational_life,
            self.model_attributes.table_nemomod_operational_life_storage: df_operational_life_storage
        }

        return dict_return



    ##  format OutputActivityRatio for NemoMod
    def format_nemomod_table_output_activity_ratio(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_technology: ds.AttributeTable = None
    ) -> pd.DataFrame:
        """
            Format the OutputActivityRatio input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
            - attribute_technology: AttributeTable for technology, used to separate technologies from storage and identify primary fuels.
        """

        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        attribute_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc) if (attribute_technology is None) else attribute_technology

        # get dummy techs
        dict_fuels_to_dummy_techs = self.get_dummy_techs(attribute_technology = attribute_technology)
        df_out_dummies = pd.DataFrame({self.field_nemomod_fuel: list(dict_fuels_to_dummy_techs.keys())})
        df_out_dummies[self.field_nemomod_technology] = df_out_dummies[self.field_nemomod_fuel].replace(dict_fuels_to_dummy_techs)
        # Initialize OutputActivityRatio and add dummies
        df_out = pd.DataFrame({self.field_nemomod_technology: attribute_technology.key_values})
        df_out[self.field_nemomod_fuel] = self.cat_enfu_elec
        df_out = pd.concat([df_out, df_out_dummies], axis = 0).reset_index(drop = True)
        # finish with other variables
        df_out[self.field_nemomod_value] = 1
        df_out[self.field_nemomod_mode] = self.cat_enmo_gnrt

        # re-sort using hierarchy
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_fuel,
                self.field_nemomod_technology,
                self.field_nemomod_mode,
                self.field_nemomod_region,
                self.field_nemomod_year,
                self.field_nemomod_value
            ]
        )
        # ensure changes are made to dict
        dict_return = {self.model_attributes.table_nemomod_output_activity_ratio: df_out}

        return dict_return



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
        attribute_technology: ds.AttributeTable = None
    ) -> pd.DataFrame:
        """
            Format the RETagTechnology (renewable energy technology tag) input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_technology: AttributeTable for technology, used to identify tag. If None, use ModelAttributes default.
        """

        # set some defaults
        attribute_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc) if (attribute_technology is None) else attribute_technology
        pycat_strg = self.model_attributes.get_subsector_attribute(self.subsec_name_enst, "pycategory_primary")
        pychat_entc = self.model_attributes.get_subsector_attribute(self.subsec_name_entc, "pycategory_primary")

        # get renewable technologies - default is 0, so only need to specify those that are renewable
        df_red = attribute_technology.table
        df_out = df_red[
            df_red[pycat_strg].isin(["none"]) &
            df_red["renewable_energy_technology"].isin([1.0, 1])
        ][[attribute_technology.key, "renewable_energy_technology"]].copy().rename(
            columns = {
                attribute_technology.key: self.field_nemomod_technology,
                "renewable_energy_technology": self.field_nemomod_value
            }
        )

        # add dimensions
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_technology,
                self.field_nemomod_year,
                self.field_nemomod_value
            ]
        )

        dict_return = {self.model_attributes.table_nemomod_re_tag_technology: df_out}

        return dict_return



    ##  format ReserveMargin for NemoMod
    def format_nemomod_table_reserve_margin(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the ReserveMargin input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        dict_return = {}
        # ReserveMargin
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_reserve_margin,
                self.model_attributes.table_nemomod_reserve_margin,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                var_bounds = (0, np.inf)
            )
        )
        dict_return[self.model_attributes.table_nemomod_reserve_margin].drop([self.field_nemomod_technology], axis = 1, inplace = True)

        return dict_return



    ##  format ReserveMarginTagFuel for NemoMod
    def format_nemomod_table_reserve_margin_tag_fuel(self
    ) -> pd.DataFrame:
        """
            Format the ReserveMargin input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
        """
        # build data frame
        df_out = pd.DataFrame({
            self.field_nemomod_fuel: [self.cat_enfu_elec],
            self.field_nemomod_value: [1]
        })

        # add dimensions
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_region,
                self.field_nemomod_fuel,
                self.field_nemomod_year,
                self.field_nemomod_value
            ]
        )

        dict_return = {self.model_attributes.table_nemomod_reserve_margin_tag_fuel: df_out}

        return dict_return



    ##  format ReserveMarginTagTechnology for NemoMod
    def format_nemomod_table_reserve_margin_tag_technology(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the ReserveMarginTagTechnology input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        dict_return = {}
        # ReserveMarginTagTechnology
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_reserve_margin_tag_technology,
                self.model_attributes.table_nemomod_reserve_margin_tag_technology,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                var_bounds = (0, np.inf)
            )
        )

        return dict_return



    ##  format ResidualCapacity for NemoMod
    def format_nemomod_table_residual_capacity(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the ResidualCapacity input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        dict_return = {}
        # get some scalars
        scalar_residual_capacity = self.model_attributes.get_scalar(self.modvar_entc_nemomod_residual_capacity, "power")
        # ResidualCapacity
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_residual_capacity,
                self.model_attributes.table_nemomod_residual_capacity,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                scalar_to_nemomod_units = scalar_residual_capacity,
                var_bounds = (0, np.inf)
            )
        )

        return dict_return



    ##  format ResidualStorageCapacity for NemoMod
    def format_nemomod_table_residual_storage_capacity(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the ResidualStorageCapacity input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        dict_return = {}
        # get some scalars
        scalar_cost_capital_storage = self.get_nemomod_energy_scalar(self.modvar_enst_nemomod_residual_capacity)
        # ResidualCapacity
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_residual_capacity,
                self.model_attributes.table_nemomod_residual_storage_capacity,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                scalar_to_nemomod_units = scalar_cost_capital_storage,
                var_bounds = (0, np.inf)
            )
        )

        return dict_return



    ##  format SpecifiedAnnualDemand for NemoMod
    def format_nemomod_table_specified_annual_demand(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_time_period: ds.AttributeTable = None
    ) -> pd.DataFrame:
        """
            Format the SpecifiedAnnualDemand input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
            - attribute_time_period: AttributeTable mapping ModelAttributes.dim_time_period to year. If None, use ModelAttributes default.
        """

        ##  GET DEMANDS FROM INTEGRATED MODEL

        # calculate total grid demand for electricity
        arr_total_energy_demand = 0.0
        modvars_enfu_extract = [
            self.modvar_enfu_energy_demand_by_fuel_ccsq,
            self.modvar_enfu_energy_demand_by_fuel_inen,
            self.modvar_enfu_energy_demand_by_fuel_scoe,
            self.modvar_enfu_energy_demand_by_fuel_trns
        ]
        # loop over outputs from other energy sectors
        for modvar in modvars_enfu_extract:
            arr_enfu_tmp = self.model_attributes.get_standard_variables(df_elec_trajectories, modvar, True, "array_base", expand_to_all_cats = True)
            arr_enfu_tmp *= self.get_nemomod_energy_scalar(modvar)
        # get transmission loss and calculate final demand
        vec_transmission_loss = self.model_attributes.get_standard_variables(df_elec_trajectories, self.modvar_enfu_transmission_loss_electricity, False, "array_base", expand_to_all_cats = True, var_bounds = (0, 1))
        vec_enfu_demand_elec = np.nan_to_num(arr_enfu_tmp[:, self.ind_enfu_elec]/(1 - vec_transmission_loss[:, self.ind_enfu_elec]), 0.0, posinf = 0.0)


        ##  FORMAT AS DATA FRAME

        # initialize and add year
        df_out = pd.DataFrame({
            self.field_nemomod_value: vec_enfu_demand_elec,
            self.field_nemomod_fuel: [self.cat_enfu_elec for x in vec_enfu_demand_elec]
        })
        df_out = self.model_attributes.add_year_from_time_period(
            df_out,
            self.field_nemomod_year,
            df_elec_trajectories[self.model_attributes.dim_time_period],
            attribute_time_period = attribute_time_period
        )
        # add additional required fields, then sort
        df_out = self.add_multifields_from_key_values(
            df_out,
            [
                self.field_nemomod_id,
                self.field_nemomod_fuel,
                self.field_nemomod_region,
                self.field_nemomod_year,
                self.field_nemomod_value
            ]
        )

        return {self.model_attributes.table_nemomod_specified_annual_demand: df_out}



    ##  format SpecifiedDemandProfile for NemoMod
    def format_nemomod_table_specified_demand_profile(self,
        df_reference_demand_profile: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the SpecifiedDemandProfile input table for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_reference_demand_profile: data frame of reference demand profile for the region
        """

        # check for required fields
        fields_required = [self.field_nemomod_time_slice, self.field_nemomod_value]
        fields_required_can_add = [self.field_nemomod_id, self.field_nemomod_fuel, self.field_nemomod_region, self.field_nemomod_year]
        df_out = df_reference_demand_profile.copy()
        sf.check_fields(df_out, fields_required, msg_prepend = f"Error in format_nemomod_table_specified_demand_profile: required fields ")

        # filter if region is already included
        if (self.field_nemomod_region in df_reference_demand_profile.columns):
            df_out = df_out[
                df_out[self.field_nemomod_region].isin(self.model_attributes.configuration.get("region"))
            ]
            n = len(df_out[self.field_nemomod_region].unique())

        # specify fuels
        if (self.field_nemomod_fuel in df_reference_demand_profile.columns):
            df_out = df_out[df_out[self.field_nemomod_fuel] == self.cat_enfu_elec]
        else:
            df_out[self.field_nemomod_fuel] = self.cat_enfu_elec

        # format by repition if fields are missing
        df_out = self.add_multifields_from_key_values(df_out, fields_required_can_add)
        dict_return = {self.model_attributes.table_nemomod_specified_demand_profile: df_out}

        return dict_return



    ##  format StorageMaxChargeRate, StorageMaxDishargeRate, and StorageStartLevel for NemoMod
    def format_nemomod_table_storage_attributes(self,
        df_elec_trajectories: pd.DataFrame,
        attribute_storage: ds.AttributeTable = None,
        field_attribute_min_charge: str = "minimum_charge_fraction",
        field_tmp: str = "TMPNEW"
    ) -> pd.DataFrame:
        """
            Format the StorageMaxChargeRate, StorageMaxDishargeRate, and StorageStartLevel input tables for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
            - attribute_storage: AttributeTable used to ensure that start level meets or exceeds the minimum allowable storage charge. If None, use ModelAttributes default.
            - field_attribute_min_charge: field in attribute_storage.table used to identify minimum required storage for each type of storage. If None, use ModelAttributes default.
            - field_tmp: temporary field used in data frame
        """

        # set some defaults
        attribute_storage = self.model_attributes.get_attribute_table(self.subsec_name_enst) if (attribute_storage is None) else attribute_storage
        pycat_strg = self.model_attributes.get_subsector_attribute(self.subsec_name_enst, "pycategory_primary")
        dict_strg_to_min_charge = attribute_storage.field_maps.get(f"{attribute_storage.key}_to_{field_attribute_min_charge}")

        dict_return = {}
        # StorageStartLevel
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_storage_start_level,
                self.model_attributes.table_nemomod_storage_level_start,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                var_bounds = (0, 1)
            )
        )
        # some cleaning of the data frame
        df_tmp = dict_return[self.model_attributes.table_nemomod_storage_level_start]
        df_tmp = df_tmp[
            df_tmp[self.field_nemomod_year] == min(df_tmp[self.field_nemomod_year])
        ].drop(
            [self.field_nemomod_year],
            axis = 1
        ).reset_index(
            drop = True
        )
        # add bounds and drop the temporary field
        df_tmp[field_tmp] = df_tmp[self.field_nemomod_storage].replace(dict_strg_to_min_charge)
        bounds = list(zip(list(df_tmp[field_tmp]), list(np.ones(len(df_tmp)))))
        df_tmp[self.field_nemomod_value] = sf.vec_bounds(
            np.array(df_tmp[self.field_nemomod_value]),
            bounds
        )
        df_tmp.drop([field_tmp], axis = 1, inplace = True)
        # update
        dict_return.update({self.model_attributes.table_nemomod_storage_level_start: df_tmp})

        return dict_return



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
            - attribute_storage: AttributeTable for storage, used to identify storage characteristics. If None, use ModelAttributes default.
            - attribute_technology: AttributeTable for technology, used to identify whether or not a technology can charge a storage. If None, use ModelAttributes default.
        """


        # set some defaults
        attribute_storage = self.model_attributes.get_attribute_table(self.subsec_name_enst) if (attribute_storage is None) else attribute_storage
        attribute_technology = self.model_attributes.get_attribute_table(self.subsec_name_entc) if (attribute_technology is None) else attribute_technology
        pycat_strg = self.model_attributes.get_subsector_attribute(self.subsec_name_enst, "pycategory_primary")
        pychat_entc = self.model_attributes.get_subsector_attribute(self.subsec_name_entc, "pycategory_primary")


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

        dict_return = {
            self.model_attributes.table_nemomod_technology_from_storage: df_tech_from_storage,
            self.model_attributes.table_nemomod_technology_to_storage: df_tech_to_storage
        }

        return dict_return



    ##  format LTsGroup, TSGROUP1, TSGROUP2, and YearSplit for NemoMod
    def format_nemomod_table_tsgroup_tables(self,
        attribute_time_slice: ds.AttributeTable = None
    ) -> pd.DataFrame:
        """
            Format the LTsGroup, TIMESLICE, TSGROUP1, TSGROUP2, and YearSplit input tables for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - attribute_time_slice: AttributeTable for time slice, used to identify the maximum discharge rate. If None, use ModelAttributes default.
        """
        # retrieve the attribute and check fields
        fields_req = [
            "time_slice",
            "description",
            self.field_nemomod_tg1,
            self.field_nemomod_tg2,
            self.field_nemomod_lorder,
            "weight"
        ]
        attribute_time_slice = self.model_attributes.dict_attributes["time_slice"] if (attribute_time_slice is None) else attribute_time_slice
        sf.check_fields(attribute_time_slice.table, fields_req, msg_prepend = "Missing fields in table 'LTsGroup': ")


        ##  FORMAT THE TIMESLICE ATTRIBUTE TABLE

        df_time_slice = attribute_time_slice.table.copy().drop_duplicates().reset_index(drop = True)
        df_time_slice = df_time_slice[["time_slice", "description"]].rename(
            columns = {
                "time_slice": self.field_nemomod_value,
                "description": self.field_nemomod_description
            }
        )


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

        dict_return = {
            self.model_attributes.table_nemomod_time_slice_group_assignment: df_ltsgroup,
            self.model_attributes.table_nemomod_time_slice: df_time_slice,
            self.model_attributes.table_nemomod_ts_group_1: dict_tg[self.field_nemomod_tg1],
            self.model_attributes.table_nemomod_ts_group_2: dict_tg[self.field_nemomod_tg2],
            self.model_attributes.table_nemomod_year_split: df_year_split
        }

        return dict_return



    ##  format TotalAnnualMaxCapacity, TotalAnnualMaxCapacityInvestment, TotalAnnualMinCapacity, TotalAnnualMinCapacityInvestment for NemoMod
    def format_nemomod_table_total_capacity_tables(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the TotalAnnualMaxCapacity, TotalAnnualMaxCapacityInvestment, TotalAnnualMinCapacity, and TotalAnnualMinCapacityInvestment input tables for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        dict_return = {}
        # get some scalars
        scalar_total_annual_max_capacity = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_max_capacity, "power")
        scalar_total_annual_max_capacity_investment = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_max_capacity_investment, "power")
        scalar_total_annual_min_capacity = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_min_capacity, "power")
        scalar_total_annual_min_capacity_investment = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_min_capacity_investment, "power")

        # TotalAnnualMaxCapacity
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_total_annual_max_capacity,
                self.model_attributes.table_nemomod_total_annual_max_capacity,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                scalar_to_nemomod_units = scalar_total_annual_max_capacity,
                drop_flag = self.drop_flag_tech_capacities
            )
        )
        # TotalAnnualMaxCapacityInvestment
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_total_annual_max_capacity_investment,
                self.model_attributes.table_nemomod_total_annual_max_capacity_investment,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                scalar_to_nemomod_units = scalar_total_annual_max_capacity_investment,
                drop_flag = self.drop_flag_tech_capacities
            )
        )
        # TotalAnnualMinCapacity
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_total_annual_min_capacity,
                self.model_attributes.table_nemomod_total_annual_min_capacity,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                scalar_to_nemomod_units = scalar_total_annual_min_capacity,
                drop_flag = self.drop_flag_tech_capacities
            )
        )
        # TotalAnnualMinCapacityInvestment
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_entc_nemomod_total_annual_min_capacity_investment,
                self.model_attributes.table_nemomod_total_annual_min_capacity_investment,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_technology,
                scalar_to_nemomod_units = scalar_total_annual_min_capacity_investment,
                drop_flag = self.drop_flag_tech_capacities
            )
        )


        ##  CHECK MAX/MIN RELATIONSHIP--SWAP VALUES IF NEEDED

        # check tables - capacity
        dfs_verify = self.verify_min_max_constraint_inputs(
            dict_return[self.model_attributes.table_nemomod_total_annual_max_capacity],
            dict_return[self.model_attributes.table_nemomod_total_annual_min_capacity],
            self.field_nemomod_value,
            self.field_nemomod_value,
            field_id = self.field_nemomod_id
        )
        if dfs_verify is not None:
            dict_return.update(
                {
                    self.model_attributes.table_nemomod_total_annual_max_capacity: dfs_verify[0],
                    self.model_attributes.table_nemomod_total_annual_min_capacity: dfs_verify[1]
                }
            )

        # check tables - capacity investment
        dfs_verify = self.verify_min_max_constraint_inputs(
            dict_return[self.model_attributes.table_nemomod_total_annual_max_capacity_investment],
            dict_return[self.model_attributes.table_nemomod_total_annual_min_capacity_investment],
            self.field_nemomod_value,
            self.field_nemomod_value,
            field_id = self.field_nemomod_id
        )
        if dfs_verify is not None:
            dict_return.update(
                {
                    self.model_attributes.table_nemomod_total_annual_max_capacity_investment: dfs_verify[0],
                    self.model_attributes.table_nemomod_total_annual_min_capacity_investment: dfs_verify[1]
                }
            )

        return dict_return



    ##  format TotalAnnualMaxCapacityStorage, TotalAnnualMaxCapacityInvestmentStorage, TotalAnnualMinCapacityStorage, TotalAnnualMinCapacityInvestmentStorage for NemoMod
    def format_nemomod_table_total_capacity_storage_tables(self,
        df_elec_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Format the TotalAnnualMaxCapacityStorage, TotalAnnualMaxCapacityInvestmentStorage, TotalAnnualMinCapacityStorage, and TotalAnnualMinCapacityInvestmentStorage input tables for NemoMod based on SISEPUEDE configuration parameters, input variables, integrated model outputs, and reference tables.
            - df_elec_trajectories: data frame of model variable input trajectories
        """

        dict_return = {}
        # get some scalars
        scalar_total_annual_max_capacity_storage = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_max_capacity, "power")
        scalar_total_annual_max_capacity_investment_storage = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_max_capacity_investment, "power")
        scalar_total_annual_min_capacity_storage = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_min_capacity, "power")
        scalar_total_annual_min_capacity_investment_storage = self.model_attributes.get_scalar(self.modvar_entc_nemomod_total_annual_min_capacity_investment, "power")

        # TotalAnnualMaxCapacityStorage
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_total_annual_max_capacity_storage,
                self.model_attributes.table_nemomod_total_annual_max_capacity_storage,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                scalar_to_nemomod_units = scalar_total_annual_max_capacity_storage,
                drop_flag = self.drop_flag_tech_capacities
            )
        )
        # TotalAnnualMaxCapacityInvestmentStorage
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_total_annual_max_capacity_investment_storage,
                self.model_attributes.table_nemomod_total_annual_max_capacity_investment_storage,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                scalar_to_nemomod_units = scalar_total_annual_max_capacity_investment_storage,
                drop_flag = self.drop_flag_tech_capacities
            )
        )
        # TotalAnnualMinCapacityStorage
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_total_annual_min_capacity_storage,
                self.model_attributes.table_nemomod_total_annual_min_capacity_storage,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                scalar_to_nemomod_units = scalar_total_annual_min_capacity_storage,
                drop_flag = self.drop_flag_tech_capacities
            )
        )
        # TotalAnnualMinCapacityInvestmentStorage
        dict_return.update(
            self.format_model_variable_as_nemomod_table(
                df_elec_trajectories,
                self.modvar_enst_nemomod_total_annual_min_capacity_investment_storage,
                self.model_attributes.table_nemomod_total_annual_min_capacity_investment_storage,
                [
                    self.field_nemomod_id,
                    self.field_nemomod_year,
                    self.field_nemomod_region
                ],
                self.field_nemomod_storage,
                scalar_to_nemomod_units = scalar_total_annual_min_capacity_investment_storage,
                drop_flag = self.drop_flag_tech_capacities
            )
        )


        ##  CHECK MAX/MIN RELATIONSHIP--SWAP VALUES IF NEEDED

        # check tables - capacity
        dfs_verify = self.verify_min_max_constraint_inputs(
            dict_return[self.model_attributes.table_nemomod_total_annual_max_capacity_storage],
            dict_return[self.model_attributes.table_nemomod_total_annual_min_capacity_storage],
            self.field_nemomod_value,
            self.field_nemomod_value,
            field_id = self.field_nemomod_id
        )
        if dfs_verify is not None:
            dict_return.update(
                {
                    self.model_attributes.table_nemomod_total_annual_max_capacity_storage: dfs_verify[0],
                    self.model_attributes.table_nemomod_total_annual_min_capacity_storage: dfs_verify[1]
                }
            )

        # check tables - capacity investment
        dfs_verify = self.verify_min_max_constraint_inputs(
            dict_return[self.model_attributes.table_nemomod_total_annual_max_capacity_investment_storage],
            dict_return[self.model_attributes.table_nemomod_total_annual_min_capacity_investment_storage],
            self.field_nemomod_value,
            self.field_nemomod_value,
            field_id = self.field_nemomod_id
        )
        if dfs_verify is not None:
            dict_return.update(
                {
                    self.model_attributes.table_nemomod_total_annual_max_capacity_investment_storage: dfs_verify[0],
                    self.model_attributes.table_nemomod_total_annual_min_capacity_investment_storage: dfs_verify[1]
                }
            )

        return dict_return







    ######################################################
    #    SQL FUNCTIONS FOR POPULATION OF THE DATABASE    #
    ######################################################

    def retrieve_tables_for_sql(self,
        df_elec_trajectories: pd.DataFrame,
        df_reference_capacity_factor: pd.DataFrame,
        df_reference_specified_demand_profile: pd.DataFrame,
        dict_attributes: dict = {}
    ) -> dict:
        """
            Retrieve tables from applicable inputs and format as dictionary. Returns a dictionary of the form

                {NemoModTABLE: df_table, ...}

            where NemoModTABLE is an appropriate table.

            - df_elec_trajectories: input of required variabels passed from other SISEPUEDE sectors.
            - df_reference_capacity_factor: reference data frame containing capacity factors
            - df_reference_specified_demand_profile:r eference data frame containing the specified demand profile by region
            - dict_attributes: dictionary of attribute tables that can be used to pass attributes to downstream format_nemomod_table_ functions. If passed, the following keys are used to represent attributes:
                * attribute_emission: EMISSION attribute table
                * attribute_fuel: FUEL attribute table
                * attribute_mode: MODE attribute table
                * attribute_storage: STORAGE attribute table
                * attribute_technology: TECHNOLOGY attribute table
                * attribute_time_slice: TIMESLICE attribute table
        """

        ##  INITIALIZE SHARED COMPONENTS

        # initilize attribute tables to pass--if they are not in the dictionary, they will return None, and defaults are used
        attribute_emission = dict_attributes.get("attribute_emission")
        attribute_fuel = dict_attributes.get("attribute_fuel")
        attribute_mode = dict_attributes.get("attribute_mode")
        attribute_nemomod_table = dict_attributes.get("nemomod_table")
        attribute_region =  dict_attributes.get("attribute_region")
        attribute_storage = dict_attributes.get("attribute_storage")
        attribute_technology = dict_attributes.get("attribute_technology")
        attribute_time_period = dict_attributes.get("attribute_time_period")
        attribute_time_slice = dict_attributes.get("attribute_time_slice")


        ##  BUILD TABLES FOR NEMOMOD

        dict_out = {}
        # start with basic attribute tables and time slice tables
        dict_out.update(self.format_nemomod_attribute_table_emission(attribute_emission = attribute_emission))
        dict_out.update(self.format_nemomod_attribute_table_fuel(attribute_fuel = attribute_fuel))
        dict_out.update(self.format_nemomod_attribute_table_mode_of_operation(attribute_mode = attribute_mode))
        dict_out.update(self.format_nemomod_attribute_table_region(attribute_region = attribute_region))
        dict_out.update(self.format_nemomod_attribute_table_storage(attribute_storage = attribute_storage))
        dict_out.update(self.format_nemomod_attribute_table_technology(attribute_technology = attribute_technology))
        dict_out.update(self.format_nemomod_attribute_table_year())
        dict_out.update(self.format_nemomod_table_tsgroup_tables(attribute_time_slice = attribute_time_slice))

        # DefaultParams
        dict_out.update(self.format_nemomod_table_default_parameters(attribute_nemomod_table = attribute_nemomod_table))
        # OperationalLife and OperationalLifeStorage
        dict_out.update(
            self.format_nemomod_table_operational_life(
                attribute_fuel = attribute_fuel,
                attribute_storage = attribute_storage,
                attribute_technology = attribute_technology
            )
        )
        # ReserveMarginTagFuel
        dict_out.update(self.format_nemomod_table_reserve_margin_tag_fuel())
        # RETagTechnology
        dict_out.update(self.format_nemomod_table_re_tag_technology(attribute_technology = attribute_technology))
        # TechnologyFromStorage and TechnologyToStorage
        dict_out.update(
            self.format_nemomod_table_technology_from_and_to_storage(
                attribute_storage = attribute_storage,
                attribute_technology = attribute_technology
            )
        )

        # add those dependent on input variables
        if df_elec_trajectories is not None:
            # AnnualEmissionLimit
            dict_out.update(
                self.format_nemomod_table_annual_emission_limit(
                    df_elec_trajectories,
                    attribute_emission = attribute_emission,
                    attribute_time_period = attribute_time_period
                )
            )
            # CapitalCostStorage
            dict_out.update(self.format_nemomod_table_costs_storage(df_elec_trajectories))
            # CapitalCost, FixedCost, and VariableCost -- Costs (Technology)
            dict_out.update(self.format_nemomod_table_costs_technology(df_elec_trajectories))
            # EmissionsActivityRatio - Emission Factors
            dict_out.update(self.format_nemomod_table_emissions_activity_ratio(df_elec_trajectories, attribute_time_period = attribute_time_period))
            # InputActivityRatio
            dict_out.update(self.format_nemomod_table_input_activity_ratio(df_elec_trajectories, attribute_technology = attribute_technology))
            # MinStorageCharge
            dict_out.update(self.format_nemomod_table_min_storage_charge(df_elec_trajectories, attribute_storage = attribute_storage))
            # OutputActivityRatio
            dict_out.update(self.format_nemomod_table_output_activity_ratio(df_elec_trajectories, attribute_technology = attribute_technology))
            # ReserveMargin
            dict_out.update(self.format_nemomod_table_reserve_margin(df_elec_trajectories))
            # ReserveMarginTagTechnology
            dict_out.update(self.format_nemomod_table_reserve_margin_tag_technology(df_elec_trajectories))
            # ResidualCapacity
            dict_out.update(self.format_nemomod_table_residual_capacity(df_elec_trajectories))
            # ResidualStorageCapacity
            dict_out.update(self.format_nemomod_table_residual_storage_capacity(df_elec_trajectories))
            # SpecifiedAnnualDemand
            dict_out.update(self.format_nemomod_table_specified_annual_demand(df_elec_trajectories, attribute_time_period = attribute_time_period))
            # StorageMaxChargeRate (if included), StorageMaxDishargeRate (if included), and StorageStartLevel
            dict_out.update(self.format_nemomod_table_storage_attributes(df_elec_trajectories))
            # TotalAnnualMax/MinCapacity +/-Investment
            dict_out.update(self.format_nemomod_table_total_capacity_tables(df_elec_trajectories))
            # TotalAnnualMax/MinCapacity +/-Investment Storage
            dict_out.update(self.format_nemomod_table_total_capacity_storage_tables(df_elec_trajectories))

        # CapacityFactor
        if df_reference_capacity_factor is not None:
            dict_out.update(
                self.format_nemomod_table_capacity_factor(
                    df_reference_capacity_factor,
                    attribute_technology = attribute_technology,
                    attribute_region = attribute_region
                )
            )
        # SpecifiedDemandProfile
        if df_reference_specified_demand_profile is not None:
            dict_out.update(self.format_nemomod_table_specified_demand_profile(df_reference_specified_demand_profile))


        ##  Prepare data for write
        for table in dict_out.keys():
            dict_dtype = {}
            for k in self.dict_fields_nemomod_to_type.keys():
                dict_dtype.update({k: self.dict_fields_nemomod_to_type[k]}) if (k in dict_out[table].columns) else None

            dict_out[table] = dict_out[table].astype(dict_dtype)

        return dict_out
