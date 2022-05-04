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

class NonElectricEnergy:

    def __init__(self, attributes: ds.ModelAttributes):

        self.model_attributes = attributes
        self.required_dimensions = self.get_required_dimensions()
        self.required_subsectors, self.required_base_subsectors = self.get_required_subsectors()
        self.required_variables, self.output_variables = self.get_neenergy_input_output_fields()

        ##  set some model fields to connect to the attribute tables

        # Energy Fuel model variables
        self.modvar_enfu_ef_combustion_co2 = ":math:\\text{CO}_2 Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_mobile_ch4 = ":math:\\text{CH}_4 Mobile Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_mobile_n2o = ":math:\\text{N}_2\\text{O} Mobile Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_stationary_ch4 = ":math:\\text{CH}_4 Stationary Combustion Emission Factor"
        self.modvar_enfu_ef_combustion_stationary_n2o = ":math:\\text{N}_2\\text{O} Stationary Combustion Emission Factor"

        # Industrial Energy model variables
        self.modvar_inen_demscalar = "Industrial Energy Demand Scalar"
        self.modvar_inen_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Industrial Energy"
        self.modvar_inen_emissions_co2 = ":math:\\text{CO}_2 Emissions from Industrial Energy"
        self.modvar_inen_emissions_n2o = ":math:\\text{N}_2\\text{O} Emissions from Industrial Energy"
        self.modvar_inen_energy_demand_electricity = "Electrical Energy Demand from Industrial Energy"
        self.modvar_inen_energy_demand_electricity_agg = "Total Electrical Energy Demand from Industrial Energy"
        self.modvar_inen_energy_demand_total = "Energy Demand from Industrial Energy"
        self.modvar_inen_energy_demand_total_agg = "Total Energy Demand from Industrial Energy"
        self.modvar_inen_en_gdp_intensity_factor = "GDP Energy Intensity Factor"
        self.modvar_inen_en_prod_intensity_factor = "Production Energy Intesity Factor"
        self.modvar_inen_frac_en_coal = "Industrial Energy Fraction Coal"
        self.modvar_inen_frac_en_coke = "Industrial Energy Fraction Coke"
        self.modvar_inen_frac_en_diesel = "Industrial Energy Fraction Diesel"
        self.modvar_inen_frac_en_electricity = "Industrial Energy Fraction Electricity"
        self.modvar_inen_frac_en_furnace_gas = "Industrial Energy Fraction Furnace Gas"
        self.modvar_inen_frac_en_gasoline = "Industrial Energy Fraction Gasoline"
        self.modvar_inen_frac_en_hydrogen = "Industrial Energy Fraction Hydrogen"
        self.modvar_inen_frac_en_kerosene = "Industrial Energy Fraction Kerosene"
        self.modvar_inen_frac_en_natural_gas = "Industrial Energy Fraction Natural Gas"
        self.modvar_inen_frac_en_oil = "Industrial Energy Fraction Oil"
        self.modvar_inen_frac_en_pliqgas = "Industrial Energy Fraction Petroleum Liquid Gas"
        self.modvar_inen_frac_en_solar = "Industrial Energy Fraction Solar"
        self.modvar_inen_frac_en_solid_biomass = "Industrial Energy Fraction Solid Biomass"
        # fuel fractions to check summation over
        self.modvar_inen_list_fuel_fractions = [
            self.modvar_inen_frac_en_coal,
            self.modvar_inen_frac_en_coke,
            self.modvar_inen_frac_en_diesel,
            self.modvar_inen_frac_en_electricity,
            self.modvar_inen_frac_en_furnace_gas,
            self.modvar_inen_frac_en_gasoline,
            self.modvar_inen_frac_en_hydrogen,
            self.modvar_inen_frac_en_kerosene,
            self.modvar_inen_frac_en_natural_gas,
            self.modvar_inen_frac_en_oil,
            self.modvar_inen_frac_en_pliqgas,
            self.modvar_inen_frac_en_solar,
            self.modvar_inen_frac_en_solid_biomass
        ]



        # variables from other sectors
        self.modvar_ippu_qty_total_production = "Industrial Production"

        # add other model classes
        self.model_socioeconomic = Socioeconomic(self.model_attributes)
        self.model_ippu = IPPU(self.model_attributes)

        # optional integration variables (uses calls to other model classes)
        self.integration_variables = self.set_integrated_variables()

        ##  MISCELLANEOUS VARIABLES
        self.time_periods, self.n_time_periods = self.model_attributes.get_time_periods()
        self.enfu_fuel_electricity = self.get_electricity_fuel()



    ##  FUNCTIONS FOR MODEL ATTRIBUTE DIMENSIONS

    def check_df_fields(self, df_neenergy_trajectories):
        check_fields = self.required_variables
        # check for required variables
        if not set(check_fields).issubset(df_neenergy_trajectories.columns):
            set_missing = list(set(check_fields) - set(df_neenergy_trajectories.columns))
            set_missing = sf.format_print_list(set_missing)
            raise KeyError(f"IPPU projection cannot proceed: The fields {set_missing} are missing.")

    def get_electricity_fuel(self):
        return self.model_attributes.get_categories_from_attribute_characteristic("Energy Fuels", {self.model_attributes.field_enfu_electricity_demand_category: 1})[0]

    def get_required_subsectors(self):
        ## TEMPORARY
        subsectors = ["Industrial Energy", "Energy Fuels"]#self.model_attributes.get_setor_subsectors("Energy")
        subsectors_base = subsectors.copy()
        subsectors += ["Economy", "General"]
        return subsectors, subsectors_base

    def get_required_dimensions(self):
        ## TEMPORARY - derive from attributes later
        required_doa = [self.model_attributes.dim_time_period]
        return required_doa

    def get_neenergy_input_output_fields(self):
        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(self.required_subsectors)

        return required_vars + self.get_required_dimensions(), output_vars


    ##  function to set alternative sets of input variables; leave empty for now
    def get_neenergy_optional_switch_variables(self) -> dict:
        """
           get_neenergy_optional_switch_variables() defines dictionaries of lists of variables. Returns a nested dictionary specified in the class.

           Output Structure
           ----------------
           {
               "varset_1": {
                   "primary": [primary variables...],
                   "secondary": [primary variables...]
                },
               "varset_2": {
                   "primary": [primary variables...],
                   "secondary": [primary variables...]
                },
               ...
           }

           Notes
           -----
           - In each dictionary, variables from the "primary" key *or* variables from the "secondary" key must be defined. In general, "primary" variables are associated with integration. In the absence of these variables, secondary variables are generally calculated endogenously.
           - Each variable set represents a different approach
           - If all variables are defined in the input data frame, then the approach associated with "primary" variables is used.
        """

        return {}


    # variables required to integration
    def set_integrated_variables(self):
        # set the integration variables
        list_vars_required_for_integration = [
            self.modvar_ippu_qty_total_production
        ]

        # in Energy, update required variables
        for modvar in list_vars_required_for_integration:
            subsec = self.model_attributes.get_variable_subsector(modvar)
            new_vars = self.model_attributes.build_varlist(subsec, modvar)
            self.required_variables += new_vars

        # sot required variables and ensure no double counting
        self.required_variables = list(set(self.required_variables))
        self.required_variables.sort()

        return list_vars_required_for_integration




    ######################################
    #    SUBSECTOR SPECIFIC FUNCTIONS    #
    ######################################



    ########################################
    ###                                  ###
    ###    PRIMARY PROJECTION METHODS    ###
    ###                                  ###
    ########################################

    ##  industrial energy model
    def project_industrial_energy(
        self,
        df_neenergy_trajectories: pd.DataFrame,
        vec_gdp: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
            project_industrial_energy can be called from other sectors to simplify calculation of industrial energy.

            Function Arguments
            ------------------
            df_neenergy_trajectories: pd.DataFrame of input variables

            vec_gdp: np.ndarray vector of gdp (requires len(vec_gdp) == len(df_neenergy_trajectories))

            dict_dims: dict of dimensions (returned from check_projection_input_df). Default is None.

            n_projection_time_periods: int giving number of time periods (returned from check_projection_input_df). Default is None.

            projection_time_periods: list of time periods (returned from check_projection_input_df). Default is None.


            Notes
            -----
            If any of dict_dims, n_projection_time_periods, or projection_time_periods are unspecified (expected if ran outside of Energy.project()), self.model_attributes.check_projection_input_df wil be run

        """

        # allows production to be run outside of the project method
        if type(None) in set([type(x) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            dict_dims, df_neenergy_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_neenergy_trajectories, True, True, True)


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_enfu = self.model_attributes.get_subsector_attribute("Energy Fuels", "pycategory_primary")
        pycat_inen = self.model_attributes.get_subsector_attribute("Industrial Energy", "pycategory_primary")
        pycat_ippu = self.model_attributes.get_subsector_attribute("IPPU", "pycategory_primary")
        # attribute tables
        attr_enfu = self.model_attributes.dict_attributes[pycat_enfu]
        attr_inen = self.model_attributes.dict_attributes[pycat_inen]
        attr_ippu = self.model_attributes.dict_attributes[pycat_ippu]


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]


        ############################
        #    MODEL CALCULATIONS    #
        ############################

        # first, retrieve energy fractions and ensure they sum to 1
        dict_arrs_inen_frac_energy = sa.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_cs_integrated,
            model_energy.modvar_inen_list_fuel_fractions,
            1,
            force_sum_equality = True,
            msg_append = "Energy fractions by category do not sum to 1. See definition of dict_arrs_inen_frac_energy."
        )


        ##  GET ENERGY INTENSITIES

        # get production-based emissions - start with production, energy demand
        arr_inen_prod = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_ippu_qty_total_production, True, "array_base", expand_to_all_cats = True)
        arr_inen_prod_energy_intensity = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_inen_en_prod_intensity_factor, True, "array_base", expand_to_all_cats = True)
        scalar_inen_prod_intensity_to_total_prod = self.model_attributes.get_mass_equivalent(
            self.model_attributes.get_variable_characteristic(self.modvar_inen_en_prod_intensity_factor, "$UNIT-MASS$"),
            self.model_attributes.get_variable_characteristic(self.modvar_ippu_qty_total_production, "$UNIT-MASS$")
        )
        # energy intensity due to production in terms of units self.modvar_ippu_qty_total_production
        arr_inen_energy_demand = arr_inen_prod*arr_inen_prod_energy_intensity*scalar_inen_prod_intensity_to_total_prod
        # gdp-based emissions - get intensity, multiply by gdp, and scale to match energy units of production
        arr_inen_gdp_energy_intensity = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_inen_en_gdp_intensity_factor, True, "array_base", expand_to_all_cats = True)
        scalar_inen_gdp_energy_to_prod_energy = self.model_attributes.get_energy_equivalent(
            self.model_attributes.get_variable_characteristic(self.modvar_inen_en_gdp_intensity_factor, "$UNIT-ENERGY$"),
            self.model_attributes.get_variable_characteristic(self.modvar_inen_en_prod_intensity_factor, "$UNIT-ENERGY$")
        )
        arr_inen_energy_demand += (arr_inen_gdp_energy_intensity.transpose() * vec_gdp).transpose()*scalar_inen_gdp_energy_to_prod_energy


        ##  GET EMISSION FACTORS

        # methane - scale to ensure energy units are the same
        arr_inen_ef_by_fuel_ch4 = sa.model_attributes.get_standard_variables(df_cs_integrated, self.modvar_enfu_ef_combustion_stationary_ch4, return_type = "array_units_corrected")
        arr_inen_ef_by_fuel_ch4 *= self.model_attributes.get_energy_equivalent(
            self.model_attributes.get_variable_characteristic(self.modvar_enfu_ef_combustion_stationary_ch4, "$UNIT-ENERGY$"),
            self.model_attributes.get_variable_characteristic(self.modvar_inen_en_prod_intensity_factor, "$UNIT-ENERGY$")
        )
        # carbon dioxide - scale to ensure energy units are the same
        arr_inen_ef_by_fuel_co2 = sa.model_attributes.get_standard_variables(df_cs_integrated, self.modvar_enfu_ef_combustion_co2, return_type = "array_units_corrected")
        arr_inen_ef_by_fuel_co2 *= self.model_attributes.get_energy_equivalent(
            self.model_attributes.get_variable_characteristic(self.modvar_enfu_ef_combustion_co2, "$UNIT-ENERGY$"),
            self.model_attributes.get_variable_characteristic(self.modvar_inen_en_prod_intensity_factor, "$UNIT-ENERGY$")
        )
        # nitrous oxide - scale to ensure energy units are the same
        arr_inen_ef_by_fuel_n2o = sa.model_attributes.get_standard_variables(df_cs_integrated, self.modvar_enfu_ef_combustion_stationary_n2o, return_type = "array_units_corrected")
        arr_inen_ef_by_fuel_n2o *= self.model_attributes.get_energy_equivalent(
            self.model_attributes.get_variable_characteristic(self.modvar_enfu_ef_combustion_stationary_n2o, "$UNIT-ENERGY$"),
            self.model_attributes.get_variable_characteristic(self.modvar_inen_en_prod_intensity_factor, "$UNIT-ENERGY$")
        )


        ##  CALCULATE EMISSIONS AND ELECTRICITY DEMAND

        # initialize electrical demand to pass and output emission arrays
        arr_inen_demand_electricity = 0.0
        arr_inen_demand_electricity_total = 0.0
        arr_inen_demand_total = 0.0
        arr_inen_demand_total_total = 0.0
        arr_inen_emissions_ch4 = 0.0
        arr_inen_emissions_co2 = 0.0
        arr_inen_emissions_n2o = 0.0
        # loop over fuels to
        for var_ener_frac in self.modvar_inen_list_fuel_fractions:
            # retrive the fuel category
            cat_fuel = ds.clean_schema(self.model_attributes.get_variable_attribute(var_ener_frac, pycat_enfu))
            # get the demand for the current fuel
            arr_inen_endem_cur_fuel = dict_arrs_inen_frac_energy[var_ener_frac].copy()
            arr_inen_endem_cur_fuel *= arr_inen_energy_demand
            # get the category value index and
            index_cat_fuel = attr_enfu.get_key_value_index(cat_fuel)
            arr_inen_emissions_ch4 += arr_inen_endem_cur_fuel.transpose()*arr_inen_ef_by_fuel_ch4[:, index_cat_fuel]
            arr_inen_emissions_co2 += arr_inen_endem_cur_fuel.transpose()*arr_inen_ef_by_fuel_co2[:, index_cat_fuel]
            arr_inen_emissions_n2o += arr_inen_endem_cur_fuel.transpose()*arr_inen_ef_by_fuel_n2o[:, index_cat_fuel]
            # add electricity demand and total energy demand
            arr_inen_demand_electricity += arr_inen_endem_cur_fuel if (cat_fuel == self.enfu_fuel_electricity) else 0.0
            arr_inen_demand_electricity_total += arr_inen_endem_cur_fuel.sum(axis = 1) if (cat_fuel == self.enfu_fuel_electricity) else 0.0
            arr_inen_demand_total += arr_inen_endem_cur_fuel
            arr_inen_demand_total_total += arr_inen_endem_cur_fuel.sum(axis = 1)

        # transpose outputs
        arr_inen_emissions_ch4 = arr_inen_emissions_ch4.transpose()
        arr_inen_emissions_co2 = arr_inen_emissions_co2.transpose()
        arr_inen_emissions_n2o = arr_inen_emissions_n2o.transpose()
        # set energy data frames
        scalar_energy = self.model_attributes.get_scalar(self.modvar_inen_en_prod_intensity_factor, "energy")


        ##  BUILD OUTPUT DFs

        df_out += [
            self.model_attributes.array_to_df(arr_inen_emissions_ch4, self.modvar_inen_emissions_ch4, False, True),
            self.model_attributes.array_to_df(arr_inen_emissions_co2, self.modvar_inen_emissions_co2, False, True),
            self.model_attributes.array_to_df(arr_inen_emissions_n2o, self.modvar_inen_emissions_n2o, False, True),
            self.model_attributes.array_to_df(arr_inen_demand_electricity*scalar_energy, self.modvar_inen_energy_demand_electricity, False, True),
            self.model_attributes.array_to_df(arr_inen_demand_electricity_total*scalar_energy, self.modvar_inen_energy_demand_electricity_agg, False),
            self.model_attributes.array_to_df(arr_inen_demand_total*scalar_energy, self.modvar_inen_energy_demand_total, False, True),
            self.model_attributes.array_to_df(arr_inen_demand_total_total*scalar_energy, self.modvar_inen_energy_demand_total_agg, False)
        ]

        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")
        self.model_attributes.add_subsector_emissions_aggregates(df_out, ["Industrial Energy"], False)

        return df_out


    ##  transportation
    def project_transportation():

        return 0


    ##  other energy: stationary emissions and carbon capture and sequestration
    def project_oesc():

        return 0


    ##  primary method
    def project(self, df_neenergy_trajectories):

        """
            The Energy.project() method takes a data frame of input variables (ordered by time series) and returns a data frame of output variables (model projections for energy--including industrial energy, transportation, stationary emissions, carbon capture and sequestration, and electricity) the same order.

            Function Arguments
            ------------------
            df_neenergy_trajectories: pd.DataFrame with all required input fields as columns. The model will not run if any required variables are missing, but errors will detail which fields are missing.

            Notes
            -----
            - The .project() method is designed to be parallelized or called from command line via __main__ in run_sector_models.py.
            - df_neenergy_trajectories should have all input fields required (see IPPU.required_variables for a list of variables to be defined)
            - the df_neenergy_trajectories.project method will run on valid time periods from 1 .. k, where k <= n (n is the number of time periods). By default, it drops invalid time periods. If there are missing time_periods between the first and maximum, data are interpolated.
        """

        ##  CHECKS

        # make sure socioeconomic variables are added and
        df_neenergy_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_neenergy_trajectories)
        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_neenergy_trajectories)
        dict_dims, df_neenergy_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_neenergy_trajectories, True, True, True)


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_fuel = self.model_attributes.get_subsector_attribute("Energy Fuels", "pycategory_primary")
        pycat_gnrl = self.model_attributes.get_subsector_attribute("General", "pycategory_primary")
        pycat_inen = self.model_attributes.get_subsector_attribute("Industrial Energy", "pycategory_primary")
        pycat_ippu = self.model_attributes.get_subsector_attribute("IPPU", "pycategory_primary")
        pycat_oesc = self.model_attributes.get_subsector_attribute("Other Energy: Stationary Emissions and Carbon Capture and Sequestration", "pycategory_primary")
        pycat_trns = self.model_attributes.get_subsector_attribute("Transportation", "pycategory_primary")
        # attribute tables
        attr_fuel = self.model_attributes.dict_attributes[pycat_fuel]
        attr_gnrl = self.model_attributes.dict_attributes[pycat_gnrl]
        attr_inen = self.model_attributes.dict_attributes[pycat_inen]
        attr_ippu = self.model_attributes.dict_attributes[pycat_ippu]
        attr_oesc = self.model_attributes.dict_attributes[pycat_oesc]
        attr_trns = self.model_attributes.dict_attributes[pycat_trns]


        ##  ECON/GNRL VECTOR AND ARRAY INITIALIZATION

        # get some vectors from the se model
        vec_gdp = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.model_socioeconomic.modvar_econ_gdp, False, return_type = "array_base")
        vec_pop = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.model_socioeconomic.modvar_gnrl_pop_total, False, return_type = "array_base")
        array_pop = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.model_socioeconomic.modvar_gnrl_subpop, False, return_type = "array_base")
        vec_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_gdp_per_capita"])
        vec_rates_gdp = np.array(df_se_internal_shared_variables["vec_rates_gdp"].dropna())
        vec_rates_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_rates_gdp_per_capita"].dropna())

        #
        df_out = self.project_industrial_energy(df_neenergy_trajectories, vec_gdp, dict_dims, n_projection_time_periods, projection_time_periods)

        return df_out
