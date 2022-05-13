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
        self.modvar_enfu_volumetric_energy_density = "Volumetric Energy Density"

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

        # Transportation variables
        self.modvar_trns_average_vehicle_load_freight = "Average Freight Vehicle Load"
        self.modvar_trns_average_passenger_occupancy = "Average Passenger Vehicle Occupancy Rate"
        self.modvar_trns_electrical_efficiency = "Electrical Vehicle Efficiency"
        self.modvar_trns_ef_combustion_mobile_biofuels_ch4 = ":math:\\text{CH}_4 Biofuels Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_diesel_ch4 = ":math:\\text{CH}_4 Diesel Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_gasoline_ch4 = ":math:\\text{CH}_4 Gasoline Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_kerosene_ch4 = ":math:\\text{CH}_4 Kerosene Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_natural_gas_ch4 = ":math:\\text{CH}_4 Natural Gas Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_biofuels_n2o = ":math:\\text{N}_2\\text{O} Biofuels Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_diesel_n2o = ":math:\\text{N}_2\\text{O} Diesel Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_gasoline_n2o = ":math:\\text{N}_2\\text{O} Gasoline Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_kerosene_n2o = ":math:\\text{N}_2\\text{O} Kerosene Mobile Combustion Emission Factor"
        self.modvar_trns_ef_combustion_mobile_natural_gas_n2o = ":math:\\text{N}_2\\text{O} Natural Gas Mobile Combustion Emission Factor"
        self.modvar_trns_fuel_demand_biofuels_agg = "Total Fuel Demand Biofuels"
        self.modvar_trns_fuel_demand_diesel_agg = "Total Fuel Demand Diesel"
        self.modvar_trns_fuel_demand_gasoline_agg = "Total Fuel Demand Gasoline"
        self.modvar_trns_fuel_demand_hydrogen_agg = "Total Fuel Demand Hydrogen"
        self.modvar_trns_fuel_demand_kerosene_agg = "Total Fuel Demand Biofuels"
        self.modvar_trns_fuel_demand_natural_gas_agg = "Total Fuel Demand NaturalGas"
        self.modvar_trns_fuel_efficiency_biofuels = "Fuel Efficiency Biofuels"
        self.modvar_trns_fuel_efficiency_diesel = "Fuel Efficiency Diesel"
        self.modvar_trns_fuel_efficiency_gasoline = "Fuel Efficiency Gasoline"
        self.modvar_trns_fuel_efficiency_hydrogen = "Fuel Efficiency Hydrogen"
        self.modvar_trns_fuel_efficiency_kerosene = "Fuel Efficiency Kerosene"
        self.modvar_trns_fuel_efficiency_natural_gas = "Fuel Efficiency Natural Gas"
        self.modvar_trns_modeshare_freight = "Freight Transportation Mode Share"
        self.modvar_trns_modeshare_public_private = "Private and Public Transportation Mode Share"
        self.modvar_trns_modeshare_regional = "Regional Transportation Mode Share"
        self.modvar_trns_fuel_fraction_biofuels = "Transportation Mode Fuel Fraction Biofuels"
        self.modvar_trns_fuel_fraction_diesel = "Transportation Mode Fuel Fraction Diesel"
        self.modvar_trns_fuel_fraction_electricity = "Transportation Mode Fuel Fraction Electricity"
        self.modvar_trns_fuel_fraction_gasoline = "Transportation Mode Fuel Fraction Gasoline"
        self.modvar_trns_fuel_fraction_hydrogen = "Transportation Mode Fuel Fraction Hydrogen"
        self.modvar_trns_fuel_fraction_kerosene = "Transportation Mode Fuel Fraction Kerosene"
        self.modvar_trns_fuel_fraction_natural_gas = "Transportation Mode Fuel Fraction Natural Gas"
        self.modvar_tnrs_energy_demand_electricity = "Electrical Energy Demand from Transportation"
        self.modvar_tnrs_energy_demand_electricity_agg = "Total Electrical Energy Demand from Transportation"
        self.modvar_trns_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Transportation"
        self.modvar_trns_emissions_co2 = ":math:\\text{CO}_2 Emissions from Transportation"
        self.modvar_trns_emissions_n2o = ":math:\\text{N}_2\\text{O} Emissions from Transportation"
        self.modvar_trns_vehicle_distance_traveled = "Total Vehicle Distance Traveled"

        # Transportation Demand variables
        self.modvar_trde_demand_scalar = "Transportation Demand Scalar"
        self.modvar_trde_elasticity_mtkm_to_gdp = "Elasticity of Megatonne-Kilometer Demand to GDP"
        self.modvar_trde_elasticity_pkm_to_gdp = "Elasticity of Passenger-Kilometer Demand per Capita to GDP per Capita"
        self.modvar_trde_demand_initial_mtkm = "Initial Megatonne-Kilometer Demand"
        self.modvar_trde_demand_initial_pkm_per_capita = "Initial per Capita Passenger-Kilometer Demand"
        self.modvar_trde_demand_mtkm = "Megatonne-Kilometer Demand"
        self.modvar_trde_demand_pkm = "Passenger-Kilometer Demand"




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

        # fuel variables dictionary for transportation
        self.dict_trns_fuel_categories_to_fuel_variables, self.dict_trns_fuel_categories_to_unassigned_fuel_variables = self.get_dict_trns_fuel_categories_to_fuel_variables()
        # some derivate lists of variables
        self.modvars_trns_list_fuel_fraction = self.model_attributes.get_vars_by_assigned_class_from_akaf(
            self.dict_trns_fuel_categories_to_fuel_variables,
            "fuel_fraction"
        )
        self.modvars_trns_list_fuel_efficiency = self.model_attributes.get_vars_by_assigned_class_from_akaf(
            self.dict_trns_fuel_categories_to_fuel_variables,
            "fuel_efficiency"
        )



    ##  FUNCTIONS FOR MODEL ATTRIBUTE DIMENSIONS

    def check_df_fields(self,
        df_neenergy_trajectories: pd.DataFrame,
        subsector: str = "All",
        var_type: str = "input",
        msg_prepend: str = None
    ):
        if subsector == "All":
            check_fields = self.required_variables
            msg_prepend = "Energy"
        else:
            self.model_attributes.check_subsector(subsector)
            if var_type == "input":
                check_fields, ignore_fields = self.model_attributes.get_input_output_fields(["Economy", "General", subsector])
            elif var_type == "output":
                ignore_fields, check_fields = self.model_attributes.get_input_output_fields([subsector])
            else:
                raise ValueError(f"Invalid var_type '{var_type}' in check_df_fields: valid types are 'input', 'output'")
            msg_prepend = msg_prepend if (msg_prepend is not None) else subsector
        sf.check_fields(df_neenergy_trajectories, check_fields, f"{msg_prepend} projection cannot proceed: fields ")

    def get_electricity_fuel(self):
        return self.model_attributes.get_categories_from_attribute_characteristic("Energy Fuels", {self.model_attributes.field_enfu_electricity_demand_category: 1})[0]

    def get_required_subsectors(self):
        ## TEMPORARY
        subsectors = ["Industrial Energy", "Energy Fuels", "Transportation", "Transportation Demand"]#self.model_attributes.get_setor_subsectors("Energy")
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

    ##  transportation variables from fuel categories as specified by a matchstring
    def get_dict_trns_fuel_categories_to_fuel_variables(self):
        """
            use get_dict_trns_fuel_categories_to_fuel_variables to return a dictionary with fuel categories as keys based on the Transportation attribute table;
            {cat_fuel: {"fuel_efficiency": VARNAME_FUELEFFICIENCY, ...}}

            for each key, the dict includes variables associated with the fuel cat_fuel:

            - "fuel_efficiency"
            - "fuel_fraction"
            - "ef_ch4"
            - "ef_n2o"

        """

        dict_out = self.model_attributes.assign_keys_from_attribute_fields(
            "Transportation",
            "cat_fuel",
            {
                "Fuel Efficiency": "fuel_efficiency",
                "Fuel Fraction": "fuel_fraction",
                ":math:\\text{CH}_4": "ef_ch4",
                ":math:\\text{N}_2\\text{O}": "ef_n2o",
                "Total Fuel Demand": "total_fuel_demand"
            },
            "varreqs_partial",
            True
        )

        return dict_out


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
        dict_arrs_inen_frac_energy = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_neenergy_trajectories,
            self.modvar_inen_list_fuel_fractions,
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
        arr_inen_ef_by_fuel_ch4 = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_ef_combustion_stationary_ch4, return_type = "array_units_corrected")
        arr_inen_ef_by_fuel_ch4 *= self.model_attributes.get_energy_equivalent(
            self.model_attributes.get_variable_characteristic(self.modvar_enfu_ef_combustion_stationary_ch4, "$UNIT-ENERGY$"),
            self.model_attributes.get_variable_characteristic(self.modvar_inen_en_prod_intensity_factor, "$UNIT-ENERGY$")
        )
        # carbon dioxide - scale to ensure energy units are the same
        arr_inen_ef_by_fuel_co2 = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_ef_combustion_co2, return_type = "array_units_corrected")
        arr_inen_ef_by_fuel_co2 *= self.model_attributes.get_energy_equivalent(
            self.model_attributes.get_variable_characteristic(self.modvar_enfu_ef_combustion_co2, "$UNIT-ENERGY$"),
            self.model_attributes.get_variable_characteristic(self.modvar_inen_en_prod_intensity_factor, "$UNIT-ENERGY$")
        )
        # nitrous oxide - scale to ensure energy units are the same
        arr_inen_ef_by_fuel_n2o = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_ef_combustion_stationary_n2o, return_type = "array_units_corrected")
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



    ##  transportation emissions
    def project_transportation(self,
        df_neenergy_trajectories: pd.DataFrame,
        vec_pop: np.ndarray,
        vec_rates_gdp: np.ndarray,
        vec_rates_gdp_per_capita: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
            project_transportation can be called from other sectors to simplify calculation of transportation emissions and associated metrics. Requires NonElectricEnergy.project_transportation_demand() and all variables from the transportation demand sector

            Function Arguments
            ------------------
            df_neenergy_trajectories: pd.DataFrame of input variables

            vec_pop: np.ndarray vector of population (requires len(vec_rates_gdp) == len(df_neenergy_trajectories))

            vec_rates_gdp: np.ndarray vector of gdp growth rates (v_i = growth rate from t_i to t_{i + 1}) (requires len(vec_rates_gdp) == len(df_neenergy_trajectories) - 1)

            vec_rates_gdp_per_capita: np.ndarray vector of gdp per capita growth rates (v_i = growth rate from t_i to t_{i + 1}) (requires len(vec_rates_gdp_per_capita) == len(df_neenergy_trajectories) - 1)

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

        # check fields - transportation demand; if not present, add to the dataframe
        self.check_df_fields(df_neenergy_trajectories, "Transportation")
        try:
            self.check_df_fields(df_neenergy_trajectories, "Transportation Demand", "output", "Transportation")
        except:
            df_transport_demand = self.project_transportation_demand(
                df_neenergy_trajectories,
                vec_pop,
                vec_rates_gdp,
                vec_rates_gdp_per_capita,
                dict_dims,
                n_projection_time_periods,
                projection_time_periods
            )
            df_neenergy_trajectories = sf.merge_output_df_list([df_neenergy_trajectories, df_transport_demand], self.model_attributes, "concatenate")


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_enfu = self.model_attributes.get_subsector_attribute("Energy Fuels", "pycategory_primary")
        pycat_trde = self.model_attributes.get_subsector_attribute("Transportation Demand", "pycategory_primary")
        pycat_trns = self.model_attributes.get_subsector_attribute("Transportation", "pycategory_primary")
        # attribute tables
        attr_enfu = self.model_attributes.dict_attributes[pycat_enfu]
        attr_trde = self.model_attributes.dict_attributes[pycat_trde]
        attr_trns = self.model_attributes.dict_attributes[pycat_trns]


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]



        ############################
        #    MODEL CALCULATIONS    #
        ############################


        ##  START WITH DEMANDS

        # start with demands and map categories in attribute to associated variable
        dict_trns_vars_to_trde_cats = self.model_attributes.get_ordered_category_attribute("Transportation", "cat_transportation_demand", "key_varreqs_partial", True, dict, True)
        dict_trns_vars_to_trde_cats = sf.reverse_dict(dict_trns_vars_to_trde_cats)
        array_trns_total_vehicle_demand = 0.0
        # get occupancy and freight occupancies
        array_trns_avg_load_freight = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trns_average_vehicle_load_freight, return_type = "array_base", expand_to_all_cats = True)
        array_trns_occ_rate_passenger = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trns_average_passenger_occupancy, return_type = "array_base", expand_to_all_cats = True)
        # convert average load to same units as demand
        array_trns_avg_load_freight *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_trns_average_vehicle_load_freight,
            self.modvar_trde_demand_mtkm,
            "mass"
        )
        # convert freight vehicle demand to same length units as passenger
        scalar_tnrs_length_demfrieght_to_dempass = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_trde_demand_mtkm,
            self.modvar_trde_demand_pkm,
            "length"
        )

        # loop over the demand categories to get transportation demand
        for category in dict_trns_vars_to_trde_cats.keys():
            # get key index, model variable, and the current demand
            index_key = self.model_attributes.get_attribute_table("Transportation Demand").get_key_value_index(category)
            modvar = self.model_attributes.get_variable_from_category("Transportation Demand", category, "partial")
            vec_trde_dem_cur = self.model_attributes.get_standard_variables(df_neenergy_trajectories, modvar, return_type = "array_base", expand_to_all_cats = True)[:, index_key]
            # retrieve the demand mix, convert to total activity-demand by category, then divide by freight/occ_rate
            array_trde_dem_cur_by_cat = self.model_attributes.get_standard_variables(
                df_neenergy_trajectories,
                dict_trns_vars_to_trde_cats[category],
                return_type = "array_base",
                expand_to_all_cats = True,
                var_bounds = (0, 1),
                force_boundary_restriction = True
            )
            # ru
            array_trde_dem_cur_by_cat = (array_trde_dem_cur_by_cat.transpose()*vec_trde_dem_cur).transpose()
            """
            freight and passenger should be mutually exclusive categories
            - e.g., if the iterating variable category == "freight", then array_trde_dem_cur_by_cat*array_trns_occ_rate_passenger should be 0
            - if category != "freight", then array_trde_dem_cur_by_cat*array_trns_avg_load_freight should be 0)

            - demand length units should be in terms of 'modvar_trns_average_passenger_occupancy' (see scalar multiplication)
            """
            array_trde_vehicle_dem_cur_by_cat = np.nan_to_num(array_trde_dem_cur_by_cat/array_trns_avg_load_freight, 0.0, neginf = 0.0, posinf = 0.0)*scalar_tnrs_length_demfrieght_to_dempass
            array_trde_vehicle_dem_cur_by_cat += np.nan_to_num(array_trde_dem_cur_by_cat/array_trns_occ_rate_passenger, 0.0, neginf = 0.0, posinf = 0.0)
            # update total vehicle-km demand
            array_trns_total_vehicle_demand += array_trde_vehicle_dem_cur_by_cat

        # add the vehicle distance to output using the units modvar_trde_demand_pkm
        scalar_trns_total_vehicle_demand = self.model_attributes.get_scalar(self.modvar_trde_demand_pkm, "length")
        df_out.append(
            self.model_attributes.array_to_df(array_trns_total_vehicle_demand*scalar_trns_total_vehicle_demand, self.modvar_trns_vehicle_distance_traveled, False, True),
        )


        ##  LOOP OVER FUELS

        # first, retrieve fuel-mix fractions and ensure they sum to 1
        dict_arrs_trns_frac_fuel = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_neenergy_trajectories,
            self.modvars_trns_list_fuel_fraction,
            1,
            force_sum_equality = False,
            msg_append = "Energy fractions by category do not sum to 1. See definition of dict_arrs_trns_frac_fuel."
        )
        # get carbon dioxide combustion factors (corrected to output units)
        arr_trns_ef_by_fuel_co2 = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_ef_combustion_co2, return_type = "array_units_corrected", expand_to_all_cats = True)
        arr_trns_energy_density_fuel = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_enfu_volumetric_energy_density, return_type = "array_units_corrected", expand_to_all_cats = True)

        # initialize electrical demand to pass and output emission arrays
        arr_trns_demand_electricity = 0.0
        arr_trns_demand_electricity_total = 0.0
        arr_trns_emissions_ch4 = 0.0
        arr_trns_emissions_co2 = 0.0
        arr_trns_emissions_n2o = 0.0

        # loop over fuels to calculate emissions and demand associated with each fuel
        fuels_loop = sorted(list(self.dict_trns_fuel_categories_to_fuel_variables.keys()))
        for cat_fuel in fuels_loop:

            # initialize the fuel demand
            vec_fuel_demand = 0

            # set some model variables
            dict_tfc_to_fv_cur = self.dict_trns_fuel_categories_to_fuel_variables.get(cat_fuel)
            modvar_trns_ef_ch4_cur = dict_tfc_to_fv_cur.get("ef_ch4")
            modvar_trns_ef_n2o_cur = dict_tfc_to_fv_cur.get("ef_n2o")
            modvar_trns_fuel_efficiency_cur = dict_tfc_to_fv_cur.get("fuel_efficiency")
            modvar_trns_fuel_fraction_cur = dict_tfc_to_fv_cur.get("fuel_fraction")
            modvar_trns_total_volumetric_fuel_dem_cur = dict_tfc_to_fv_cur.get("total_fuel_demand")

            # set some scalars for use in the calculations
            scalar_trns_fuel_efficiency_to_demand = self.model_attributes.get_variable_unit_conversion_factor(
                modvar_trns_fuel_efficiency_cur,
                self.modvar_trde_demand_pkm,
                "length"
            )

            # get the index and vector of co2 emission factors
            ind_enfu_cur = attr_enfu.get_key_value_index(cat_fuel)
            vec_trns_ef_by_fuel_co2_cur = arr_trns_ef_by_fuel_co2[:, ind_enfu_cur]
            vec_trns_volumetric_enerdensity_by_fuel = arr_trns_energy_density_fuel[:, ind_enfu_cur]
            # get arrays
            arr_trns_fuel_fraction_cur = dict_arrs_trns_frac_fuel.get(modvar_trns_fuel_fraction_cur)
            arr_trns_ef_ch4_cur = self.model_attributes.get_standard_variables(df_neenergy_trajectories, modvar_trns_ef_ch4_cur, return_type = "array_units_corrected", expand_to_all_cats = True) if (modvar_trns_ef_ch4_cur is not None) else 0
            arr_trns_ef_n2o_cur = self.model_attributes.get_standard_variables(df_neenergy_trajectories, modvar_trns_ef_n2o_cur, return_type = "array_units_corrected", expand_to_all_cats = True) if (modvar_trns_ef_n2o_cur is not None) else 0
            arr_trns_fuel_efficiency_cur = self.model_attributes.get_standard_variables(df_neenergy_trajectories, modvar_trns_fuel_efficiency_cur, return_type = "array_base", expand_to_all_cats = True)

            # current demand associate with the fuel (in terms of modvar_trde_demand_pkm)
            arr_trns_vehdem_cur_fuel = array_trns_total_vehicle_demand*arr_trns_fuel_fraction_cur

            if (arr_trns_fuel_efficiency_cur is not None):

                # get demand for fuel in terms of modvar_trns_fuel_efficiency_cur, then get scalars to conert to emission factor fuel volume units
                arr_trns_fueldem_cur_fuel = np.nan_to_num(arr_trns_vehdem_cur_fuel/arr_trns_fuel_efficiency_cur, neginf = 0.0, posinf = 0.0)
                arr_trns_energydem_cur_fuel = (arr_trns_fueldem_cur_fuel.transpose()*vec_trns_volumetric_enerdensity_by_fuel).transpose()
                arr_trns_energydem_cur_fuel *= self.model_attributes.get_variable_unit_conversion_factor(
                    modvar_trns_fuel_efficiency_cur,
                    self.modvar_enfu_volumetric_energy_density,
                    "volume"
                )
                # add total fuel to output variable
                vec_fuel_demand += np.sum(arr_trns_fueldem_cur_fuel, axis = 1)


                ##  CH4 EMISSIONS

                # get scalar to prepare fuel energies for the emission factor
                scalar_fuel_energy_to_ef_ch4 = self.model_attributes.get_variable_unit_conversion_factor(
                    self.modvar_enfu_volumetric_energy_density,
                    modvar_trns_ef_ch4_cur,
                    "energy"
                ) if (modvar_trns_ef_ch4_cur is not None) else 0
                arr_trns_fuel_energydem_cur_fuel_ch4 = arr_trns_energydem_cur_fuel*scalar_fuel_energy_to_ef_ch4
                arr_emissions_ch4_cur_fuel = arr_trns_ef_ch4_cur*arr_trns_fuel_energydem_cur_fuel_ch4
                arr_trns_emissions_ch4 += arr_emissions_ch4_cur_fuel


                ##  CO2 EMISSIONS

                # get scalar to prepare fuel energies for the emission factor
                scalar_fuel_energy_to_ef_co2 = self.model_attributes.get_variable_unit_conversion_factor(
                    self.modvar_enfu_volumetric_energy_density,
                    self.modvar_enfu_ef_combustion_co2,
                    "energy"
                )
                arr_trns_fuel_energydem_cur_fuel_co2 = arr_trns_energydem_cur_fuel*scalar_fuel_energy_to_ef_co2
                arr_emissions_co2_cur_fuel = (arr_trns_fuel_energydem_cur_fuel_co2.transpose()*vec_trns_ef_by_fuel_co2_cur).transpose()
                arr_trns_emissions_co2 += arr_emissions_co2_cur_fuel

                ##  N2O EMISSIONS

                # n2o scalar
                scalar_fuel_energy_to_ef_n2o = self.model_attributes.get_variable_unit_conversion_factor(
                    self.modvar_enfu_volumetric_energy_density,
                    modvar_trns_ef_n2o_cur,
                    "energy"
                ) if (modvar_trns_ef_n2o_cur is not None) else 0
                arr_trns_fuel_energydem_cur_fuel_n2o = arr_trns_energydem_cur_fuel*scalar_fuel_energy_to_ef_n2o
                arr_emissions_n2o_cur_fuel = arr_trns_ef_n2o_cur*arr_trns_fuel_energydem_cur_fuel_n2o
                arr_trns_emissions_n2o += arr_emissions_n2o_cur_fuel

            elif cat_fuel == self.enfu_fuel_electricity:

                # get scalar for energy
                scalar_electric_eff_to_distance_equiv = self.model_attributes.get_variable_unit_conversion_factor(
                    self.modvar_trns_electrical_efficiency,
                    self.modvar_trde_demand_pkm,
                    "length"
                )
                # get demand for fuel in terms of modvar_trns_fuel_efficiency_cur, then get scalars to conert to emission factor fuel volume units
                arr_trns_elect_efficiency_cur = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trns_electrical_efficiency, return_type = "array_base", expand_to_all_cats = True)
                arr_trns_elect_efficiency_cur *= scalar_electric_eff_to_distance_equiv
                arr_trns_energydem_elec = arr_trns_vehdem_cur_fuel/arr_trns_elect_efficiency_cur
                # write in terms of output units
                arr_trns_energydem_elec *= self.model_attributes.get_scalar(self.modvar_trns_electrical_efficiency, "energy")
                arr_trns_energydem_elec = np.nan_to_num(arr_trns_energydem_elec, posinf = 0, neginf = 0)

            # add total fuel volumetric fuel demand
            if modvar_trns_fuel_efficiency_cur is not None:
                vec_fuel_demand *= self.model_attributes.get_scalar(modvar_trns_fuel_efficiency_cur, "volume")
                df_out.append(
                    self.model_attributes.array_to_df(vec_fuel_demand, modvar_trns_total_volumetric_fuel_dem_cur, False, False),
                )

        # add aggregate emissions
        df_out += [
            self.model_attributes.array_to_df(arr_trns_emissions_ch4, self.modvar_trns_emissions_ch4, False),
            self.model_attributes.array_to_df(arr_trns_emissions_co2, self.modvar_trns_emissions_co2, False),
            self.model_attributes.array_to_df(arr_trns_emissions_n2o, self.modvar_trns_emissions_n2o, False),
            self.model_attributes.array_to_df(arr_trns_energydem_elec, self.modvar_tnrs_energy_demand_electricity, False, True),
            self.model_attributes.array_to_df(np.sum(arr_trns_energydem_elec, axis = 1), self.modvar_tnrs_energy_demand_electricity_agg, False)
        ]


        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")
        self.model_attributes.add_subsector_emissions_aggregates(df_out, ["Transportation"], False)

        return df_out



    ##  transportation demands
    def project_transportation_demand(self,
        df_neenergy_trajectories: pd.DataFrame,
        vec_pop: np.ndarray,
        vec_rates_gdp: np.ndarray,
        vec_rates_gdp_per_capita: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
            project_transportation_demand can be called from other sectors to simplify calculation of transportation demands and associated metrics.

            Function Arguments
            ------------------
            df_neenergy_trajectories: pd.DataFrame of input variables

            vec_pop: np.ndarray vector of population (requires len(vec_rates_gdp) == len(df_neenergy_trajectories))

            vec_rates_gdp: np.ndarray vector of gdp growth rates (v_i = growth rate from t_i to t_{i + 1}) (requires len(vec_rates_gdp) == len(df_neenergy_trajectories) - 1)

            vec_rates_gdp_per_capita: np.ndarray vector of gdp per capita growth rates (v_i = growth rate from t_i to t_{i + 1}) (requires len(vec_rates_gdp_per_capita) == len(df_neenergy_trajectories) - 1)

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
        pycat_trde = self.model_attributes.get_subsector_attribute("Transportation Demand", "pycategory_primary")
        pycat_trns = self.model_attributes.get_subsector_attribute("Transportation", "pycategory_primary")
        # attribute tables
        attr_enfu = self.model_attributes.dict_attributes[pycat_enfu]
        attr_trde = self.model_attributes.dict_attributes[pycat_trde]
        attr_trns = self.model_attributes.dict_attributes[pycat_trns]


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]


        ############################
        #    MODEL CALCULATIONS    #
        ############################

        # get the demand scalar
        array_trde_demscalar = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trde_demand_scalar, return_type = "array_base", expand_to_all_cats = True, var_bounds = (0, np.inf))
        # start with freight/megaton km demands
        array_trde_dem_init_freight = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trde_demand_initial_mtkm, return_type = "array_base", expand_to_all_cats = True)
        array_trde_elast_freight_demand_to_gdp = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trde_elasticity_mtkm_to_gdp, return_type = "array_base", expand_to_all_cats = True)
        array_trde_growth_freight_dem_by_cat = sf.project_growth_scalar_from_elasticity(vec_rates_gdp, array_trde_elast_freight_demand_to_gdp, False, "standard")
        # multiply and add to the output
        array_trde_freight_dem_by_cat = array_trde_dem_init_freight[0]*array_trde_growth_freight_dem_by_cat
        array_trde_freight_dem_by_cat *= array_trde_demscalar
        df_out.append(
            self.model_attributes.array_to_df(array_trde_freight_dem_by_cat, self.modvar_trde_demand_mtkm, False, True)
        )

        # deal with person-km
        array_trde_dem_init_passenger = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trde_demand_initial_pkm_per_capita, return_type = "array_base", expand_to_all_cats = True)
        array_trde_elast_passenger_demand_to_gdppc = self.model_attributes.get_standard_variables(df_neenergy_trajectories, self.modvar_trde_elasticity_pkm_to_gdp, return_type = "array_base", expand_to_all_cats = True)
        array_trde_growth_passenger_dem_by_cat = sf.project_growth_scalar_from_elasticity(vec_rates_gdp_per_capita, array_trde_elast_passenger_demand_to_gdppc, False, "standard")
        # project the growth in per capita, multiply by population, then add it to the output
        array_trde_passenger_dem_by_cat = array_trde_dem_init_passenger[0]*array_trde_growth_passenger_dem_by_cat
        array_trde_passenger_dem_by_cat = (array_trde_passenger_dem_by_cat.transpose()*vec_pop).transpose()
        array_trde_passenger_dem_by_cat *= array_trde_demscalar
        df_out.append(
            self.model_attributes.array_to_df(array_trde_passenger_dem_by_cat, self.modvar_trde_demand_pkm, False, True)
        )

        # build output dataframe
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")

        return df_out



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
            - df_neenergy_trajectories should have all input fields required (see Energy.required_variables for a list of variables to be defined)
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


        ##  OUTPUT INITIALIZATION

        df_out = [df_neenergy_trajectories[self.required_dimensions].copy()]



        #########################################
        #    MODEL CALCULATIONS BY SUBSECTOR    #
        #########################################

        # add industrial energy, transportation, and OESC
        df_out.append(self.project_industrial_energy(df_neenergy_trajectories, vec_gdp, dict_dims, n_projection_time_periods, projection_time_periods))
        df_out.append(self.project_transportation(df_neenergy_trajectories, vec_pop, vec_rates_gdp, vec_rates_gdp_per_capita, dict_dims, n_projection_time_periods, projection_time_periods))

        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")

        return df_out
