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



    ################################################################
    #    SETUP TABLE TRANSFORMATION FUNCTIONS TO FORMAT FOR SQL    #
    ################################################################

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
    def format_nemomod_reserve_margin(self,
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

        return None


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
