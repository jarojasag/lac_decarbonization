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
