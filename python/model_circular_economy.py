import support_functions as sf
import data_structures as ds
import pandas as pd
import numpy as np
import time

##############################
###                        ###
###    CIRCULAR ECONOMY    ###
###                        ###
##############################

class CircularEconomy:

    def __init__(self, attributes: ds.ModelAttributes):

        self.model_attributes = attributes
        self.required_dimensions = self.get_required_dimensions()
        self.required_subsectors, self.required_base_subsectors = self.get_required_subsectors()
        self.required_variables, self.output_variables = self.get_ce_input_output_fields()

        ##  set some model fields to connect to the attribute tables

        # domestic solid waste model variables
        self.modvar_waso_boc = "BOC"
        self.modvar_waso_frac_dom_compo = "Composition Fraction Domestic Solid Waste"
        self.modvar_waso_frac_ind_compo = "Composition Fraction Industrial Solid Waste"
        self.modvar_waso_dem_energy_recycle = "Energy Demand to Recycle Waste"
        self.modvar_waso_dsw_pc_init = "Initial Per Capita Annual Domestic Solid Waste Generated"
        self.modvar_waso_dsw_elast = "Elasticity of Waste Produced to GDP/Capita"
        self.modvar_waso_dsw_elast = "Initial Per VA Solid Industrial Waste Generated"
        self.modvar_waso_edrf = "Virgin Production Energy Demand Reduction Factor"
        self.modvar_waso_frac_dsw_nr_burned = "Fraction of Non-Recycled Domestic Solid Waste Burned"
        self.modvar_waso_frac_dsw_nr_landfilled = "Fraction of Non-Recycled Domestic Solid Waste Landfilled"
        self.modvar_waso_frac_dsw_nr_open = "Fraction of Non-Recycled Domestic Solid Waste Open Dumps"
        self.modvar_waso_frac_dsw_recycle = "Fraction of Waste Recycled"
        self.modvar_waso_frac_landfill_gas_captured = "Fraction of Landfill Gas Captured at Landfills"
        self.modvar_waso_k = "K"


        # economy and general variables
        self.modvar_econ_gdp = "GDP"
        self.modvar_econ_va = "Value Added"
        self.modvar_gnrl_area = "Area of Country"
        self.modvar_gnrl_occ = "National Occupation Rate"
        self.modvar_gnrl_subpop = "Population"
        self.modvar_gnrl_pop_total = "Total Population"

        ##  MISCELLANEOUS VARIABLES

        self.time_periods, self.n_time_periods = self.model_attributes.get_time_periods()

        # TEMP:SET TO DERIVE FROM ATTRIBUTE TABLES---
        self.landfill_gas_frac_methane = 0.5


    ##  FUNCTIONS FOR MODEL ATTRIBUTE DIMENSIONS

    def check_df_fields(self, df_ce_trajectories):
        check_fields = self.required_variables
        # check for required variables
        if not set(check_fields).issubset(df_ce_trajectories.columns):
            set_missing = list(set(check_fields) - set(df_ce_trajectories.columns))
            set_missing = sf.format_print_list(set_missing)
            raise KeyError(f"Circular Economy projection cannot proceed: The fields {set_missing} are missing.")


    def get_required_subsectors(self):
        subsectors = list(sf.subset_df(self.model_attributes.dict_attributes["abbreviation_subsector"].table, {"sector": ["Circular Economy"]})["subsector"])
        subsectors_base = subsectors.copy()
        subsectors += ["Economy", "General"]
        return subsectors, subsectors_base

    def get_required_dimensions(self):
        ## TEMPORARY - derive from attributes later
        required_doa = [self.model_attributes.dim_time_period]
        return required_doa

    def get_ce_input_output_fields(self):
        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(self.required_subsectors)
        return required_vars + self.get_required_dimensions(), output_vars
