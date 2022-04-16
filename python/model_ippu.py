import support_functions as sf
import data_structures as ds
from model_socioeconomic import Socioeconomic
import pandas as pd
import numpy as np
import time


#########################
###                   ###
###     IPPU MODEL    ###
###                   ###
#########################

class IPPU:

    def __init__(self, attributes: ds.ModelAttributes):

        self.model_attributes = attributes
        self.required_dimensions = self.get_required_dimensions()
        self.required_subsectors, self.required_base_subsectors = self.get_required_subsectors()
        self.required_variables, self.output_variables = self.get_ippu_input_output_fields()

        ##  set some model fields to connect to the attribute tables

        # ippu model variables
        self.modvar_ippu_change_net_imports = "Change to Net Imports of Recyclable Products"
        self.modvar_ippu_ef_ch4_per_prod = ":math:\\text{CH}_4 Production Emission Factor"
        self.modvar_ippu_ef_co2_per_gdp = ":math:\\text{CO}_2 GDP Emission Factor"
        self.modvar_ippu_ef_co2_per_prod = ":math:\\text{CO}_2 Production Emission Factor"
        self.modvar_ippu_ef_hfc_per_gdp = "HFC Emission GDP Factor Placeholder"
        self.modvar_ippu_ef_hfc_per_prod = "HFC Emission Production Factor Placeholder"
        self.modvar_ippu_ef_n2o_per_gdp = ":math:\\text{N}_2\\text{O} GDP Emission Factor"
        self.modvar_ippu_ef_n2o_per_prod = ":math:\\text{N}_2\\text{O} Production Emission Factor"
        self.modvar_ippu_ef_nf3_per_prod = ":math:\\text{NF}_3 Production Emission Factor"
        self.modvar_ippu_ef_pfc_per_gdp = "PFC Emission GDP Factor Placeholder"
        self.modvar_ippu_ef_pfc_per_prod = "PFC Emission Production Factor Placeholder"
        self.modvar_ippu_ef_sf6_per_gdp = ":math:\\text{SF}_6 GDP Emission Factor"
        self.modvar_ippu_ef_sf6_per_prod = ":math:\\text{SF}_6 Production Emission Factor"
        self.modvar_ippu_elast_ind_prod_to_gdp = "Elasticity of Industrial Production to GDP"
        self.modvar_ippu_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Industry"
        self.modvar_ippu_emissions_co2 = ":math:\\text{CO}_2 Emissions from Industry"
        self.modvar_ippu_emissions_hfc = "HFC Emissions Placeholder"
        self.modvar_ippu_emissions_n2o = ":math:\\text{N}_2\\text{O} Emissions from Industry"
        self.modvar_ippu_emissions_pfc = "PFC Emissions Placeholder"
        self.modvar_ippu_emissions_nf3 = ":math:\\text{NF}_3 Emissions from Industry"
        self.modvar_ippu_emissions_sf6 = ":math:\\text{SF}_6 Emissions from Industry"
        self.modvar_ippu_prod_qty_init = "Initial Industrial Production"
        self.modvar_ippu_wwf_cod = "COD Wastewater Factor"
        self.modvar_ippu_wwf_vol = "Wastewater Production Factor"
        self.modvar_ippu_qty_total_production = "Industrial Production"

        # variables from other sectors
        self.modvar_waso_waste_total_recycled = "Total Waste Recycled"

        # add other model classes
        self.model_socioeconomic = Socioeconomic(self.model_attributes)

        # optional integration variables (uses calls to other model classes)
        self.integration_variables = self.set_integrated_variables()

        ##  MISCELLANEOUS VARIABLES
        self.time_periods, self.n_time_periods = self.model_attributes.get_time_periods()



    ##  FUNCTIONS FOR MODEL ATTRIBUTE DIMENSIONS

    def check_df_fields(self, df_ippu_trajectories):
        check_fields = self.required_variables
        # check for required variables
        if not set(check_fields).issubset(df_ippu_trajectories.columns):
            set_missing = list(set(check_fields) - set(df_ippu_trajectories.columns))
            set_missing = sf.format_print_list(set_missing)
            raise KeyError(f"IPPU projection cannot proceed: The fields {set_missing} are missing.")

    def get_required_subsectors(self):
        subsectors = list(sf.subset_df(self.model_attributes.dict_attributes["abbreviation_subsector"].table, {"sector": ["IPPU"]})["subsector"])
        subsectors_base = subsectors.copy()
        subsectors += ["Economy", "General"]
        return subsectors, subsectors_base

    def get_required_dimensions(self):
        ## TEMPORARY - derive from attributes later
        required_doa = [self.model_attributes.dim_time_period]
        return required_doa

    def get_ippu_input_output_fields(self):
        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(self.required_subsectors)
        return required_vars + self.get_required_dimensions(), output_vars

    def set_integrated_variables(self):
        list_vars_required_for_integration = [
            self.modvar_waso_waste_total_recycled
        ]

        return list_vars_required_for_integration


    ######################################
    #    SUBSECTOR SPECIFIC FUNCTIONS    #
    ######################################


    # project industrial production—broken out so that other sectors can call it
    def project_industrial_production(self,
        df_ippu_trajectories: pd.DataFrame,
        vec_rates_gdp: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> np.ndarray:
        """
            project_industrial_production can be called from other sectors to simplify calculation of industrial production. If any of dict_dims, n_projection_time_periods, or projection_time_periods are unspecified (expected if ran outside of IPPU.project()), self.model_attributes.check_projection_input_df wil be run

            dict_dims: dict of dimensions (returned from check_projection_input_df). Default is None.

            n_projection_time_periods: integer giving number of time periods (returned from check_projection_input_df). Default is None.

            projection_time_periods: list of time periods (returned from check_projection_input_df). Default is None.
        """
        # allows production to be run outside of the project method
        if type(None) in set([type(x) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            dict_dims, df_ippu_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_ippu_trajectories, True, True, True)

        # get initial production and apply elasticities to gdp to calculate growth in production
        array_ippu_prod_init_by_cat = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_prod_qty_init, False, return_type = "array_base", var_bounds = (0, np.inf), expand_to_all_cats = True)
        array_ippu_elasticity_prod_to_gdp = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_elast_ind_prod_to_gdp, False, return_type = "array_base", expand_to_all_cats = True)
        array_ippu_ind_growth = sf.project_growth_scalar_from_elasticity(vec_rates_gdp, array_ippu_elasticity_prod_to_gdp, False, "standard")
        array_ippu_ind_prod = array_ippu_prod_init_by_cat[0]*array_ippu_ind_growth

        return array_ippu_ind_prod


    # project method for IPPU
    def project(self, df_ippu_trajectories: pd.DataFrame) -> pd.DataFrame:

        # make sure socioeconomic variables are added and
        df_ippu_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_ippu_trajectories)
        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_ippu_trajectories)
        dict_dims, df_ippu_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_ippu_trajectories, True, True, True)

        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_gnrl = self.model_attributes.get_subsector_attribute("General", "pycategory_primary")
        pycat_ippu = self.model_attributes.get_subsector_attribute("IPPU", "pycategory_primary")
        pycat_waso = self.model_attributes.get_subsector_attribute("Solid Waste", "pycategory_primary")
        # attribute tables
        attr_gnrl = self.model_attributes.dict_attributes[pycat_gnrl]
        attr_ippu = self.model_attributes.dict_attributes[pycat_ippu]
        attr_waso = self.model_attributes.dict_attributes[pycat_waso]


        ##  ECON/GNRL VECTOR AND ARRAY INITIALIZATION

        # get some vectors
        vec_gdp = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.model_socioeconomic.modvar_econ_gdp, False, return_type = "array_base")
        vec_pop = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.model_socioeconomic.modvar_gnrl_pop_total, False, return_type = "array_base")
        array_pop = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.model_socioeconomic.modvar_gnrl_subpop, False, return_type = "array_base")
        vec_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_gdp_per_capita"])
        vec_rates_gdp = np.array(df_se_internal_shared_variables["vec_rates_gdp"].dropna())
        vec_rates_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_rates_gdp_per_capita"].dropna())


        ##  OUTPUT INITIALIZATION
        df_out = [df_ippu_trajectories[self.required_dimensions].copy()]


        ######################################################
        #    INDUSTRIAL PRODUCTION + RECYCLING ADJUSTMENT    #
        ######################################################

        # initialize production + initialize change to net imports as 0 (reduce categories later)
        array_ippu_production = self.project_industrial_production(df_ippu_trajectories, vec_rates_gdp, dict_dims, n_projection_time_periods, projection_time_periods)
        array_ippu_change_net_imports = np.zeros((array_ippu_production.shape[0], attr_ippu.n_key_values))


        ##  PERFORM THE RECYCLING ADJUSTMENT (if recycling data are provided from the waste model)

        array_ippu_recycled = self.model_attributes.get_optional_or_integrated_standard_variable(df_ippu_trajectories, self.modvar_waso_waste_total_recycled, None, True, "array_base")
        if type(array_ippu_recycled) != type(None):
            # if recycling totals are passed from the waste model, convert to ippu categories
            cats_waso_recycle = sa.model_attributes.get_variable_categories(self.modvar_waso_waste_total_recycled)
            dict_repl = attr_waso.field_maps[f"{pycat_waso}_to_{pycat_ippu}"]
            cats_ippu_recycle = [ds.clean_schema(dict_repl[x]) for x in cats_waso_recycle]
            array_ippu_recycled_waste = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(
                array_ippu_recycled[1],
                None,
                output_cats = cats_ippu_recycle,
                output_subsec = "IPPU"
            )
            # units correction to ensure consistency from waso -> ippu
            factor_ippu_waso_recycle_to_ippu_recycle = self.model_attributes.get_mass_equivalent(
                self.model_attributes.get_variable_characteristic(self.modvar_waso_waste_total_recycled, "$UNIT-MASS$"),
                self.model_attributes.get_variable_characteristic(self.modvar_ippu_prod_qty_init, "$UNIT-MASS$")
            )
            array_ippu_recycled_waste *= factor_ippu_waso_recycle_to_ippu_recycle
            array_ippu_production += array_ippu_recycled_waste
            # next, check for industrial categories whose production is affected by recycling, then adjust downwards
            vec_ippu_cats_to_adjust_from_recycling = [ds.clean_schema(x) for x in sa.model_attributes.get_ordered_category_attribute("IPPU", "target_recycling_cat_industry_to_adjust")]
            w = [i for i in range(len(vec_ippu_cats_to_adjust_from_recycling)) if (vec_ippu_cats_to_adjust_from_recycling[i] != "none") and (vec_ippu_cats_to_adjust_from_recycling[i] in attr_ippu.key_values)]
            if len(w) > 0:
                array_ippu_recycled_waste_adj = array_ippu_recycled_waste[:, w].copy()
                array_ippu_recycled_waste_adj = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(
                    array_ippu_recycled_waste_adj,
                    None,
                    output_cats = np.array(vec_ippu_cats_to_adjust_from_recycling)[w],
                    output_subsec = "IPPU"
                )
                # inititialize production, then get change to net imports (anything negative) and reduce virgin production accordingly
                array_ippu_production = array_ippu_production - array_ippu_recycled_waste_adj
                array_ippu_change_net_imports = sf.vec_bounds(array_ippu_production, (-np.inf, 0))
                array_ippu_production = sf.vec_bounds(array_ippu_production, (0, np. inf))

        # get production in terms of output variable (should be 1, and add net imports and production to output dataframe)
        array_ippu_production *= self.model_attributes.get_mass_equivalent(
            self.model_attributes.get_variable_characteristic(self.modvar_ippu_prod_qty_init, "$UNIT-MASS$"),
            self.model_attributes.get_variable_characteristic(self.modvar_ippu_qty_total_production, "$UNIT-MASS$")
        )
        df_out += [
            self.model_attributes.array_to_df(array_ippu_change_net_imports, self.modvar_ippu_change_net_imports, False, True),
            self.model_attributes.array_to_df(array_ippu_production, self.modvar_ippu_qty_total_production, False, True)
        ]


        ##  USE EMISSION FACTORS TO ESTIMATE EMISSIONS FROM PROCESSES AND PRODUCT USE - LOOP OVER A DICTIONARY GIVEN THE NUMBER OF FACTORS

        # dictionary variables mapping emission variable to factor tuples (gdp, production). No factor gives ""
        dict_ippu_simple_efs = {
            self.modvar_ippu_emissions_ch4: ("", self.modvar_ippu_ef_ch4_per_prod),
            self.modvar_ippu_emissions_co2: (self.modvar_ippu_ef_co2_per_gdp, self.modvar_ippu_ef_co2_per_prod),
            self.modvar_ippu_emissions_hfc: (self.modvar_ippu_ef_hfc_per_gdp, self.modvar_ippu_ef_hfc_per_prod),
            self.modvar_ippu_emissions_n2o: (self.modvar_ippu_ef_n2o_per_gdp, self.modvar_ippu_ef_n2o_per_prod),
            self.modvar_ippu_emissions_nf3: ("", self.modvar_ippu_ef_nf3_per_prod),
            self.modvar_ippu_emissions_pfc: (self.modvar_ippu_ef_pfc_per_gdp, self.modvar_ippu_ef_pfc_per_prod),
            self.modvar_ippu_emissions_sf6: (self.modvar_ippu_ef_sf6_per_gdp, self.modvar_ippu_ef_sf6_per_prod)
        }

        # process is identical across emission factors -- sum gdp-driven and production-driven factors
        for modvar in dict_ippu_simple_efs.keys():
            # get variables and initialize total emissions
            modvar_ef_gdp = dict_ippu_simple_efs[modvar][0]
            modvar_ef_prod = dict_ippu_simple_efs[modvar][1]
            array_ippu_emission = np.zeros((len(df_ippu_trajectories), attr_ippu.n_key_values))
            # check if there is a gdp driven factor
            if modvar_ef_gdp != "":
                array_ippu_emission_cur = self.model_attributes.get_standard_variables(df_ippu_trajectories, modvar_ef_gdp, False, return_type = "array_units_corrected", expand_to_all_cats = True)
                array_ippu_emission += (array_ippu_emission_cur.transpose()*vec_gdp).transpose()
            # check if there is a production driven factor
            if modvar_ef_prod != "":
                scalar_ippu_mass = self.model_attributes.get_mass_equivalent(
                    self.model_attributes.get_variable_characteristic(self.modvar_ippu_qty_total_production, "$UNIT-MASS$"),
                    self.model_attributes.get_variable_characteristic(modvar_ef_prod, "$UNIT-MASS$")
                )
                array_ippu_emission_cur = self.model_attributes.get_standard_variables(df_ippu_trajectories, modvar_ef_prod, False, return_type = "array_units_corrected", expand_to_all_cats = True)
                array_ippu_emission += array_ippu_emission_cur*array_ippu_production
            # add to output dataframe
            df_out += [
                self.model_attributes.array_to_df(array_ippu_emission, modvar, False, True)
            ]


        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")
        self.model_attributes.add_subsector_emissions_aggregates(df_out, self.required_base_subsectors, False)

        return df_out
