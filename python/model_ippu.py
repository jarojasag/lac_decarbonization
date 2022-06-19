import support_functions as sf
import data_structures as ds
from model_socioeconomic import Socioeconomic
import pandas as pd
import numpy as np
import time
import warnings


#########################
###                   ###
###     IPPU MODEL    ###
###                   ###
#########################

class IPPU:

    def __init__(self, attributes: ds.ModelAttributes):

        # some subector reference variables
        self.subsec_name_econ = "Economy"
        self.subsec_name_gnrl = "General"
        self.subsec_name_ippu = "IPPU"
        self.subsec_name_waso = "Solid Waste"

        # initialzie dynamic variables
        self.model_attributes = attributes
        self.required_dimensions = self.get_required_dimensions()
        self.required_subsectors, self.required_base_subsectors = self.get_required_subsectors()
        self.required_variables, self.output_variables = self.get_ippu_input_output_fields()


        ##  SET MODEL VARIABLES

        # ippu model variables
        self.modvar_ippu_change_net_imports = "Change to Net Imports of Recyclable Products"
        self.modvar_ippu_clinker_fraction_cement = "Clinker Fraction of Cement"
        self.modvar_ippu_ef_ch4_per_prod_process = ":math:\\text{CH}_4 Production Process Emission Factor"
        self.modvar_ippu_ef_co2_per_prod_process = ":math:\\text{CO}_2 Production Process Emission Factor"
        self.modvar_ippu_ef_co2_per_prod_produse = ":math:\\text{CO}_2 Product Use Emission Factor"
        self.modvar_ippu_ef_co2_per_prod_process_clinker = ":math:\\text{CO}_2 Clinker Production Process Emission Factor"
        self.modvar_ippu_ef_n2o_per_gdp_process = ":math:\\text{N}_2\\text{O} GDP Production Process Emission Factor"
        self.modvar_ippu_ef_n2o_per_prod_process = ":math:\\text{N}_2\\text{O} Production Process Emission Factor"
        self.modvar_ippu_ef_nf3_per_prod_process = ":math:\\text{NF}_3 Production Process Emission Factor"
        self.modvar_ippu_ef_octafluoro_per_prod_process = "Octafluorooxolane Production Process Emission Factor"
        self.modvar_ippu_ef_sf6_per_gdp_process = ":math:\\text{SF}_6 GDP Production Process Emission Factor"
        self.modvar_ippu_ef_sf6_per_prod_process = ":math:\\text{SF}_6 Production Process Emission Factor"
        self.modvar_ippu_ef_hfc23_per_gdp_produse = "HFC-23 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc23_per_prod_process = "HFC-23 Production Process Emission Factor"
        self.modvar_ippu_ef_hfc32_per_gdp_produse = "HFC-32 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc32_per_prod_process = "HFC-32 Production Process Emission Factor"
        self.modvar_ippu_ef_hfc125_per_gdp_produse = "HFC-125 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc134a_per_gdp_produse = "HFC-134a GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc143a_per_gdp_produse = "HFC-143a GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc152a_per_gdp_produse = "HFC-152a GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc227ea_per_gdp_produse = "HFC-227ea GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc236fa_per_gdp_produse = "HFC-236fa GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc245fa_per_gdp_produse = "HFC-245fa GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc365mfc_per_gdp_produse = "HFC-365mfc GDP Product Use Emission Factor"
        self.modvar_ippu_ef_hfc4310mee_per_gdp_produse = "HFC-43-10mee GDP Product Use Emission Factor"
        self.modvar_ippu_ef_pfc14_per_gdp_produse = "PFC-14 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_pfc14_per_prod_process = "PFC-14 Production Process Emission Factor"
        self.modvar_ippu_ef_pfc116_per_gdp_produse = "PFC-116 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_pfc116_per_prod_process = "PFC-116 Production Process Emission Factor"
        self.modvar_ippu_ef_pfc218_per_prod_process = "PFC-218 Production Process Emission Factor"
        self.modvar_ippu_ef_pfc3110_per_gdp_produse = "PFC-31-10 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_pfc5114_per_gdp_produse = "PFC-51-14 GDP Product Use Emission Factor"
        self.modvar_ippu_ef_pfc1114_per_prod_process = "PFC-1114 Production Process Emission Factor"
        self.modvar_ippu_ef_pfcc318_per_prod_process = "PFC-C-318 Production Process Emission Factor"
        self.modvar_ippu_ef_pfcc1418_per_prod_process = "PFC-C-1418 Production Process Emission Factor"
        self.modvar_ippu_elast_ind_prod_to_gdp = "Elasticity of Industrial Production to GDP"
        self.modvar_ippu_elast_produserate_to_gdppc = "Elasticity of Product Use Rate to GDP per Capita"
        self.modvar_ippu_emissions_other_nonenergy_co2 = "Initial Other Non-Energy :math:\\text{CO}_2 Emissions"
        self.modvar_ippu_emissions_process_ch4 = ":math:\\text{CH}_4 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_process_co2 = ":math:\\text{CO}_2 Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_co2 = ":math:\\text{CO}_2 Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_hfc = "HFC Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_hfc = "HFC Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_n2o = ":math:\\text{N}_2\\text{O} Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_process_other_fcs = "Other Fluorinated Compound Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_process_pfc = "PFC Emissions from Industrial Production Processes"
        self.modvar_ippu_emissions_produse_pfc = "PFC Emissions from Industrial Product Use"
        self.modvar_ippu_emissions_process_sf6 = ":math:\\text{SF}_6 Emissions from Industrial Production Processes"
        self.modvar_ippu_max_recycled_material_ratio = "Maximum Recycled Material Ratio in Virgin Process"
        self.modvar_ippu_net_imports_clinker = "Net Imports of Cement Clinker"
        self.modvar_ippu_prod_qty_init = "Initial Industrial Production"
        self.modvar_ippu_scalar_production = "Industrial Production Scalar"
        self.modvar_ippu_useinit_nonenergy_fuel = "Initial Non-Energy Fuel Use"
        self.modvar_ippu_wwf_cod = "COD Wastewater Factor"
        self.modvar_ippu_wwf_vol = "Wastewater Production Factor"
        self.modvar_ippu_qty_total_production = "Industrial Production"
        self.modvar_ippu_qty_recycled_used_in_production = "Recycled Material Used in Industrial Production"

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
        subsectors = self.model_attributes.get_sector_subsectors(self.subsec_name_ippu)
        subsectors_base = subsectors.copy()
        subsectors += [self.subsec_name_econ, self.subsec_name_gnrl]
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


    ################################
    #    BASIC SHARED FUNCTIONS    #
    ################################

    #
    def calculate_emissions_by_gdp_and_production(self,
        df_ippu_trajectories: pd.DataFrame,
        array_production: np.ndarray,
        vec_gdp: np.ndarray,
        dict_base_emissions: dict,
        dict_simple_efs: dict,
        modvar_prod_mass: str = None
    ) -> list:
        """
            Calculate emissions driven by GDP/Production and different factors. Takes a production array (to drive production-based emissions), a gdp vector, and dictionaries that contain (a) output variables as keys and (b) lists of input gdp and/or production variables.


            Function Arguments
            ------------------
            df_ippu_trajectories: pd.DataFrame with all required input variable trajectories.

            array_production: an array of production by industrial category

            vec_gdp: vector of gdp

            dict_simple_efs: dict of the form {modvar_emission_out: ([modvar_factor_gdp_1, ...], [modvar_factor_production_1, ... ])}. Allows for multiple gasses to be summed over.

            modvar_prod_mass: variable with mass of production denoted in array_production; used to match emission factors


            Notes
            -----

        """

        # get the attribute table and initialize output
        attr_ippu = self.model_attributes.get_attribute_table(self.subsec_name_ippu)
        modvar_prod_mass = self.modvar_ippu_qty_total_production if (modvar_prod_mass is None) else modvar_prod_mass
        df_out = []

        # process is identical across emission factors -- sum gdp-driven and production-driven factors
        for modvar in dict_simple_efs.keys():
            # get variables and initialize total emissions
            all_modvar_ef_gdp = dict_simple_efs[modvar][0]
            all_modvar_ef_prod = dict_simple_efs[modvar][1]
            array_emission = np.zeros((len(df_ippu_trajectories), attr_ippu.n_key_values))

            # check if there are gdp driven factors
            if isinstance(vec_gdp, np.ndarray):
                for modvar_ef_gdp in all_modvar_ef_gdp:
                    array_emission_cur = self.model_attributes.get_standard_variables(df_ippu_trajectories, modvar_ef_gdp, False, return_type = "array_units_corrected", expand_to_all_cats = True)
                    if vec_gdp.shape == array_emission_cur.shape:
                        array_emission += array_emission_cur*vec_gdp
                    else:
                        array_emission += (array_emission_cur.transpose()*vec_gdp).transpose()

            # check if there is a production driven factor
            if isinstance(array_production, np.ndarray):
                for modvar_ef_prod in all_modvar_ef_prod:
                    scalar_ippu_mass = self.model_attributes.get_variable_unit_conversion_factor(
                        modvar_ef_prod,
                        modvar_prod_mass,
                        "mass"
                    )
                    array_emission_cur = self.model_attributes.get_standard_variables(df_ippu_trajectories, modvar_ef_prod, False, return_type = "array_units_corrected", expand_to_all_cats = True)
                    array_emission += array_emission_cur*array_production/scalar_ippu_mass

            # add any baseline emissions from elsewhere
            array_emission += dict_base_emissions.get(modvar, 0)
            subsec = self.model_attributes.get_variable_subsector(modvar, throw_error_q = False)

            if subsec is not None:
                # add to output dataframe if it's a valid model variable
                df_out += [
                    self.model_attributes.array_to_df(array_emission, modvar, False, True)
                ]

        return df_out



    ######################################
    #    SUBSECTOR SPECIFIC FUNCTIONS    #
    ######################################


    # project industrial production—broken out so that other sectors can call it
    def project_industrial_production(self,
        df_ippu_trajectories: pd.DataFrame,
        vec_rates_gdp: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None,
        modvar_elast_ind_prod_to_gdp: str = None,
        modvar_prod_qty_init: str = None,
        modvar_scalar_prod: str = None
    ) -> np.ndarray:
        """
            project_industrial_production() can be called from other sectors to simplify calculation of industrial production.

            Function Arguments
            ------------------
            df_ippu_trajectories: pd.DataFrame of input variable trajectories.

            vec_rates_gdp: vector of rates of change to gdp (length = len(df_ippu_trajectories) - 1)

            dict_dims: dict of dimensions (returned from check_projection_input_df). Default is None.

            n_projection_time_periods: int giving number of time periods (returned from check_projection_input_df). Default is None.

            projection_time_periods: list of time periods (returned from check_projection_input_df). Default is None.

            modvar_prod_qty_init: model variable giving initial production quantity

            modvar_elast_ind_prod_to_gdp: model variable giving elasticity of production to gdp


            Notes
            -----
            - If any of dict_dims, n_projection_time_periods, or projection_time_periods are unspecified (expected if ran outside of IPPU.project()), self.model_attributes.check_projection_input_df wil be run
        """
        # allows production to be run outside of the project method
        if type(None) in set([type(x) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            dict_dims, df_ippu_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_ippu_trajectories, True, True, True)

        # set defaults
        modvar_elast_ind_prod_to_gdp = self.modvar_ippu_elast_ind_prod_to_gdp if (modvar_elast_ind_prod_to_gdp is None) else modvar_elast_ind_prod_to_gdp
        modvar_prod_qty_init = self.modvar_ippu_prod_qty_init if (modvar_prod_qty_init is None) else modvar_prod_qty_init
        modvar_scalar_prod = self.modvar_ippu_scalar_production if (modvar_scalar_prod is None) else modvar_scalar_prod

        # get initial production and apply elasticities to gdp to calculate growth in production
        array_ippu_prod_init_by_cat = self.model_attributes.get_standard_variables(df_ippu_trajectories, modvar_prod_qty_init, False, return_type = "array_base", var_bounds = (0, np.inf), expand_to_all_cats = True)
        array_ippu_elasticity_prod_to_gdp = self.model_attributes.get_standard_variables(df_ippu_trajectories, modvar_elast_ind_prod_to_gdp, False, return_type = "array_base", expand_to_all_cats = True)
        array_ippu_ind_growth = sf.project_growth_scalar_from_elasticity(vec_rates_gdp, array_ippu_elasticity_prod_to_gdp, False, "standard")
        array_ippu_ind_prod = array_ippu_prod_init_by_cat[0]*array_ippu_ind_growth
        # set exogenous scaling of production
        array_prod_scalar = self.model_attributes.get_standard_variables(df_ippu_trajectories, modvar_scalar_prod, False, return_type = "array_base", var_bounds = (0, np.inf), expand_to_all_cats = True)
        array_ippu_ind_prod *= array_prod_scalar

        return array_ippu_ind_prod


    ##  project production and adjust recycling
    def get_production_with_recycling_adjustment(self,
        df_ippu_trajectories: pd.DataFrame,
        vec_rates_gdp: np.ndarray,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None,
        modvar_change_net_imports: str = None,
        modvar_elast_ind_prod_to_gdp: str = None,
        modvar_max_recycled_material_ratio: str = None,
        modvar_prod_qty_init: str = None,
        modvar_qty_recycled_used_in_production: str = None,
        modvar_qty_total_production: str = None,
        modvar_waste_total_recycled: str = None
    ) -> tuple:
        """
            get_production_with_recycling_adjustment() can be called to retrieve production and perform the adjustment to virgin production due to recycling changes from the CircularEconomy model.

            Function Arguments
            ------------------
            df_ippu_trajectories: pd.DataFrame of input variable trajectories.

            vec_rates_gdp: vector of rates of change of gdp. Entry at index t is the change from time t-1 to t (length = len(df_ippu_trajectories) - 1)

            dict_dims: dict of dimensions (returned from check_projection_input_df). Default is None.

            n_projection_time_periods: int giving number of time periods (returned from check_projection_input_df). Default is None.

            projection_time_periods: list of time periods (returned from check_projection_input_df). Default is None.


            Keyword Arguments (variables)
            -----------------------------
            modvar_change_net_imports: model variable denoting the change to net imports

            modvar_prod_qty_init: model variable denoting the initial production quantity

            modvar_elast_ind_prod_to_gdp: model variable denoting the elasticity of production to gdp

            modvar_max_recycled_material_ratio: model variable denoting the maximum fraction of virgin production that can be replaced by recylables (e.g., cullet in glass production)

            modvar_qty_total_production: model variable denoting total industrial production

            modvar_waste_total_recycled: model variable denoted the total waste recycled (from CircularEconomy)

            Notes
            -----
            - If any of dict_dims, n_projection_time_periods, or projection_time_periods are unspecified (expected if ran outside of IPPU.project()), self.model_attributes.check_projection_input_df wil be run
        """

        # set defaults
        modvar_change_net_imports = self.modvar_ippu_change_net_imports if (modvar_change_net_imports is None) else modvar_change_net_imports
        modvar_elast_ind_prod_to_gdp = self.modvar_ippu_elast_ind_prod_to_gdp if (modvar_elast_ind_prod_to_gdp is None) else modvar_elast_ind_prod_to_gdp
        modvar_max_recycled_material_ratio = self.modvar_ippu_max_recycled_material_ratio if (modvar_max_recycled_material_ratio is None) else modvar_max_recycled_material_ratio
        modvar_prod_qty_init = self.modvar_ippu_prod_qty_init if (modvar_prod_qty_init is None) else modvar_prod_qty_init
        modvar_qty_recycled_used_in_production = self.modvar_ippu_qty_recycled_used_in_production if (modvar_qty_recycled_used_in_production is None) else modvar_qty_recycled_used_in_production
        modvar_qty_total_production = self.modvar_ippu_qty_total_production if (modvar_qty_total_production is None) else modvar_qty_total_production
        modvar_waste_total_recycled = self.modvar_waso_waste_total_recycled if (modvar_waste_total_recycled is None) else modvar_waste_total_recycled

        # allows production to be run outside of the project method
        if type(None) in set([type(x) for x in [dict_dims, n_projection_time_periods, projection_time_periods]]):
            dict_dims, df_ippu_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_ippu_trajectories, True, True, True)

        # get some attribute info
        pycat_ippu = self.model_attributes.get_subsector_attribute(self.subsec_name_ippu, "pycategory_primary")
        pycat_waso = self.model_attributes.get_subsector_attribute(self.subsec_name_waso, "pycategory_primary")
        attr_ippu = self.model_attributes.dict_attributes[pycat_ippu]
        attr_waso = self.model_attributes.dict_attributes[pycat_waso]

        # get recycling
        array_ippu_recycled = self.model_attributes.get_optional_or_integrated_standard_variable(
            df_ippu_trajectories,
            modvar_waste_total_recycled,
            None,
            True,
            "array_base"
        )

        # initialize production + initialize change to net imports as 0 (reduce categories later)
        array_ippu_production = self.project_industrial_production(
            df_ippu_trajectories,
            vec_rates_gdp,
            dict_dims,
            n_projection_time_periods,
            projection_time_periods,
            modvar_elast_ind_prod_to_gdp,
            modvar_prod_qty_init
        )
        array_ippu_change_net_imports = np.zeros(array_ippu_production.shape)

        # perform adjustments to production if recycling is denoted
        if array_ippu_recycled is not None:
            # if recycling totals are passed from the waste model, convert to ippu categories
            cats_waso_recycle = self.model_attributes.get_variable_categories(modvar_waste_total_recycled)
            dict_repl = attr_waso.field_maps[f"{pycat_waso}_to_{pycat_ippu}"]
            cats_ippu_recycle = [ds.clean_schema(dict_repl[x]) for x in cats_waso_recycle]
            array_ippu_recycled_waste = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(
                array_ippu_recycled[1],
                None,
                output_cats = cats_ippu_recycle,
                output_subsec = self.subsec_name_ippu
            )
            # units correction to ensure consistency from waso -> ippu
            factor_ippu_waso_recycle_to_ippu_recycle = self.model_attributes.get_variable_unit_conversion_factor(
                modvar_waste_total_recycled,
                modvar_prod_qty_init,
                "mass"
            )
            array_ippu_recycled_waste *= factor_ippu_waso_recycle_to_ippu_recycle
            array_ippu_production += array_ippu_recycled_waste

            # next, check for industrial categories whose production is affected by recycling, then adjust downwards
            cats_ippu_to_recycle_ordered = self.model_attributes.get_ordered_category_attribute(self.subsec_name_ippu, "target_cat_industry_to_adjust_with_recycling")
            vec_ippu_cats_to_adjust_from_recycling = [ds.clean_schema(x) for x in cats_ippu_to_recycle_ordered]

            # get indexes of of valid categories specified for recycling adjustments
            w = [i for i in range(len(vec_ippu_cats_to_adjust_from_recycling)) if (vec_ippu_cats_to_adjust_from_recycling[i] != "none") and (vec_ippu_cats_to_adjust_from_recycling[i] in attr_ippu.key_values)]
            if len(w) > 0:
                # maximum proportion of virgin production (e.g., fraction of glass that is cullet) that can be replaced by recycled materials--if not specifed, default to 1
                array_ippu_maxiumum_recycling_ratio = self.model_attributes.get_standard_variables(
                    df_ippu_trajectories,
                    self.modvar_ippu_max_recycled_material_ratio,
                    False,
                    return_type = "array_base",
                    var_bounds = (0, 1),
                    expand_to_all_cats = True,
                    all_cats_missing_val = 1.0
                )
                array_ippu_recycled_waste_adj = array_ippu_recycled_waste[:, w].copy()
                array_ippu_recycled_waste_adj = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(
                    array_ippu_recycled_waste_adj,
                    None,
                    output_cats = np.array(vec_ippu_cats_to_adjust_from_recycling)[w],
                    output_subsec = self.subsec_name_ippu
                )
                # inititialize production, then get change to net imports (anything negative) and reduce virgin production accordingly
                array_ippu_production_base = array_ippu_production*(1 - array_ippu_maxiumum_recycling_ratio)
                array_ippu_production = array_ippu_production*array_ippu_maxiumum_recycling_ratio - array_ippu_recycled_waste_adj
                # array of changes to net imports has to be mapped back to the original recycling categories
                array_ippu_change_net_imports = sf.vec_bounds(array_ippu_production, (-np.inf, 0))
                array_ippu_change_net_imports = self.model_attributes.swap_array_categories(
                    array_ippu_change_net_imports,
                    np.array(vec_ippu_cats_to_adjust_from_recycling)[w],
                    np.array(attr_ippu.key_values)[w],
                    self.subsec_name_ippu
                )
                array_ippu_production = sf.vec_bounds(array_ippu_production, (0, np.inf)) + array_ippu_production_base
                array_ippu_production += array_ippu_change_net_imports


        # ensure net imports are in the proper mass units
        array_ippu_change_net_imports *= self.model_attributes.get_variable_unit_conversion_factor(
            modvar_prod_qty_init,
            modvar_change_net_imports,
            "mass"
        )
        # get production in terms of output variable (should be 1, and add net imports and production to output dataframe)
        array_ippu_production *= self.model_attributes.get_variable_unit_conversion_factor(
            modvar_prod_qty_init,
            modvar_qty_total_production,
            "mass"
        )
        df_out = [
            self.model_attributes.array_to_df(array_ippu_change_net_imports, modvar_change_net_imports, False, True),
            self.model_attributes.array_to_df(array_ippu_production, modvar_qty_total_production, False, True),
            self.model_attributes.array_to_df(array_ippu_production, modvar_qty_recycled_used_in_production, False, True)
        ]

        return array_ippu_production, df_out




    # project method for IPPU
    def project(self, df_ippu_trajectories: pd.DataFrame) -> pd.DataFrame:

        """
            project() takes a data frame of input variables (ordered by time series) and returns a data frame of output variables (model projections for industrial processes and product use--excludes industrial energy (see Energy class)) the same order.

            Function Arguments
            ------------------
            df_ippu_trajectories: pd.DataFrame with all required input variable trajectories.


            Notes
            -----
            - The .project() method is designed to be parallelized or called from command line via __main__ in run_sector_models.py.
            - df_ippu_trajectories should have all input fields required (see IPPU.required_variables for a list of variables to be defined).  The model will not run if any required variables are missing, but errors will detail which fields are missing.
            - the df_ippu_trajectories.project method will run on valid time periods from 1 .. k, where k <= n (n is the number of time periods). By default, it drops invalid time periods. If there are missing time_periods between the first and maximum, data are interpolated.
        """

        # make sure socioeconomic variables are added and
        df_ippu_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_ippu_trajectories)
        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_ippu_trajectories)
        dict_dims, df_ippu_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_ippu_trajectories, True, True, True)

        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_gnrl = self.model_attributes.get_subsector_attribute(self.subsec_name_gnrl, "pycategory_primary")
        pycat_ippu = self.model_attributes.get_subsector_attribute(self.subsec_name_ippu, "pycategory_primary")
        pycat_waso = self.model_attributes.get_subsector_attribute(self.subsec_name_waso, "pycategory_primary")
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

        ##  PERFORM THE RECYCLING ADJUSTMENT (if recycling data are provided from the waste model)
        array_ippu_production = self.get_production_with_recycling_adjustment(df_ippu_trajectories, vec_rates_gdp)
        df_out += array_ippu_production[1]
        array_ippu_production = array_ippu_production[0]


        ############################
        #    PRODUCTION PROCESS    #
        ############################

        ##  GET CEMENT CLINKER EMISSIONS BEFORE GENERALIZED APPROACH

        scalar_ippu_mass_clinker = self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_ippu_ef_co2_per_prod_process_clinker,
            self.modvar_ippu_qty_total_production,
            "mass"
        )
        array_ippu_emissions_clinker = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_ef_co2_per_prod_process_clinker, False, return_type = "array_units_corrected", expand_to_all_cats = True)/scalar_ippu_mass_clinker
        # get net imports and convert to units of production
        array_ippu_net_imports_clinker = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_net_imports_clinker
        , False, return_type = "array_base", expand_to_all_cats = True)
        array_ippu_net_imports_clinker *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_ippu_net_imports_clinker,
            self.modvar_ippu_qty_total_production,
            "mass"
        )
        # get production of clinker, remove net imports (and cap reduction to 0), and calculate emissions
        array_ippu_production_clinker = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_clinker_fraction_cement, False, return_type = "array_base", expand_to_all_cats = True)
        array_ippu_production_clinker = sf.vec_bounds(array_ippu_production_clinker*array_ippu_production - array_ippu_net_imports_clinker, (0, np.inf))
        array_ippu_emissions_clinker *= array_ippu_production_clinker


        ##  GENERAL EMISSIONS

        # dictionary that contains some baseline emissions from secondary sources (e.g., cement clinker) -- will add to those calculated from dict_ippu_simple_efs
        dict_ippu_proc_emissions_to_add = {
            self.modvar_ippu_emissions_process_co2: array_ippu_emissions_clinker
        }
        # dictionary variables mapping emission variable to component tuples (gdp, production). No factor is []
        dict_ippu_proc_simple_efs = {
            self.modvar_ippu_emissions_process_ch4: (
                [],
                [
                    self.modvar_ippu_ef_ch4_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_co2: (
                [],
                [
                    self.modvar_ippu_ef_co2_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_hfc: (
                [],
                [
                    self.modvar_ippu_ef_hfc23_per_prod_process,
                    self.modvar_ippu_ef_hfc32_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_n2o: (
                [
                    self.modvar_ippu_ef_n2o_per_gdp_process
                ],
                [
                    self.modvar_ippu_ef_n2o_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_other_fcs: (
                [],
                [
                    self.modvar_ippu_ef_nf3_per_prod_process,
                    self.modvar_ippu_ef_octafluoro_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_pfc: (
                [],
                [
                    self.modvar_ippu_ef_pfc14_per_prod_process,
                    self.modvar_ippu_ef_pfc116_per_prod_process,
                    self.modvar_ippu_ef_pfc218_per_prod_process,
                    self.modvar_ippu_ef_pfc1114_per_prod_process,
                    self.modvar_ippu_ef_pfcc318_per_prod_process,
                    self.modvar_ippu_ef_pfcc1418_per_prod_process
                ]
            ),
            self.modvar_ippu_emissions_process_sf6: (
                [
                    self.modvar_ippu_ef_sf6_per_gdp_process
                ],
                [
                    self.modvar_ippu_ef_sf6_per_prod_process
                ]
            )
        }

        # use dictionary to calculate emissions
        df_out += self.calculate_emissions_by_gdp_and_production(
            df_ippu_trajectories,
            array_ippu_production,
            vec_gdp,
            dict_ippu_proc_emissions_to_add,
            dict_ippu_proc_simple_efs,
            self.modvar_ippu_qty_total_production
        )



        #####################
        #    PRODUCT USE    #
        #####################

        ##  PRODUCT USE FROM PARAFFIN WAX AND LUBRICANTS
        array_ippu_useinit_nonenergy_fuel = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_useinit_nonenergy_fuel, False, return_type = "array_base", expand_to_all_cats = True)
        array_ippu_pwl_growth = sf.project_growth_scalar_from_elasticity(vec_rates_gdp, np.ones(len(array_ippu_useinit_nonenergy_fuel)), False, "standard")
        array_ippu_emissions_produse_nonenergy_fuel = np.outer(array_ippu_pwl_growth, array_ippu_useinit_nonenergy_fuel[0])
        array_ippu_production_scalar = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_scalar_production, False, return_type = "array_base", var_bounds = (0, np.inf), expand_to_all_cats = True)
        # get the emission factor and project emissions (unitless emissions)
        array_ippu_ef_co2_produse = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_ef_co2_per_prod_produse, False, return_type = "array_base", expand_to_all_cats = True)
        array_ippu_emissions_produse_nonenergy_fuel *= array_ippu_ef_co2_produse*self.model_attributes.get_scalar(self.modvar_ippu_useinit_nonenergy_fuel, "mass")
        array_ippu_emissions_produse_nonenergy_fuel *= array_ippu_production_scalar
        array_ippu_elasticity_produse = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_elast_produserate_to_gdppc, False, return_type = "array_base", expand_to_all_cats = True)
        array_ippu_gdp_scalar_produse = sf.project_growth_scalar_from_elasticity(vec_rates_gdp_per_capita, array_ippu_elasticity_produse, False, "standard")
        # this scalar array accounts for elasticity changes in per/gdp product use rates due to increases in gdp/capita, increases in gdp, and exogenously-defined reductions to production
        array_ippu_gdp_scalar_produse = (array_ippu_gdp_scalar_produse.transpose()*np.concatenate([np.ones(1), np.cumprod(1 + vec_rates_gdp)])).transpose()
        array_ippu_gdp_scalar_produse = array_ippu_gdp_scalar_produse * vec_gdp[0]
        array_ippu_gdp_scalar_produse *= array_ippu_production_scalar

        ##  OTHER EMISSIONS (very small--NMVOC, e.g.)
        array_ippu_emissions_other_nonenergy_co2 = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_emissions_other_nonenergy_co2, False, return_type = "array_units_corrected", expand_to_all_cats = True)
        array_ippu_emissions_other_nonenergy_co2 = array_ippu_emissions_other_nonenergy_co2[0]*sf.project_growth_scalar_from_elasticity(vec_rates_gdp, np.ones(array_ippu_emissions_other_nonenergy_co2.shape), False, "standard")
        array_ippu_emissions_other_nonenergy_co2 *= array_ippu_production_scalar

        ##  OTHER PRODUCT USE EMISSIONS (HIGH DEGREE OF VARIATION BY COUNTRY)

        """
        # get emission factor
        array_ippu_net_imports_clinker = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_net_imports_clinker
        , False, return_type = "array_base", expand_to_all_cats = True)
        array_ippu_net_imports_clinker *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_ippu_net_imports_clinker,
            self.modvar_ippu_qty_total_production,
            "mass"
        )
        # get production of clinker, remove net imports (and cap reduction to 0), and calculate emissions
        array_ippu_production_clinker = self.model_attributes.get_standard_variables(df_ippu_trajectories, self.modvar_ippu_clinker_fraction_cement, False, return_type = "array_base", expand_to_all_cats = True)
        array_ippu_production_clinker = sf.vec_bounds(array_ippu_production_clinker*array_ippu_production - array_ippu_net_imports_clinker, (0, np.inf))
        array_ippu_emissions_clinker *= array_ippu_production_clinker
        """


        ##  OTHER PRODUCT USE

        dict_ippu_produse_emissions_to_add = {
            self.modvar_ippu_emissions_produse_co2: array_ippu_emissions_produse_nonenergy_fuel + array_ippu_emissions_other_nonenergy_co2
        }

        dict_ippu_produse_simple_efs = {
            self.modvar_ippu_emissions_produse_co2: (
                [],
                []
            ),
            self.modvar_ippu_emissions_produse_hfc: (
                [
                    self.modvar_ippu_ef_hfc23_per_gdp_produse,
                    self.modvar_ippu_ef_hfc32_per_gdp_produse,
                    self.modvar_ippu_ef_hfc125_per_gdp_produse,
                    self.modvar_ippu_ef_hfc134a_per_gdp_produse,
                    self.modvar_ippu_ef_hfc143a_per_gdp_produse,
                    self.modvar_ippu_ef_hfc152a_per_gdp_produse,
                    self.modvar_ippu_ef_hfc227ea_per_gdp_produse,
                    self.modvar_ippu_ef_hfc236fa_per_gdp_produse,
                    self.modvar_ippu_ef_hfc245fa_per_gdp_produse,
                    self.modvar_ippu_ef_hfc365mfc_per_gdp_produse,
                    self.modvar_ippu_ef_hfc4310mee_per_gdp_produse
                ],
                []
            ),
            self.modvar_ippu_emissions_produse_pfc: (
                [
                    self.modvar_ippu_ef_pfc3110_per_gdp_produse,
                    self.modvar_ippu_ef_pfc5114_per_gdp_produse,
                    self.modvar_ippu_ef_pfc14_per_gdp_produse,
                    self.modvar_ippu_ef_pfc116_per_gdp_produse
                ],
                []
            )
        }
        """

        """

        # use dictionary to calculate emissions
        df_out += self.calculate_emissions_by_gdp_and_production(
            df_ippu_trajectories,
            0,
            array_ippu_gdp_scalar_produse,
            dict_ippu_produse_emissions_to_add,
            dict_ippu_produse_simple_efs,
            self.modvar_ippu_qty_total_production
        )


        # non-standard emission fields to include in emission total for IPPU
        vars_additional_sum = [
            self.modvar_ippu_emissions_process_hfc,
            self.modvar_ippu_emissions_produse_hfc,
            self.modvar_ippu_emissions_process_pfc,
            self.modvar_ippu_emissions_produse_pfc,
            self.modvar_ippu_emissions_process_other_fcs
        ]
        # concatenate and add subsector emission totals
        df_out = sf.merge_output_df_list(df_out, self.model_attributes, "concatenate")
        self.model_attributes.add_subsector_emissions_aggregates(df_out, self.required_base_subsectors, False)
        self.model_attributes.add_specified_total_fields_to_emission_total(df_out, vars_additional_sum)

        return df_out
