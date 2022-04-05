import support_functions as sf
import data_structures as ds
from model_socioeconomic import Socioeconomic
import pandas as pd
import numpy as np
import time

class CircularEconomy:

    def __init__(self, attributes: ds.ModelAttributes):

        self.model_attributes = attributes
        self.required_dimensions = self.get_required_dimensions()
        self.required_subsectors, self.required_base_subsectors = self.get_required_subsectors()
        self.required_variables, self.output_variables = self.get_ce_input_output_fields()
        self.required_variables_wali, self.output_variables_wali = self.get_ce_input_output_fields([x for x in self.required_subsectors if (x != "Solid Waste")])

        ##  set some model fields to connect to the attribute tables

        # liquid waste model variables
        self.modvar_wali_bod_correction = "BOD Correction Factor for TOW"
        self.modvar_wali_bod_per_capita = "BOD per Capita"
        self.modvar_wali_cod_per_gdp = "COD per GDP"
        self.modvar_wali_frac_nitrogen_removed_in_treatment = "Nitrogen Treatment Removal Fraction"
        self.modvar_wali_frac_protein_with_red_meat = "Fraction of Protein in Diet with Red Meat"
        self.modvar_wali_frac_protein_without_red_meat = "Fraction of Protein in Diet without Red Meat"
        self.modvar_wali_init_pcap_wwgen = "Initial Per Capita Annual Domestic Wastewater Generated"
        self.modvar_wali_init_pgdp_wwgen = "Initial Per GDP Annual Industrial Wastewater Generated"
        self.modvar_wali_logelast_ww_to_gdppc = "Log Elasticity DWW Production to GDP Per Capita"
        self.modvar_wali_max_bod_capac = "Maximum BOD :math:\\text{CH}_4 Producing Capacity"
        self.modvar_wali_max_cod_capac = "Maximum COD :math:\\text{CH}_4 Producing Capacity"
        self.modvar_wali_nitrogen_density_ww_ind = "Nitrogen Density of Industrial Wastewater"
        self.modvar_wali_optional_elasticity_protein_to_gdppc = "(Optional) Elasticity of Protein in Diet to GDP per Capita"
        self.modvar_wali_param_fnoncon = "Factor for Nitrogen in Non-Consumed Protein Disposed in Sewer System"
        self.modvar_wali_param_nhh = "Scalar to Account for Nitrogen in Household Products"
        self.modvar_wali_protein_per_capita = "Average Protein Consumption Per Capita"
        self.modvar_wali_treatpath_aerobic = "Treatment Fraction Aerobic"
        self.modvar_wali_treatpath_anaerobic = "Treatment Fraction Anaerobic"
        self.modvar_wali_treatpath_septic = "Treatment Fraction Septic"
        self.modvar_wali_treatpath_latrine_improved = "Treatment Fraction Improved Latrine"
        self.modvar_wali_treatpath_latrine_unimproved = "Treatment Fraction Unimproved Latrine"
        self.modvar_wali_treatpath_untreated_no_sewerage = "Treatment Fraction Untreated No Sewerage"
        self.modvar_wali_treatpath_untreated_with_sewerage = "Treatment Fraction Untreated With Sewerage"

        # domestic solid waste model variables
        self.modvar_waso_annual_vkmt_per_collection_vehicle = "Average VKMT Per Waste Collection Vehicle"
        self.modvar_waso_annual_waste_collected_per_collection_vehicle = "Average Annual Waste Transported Per Waste Collection Vehicle"
        self.modvar_waso_composition_isw = "Initial Composition Fraction Industrial Solid Waste"
        self.modvar_waso_ef_ch4_biogas = ":math:\\text{CH}_4 Anaerobic Biogas Emission Factor"
        self.modvar_waso_ef_ch4_compost = ":math:\\text{CH}_4 Composting Emission Factor"
        self.modvar_waso_ef_ch4_incineration_isw = ":math:\\text{CH}_4 ISW Incineration Emission Factor"
        self.modvar_waso_ef_ch4_incineration_msw = ":math:\\text{CH}_4 MSW Incineration Emission Factor"
        self.modvar_waso_ef_n2o_compost = ":math:\\text{N}_2\\text{O} Composting Emission Factor"
        self.modvar_waso_ef_n2o_incineration = ":math:\\text{N}_2\\text{O} Incineration Emission Factor"
        self.modvar_waso_elast_msw = "Elasticity of Municipal Solid Waste Produced to GDP per Capita"
        self.modvar_waso_emissions_ch4_biogas = ":math:\\text{CH}_4 Emissions from Anearobic Biogas"
        self.modvar_waso_emissions_ch4_compost = ":math:\\text{CH}_4 Emissions from Composting"
        self.modvar_waso_emissions_ch4_incineration = ":math:\\text{CH}_4 Emissions from Incineration"
        self.modvar_waso_emissions_ch4_landfill = ":math:\\text{CH}_4 Emissions from Landfills"
        self.modvar_waso_emissions_n2o_compost = ":math:\\text{N}_2\\text{O} Emissions from Composting"
        self.modvar_waso_emissions_n2o_incineration = ":math:\\text{N}_2\\text{O} Emissions from Incineration"
        self.modvar_waso_frac_ch4_flared_composting = "Fraction of Methane Flared at Composting Facilities"
        self.modvar_waso_frac_biogas = "Fraction of Waste Treated Anaerobically"
        self.modvar_waso_frac_compost = "Fraction of Waste Composted"
        self.modvar_waso_frac_landfill_gas_ch4_to_energy = "Fraction of Landfill Gas Recovered for Energy"
        self.modvar_waso_frac_nonrecycled_incinerated = "Fraction of Non-Recycled Solid Waste Incinerated"
        self.modvar_waso_frac_nonrecycled_landfilled = "Fraction of Non-Recycled Solid Waste Landfilled"
        self.modvar_waso_frac_nonrecycled_opendump = "Fraction of Non-Recycled Solid Waste Open Dumps"
        self.modvar_waso_frac_recovered_for_energy_incineration = "Fraction of Incineration Recovered for Energy"
        self.modvar_waso_frac_recycled = "Fraction of Waste Recycled"
        self.modvar_waso_init_composition_msw = "Initial Composition Fraction Municipal Solid Waste"
        self.modvar_waso_init_isw_generated_pgdp = "Per GDP Industrial Solid Waste Generated"
        self.modvar_waso_init_msw_generated_pc = "Initial Per Capita Municipal Solid Waste Generated"
        self.modvar_waso_mcf_landfills_average = "Average Methane Correction Factor at Landfills"
        self.modvar_waso_physparam_k = "K"
        self.modvar_waso_recovered_biogas = "Biogas Recovered from Anaerobic Facilities"
        self.modvar_waso_recovered_ch4_landfill_gas = ":math:\\text{CH}_4 Recovered from Landfill Gas"
        self.modvar_waso_rf_biogas = "Biogas Recovery Factor"
        self.modvar_waso_rf_landfill_gas_recovered = "Fraction of Landfill Gas Captured at Landfills"
        self.modvar_waso_rf_landfill_gas_to_ch4 = ":math:\\text{CH}_4 Recovery Factor Landfill Gas"
        self.modvar_waso_waste_per_capita_scalar = "Waste Per Capita Scale Factor"
        self.modvar_waso_waste_total_biogas = "Total Waste Anaerobic Biogas"
        self.modvar_waso_waste_total_compost = "Total Waste Composted"
        self.modvar_waso_waste_total_incinerated = "Total Waste Incinerated"
        self.modvar_waso_waste_total_produced = "Total Solid Waste Produced"
        self.modvar_waso_waste_total_landfilled = "Total Waste Landfilled"
        self.modvar_waso_waste_total_open_dumped = "Total Waste Open Dumped"
        self.modvar_waso_waste_total_recycled = "Total Waste Recycled"

        # wastewater treatment
        self.modvar_trww_ef_n2o_wastewater_treatment = ":math:\\text{N}_2\\text{O} Wastewater Treatment Emission Factor"
        self.modvar_trww_emissions_ch4_treatment = ":math:\\text{CH}_4 Emissions from Wastewater Treatment"
        self.modvar_trww_emissions_n2o_treatment = ":math:\\text{N}_2\\text{O} Emissions from Wastewater Treatment"
        self.modvar_trww_emissions_n2o_effluent = ":math:\\text{N}_2\\text{O} Emissions from Wastewater Effluent"
        self.modvar_trww_frac_tow_removed = "Fraction of Total Organic Waste Removed"
        self.modvar_trww_krem = ":math:\\text{K}_{REM} Sludge Factor"
        self.modvar_trww_mcf = "Methane Correction Factor"
        self.modvar_trww_septic_sludge_compliance = "Septic Sludge Compliance Fraction"
        self.modvar_trww_sludge_produced = "Mass of Sludge Produced"
        self.modvar_trww_total_bod_treated = "Total BOD Treated"
        self.modvar_trww_total_cod_treated = "Total COD Treated"
        self.modvar_trww_total_tow_bod_in_effluent = "Total BOD Organic Waste in Effluent"
        self.modvar_trww_total_tow_cod_in_effluent = "Total COD Organic Waste in Effluent"
        self.modvar_trww_vol_ww_treated = "Volume of Wastewater Treated"

        # other sectors' variables, used in integration
        self.modvar_lvst_net_imports = "Change to Net Imports of Livestock"
        self.modvar_lvst_pop = "Livestock Head Count"


        ##  MISCELLANEOUS VARIABLES

        self.time_periods, self.n_time_periods = self.model_attributes.get_time_periods()
        self.vars_wali_to_trww = self.model_attributes.get_ordered_vars_by_nonprimary_category("Liquid Waste", "Wastewater Treatment", "key_varreqs_all")

        # TEMP:SET TO DERIVE FROM ATTRIBUTE TABLES---
        self.landfill_gas_frac_methane = 0.5
        # fraction of protein composed of nitrogen
        self.factor_f_npr = 0.16
        self.factor_n2on_to_n2o = float(11/7)

        # add socioeconomic
        self.model_socioeconomic = Socioeconomic(self.model_attributes)




    ##  FUNCTIONS FOR MODEL ATTRIBUTE DIMENSIONS

    def check_df_fields(self, df_ce_trajectories, check_fields = None):
        if check_fields == None:
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

    def get_ce_input_output_fields(self, subsectors = None):
        if subsectors == None:
            subsectors = self.required_subsectors
        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(subsectors)
        return required_vars + self.get_required_dimensions(), output_vars




    #####################################
    ###                               ###
    ###    PRIMARY MODEL FUNCTIONS    ###
    ###                               ###
    #####################################


    ## project protein consumption
    def project_protein_consumption(self, df_ce_trajectories: pd.DataFrame, vec_pop: np.ndarray, vec_rates_gdp_per_capita: np.ndarray = None) -> np.array:
        """
            Projects protein consumption (in kg) based on livestock growth, or, if not integrated, a specified elasticity
        """
        # get scalar that represents the impact of a reduction of protein in the vegetarian diet
        vec_wali_frac_protein_in_diet_with_rm = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_frac_protein_with_red_meat, True, return_type = "array_base")
        vec_wali_frac_protein_in_diet_without_rm = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_frac_protein_without_red_meat, True, return_type = "array_base")
        vec_wali_protein_scalar_no_rm = vec_wali_frac_protein_in_diet_without_rm/vec_wali_frac_protein_in_diet_with_rm
        vec_gnrl_frac_eating_red_meat = self.model_attributes.get_standard_variables(df_ce_trajectories, self.model_socioeconomic.modvar_gnrl_frac_eating_red_meat, True, return_type = "array_base", var_bounds = (0, 1))
        vec_wali_protein_scalar = (vec_gnrl_frac_eating_red_meat + vec_wali_protein_scalar_no_rm*(1 - vec_gnrl_frac_eating_red_meat)).flatten()
        # get protein consumed per person in kg/year
        vec_wali_protein_per_capita = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_protein_per_capita, False, return_type = "array_base")*self.model_attributes.configuration.get("days_per_year")
        # get livestock population (a) and net imports (b) if available; otherwise, default to an elasticity
        modvar_proj_protein_driver_a, array_project_protein_driver_a = self.model_attributes.get_optional_or_integrated_standard_variable(df_ce_trajectories, self.modvar_lvst_pop, self.modvar_wali_optional_elasticity_protein_to_gdppc, True, "array_base")
        modvar_proj_protein_driver_b, array_project_protein_driver_b = self.model_attributes.get_optional_or_integrated_standard_variable(df_ce_trajectories, self.modvar_lvst_net_imports, self.modvar_wali_optional_elasticity_protein_to_gdppc, True, "array_base")

        # project depending on availability
        if modvar_proj_protein_driver_a == self.modvar_lvst_pop:
            """
                use estimate of total animal weight for increase in protein content in diet
                - note that projections of animal demand takes into account shifts in diet away from red meat
                - however, we still have to correct for the reduction of protein in non-red meat diets
            """
            array_lvst_total_dem = array_project_protein_driver_a + array_project_protein_driver_b
            vec_lvst_weights = self.model_attributes.get_ordered_category_attribute("Livestock", "animal_weight_kg")
            vec_protein_growth = np.sum(array_lvst_total_dem*vec_lvst_weights, axis = 1)
            vec_protein_growth = np.concatenate([np.ones(1), np.cumprod(vec_protein_growth[1:]/vec_protein_growth[0:-1])])
        else:
            if type(vec_rates_gdp_per_capita) == type(None):
                raise ValueError(f"Error in project_protein_consumption: Livestock growth rates not found in data frame. To use the '{self.modvar_wali_optional_elasticity_protein_to_gdppc}' variable, specify a vector of gdp growth rates.")
            # in this case, array_project_protein_driver_a == array_project_protein_driver_a
            vec_wali_elast_protein = array_project_protein_driver_a.flatten()
            vec_protein_growth = sf.project_growth_scalar_from_elasticity(vec_rates_gdp_per_capita, vec_wali_elast_protein, False, "standard")
        # total protein
        vec_wali_protein_kg = vec_wali_protein_per_capita*vec_pop*vec_protein_growth*vec_wali_protein_scalar

        return vec_wali_protein_kg


    ##  project emissions and outputs from liquid waste and wastewater treatment subsectors
    def project_waste_liquid(self,
        df_ce_trajectories: pd.DataFrame,
        df_se_internal_shared_variables: pd.DataFrame = None,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
            - CircularEconomy.project_waste_liquid takes a data frame (ordered by time series) and returns a data frame of the same order
            - designed to be parallelized or called from command line via __main__ in run_afolu.py
            - df_ce_trajectories should have all input fields required (see CircularEconomy.required_variables for a list of variables to be defined) for the Liquid Waste and Wastewater Treatment sectors
            - the df_ce_trajectories.project_waste_liquid method will run on valid time periods from 1 .. k, where k <= n (n is the number of time periods). By default, it drops invalid time periods. If there are missing time_periods between the first and maximum, data are interpolated.

            - df_ce_trajectories: data frame of input trajectories

            - df_se_internal_shared_variables: Default = None. Data frame of socioeconomic projections that are used internally. If none, the socioeconomic model will be called to project based on the input data frame.

            - dict_dims: dictionary of scenario dimensions (if applicable). Default = None. If none, ModelAttribute.check_projection_input_df() will be run to obtain it.

            - n_projection_time_periods: number of time periods in the projection. Default = None. If none, ModelAttribute.check_projection_input_df() will be run to obtain it.

            - projection_time_periods: list of time periods in the projection. Default = None. If none, ModelAttribute.check_projection_input_df() will be run to obtain it.
        """

        ##  CHECKS

        # make sure socioeconomic variables are added and
        if type(df_se_internal_shared_variables) == type(None):
            df_ce_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_ce_trajectories)
        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_ce_trajectories, self.required_variables_wali)
        if type(None) in [type(dict_dims), type(n_projection_time_periods), type(projection_time_periods)]:
            dict_dims, df_ce_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_ce_trajectories, True, True, True)


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_gnrl = self.model_attributes.get_subsector_attribute("General", "pycategory_primary")
        pycat_trww = self.model_attributes.get_subsector_attribute("Wastewater Treatment", "pycategory_primary")
        pycat_wali = self.model_attributes.get_subsector_attribute("Liquid Waste", "pycategory_primary")
        # attribute tables
        attr_gnrl = self.model_attributes.dict_attributes[pycat_gnrl]
        attr_trww = self.model_attributes.dict_attributes[pycat_trww]
        attr_wali = self.model_attributes.dict_attributes[pycat_wali]


        ##  ECON/GNRL VECTOR AND ARRAY INITIALIZATION

        # get some vectors
        vec_gdp = self.model_attributes.get_standard_variables(df_ce_trajectories, self.model_socioeconomic.modvar_econ_gdp, False, return_type = "array_base")
        vec_pop = self.model_attributes.get_standard_variables(df_ce_trajectories, self.model_socioeconomic.modvar_gnrl_pop_total, False, return_type = "array_base")
        array_pop = self.model_attributes.get_standard_variables(df_ce_trajectories, self.model_socioeconomic.modvar_gnrl_subpop, False, return_type = "array_base")
        vec_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_gdp_per_capita"])
        vec_rates_gdp = np.array(df_se_internal_shared_variables["vec_rates_gdp"].dropna())
        vec_rates_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_rates_gdp_per_capita"].dropna())


        ##  OUTPUT INITIALIZATION

        df_out = [df_ce_trajectories[self.required_dimensions].copy()]


        ######################
        #    LIQUID WASTE    #
        ######################

        ##  GET INITIAL WW GENERATED + BASED ON BOD/PERSON + COD/GDP, SET IMPLIED FRACTION OF BOD/M3 WW

        # bod/cod
        vec_wali_bod_percap_init = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_bod_per_capita, True, return_type = "array_units_corrected")[0, :]*self.model_attributes.configuration.get("days_per_year")
        vec_wali_bod_correction = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_bod_correction, False, return_type = "array_base")
        array_wali_bod_percap = np.outer(vec_wali_bod_correction, vec_wali_bod_percap_init)
        array_wali_cod_pergdp = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_cod_per_gdp, True, return_type = "array_units_corrected")
        # get elasticity of wastewater
        vec_wali_logelastic = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_logelast_ww_to_gdppc, False, return_type = "array_base")
        vec_wali_scale_percapita_dem = sf.project_growth_scalar_from_elasticity(vec_rates_gdp_per_capita, vec_wali_logelastic, False, "log")
        # volume per capita (m3)
        array_wali_vol_domww_percap = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_init_pcap_wwgen, True, return_type = "array_base")
        array_wali_vol_domww_percap = (array_wali_vol_domww_percap.transpose() * vec_wali_bod_correction).transpose()
        array_wali_vol_indww_per_gdp = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_init_pgdp_wwgen, True, return_type = "array_base")
        # scale per capita volume and bod/person (representing increases)
        array_wali_bod_percap = (array_wali_bod_percap.transpose()*vec_wali_scale_percapita_dem).transpose()
        array_wali_vol_domww_percap = (array_wali_vol_domww_percap.transpose()*vec_wali_scale_percapita_dem).transpose()
        # total bod (kg), cod (tonne), and wastewater (m3) generated
        array_wali_bod_total = (array_wali_bod_percap.transpose()*array_pop.transpose()).transpose()
        array_wali_domww_total = (array_wali_vol_domww_percap.transpose()*array_pop.transpose()).transpose()
        array_wali_cod_total = (array_wali_cod_pergdp.transpose()*vec_gdp).transpose()
        array_wali_indww_total = (array_wali_vol_indww_per_gdp.transpose()*vec_gdp).transpose()


        ##  CALCULATE TOTALS SENT TO EACH TREATMENT PATH

        #
        # DOM WW IS OK
        # TMP: INDUSTRIAL CAN TO BE IMPROVED TO INTEGRATE PRODUCTION BY INDUSTRY
        #
        cats_dom_ww = list(attr_wali.table[attr_wali.table[pycat_gnrl] != "none"][pycat_wali])
        cats_ind_ww = list(attr_wali.table[attr_wali.table["industrial_category"] != "none"][pycat_wali])
        # initialize bod/cod (oxygen demand) and volume by category (as transpose)
        array_trww_total_bod_by_pathway = np.zeros((len(attr_trww.key_values), n_projection_time_periods))
        array_trww_total_cod_by_pathway = array_trww_total_bod_by_pathway.copy()
        array_trww_total_ww_bod_by_pathway = array_trww_total_bod_by_pathway.copy()
        array_trww_total_ww_cod_by_pathway = array_trww_total_bod_by_pathway.copy()

        ##  GET TOTALS BY TREATMENT PATHWAY

        # domestiic
        for cdw in cats_dom_ww:
            # get population category
            cat_gnrl = ds.clean_schema(self.model_attributes.dict_attributes[pycat_wali].field_maps[f"{pycat_wali}_to_{pycat_gnrl}"][cdw])
            ind_gnrl = attr_gnrl.get_key_value_index(cat_gnrl)
            # the associated vector of wastewater produced + bod produced
            vec_bod = array_wali_bod_total[:, ind_gnrl]
            vec_ww = array_wali_domww_total[:, ind_gnrl]
            # get the treatment pathway
            vars_treatment_path = []
            for var in self.vars_wali_to_trww:
                vars_treatment_path += self.model_attributes.build_varlist("Liquid Waste", var, [cdw])
            array_pathways = sf.check_row_sums(np.array(df_ce_trajectories[vars_treatment_path]), msg_pass = f" 'df_ce_trajectories[vars_treatment_path]' for wali category '{cdw}'")
            # add to output arrays
            array_trww_total_bod_by_pathway += (array_pathways.transpose()*vec_bod)
            array_trww_total_ww_bod_by_pathway += (array_pathways.transpose()*vec_ww)

        # industrial
        for cdw in cats_ind_ww:
            ind_industry = 0
            # the associated vector of wastewater produced + bod produced
            vec_cod = array_wali_cod_total[:, ind_industry]
            vec_ww = array_wali_indww_total[:, ind_industry]
            # get the treatment pathway
            vars_treatment_path = []
            for var in self.vars_wali_to_trww:
                vars_treatment_path += self.model_attributes.build_varlist("Liquid Waste", var, [cdw])
            array_pathways = sf.check_row_sums(np.array(df_ce_trajectories[vars_treatment_path]), msg_pass = f" 'df_ce_trajectories[vars_treatment_path]' for wali category '{cdw}'")
            # add to output arrays
            array_trww_total_cod_by_pathway += (array_pathways.transpose()*vec_cod)
            array_trww_total_ww_cod_by_pathway += (array_pathways.transpose()*vec_ww)

        # total bod (kg -> tonne), cod (tonne), and ww vol (m3) -- get factor, which is applied only to the data frame (to presreve array_trww_total_bod_by_pathway in units of emissions mass for downstream calculations)
        factor_trww_emissions_mass_to_tow_mass = self.model_attributes.get_mass_equivalent(self.model_attributes.configuration.get("emissions_mass").lower(), self.model_attributes.get_variable_characteristic(self.modvar_trww_sludge_produced, "$UNIT-MASS$"))
        array_trww_total_bod_by_pathway = array_trww_total_bod_by_pathway.transpose()
        array_trww_total_cod_by_pathway = array_trww_total_cod_by_pathway.transpose()
        array_trww_total_ww_bod_by_pathway = array_trww_total_ww_bod_by_pathway.transpose()
        array_trww_total_ww_cod_by_pathway = array_trww_total_ww_cod_by_pathway.transpose()
        array_trww_total_ww_by_pathway = array_trww_total_ww_bod_by_pathway + array_trww_total_ww_cod_by_pathway
        # data frame for output
        df_trww_total_bod_by_pathway = self.model_attributes.array_to_df(array_trww_total_bod_by_pathway*factor_trww_emissions_mass_to_tow_mass, self.modvar_trww_total_bod_treated)
        df_trww_total_cod_by_pathway = self.model_attributes.array_to_df(array_trww_total_cod_by_pathway*factor_trww_emissions_mass_to_tow_mass, self.modvar_trww_total_cod_treated)
        df_trww_total_ww_by_pathway = self.model_attributes.array_to_df(array_trww_total_ww_by_pathway, self.modvar_trww_vol_ww_treated)
        # add to output
        df_out += [
            df_trww_total_bod_by_pathway,
            df_trww_total_cod_by_pathway,
            df_trww_total_ww_by_pathway
        ]


        ##  GET METHANE EMISSIONS FROM EACH TREATMENT PROCESS

        # get maximum methane production capacity for bod/cod (in co2e - i.e., using array_units_corrected)
        vec_wali_bod_max_bo = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_max_bod_capac, False, return_type = "array_units_corrected")
        vec_wali_cod_max_bo = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_max_cod_capac, False, return_type = "array_units_corrected")
        # get arrays for the treatment-specific methane correction factor,
        array_trww_mcf = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_trww_mcf, True, return_type = "array_base")
        array_trww_frac_tow_removed = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_trww_frac_tow_removed, True, return_type = "array_base")
        # get some specific factors and merge them to all categories (aerobic + septic, for sludge removal)
        array_trww_krem = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_trww_krem, True, return_type = "array_base")
        array_trww_krem = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(array_trww_krem, self.modvar_trww_krem)
        array_trww_septic_compliance = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_trww_septic_sludge_compliance, True, return_type = "array_base")
        array_trww_septic_compliance = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(array_trww_septic_compliance, self.modvar_trww_septic_sludge_compliance)
        # get treatment pathways that produce sludge
        array_mask_sludge = np.sign(array_trww_krem) + np.sign(array_trww_septic_compliance)
        # next, once krem has been used, replace 0s with 1s and used to divide to estimate the total mass of sludge (which is passed to the solid waste model)
        sf.repl_array_val_twodim(array_trww_krem, 0, 1)
        # calcualte total organic waste removed by type as sludge (use TOW_{REM} values from table 6.6B in IPCC GNGHG Inventories 2019) BOD then COD
        array_trww_tow_bod_removed_sludge = (array_trww_frac_tow_removed + array_trww_septic_compliance*0.5)*array_trww_total_bod_by_pathway*array_mask_sludge
        array_trww_tow_bod_not_removed = array_trww_total_bod_by_pathway - array_trww_tow_bod_removed_sludge
        array_trww_tow_cod_removed_sludge = (array_trww_frac_tow_removed + array_trww_septic_compliance*0.5)*array_trww_total_cod_by_pathway*array_mask_sludge
        array_trww_tow_cod_not_removed = array_trww_total_cod_by_pathway - array_trww_tow_cod_removed_sludge
        # apply methane correction factor to estimate methane emissions (these are in co2e
        array_trww_emissions_ch4_bod = ((array_trww_tow_bod_not_removed*array_trww_mcf).transpose()*vec_wali_bod_max_bo).transpose()
        array_trww_emissions_ch4_cod = ((array_trww_tow_cod_not_removed*array_trww_mcf).transpose()*vec_wali_cod_max_bo).transpose()
        array_trww_bod_equivalent_removed_sludge = array_trww_tow_bod_removed_sludge + (array_trww_tow_cod_removed_sludge.transpose()*(vec_wali_cod_max_bo/vec_wali_bod_max_bo)).transpose()
        array_trww_emissions_ch4_treatment = array_trww_emissions_ch4_bod + array_trww_emissions_ch4_cod
        # get sludge mass and mass of tow in effluent (convert to tonnes)
        array_trww_mass_removed_sludge = (array_trww_bod_equivalent_removed_sludge/array_trww_krem)*factor_trww_emissions_mass_to_tow_mass
        array_trww_tow_bod_effluent = array_trww_tow_bod_not_removed*(1 - array_trww_mcf)*factor_trww_emissions_mass_to_tow_mass
        array_trww_tow_cod_effluent = array_trww_tow_cod_not_removed*(1 - array_trww_mcf)*factor_trww_emissions_mass_to_tow_mass
        # data frames for output
        df_trww_emissions_ch4_treatment = self.model_attributes.array_to_df(array_trww_emissions_ch4_treatment, self.modvar_trww_emissions_ch4_treatment)
        df_trww_mass_removed_sludge = self.model_attributes.array_to_df(array_trww_mass_removed_sludge, self.modvar_trww_sludge_produced, reduce_from_all_cats_to_specified_cats = True)
        df_trww_tow_bod_effluent = self.model_attributes.array_to_df(array_trww_tow_bod_effluent, self.modvar_trww_total_tow_bod_in_effluent)
        df_trww_tow_cod_effluent = self.model_attributes.array_to_df(array_trww_tow_cod_effluent, self.modvar_trww_total_tow_cod_in_effluent)

        # add to output
        df_out += [
            df_trww_emissions_ch4_treatment,
            df_trww_mass_removed_sludge,
            df_trww_tow_bod_effluent,
            df_trww_tow_cod_effluent
        ]


        ######################
        #   N2O EMISSIONS    #
        ######################

        ##  START BY CALCULATING TOTAL NITROGEN

        #  calcualte the protein content (kg) and total nitrogen in domestic wastewater using V5, C6, Equation 6.10 from IPCC GNGHGI (2019R) - factors are default
        vec_wali_protein = self.project_protein_consumption(df_ce_trajectories, vec_pop, vec_rates_gdp_per_capita)
        # use the BOD commercial/industrial correction factor as f_indcom from 6.10
        vec_wali_findcom = vec_wali_bod_correction
        vec_wali_fnoncon = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_param_fnoncon, False, return_type = "array_base")
        vec_wali_nhh = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_param_nhh, False, return_type = "array_base")

        # get total domestic nitrogen
        vec_wali_total_nitrogen_dom = vec_wali_protein*vec_wali_findcom*vec_wali_fnoncon*vec_wali_nhh*self.factor_f_npr
        # use BOD array to allocate domestic wastewater nitrogen (assume it's uniformly distributed)
        array_trww_total_nitrogen_dom = (array_trww_total_bod_by_pathway.transpose()/np.sum(array_trww_total_bod_by_pathway, axis = 1))
        array_trww_total_nitrogen_dom = (array_trww_total_nitrogen_dom*vec_wali_total_nitrogen_dom).transpose()
        # get total industrial nitrogen
        vec_wali_nitrogen_density_ind = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_nitrogen_density_ww_ind, False, return_type = "array_base")
        # use COD array to allocate industrial wastewater nitrogen (assume it's uniformly distributed)
        array_trww_total_nitrogen_ind = (array_trww_total_cod_by_pathway.transpose()/np.sum(array_trww_total_cod_by_pathway, axis = 1))
        array_trww_total_nitrogen_ind = (array_trww_total_nitrogen_ind*vec_wali_nitrogen_density_ind).transpose()*array_trww_total_ww_cod_by_pathway
        # get total nitrogen in each treatment pathway and find total removed by treatment
        array_trww_total_nitrogen = array_trww_total_nitrogen_dom + array_trww_total_nitrogen_ind
        array_trww_frac_n_removed = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_wali_frac_nitrogen_removed_in_treatment, False, return_type = "array_base", var_bounds = (0, 1))
        array_trww_total_nitrogen_effluent = array_trww_total_nitrogen*(1 - array_trww_frac_n_removed)
        # retrieve the emission factors, which are g/g (unitless)
        array_trww_ef_n2o_ww = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_trww_ef_n2o_wastewater_treatment, False, return_type = "array_base")
        # nitrogen emissions in kg (first component) converted to emissions mass--assumes both industry and domestic have same units, kg
        factor_trww_mass_protein_to_emission_mass = self.model_attributes.get_scalar(self.modvar_wali_protein_per_capita, "mass")
        array_trww_emissions_n2o_treatment = array_trww_total_nitrogen*array_trww_ef_n2o_ww*self.factor_n2on_to_n2o*factor_trww_mass_protein_to_emission_mass
        array_trww_emissions_n2o_effluent = array_trww_total_nitrogen_effluent.transpose()*array_trww_ef_n2o_ww[:, attr_trww.get_key_value_index("untreated_no_sewerage")]*self.factor_n2on_to_n2o*factor_trww_mass_protein_to_emission_mass
        array_trww_emissions_n2o_effluent = array_trww_emissions_n2o_effluent.transpose()
        # set to data frame and add to the output
        df_trww_emissions_n2o_treatment = self.model_attributes.array_to_df(array_trww_emissions_n2o_treatment, self.modvar_trww_emissions_n2o_treatment, True)
        df_trww_emissions_n2o_effluent = self.model_attributes.array_to_df(array_trww_emissions_n2o_effluent, self.modvar_trww_emissions_n2o_effluent, True)

        df_out += [
            df_trww_emissions_n2o_effluent,
            df_trww_emissions_n2o_treatment
        ]

        df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)

        return df_out

    ##  TEMP ORDERING, MOVE TO LAST AFTER FINISHING SOLID WASTE
    ##  primary method for integrated liquid/solid waste
    def project(self, df_ce_trajectories: pd.DataFrame) -> pd.DataFrame:

        """
            - CircularEconomy.project takes a data frame (ordered by time series) and returns a data frame of the same order
            - designed to be parallelized or called from command line via __main__ in run_afolu.py
            - df_ce_trajectories should have all input fields required (see CircularEconomy.required_variables for a list of variables to be defined)
            - the df_ce_trajectories.project method will run on valid time periods from 1 .. k, where k <= n (n is the number of time periods). By default, it drops invalid time periods. If there are missing time_periods between the first and maximum, data are interpolated.
        """


        ##  CHECKS

        # make sure socioeconomic variables are added and
        df_ce_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_ce_trajectories)
        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_ce_trajectories)
        dict_dims, df_ce_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_ce_trajectories, True, True, True)


        # initialize by running waste
        df_out = [self.project_waste_liquid(df_ce_trajectories, df_se_internal_shared_variables, dict_dims, n_projection_time_periods, projection_time_periods)]
        # then, build input data frame for solid waste, which includes sludge totals that are reported from liquid waste
        df_waso_sludge = self.model_attributes.get_optional_or_integrated_standard_variable(df_out[0], self.modvar_trww_sludge_produced, None, True, "data_frame")

        df_in = pd.concat([df_ce_trajectories, df_waso_sludge[1]], axis = 1) if df_waso_sludge else df_ce_trajectories

        df_out += [
            self.project_waste_solid(df_in, df_se_internal_shared_variables, dict_dims, n_projection_time_periods, projection_time_periods)
        ]

        df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)

        #self.model_attributes.add_subsector_emissions_aggregates(df_out, self.required_base_subsectors, False)
        # TEMP UNTIL SOLID WASTE IS COMPLETE
        self.model_attributes.add_subsector_emissions_aggregates(df_out, ["Wastewater Treatment"], False)

        return df_out

    ##  project emissions and outputs from solid waste (excluding recylcing energy and process emissions, which are handled in IPPU)
    def project_waste_solid(self,
        df_ce_trajectories: pd.DataFrame,
        df_se_internal_shared_variables: pd.DataFrame = None,
        dict_dims: dict = None,
        n_projection_time_periods: int = None,
        projection_time_periods: list = None
    ) -> pd.DataFrame:

        """
            - CircularEconomy.project_waste_solid takes a data frame (ordered by time series) and returns a data frame of the same order
            - designed to be parallelized or called from command line via __main__ in run_afolu.py
            - df_ce_trajectories should have all input fields required (see CircularEconomy.required_variables for a list of variables to be defined) for the Solid Waste sector
            - the df_ce_trajectories.project_waste_liquid method will run on valid time periods from 1 .. k, where k <= n (n is the number of time periods). By default, it drops invalid time periods. If there are missing time_periods between the first and maximum, data are interpolated.

            - df_ce_trajectories: data frame of input trajectories

            - df_se_internal_shared_variables: Default = None. Data frame of socioeconomic projections that are used internally. If none, the socioeconomic model will be called to project based on the input data frame.

            - dict_dims: dictionary of scenario dimensions (if applicable). Default = None. If none, ModelAttribute.check_projection_input_df() will be run to obtain it.

            - n_projection_time_periods: number of time periods in the projection. Default = None. If none, ModelAttribute.check_projection_input_df() will be run to obtain it.

            - projection_time_periods: list of time periods in the projection. Default = None. If none, ModelAttribute.check_projection_input_df() will be run to obtain it.
        """

        ##  CHECKS

        # make sure socioeconomic variables are added and
        if type(df_se_internal_shared_variables) == type(None):
            df_ce_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_ce_trajectories)
        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_ce_trajectories, self.required_variables_wali)
        if type(None) in [type(dict_dims), type(n_projection_time_periods), type(projection_time_periods)]:
            dict_dims, df_ce_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_ce_trajectories, True, True, True)


        ##  CATEGORY AND ATTRIBUTE INITIALIZATION
        pycat_gnrl = self.model_attributes.get_subsector_attribute("General", "pycategory_primary")
        pycat_trww = self.model_attributes.get_subsector_attribute("Wastewater Treatment", "pycategory_primary")
        pycat_waso = self.model_attributes.get_subsector_attribute("Solid Waste", "pycategory_primary")
        # attribute tables
        attr_gnrl = self.model_attributes.dict_attributes[pycat_gnrl]
        attr_trww = self.model_attributes.dict_attributes[pycat_trww]
        attr_waso = self.model_attributes.dict_attributes[pycat_waso]


        ##  ECON/GNRL VECTOR AND ARRAY INITIALIZATION

        # get some vectors
        vec_gdp = self.model_attributes.get_standard_variables(df_ce_trajectories, self.model_socioeconomic.modvar_econ_gdp, False, return_type = "array_base")
        vec_pop = self.model_attributes.get_standard_variables(df_ce_trajectories, self.model_socioeconomic.modvar_gnrl_pop_total, False, return_type = "array_base")
        array_pop = self.model_attributes.get_standard_variables(df_ce_trajectories, self.model_socioeconomic.modvar_gnrl_subpop, False, return_type = "array_base")
        vec_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_gdp_per_capita"])
        vec_rates_gdp = np.array(df_se_internal_shared_variables["vec_rates_gdp"].dropna())
        vec_rates_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_rates_gdp_per_capita"].dropna())


        ##  OUTPUT INITIALIZATION

        df_out = [df_ce_trajectories[self.required_dimensions].copy()]



        ######################
        #    SOLID WASTE     #
        ######################

        ##  estimate total waste generated by stream (dom + ind) -- keep everything in tonnes

        # municipal components
        factor_waso_init_pc_waste = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_waso_init_msw_generated_pc, False, return_type = "array_base")[0]
        vec_waso_init_msw_composition = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_waso_init_composition_msw, True, return_type = "array_base")[0]
        array_waso_elasticity_waste_prod = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_waso_elast_msw, False, return_type = "array_base")
        array_waso_growth_msw_by_cat = sf.project_growth_scalar_from_elasticity(vec_rates_gdp_per_capita, array_waso_elasticity_waste_prod, False, "standard")
        array_waso_scale_msw = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_waso_waste_per_capita_scalar, False, return_type = "array_base")
        # estimate total waste in each category
        array_waso_msw_total_by_category = np.outer(factor_waso_init_pc_waste*vec_pop, vec_waso_init_msw_composition)
        array_waso_msw_total_by_category *= array_waso_growth_msw_by_cat*array_waso_scale_msw
        # then, check for sludge in input dataframe
        array_waso_sludge = self.model_attributes.get_optional_or_integrated_standard_variable(df_ce_trajectories, self.modvar_trww_sludge_produced, None, True, "array_base")
        if array_waso_sludge:
            # convert to total sludge, then get the correct cateogry and add (should be a unique sludge category)
            array_waso_sludge = np.sum(array_waso_sludge[1], axis = 1)
            cat_sludge = self.model_attributes.get_categories_from_attribute_characteristic("Solid Waste", {"sewage_sludge_category": 1})
            # if a category is defined, add to the solid waste table
            if len(cat_sludge) > 0:
                cat_sludge = cat_sludge[0]
                ind = attr_waso.get_key_value_index(cat_sludge)
                # multiply by factor to ensure that sludge units are in the same as msw
                array_waso_sludge *= self.model_attributes.get_mass_equivalent(
                    self.model_attributes.get_variable_characteristic(self.modvar_trww_sludge_produced, "$UNIT-MASS$"),
                    self.model_attributes.get_variable_characteristic(self.modvar_waso_init_msw_generated_pc, "$UNIT-MASS$")
                )
                array_waso_msw_total_by_category[:, ind] += array_waso_sludge

        # industrial - include multiplication by factor to write industrial waste in same units as msw
        vec_waso_init_pgdp_waste = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_waso_init_isw_generated_pgdp, False, return_type = "array_base")
        vec_waso_init_pgdp_waste *= self.model_attributes.get_mass_equivalent(
            self.model_attributes.get_variable_characteristic(self.modvar_waso_init_isw_generated_pgdp, "$UNIT-MASS$"),
            self.model_attributes.get_variable_characteristic(self.modvar_waso_init_msw_generated_pc, "$UNIT-MASS$")
        )
        vec_waso_isw_composition = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_waso_composition_isw, True, return_type = "array_base")[0]
        array_waso_isw_total_by_category = np.outer(vec_waso_init_pgdp_waste*vec_gdp, vec_waso_isw_composition)
        # initialize total waste array, which will be reduced through recylcing and composting before being divided up between incineration, landfilling, and open dumping
        array_waso_total_by_category = array_waso_isw_total_by_category + array_waso_msw_total_by_category
        df_waso_total_produced_by_category = self.model_attributes.array_to_df(array_waso_total_by_category, self.modvar_waso_waste_total_produced, False)
        df_out += [df_waso_total_produced_by_category]


        ##  Recylcing and Compostic/Anaerobic Treatment for Biogas - assume categories for recycling and composition are mutually exclusive, allowing us to subtract successive values from array_waso_total_by_category

        # estimate total waste recycled
        array_waso_waste_recycled = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_waso_frac_recycled, False, return_type = "array_base", var_bounds = (0, 1))
        array_waso_waste_recycled = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(array_waso_waste_recycled, self.modvar_waso_frac_recycled)
        array_waso_waste_recycled *= array_waso_total_by_category
        array_waso_total_by_category -= array_waso_waste_recycled
        # initialize arrays for compost and biogas, but ensure their totals do not exceed 1
        dict_waso_comp_biogas_check = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_ce_trajectories,
            [self.modvar_waso_frac_compost, self.modvar_waso_frac_biogas],
            1
        )
        array_waso_waste_compost = dict_waso_comp_biogas_check[self.modvar_waso_frac_compost]
        array_waso_waste_biogas = dict_waso_comp_biogas_check[self.modvar_waso_frac_biogas]
        # estimate total waste to compost
        array_waso_waste_compost = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(array_waso_waste_compost, self.modvar_waso_frac_compost)
        array_waso_waste_compost *= array_waso_total_by_category
        # estimate total waste to anaerobic treatment
        array_waso_waste_biogas = self.model_attributes.merge_array_var_partial_cat_to_array_all_cats(array_waso_waste_biogas, self.modvar_waso_frac_biogas)
        array_waso_waste_biogas *= array_waso_total_by_category
        array_waso_total_by_category -= (array_waso_waste_biogas + array_waso_waste_compost)
        # estimate ch4 emissions from composting/biogas
        array_waso_ef_ch4_composting = self.model_attributes.get_standard_variables(df_ce_trajectories, self.modvar_waso_ef_ch4_compost, False, return_type = "array_base")



        self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_ce_trajectories,
            [self.modvar_waso_frac_nonrecycled_incinerated, self.modvar_waso_frac_nonrecycled_landfilled, self.modvar_waso_frac_nonrecycled_opendump],
            1
        )

        # get gas collection total

        ##  estimate emissions+totals from non-collection


        # estimate emissions+totals from incineration
        # get total associated with energy



        # estimate emissions+totals from landfilling
        #

        # add waste totals to df out
        df_waso_waste_biogas = self.model_attributes.array_to_df(array_waso_waste_biogas, self.modvar_waso_waste_total_biogas, False, True)
        df_waso_waste_compost = self.model_attributes.array_to_df(array_waso_waste_compost, self.modvar_waso_waste_total_compost, False, True)
        df_waso_waste_recycled = self.model_attributes.array_to_df(array_waso_waste_recycled, self.modvar_waso_waste_total_recycled, False, True)

        df_out += [
            df_waso_waste_biogas,
            df_waso_waste_compost,
            df_waso_waste_recycled
        ]

        df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)

        return df_out
