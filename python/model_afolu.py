import support_functions as sf
import data_structures as ds
from model_socioeconomic import Socioeconomic
import pandas as pd
import numpy as np
import time

##########################
###                    ###
###     AFOLU MODEL    ###
###                    ###
##########################

class AFOLU:

    def __init__(self, attributes: ds.ModelAttributes):

        # some subector reference variables
        self.subsec_name_agrc = "Agriculture"
        self.subsec_name_econ = "Economy"
        self.subsec_name_frst = "Forest"
        self.subsec_name_gnrl = "General"
        self.subsec_name_lndu = "Land Use"
        self.subsec_name_lsmm = "Livestock Manure Management"
        self.subsec_name_lvst = "Livestock"
        self.subsec_name_soil = "Soil Management"

        # initialzie dynamic variables
        self.model_attributes = attributes
        self.required_dimensions = self.get_required_dimensions()
        self.required_subsectors, self.required_base_subsectors = self.get_required_subsectors()
        self.required_variables, self.output_variables = self.get_afolu_input_output_fields()


        ##  SET MODEL FIELDS

        # agricultural model variables
        self.modvar_agrc_area_prop_calc = "Cropland Area Proportion"
        self.modvar_agrc_area_prop_init = "Initial Cropland Area Proportion"
        self.modvar_agrc_area_crop = "Crop Area"
        self.modvar_agrc_ef_ch4 = ":math:\\text{CH}_4 Crop Anaerobic Decomposition Emission Factor"
        self.modvar_agrc_ef_co2_biomass = ":math:\\text{CO}_2 Crop Biomass Emission Factor"
        self.modvar_agrc_ef_co2_soil_carbon = ":math:\\text{CO}_2 Crop Soil Carbon Emission Factor"
        self.modvar_agrc_ef_n2o_burning = ":math:\\text{N}_2\\text{O} Crop Biomass Burning Emission Factor"
        self.modvar_agrc_ef_n2o_fertilizer = ":math:\\text{N}_2\\text{O} Crop Fertilizer and Lime Emission Factor"
        self.modvar_agrc_elas_crop_demand_income = "Crop Demand Income Elasticity"
        self.modvar_agrc_emissions_ch4_crops = ":math:\\text{CH}_4 Emissions from Crop Activity"
        self.modvar_agrc_emissions_co2_crops = ":math:\\text{CO}_2 Emissions from Crop Activity"
        self.modvar_agrc_emissions_n2o_crops = ":math:\\text{N}_2\\text{O} Emissions from Crop Activity"
        self.modvar_agrc_frac_animal_feed = "Crop Fraction Animal Feed"
        self.modvar_agrc_net_imports = "Change to Net Imports of Crops"
        self.modvar_agrc_yf = "Crop Yield Factor"
        self.modvar_agrc_yield = "Crop Yield"

        # forest model variables
        self.modvar_frst_elas_wood_demand = "Elasticity of Wood Products Demand to Value Added"
        self.modvar_frst_ef_fires = "Forest Fire Emission Factor"
        self.modvar_frst_ef_ch4 = "Forest Methane Emissions"
        self.modvar_frst_emissions_sequestration = ":math:\\text{CO}_2 Emissions from Forest Sequestration"
        self.modvar_frst_emissions_methane = ":math:\\text{CH}_4 Emissions from Forests"
        self.modvar_frst_sq_co2 = "Forest Sequestration Emission Factor"

        # land use model variables
        self.modvar_lndu_area_by_cat = "Land Use Area"
        self.modvar_lndu_area_converted_from_type = "Area of Land Use Area Conversion Away from Type"
        self.modvar_lndu_area_converted_to_type = "Area of Land Use Area Conversion To Type"
        self.modvar_lndu_ef_co2_conv = ":math:\\text{CO}_2 Land Use Conversion Emission Factor"
        self.modvar_lndu_emissions_conv = ":math:\\text{CO}_2 Emissions from Land Use Conversion"
        self.modvar_lndu_emissions_ch4_from_wetlands = ":math:\\text{CH}_4 Emissions from Wetlands"
        self.modvar_lndu_emissions_n2o_from_pastures = ":math:\\text{N}_2\\text{O} Emissions from Pastures"
        self.modvar_lndu_emissions_co2_from_pastures = ":math:\\text{CO}_2 Emissions from Pastures"
        self.modvar_lndu_initial_frac = "Initial Land Use Area Proportion"
        self.modvar_lndu_ef_ch4_boc = "Land Use BOC :math:\\text{CH}_4 Emission Factor"
        self.modvar_lndu_ef_co2_soilcarb = "Land Use Soil Carbon :math:\\text{CO}_2 Emission Factor"
        self.modvar_lndu_prob_transition = "Unadjusted Land Use Transition Probability"
        self.modvar_lndu_reallocation_factor = "Land Use Yield Reallocation Factor"
        self.modvar_lndu_vdes = "Vegetarian Diet Exchange Scalar"
        # some key categories
        self.cat_lndu_crop = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_lndu, {"crop_category": 1})[0]
        self.cat_lndu_pstr = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_lndu, {"pasture_category": 1})[0]

        # livestock model variables
        self.modvar_lvst_animal_weight = "Animal Weight"
        self.modvar_lvst_carrying_capacity_scalar = "Carrying Capacity Scalar"
        self.modvar_lvst_dry_matter_consumption = "Daily Dry Matter Consumption"
        self.modvar_lvst_ef_ch4_ef = ":math:\\text{CH}_4 Enteric Fermentation Emission Factor"
        self.modvar_lvst_elas_lvst_demand = "Elasticity of Livestock Demand to GDP per Capita"
        self.modvar_lvst_emissions_ch4_ef = ":math:\\text{CH}_4 Emissions from Livestock Enteric Fermentation"
        self.modvar_lvst_frac_exc_n_in_dung = "Fraction Nitrogen Excretion in Dung"
        self.modvar_lvst_frac_mm_anaerobic_digester = "Livestock Manure Management Fraction Anaerobic Digester"
        self.modvar_lvst_frac_mm_anaerobic_lagoon = "Livestock Manure Management Fraction Anaerobic Lagoon"
        self.modvar_lvst_frac_mm_composting = "Livestock Manure Management Fraction Composting"
        self.modvar_lvst_frac_mm_daily_spread = "Livestock Manure Management Fraction Daily Spread"
        self.modvar_lvst_frac_mm_deep_bedding = "Livestock Manure Management Fraction Deep Bedding"
        self.modvar_lvst_frac_mm_dry_lot = "Livestock Manure Management Fraction Dry Lot"
        self.modvar_lvst_frac_mm_incineration = "Livestock Manure Management Fraction Incineration"
        self.modvar_lvst_frac_mm_liquid_slurry = "Livestock Manure Management Fraction Liquid Slurry"
        self.modvar_lvst_frac_mm_poultry_manure = "Livestock Manure Management Fraction Poultry Manure"
        self.modvar_lvst_frac_mm_ppr = "Livestock Manure Management Fraction Paddock Pasture Range"
        self.modvar_lvst_frac_mm_solid_storage = "Livestock Manure Management Fraction Solid Storage"
        self.modvar_lvst_genfactor_nitrogen = "Daily Nitrogen Generation Factor"
        self.modvar_lvst_genfactor_volatile_solids = "Daily Volatile Solid Generation Factor"
        self.modvar_lvst_b0_manure_ch4 = "Maximum Manure :math:\\text{CH}_4 Generation Capacity"
        self.modvar_lvst_net_imports = "Change to Net Imports of Livestock"
        self.modvar_lvst_pop = "Livestock Head Count"
        self.modvar_lvst_pop_init = "Initial Livestock Head Count"
        self.modvar_lvst_total_animal_mass = "Total Domestic Animal Mass"
        self.modvar_list_lvst_mm_fractions = [
            self.modvar_lvst_frac_mm_anaerobic_digester,
            self.modvar_lvst_frac_mm_anaerobic_lagoon,
            self.modvar_lvst_frac_mm_composting,
            self.modvar_lvst_frac_mm_daily_spread,
            self.modvar_lvst_frac_mm_deep_bedding,
            self.modvar_lvst_frac_mm_dry_lot,
            self.modvar_lvst_frac_mm_incineration,
            self.modvar_lvst_frac_mm_liquid_slurry,
            self.modvar_lvst_frac_mm_poultry_manure,
            self.modvar_lvst_frac_mm_ppr,
            self.modvar_lvst_frac_mm_solid_storage
        ]
        # some categories
        self.cat_lsmm_incineration = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_lsmm, {"incineration_category": 1})[0]
        self.cat_lsmm_pasture = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_lsmm, {"pasture_category": 1})[0]

        # manure management variables
        self.modvar_lsmm_dung_incinerated = "Dung Incinerated"
        self.modvar_lsmm_ef_direct_n2o = ":math:\\text{N}_2\\text{O} Manure Management Emission Factor"
        self.modvar_lsmm_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Manure Management"
        self.modvar_lsmm_emissions_n2o = ":math:\\text{N}_2\\text{O} Emissions from Manure Management"
        self.modvar_lsmm_frac_loss_leaching = "Fraction of Nitrogen Lost to Leaching"
        self.modvar_lsmm_frac_loss_volatilisation = "Fraction of Nitrogen Lost to Volatilisation"
        self.modvar_lsmm_frac_n_available_used = "Fraction of Nitrogen Used in Fertilizer"
        self.modvar_lsmm_mcf_by_pathway = "Manure Management Methane Correction Factor"
        self.modvar_lsmm_n_from_bedding = "Nitrogen from Bedding per Animal"
        self.modvar_lsmm_n_from_codigestates = "Nitrogen from Co-Digestates Factor"
        self.modvar_lsmm_n_to_pastures = "Total Nitrogen to Pastures"
        self.modvar_lsmm_n_to_fertilizer= "Nitrogen Available for Fertilizer"
        self.modvar_lsmm_n_to_fertilizer_agg_dung = "Total Nitrogen Available for Fertilizer from Dung"
        self.modvar_lsmm_n_to_fertilizer_agg_urea = "Total Nitrogen Available for Fertilizer from Urea"
        self.modvar_lsmm_ratio_n2_to_n2o = "Ratio of :math:\\text{N}_2 to :math:\\text{N}_2\\text{O}"
        self.modvar_lsmm_rf_biogas = "Biogas Recovery Factor at LSMM Anaerobic Digesters"
        self.modvar_lsmm_rf_biogas_recovered = "LSMM Biogas Recovered from Anaerobic Digesters"

        # soil management variables
        self.modvar_soil_ef1_n_managed_soils_rice = ":math:\\text{EF}_{1FR} - \\text{N}_2\\text{O} Rice Fields"
        self.modvar_soil_ef1_n_managed_soils_org_fert = ":math:\\text{EF}_1 - \\text{N}_2\\text{O} Organic Amendments and Fertilizer"
        self.modvar_soil_ef1_n_managed_soils_syn_fert = ":math:\\text{EF}_1 - \\text{N}_2\\text{O} Synthetic Fertilizer"
        self.modvar_soil_ef2_n_organic_soils = ":math:\\text{EF}_2 - \\text{N}_2\\text{O} Emissions from Drained and Managed Organic Soils"
        self.modvar_soil_ef3_n_prp = "EF3 N Pasture Range and Paddock"
        self.modvar_soil_ef4_n_volatilisation = "EF4 N Volatilisation and Re-Deposition Emission Factor"
        self.modvar_soil_ef5_n_leaching = "EF5 N Leaching and Runoff Emission Factor"
        self.modvar_soil_fertuse_nonurea = "Initial Non-Urea Synthetic Fertilizer Use"
        self.modvar_soil_fertuse_urea = "Initial Urea Synthetic Fertilizer Use"
        self.modvar_soil_emissions_co2_lime_urea = ":math:\\text{CO}_2 Emissions from Lime and Urea"
        self.modvar_soil_emissions_fert = ":math:\\text{N}_2\\text{O} Emissions from Fertilizer Use"
        self.modvar_soil_emissions_organic_soils = ":math:\\text{N}_2\\text{O} Emissions from Organic Soils"

        # variable required for integration
        self.integration_variables = self.set_integrated_variables()

        ##  MISCELLANEOUS VARIABLES

        self.time_periods, self.n_time_periods = self.model_attributes.get_time_periods()
        self.factor_c_to_co2 = float(11/3)
        self.factor_n2on_to_n2o = float(11/7)

        # TEMP:SET TO DERIVE FROM ATTRIBUTE TABLES---
        self.cat_lndu_crop = "croplands"
        self.cat_lndu_grazing = "grasslands"

        # add socioeconomic
        self.model_socioeconomic = Socioeconomic(self.model_attributes)


    ##  FUNCTIONS FOR MODEL ATTRIBUTE DIMENSIONS

    def check_df_fields(self, df_afolu_trajectories):
        check_fields = self.required_variables
        # check for required variables
        if not set(check_fields).issubset(df_afolu_trajectories.columns):
            set_missing = list(set(check_fields) - set(df_afolu_trajectories.columns))
            set_missing = sf.format_print_list(set_missing)
            raise KeyError(f"AFOLU projection cannot proceed: The fields {set_missing} are missing.")


    def get_required_subsectors(self):
        subsectors = self.model_attributes.get_sector_subsectors("AFOLU")
        subsectors = [x for x in subsectors if (x not in [self.subsec_name_soil])] #TEMPORARY
        subsectors_base = subsectors.copy()
        subsectors += [self.subsec_name_econ, "General"]
        return subsectors, subsectors_base

    def get_required_dimensions(self):
        ## TEMPORARY - derive from attributes later
        required_doa = [self.model_attributes.dim_time_period]
        return required_doa

    def get_afolu_input_output_fields(self):
        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(self.required_subsectors)
        return required_vars + self.get_required_dimensions(), output_vars

    def set_integrated_variables(self):
        list_vars_required_for_integration = []
        return list_vars_required_for_integration


    ######################################
    #    SUBSECTOR SPECIFIC FUNCTIONS    #
    ######################################


    ###   AGRICULTURE

    def check_cropland_fractions(self, df_in, frac_type = "initial", thresh_for_correction: float = 0.01):

            if frac_type not in ["initial", "calculated"]:
                raise ValueError(f"Error in frac_type '{frac_type}': valid values are 'initial' and 'calculated'.")
            else:
                varname = self.modvar_agrc_area_prop_init if (frac_type == "initial") else self.modvar_agrc_area_prop_calc

            arr = self.model_attributes.get_standard_variables(df_in, varname, True, "array_base")
            totals = sum(arr.transpose())
            m = max(np.abs(totals - 1))

            if m > thresh_for_correction:
                raise ValueError(f"Invalid crop areas found in check_cropland_fractions. The maximum fraction total was {m}; the maximum allowed deviation from 1 is {thresh_for_correction}.")
            else:
                arr = (arr.transpose()/totals).transpose()

            return arr


    ###   LAND USE

    ## apply a scalar to columns or points, then adjust transition probabilites out of a state accordingly. Applying large scales will lead to dominance, and eventually singular values
    def adjust_transition_matrix(self,
        mat: np.ndarray,
        dict_tuples_scale: dict,
        ignore_diag_on_col_scale: bool = False,
        # set max/min for scaled values. Can be used to prevent increasing a single probability to 1
        mat_bounds: tuple = (0, 1),
        response_columns = None
    ) -> np.ndarray:
        """
            dict_tuples_scale has tuples for keys
            - to scale an entire column, enter a single tuple (j, )
            - to scale a point, use (i, j)
            - no way to scale a row--in a row-stochastic matrix, this wouldn't make sense
        """

        # assume that the matrix is square - get the scalar, then get the mask to use adjust transition probabilities not specified as a scalar
        mat_scale = np.ones(mat.shape)
        mat_pos_scale = np.zeros(mat.shape)
        mat_mask = np.ones(mat.shape)

        # assign columns that will be adjusted in response to changes - default to all that aren't scaled
        if (response_columns == None):
            mat_mask_response_nodes = np.ones(mat.shape)
        else:
            mat_mask_response_nodes = np.zeros(mat.shape)
            for col in [x for x in response_columns if x < mat.shape[0]]:
                mat_mask_response_nodes[:, col] = 1

        m = mat_scale.shape[0]


        ##  PERFORM SCALING

        # adjust columns first
        for ind in [x for x in dict_tuples_scale.keys() if len(x) == 1]:
            # overwrite the column
            mat_scale[:, ind[0]] = np.ones(m)*dict_tuples_scale[ind]
            mat_pos_scale[:, ind[0]] = np.ones(m)
            mat_mask[:, ind[0]] = np.zeros(m)
        # it may be of interest to ignore the diagonals when scaling columns
        if ignore_diag_on_col_scale:
            mat_diag = np.diag(tuple(np.ones(m)))
            # reset ones on the diagonal
            mat_scale = (np.ones(mat.shape) - mat_diag)*mat_scale + mat_diag
            mat_pos_scale = sf.vec_bounds(mat_pos_scale - mat_diag, (0, 1))
            mat_mask =  sf.vec_bounds(mat_mask + mat_diag, (0, 1))
        # next, adjust points - operate on the transpose of the matrix
        for ind in [x for x in dict_tuples_scale.keys() if len(x) == 2]:
            mat_scale[ind[0], ind[1]] = dict_tuples_scale[ind]
            mat_pos_scale[ind[0], ind[1]] = 1
            mat_mask[ind[0], ind[1]] = 0


        """
            Get the total that needs to be removed from masked elements (those that are not scaled)

            NOTE: bound scalars at the low end by 0 (if mask_shift_total_i > sums_row_mask_i, then the scalar is negative.
            This occurs if the row total of the adjusted values exceeds 1)
            Set mask_scalar using a minimum value of 0 and implement row normalization—if there's no way to rebalance response columns, everything gets renormalized
            We correct for this below by implementing row normalization to mat_out
        """
        # get new mat and restrict values to 0, 1
        mat_new_scaled = sf.vec_bounds(mat*mat_scale, mat_bounds)
        sums_row = sum(mat_new_scaled.transpose())
        sums_row_mask = sum((mat_mask_response_nodes*mat_mask*mat).transpose())
        # get shift and positive scalar to apply to valid masked elements
        mask_shift_total = sums_row - 1
        mask_scalar = sf.vec_bounds((sums_row_mask - mask_shift_total)/sums_row_mask, (0, np.inf))
        # get the masked nodes, multiply by the response scalar (at applicable columns, denoted by mat_mask_response_nodes), then add to
        mat_out = ((mat_mask_response_nodes*mat_mask*mat).transpose() * mask_scalar).transpose()
        mat_out += sf.vec_bounds(mat_mask*(1 - mat_mask_response_nodes), (0, 1))*mat
        mat_out += mat_pos_scale*mat_new_scaled
        mat_out = (mat_out.transpose()/sum(mat_out.transpose())).transpose()

        return sf.vec_bounds(mat_out, (0, 1))


    ##  check the shape of transition/emission factor matrices sent to project_land_use
    def check_markov_shapes(self, arrs: np.ndarray, function_var_name:str):
            # get land use info
            pycat_lndu = self.model_attributes.get_subsector_attribute(self.subsec_name_lndu, "pycategory_primary")
            attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]

            if len(arrs.shape) < 3:
                raise ValueError(f"Invalid shape for array {function_var_name}; the array must be a list of square matrices.")
            elif arrs.shape[1:3] != (attr_lndu.n_key_values, attr_lndu.n_key_values):
                raise ValueError(f"Invalid shape of matrices in {function_var_name}. They must have shape ({attr_lndu.n_key_values}, {attr_lndu.n_key_values}).")

    ##  get the transition and emission factors matrices from the data frame
    def get_markov_matrices(self,
        df_ordered_trajectories: pd.DataFrame,
        n_tp = None,
        thresh_correct: float = 0.0001
    ) -> tuple:
        """
            - assumes that the input data frame is ordered by time_period
            - n_tp gives the number of time periods. Default value is None, which implies all time periods
            - thresh_correct is used to decide whether or not to correct the transition matrix (assumed to be row stochastic) to sum to 1; if the abs of the sum is outside this range, an error will be thrown
            - fields_pij and fields_efc will be properly ordered by categories for this transformation
        """
        n_tp = n_tp if (n_tp != None) else self.n_time_periods
        fields_pij = self.model_attributes.dict_model_variables_to_variables[self.modvar_lndu_prob_transition]
        fields_efc = self.model_attributes.dict_model_variables_to_variables[self.modvar_lndu_ef_co2_conv]
        sf.check_fields(df_ordered_trajectories, fields_pij + fields_efc)

        pycat_landuse = self.model_attributes.get_subsector_attribute(self.subsec_name_lndu, "pycategory_primary")

        n_categories = len(self.model_attributes.dict_attributes[pycat_landuse].key_values)

        # fetch arrays of transition probabilities and co2 emission factors
        arr_pr = np.array(df_ordered_trajectories[fields_pij])
        arr_pr = arr_pr.reshape((n_tp, n_categories, n_categories))
        arr_ef = np.array(df_ordered_trajectories[fields_efc])
        arr_ef = arr_ef.reshape((n_tp, n_categories, n_categories))

        return arr_pr, arr_ef


    ##  project demand for ag/livestock
    def project_per_capita_demand(self,
        dem_0: np.ndarray, # initial demand (e.g., total yield/livestock produced per acre) ()
        pop: np.ndarray, # population (vec_pop)
        gdp_per_capita_rates: np.ndarray, # driver of demand growth: gdp/capita (vec_rates_gdp_per_capita)
        elast: np.ndarray, # elasticity of demand per capita to growth in gdp/capita (e.g., arr_lvst_elas_demand)
        dem_pc_scalar_exog = None, # exogenous demand per capita scalar representing other changes in the exogenous per-capita demand (can be used to represent population changes)
        return_type: type = float # return type of array
    ) -> np.ndarray:

        # get the demand scalar to apply to per-capita demands
        dem_scale_proj_pc = (gdp_per_capita_rates.transpose()*elast[0:-1].transpose()).transpose()
        dem_scale_proj_pc = np.cumprod(1 + dem_scale_proj_pc, axis = 0)
        dem_scale_proj_pc = np.concatenate([np.ones((1,len(dem_scale_proj_pc[0]))), dem_scale_proj_pc])

        # estimate demand for livestock (used in CBA) - start with livestock population per capita
        if type(dem_pc_scalar_exog) == type(None):
            vec_demscale_exog = np.ones(len(pop))
        else:
            if dem_pc_scalar_exog.shape == pop.shape:
                arr_dem_base = np.outer(pop*dem_pc_scalar_exog, dem_0/pop[0])
            elif dem_pc_scalar_exog.shape == dem_scale_proj_pc.shape:
                arr_pc = (dem_0/pop[0])*dem_pc_scalar_exog
                arr_dem_base = (pop*arr_pc.transpose()).transpose()
            else:
                raise ValueError(f"Invalid shape of dem_pc_scalar_exog: valid shapes are '{pop.shape}' and '{dem_scale_proj_pc.shape}'.")

        # get the total demand
        arr_dem_base = np.array(dem_scale_proj_pc*arr_dem_base).astype(return_type)

        return arr_dem_base


    ##  integrated land use model, which performas required land use transition adjustments
    def project_integrated_land_use(self,
        vec_initial_area: np.ndarray,
        arrs_transitions: np.ndarray,
        arrs_efs: np.ndarray,
        arr_agrc_nonfeeddem_yield: np.ndarray,
        arr_agrc_yield_factors: np.ndarray,
        arr_lndu_yield_by_lvst: np.ndarray,
        arr_lvst_dem: np.ndarray,
        vec_agrc_frac_cropland_area: np.ndarray,
        vec_lndu_yrf: np.ndarray,
        vec_lvst_pop_init: np.ndarray,
        vec_lvst_pstr_weights: np.ndarray,
        vec_lvst_scale_cc: np.ndarray,
        n_tp: int = None
    ) -> tuple:

        t0 = time.time()

        # check shapes
        n_tp = n_tp if (n_tp != None) else self.n_time_periods
        self.check_markov_shapes(arrs_transitions, "arrs_transitions")
        self.check_markov_shapes(arrs_efs, "arrs_efs")

        # get attributes
        pycat_agrc = self.model_attributes.get_subsector_attribute(self.subsec_name_agrc, "pycategory_primary")
        attr_agrc = self.model_attributes.dict_attributes[pycat_agrc]
        pycat_lndu = self.model_attributes.get_subsector_attribute(self.subsec_name_lndu, "pycategory_primary")
        attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]
        pycat_lvst = self.model_attributes.get_subsector_attribute(self.subsec_name_lvst, "pycategory_primary")
        attr_lvst = self.model_attributes.dict_attributes[pycat_lvst]
        # set some commonly called attributes and indices in arrays
        m = attr_lndu.n_key_values
        ind_crop = attr_lndu.get_key_value_index(self.cat_lndu_crop)
        ind_pstr = attr_lndu.get_key_value_index(self.cat_lndu_pstr)

        # initialize variables
        arr_lvst_dem_gr = np.cumprod(arr_lvst_dem/arr_lvst_dem[0], axis = 0)
        vec_lvst_cc_init = vec_lvst_pop_init/(vec_initial_area[ind_pstr]*vec_lvst_pstr_weights)

        # intilize output arrays, including land use, land converted, emissions, and adjusted transitions
        arr_agrc_frac_cropland = np.array([vec_agrc_frac_cropland_area for k in range(n_tp)])
        arr_agrc_net_import_increase = np.zeros((n_tp, attr_agrc.n_key_values))
        arr_agrc_yield = np.array([(vec_initial_area[ind_crop]*vec_agrc_frac_cropland_area*arr_agrc_yield_factors[0]) for k in range(n_tp)])
        arr_emissions_conv = np.zeros((n_tp, attr_lndu.n_key_values))
        arr_land_use = np.array([vec_initial_area for k in range(n_tp)])
        arr_lvst_net_import_increase = np.zeros((n_tp, attr_lvst.n_key_values))
        arrs_land_conv = np.zeros((n_tp, attr_lndu.n_key_values, attr_lndu.n_key_values))
        arrs_transitions_adj = np.zeros(arrs_transitions.shape)
        arrs_yields_per_livestock = np.array([arr_lndu_yield_by_lvst for k in range(n_tp)])

        # initialize running matrix of land use and iteration index i
        x = vec_initial_area
        i = 0

        while i < n_tp - 1:
            # check emission factor index
            i_ef = i if (i < len(arrs_efs)) else len(arrs_efs) - 1
            if i_ef != i:
                print(f"No emission factor matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.")
            # check transition matrix index
            i_tr = i if (i < len(arrs_transitions)) else len(arrs_transitions) - 1
            if i_tr != i:
                print(f"No transition matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.")

            # calculate the unadjusted land use areas (projected to time step i + 1)
            area_crop_cur = x[ind_crop]
            area_crop_proj = np.dot(x, arrs_transitions[i_tr][:, ind_crop])
            area_pstr_cur = x[ind_pstr]
            area_pstr_proj = np.dot(x, arrs_transitions[i_tr][:, ind_pstr])

            vec_agrc_cropland_area_proj = area_crop_proj*arr_agrc_frac_cropland[i]

            # LIVESTOCK - calculate carrying capacities, demand used for pasture reallocation, and net surplus
            vec_lvst_cc_proj = vec_lvst_scale_cc[i + 1]*vec_lvst_cc_init
            vec_lvst_prod_proj = vec_lvst_cc_proj*area_pstr_proj*vec_lvst_pstr_weights
            vec_lvst_net_surplus = np.nan_to_num(arr_lvst_dem[i + 1] - vec_lvst_prod_proj)
            vec_lvst_reallocation = vec_lvst_net_surplus*vec_lndu_yrf[i + 1] # demand for livestock met by reallocating land
            vec_lvst_net_import_increase = vec_lvst_net_surplus - vec_lvst_reallocation # demand for livestock met by increasing net imports (neg => net exports)

            # calculate required increase in transition probabilities
            area_lndu_pstr_increase = sum(np.nan_to_num(vec_lvst_reallocation/vec_lvst_cc_proj))
            scalar_lndu_pstr = (area_pstr_cur + area_lndu_pstr_increase)/np.dot(x, arrs_transitions[i_tr][:, ind_pstr])

            # AGRICULTURE - calculate demand increase in crops, which is a function of gdp/capita (exogenous) and livestock demand (used for feed)
            vec_agrc_feed_dem_yield = sum((arr_lndu_yield_by_lvst*arr_lvst_dem_gr[i + 1]).transpose())
            vec_agrc_total_dem_yield = (arr_agrc_nonfeeddem_yield[i + 1] + vec_agrc_feed_dem_yield)
            vec_agrc_dem_cropareas = vec_agrc_total_dem_yield/arr_agrc_yield_factors[i + 1]
            vec_agrc_net_surplus_cropland_area_cur = vec_agrc_dem_cropareas - vec_agrc_cropland_area_proj
            vec_agrc_reallocation = vec_agrc_net_surplus_cropland_area_cur*vec_lndu_yrf[i + 1]
            # get surplus yield (increase to net imports)
            vec_agrc_net_imports_increase_yield = (vec_agrc_net_surplus_cropland_area_cur - vec_agrc_reallocation)*arr_agrc_yield_factors[i + 1]
            vec_agrc_cropareas_adj = vec_agrc_cropland_area_proj + vec_agrc_reallocation
            scalar_lndu_crop = sum(vec_agrc_cropareas_adj)/np.dot(x, arrs_transitions[i_tr][:, ind_crop])

            # adjust the transition matrix
            trans_adj = self.adjust_transition_matrix(arrs_transitions[i_tr], {(ind_pstr, ): scalar_lndu_pstr, (ind_crop, ): scalar_lndu_crop})
            # calculate final land conversion and emissions
            arr_land_conv = (trans_adj.transpose()*x.transpose()).transpose()
            vec_emissions_conv = sum((trans_adj*arrs_efs[i_ef]).transpose()*x.transpose())

            if i + 1 < n_tp:
                # update arrays
                rng_agrc = list(range((i + 1)*attr_agrc.n_key_values, (i + 2)*attr_agrc.n_key_values))
                np.put(arr_agrc_net_import_increase, rng_agrc, np.round(vec_agrc_net_imports_increase_yield), 2)
                np.put(arr_agrc_frac_cropland, rng_agrc, vec_agrc_cropareas_adj/sum(vec_agrc_cropareas_adj))
                np.put(arr_agrc_yield, rng_agrc, vec_agrc_total_dem_yield)
                arr_lvst_net_import_increase[i + 1] = np.round(vec_lvst_net_import_increase).astype(int)

            # non-ag arrays
            rng_put = np.arange((i)*attr_lndu.n_key_values, (i + 1)*attr_lndu.n_key_values)
            np.put(arr_land_use, rng_put, x)
            np.put(arr_emissions_conv, rng_put, vec_emissions_conv)

            arrs_land_conv[i] = arr_land_conv
            arrs_transitions_adj[i] = trans_adj

            # update land use vector and iterate
            x = np.matmul(x, trans_adj)
            i += 1

        # add on final time step
        trans_adj = arrs_transitions[len(arrs_transitions) - 1]
        # calculate final land conversion and emissions
        arr_land_conv = (trans_adj.transpose()*x.transpose()).transpose()
        vec_emissions_conv = sum((trans_adj*arrs_efs[len(arrs_efs) - 1]).transpose()*x.transpose())
        # add to tables
        rng_put = np.arange((i)*attr_lndu.n_key_values, (i + 1)*attr_lndu.n_key_values)
        np.put(arr_land_use, rng_put, x)
        np.put(arr_emissions_conv, rng_put, vec_emissions_conv)
        arrs_land_conv[i] = arr_land_conv
        arrs_transitions_adj[i] = trans_adj

        return (
            arr_agrc_frac_cropland,
            arr_agrc_net_import_increase,
            arr_agrc_yield,
            arr_emissions_conv,
            arr_land_use,
            arr_lvst_net_import_increase,
            arrs_land_conv,
            arrs_transitions_adj,
            arrs_yields_per_livestock
        )


    ##  project land use
    def project_land_use(
        self,
        vec_initial_area: np.ndarray,
        arrs_transitions: np.ndarray,
        arrs_efs: np.ndarray,
        n_tp: int = None
    ) -> tuple:

        t0 = time.time()

        np.seterr(divide = "ignore", invalid = "ignore")

        # check shapes
        n_tp = n_tp if (n_tp != None) else self.n_time_periods
        self.check_markov_shapes(arrs_transitions, "arrs_transitions")
        self.check_markov_shapes(arrs_efs, "arrs_efs")

        # get land use info
        pycat_lndu = self.model_attributes.get_subsector_attribute(self.subsec_name_lndu, "pycategory_primary")
        attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]

        # intilize the land use and conversion emissions array
        shp_init = (n_tp, attr_lndu.n_key_values)
        arr_land_use = np.zeros(shp_init)
        arr_emissions_conv = np.zeros(shp_init)
        arrs_land_conv = np.zeros((n_tp, attr_lndu.n_key_values, attr_lndu.n_key_values))

        # initialize running matrix of land use and iteration index i
        x = vec_initial_area
        i = 0

        while i < n_tp:
            # check emission factor index
            i_ef = i if (i < len(arrs_efs)) else len(arrs_efs) - 1
            if i_ef != i:
                print(f"No emission factor matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.")
            # check transition matrix index
            i_tr = i if (i < len(arrs_transitions)) else len(arrs_transitions) - 1
            if i_tr != i:
                print(f"No transition matrix found for time period {self.time_periods[i]}; using the matrix from period {len(arrs_efs) - 1}.")
            # calculate land use, conversions, and emissions
            vec_emissions_conv = sum((arrs_transitions[i_tr] * arrs_efs[i_ef]).transpose()*x.transpose())
            arr_land_conv = (arrs_transitions[i_tr].transpose()*x.transpose()).transpose()
            # update matrices
            rng_put = np.arange(i*attr_lndu.n_key_values, (i + 1)*attr_lndu.n_key_values)
            np.put(arr_land_use, rng_put, x)
            np.put(arr_emissions_conv, rng_put, vec_emissions_conv)
            np.put(arrs_land_conv, np.arange(i*attr_lndu.n_key_values**2, (i + 1)*attr_lndu.n_key_values**2), arr_land_conv)
            # update land use vector
            x = np.matmul(x, arrs_transitions[i_tr])

            i += 1

        t1 = time.time()
        t_elapse = round(t1 - t0, 2)
        print(f"Land use projection complete in {t_elapse} seconds.")

        return arr_emissions_conv, arr_land_use, arrs_land_conv


    ##  LIVESTOCK

    def reassign_pops_from_proj_to_carry(self, arr_lu_derived, arr_dem_based):
        """
            Before assigning net imports, there are many non-grazing animals to consider (note that these animals are generally not emission-intensive animals)
            Due to 0 graze area, their estimated population is infinite, or stored as a negative
            We assign their population as the demand-estimated population
        """
        if arr_lu_derived.shape != arr_dem_based.shape:
            raise ValueError(f"Error in reassign_pops_from_proj_to_carry: array dimensions do not match: arr_lu_derived = {arr_lu_derived.shape}, arr_dem_based = {arr_dem_based.shape}.")

        cols = np.where(arr_lu_derived[0] < 0)[0]
        n_row, n_col = arr_lu_derived.shape

        for w in cols:
            rng = np.arange(w*n_row, (w + 1)*n_row)
            np.put(arr_lu_derived.transpose(), rng, arr_dem_based[:, w])

        return arr_lu_derived



    ####################################
    ###                              ###
    ###    PRIMARY MODEL FUNCTION    ###
    ###                              ###
    ####################################

    def project(self, df_afolu_trajectories: pd.DataFrame) -> pd.DataFrame:

        """
            The project() method takes a data frame of input variables (ordered by time series) and returns a data frame of output variables (model projections for agriculture and livestock, forestry, and land use) the same order.

            Function Arguments
            ------------------
            df_afolu_trajectories: pd.DataFrame with all required input fields as columns. The model will not run if any required variables are missing, but errors will detail which fields are missing.


            Notes
            -----
            - The .project() method is designed to be parallelized or called from command line via __main__ in run_sector_models.py.
            - df_afolu_trajectories should have all input fields required (see AFOLU.required_variables for a list of variables to be defined)
            - the df_afolu_trajectories.project method will run on valid time periods from 1 .. k, where k <= n (n is the number of time periods). By default, it drops invalid time periods. If there are missing time_periods between the first and maximum, data are interpolated.
        """

        ##  CHECKS

        # make sure socioeconomic variables are added and
        df_afolu_trajectories, df_se_internal_shared_variables = self.model_socioeconomic.project(df_afolu_trajectories)
        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_afolu_trajectories)
        dict_dims, df_afolu_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_afolu_trajectories, True, True, True)


        ##  CATEGORY INITIALIZATION
        pycat_agrc = self.model_attributes.get_subsector_attribute(self.subsec_name_agrc, "pycategory_primary")
        pycat_frst = self.model_attributes.get_subsector_attribute(self.subsec_name_frst, "pycategory_primary")
        pycat_lndu = self.model_attributes.get_subsector_attribute(self.subsec_name_lndu, "pycategory_primary")
        pycat_lsmm = self.model_attributes.get_subsector_attribute(self.subsec_name_lsmm, "pycategory_primary")
        pycat_lvst = self.model_attributes.get_subsector_attribute(self.subsec_name_lvst, "pycategory_primary")
        # attribute tables
        attr_agrc = self.model_attributes.dict_attributes[pycat_agrc]
        attr_frst = self.model_attributes.dict_attributes[pycat_frst]
        attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]
        attr_lsmm = self.model_attributes.dict_attributes[pycat_lsmm]
        attr_lvst = self.model_attributes.dict_attributes[pycat_lvst]


        ##  ECON/GNRL VECTOR AND ARRAY INITIALIZATION

        # get some vectors
        vec_gdp = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.model_socioeconomic.modvar_econ_gdp, False, return_type = "array_base")
        vec_pop = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.model_socioeconomic.modvar_gnrl_pop_total, False, return_type = "array_base")
        vec_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_gdp_per_capita"])
        vec_rates_gdp = np.array(df_se_internal_shared_variables["vec_rates_gdp"].dropna())
        vec_rates_gdp_per_capita = np.array(df_se_internal_shared_variables["vec_rates_gdp_per_capita"].dropna())


        ##  OUTPUT INITIALIZATION

        df_out = [df_afolu_trajectories[self.required_dimensions].copy()]


        ########################################
        #    LAND USE - UNADJUSTED VARIABLES   #
        ########################################

        # area of the country + the applicable scalar used to convert outputs
        area = float(self.model_attributes.get_standard_variables(df_afolu_trajectories, self.model_socioeconomic.modvar_gnrl_area, return_type = "array_base")[0])
        scalar_lndu_input_area_to_output_area = self.model_attributes.get_scalar(self.model_socioeconomic.modvar_gnrl_area,"area")
        # get the initial distribution of land
        vec_modvar_lndu_initial_frac = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lndu_initial_frac, return_type = "array_base")[0]
        vec_modvar_lndu_initial_area = vec_modvar_lndu_initial_frac*area
        self.vec_modvar_lndu_initial_area = vec_modvar_lndu_initial_area
        self.mat_trans_unadj, self.mat_ef = self.get_markov_matrices(df_afolu_trajectories, n_projection_time_periods)
        # factor for reallocating land in adjustment
        vec_lndu_reallocation_factor = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lndu_reallocation_factor, False, "array_base")
        # common indices
        cat_lndu_ind_crop = attr_lndu.get_key_value_index(self.cat_lndu_crop)
        cat_lndu_ind_pstr = attr_lndu.get_key_value_index(self.cat_lndu_pstr)


        ###########################
        #    CALCULATE DEMANDS    #
        ###########################

        ##  livestock demands (calculated exogenously)

        # variables requried to estimate demand
        vec_modvar_lvst_pop_init = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_pop_init, True, "array_base")[0]
        fields_lvst_elas = self.model_attributes.switch_variable_category(self.subsec_name_lvst, self.modvar_lvst_elas_lvst_demand, "demand_elasticity_category")
        arr_lvst_elas_demand = np.array(df_afolu_trajectories[fields_lvst_elas])
        # get the "vegetarian" factor and use to estimate livestock pop
        vec_lvst_demscale = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.model_socioeconomic.modvar_gnrl_frac_eating_red_meat, False, "array_base", var_bounds = (0, np.inf))
        arr_lvst_dem_pop = self.project_per_capita_demand(vec_modvar_lvst_pop_init, vec_pop, vec_rates_gdp_per_capita, arr_lvst_elas_demand, vec_lvst_demscale, int)
        # get weights for allocating grazing area and feed requirement to animals - based on first year only
        vec_lvst_base_graze_weights = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_dry_matter_consumption, True, "array_base")[0]
        vec_lvst_feed_allocation_weights = (vec_modvar_lvst_pop_init*vec_lvst_base_graze_weights)/np.dot(vec_modvar_lvst_pop_init, vec_lvst_base_graze_weights)
        # get information used to calculate carrying capacity of land
        vec_lvst_carry_capacity_scale = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_carrying_capacity_scalar, False, "array_base", var_bounds = (0, np.inf))


        ##  agricultural demands

        # variables required for demand
        arr_agrc_elas_crop_demand = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_elas_crop_demand_income, False, "array_base")
        arr_agrc_frac_feed = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_frac_animal_feed, False, "array_base")
        # get initial cropland area
        area_agrc_cropland_init = area*vec_modvar_lndu_initial_frac[cat_lndu_ind_crop]
        vec_agrc_frac_cropland_area = self.check_cropland_fractions(df_afolu_trajectories, "initial")[0]
        vec_agrc_cropland_area = area_agrc_cropland_init*vec_agrc_frac_cropland_area
        # get initial yield
        arr_agrc_yf = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_yf, True, "array_base")
        arr_agrc_yf *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_agrc_yf,
            self.model_socioeconomic.modvar_gnrl_area,
            "area"
        )
        vec_agrc_yield_init = arr_agrc_yf[0]*vec_agrc_cropland_area
        # split into yield for livestock feed (responsive to changes in livestock population) and yield for consumption and export (nonlvstfeed)
        vec_agrc_yield_init_lvstfeed = vec_agrc_yield_init*arr_agrc_frac_feed[0]
        vec_agrc_yield_init_nonlvstfeed = vec_agrc_yield_init - vec_agrc_yield_init_lvstfeed
        # project ag demand for crops that are driven by gdp/capita - set demand scalar for crop demand (increases based on reduction in red meat demand) - depends on how many people eat red meat (vec_lvst_demscale)
        vec_agrc_diet_exchange_scalar = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lndu_vdes, False, "array_base", var_bounds = (0, np.inf))
        vec_agrc_demscale = vec_lvst_demscale + vec_agrc_diet_exchange_scalar - vec_lvst_demscale*vec_agrc_diet_exchange_scalar
        # get categories that need to be scaled
        vec_agrc_scale_demands_for_veg = np.array(self.model_attributes.get_ordered_category_attribute(self.subsec_name_agrc, "apply_vegetarian_exchange_scalar"))
        arr_agrc_demscale = np.outer(vec_agrc_demscale, vec_agrc_scale_demands_for_veg)
        arr_agrc_demscale = arr_agrc_demscale + np.outer(np.ones(len(vec_agrc_demscale)), 1 - vec_agrc_scale_demands_for_veg)
        arr_agrc_nonfeeddem_yield = self.project_per_capita_demand(vec_agrc_yield_init_nonlvstfeed, vec_pop, vec_rates_gdp_per_capita, arr_agrc_elas_crop_demand, arr_agrc_demscale, float)
        # array gives the total yield of crop type i allocated to livestock type j at time 0
        arr_lndu_yield_i_reqd_lvst_j_init = np.outer(vec_agrc_yield_init_lvstfeed, vec_lvst_feed_allocation_weights)


        ################################################
        #    CALCULATE LAND USE + AGRC/LVST DRIVERS    #
        ################################################

        # get land use projections (np arrays) - note, arrs_land_conv returns a list of matrices for troubleshooting
        arr_agrc_frac_cropland, arr_agrc_net_import_increase, arr_agrc_yield, arr_lndu_emissions_conv, arr_land_use, arr_lvst_net_import_increase, arrs_lndu_land_conv, self.mat_trans_adj, self.yields_per_livestock = self.project_integrated_land_use(
            vec_modvar_lndu_initial_area,
            self.mat_trans_unadj,
            self.mat_ef,
            arr_agrc_nonfeeddem_yield,
            arr_agrc_yf,
            arr_lndu_yield_i_reqd_lvst_j_init,
            arr_lvst_dem_pop,
            vec_agrc_frac_cropland_area,
            vec_lndu_reallocation_factor,
            vec_modvar_lvst_pop_init,
            vec_lvst_feed_allocation_weights,
            vec_lvst_carry_capacity_scale,
            n_projection_time_periods
        )
        self.arrs_lndu_land_conv = arrs_lndu_land_conv
        # assign some dfs that are used below in other subsectors
        df_agrc_frac_cropland = self.model_attributes.array_to_df(arr_agrc_frac_cropland, self.modvar_agrc_area_prop_calc)
        df_land_use = self.model_attributes.array_to_df(arr_land_use, self.modvar_lndu_area_by_cat)

        # calculate land use conversions
        arrs_lndu_conv_to = np.array([np.sum(x - np.diag(np.diagonal(x)), axis = 0) for x in arrs_lndu_land_conv])
        arrs_lndu_conv_from = np.array([np.sum(x - np.diag(np.diagonal(x)), axis = 1) for x in arrs_lndu_land_conv])

        # add to output data frame
        df_out += [
            df_agrc_frac_cropland,
            self.model_attributes.array_to_df(arr_agrc_net_import_increase, self.modvar_agrc_net_imports),
            self.model_attributes.array_to_df(arr_lndu_emissions_conv, self.modvar_lndu_emissions_conv, True),
            self.model_attributes.array_to_df(arr_agrc_yield, self.modvar_agrc_yield),
            self.model_attributes.array_to_df(arr_land_use*scalar_lndu_input_area_to_output_area, self.modvar_lndu_area_by_cat),
            self.model_attributes.array_to_df(arrs_lndu_conv_to*scalar_lndu_input_area_to_output_area, self.modvar_lndu_area_converted_to_type),
            self.model_attributes.array_to_df(arrs_lndu_conv_from*scalar_lndu_input_area_to_output_area, self.modvar_lndu_area_converted_from_type),
            self.model_attributes.array_to_df(arr_lvst_net_import_increase, self.modvar_lvst_net_imports)
        ]


        ##  EXISTENCE EMISSIONS FOR OTHER LANDS, INCLUDING AG ACTIVITY ON PASTURES

        # get CO2 emissions from soil carbon in pastures
        arr_lndu_ef_co2_soilcarb = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lndu_ef_co2_soilcarb, True, "array_units_corrected")
        arr_lndu_ef_co2_soilcarb *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_lndu_ef_co2_soilcarb,
            "area"
        )
        arr_lndu_area_co2_soilcarb = np.array(df_land_use[self.model_attributes.build_target_varlist_from_source_varcats(self.modvar_lndu_ef_co2_soilcarb, self.modvar_lndu_area_by_cat)])
        # get CH4 emissions from wetlands
        arr_lndu_ef_ch4_boc = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lndu_ef_ch4_boc, True, "array_units_corrected")
        arr_lndu_ef_ch4_boc *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_lndu_ef_ch4_boc,
            "area"
        )
        arr_lndu_area_ch4_boc = np.array(df_land_use[self.model_attributes.build_target_varlist_from_source_varcats(self.modvar_lndu_ef_ch4_boc, self.modvar_lndu_area_by_cat)])

        df_out += [
            self.model_attributes.array_to_df(arr_lndu_area_co2_soilcarb*arr_lndu_ef_co2_soilcarb, self.modvar_lndu_emissions_co2_from_pastures),
            self.model_attributes.array_to_df(arr_lndu_area_ch4_boc*arr_lndu_ef_ch4_boc, self.modvar_lndu_emissions_ch4_from_wetlands)
        ]



        ##################
        #    FORESTRY    #
        ##################

        # get ordered fields from land use
        fields_lndu_forest_ordered = [self.model_attributes.matchstring_landuse_to_forests + x for x in self.model_attributes.dict_attributes[pycat_frst].key_values]
        arr_area_frst = np.array(df_land_use[self.model_attributes.build_varlist(self.subsec_name_lndu, variable_subsec = self.modvar_lndu_area_by_cat, restrict_to_category_values = fields_lndu_forest_ordered)])
        # get different variables
        arr_frst_ef_sequestration = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_frst_sq_co2, True, "array_units_corrected")
        arr_frst_ef_sequestration *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_frst_sq_co2,
            "area"
        )
        arr_frst_ef_methane = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_frst_ef_ch4, True, "array_units_corrected")
        arr_frst_ef_methane *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_frst_ef_ch4,
            "area"
        )
        # build output variables
        df_out += [
            self.model_attributes.array_to_df(-1*arr_area_frst*arr_frst_ef_sequestration, self.modvar_frst_emissions_sequestration),
            self.model_attributes.array_to_df(arr_area_frst*arr_frst_ef_methane, self.modvar_frst_emissions_methane)
        ]

        ##  NEEDED: FOREST FIRES (ADD HERE)
        ##  NEEDED: WOOD PRODUCTS (ADD HERE)



        #####################
        #    AGRICULTURE    #
        #####################

        # get area of cropland
        field_crop_array = self.model_attributes.build_varlist(self.subsec_name_lndu, variable_subsec = self.modvar_lndu_area_by_cat, restrict_to_category_values = [self.cat_lndu_crop])[0]
        vec_cropland_area = np.array(df_land_use[field_crop_array])
        # fraction of cropland represented by each crop
        arr_agrc_frac_cropland_area = self.check_cropland_fractions(df_agrc_frac_cropland, "calculated")
        arr_agrc_crop_area = (arr_agrc_frac_cropland_area.transpose()*vec_cropland_area.transpose()).transpose()
        # unit-corrected emission factors - ch4
        arr_agrc_ef_ch4 = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ef_ch4, True, "array_units_corrected", expand_to_all_cats = True)
        arr_agrc_ef_ch4 *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_agrc_ef_ch4,
            "area"
        )
        # soil carbon
        arr_agrc_ef_co2_soil_carbon = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ef_co2_soil_carbon, True, "array_units_corrected", expand_to_all_cats = True)
        arr_agrc_ef_co2_soil_carbon *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_agrc_ef_co2_soil_carbon,
            "area"
        )
        # biomass
        arr_agrc_ef_co2_biomass = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ef_co2_biomass, True, "array_units_corrected", expand_to_all_cats = True)
        arr_agrc_ef_co2_biomass *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_agrc_ef_co2_biomass,
            "area"
        )
        # biomass burning n2o
        arr_agrc_ef_n2o = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ef_n2o_burning, True, "array_units_corrected")
        arr_agrc_ef_n2o *= self.model_attributes.get_variable_unit_conversion_factor(
            self.model_socioeconomic.modvar_gnrl_area,
            self.modvar_agrc_ef_n2o_burning,
            "area"
        )

        # add to output dataframe
        df_out += [
            self.model_attributes.array_to_df(arr_agrc_crop_area*scalar_lndu_input_area_to_output_area, self.modvar_agrc_area_crop),
            self.model_attributes.array_to_df(arr_agrc_ef_ch4*arr_agrc_crop_area, self.modvar_agrc_emissions_ch4_crops, reduce_from_all_cats_to_specified_cats = True),
            self.model_attributes.array_to_df((arr_agrc_ef_co2_soil_carbon + arr_agrc_ef_co2_biomass)*arr_agrc_crop_area, self.modvar_agrc_emissions_co2_crops),
            self.model_attributes.array_to_df(arr_agrc_ef_n2o*arr_agrc_crop_area, self.modvar_agrc_emissions_n2o_crops)
        ]



        ###################
        #    LIVESTOCK    #
        ###################

        # get area of grassland/pastures
        field_lvst_graze_array = self.model_attributes.build_varlist(self.subsec_name_lndu, variable_subsec = self.modvar_lndu_area_by_cat, restrict_to_category_values = [self.cat_lndu_grazing])[0]
        vec_lvst_graze_area = np.array(df_land_use[field_lvst_graze_array])
        # estimate the total number of livestock that are raised
        arr_lvst_pop = arr_lvst_dem_pop - arr_lvst_net_import_increase
        arr_lvst_total_weight = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_animal_weight, True, "array_base", expand_to_all_cats = True)
        arr_lvst_total_animal_mass = arr_lvst_pop*arr_lvst_total_weight
        arr_lvst_aggregate_animal_mass = np.sum(arr_lvst_total_animal_mass, axis = 1)
        arr_lvst_aggregate_animal_mass *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lvst_total_animal_mass,
            "mass"
        )
        # get the enteric fermentation emission factor
        arr_lvst_emissions_ch4_ef = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_ef_ch4_ef, True, "array_units_corrected")
        # add to output dataframe
        df_out += [
            self.model_attributes.array_to_df(arr_lvst_emissions_ch4_ef*arr_lvst_pop, self.modvar_lvst_emissions_ch4_ef),
            self.model_attributes.array_to_df(arr_lvst_pop, self.modvar_lvst_pop),
            self.model_attributes.array_to_df(arr_lvst_aggregate_animal_mass, self.modvar_lvst_total_animal_mass)
        ]


        ##  MANURE MANAGEMENT DATA

        # nitrogen and volative solids generated (passed to manure management--unitless, so they take the mass of modvar_lvst_animal_weight)
        arr_lvst_nitrogen = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_genfactor_nitrogen, True, "array_base", expand_to_all_cats = True)
        arr_lvst_nitrogen *= arr_lvst_total_animal_mass*self.model_attributes.configuration.get("days_per_year")
        arr_lvst_volatile_solids = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_genfactor_volatile_solids, True, "array_base", expand_to_all_cats = True)
        arr_lvst_volatile_solids *= arr_lvst_total_animal_mass*self.model_attributes.configuration.get("days_per_year")
        arr_lvst_b0 = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_b0_manure_ch4, True, "array_units_corrected_gas", expand_to_all_cats = True)



        #####################################
        #    LIVESTOCK MANURE MANAGEMENT    #
        #####################################

        # first, retrieve energy fractions and ensure they sum to 1
        dict_arrs_lsmm_frac_manure = self.model_attributes.get_multivariables_with_bounded_sum_by_category(
            df_afolu_trajectories,
            self.modvar_list_lvst_mm_fractions,
            1,
            force_sum_equality = True,
            msg_append = "Energy fractions by category do not sum to 1. See definition of dict_arrs_inen_frac_energy."
        )

        # get variables that can be indexed below
        arr_lsmm_ef_direct_n2o = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lsmm_ef_direct_n2o, True, "array_base", expand_to_all_cats = True)
        arr_lsmm_frac_lost_leaching = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lsmm_frac_loss_leaching, True, "array_base", expand_to_all_cats = True, var_bounds = (0, 1))
        arr_lsmm_frac_lost_volatilisation = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lsmm_frac_loss_volatilisation, True, "array_base", expand_to_all_cats = True, var_bounds = (0, 1))
        arr_lsmm_mcf_by_pathway = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lsmm_mcf_by_pathway, True, "array_base", expand_to_all_cats = True, var_bounds = (0, 1))
        arr_lsmm_n_from_bedding = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lsmm_n_from_bedding, True, "array_base", expand_to_all_cats = True)
        arr_lsmm_n_from_codigestates = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lsmm_n_from_codigestates, True, "array_base", expand_to_all_cats = True, var_bounds = (0, np.inf))
        arr_lsmm_rf_biogas = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lsmm_rf_biogas, True, "array_base", expand_to_all_cats = True, var_bounds = (0, 1))
        vec_lsmm_frac_n_in_dung = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_frac_exc_n_in_dung, False, "array_base", var_bounds = (0, 1))
        vec_lsmm_ratio_n2_to_n2o = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lsmm_ratio_n2_to_n2o, False, "array_base")

        # soil EF4/EF5 from Table 11.3 - use average fractions from grasslands
        arr_soil_ef_ef4 = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_ef4_n_volatilisation, False, "array_base")
        vec_soil_ef_ef4 = np.sum(arr_soil_ef_ef4, axis = 1)/2
        print("TEMPORARY WORK -- ASSUMED 50% DRY/WET--TO GET RUNNING - FIX TO READ FROM VARS!")
        vec_soil_ef_ef5 = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_soil_ef5_n_leaching, False, "array_base")

        # convert bedding/co-digestates to animal weight masses
        arr_lsmm_n_from_bedding *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lsmm_n_from_bedding,
            self.modvar_lvst_animal_weight,
            "mass"
        )

        # initialize output arrays
        arr_lsmm_biogas_recovered = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        arr_lsmm_emission_ch4 = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        arr_lsmm_emission_n2o = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        arr_lsmm_nitrogen_available = np.zeros(arr_lsmm_ef_direct_n2o.shape)
        # initialize some aggregations
        vec_lsmm_nitrogen_to_fertilizer_dung = 0
        vec_lsmm_nitrogen_to_fertilizer_urea = 0

        # loop over manure pathways to
        for var_lvst_mm_frac in self.modvar_list_lvst_mm_fractions:
            # get the current variable
            arr_lsmm_fracs_by_lvst = dict_arrs_lsmm_frac_manure[var_lvst_mm_frac]
            arr_lsmm_total_nitrogen_cur = arr_lvst_nitrogen*arr_lsmm_fracs_by_lvst
            # retrive the livestock management category
            cat_lsmm = ds.clean_schema(self.model_attributes.get_variable_attribute(var_lvst_mm_frac, pycat_lsmm))
            index_cat_lsmm = attr_lsmm.get_key_value_index(cat_lsmm)


            ##  METHANE EMISSIONS

            # get MCF, b0, and total volatile solids - USE EQ. 10.23
            vec_lsmm_mcf_cur = arr_lsmm_mcf_by_pathway[:, index_cat_lsmm]
            arr_lsmm_emissions_ch4_cur = arr_lvst_b0*arr_lvst_volatile_solids*arr_lsmm_fracs_by_lvst
            arr_lsmm_emissions_ch4_cur = (arr_lsmm_emissions_ch4_cur.transpose()*vec_lsmm_mcf_cur).transpose()
            # get biogas recovery
            arr_lsmm_biogas_recovered_cur = (arr_lsmm_emissions_ch4_cur.transpose()*arr_lsmm_rf_biogas[:, index_cat_lsmm]).transpose()
            arr_lsmm_emissions_ch4_cur -= arr_lsmm_biogas_recovered_cur
            arr_lsmm_biogas_recovered[:, index_cat_lsmm] = np.sum(arr_lsmm_biogas_recovered_cur, axis = 1)
            # adjust
            arr_lsmm_emissions_ch4_cur *= self.model_attributes.get_scalar(self.modvar_lvst_animal_weight, "mass")
            arr_lsmm_emission_ch4[:, index_cat_lsmm] = np.sum(arr_lsmm_emissions_ch4_cur, axis = 1)


            ##  NITROGEN EMISSIONS AND FERTILIZER AVAILABILITY

            # get total nitrogen deposited
            vec_lsmm_nitrogen_treated_cur = np.sum(arr_lsmm_total_nitrogen_cur, axis = 1)
            vec_lsmm_n_from_bedding = arr_lsmm_n_from_bedding[:, index_cat_lsmm]
            vec_lsmm_n_from_codigestates = arr_lsmm_n_from_codigestates[:, index_cat_lsmm]
            # get nitrogen from bedding per animal
            vec_lsmm_n_from_bedding *= np.sum(arr_lvst_pop*arr_lsmm_fracs_by_lvst, axis = 1)

            # get totals lost to different pathways
            vec_lsmm_frac_lost_direct = sf.vec_bounds((1 + vec_lsmm_ratio_n2_to_n2o)*arr_lsmm_ef_direct_n2o[:, index_cat_lsmm], (0, 1))
            vec_lsmm_frac_lost_leaching = arr_lsmm_frac_lost_leaching[:, index_cat_lsmm]
            vec_lsmm_frac_lost_volatilisation = arr_lsmm_frac_lost_volatilisation[:, index_cat_lsmm]
            # apply the limiter, which prevents their total from exceeding 1
            vec_lsmm_frac_lost_direct, vec_lsmm_frac_lost_leaching, vec_lsmm_frac_lost_volatilisation = sf.vector_limiter(
                [
                    vec_lsmm_frac_lost_direct,
                    vec_lsmm_frac_lost_leaching,
                    vec_lsmm_frac_lost_volatilisation
                ],
                (0, 1)
            )
            vec_lsmm_frac_loss_ms = vec_lsmm_frac_lost_leaching + vec_lsmm_frac_lost_volatilisation + vec_lsmm_frac_lost_direct
            vec_lsmm_n_lost = vec_lsmm_nitrogen_treated_cur*(1 + vec_lsmm_n_from_codigestates)*self.factor_n2on_to_n2o

            # 10.25 FOR DIRECT EMISSIONS
            arr_lsmm_emission_n2o[:, index_cat_lsmm] = vec_lsmm_n_lost*arr_lsmm_ef_direct_n2o[:, index_cat_lsmm]
            # 10.28 FOR LOSSES DUE TO VOLATILISATION
            arr_lsmm_emission_n2o[:, index_cat_lsmm] += vec_lsmm_n_lost*vec_soil_ef_ef4*vec_lsmm_frac_lost_volatilisation
            # 10.29 FOR LOSSES DUE TO LEACHING
            arr_lsmm_emission_n2o[:, index_cat_lsmm] += vec_lsmm_n_lost*vec_soil_ef_ef5*vec_lsmm_frac_lost_leaching
            # BASED ON EQ. 10.34A in IPCC GNGHGI 2019R FOR NITROGEN AVAILABILITY - note: co-digestates are entered as an inflation factor
            vec_lsmm_nitrogen_available = (vec_lsmm_nitrogen_treated_cur*(1 + vec_lsmm_n_from_codigestates))*(1 - vec_lsmm_frac_loss_ms) + vec_lsmm_n_from_bedding

            if cat_lsmm == self.cat_lsmm_incineration:
                # if incinerating, send urea nitrogen to pasture
                index_cat_lsmm_pasture = attr_lsmm.get_key_value_index(self.cat_lsmm_pasture)
                arr_lsmm_nitrogen_available[:, index_cat_lsmm_pasture] += vec_lsmm_nitrogen_available*(1 - vec_lsmm_frac_n_in_dung)
                # get incinerated dung
                vec_lsmm_nitrogen_to_incinerator = vec_lsmm_nitrogen_available*vec_lsmm_frac_n_in_dung
                vec_lsmm_nitrogen_to_incinerator *= self.model_attributes.get_variable_unit_conversion_factor(
                    self.modvar_lvst_animal_weight,
                    self.modvar_lsmm_dung_incinerated,
                    "mass"
                )
                # add to output
                df_out += [
                    self.model_attributes.array_to_df(vec_lsmm_nitrogen_to_incinerator, self.modvar_lsmm_dung_incinerated)
                ]

            else:
                # add to total by animal and splits by dung/urea (used in Soil Management subsector)
                arr_lsmm_nitrogen_available[:, index_cat_lsmm] += vec_lsmm_nitrogen_available
                vec_lsmm_nitrogen_to_fertilizer_dung += vec_lsmm_nitrogen_available*vec_lsmm_frac_n_in_dung
                vec_lsmm_nitrogen_to_fertilizer_urea += vec_lsmm_nitrogen_available*(1 - vec_lsmm_frac_n_in_dung)

        # get pasture and fertilizer totals
        vec_lsmm_nitrogen_to_pasture = arr_lsmm_nitrogen_available[:, attr_lsmm.get_key_value_index(self.cat_lsmm_pasture)].copy()
        vec_lsmm_nitrogen_to_pasture *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_pastures,
            "mass"
        )
        arr_lsmm_nitrogen_available[:, attr_lsmm.get_key_value_index(self.cat_lsmm_pasture)] = np.zeros(len(arr_lsmm_nitrogen_available))


        ##  UNITS CONVERSTION

        # biogas recovery
        arr_lsmm_biogas_recovered *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_rf_biogas_recovered,
            "mass"
        )
        # total nitrogen available for fertilizer by pathway
        arr_lsmm_nitrogen_available *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_fertilizer,
            "mass"
        )
        # nitrogen available from dung/urea
        vec_lsmm_nitrogen_to_fertilizer_dung *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_fertilizer_agg_dung,
            "mass"
        )
        vec_lsmm_nitrogen_to_fertilizer_urea *= self.model_attributes.get_variable_unit_conversion_factor(
            self.modvar_lvst_animal_weight,
            self.modvar_lsmm_n_to_fertilizer_agg_urea,
            "mass"
        )
        # n2o emissions
        arr_lsmm_emission_n2o *= self.model_attributes.get_scalar(self.modvar_lsmm_emissions_n2o, "gas")
        arr_lsmm_emission_n2o *= self.model_attributes.get_scalar(self.modvar_lvst_animal_weight, "mass")

        df_out += [
            self.model_attributes.array_to_df(arr_lsmm_emission_ch4, self.modvar_lsmm_emissions_ch4),
            self.model_attributes.array_to_df(arr_lsmm_emission_n2o, self.modvar_lsmm_emissions_n2o),
            self.model_attributes.array_to_df(vec_lsmm_nitrogen_to_pasture, self.modvar_lsmm_n_to_pastures),
            self.model_attributes.array_to_df(arr_lsmm_nitrogen_available, self.modvar_lsmm_n_to_fertilizer),
            self.model_attributes.array_to_df(vec_lsmm_nitrogen_to_fertilizer_dung, self.modvar_lsmm_n_to_fertilizer_agg_dung),
            self.model_attributes.array_to_df(vec_lsmm_nitrogen_to_fertilizer_urea, self.modvar_lsmm_n_to_fertilizer_agg_urea),
            self.model_attributes.array_to_df(arr_lsmm_biogas_recovered, self.modvar_lsmm_rf_biogas_recovered, reduce_from_all_cats_to_specified_cats = True)
        ]




        #########################
        #    SOIL MANAGEMENT    #
        #########################

        # get mineral soils based on land use categories converted to cropland
        cats_lndu_to_crop_mineral = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_lndu, {"mineralization_in_land_use_conversion_to_managed": 1})
        inds_lndu_to_crop_mineral = [attr_lndu.get_key_value_index(x) for x in cats_lndu_to_crop_mineral]
        ind_lndu_cropland = attr_lndu.get_key_value_index(self.cat_lndu_crop)
        vec_lndu_mask_crop_mineral = np.zeros(len(attr_lndu.key_values))
        np.put(vec_lndu_mask_crop_mineral, inds_lndu_to_crop_mineral, 1)
        vec_lndu_land_conv_to_cropland_mineral_soils = np.array([np.dot(x[:, ind_lndu_cropland], vec_lndu_mask_crop_mineral) for x in arrs_lndu_land_conv])

        # other soils





        df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)
        self.model_attributes.add_subsector_emissions_aggregates(df_out, self.required_base_subsectors, False)

        return df_out
