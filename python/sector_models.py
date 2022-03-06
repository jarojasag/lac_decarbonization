import support_functions as sf
import data_structures as ds
import pandas as pd
import numpy as np
import time

######################
#     AFOLU MODEL    #
######################

class AFOLU:

    def __init__(self, attributes: ds.ModelAttributes):

        self.model_attributes = attributes
        self.required_dimensions = self.get_required_dimensions()
        self.required_subsectors, self.required_base_subsectors = self.get_required_subsectors()
        self.required_variables, self.output_variables = self.get_afolu_input_output_fields()

        ##  set some model fields to connect to the attribute tables

        # agricultural model variables
        self.modvar_agrc_area_prop_calc = "Cropland Area Proportion"
        self.modvar_agrc_area_prop_init = "Initial Cropland Area Proportion"
        self.modvar_agrc_area_crop = "Crop Area"
        self.modvar_agrc_ef_ch4 = ":math:\\text{CH}_4 Crop Anaerobic Decomposition Emission Factor"
        self.modvar_agrc_ef_co2 = ":math:\\text{CO}_2 Crop Soil Carbon Emission Factor"
        self.modvar_agrc_ef_co2_yield = ":math:\\text{CO}_2 Crop Biomass Emission Factor"
        self.modvar_agrc_ef_n2o = ":math:\\text{N}_2\\text{O} Crop Biomass Burning Emission Factor"
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
        self.modvar_lndu_ef_co2_conv = ":math:\\text{CO}_2 Land Use Conversion Emission Factor"
        self.modvar_lndu_emissions_conv = ":math:\\text{CO}_2 Emissions from Land Use Conversion"
        self.modvar_lndu_emissions_ch4_from_wetlands = ":math:\\text{CH}_4 Emissions from Wetlands"
        self.modvar_lndu_emissions_n2o_from_pastures = ":math:\\text{N}_2\\text{O} Emissions from Pastures"
        self.modvar_lndu_emissions_co2_from_pastures = ":math:\\text{CO}_2 Emissions from Pastures"
        self.modvar_lndu_initial_frac = "Initial Land Use Area Proportion"
        self.modvar_lndu_ef_ch4_boc = "Land Use BOC :math:\\text{CH}_4 Emission Factor"
        self.modvar_lndu_ef_n2o_past = "Land Use Pasture :math:\\text{N}_2\\text{O} Emission Factor"
        self.modvar_lndu_ef_co2_soilcarb = "Land Use Soil Carbon :math:\\text{CO}_2 Emission Factor"
        self.modvar_lndu_prob_transition = "Unadjusted Land Use Transition Probability"
        self.modvar_lndu_reallocation_factor = "Land Use Yield Reallocation Factor"
        self.modvar_lndu_vdes = "Vegetarian Diet Exchange Scalar"
        # livestock model variables
        self.modvar_lvst_carrying_capacity_scalar = "Carrying Capacity Scalar"
        self.modvar_lvst_dry_matter_consumption = "Daily Dry Matter Consumption"
        self.modvar_lvst_ef_ch4_ef = ":math:\\text{CH}_4 Enteric Fermentation Emission Factor"
        self.modvar_lvst_ef_ch4_mm = ":math:\\text{CH}_4 Manure Management Emission Factor"
        self.modvar_lvst_ef_n2o_mm = ":math:\\text{N}_2\\text{O} Manure Management Emission Factor"
        self.modvar_lvst_elas_lvst_demand = "Elasticity of Livestock Demand to GDP per Capita"
        self.modvar_lvst_emissions_ch4_ef = ":math:\\text{CH}_4 Emissions from Livestock Enteric Fermentation"
        self.modvar_lvst_emissions_ch4_mm = ":math:\\text{CH}_4 Emissions from Livestock Manure"
        self.modvar_lvst_emissions_n2o_mm = ":math:\\text{N}_2\\text{O} Emissions from Livestock Manure"
        self.modvar_lvst_frac_eating_red_meat = "Fraction Eating Red Meat"
        self.modvar_lvst_net_imports = "Change to Net Imports of Livestock"
        self.modvar_lvst_pop = "Livestock Head Count"
        self.modvar_lvst_pop_init = "Initial Livestock Head Count"
        # economy and general variables
        self.modvar_econ_gdp = "GDP"
        self.modvar_econ_va = "Value Added"
        self.modvar_gnrl_area = "Area of Country"
        self.modvar_gnrl_occ = "National Occupation Rate"
        self.modvar_gnrl_subpop = "Population"
        self.modvar_gnrl_pop_total = "Total Population"

        ##  MISCELLANEOUS VARIABLES

        self.time_periods, self.n_time_periods = self.get_time_periods()

        # TEMP:SET TO DERIVE FROM ATTRIBUTE TABLES---
        self.cat_lndu_crop = "croplands"
        self.cat_lndu_grazing = "grasslands"


    ##  FUNCTIONS FOR MODEL ATTRIBUTE DIMENSIONS

    def check_df_fields(self, df_afolu_trajectories):
        check_fields = self.required_variables
        # check for required variables
        if not set(check_fields).issubset(df_afolu_trajectories.columns):
            set_missing = list(set(check_fields) - set(df_afolu_trajectories.columns))
            set_missing = sf.format_print_list(set_missing)
            raise KeyError(f"AFOLU projection cannot proceed: The fields {set_missing} are missing.")


    def get_required_subsectors(self):
        subsectors = list(sf.subset_df(self.model_attributes.dict_attributes["abbreviation_subsector"].table, {"sector": ["AFOLU"]})["subsector"])
        subsectors_base = subsectors.copy()
        subsectors += ["Economy", "General"]
        return subsectors, subsectors_base

    def get_required_dimensions(self):
        ## TEMPORARY - derive from attributes later
        required_doa = [self.model_attributes.dim_time_period]
        return required_doa

    def get_afolu_input_output_fields(self):
        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(self.required_subsectors)
        return required_vars + self.get_required_dimensions(), output_vars


    def get_time_periods(self):
        pydim_time_period = self.model_attributes.get_dimensional_attribute("time_period", "pydim")
        time_periods = self.model_attributes.dict_attributes[pydim_time_period].key_values
        return time_periods, len(time_periods)


    ##  STREAMLINING FUNCTIONS

    # loop over a dictionary of simple variables that map an emission factor to a driver within the sector
    def get_simple_input_to_output_emission_arrays(self, df_ef: pd.DataFrame, df_driver: pd.DataFrame, dict_vars: dict, variable_driver: str):
        """
            NOTE: this only works w/in subsector
        """
        df_out = []
        subsector_driver = self.model_attributes.dict_model_variable_to_subsector[variable_driver]

        for var in dict_vars.keys():
            subsector_var = self.model_attributes.dict_model_variable_to_subsector[var]
            if subsector_driver != subsector_driver:
                warnings.warn(f"In get_simple_input_to_output_emission_arrays, driver variable '{variable_driver}' and emission variable '{var}' are in different sectors. This instance will be skipped.")
            else:
                # get emissions factor fields and apply scalar using get_standard_variables
                arr_ef = np.array(self.model_attributes.get_standard_variables(df_ef, var, True, "array_units_corrected"))
                # get the emissions driver array (driver must h)
                arr_driver = np.array(df_driver[self.model_attributes.build_target_varlist_from_source_varcats(var, variable_driver)])
                df_out.append(self.model_attributes.array_to_df(arr_driver*arr_ef, dict_vars[var]))

        return df_out


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
            pycat_lndu = self.model_attributes.get_subsector_attribute("Land Use", "pycategory_primary")
            attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]

            if len(arrs.shape) < 3:
                raise ValueError(f"Invalid shape for array {function_var_name}; the array must be a list of square matrices.")
            elif arrs.shape[1:3] != (attr_lndu.n_key_values, attr_lndu.n_key_values):
                raise ValueError(f"Invalid shape of matrices in {function_var_name}. They must have shape ({attr_lndu.n_key_values}, {attr_lndu.n_key_values}).")

    ##  get the transition and emission factors matrices from the data frame
    def get_markov_matrices(self, df_ordered_trajectories, thresh_correct = 0.0001):
        """
            - assumes that the input data frame is ordered by time_period
            - thresh_correct is used to decide whether or not to correct the transition matrix (assumed to be row stochastic) to sum to 1; if the abs of the sum is outside this range, an error will be thrown
            - fields_pij and fields_efc will be properly ordered by categories for this transformation
        """

        fields_pij = self.model_attributes.dict_model_variables_to_variables[self.modvar_lndu_prob_transition]
        fields_efc = self.model_attributes.dict_model_variables_to_variables[self.modvar_lndu_ef_co2_conv]
        sf.check_fields(df_ordered_trajectories, fields_pij + fields_efc)

        pycat_landuse = self.model_attributes.get_subsector_attribute("Land Use", "pycategory_primary")

        n_categories = len(self.model_attributes.dict_attributes[pycat_landuse].key_values)

        # fetch arrays of transition probabilities and co2 emission factors
        arr_pr = np.array(df_ordered_trajectories[fields_pij])
        arr_pr = arr_pr.reshape((self.n_time_periods, n_categories, n_categories))
        arr_ef = np.array(df_ordered_trajectories[fields_efc])
        arr_ef = arr_ef.reshape((self.n_time_periods, n_categories, n_categories))

        return arr_pr, arr_ef


    ##  project demand for ag/livestock
    def project_per_capita_demand(self,
        dem_0: np.ndarray, # initial demand (e.g., total yield/livestock produced per acre) ()
        pop: np.ndarray, # population (vec_pop)
        gdp_per_capita_rates: np.ndarray, # driver of demand growth: gdp/capita (vec_rates_gdp_per_capita)
        elast: np.ndarray, # elasticity of demand per capita to growth in gdp/capita (e.g., arr_lvst_elas_demand)
        dem_pc_scalar_exog = None, # exogenous demand per capita scalar representing other changes in the exogenous per-capita demand (can be used to represent population changes)
        # self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_frac_eating_red_meat, False, "array_base")
        return_type: type = float # return type of array
    ):

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
        vec_lvst_scale_cc: np.ndarray
    ) -> tuple:

        t0 = time.time()

        # check shapes
        self.check_markov_shapes(arrs_transitions, "arrs_transitions")
        self.check_markov_shapes(arrs_efs, "arrs_efs")

        # get attributes
        pycat_agrc = self.model_attributes.get_subsector_attribute("Agriculture", "pycategory_primary")
        attr_agrc = self.model_attributes.dict_attributes[pycat_agrc]
        pycat_lndu = self.model_attributes.get_subsector_attribute("Land Use", "pycategory_primary")
        attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]
        pycat_lvst = self.model_attributes.get_subsector_attribute("Livestock", "pycategory_primary")
        attr_lvst = self.model_attributes.dict_attributes[pycat_lvst]
        # set some commonly called attributes and indices in arrays
        m = attr_lndu.n_key_values
        ind_crop = attr_lndu.get_key_value_index("croplands")
        ind_pstr = attr_lndu.get_key_value_index("grasslands")

        # initialize variables
        arr_lvst_dem_gr = np.cumprod(arr_lvst_dem/arr_lvst_dem[0], axis = 0)
        vec_lvst_cc_init = vec_lvst_pop_init/(vec_initial_area[ind_pstr]*vec_lvst_pstr_weights)

        # intilize output arrays, including land use, land converted, emissions, and adjusted transitions
        arr_agrc_frac_cropland = np.array([vec_agrc_frac_cropland_area for k in range(self.n_time_periods)])
        arr_agrc_net_import_increase = np.zeros((self.n_time_periods, attr_agrc.n_key_values))
        arr_agrc_yield = np.array([(vec_initial_area[ind_crop]*vec_agrc_frac_cropland_area*arr_agrc_yield_factors[0]) for k in range(self.n_time_periods)])
        arr_emissions_conv = np.zeros((self.n_time_periods, attr_lndu.n_key_values))
        arr_land_use = np.array([vec_initial_area for k in range(self.n_time_periods)])
        arr_lvst_net_import_increase = np.zeros((self.n_time_periods, attr_lvst.n_key_values))
        arrs_land_conv = np.zeros((self.n_time_periods, attr_lndu.n_key_values, attr_lndu.n_key_values))
        arrs_transitions_adj = np.zeros(arrs_transitions.shape)
        arrs_yields_per_livestock = np.array([arr_lndu_yield_by_lvst for k in range(self.n_time_periods)])

        # initialize running matrix of land use and iteration index i
        x = vec_initial_area
        i = 0

        while i < self.n_time_periods - 1:
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
            vec_agrc_total_yield = (arr_agrc_nonfeeddem_yield[i + 1] + vec_agrc_feed_dem_yield)
            vec_agrc_dem_cropareas = vec_agrc_total_yield/arr_agrc_yield_factors[i + 1]
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

            if i + 1 < self.n_time_periods:
                # update arrays
                rng_agrc = list(range((i + 1)*attr_agrc.n_key_values, (i + 2)*attr_agrc.n_key_values))
                np.put(arr_agrc_net_import_increase, rng_agrc, np.round(vec_agrc_net_imports_increase_yield), 2)
                np.put(arr_agrc_frac_cropland, rng_agrc, vec_agrc_cropareas_adj/sum(vec_agrc_cropareas_adj))
                np.put(arr_agrc_yield, rng_agrc, vec_agrc_total_yield)
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
    def project_land_use(self, vec_initial_area: np.ndarray, arrs_transitions: np.ndarray, arrs_efs: np.ndarray):

        t0 = time.time()

        # check shapes
        self.check_markov_shapes(arrs_transitions, "arrs_transitions")
        self.check_markov_shapes(arrs_efs, "arrs_efs")

        # get land use info
        pycat_lndu = self.model_attributes.get_subsector_attribute("Land Use", "pycategory_primary")
        attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]

        # intilize the land use and conversion emissions array
        shp_init = (self.n_time_periods, attr_lndu.n_key_values)
        arr_land_use = np.zeros(shp_init)
        arr_emissions_conv = np.zeros(shp_init)
        arrs_land_conv = np.zeros((self.n_time_periods, attr_lndu.n_key_values, attr_lndu.n_key_values))

        # initialize running matrix of land use and iteration index i
        x = vec_initial_area
        i = 0

        while i < self.n_time_periods:
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

    def project(self, df_afolu_trajectories):

        """
            - AFOLU.project takes a data frame (ordered by time series) and returns a data frame of the same order
            - designed to be parallelized or called from command line via __main__ in run_afolu.py
        """

        ##  CHECKS

        # check for internal variables and add if necessary; note, this can be defined for different variables (see model attributes)
        self.model_attributes.manage_pop_to_df(df_afolu_trajectories, "add")
        df_afolu_trajectories.sort_values(by = [self.model_attributes.dim_time_period], inplace = True)
        # check that all required fields are contained—assume that it is ordered by time period
        self.check_df_fields(df_afolu_trajectories)


        ##  CATEGORY INITIALIZATION

        pycat_agrc = self.model_attributes.get_subsector_attribute("Agriculture", "pycategory_primary")
        pycat_frst = self.model_attributes.get_subsector_attribute("Forest", "pycategory_primary")
        pycat_lndu = self.model_attributes.get_subsector_attribute("Land Use", "pycategory_primary")
        pycat_lvst = self.model_attributes.get_subsector_attribute("Livestock", "pycategory_primary")
        # attribute tables
        attr_agrc = self.model_attributes.dict_attributes[pycat_agrc]
        attr_frst = self.model_attributes.dict_attributes[pycat_frst]
        attr_lndu = self.model_attributes.dict_attributes[pycat_lndu]
        attr_lvst = self.model_attributes.dict_attributes[pycat_lvst]

        ##  FIELD INITIALIZATION

        # get the gdp and total population fields
        field_gdp = self.model_attributes.build_varlist("Economy", variable_subsec = self.modvar_econ_gdp)[0]
        field_pop = self.model_attributes.build_varlist("General", variable_subsec = self.modvar_gnrl_pop_total)[0]


        ##  ECON/GNRL VECTOR AND ARRAY INITIALIZATION

        # get some vectors
        vec_gdp = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_econ_gdp, False, return_type = "array_base")#np.array(df_afolu_trajectories[field_gdp])
        vec_pop = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_gnrl_pop_total, False, return_type = "array_base")
        vec_gdp_per_capita = vec_gdp/vec_pop
        # growth rates
        vec_rates_gdp = vec_gdp[1:]/vec_gdp[0:-1] - 1
        vec_rates_gdp_per_capita = vec_gdp_per_capita[1:]/vec_gdp_per_capita[0:-1] - 1


        ##  OUTPUT INITIALIZATION

        df_out = [df_afolu_trajectories[self.required_dimensions].copy()]


        ########################################
        #    LAND USE - UNADJUSTED VARIABLES   #
        ########################################

        # area of the country
        area = float(self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_gnrl_area, return_type = "array_base")[0])
        # get the initial distribution of land
        vec_modvar_lndu_initial_frac = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lndu_initial_frac, return_type = "array_base")[0]
        vec_modvar_lndu_initial_area = vec_modvar_lndu_initial_frac*area
        self.vec_modvar_lndu_initial_area = vec_modvar_lndu_initial_area
        self.mat_trans_unadj, self.mat_ef = self.get_markov_matrices(df_afolu_trajectories)
        # factor for reallocating land in adjustment
        vec_lndu_reallocation_factor = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lndu_reallocation_factor, False, "array_base")
        # common indices
        cat_lndu_ind_pstr = self.model_attributes.dict_attributes["cat_landuse"].get_key_value_index("grasslands")
        cat_lndu_ind_crop = self.model_attributes.dict_attributes["cat_landuse"].get_key_value_index("croplands")

        ###########################
        #    CALCULATE DEMANDS    #
        ###########################

        ##  livestock demands (calculated exogenously)

        # variables requried to estimate demand
        vec_modvar_lvst_pop_init = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_pop_init, True, "array_base")[0]
        fields_lvst_elas = self.model_attributes.switch_variable_category("Livestock", self.modvar_lvst_elas_lvst_demand, "demand_elasticity_category")
        arr_lvst_elas_demand = np.array(df_afolu_trajectories[fields_lvst_elas])
        # get the "vegetarian" factor and use to estimate livestock pop
        vec_lvst_demscale = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_frac_eating_red_meat, False, "array_base", var_bounds = (0, np.inf))
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
        vec_agrc_yield_init = arr_agrc_yf[0]*vec_agrc_cropland_area
        # split into yield for livestock feed (responsive to changes in livestock population) and yield for consumption and export (nonlvstfeed)
        vec_agrc_yield_init_lvstfeed = vec_agrc_yield_init*arr_agrc_frac_feed[0]
        vec_agrc_yield_init_nonlvstfeed = vec_agrc_yield_init - vec_agrc_yield_init_lvstfeed
        # project ag demand for crops that are driven by gdp/capita - set demand scalar for crop demand (increases based on reduction in red meat demand) - depends on how many people eat red meat (vec_lvst_demscale)
        vec_agrc_diet_exchange_scalar = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lndu_vdes, False, "array_base", var_bounds = (0, np.inf))
        vec_agrc_demscale = vec_lvst_demscale + vec_agrc_diet_exchange_scalar - vec_lvst_demscale*vec_agrc_diet_exchange_scalar
        # get categories that need to be scaled
        vec_agrc_scale_demands_for_veg = np.array(self.model_attributes.get_ordered_category_attribute("Agriculture", "apply_vegetarian_exchange_scalar"))
        arr_agrc_demscale = np.outer(vec_agrc_demscale, vec_agrc_scale_demands_for_veg)
        arr_agrc_demscale = arr_agrc_demscale + np.outer(np.ones(len(vec_agrc_demscale)), 1 - vec_agrc_scale_demands_for_veg)
        arr_agrc_nonfeeddem_yield = self.project_per_capita_demand(vec_agrc_yield_init_nonlvstfeed, vec_pop, vec_rates_gdp_per_capita, arr_agrc_elas_crop_demand, arr_agrc_demscale, float)
        # array gives the total yield of crop type i allocated to livestock type j at time 0
        arr_lndu_yield_i_reqd_lvst_j_init = np.outer(vec_agrc_yield_init_lvstfeed, vec_lvst_feed_allocation_weights)


        ################################################
        #    CALCULATE LAND USE + AGRC/LVST DRIVERS    #
        ################################################

        # get land use projections (np arrays) - note, arrs_land_conv returns a list of matrices for troubleshooting
        arr_agrc_frac_cropland, arr_agrc_net_import_increase, arr_agrc_yield, arr_lndu_emissions_conv, arr_land_use, arr_lvst_net_import_increase, self.arrs_land_conv, self.mat_trans_adj, self.yields_per_livestock = self.project_integrated_land_use(
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
            vec_lvst_carry_capacity_scale
        )

        # scale emissions
        #arr_lndu_emissions_conv *= self.model_attributes.get_scalar(self.modvar_lndu_ef_co2_conv, "total")
        df_lndu_emissions_conv = self.model_attributes.array_to_df(arr_lndu_emissions_conv, self.modvar_lndu_emissions_conv, True)
        # assign other variables
        df_agrc_frac_cropland = self.model_attributes.array_to_df(arr_agrc_frac_cropland, self.modvar_agrc_area_prop_calc)
        df_agrc_net_import_increase = self.model_attributes.array_to_df(arr_agrc_net_import_increase, self.modvar_agrc_net_imports)
        df_agrc_yield = self.model_attributes.array_to_df(arr_agrc_yield, self.modvar_agrc_yield)
        df_land_use = self.model_attributes.array_to_df(arr_land_use, self.modvar_lndu_area_by_cat)
        df_lvst_net_import_increase = self.model_attributes.array_to_df(arr_lvst_net_import_increase, self.modvar_lvst_net_imports)

        # add to output data frame
        df_out += [
            df_agrc_frac_cropland,
            df_agrc_net_import_increase,
            df_lndu_emissions_conv,
            df_land_use,
            df_lvst_net_import_increase
        ]


        ##  EXISTENCE EMISSIONS FOR OTHER LANDS, INCLUDING AG ACTIVITY ON PASTURES

        # dictionary variables mapping emission factor variables to output variables
        dict_modvars_lndu_simple_efs = {
            self.modvar_lndu_ef_n2o_past: self.modvar_lndu_emissions_n2o_from_pastures,
            self.modvar_lndu_ef_co2_soilcarb: self.modvar_lndu_emissions_co2_from_pastures,
            self.modvar_lndu_ef_ch4_boc: self.modvar_lndu_emissions_ch4_from_wetlands
        }
        # add to output dataframe
        df_out += self.get_simple_input_to_output_emission_arrays(df_afolu_trajectories, df_land_use, dict_modvars_lndu_simple_efs, self.modvar_lndu_area_by_cat)



        ##################
        #    FORESTRY    #
        ##################

        # get ordered fields from land use
        fields_lndu_forest_ordered = [self.model_attributes.matchstring_landuse_to_forests + x for x in self.model_attributes.dict_attributes[pycat_frst].key_values]
        arr_area_frst = np.array(df_land_use[self.model_attributes.build_varlist("Land Use", variable_subsec = self.modvar_lndu_area_by_cat, restrict_to_category_values = fields_lndu_forest_ordered)])
        # get different variables
        arr_frst_ef_sequestration = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_frst_sq_co2, True, "array_units_corrected")
        arr_frst_ef_methane = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_frst_ef_ch4, True, "array_units_corrected")
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
        field_crop_array = self.model_attributes.build_varlist("Land Use", variable_subsec = self.modvar_lndu_area_by_cat, restrict_to_category_values = [self.cat_lndu_crop])[0]
        vec_cropland_area = np.array(df_land_use[field_crop_array])
        # fraction of cropland represented by each crop
        arr_agrc_frac_cropland_area = self.check_cropland_fractions(df_agrc_frac_cropland, "calculated")
        arr_agrc_crop_area = (arr_agrc_frac_cropland_area.transpose()*vec_cropland_area.transpose()).transpose()
        # unit-corrected emission factors
        arr_agrc_ef_ch4 = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ef_ch4, True, "array_units_corrected")
        arr_agrc_ef_co2 = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ef_co2, True, "array_units_corrected")
        arr_agrc_ef_co2_yield = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ef_co2_yield, True, "array_units_corrected")
        arr_agrc_ef_n2o = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ef_n2o, True, "array_units_corrected")

        # add to output dataframe
        df_out += [
            self.model_attributes.array_to_df(arr_agrc_crop_area, self.modvar_agrc_area_crop),
            self.model_attributes.array_to_df(arr_agrc_ef_ch4*arr_agrc_crop_area, self.modvar_agrc_emissions_ch4_crops),
            self.model_attributes.array_to_df(arr_agrc_ef_co2*arr_agrc_crop_area + arr_agrc_yield*arr_agrc_ef_co2_yield, self.modvar_agrc_emissions_co2_crops),
            self.model_attributes.array_to_df(arr_agrc_ef_n2o*arr_agrc_crop_area, self.modvar_agrc_emissions_n2o_crops)
        ]



        ###################
        #    LIVESTOCK    #
        ###################

        # get area of grassland/pastures
        field_lvst_graze_array = self.model_attributes.build_varlist("Land Use", variable_subsec = self.modvar_lndu_area_by_cat, restrict_to_category_values = [self.cat_lndu_grazing])[0]
        vec_lvst_graze_area = np.array(df_land_use[field_lvst_graze_array])
        # estimate the total number of livestock that are raised, then get emission factor
        arr_lvst_emissions_ch4_ef = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_ef_ch4_ef, True, "array_units_corrected")
        arr_lvst_emissions_ch4_mm = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_ef_ch4_mm, True, "array_units_corrected")
        arr_lvst_emissions_n2o_mm = self.model_attributes.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_ef_n2o_mm, True, "array_units_corrected")
        arr_lvst_pop = arr_lvst_dem_pop - arr_lvst_net_import_increase
        # add to output dataframe
        df_out += [
            self.model_attributes.array_to_df(arr_lvst_emissions_ch4_ef*arr_lvst_pop, self.modvar_lvst_emissions_ch4_ef),
            self.model_attributes.array_to_df(arr_lvst_emissions_ch4_mm*arr_lvst_pop, self.modvar_lvst_emissions_ch4_mm),
            self.model_attributes.array_to_df(arr_lvst_emissions_n2o_mm*arr_lvst_pop, self.modvar_lvst_emissions_n2o_mm),
            self.model_attributes.array_to_df(arr_lvst_pop, self.modvar_lvst_pop)
        ]


        df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)
        self.model_attributes.add_subsector_emissions_aggregates(df_out, self.required_base_subsectors, False)

        return df_out
