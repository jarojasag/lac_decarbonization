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
        self.modvar_agrc_area_prop = "Cropland Area Proportion"
        self.modvar_agrc_area_crop = "Crop Area"
        self.modvar_agrc_ef_ch4 = ":math:\\text{CH}_4 Crop Activity Emission Factor"
        self.modvar_agrc_ef_n2o = ":math:\\text{CO}_2 Crop Activity Emission Factor"
        self.modvar_agrc_ef_co2 = ":math:\\text{N}_2\\text{O} Crop Activity Emission Factor"
        self.modvar_agrc_elas_crop_demand_income = "Crop Demand Income Elasticity"
        self.modvar_agrc_emissions_ch4_crops = ":math:\\text{CH}_4 Emissions from Crop Activity"
        self.modvar_agrc_emissions_co2_crops = ":math:\\text{CO}_2 Emissions from Crop Activity"
        self.modvar_agrc_emissions_n2o_crops = ":math:\\text{N}_2\\text{O} Emissions from Crop Activity"
        self.modvar_agrc_net_imports = "Crop Surplus Demand"
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
        self.modvar_lvst_frac_meat_import = "Fraction of Meat Consumption from Imports"
        self.modvar_lvst_meat_demand_scalar = "Red Meat Demand Scalar"
        self.modvar_lvst_net_imports = "Livestock Surplus Demand"
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
        self.cat_lu_crop = "croplands"
        self.cat_lu_grazing = "grasslands"
        self.varchar_str_emission_gas = "$EMISSION-GAS$"
        self.varchar_str_unit_mass = "$UNIT-MASS$"

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


    # define a function to clean up code
    def get_standard_variables(self, df_in, modvar, override_vector_for_single_mv_q: bool = False, return_type: str = "data_frame"):
        flds = self.model_attributes.dict_model_variables_to_variables[modvar]
        flds = flds[0] if ((len(flds) == 1) and not override_vector_for_single_mv_q) else flds

        valid_rts = ["data_frame", "array_base", "array_units_corrected"]
        if return_type not in valid_rts:
            vrts = sf.format_print_list(valid_rts)
            raise ValueError(f"Invalid return_type in get_standard_variables: valid types are {vrts}.")

        # initialize output, apply various common transformations based on type
        out = df_in[flds]
        if return_type != "data_frame":
            out = np.array(out)
            if return_type == "array_units_corrected":
                out *= self.get_scalar(modvar, "total")

        return out


    def get_time_periods(self):
        pydim_time_period = self.model_attributes.get_dimensional_attribute("time_period", "pydim")
        time_periods = self.model_attributes.dict_attributes[pydim_time_period].key_values
        return time_periods, len(time_periods)


    ##  STREAMLINING FUNCTIONS

    # convert an array to a varibale out dataframe
    def array_to_df(self, arr_in, modvar: str, include_scalars = False) -> pd.DataFrame:
        # get subsector and fields to name based on variable
        subsector = self.model_attributes.dict_model_variable_to_subsector[modvar]
        fields = self.model_attributes.build_varlist(subsector, variable_subsec = modvar)

        scalar_em = 1
        scalar_me = 1
        if include_scalars:
            # get scalars
            gas = self.model_attributes.get_variable_characteristic(modvar, self.varchar_str_emission_gas)
            mass = self.model_attributes.get_variable_characteristic(modvar, self.varchar_str_unit_mass)
            # will conver ch4 to co2e e.g. + kg to MT
            scalar_em = 1 if not gas else self.model_attributes.get_gwp(gas.lower())
            scalar_me = 1 if not mass else self.model_attributes.get_mass_equivalent(mass.lower())

        # raise error if there's a shape mismatch
        if len(fields) != arr_in.shape[1]:
            flds_print = sf.format_print_list(fields)
            raise ValueError(f"Array shape mismatch for fields {flds_print}: the array only has {arr_in.shape[1]} columns.")

        return pd.DataFrame(arr_in*scalar_em*scalar_me, columns = fields)

    # some scalars
    def get_scalar(self, modvar: str, return_type: str = "total"):

        valid_rts = ["total", "gas", "mass"]
        if return_type not in valid_rts:
            tps = sf.format_print_list(valid_rts)
            raise ValueError(f"Invalid return type '{return_type}' in get_scalar: valid types are {tps}.")

        # get scalars
        gas = self.model_attributes.get_variable_characteristic(modvar, self.varchar_str_emission_gas)
        scalar_gas = 1 if not gas else self.model_attributes.get_gwp(gas.lower())
        #
        mass = self.model_attributes.get_variable_characteristic(modvar, self.varchar_str_unit_mass)
        scalar_mass = 1 if not mass else self.model_attributes.get_mass_equivalent(mass.lower())

        if return_type == "gas":
            out = scalar_gas
        elif return_type == "mass":
            out = scalar_mass
        elif return_type == "total":
            out = scalar_gas*scalar_mass

        return out

    # loop over a dictionary of simple variables that map an emission factor () to build out
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
                arr_ef = np.array(self.get_standard_variables(df_ef, var, True, "array_units_corrected"))
                # get the emissions driver array (driver must h)
                arr_driver = np.array(df_driver[self.model_attributes.build_target_varlist_from_source_varcats(var, variable_driver)])
                df_out.append(self.array_to_df(arr_driver*arr_ef, dict_vars[var]))

        return df_out

    # add subsector emission totals
    def add_subsector_emissions_aggregates(self, df_in: pd.DataFrame, stop_on_missing_fields_q: bool = False):
        # loop over base subsectors
        for subsector in self.required_base_subsectors:
            vars_subsec = self.model_attributes.dict_model_variables_by_subsector[subsector]
            # add subsector abbreviation
            fld_nam = self.model_attributes.get_subsector_attribute(subsector, "abv_subsector")
            fld_nam = f"emission_co2e_subsector_total_{fld_nam}"

            flds_add = []
            for var in vars_subsec:
                var_type = self.model_attributes.get_variable_attribute(var, "variable_type").lower()
                gas = self.model_attributes.get_variable_characteristic(var, "$EMISSION-GAS$")
                if (var_type == "output") and gas:
                    flds_add +=  self.model_attributes.dict_model_variables_to_variables[var]

            # check for missing fields; notify
            missing_fields = [x for x in flds_add if x not in df_in.columns]
            if len(missing_fields) > 0:
                str_mf = print_setdiff(set(df_in.columns), set(flds_add))
                str_mf = f"Missing fields {str_mf}.%s"
                if stop_on_missing_fields_q:
                    raise ValueError(str_mf%(" Subsector emission totals will not be added."))
                else:
                    warnings.warn(str_mf%(" Subsector emission totals will exclude these fields."))

            keep_fields = [x for x in flds_add if x in df_in.columns]
            df_in[fld_nam] = df_in[keep_fields].sum(axis = 0)


    ######################################
    #    SUBSECTOR SPECIFIC FUNCTIONS    #
    ######################################


    ###   AGRICULTURE

    def check_cropland_fractions(self, df_in, thresh_for_correction: float = 0.01):

            arr = self.get_standard_variables(df_in, self.modvar_agrc_area_prop, True, "array_base")
            totals = sum(arr.transpose())
            m = max(np.abs(totals - 1))

            if m > thresh_for_correction:
                raise ValueError(f"Invalid crop areas found in check_cropland_fractions. The maximum fraction total was {m}; the maximum allowed deviation from 1 is {thresh_for_correction}.")
            else:
                arr = (arr.transpose()/totals).transpose()

            return arr


    ###   LAND USE

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

        # running matrix Q_i; initialize as identity. initialize running matrix of land use are
        Q_i = np.identity(attr_lndu.n_key_values)
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
            vec_land_use = np.matmul(vec_initial_area, Q_i)
            vec_emissions_conv = sum((arrs_transitions[i_tr] * arrs_efs[i_ef]).transpose()*x.transpose())
            arr_land_conv = (arrs_transitions[i_tr].transpose()*x.transpose()).transpose()

            # update matrices
            rng_put = np.arange(i*attr_lndu.n_key_values, (i + 1)*attr_lndu.n_key_values)
            np.put(arr_land_use, rng_put, vec_land_use)
            np.put(arr_emissions_conv, rng_put, vec_emissions_conv)
            np.put(arrs_land_conv, np.arange(i*attr_lndu.n_key_values**2, (i + 1)*attr_lndu.n_key_values**2), arr_land_conv)

            # update transition matrix and land use matrix
            Q_i = np.matmul(Q_i, arrs_transitions[i_tr])
            x = vec_land_use

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
        vec_gdp = self.get_standard_variables(df_afolu_trajectories, self.modvar_econ_gdp, False, return_type = "array_base")#np.array(df_afolu_trajectories[field_gdp])
        vec_pop = self.get_standard_variables(df_afolu_trajectories, self.modvar_gnrl_pop_total, False, return_type = "array_base")
        vec_gdp_per_capita = vec_gdp/vec_pop
        # growth rates
        vec_rates_gdp = vec_gdp[1:]/vec_gdp[0:-1] - 1
        vec_rates_gdp_per_capita = vec_gdp_per_capita[1:]/vec_gdp_per_capita[0:-1] - 1


        ##  OUTPUT INITIALIZATION

        df_out = [df_afolu_trajectories[self.required_dimensions].copy()]



        ##################
        #    LAND USE    #
        ##################

        # area of the country
        area = float(self.get_standard_variables(df_afolu_trajectories, self.modvar_gnrl_area, return_type = "array_base")[0])

        ##  LU MARKOV

        # get the initial distribution of land
        vec_modvar_lndu_initial_frac = self.get_standard_variables(df_afolu_trajectories, self.modvar_lndu_initial_frac, return_type = "array_base")[0]
        vec_modvar_lndu_initial_area = vec_modvar_lndu_initial_frac*area
        self.vec_modvar_lndu_initial_area = vec_modvar_lndu_initial_area
        self.mat_trans, self.mat_ef = self.get_markov_matrices(df_afolu_trajectories)
        # get land use projections (np arrays) - note, arrs_land_conv returns a list of matrices for troubleshooting
        arr_lndu_emissions_conv, arr_land_use, arrs_land_conv = self.project_land_use(vec_modvar_lndu_initial_area, *self.get_markov_matrices(df_afolu_trajectories))
        # scale emissions
        arr_lndu_emissions_conv *= self.get_scalar(self.modvar_lndu_ef_co2_conv, "total")
        df_lndu_emissions_conv = self.array_to_df(arr_lndu_emissions_conv, self.modvar_lndu_emissions_conv)
        df_land_use = self.array_to_df(arr_land_use, self.modvar_lndu_area_by_cat)
        # add to output data frame
        df_out.append(df_lndu_emissions_conv)
        df_out.append(df_land_use)

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
        arr_frst_ef_sequestration = self.get_standard_variables(df_afolu_trajectories, self.modvar_frst_sq_co2, True, "array_units_corrected")
        arr_frst_ef_methane = self.get_standard_variables(df_afolu_trajectories, self.modvar_frst_ef_ch4, True, "array_units_corrected")
        # build output variables
        df_out += [
            self.array_to_df(-1*arr_area_frst*arr_frst_ef_sequestration, self.modvar_frst_emissions_sequestration),
            self.array_to_df(arr_area_frst*arr_frst_ef_methane, self.modvar_frst_emissions_methane)
        ]

        ##  NEEDED: FOREST FIRES (ADD HERE)
        ##  NEEDED: WOOD PRODUCTS (ADD HERE)



        #####################
        #    AGRICULTURE    #
        #####################

        # get area of cropland
        field_crop_array = self.model_attributes.build_varlist("Land Use", variable_subsec = self.modvar_lndu_area_by_cat, restrict_to_category_values = [self.cat_lu_crop])[0]
        vec_cropland_area = np.array(df_land_use[field_crop_array])
        # fraction of cropland represented by each crop
        arr_agrc_frac_cropland_area = self.check_cropland_fractions(df_afolu_trajectories)
        arr_agrc_crop_area = (arr_agrc_frac_cropland_area.transpose()*vec_cropland_area.transpose()).transpose()
        # area-corrected emission factors
        arr_agrc_ef_ch4 = self.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ef_ch4, True, "array_units_corrected")
        arr_agrc_ef_co2 = self.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ef_co2, True, "array_units_corrected")
        arr_agrc_ef_n2o = self.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_ef_n2o, True, "array_units_corrected")
        # estimate yield capacity
        arr_agrc_yf = self.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_yf, True, "array_base")
        arr_yield = arr_agrc_yf*arr_agrc_crop_area
        # estimate demand for crops (used in CBA)
        arr_agrc_elas_crop_demand = self.get_standard_variables(df_afolu_trajectories, self.modvar_agrc_elas_crop_demand_income, False, "array_base")
        arr_agrc_yield_dem_scale_proj = (vec_rates_gdp_per_capita.transpose()*arr_agrc_elas_crop_demand[0:-1].transpose()).transpose()
        arr_agrc_yield_dem_scale_proj = np.cumprod(1 + arr_agrc_yield_dem_scale_proj, axis = 0)
        arr_agrc_yield_dem_scale_proj = np.concatenate([np.ones((1,len(arr_agrc_yield_dem_scale_proj[0]))), arr_agrc_yield_dem_scale_proj])
        # estimate net imports (surplus demand)
        arr_agrc_net_imports = arr_agrc_yield_dem_scale_proj*arr_yield[0] - arr_yield
        # add to output dataframe
        df_out += [
            self.array_to_df(arr_agrc_crop_area, self.modvar_agrc_area_crop),
            self.array_to_df(arr_yield, self.modvar_agrc_yield),
            self.array_to_df(arr_agrc_ef_ch4, self.modvar_agrc_emissions_ch4_crops),
            self.array_to_df(arr_agrc_ef_co2, self.modvar_agrc_emissions_co2_crops),
            self.array_to_df(arr_agrc_ef_n2o, self.modvar_agrc_emissions_n2o_crops),
            self.array_to_df(arr_agrc_net_imports, self.modvar_agrc_net_imports)
        ]



        ###################
        #    LIVESTOCK    #
        ###################

        # get area of grassland/pastures
        field_lvst_graze_array = self.model_attributes.build_varlist("Land Use", variable_subsec = self.modvar_lndu_area_by_cat, restrict_to_category_values = [self.cat_lu_grazing])[0]
        vec_lvst_graze_area = np.array(df_land_use[field_lvst_graze_array])
        # get weights for allocating grazing area to animals - based on first year only
        vec_lvst_base_graze_weights = self.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_dry_matter_consumption, True, "array_base")[0]
        vec_modvar_lvst_pop_init = self.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_pop_init, True, "array_base")[0]
        vec_lvst_grassland_allocation_weights = (vec_modvar_lvst_pop_init*vec_lvst_base_graze_weights)/np.dot(vec_modvar_lvst_pop_init, vec_lvst_base_graze_weights)
        # estimate the total area used for grazing, then get the number of livestock/area
        arr_lvst_graze_area = np.outer(vec_lvst_graze_area, vec_lvst_grassland_allocation_weights)
        vec_lvst_carry_capacity_scale = self.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_carrying_capacity_scalar, False, "array_base")
        vec_lvst_carry_capacity = vec_modvar_lvst_pop_init/arr_lvst_graze_area[0]
        arr_lvst_carry_capacity = np.outer(vec_lvst_carry_capacity_scale, vec_lvst_carry_capacity)
        # estimate the total number of livestock that are raised, then get emission factors
        arr_lvst_pop = np.array(arr_lvst_carry_capacity*arr_lvst_graze_area).astype(int)
        arr_lvst_emissions_ch4_ef = self.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_ef_ch4_ef, True, "array_units_corrected")
        arr_lvst_emissions_ch4_mm = self.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_ef_ch4_mm, True, "array_units_corrected")
        arr_lvst_emissions_n2o_mm = self.get_standard_variables(df_afolu_trajectories, self.modvar_lvst_ef_n2o_mm, True, "array_units_corrected")
        # estimate demand for livestock (used in CBA)
        fields_lvst_elas = self.model_attributes.switch_variable_category("Livestock", self.modvar_lvst_elas_lvst_demand, "demand_elasticity_category")
        arr_lvst_elas_demand = np.array(df_afolu_trajectories[fields_lvst_elas])
        # get the demand scalar, then apply to the initial population
        arr_lvst_dem_scale_proj = (vec_rates_gdp_per_capita.transpose()*arr_lvst_elas_demand[0:-1].transpose()).transpose()
        arr_lvst_dem_scale_proj = np.cumprod(1 + arr_lvst_dem_scale_proj, axis = 0)
        arr_lvst_dem_scale_proj= np.concatenate([np.ones((1,len(arr_lvst_dem_scale_proj[0]))), arr_lvst_dem_scale_proj])
        arr_lvst_dem_pop = np.array(arr_lvst_dem_scale_proj*vec_modvar_lvst_pop_init).astype(int)
        # clean the population and grab net imports
        arr_lvst_pop = self.reassign_pops_from_proj_to_carry(arr_lvst_pop, arr_lvst_dem_pop)
        arr_lvst_net_imports = arr_lvst_dem_pop - arr_lvst_pop

        # add to output dataframe
        df_out += [
            self.array_to_df(arr_lvst_emissions_ch4_ef*arr_lvst_pop, self.modvar_lvst_emissions_ch4_ef),
            self.array_to_df(arr_lvst_emissions_ch4_mm*arr_lvst_pop, self.modvar_lvst_emissions_ch4_mm),
            self.array_to_df(arr_lvst_emissions_n2o_mm*arr_lvst_pop, self.modvar_lvst_emissions_n2o_mm),
            self.array_to_df(arr_lvst_pop, self.modvar_lvst_pop),
            self.array_to_df(arr_lvst_net_imports, self.modvar_lvst_net_imports)
        ]


        df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)
        self.add_subsector_emissions_aggregates(df_out, False)

        return df_out
