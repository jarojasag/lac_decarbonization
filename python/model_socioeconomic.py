import support_functions as sf
import data_structures as ds
import pandas as pd
import numpy as np

class Socioeconomic:

    def __init__(self, attributes: ds.ModelAttributes):

        self.model_attributes = attributes
        self.required_dimensions = self.get_required_dimensions()
        self.required_subsectors, self.required_base_subsectors = self.get_required_subsectors()
        self.required_variables, self.output_variables = self.get_se_input_output_fields()


        ##  set some model fields to connect to the attribute tables

        # economy and general variables
        self.modvar_econ_gdp = "GDP"
        self.modvar_econ_va = "Value Added"
        self.modvar_gnrl_area = "Area of Country"
        self.modvar_gnrl_frac_eating_red_meat = "Fraction Eating Red Meat"
        self.modvar_gnrl_occ = "National Occupation Rate"
        self.modvar_gnrl_subpop = "Population"
        self.modvar_gnrl_pop_total = "Total Population"

        ##  MISCELLANEOUS VARIABLES

        self.time_periods, self.n_time_periods = self.model_attributes.get_time_periods()


    ##  FUNCTIONS FOR MODEL ATTRIBUTE DIMENSIONS

    def check_df_fields(self, df_se_trajectories):
        check_fields = self.required_variables
        # check for required variables
        if not set(check_fields).issubset(df_se_trajectories.columns):
            set_missing = list(set(check_fields) - set(df_se_trajectories.columns))
            set_missing = sf.format_print_list(set_missing)
            raise KeyError(f"Socioconomic projection cannot proceed: The fields {set_missing} are missing.")

    def get_required_subsectors(self):
        subsectors = list(sf.subset_df(self.model_attributes.dict_attributes["abbreviation_subsector"].table, {"sector": ["Socioeconomic"]})["subsector"])
        subsectors_base = subsectors.copy()

        return subsectors, subsectors_base

    def get_required_dimensions(self):
        ## TEMPORARY - derive from attributes later
        required_doa = [self.model_attributes.dim_time_period]
        return required_doa

    def get_se_input_output_fields(self):
        required_doa = [self.model_attributes.dim_time_period]
        required_vars, output_vars = self.model_attributes.get_input_output_fields(self.required_subsectors)
        return required_vars + self.get_required_dimensions(), output_vars

    # projection for socioeconomic is slightly different;
    def project(self, df_se_trajectories: pd.DataFrame) -> tuple:

        """
            the project() method returns a tuple:

            (1) the first element of the return tuple is a modified version of df_se_trajectories data frame that includes socioeconomic projections. This should be passed to other models.

            (2) the second element of the return tuple is a data frame with n_time_periods - 1 rows that represents growth rates in the socioeconomic sector. Row i represents the growth rate from time i to time i + 1.

            Function Arguments
            ------------------
            df_se_trajectories: pd.DataFrame with input variable trajectories for the Socioeconomic model.

        """
        # add population and interpolate if necessary
        self.model_attributes.manage_pop_to_df(df_se_trajectories, "add")
        self.check_df_fields(df_se_trajectories)
        dict_dims, df_se_trajectories, n_projection_time_periods, projection_time_periods = self.model_attributes.check_projection_input_df(df_se_trajectories, True, True, True)

        #
        # MODIFY OCCUPANCY RATE LATER
        #

        # get some vectors
        vec_gdp = self.model_attributes.get_standard_variables(df_se_trajectories, self.modvar_econ_gdp, False, return_type = "array_base")#np.array(df_afolu_trajectories[field_gdp])
        vec_pop = self.model_attributes.get_standard_variables(df_se_trajectories, self.modvar_gnrl_pop_total, False, return_type = "array_base")
        vec_gdp_per_capita = vec_gdp/vec_pop

        # growth rates
        vec_rates_gdp = vec_gdp[1:]/vec_gdp[0:-1] - 1
        vec_rates_gdp_per_capita = vec_gdp_per_capita[1:]/vec_gdp_per_capita[0:-1] - 1

        # get internal variables that are shared between downstream sectors
        time_periods_df = np.array(df_se_trajectories[self.model_attributes.dim_time_period])[0:-1]
        df_se_internal_shared_variables = df_se_trajectories[[self.model_attributes.dim_time_period]].copy()
        df_se_internal_shared_variables["vec_gdp_per_capita"] = vec_gdp_per_capita
        df_se_internal_shared_variables = pd.merge(
            df_se_internal_shared_variables,
            pd.DataFrame({self.model_attributes.dim_time_period: time_periods_df, "vec_rates_gdp": vec_rates_gdp, "vec_rates_gdp_per_capita": vec_rates_gdp_per_capita}),
            how = "left"
        )

        return (df_se_trajectories, df_se_internal_shared_variables)
