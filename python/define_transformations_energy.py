from attribute_table import AttributeTable
import auxiliary_definitions_transformations as adt
import ingestion as ing
import logging
import model_afolu as mafl
import model_attributes as ma
import model_electricity as ml
import model_energy as me
import numpy as np
import os, os.path
import pandas as pd
import setup_analysis as sa
from sisepuede_file_structure import *
import support_classes as sc
import support_functions as sf
import time
from typing import Union
import warnings



class TransformationsEnergy:
    """
    Build energy transformations using general transformations defined in
        auxiliary_definitions_transformations. Wraps more general forms from 
        auxiliary_definitions_transformations into functions and classes
        with shared ramps, paramters, and build functionality.

    Initialization Arguments
	------------------------
	- model_attributes: ModelAttributes object used to manage variables and
		coordination
    - dict_config: configuration dictionary used to pass parameters to 
        transformations. See ?TransformationEnergy._initialize_parameters() for
        more information on requirements.
    - dir_jl: location of Julia directory containing Julia environment and 
        support modules
    - fp_nemomod_reference_files: directory housing reference files called by
		NemoMod when running electricity model. Required to access data in 
        ElectricEnergy. Needs the following CSVs:

        * Required keys or CSVs (without extension):
            (1) CapacityFactor
            (2) SpecifiedDemandProfile

    Optional Arguments
    ------------------
	- fp_nemomod_temp_sqlite_db: optional file path to use for SQLite database
		used in Julia NemoMod Electricity model
        * If None, defaults to a temporary path sql database
    - logger: optional logger object
    """
    
    def __init__(self,
        model_attributes: ma.ModelAttributes,
        dict_config: Dict,
        dir_jl: str,
        fp_nemomod_reference_files: str,
		fp_nemomod_temp_sqlite_db: Union[str, None] = None,
		logger: Union[logging.Logger, None] = None,
    ):

        self.logger = logger

        self._initialize_attributes(model_attributes)
        self._initialize_config(dict_config = dict_config)
        self._initialize_models(dir_jl, fp_nemomod_reference_files)
        self._initialize_parameters(dict_config = dict_config)
        self._initialize_ramp()
        self._initialize_transformations()




    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################
    
    def get_entc_cats_max_investment_ramp(self,
        dict_config: Union[Dict, None] = None,
    ) -> List[str]:
        """
        Set categories to which a cap on maximum investment is applied in the 
            renewables target shift.  If dict_config is None, uses self.config.
        
        Keyword Arguments
        -----------------
        - dict_config: dictionary mapping input configuration arguments to key 
            values. Must include the following keys:

            * categories_max_investment_ramp: list of categories to tag as 
                renewable for the Renewable Targets transformation.
        """
        dict_config = self.config if not isinstance(dict_config, dict) else dict_config

        cats_entc_max_investment_ramp = self.config.get(self.key_config_cats_entc_max_investment_ramp)
        cats_entc_max_investment_ramp = (
            list(cats_entc_max_investment_ramp)
            if isinstance(cats_entc_max_investment_ramp, list) or isinstance(cats_entc_max_investment_ramp, np.ndarray)
            else [
                "pp_geothermal",
                "pp_nuclear"
            ]
        )

        return cats_entc_max_investment_ramp



    def get_entc_cats_renewable(self,
        dict_config: Union[Dict, None] = None,
    ) -> List[str]:
        """
        Set renewable categories based on the input dictionary dict_config. If 
            dict_config is None, uses self.config.
        
        Keyword Arguments
        -----------------
        - dict_config: dictionary mapping input configuration arguments to key 
            values. Must include the following keys:

            * categories_entc_renewable: list of categories to tag as renewable 
                for the Renewable Targets transformation.
        """
        dict_config = self.config if not isinstance(dict_config, dict) else dict_config

        cats_renewable = dict_config.get(self.key_config_cats_entc_renewable)
        cats_renewable = (
            list(cats_renewable)
            if isinstance(cats_renewable, list) or isinstance(cats_renewable, np.ndarray)
            else [
                "pp_geothermal",
                "pp_hydropower",
                "pp_ocean",
                "pp_solar",
                "pp_wind"
            ]
        )

        return cats_renewable



    def get_ramp_characteristics(self,
        dict_config: Union[Dict, None] = None,
    ) -> List[str]:
        """
        Get parameters for the implementation of transformations. Returns a 
            tuple with the following elements:

            (
                n_tp_ramp,
                vir_renewable_cap_delta_frac,
                vir_renewable_cap_max_frac,
                year_0_ramp, 
            )
        
        If dict_config is None, uses self.config.

        NOTE: Requires those keys in dict_config to set. If not found, will set
            the following defaults:
                * year_0_ramp: 9th year (10th time period)
                * n_tp_ramp: n_tp - t0_ramp - 1 (ramps to 1 at final time 
                    period)
                * vir_renewable_cap_delta_frac: 0.1
                * vir_renewable_cap_max_frac: 0.01 

        Keyword Arguments
        -----------------
        - dict_config: dictionary mapping input configuration arguments to key 
            values. Must include the following keys:

            * categories_entc_renewable: list of categories to tag as renewable 
                for the Renewable Targets transformation.
        """

        dict_config = self.config if not isinstance(dict_config, dict) else dict_config
        n_tp = len(self.time_periods.all_time_periods)

        # get first year of non-baseline
        default_year = self.time_periods.all_years[min(9, n_tp - 1)]
        year_0_ramp = dict_config.get(self.key_config_year_0_ramp)
        year_0_ramp = (
            self.time_periods.all_years[default_year] 
            if not sf.isnumber(year_0_ramp, integer = True)
            else year_0_ramp
        )

        # shift by 2--1 to account for baseline having no uncertainty, 1 for py reindexing
        default_n_tp_ramp = n_tp - self.time_periods.year_to_tp(year_0_ramp) - 1
        n_tp_ramp = dict_config.get(self.key_config_n_tp_ramp)
        n_tp_ramp = (
            default_n_tp_ramp
            if not sf.isnumber(n_tp_ramp, integer = True)
            else n_tp_ramp
        )

        # get VIR (get_vir_max_capacity) delta_frac
        default_vir_renewable_cap_delta_frac = 0.01
        vir_renewable_cap_delta_frac = dict_config.get(self.key_config_vir_renewable_cap_delta_frac)
        vir_renewable_cap_delta_frac = (
            default_vir_renewable_cap_delta_frac
            if not sf.isnumber(vir_renewable_cap_delta_frac)
            else vir_renewable_cap_delta_frac
        )
        vir_renewable_cap_delta_frac = float(sf.vec_bounds(vir_renewable_cap_delta_frac, (0.0, 1.0)))

        # get VIR (get_vir_max_capacity) max_frac
        default_vir_renewable_cap_max_frac = 0.05
        vir_renewable_cap_max_frac = dict_config.get(self.key_config_vir_renewable_cap_max_frac)
        vir_renewable_cap_max_frac = (
            default_vir_renewable_cap_max_frac
            if not sf.isnumber(vir_renewable_cap_max_frac)
            else vir_renewable_cap_max_frac
        )
        vir_renewable_cap_max_frac = float(sf.vec_bounds(vir_renewable_cap_max_frac, (0.0, 1.0)))

        tup_out = (
            n_tp_ramp,
            vir_renewable_cap_delta_frac,
            vir_renewable_cap_max_frac,
            year_0_ramp, 
        )

        return tup_out


    
    def get_dict_entc_renewable_target_msp(self,
        cats_renewable: Union[List[str], None], 
        dict_config: Union[Dict, None] = None,
    ) -> List[str]:
        """
        Set any targets for renewable energy categories. Relies on 
            cats_renewable to verify keys in renewable_target_entc
        
        Keyword Arguments
        -----------------
        - dict_config: dictionary mapping input configuration arguments to key 
            values. Must include the following keys:

            * dict_entc_renewable_target_msp: dictionary of renewable energy
                categories mapped to MSP targets under the renewable target
                transformation
        """
        attr_tech = self.model_attributes.dict_attributes.get("cat_technology")
        dict_config = self.config if not isinstance(dict_config, dict) else dict_config
        cats_renewable = [x for x in cats_renewable if x in attr_tech.key_values]

        dict_entc_renewable_target_msp = dict_config.get(self.key_config_dict_entc_renewable_target_msp)
        dict_entc_renewable_target_msp = (
            {}
            if not isinstance(dict_entc_renewable_target_msp, dict)
            else dict(
                (k, v) for k, v in dict_entc_renewable_target_msp.items() 
                if (k in cats_renewable) and (sf.isnumber(v))
            )
        )

        return dict_entc_renewable_target_msp




    def _initialize_attributes(self,
        model_attributes: ma.ModelAttributes,
    ) -> None:
        """
        Initialize the model attributes object. Checks implementation and throws
            an error if issues arise. Sets the following properties

            * self.attribute_strategy
            * self.model_attributes
            * self.time_periods (support_classes.TimePeriods object)
        """

        # run checks and throw and
        error_q = False
        error_q = error_q | (model_attributes is None)
        if error_q:
            raise RuntimeError(f"Error: invalid specification of model_attributes in transformations_energy")

        attribute_strategy = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_strategy_id}")
        time_periods = sc.TimePeriods(model_attributes)


        ##  SET PROPERTIES
        
        self.attribute_strategy = attribute_strategy
        self.model_attributes = model_attributes
        self.time_periods = time_periods

        return None



    def _initialize_config(self,
        dict_config: Union[Dict[str, Any], None],
    ) -> None:
        """
        Define the configuration dictionary and paramter keys. Sets the 
            following properties:

            * self.config (configuration dictionary)
            * self.key_* (keys)
            
        Function Arguments
        ------------------
        - dict_config: dictionary mapping input configuration arguments to key 
            values. Can include the following keys:

            * "categories_entc_max_investment_ramp": list of categories to apply
                self.vec_implementation_ramp_renewable_cap to with a maximum
                investment cap (implemented *after* turning on renewable target)
            * "categories_entc_renewable": list of categories to tag as 
                renewable for the Renewable Targets transformation (sets 
                self.cats_renewable)
            * "dict_entc_renewable_target_msp": optional dictionary mapping 
                renewable ENTC categories to MSP fractions to use in the 
                Renewable Targets trasnsformationl. Can be used to ensure some
                minimum contribution of certain renewables--e.g.,

                    {
                        "pp_hydropower": 0.1,
                        "pp_solar": 0.15
                    }

                will ensure that hydropower is at least 10% of the mix and solar
                is at least 15%. 

            * "n_tp_ramp": number of time periods to use to ramp up. If None or
                not specified, builds to full implementation by the final time
                period
            * "vir_renewable_cap_delta_frac": change (applied downward from 
                "vir_renewable_cap_max_frac") in cap for for new technology
                capacities available to build in time period while transitioning
                to renewable capacties. Default is 0.01 (will decline by 1% each
                time period after "year_0_ramp")
            * "vir_renewable_cap_max_frac": cap for for new technology 
                capacities available to build in time period while transitioning
                to renewable capacties; entered as a fraction of estimated
                capacity in "year_0_ramp". Default is 0.05
            * "year_0_ramp": last year with no diversion from baseline strategy
                (baseline for implementation ramp)
        """

        dict_config = {} if not isinstance(dict_config, dict) else dict_config

        # set parameters
        self.config = dict_config
        self.key_config_cats_entc_max_investment_ramp = "categories_entc_max_investment_ramp"
        self.key_config_cats_entc_renewable = "categories_entc_renewable"
        self.key_config_dict_entc_renewable_target_msp = "dict_entc_renewable_target_msp"
        self.key_config_n_tp_ramp = "n_tp_ramp"
        self.key_config_vir_renewable_cap_delta_frac = "vir_renewable_cap_delta_frac"
        self.key_config_vir_renewable_cap_max_frac = "vir_renewable_cap_max_frac"
        self.key_config_year_0_ramp = "year_0_ramp" 

        return None



    def _initialize_models(self,
        dir_jl: str,
        fp_nemomod_reference_files: str,
    ) -> None:
        """
        Define model objects for use in variable access and base estimates.

        Function Arguments
        ------------------
        - dir_jl: location of Julia directory containing Julia environment and 
        support modules
        - fp_nemomod_reference_files: directory housing reference files called 
            by NemoMod when running electricity model. Required to access data 
            in ElectricEnergy. Needs the following CSVs:

            * Required keys or CSVs (without extension):
                (1) CapacityFactor
                (2) SpecifiedDemandProfile
        """

        model_afolu = mafl.AFOLU(self.model_attributes)
        model_electricity = ml.ElectricEnergy(
            self.model_attributes, 
            dir_jl,
            fp_nemomod_reference_files,
            initialize_julia = False
        )

        self.model_afolu = model_afolu
        self.model_electricity = model_electricity
        self.model_energy = model_electricity.model_energy

        return None


    
    def _initialize_parameters(self,
        dict_config: Union[Dict[str, Any], None] = None,
    ) -> None:
        """
        Define key parameters for transformation. For keys needed to initialize
            and define these parameters, see ?self._initialize_config
      
        """

        dict_config = self.config if not isinstance(dict_config, dict) else dict_config

        # get parameters from configuration dictionary
        cats_entc_max_investment_ramp = self.get_entc_cats_max_investment_ramp()
        cats_renewable = self.get_entc_cats_renewable()
        (
            n_tp_ramp,
            vir_renewable_cap_delta_frac,
            vir_renewable_cap_max_frac,
            year_0_ramp
        ) = self.get_ramp_characteristics()

        dict_entc_renewable_target_msp = self.get_dict_entc_renewable_target_msp(cats_renewable)


        ##  SET PROPERTIES

        self.cats_entc_max_investment_ramp = cats_entc_max_investment_ramp
        self.cats_renewable = cats_renewable
        self.dict_entc_renewable_target_msp = dict_entc_renewable_target_msp
        self.n_tp_ramp = n_tp_ramp
        self.vir_renewable_cap_delta_frac = vir_renewable_cap_delta_frac
        self.vir_renewable_cap_max_frac = vir_renewable_cap_max_frac
        self.year_0_ramp = year_0_ramp

        return None
    


    def _initialize_ramp(self,
    ) -> None: 
        """
        Initialize the ramp vector for implementing transformations. Sets the 
            following properties:

            * self.dict_entc_renewable_target_cats_max_investment
            * self.vec_implementation_ramp
            * self.vec_implementation_ramp_renewable_cap
        """
        
        vec_implementation_ramp = self.build_implementation_ramp_vector()
        vec_implementation_ramp_renewable_cap = self.get_vir_max_capacity(vec_implementation_ramp)

        dict_entc_renewable_target_cats_max_investment = dict(
            (
                x, 
                {
                    "vec": vec_implementation_ramp_renewable_cap,
                    "type": "scalar"
                }
            ) for x in self.cats_entc_max_investment_ramp
        )
        

        ##  SET PROPERTIES
        self.dict_entc_renewable_target_cats_max_investment = dict_entc_renewable_target_cats_max_investment
        self.vec_implementation_ramp = vec_implementation_ramp
        self.vec_implementation_ramp_renewable_cap = vec_implementation_ramp_renewable_cap

        return None



    def _log(self,
		msg: str,
		type_log: str = "log",
		**kwargs
	) -> None:
        """
		Clean implementation of sf._optional_log in-line using default logger. See ?sf._optional_log for more information

		Function Arguments
		------------------
		- msg: message to log

		Keyword Arguments
		-----------------
		- type_log: type of log to use
		- **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
		"""
        sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)

        return None



    ####################################
    #    OTHER SUPPORTING FUNCTIONS    #
    ####################################

    def build_implementation_ramp_vector(self,
        year_0: Union[int, None] = None,
        n_years_ramp: Union[int, None] = None,
    ) -> np.ndarray:
        """
        Build the implementation ramp vector

        Function Arguments
		------------------

        Keyword Arguments
		-----------------
		- year_0: last year without change from baseline
        - n_years_ramp: number of years to go from 0 to 1
        """
        year_0 = self.year_0_ramp if (year_0 is None) else year_0
        n_years_ramp = self.n_tp_ramp if (n_years_ramp is None) else n_years_ramp

        tp_0 = self.time_periods.year_to_tp(year_0) #10
        n_tp = len(self.time_periods.all_time_periods) #25

        vec_out = np.array([max(0, min((x - tp_0)/n_years_ramp, 1)) for x in range(n_tp)])

        return vec_out



    def get_vir_max_capacity(self,
        vec_implementation_ramp: np.ndarray,
        delta_frac: Union[float, None] = None,
        dict_values_to_inds: Union[Dict, None] = None,
        max_frac: Union[float, None] = None,
    ) -> np.ndarray:
        """
        Buil a new value for the max_capacity based on vec_implementation_ramp.
            Starts with max_frac of a technicology's maximum residual capacity
            in the first period when vec_implementation_ramp != 0, then declines
            by delta_frac the specified number of time periods. Ramp down a cap 
            based on the renewable energy target.

        Function Arguments
        ------------------
        - vec_implementation_ramp: vector of lever implementation ramp to use as
            reference

        Keyword Arguments
        -----------------
        - delta_frac: delta to apply at each time period after the first time
            non-0 vec_implementation_ramp time_period. Defaults to 
            self.vir_renewable_cap_delta_frac if unspecified
        - dict_values_to_inds: optional dictionary mapping a value to row indicies
            to pass the value to. Can be used, for example, to provide a cap on new
            investments in early time periods. 
         - max_frac: fraction of maximum residual capacity to use as cap in first
            time period where vec_implementation_ramp > 0. Defaults to
            self.vir_renewable_cap_max_frac if unspecified
        """

        delta_frac = (
            self.vir_renewable_cap_delta_frac
            if not sf.isnumber(delta_frac)
            else float(sf.vec_bounds(delta_frac, (0.0, 1.0)))
        )
        max_frac = (
            self.vir_renewable_cap_max_frac
            if not sf.isnumber(max_frac)
            else float(sf.vec_bounds(max_frac, (0.0, 1.0)))
        )

        vec_implementation_ramp_max_capacity = np.ones(len(vec_implementation_ramp))
        i0 = None

        for i in range(len(vec_implementation_ramp)):
            if vec_implementation_ramp[i] == 0:
                vec_implementation_ramp_max_capacity[i] = -999
            else:
                i0 = i if (i0 is None) else i0
                vec_implementation_ramp_max_capacity[i] = max(max_frac - delta_frac*(i - i0), 0.0)


        if isinstance(dict_values_to_inds, dict):
            for k in dict_values_to_inds.keys():
                np.put(vec_implementation_ramp_max_capacity, dict_values_to_inds.get(k), k)

        return vec_implementation_ramp_max_capacity 




    def _initialize_transformations(self,
    ) -> None:
        """
        Initialize all sc.Transformation objects used to manage the construction
            of transformations. 

        NOTE: This is the key function mapping each function to a transformation
            name.
            
        Sets the following properties:

            * self.transformation_***
            * self
        """

        attr_strategy = self.model_attributes.dict_attributes.get(f"dim_{self.model_attributes.dim_strategy_id}")
        all_transformations = []
        dict_transformations = {}

        ##  CCSQ


        ##  ENTC

        #self.entc_clean_grid = sc.Transformation(
        #    "FGTV: All Fugitive Emissions Transformations", 
        #    self.all_transformations_fgtv, 
        #    attr_strat
        #)

        self.entc_renewable_electricity = sc.Transformation(
            "ENTC: 95% of electricity is generated by renewables in 2050", 
            self.transformation_entc_renewables_target, 
            attr_strategy
        )
        all_transformations.append(self.entc_renewable_electricity)


        ##  FGTV

        self.fgtv_all = sc.Transformation(
            "FGTV: All Fugitive Emissions Transformations", 
            self.transformation_fgtv_all, 
            attr_strategy
        )
        all_transformations.append(self.entc_renewable_electricity)


        self.fgtv_maximize_flaring = sc.Transformation(
            "FGTV: Maximize flaring", 
            self.transformation_fgtv_maximize_flaring, 
            attr_strategy
        )
        all_transformations.append(self.entc_renewable_electricity)


        self.fgtv_minimize_leaks = sc.Transformation(
            "FGTV: Minimize leaks", 
            self.transformation_fgtv_minimize_leaks, 
            attr_strategy
        )
        all_transformations.append(self.entc_renewable_electricity)

        ##  INEN
        




        ##  TRNS/TRDE

        self.trde_reduce_demand = sc.Transformation(
            "TRNS: Reduce demand for transport", 
            self.transformation_trde_reduce_demand, 
            attr_strategy
        )
        all_transformations.append(self.trde_reduce_demand)


        self.trns_all = sc.Transformation(
            "TRNS: All Transportation Transformations", 
            self.transformation_trns_all, 
            attr_strategy
        )
        all_transformations.append(self.trns_all)


        self.trns_all_with_clean_grid = sc.Transformation(
            "TRNS: All Transportation Transformations with Renewable Grid and Green Hydrogen", 
            self.transformation_trns_all_with_clean_grid, 
            attr_strategy
        )
        all_transformations.append(self.trns_all_with_clean_grid)


        self.trns_bundle_demand_management = sc.Transformation(
            "TRNS: Demand management bundle", 
            self.transformation_trns_bundle_demand_management, 
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_demand_management)


        self.trns_bundle_efficiency = sc.Transformation(
            "TRNS: Efficiency bundle", 
            self.transformation_trns_bundle_efficiency, 
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_efficiency)


        self.trns_bundle_fuel_swtich = sc.Transformation(
            "TRNS: Fuel switch bundle", 
            self.transformation_trns_bundle_fuel_switch, 
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_fuel_swtich)


        self.trns_bundle_mode_shift = sc.Transformation(
            "TRNS: Mode shift bundle", 
            self.transformation_trns_bundle_mode_shift, 
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_mode_shift)


        self.trns_electrify_light_duty_road = sc.Transformation(
            "TRNS: Electrify light duty road transport", 
            self.transformation_trns_electrify_road_light_duty, 
            attr_strategy
        )
        all_transformations.append(self.trns_electrify_light_duty_road)


        self.trns_electrify_rail = sc.Transformation(
            "TRNS: Electrify rail", 
            self.transformation_trns_electrify_rail, 
            attr_strategy
        )
        all_transformations.append(self.trns_electrify_rail)


        self.trns_fuel_switch_maritime = sc.Transformation(
            "TRNS: Fuel switch maritime", 
            self.transformation_trns_fuel_switch_maritime, 
            attr_strategy
        )
        all_transformations.append(self.trns_fuel_switch_maritime)


        self.trns_fuel_switch_medium_duty_road = sc.Transformation(
            "TRNS: Fuel switch medium duty road transport", 
            self.transformation_trns_fuel_switch_road_medium_duty, 
            attr_strategy
        )
        all_transformations.append(self.trns_fuel_switch_medium_duty_road)


        self.trns_increase_efficiency_electric = sc.Transformation(
            "TRNS: Increase transportation electricity energy efficiency", 
            self.transformation_trns_increase_efficiency_electric, 
            attr_strategy
        )
        all_transformations.append(self.trns_increase_efficiency_electric)


        self.trns_increase_efficiency_non_electric = sc.Transformation(
            "TRNS: Increase transportation non-electricity energy efficiency", 
            self.transformation_trns_increase_efficiency_non_electric, 
            attr_strategy
        )
        all_transformations.append(self.trns_increase_efficiency_non_electric)


        self.trns_increase_occupancy_light_duty = sc.Transformation(
            "TRNS: Increase occupancy for private vehicles", 
            self.transformation_trns_increase_occupancy_light_duty, 
            attr_strategy
        )
        all_transformations.append(self.trns_increase_occupancy_light_duty)


        self.trns_mode_shift_freight = sc.Transformation(
            "TRNS: Mode shift freight", 
            self.transformation_trns_mode_shift_freight, 
            attr_strategy
        )
        all_transformations.append(self.trns_mode_shift_freight)


        self.trns_mode_shift_public_private = sc.Transformation(
            "TRNS: Mode shift passenger vehicles to others", 
            self.transformation_trns_mode_shift_public_private, 
            attr_strategy
        )
        all_transformations.append(self.trns_mode_shift_public_private)


        self.trns_mode_shift_regional = sc.Transformation(
            "TRNS: Mode shift regional passenger travel", 
            self.transformation_trns_mode_shift_regional, 
            attr_strategy
        )
        all_transformations.append(self.trns_mode_shift_regional)

    

        ## specify dictionary of transformations

        dict_transformations = dict(
            (x.id, x) for x in all_transformations
        )
        self.dict_transformations = dict_transformations




    ##############################
    #    CCSQ TRANSFORMATIONS    #
    ##############################






    ##############################
    #    ENTC TRANSFORMATIONS    #
    ##############################

    def transformation_support_entc_clean_grid(self,
        df_entc: pd.DataFrame,
        strat: int,
        include_hydrogen = True,
    ) -> pd.DataFrame:
        """
        Function used to implement "clean grid" transformation (shared 
            repeatability), which includes 95% renewable energy target and green 
            hydrogen. Shared across numerous ENTC and EN functions. Set
            `include_hydrogen = False` to exclude the green hydrogen component.
        """
        # ENTC: 95% of today's fossil-fuel electricity is generated by renewables in 2050
        df_strat_cur = self.transformation_entc_renewables_target(
            df_entc,
            strat
        )

        # ENTC: add green hydrogen
        df_strat_cur = (
            self.transformation_support_entc_green_hydrogen(df_strat_cur, strat)
            if include_hydrogen
            else df_strat_cur
        )

        return df_strat_cur


    
    def transformation_support_entc_green_hydrogen(self,
        df_entc: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement "green hydrogen" transformation requirements by forcing at 
            least 95% of hydrogen production to come from electrolysis.
        """

        df_strat_cur = adt.transformation_entc_hydrogen_electrolysis(
            df_entc,
            0.95,
            self.vec_implementation_ramp,
            self.model_attributes,
            self.model_elecricity,
            strategy_id = strat
        )

        return df_strat_cur



    def transformation_entc_renewables_target(self,
        df_entc: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "renewables target" transformation (shared repeatability),
            which includes 95% renewable energy target and green hydrogen
        """

        df_strat_cur = adt.transformation_entc_renewable_target(
            df_entc,
            0.95,
            self.cats_renewable,
            self.vec_implementation_ramp,
            self.model_attributes,
            self.model_electricity,
            dict_cats_entc_max_investment = self.dict_entc_renewable_target_cats_max_investment,
            magnitude_renewables = self.dict_entc_renewable_target_msp,
            strategy_id = strat
        )

        return df_strat_cur


        



    ##############################
    #    FGTV TRANSFORMATIONS    #
    ##############################

    def transformation_fgtv_all(self,
        df_fgtv: pd.DataFrame,
        strat: int
    ) -> pd.DataFrame:
        """
        Implement all fugitive emission transformations on data frame df_fgtv.
        """
        # FGTV: MINIMIZE LEAKS STRATEGY (FGTV)
        df_strat_cur = self.transformation_fgtv_minimize_leaks(df_fgtv, strat)

        # FGTV: MAXIMIZE FLARING STRATEGY (FGTV)
        df_strat_cur = self.transformation_fgtv_maximize_flaring(df_strat_cur, strat)

        return df_strat_cur


        
    def transformation_fgtv_maximize_flaring(self,
        df_input: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Maximize Flaring" FGTV transformation on input DataFrame
            df_input
        """
        df_strat_cur = adt.transformation_fgtv_maximize_flaring(
            df_input,
            0.8, 
            self.vec_implementation_ramp,
            self.model_attributes,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur



    def transformation_fgtv_minimize_leaks(self,
        df_input: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Minimize Leaks" FGTV transformation on input DataFrame
            df_input
        """
        df_strat_cur = adt.transformation_fgtv_reduce_leaks(
            df_input,
            0.8, 
            self.vec_implementation_ramp,
            self.model_attributes,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur






    ##############################
    #    INEN TRANSFORMATIONS    #
    ##############################

    def transformation_fgtv_maximize_flaring(self,
        df_input: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Maximize Flaring" FGTV transformation on input DataFrame
            df_input
        """
        df_strat_cur = adt.transformation_fgtv_maximize_flaring(
            df_input,
            0.8, 
            self.vec_implementation_ramp,
            self.model_attributes,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur





    ##############################
    #    TRNS TRANSFORMATIONS    #
    ##############################

    def transformation_trde_reduce_demand(self,
        df_trde: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Reduce Demand" TRDE transformation on input DataFrame
            df_trde
        """

        df_out = adt.transformation_trde_reduce_demand(
            df_trns,
            0.25, 
            self.vec_implementation_ramp,
            self.model_attributes,
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out




    def transformation_trns_all(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement All TRNS transformations on input DataFrame df_trns
        """

        df_out = self.transformation_trns_bundle_demand_management(df_trns, strat)
        df_out = self.transformation_trns_bundle_efficiency(df_out, strat)
        df_out = self.transformation_trns_bundle_fuel_switch(df_out, strat)
        df_out = self.transformation_trns_bundle_mode_shift(df_out, strat)

        return df_out
    
    
    
    def transformation_trns_all_with_clean_grid(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement All TRNS transformations + clean grid (renewable targets + 
        green hydrogen) on input DataFrame df_trns
        """

        df_out = self.transformation_trns_all(df_trns, strat)
        df_out = self.transformation_support_entc_clean_grid(df_out, strat)

        return df_out
    
    
    
    def transformation_trns_bundle_demand_management(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "" TRNS transformation on input DataFrame
            df_trns
        """

        df_out = self.transformation_trde_reduce_demand(df_trns, strat)
        df_out = self.transformation_trns_increase_occupancy_light_duty(df_out, strat)

        return df_out
    
    
    
    
    def transformation_trns_bundle_efficiency(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "" TRNS transformation on input DataFrame
            df_trns
        """

        df_out = self.transformation_trns_increase_efficiency_electric(df_trns, strat)
        df_out = self.transformation_trns_increase_efficiency_non_electric(df_out, strat)

        return df_out
    
    
    
    def transformation_trns_bundle_fuel_switch(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel-Switch Bundle" TRNS transformation on input DataFrame
            df_trns
        """

        df_strat_cur = self.transformation_trns_electrify_road_light_duty(df_trns, strat)
        df_strat_cur = self.transformation_trns_electrify_rail(df_strat_cur, strat)
        df_strat_cur = self.transformation_trns_fuel_switch_maritime(df_strat_cur, strat)
        df_strat_cur = self.transformation_trns_fuel_switch_road_medium_duty(df_strat_cur, strat)
    
        return df_strat_cur

    
    
    def transformation_trns_bundle_mode_shift(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Mode Shift Bundle" TRNS transformation on input DataFrame
            df_trns
        """
        df_strat_cur = self.transformation_trns_mode_shift_freight(df_trns, strat)
        df_strat_cur = self.transformation_trns_mode_shift_public_private(df_strat_cur, strat)
        df_strat_cur = self.transformation_trns_mode_shift_regional(df_strat_cur, strat)

        return df_strat_cur



    def transformation_trns_electrify_road_light_duty(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Electrify Light-Duty" TRNS transformation on input 
            DataFrame df_trns
        """

        df_out = adt.transformation_trns_fuel_shift_to_target(
            df_trns,
            0.7,
            self.vec_implementation_ramp,
            self.model_attributes,
            categories = ["road_light"],
            dict_modvar_specs = {
                self.model_energy.modvar_trns_fuel_fraction_electricity: 1.0
            },
            magnitude_type = "transfer_scalar",
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    
    def transformation_trns_electrify_rail(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Electrify Rail" TRNS transformation on input DataFrame
            df_trns
        """
        model_energy = self.model_energy

        df_out = adt.transformation_trns_fuel_shift_to_target(
            df_trns,
            0.25,
            self.vec_implementation_ramp,
            self.model_attributes,
            categories = ["rail_freight", "rail_passenger"],
            dict_modvar_specs = {
                model_energy.modvar_trns_fuel_fraction_electricity: 1.0
            },
            magnitude_type = "transfer_scalar",
            model_energy = model_energy,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    
    def transformation_trns_fuel_switch_maritime(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel-Swich Maritime" TRNS transformation on input 
            DataFrame df_trns
        """
        model_energy = self.model_energy

        # transfer 70% of diesel + gasoline to hydrogen
        df_out = adt.transformation_trns_fuel_shift_to_target(
            df_trns,
            0.7,
            self.vec_implementation_ramp,
            self.model_attributes,
            categories = ["water_borne"],
            dict_modvar_specs = {
                model_energy.modvar_trns_fuel_fraction_hydrogen: 1.0
            },
            modvars_source = [
                model_energy.modvar_trns_fuel_fraction_diesel,
                model_energy.modvar_trns_fuel_fraction_gasoline
            ],
            magnitude_type = "transfer_scalar",
            model_energy = model_energy,
            strategy_id = strat
        )

        # transfer remaining diesel + gasoline to hydrogen
        df_out = adt.transformation_trns_fuel_shift_to_target(
            df_out,
            1.0,
            self.vec_implementation_ramp,
            self.model_attributes,
            categories = ["water_borne"],
            dict_modvar_specs = {
                model_energy.modvar_trns_fuel_fraction_electricity: 1.0
            },
            modvars_source = [
                model_energy.modvar_trns_fuel_fraction_diesel,
                model_energy.modvar_trns_fuel_fraction_gasoline
            ],
            magnitude_type = "transfer_scalar",
            model_energy = model_energy,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    def transformation_trns_fuel_switch_road_medium_duty(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel-Switch Medium Duty" TRNS transformation on input 
            DataFrame df_trns
        """
        model_energy = self.model_energy

        # transfer 70% of diesel + gasoline to electricity
        df_out = adt.transformation_trns_fuel_shift_to_target(
            df_trns,
            0.7,
            self.vec_implementation_ramp,
            self.model_attributes,
            categories = ["road_heavy_freight", "road_heavy_regional", "public"],
            dict_modvar_specs = {
                model_energy.modvar_trns_fuel_fraction_electricity: 1.0
            },
            modvars_source = [
                model_energy.modvar_trns_fuel_fraction_diesel,
                model_energy.modvar_trns_fuel_fraction_gasoline
            ],
            magnitude_type = "transfer_scalar",
            model_energy = model_energy,
            strategy_id = strat
        )

        # transfer remaining diesel + gasoline to hydrogen
        df_out = adt.transformation_trns_fuel_shift_to_target(
            df_out,
            1.0,
            self.vec_implementation_ramp,
            self.model_attributes,
            categories = ["road_heavy_freight", "road_heavy_regional", "public"],
            dict_modvar_specs = {
                model_energy.modvar_trns_fuel_fraction_hydrogen: 1.0
            },
            modvars_source = [
                model_energy.modvar_trns_fuel_fraction_diesel,
                model_energy.modvar_trns_fuel_fraction_gasoline
            ],
            magnitude_type = "transfer_scalar",
            model_energy = model_energy,
            strategy_id = strat
        )
    
        return df_out
    

    
    def transformation_trns_increase_efficiency_electric(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Electric Efficiency" TRNS transformation on 
            input DataFrame df_trns
        """
        df_out = adt.transformation_trns_increase_energy_efficiency_electric(
            df_trns,
            0.25, 
            self.vec_implementation_ramp,
            self.model_attributes,
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out



    def transformation_trns_increase_efficiency_non_electric(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Non-Electric Efficiency" TRNS transformation on 
            input DataFrame df_trns
        """
        df_out = adt.transformation_trns_increase_energy_efficiency_non_electric(
            df_trns,
            0.25, 
            self.vec_implementation_ramp,
            self.model_attributes,
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out



    def transformation_trns_increase_occupancy_light_duty(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Vehicle Occupancy" TRNS transformation on input 
            DataFrame df_trns
        """

        df_out = adt.transformation_trns_increase_vehicle_occupancy(
            df_trns,
            0.25, 
            self.vec_implementation_ramp,
            self.model_attributes,
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out



    def transformation_trns_mode_shift_freight(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Mode Shift Freight" TRNS transformation on input 
            DataFrame df_trns
        """
        df_out = adt.transformation_general(
            df_trns,
            self.model_attributes,
            {
                self.model_energy.modvar_trns_modeshare_freight: {
                    "bounds": (0, 1),
                    "magnitude": 0.2,
                    "magnitude_type": "transfer_value_scalar",
                    "categories_source": ["aviation", "road_heavy_freight"],
                    "categories_target": {
                        "rail_freight": 1.0
                    },
                    "vec_ramp": self.vec_implementation_ramp
                }
            },
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out



    def transformation_trns_mode_shift_public_private(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Mode Shift Passenger Vehicles to Others" TRNS 
            transformation on input DataFrame df_trns
        """

        df_out = adt.transformation_general(
            df_trns,
            self.model_attributes,
            {
                self.model_energy.modvar_trns_modeshare_public_private: {
                    "bounds": (0, 1),
                    "magnitude": 0.3,
                    "magnitude_type": "transfer_value_scalar",
                    "categories_source": ["road_light"],
                    "categories_target": {
                        "human_powered": (1/6),
                        "powered_bikes": (2/6),
                        "public": 0.5
                    },
                    "vec_ramp": self.vec_implementation_ramp
                }
            },
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    
    def transformation_trns_mode_shift_regional(self,
        df_trns: pd.DataFrame,
        strat: int,
    ) -> pd.DataFrame:
        """
        Implement the "Mode Shift Regional Travel" TRNS transformation on input 
            DataFrame df_trns
        """

        df_out = adt.transformation_general(
            df_trns,
            self.model_attributes,
            {
                self.model_energy.modvar_trns_modeshare_regional: {
                    "bounds": (0, 1),
                    "magnitude": 0.25,
                    "magnitude_type": "transfer_value_scalar",
                    "categories_source": ["aviation"],
                    "categories_target": {
                        "rail_passenger": 0.5,
                        "road_heavy_regional": 0.5
                    },
                    "vec_ramp": self.vec_implementation_ramp
                }
            },
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out