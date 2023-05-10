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



class TransformationsAFOLU:
    """
    Build AFOLU transformations using general transformations defined in
        auxiliary_definitions_transformations. Wraps more general forms from 
        auxiliary_definitions_transformations into functions and classes
        with shared ramps, paramters, and build functionality.

    NOTE: To update transformations, users need to follow three steps:

        1. Use or create a function in auxiliarty_definitions_transformations, 
            which modifies an input DataFrame, using the ModelAttributes object
            and any approprite SISEPUEDE models. 

        2. If the transformation is not the composition of existing 
            transformation functions, create a transformation definition 
            function using the following functional template:

             def transformation_sabv_###(
                df_input: pd.DataFrame,
                strat: Union[int, None] = None,
                **kwargs
             ) -> pd.DataFrame:
                #DOCSTRING
                ...
                return df_out

            This function is where parameterizations are defined (can be passed 
                through dict_config too if desired, but not done here)

            If using the composition of functions, can leverage the 
            sc.Transformation composition functionality, which lets the user
            enter lists of functions (see ?sc.Transformation for more 
            information)

        3. Finally, define the Transformation object using the 
            `sc.Transformation` class, which connects the function to the 
            Strategy name in attribute_strategy_id, assigns an id, and 
            simplifies the organization and running of strategies. 


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
        field_region: Union[str, None] = None,
		fp_nemomod_temp_sqlite_db: Union[str, None] = None,
		logger: Union[logging.Logger, None] = None,
    ):

        self.logger = logger

        self._initialize_attributes(field_region, model_attributes)
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


    
    def get_entc_dict_renewable_target_msp(self,
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



    def get_inen_parameters(self,
        dict_config: Union[Dict, None] = None,
    ) -> List[str]:
        """
        Get parameters for the implementation of transformations. Returns a 
            tuple with the following elements (dictionary keys, if present, are 
            shown within after comments; otherwise, calculated internally):

            (
                cats_inen_high_heat, # key "categories_inen_high_heat",
                cats_inen_not_high_heat,
                frac_inen_high_temp_elec_hydg, # key "frac_inen_low_temp_elec"
                frac_inen_low_temp_elec, # key "frac_inen_low_temp_elec"
                frac_inen_shift_denom, 
            )
        
        If dict_config is None, uses self.config.

        NOTE: Requires keys in dict_config to set. If not found, will set the 
            following defaults:
                * cats_inen_high_heat: [
                    "cement", 
                    "chemicals", 
                    "glass", 
                    "lime_and_carbonite", 
                    "metals"
                ]
                * cats_inen_not_high_heat: derived from INEN Fuel Fraction 
                    variables and cats_inen_high_heat (complement)
                * frac_inen_high_temp_elec_hydg: (electrification and hydrogen
                    potential fractionation of industrial energy demand, 
                    targeted at high temperature demands: 50% of 1/2 of 90% of 
                    total INEN energy demand for each fuel)
                * frac_inen_low_temp_elec: 0.95*0.45 (electrification potential 
                    fractionation of industrial energy demand, targeted at low 
                    temperature demands: 95% of 1/2 of 90% of total INEN energy 
                    demand)
            
            The value of `frac_inen_shift_denom` is 
                frac_inen_low_temp_elec + 2*frac_inen_high_temp_elec_hydg


        Keyword Arguments
        -----------------
        - dict_config: dictionary mapping input configuration arguments to key 
            values. Must include the following keys:

            * categories_entc_renewable: list of categories to tag as renewable 
                for the Renewable Targets transformation.
        """

        attr_industry = self.model_attributes.dict_attributes.get("cat_industry")
        dict_config = self.config if not isinstance(dict_config, dict) else dict_config


        # INEN high heat and non-high heat categories
        default_cats_inen_high_heat = [
            x for x in attr_industry.key_values 
            if x in [
                "cement", 
                "chemicals", 
                "glass", 
                "lime_and_carbonite", 
                "metals"
            ]
        ]
        cats_inen_high_heat = dict_config.get(self.key_config_cats_inen_high_heat)
        cats_inen_high_heat = (
            default_cats_inen_high_heat
            if cats_inen_high_heat is None
            else [x for x in cats_inen_high_heat if x in attr_industry.key_values]
        )

        # get categories without high heat
        modvars_inen_fuel_switching = adt.transformation_inen_shift_modvars(
            None,
            None,
            None,
            self.model_attributes,
            return_modvars_only = True
        )
        cats_inen_fuel_switching = set({})
        for modvar in modvars_inen_fuel_switching:
            cats_inen_fuel_switching = cats_inen_fuel_switching | set(self.model_attributes.get_variable_categories(modvar))
        cats_inen_not_high_heat = sorted(list(cats_inen_fuel_switching - set(cats_inen_high_heat))) 

        # fraction of energy that can be moved to electric/hydrogen, representing high heat transfer
        default_frac_inen_high_temp_elec_hydg = 0.5*0.45
        frac_inen_high_temp_elec_hydg = dict_config.get(self.key_config_frac_inen_high_temp_elec_hydg)
        frac_inen_high_temp_elec_hydg = (
            default_frac_inen_high_temp_elec_hydg
            if not sf.isnumber(frac_inen_high_temp_elec_hydg)
            else frac_inen_high_temp_elec_hydg
        )

        # fraction of energy that can be moved to electric, representing low heat transfer
        default_frac_inen_low_temp_elec = 0.95*0.45
        frac_inen_low_temp_elec = dict_config.get(self.key_config_frac_inen_low_temp_elec)
        frac_inen_low_temp_elec = (
            default_frac_inen_low_temp_elec
            if not sf.isnumber(frac_inen_low_temp_elec)
            else frac_inen_low_temp_elec
        )

        frac_inen_shift_denom = frac_inen_low_temp_elec + 2*frac_inen_high_temp_elec_hydg

        
        # setup return
        tup_out = (
            cats_inen_high_heat,
            cats_inen_not_high_heat,
            frac_inen_high_temp_elec_hydg,
            frac_inen_low_temp_elec,
            frac_inen_shift_denom,
        )
        
        return tup_out



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



    def _initialize_attributes(self,
        field_region: Union[str, None],
        model_attributes: ma.ModelAttributes,
    ) -> None:
        """
        Initialize the model attributes object. Checks implementation and throws
            an error if issues arise. Sets the following properties

            * self.attribute_strategy
            * self.key_region
            * self.model_attributes
            * self.regions (support_classes.Regions object)
            * self.time_periods (support_classes.TimePeriods object)
        """

        # run checks and throw and
        error_q = False
        error_q = error_q | (model_attributes is None)
        if error_q:
            raise RuntimeError(f"Error: invalid specification of model_attributes in transformations_energy")

        # get strategy attribute, baseline strategy, and some fields
        attribute_strategy = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_strategy_id}")
        baseline_strategy = int(
            attribute_strategy.table[
                attribute_strategy.table["baseline_strategy_id"] == 1
            ][attribute_strategy.key].iloc[0]
        )
        field_region = model_attributes.dim_region if (field_region is None) else field_region

        # set some useful classes
        time_periods = sc.TimePeriods(model_attributes)
        regions = sc.Regions(model_attributes)


        ##  SET PROPERTIES
        
        self.attribute_strategy = attribute_strategy
        self.baseline_strategy = baseline_strategy
        self.key_region = field_region
        self.key_strategy = attribute_strategy.key
        self.model_attributes = model_attributes
        self.time_periods = time_periods
        self.regions = regions

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
        self.key_config_cats_inen_high_heat = "categories_inen_high_heat",
        self.key_config_dict_entc_renewable_target_msp = "dict_entc_renewable_target_msp"
        self.key_config_frac_inen_high_temp_elec_hydg = "frac_inen_low_temp_elec"
        self.key_config_frac_inen_low_temp_elec = "frac_inen_low_temp_elec"
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

        dict_entc_renewable_target_msp = self.get_entc_dict_renewable_target_msp(cats_renewable)

        # get some INEN paraemeters ()
        (
            cats_inen_high_heat,
            cats_inen_not_high_heat,
            frac_inen_high_temp_elec_hydg,
            frac_inen_low_temp_elec,
            frac_inen_shift_denom,
        ) = self.get_inen_parameters()


        ##  SET PROPERTIES

        self.cats_entc_max_investment_ramp = cats_entc_max_investment_ramp
        self.cats_inen_high_heat = cats_inen_high_heat
        self.cats_inen_not_high_heat = cats_inen_not_high_heat
        self.cats_renewable = cats_renewable
        self.dict_entc_renewable_target_msp = dict_entc_renewable_target_msp
        self.frac_inen_high_temp_elec_hydg = frac_inen_high_temp_elec_hydg
        self.frac_inen_low_temp_elec = frac_inen_low_temp_elec
        self.frac_inen_shift_denom = frac_inen_shift_denom
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



    def _initialize_transformations(self,
    ) -> None:
        """
        Initialize all sc.Transformation objects used to manage the construction
            of transformations. Note that each transformation == a strategy.

        NOTE: This is the key function mapping each function to a transformation
            name.
            
        Sets the following properties:

            * self.dict_transformations
            * self.transformation_***
        """

        attr_strategy = self.attribute_strategy
        all_transformations = []
        dict_transformations = {}


        ##############
        #    CCSQ    #
        ##############

        self.ccsq_increase_air_capture = sc.Transformation(
            "CCSQ: Increase direct air capture", 
            self.transformation_ccsq_increase_air_capture, 
            attr_strategy
        )
        all_transformations.append(self.ccsq_increase_air_capture)


        self.ccsq_increase_air_capture_with_rep = sc.Transformation(
            "CCSQ: Increase direct air capture with renewable energy production", 
            [
                self.transformation_ccsq_increase_air_capture,
                self.transformation_support_entc_clean_grid
            ],
            attr_strategy
        )
        all_transformations.append(self.ccsq_increase_air_capture_with_rep)



        ##############
        #    ENTC    #
        ##############

        self.entc_all = sc.Transformation(
            "EN: All Energy transformations", 
            [
                self.transformation_ccsq_increase_air_capture,
                self.transformation_entc_reduce_transmission_losses,
                self.transformation_entc_renewables_target,
                self.transformation_fgtv_maximize_flaring,
                self.transformation_fgtv_minimize_leaks,
                self.transformation_inen_fuel_switch_low_and_high_temp, # required instead of the two separate functions
                self.transformation_inen_maximize_efficiency_energy,
                self.transformation_inen_maximize_efficiency_production,
                self.transformation_scoe_fuel_switch_electrify,
                self.transformation_scoe_increase_applicance_efficiency,
                self.transformation_scoe_reduce_heat_energy_demand,
                self.transformation_trde_reduce_demand,
                self.transformation_trns_electrify_road_light_duty,
                self.transformation_trns_electrify_rail,
                self.transformation_trns_fuel_switch_maritime,
                self.transformation_trns_fuel_switch_road_medium_duty,
                self.transformation_trns_increase_efficiency_electric,
                self.transformation_trns_increase_efficiency_non_electric,
                self.transformation_trns_increase_occupancy_light_duty,
                self.transformation_trns_mode_shift_freight,
                self.transformation_trns_mode_shift_public_private,
                self.transformation_trns_mode_shift_regional,
                self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.entc_all)


        self.entc_bundle_efficiency = sc.Transformation(
            "EN: Efficiency bundle", 
            [
                self.transformation_inen_fuel_switch_low_temp_to_heat_pump,
                self.transformation_inen_maximize_efficiency_energy,
                self.transformation_scoe_fuel_switch_electrify,
                self.transformation_scoe_increase_applicance_efficiency,
                self.transformation_scoe_reduce_heat_energy_demand,
                self.transformation_trns_increase_efficiency_electric,
                self.transformation_trns_increase_efficiency_non_electric
            ], 
            attr_strategy
        )
        all_transformations.append(self.entc_bundle_efficiency)


        self.entc_bundle_efficiency_with_rep = sc.Transformation(
            "EN: Efficiency bundle with renewable energy production", 
            [
                self.transformation_inen_fuel_switch_low_temp_to_heat_pump,
                self.transformation_inen_maximize_efficiency_energy,
                self.transformation_scoe_fuel_switch_electrify,
                self.transformation_scoe_increase_applicance_efficiency,
                self.transformation_scoe_reduce_heat_energy_demand,
                self.transformation_trns_increase_efficiency_electric,
                self.transformation_trns_increase_efficiency_non_electric,
                self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.entc_bundle_efficiency_with_rep)


        self.entc_bundle_fuel_switch = sc.Transformation(
            "EN: Fuel switch bundle", 
            [
                self.transformation_inen_fuel_switch_low_and_high_temp,
                self.transformation_scoe_fuel_switch_electrify,
                self.transformation_trns_electrify_road_light_duty,
                self.transformation_trns_electrify_rail,
                self.transformation_trns_fuel_switch_maritime,
                self.transformation_trns_fuel_switch_road_medium_duty
            ], 
            attr_strategy
        )
        all_transformations.append(self.entc_bundle_fuel_switch)


        self.entc_bundle_fuel_switch_with_rep = sc.Transformation(
            "EN: Fuel switch bundle with renewable energy production", 
            [
                self.transformation_inen_fuel_switch_low_and_high_temp,
                self.transformation_scoe_fuel_switch_electrify,
                self.transformation_trns_electrify_road_light_duty,
                self.transformation_trns_electrify_rail,
                self.transformation_trns_fuel_switch_maritime,
                self.transformation_trns_fuel_switch_road_medium_duty,
                self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.entc_bundle_fuel_switch_with_rep)


        self.entc_least_cost = sc.Transformation(
            "ENTC: Least cost solution", 
            self.transformation_entc_least_cost, 
            attr_strategy
        )
        all_transformations.append(self.entc_least_cost)

        
        self.entc_reduce_transmission_losses = sc.Transformation(
            "ENTC: Reduce transmission losses", 
            self.transformation_entc_reduce_transmission_losses, 
            attr_strategy
        )
        all_transformations.append(self.entc_reduce_transmission_losses)


        self.entc_reduce_transmission_losses_with_rep = sc.Transformation(
            "ENTC: Reduce transmission losses with renewable energy production",
            [
                self.transformation_entc_reduce_transmission_losses,
                self.transformation_support_entc_clean_grid
            ],
            attr_strategy
        )
        all_transformations.append(self.entc_reduce_transmission_losses_with_rep)


        self.entc_renewable_electricity = sc.Transformation(
            "ENTC: 95% of electricity is generated by renewables in 2050", 
            self.transformation_entc_renewables_target, 
            attr_strategy
        )
        all_transformations.append(self.entc_renewable_electricity)



        ##############
        #    FGTV    #
        ##############

        self.fgtv_all = sc.Transformation(
            "FGTV: All Fugitive Emissions transformations", 
            [
                self.transformation_fgtv_maximize_flaring,
                self.transformation_fgtv_minimize_leaks
            ], 
            attr_strategy
        )
        all_transformations.append(self.fgtv_all)


        self.fgtv_all_with_rep = sc.Transformation(
            "FGTV: All Fugitive Emissions transformations with renewable energy production", 
            [
                self.transformation_fgtv_maximize_flaring,
                self.transformation_fgtv_minimize_leaks,
                self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.fgtv_all_with_rep)


        self.fgtv_maximize_flaring = sc.Transformation(
            "FGTV: Maximize flaring", 
            self.transformation_fgtv_maximize_flaring, 
            attr_strategy
        )
        all_transformations.append(self.fgtv_maximize_flaring)


        self.fgtv_maximize_flaring_with_rep = sc.Transformation(
            "FGTV: Maximize flaring with renewable energy production", 
            [
                self.transformation_fgtv_maximize_flaring,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.fgtv_maximize_flaring_with_rep)


        self.fgtv_minimize_leaks = sc.Transformation(
            "FGTV: Minimize leaks", 
            self.transformation_fgtv_minimize_leaks, 
            attr_strategy
        )
        all_transformations.append(self.fgtv_minimize_leaks)


        self.fgtv_minimize_leaks_with_rep = sc.Transformation(
            "FGTV: Minimize leaks with renewable energy production", 
            [
                self.transformation_fgtv_minimize_leaks,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.fgtv_minimize_leaks_with_rep)



        ##############
        #    INEN    #
        ##############

        self.inen_all = sc.Transformation(
            "INEN: All Industrial Energy transformations", 
            [
                self.transformation_inen_fuel_switch_low_and_high_temp, # use instead of both functions to avoid incorrect results w/func composition
                self.transformation_inen_maximize_efficiency_energy,
                self.transformation_inen_maximize_efficiency_production
            ], 
            attr_strategy
        )
        all_transformations.append(self.inen_all)


        self.inen_all_with_rep = sc.Transformation(
            "INEN: All Industrial Energy transformations with renewable energy production", 
            [
                self.transformation_inen_fuel_switch_low_and_high_temp, # use instead of both functions to avoid incorrect results w/func composition
                self.transformation_inen_maximize_efficiency_energy,
                self.transformation_inen_maximize_efficiency_production,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.inen_all_with_rep)


        self.inen_fuel_switch_high_temp = sc.Transformation(
            "INEN: Fuel switch medium and high-temp thermal processes to hydrogen and electricity", 
            self.transformation_inen_fuel_switch_high_temp, 
            attr_strategy
        )
        all_transformations.append(self.inen_fuel_switch_high_temp)


        self.inen_fuel_switch_high_temp_with_rep = sc.Transformation(
            "INEN: Fuel switch medium and high-temp thermal processes to hydrogen and electricity with renewable energy production", 
            [
                self.transformation_inen_fuel_switch_high_temp,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.inen_fuel_switch_high_temp_with_rep)
        

        self.inen_fuel_switch_low_temp_to_heat_pump = sc.Transformation(
            "INEN: Fuel switch low-temp thermal processes to industrial heat pumps", 
            self.transformation_inen_fuel_switch_low_temp_to_heat_pump, 
            attr_strategy
        )
        all_transformations.append(self.inen_fuel_switch_low_temp_to_heat_pump)


        self.inen_fuel_switch_low_temp_to_heat_pump_with_rep = sc.Transformation(
            "INEN: Fuel switch low-temp thermal processes to industrial heat pumps with renewable energy production", 
            [
                self.transformation_inen_fuel_switch_low_temp_to_heat_pump,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.inen_fuel_switch_low_temp_to_heat_pump_with_rep)

        
        self.inen_maximize_energy_efficiency = sc.Transformation(
            "INEN: Maximize industrial energy efficiency", 
            self.transformation_inen_maximize_efficiency_energy, 
            attr_strategy
        )
        all_transformations.append(self.inen_maximize_energy_efficiency)


        self.inen_maximize_energy_efficiency_with_rep = sc.Transformation(
            "INEN: Maximize industrial energy efficiency with renewable energy production", 
            [
                self.transformation_inen_maximize_efficiency_energy,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.inen_maximize_energy_efficiency_with_rep)
        

        self.inen_maximize_production_efficiency = sc.Transformation(
            "INEN: Maximize industrial production efficiency", 
            self.transformation_inen_maximize_efficiency_production, 
            attr_strategy
        )
        all_transformations.append(self.inen_maximize_production_efficiency)


        self.inen_maximize_production_efficiency_with_rep = sc.Transformation(
            "INEN: Maximize industrial production efficiency with renewable energy production", 
            [
                self.transformation_inen_maximize_efficiency_production,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.inen_maximize_production_efficiency_with_rep)
        


        ##############
        #    SCOE    #
        ##############

        self.scoe_all = sc.Transformation(
            "SCOE: All Stationary Combustion and Other Energy transformations", 
            [
                self.transformation_scoe_fuel_switch_electrify,
                self.transformation_scoe_increase_applicance_efficiency,
                self.transformation_scoe_reduce_heat_energy_demand,
            ], 
            attr_strategy
        )
        all_transformations.append(self.scoe_all)


        self.scoe_all_with_rep = sc.Transformation(
            "SCOE: All Stationary Combustion and Other Energy transformations with renewable energy production", 
            [
                self.transformation_scoe_fuel_switch_electrify,
                self.transformation_scoe_increase_applicance_efficiency,
                self.transformation_scoe_reduce_heat_energy_demand,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.scoe_all_with_rep)


        self.scoe_fuel_switch_electrify = sc.Transformation(
            "SCOE: Switch to electricity for heat using heat pumps, electric stoves, etc.", 
            self.transformation_scoe_fuel_switch_electrify, 
            attr_strategy
        )
        all_transformations.append(self.scoe_fuel_switch_electrify)


        self.scoe_fuel_switch_electrify_with_rep = sc.Transformation(
            "SCOE: Switch to electricity for heat using heat pumps, electric stoves, etc. with renewable energy production", 
            [
                self.transformation_scoe_fuel_switch_electrify,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.scoe_fuel_switch_electrify_with_rep)
        

        self.scoe_increase_applicance_efficiency = sc.Transformation(
            "SCOE: Increase appliance efficiency", 
            self.transformation_scoe_increase_applicance_efficiency, 
            attr_strategy
        )
        all_transformations.append(self.scoe_increase_applicance_efficiency)


        self.scoe_increase_applicance_efficiency_with_rep = sc.Transformation(
            "SCOE: Increase appliance efficiency with renewable energy production", 
            [
                self.transformation_scoe_increase_applicance_efficiency,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.scoe_increase_applicance_efficiency_with_rep)
        

        self.scoe_reduce_heat_energy_demand = sc.Transformation(
            "SCOE: Reduce end-use demand for heat energy by improving building shell", 
            self.transformation_scoe_reduce_heat_energy_demand, 
            attr_strategy
        )
        all_transformations.append(self.scoe_reduce_heat_energy_demand)


        self.scoe_reduce_heat_energy_demand_with_rep = sc.Transformation(
            "SCOE: Reduce end-use demand for heat energy by improving building shell with renewable energy production", 
            [
                self.transformation_scoe_reduce_heat_energy_demand,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.scoe_reduce_heat_energy_demand_with_rep)



        ###################
        #    TRNS/TRDE    #
        ###################

        self.trde_reduce_demand = sc.Transformation(
            "TRNS: Reduce demand for transport", 
            self.transformation_trde_reduce_demand, 
            attr_strategy
        )
        all_transformations.append(self.trde_reduce_demand)


        self.trde_reduce_demand_with_rep = sc.Transformation(
            "TRNS: Reduce demand for transport with renewable energy production", 
            [
                self.transformation_trde_reduce_demand,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trde_reduce_demand_with_rep)

        
        self.trns_all = sc.Transformation(
            "TRNS: All Transportation transformations", 
            [
                self.transformation_trde_reduce_demand,
                self.transformation_trns_electrify_road_light_duty,
                self.transformation_trns_electrify_rail,
                self.transformation_trns_fuel_switch_maritime,
                self.transformation_trns_fuel_switch_road_medium_duty,
                self.transformation_trns_increase_efficiency_electric,
                self.transformation_trns_increase_efficiency_non_electric,
                self.transformation_trns_increase_occupancy_light_duty,
                self.transformation_trns_mode_shift_freight,
                self.transformation_trns_mode_shift_public_private,
                self.transformation_trns_mode_shift_regional
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_all)


        self.trns_all_with_rep = sc.Transformation(
            "TRNS: All Transportation transformations with renewable energy production", 
            [
                self.transformation_trde_reduce_demand,
                self.transformation_trns_electrify_road_light_duty,
                self.transformation_trns_electrify_rail,
                self.transformation_trns_fuel_switch_maritime,
                self.transformation_trns_fuel_switch_road_medium_duty,
                self.transformation_trns_increase_efficiency_electric,
                self.transformation_trns_increase_efficiency_non_electric,
                self.transformation_trns_increase_occupancy_light_duty,
                self.transformation_trns_mode_shift_freight,
                self.transformation_trns_mode_shift_public_private,
                self.transformation_trns_mode_shift_regional,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_all_with_rep)

        
        self.trns_bundle_demand_management = sc.Transformation(
            "TRNS: Demand management bundle", 
            [
                self.transformation_trde_reduce_demand,
                self.transformation_trns_increase_occupancy_light_duty
            ],
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_demand_management)


        self.trns_bundle_demand_management_with_rep = sc.Transformation(
            "TRNS: Demand management bundle with renewable energy production", 
            [
                self.transformation_trde_reduce_demand,
                self.transformation_trns_increase_occupancy_light_duty,
		        self.transformation_support_entc_clean_grid
            ],
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_demand_management_with_rep)

        
        self.trns_bundle_efficiency = sc.Transformation(
            "TRNS: Efficiency bundle", 
            [
                self.transformation_trns_increase_efficiency_electric,
                self.transformation_trns_increase_efficiency_non_electric
            ],
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_efficiency)


        self.trns_bundle_efficiency_with_rep = sc.Transformation(
            "TRNS: Efficiency bundle with renewable energy production", 
            [
                self.transformation_trns_increase_efficiency_electric,
                self.transformation_trns_increase_efficiency_non_electric,
                self.transformation_support_entc_clean_grid
            ],
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_efficiency_with_rep)

        
        self.trns_bundle_fuel_swtich = sc.Transformation(
            "TRNS: Fuel switch bundle", 
            [
                self.transformation_trns_electrify_road_light_duty,
                self.transformation_trns_electrify_rail,
                self.transformation_trns_fuel_switch_maritime,
                self.transformation_trns_fuel_switch_road_medium_duty
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_fuel_swtich)


        self.trns_bundle_fuel_swtich_with_rep = sc.Transformation(
            "TRNS: Fuel switch bundle with renewable energy production", 
            [
                self.transformation_trns_electrify_road_light_duty,
                self.transformation_trns_electrify_rail,
                self.transformation_trns_fuel_switch_maritime,
                self.transformation_trns_fuel_switch_road_medium_duty,
                self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_fuel_swtich_with_rep)

        
        self.trns_bundle_mode_shift = sc.Transformation(
            "TRNS: Mode shift bundle", 
            [
                self.transformation_trns_mode_shift_freight,
                self.transformation_trns_mode_shift_public_private,
                self.transformation_trns_mode_shift_regional
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_mode_shift)


        self.trns_bundle_mode_shift_with_rep = sc.Transformation(
            "TRNS: Mode shift bundle with renewable energy production", 
            [
                self.transformation_trns_mode_shift_freight,
                self.transformation_trns_mode_shift_public_private,
                self.transformation_trns_mode_shift_regional,
                self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_bundle_mode_shift_with_rep)

        
        self.trns_electrify_light_duty_road = sc.Transformation(
            "TRNS: Electrify light duty road transport", 
            self.transformation_trns_electrify_road_light_duty, 
            attr_strategy
        )
        all_transformations.append(self.trns_electrify_light_duty_road)


        self.trns_electrify_light_duty_road_with_rep = sc.Transformation(
            "TRNS: Electrify light duty road transport with renewable energy production", 
            [
                self.transformation_trns_electrify_road_light_duty,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_electrify_light_duty_road_with_rep)

        
        self.trns_electrify_rail = sc.Transformation(
            "TRNS: Electrify rail", 
            self.transformation_trns_electrify_rail, 
            attr_strategy
        )
        all_transformations.append(self.trns_electrify_rail)


        self.trns_electrify_rail_with_rep = sc.Transformation(
            "TRNS: Electrify rail with renewable energy production", 
            [
                self.transformation_trns_electrify_rail,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_electrify_rail_with_rep)

        
        self.trns_fuel_switch_maritime = sc.Transformation(
            "TRNS: Fuel switch maritime", 
            self.transformation_trns_fuel_switch_maritime, 
            attr_strategy
        )
        all_transformations.append(self.trns_fuel_switch_maritime)


        self.trns_fuel_switch_maritime_with_rep = sc.Transformation(
            "TRNS: Fuel switch maritime with renewable energy production", 
            [
                self.transformation_trns_fuel_switch_maritime,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_fuel_switch_maritime_with_rep)


        self.trns_fuel_switch_medium_duty_road = sc.Transformation(
            "TRNS: Fuel switch medium duty road transport", 
            self.transformation_trns_fuel_switch_road_medium_duty, 
            attr_strategy
        )
        all_transformations.append(self.trns_fuel_switch_medium_duty_road)


        self.trns_fuel_switch_medium_duty_road_with_rep = sc.Transformation(
            "TRNS: Fuel switch medium duty road transport with renewable energy production", 
            [
                self.transformation_trns_fuel_switch_road_medium_duty,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_fuel_switch_medium_duty_road_with_rep)


        self.trns_increase_efficiency_electric = sc.Transformation(
            "TRNS: Increase transportation electricity energy efficiency", 
            self.transformation_trns_increase_efficiency_electric, 
            attr_strategy
        )
        all_transformations.append(self.trns_increase_efficiency_electric)


        self.trns_increase_efficiency_electric_with_rep = sc.Transformation(
            "TRNS: Increase transportation electricity energy efficiency with renewable energy production", 
            [
                self.transformation_trns_increase_efficiency_electric,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_increase_efficiency_electric_with_rep)


        self.trns_increase_efficiency_non_electric = sc.Transformation(
            "TRNS: Increase transportation non-electricity energy efficiency", 
            self.transformation_trns_increase_efficiency_non_electric, 
            attr_strategy
        )
        all_transformations.append(self.trns_increase_efficiency_non_electric)


        self.trns_increase_efficiency_non_electric_with_rep = sc.Transformation(
            "TRNS: Increase transportation non-electricity energy efficiency with renewable energy production", 
            [
                self.transformation_trns_increase_efficiency_non_electric,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_increase_efficiency_non_electric_with_rep)


        self.trns_increase_occupancy_light_duty = sc.Transformation(
            "TRNS: Increase occupancy for private vehicles", 
            self.transformation_trns_increase_occupancy_light_duty, 
            attr_strategy
        )
        all_transformations.append(self.trns_increase_occupancy_light_duty)


        self.trns_increase_occupancy_light_duty_with_rep = sc.Transformation(
            "TRNS: Increase occupancy for private vehicles with renewable energy production", 
            [
                self.transformation_trns_increase_occupancy_light_duty,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_increase_occupancy_light_duty_with_rep)


        self.trns_mode_shift_freight = sc.Transformation(
            "TRNS: Mode shift freight", 
            self.transformation_trns_mode_shift_freight, 
            attr_strategy
        )
        all_transformations.append(self.trns_mode_shift_freight)


        self.trns_mode_shift_freight_with_rep = sc.Transformation(
            "TRNS: Mode shift freight with renewable energy production", 
            [
                self.transformation_trns_mode_shift_freight,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_mode_shift_freight_with_rep)


        self.trns_mode_shift_public_private = sc.Transformation(
            "TRNS: Mode shift passenger vehicles to others", 
            self.transformation_trns_mode_shift_public_private, 
            attr_strategy
        )
        all_transformations.append(self.trns_mode_shift_public_private)


        self.trns_mode_shift_public_private_with_rep = sc.Transformation(
            "TRNS: Mode shift passenger vehicles to others with renewable energy production", 
            [
                self.transformation_trns_mode_shift_public_private,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_mode_shift_public_private_with_rep)


        self.trns_mode_shift_regional = sc.Transformation(
            "TRNS: Mode shift regional passenger travel", 
            self.transformation_trns_mode_shift_regional, 
            attr_strategy
        )
        all_transformations.append(self.trns_mode_shift_regional)


        self.trns_mode_shift_regional_with_rep = sc.Transformation(
            "TRNS: Mode shift regional passenger travel with renewable energy production", 
            [
                self.transformation_trns_mode_shift_regional,
		        self.transformation_support_entc_clean_grid
            ], 
            attr_strategy
        )
        all_transformations.append(self.trns_mode_shift_regional_with_rep)

    

        ## specify dictionary of transformations

        dict_transformations = dict(
            (x.id, x) 
            for x in all_transformations
            if x.id in attr_strategy.key_values
        )
        all_transformations = sorted(list(dict_transformations.keys()))

        # SET ADDDITIONAL PROPERTIES

        self.all_transformations = all_transformations
        self.dict_transformations = dict_transformations





    ################################################
    ###                                          ###
    ###    OTHER NON-TRANSFORMATION FUNCTIONS    ###
    ###                                          ###
    ################################################

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



    def build_strategies_long(self,
        df_input: pd.DataFrame,
        include_base_df: bool = True,
        strategies: Union[List[str], List[int], None] = None,
    ) -> pd.DataFrame:
        """
        Return a long (by model_attributes.dim_strategy_id) concatenated
            DataFrame of transformations.

        Function Arguments
		------------------
        - df_input: baseline (untransformed) data frame to use to build 
            strategies. Must contain self.key_region and 
            self.model_attributes.dim_time_period in columns

        Keyword Arguments
		-----------------
        - include_base_df: include df_input in the output DataFrame? If False,
            only includes strategies associated with transformation 
        - strategies: strategies to build for. Can be a mixture of strategy_ids
            and names. If None, runs all available. 
        """

        # INITIALIZE STRATEGIES TO LOOP OVER

        strategies = (
            self.all_transformations
            if strategies is None
            else strategies
        )
        strategies = [self.get_strategy(x) for x in strategies]
        strategies = sorted([x.id for x in strategies if x is not None])
        n = len(strategies)

        # LOOP TO BUILD
        
        t0 = time.time()
        self._log(
            f"TransformationsEnergy.build_strategies_long() starting build of {n} strategies...",
            type_log = "info"
        )
        
        df_out = df_input.copy()
        if self.key_strategy not in df_out.columns:
            df_out[self.key_strategy] = self.baseline_strategy
        df_out = [df_out for x in range(len(strategies) + 1)]

        for i, strat in enumerate(strategies):
            t0_cur = time.time()
            transformation = self.get_strategy(strat)

            if transformation is not None:
                try:
                    df_out[i + 1] = transformation(df_out[i + 1])
                    t_elapse = sf.get_time_elapsed(t0_cur)
                    self._log(
                        f"\tSuccessfully built transformation {self.key_strategy} = {transformation.id} ('{transformation.name}') in {t_elapse} seconds.",
                        type_log = "info"
                    )

                except Exception as e: 
                    df_out[i + 1] = None
                    self._log(
                        f"\tError trying to build transformation {self.key_strategy} = {transformation.id}: {e}",
                        type_log = "error"
                    )

        # concatenate, log time elapsed and completion
        df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)

        t_elapse = sf.get_time_elapsed(t0)
        self._log(
            f"TransformationsEnergy.build_strategies_long() build complete in {t_elapse} seconds.",
            type_log = "info"
        )

        return df_out

        


    
    def get_strategy(self,
        strat: Union[int, str, None],
        field_strategy_name: str = "strategy",
    ) -> None:
        """
        Get strategy `strat` based on name or id. 
        
        If strat is None or an invalid valid of strat is entered, returns None; 
            otherwise, returns the sc.Transformation object. 
            
        Function Arguments
        ------------------
        - strat: strategy id or strategy name to use to retrieve 
            sc.Trasnformation object
            
        Keyword Arguments
        ------------------
         - field_strategy_name: field in strategy_id attribute table containing
            the strategy name
        """

        if not (sf.isnumber(strat, integer = True) | isinstance(strat, str)):
            return None

        dict_name_to_strat = self.attribute_strategy.field_maps.get(f"{field_strategy_name}_to_{self.attribute_strategy.key}")

        # check strategy
        strat = dict_name_to_strat.get(strat) if isinstance(strat, str) else strat
        out = (
            None
            if strat not in self.attribute_strategy.key_values
            else self.dict_transformations.get(strat)
        )
        
        return out



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





    ############################################
    ###                                      ###
    ###    BEGIN TRANSFORMATION FUNCTIONS    ###
    ###                                      ###
    ############################################

    ##############################
    #    CCSQ TRANSFORMATIONS    #
    ##############################

    def transformation_ccsq_increase_air_capture(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Direct Air Capture" CCSQ transformation on input 
            DataFrame df_input
        """
        df_strat_cur = adt.transformation_ccsq_increase_direct_air_capture(
            df_input,
            50,
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur



    ##############################
    #    ENTC TRANSFORMATIONS    #
    ##############################

    def transformation_entc_least_cost(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Least Cost" ENTC transformation on input DataFrame
            df_input
        """
        df_strat_cur = adt.transformation_entc_least_cost_solution(
            df_input,
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_electricity = self.model_electricity,
            strategy_id = strat
        )

        return df_strat_cur



    def transformation_entc_reduce_transmission_losses(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Reduce Transmission Losses" ENTC transformation on input 
            DataFrame df_input
        """
        df_strat_cur = adt.transformation_entc_specify_transmission_losses(
            df_input,
            0.06,
            self.vec_implementation_ramp,
            self.model_attributes,
            self.model_electricity,
            field_region = self.key_region,
            strategy_id = strat
        )

        return df_strat_cur



    def transformation_entc_renewables_target(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "renewables target" transformation (shared repeatability),
            which includes 95% renewable energy target and green hydrogen
        """

        df_strat_cur = adt.transformation_entc_renewable_target(
            df_input,
            0.95,
            self.cats_renewable,
            self.vec_implementation_ramp,
            self.model_attributes,
            self.model_electricity,
            dict_cats_entc_max_investment = self.dict_entc_renewable_target_cats_max_investment,
            field_region = self.key_region,
            magnitude_renewables = self.dict_entc_renewable_target_msp,
            strategy_id = strat
        )

        return df_strat_cur



    def transformation_support_entc_clean_grid(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
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
            df_input,
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
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement "green hydrogen" transformation requirements by forcing at 
            least 95% of hydrogen production to come from electrolysis.
        """

        df_strat_cur = adt.transformation_entc_hydrogen_electrolysis(
            df_input,
            0.95,
            self.vec_implementation_ramp,
            self.model_attributes,
            self.model_electricity,
            field_region = self.key_region,
            strategy_id = strat
        )

        return df_strat_cur



    ##############################
    #    FGTV TRANSFORMATIONS    #
    ##############################
        
    def transformation_fgtv_maximize_flaring(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
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
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur



    def transformation_fgtv_minimize_leaks(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
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
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur



    ##############################
    #    INEN TRANSFORMATIONS    #
    ##############################

    def transformation_inen_fuel_switch_high_temp(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel switch medium and high-temp thermal processes to 
            hydrogen and electricity" INEN transformation on input DataFrame 
            df_input
        """
        df_strat_cur = adt.transformation_inen_shift_modvars(
            df_input,
            2*self.frac_inen_high_temp_elec_hydg,
            self.vec_implementation_ramp,
            self.model_attributes,
            categories = self.cats_inen_high_heat,
            dict_modvar_specs = {
                self.model_energy.modvar_inen_frac_en_electricity: 0.5,
                self.model_energy.modvar_inen_frac_en_hydrogen: 0.5,
            },
            field_region = self.key_region,
            magnitude_relative_to_baseline = True,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur



    def transformation_inen_fuel_switch_low_and_high_temp(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel switch low-temp thermal processes to industrial heat 
            pumps" and "Fuel switch medium and high-temp thermal processes to 
            hydrogen and electricity" INEN transformations on input DataFrame 
            df_input (note: these must be combined in a new function instead of
            as a composition due to the electricity shift in high-heat 
            categories)
        """
        # set up fractions 
        frac_shift_hh_elec = self.frac_inen_low_temp_elec + self.frac_inen_high_temp_elec_hydg
        frac_shift_hh_elec /= self.frac_inen_shift_denom

        frac_shift_hh_hydrogen = self.frac_inen_high_temp_elec_hydg
        frac_shift_hh_hydrogen /= self.frac_inen_shift_denom


        # HIGH HEAT CATS ONLY
        # Fuel switch high-temp thermal processes + Fuel switch low-temp thermal processes to industrial heat pumps
        df_out = adt.transformation_inen_shift_modvars(
            df_input,
            self.frac_inen_shift_denom,
            self.vec_implementation_ramp, 
            self.model_attributes,
            categories = self.cats_inen_high_heat,
            dict_modvar_specs = {
                self.model_energy.modvar_inen_frac_en_electricity: frac_shift_hh_elec,
                self.model_energy.modvar_inen_frac_en_hydrogen: frac_shift_hh_hydrogen,
            },
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        # LOW HEAT CATS ONLY
        # + Fuel switch low-temp thermal processes to industrial heat pumps
        df_out = adt.transformation_inen_shift_modvars(
            df_out,
            self.frac_inen_shift_denom,
            self.vec_implementation_ramp, 
            self.model_attributes,
            categories = self.cats_inen_not_high_heat,
            dict_modvar_specs = {
                self.model_energy.modvar_inen_frac_en_electricity: 1.0
            },
            field_region = self.key_region,
            magnitude_relative_to_baseline = True,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_out



    def transformation_inen_fuel_switch_low_temp_to_heat_pump(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel switch low-temp thermal processes to industrial heat 
            pumps" INEN transformation on input DataFrame df_input
        """
        df_strat_cur = adt.transformation_inen_shift_modvars(
            df_input,
            self.frac_inen_low_temp_elec,
            self.vec_implementation_ramp,
            self.model_attributes,
            dict_modvar_specs = {
                self.model_energy.modvar_inen_frac_en_electricity: 1.0
            },
            field_region = self.key_region,
            magnitude_relative_to_baseline = True,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur



    def transformation_inen_maximize_efficiency_energy(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Maximize Industrial Energy Efficiency" INEN 
            transformation on input DataFrame df_input
        """
        df_strat_cur = adt.transformation_inen_maximize_energy_efficiency(
            df_input,
            0.3, 
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur



    def transformation_inen_maximize_efficiency_production(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Maximize Industrial Production Efficiency" INEN 
            transformation on input DataFrame df_input
        """
        df_strat_cur = adt.transformation_inen_maximize_production_efficiency(
            df_input,
            0.4, 
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur



    ##############################
    #    SCOE TRANSFORMATIONS    #
    ##############################

    def transformation_scoe_fuel_switch_electrify(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Switch to electricity for heat using heat pumps, electric 
            stoves, etc." INEN transformation on input DataFrame df_input
        """
        df_strat_cur = adt.transformation_scoe_electrify_category_to_target(
            df_input,
            0.95,
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur



    def transformation_scoe_reduce_heat_energy_demand(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Reduce end-use demand for heat energy by improving 
            building shell" SCOE transformation on input DataFrame df_input
        """
        df_strat_cur = adt.transformation_scoe_reduce_demand_for_heat_energy(
            df_input,
            0.5,
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur



    def transformation_scoe_increase_applicance_efficiency(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase appliance efficiency" SCOE transformation on 
            input DataFrame df_input
        """
        df_strat_cur = adt.transformation_scoe_reduce_demand_for_appliance_energy(
            df_input,
            0.5,
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )

        return df_strat_cur



    ##############################
    #    TRNS TRANSFORMATIONS    #
    ##############################

    def transformation_trde_reduce_demand(self,
        df_trde: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Reduce Demand" TRDE transformation on input DataFrame
            df_trde
        """

        df_out = adt.transformation_trde_reduce_demand(
            df_trde,
            0.25, 
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out



    def transformation_trns_electrify_road_light_duty(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Electrify Light-Duty" TRNS transformation on input 
            DataFrame df_input
        """

        df_out = adt.transformation_trns_fuel_shift_to_target(
            df_input,
            0.7,
            self.vec_implementation_ramp,
            self.model_attributes,
            categories = ["road_light"],
            dict_modvar_specs = {
                self.model_energy.modvar_trns_fuel_fraction_electricity: 1.0
            },
            field_region = self.key_region,
            magnitude_type = "transfer_scalar",
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    
    def transformation_trns_electrify_rail(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Electrify Rail" TRNS transformation on input DataFrame
            df_input
        """
        model_energy = self.model_energy

        df_out = adt.transformation_trns_fuel_shift_to_target(
            df_input,
            0.25,
            self.vec_implementation_ramp,
            self.model_attributes,
            categories = ["rail_freight", "rail_passenger"],
            dict_modvar_specs = {
                self.model_energy.modvar_trns_fuel_fraction_electricity: 1.0
            },
            field_region = self.key_region,
            magnitude_type = "transfer_scalar",
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    
    def transformation_trns_fuel_switch_maritime(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel-Swich Maritime" TRNS transformation on input 
            DataFrame df_input
        """
        model_energy = self.model_energy

        # transfer 70% of diesel + gasoline to hydrogen
        df_out = adt.transformation_trns_fuel_shift_to_target(
            df_input,
            0.7,
            self.vec_implementation_ramp,
            self.model_attributes,
            categories = ["water_borne"],
            dict_modvar_specs = {
                self.model_energy.modvar_trns_fuel_fraction_hydrogen: 1.0
            },
            field_region = self.key_region,
            modvars_source = [
                self.model_energy.modvar_trns_fuel_fraction_diesel,
                self.model_energy.modvar_trns_fuel_fraction_gasoline
            ],
            magnitude_type = "transfer_scalar",
            model_energy = self.model_energy,
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
                self.model_energy.modvar_trns_fuel_fraction_electricity: 1.0
            },
            field_region = self.key_region,
            modvars_source = [
                self.model_energy.modvar_trns_fuel_fraction_diesel,
                self.model_energy.modvar_trns_fuel_fraction_gasoline
            ],
            magnitude_type = "transfer_scalar",
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out
    
    
    
    def transformation_trns_fuel_switch_road_medium_duty(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Fuel-Switch Medium Duty" TRNS transformation on input 
            DataFrame df_input
        """
        model_energy = self.model_energy

        # transfer 70% of diesel + gasoline to electricity
        df_out = adt.transformation_trns_fuel_shift_to_target(
            df_input,
            0.7,
            self.vec_implementation_ramp,
            self.model_attributes,
            categories = ["road_heavy_freight", "road_heavy_regional", "public"],
            dict_modvar_specs = {
                self.model_energy.modvar_trns_fuel_fraction_electricity: 1.0
            },
            field_region = self.key_region,
            modvars_source = [
                self.model_energy.modvar_trns_fuel_fraction_diesel,
                self.model_energy.modvar_trns_fuel_fraction_gasoline
            ],
            magnitude_type = "transfer_scalar",
            model_energy = self.model_energy,
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
                self.model_energy.modvar_trns_fuel_fraction_hydrogen: 1.0
            },
            field_region = self.key_region,
            modvars_source = [
                self.model_energy.modvar_trns_fuel_fraction_diesel,
                self.model_energy.modvar_trns_fuel_fraction_gasoline
            ],
            magnitude_type = "transfer_scalar",
            model_energy = self.model_energy,
            strategy_id = strat
        )
    
        return df_out
    

    
    def transformation_trns_increase_efficiency_electric(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Electric Efficiency" TRNS transformation on 
            input DataFrame df_input
        """
        df_out = adt.transformation_trns_increase_energy_efficiency_electric(
            df_input,
            0.25, 
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out



    def transformation_trns_increase_efficiency_non_electric(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Non-Electric Efficiency" TRNS transformation on 
            input DataFrame df_input
        """
        df_out = adt.transformation_trns_increase_energy_efficiency_non_electric(
            df_input,
            0.25, 
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out



    def transformation_trns_increase_occupancy_light_duty(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Vehicle Occupancy" TRNS transformation on input 
            DataFrame df_input
        """

        df_out = adt.transformation_trns_increase_vehicle_occupancy(
            df_input,
            0.25, 
            self.vec_implementation_ramp,
            self.model_attributes,
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out



    def transformation_trns_mode_shift_freight(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Mode Shift Freight" TRNS transformation on input 
            DataFrame df_input
        """
        df_out = adt.transformation_general(
            df_input,
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
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out



    def transformation_trns_mode_shift_public_private(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Mode Shift Passenger Vehicles to Others" TRNS 
            transformation on input DataFrame df_input
        """

        df_out = adt.transformation_general(
            df_input,
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
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out
    
    

    def transformation_trns_mode_shift_regional(self,
        df_input: pd.DataFrame,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Mode Shift Regional Travel" TRNS transformation on input 
            DataFrame df_input
        """

        df_out = adt.transformation_general(
            df_input,
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
            field_region = self.key_region,
            model_energy = self.model_energy,
            strategy_id = strat
        )
        
        return df_out