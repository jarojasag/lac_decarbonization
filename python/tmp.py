##############################
    #    CCSQ TRANSFORMATIONS    #
    ##############################

    def transformation_ccsq_increase_air_capture(self,
        df_input: Union[pd.DataFrame, None] = None,
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement the "Increase Direct Air Capture" CCSQ transformation on input 
            DataFrame df_input
        """

        df_input = (
            self.base_inputs 
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

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

    def transformation_entc_change_msp_max(self,
        df_input: Union[pd.DataFrame, None] = None,
        cats_to_cap: Union[List[str], None],
        strat: Union[int, None] = None,
    ) -> pd.DataFrame:
        """
        Implement a transformation for the baseline to resolve constraint
            conflicts between TotalTechnologyAnnualActivityUpperLimit/
            TotalTechnologyAnnualActivityLowerLimit if MinShareProduction is 
            Specified. 

        This transformation will turn on the MSP Max method in ElectricEnergy,
            which will cap electric production (for a given technology) at the 
            value estimated for the last non-engaged time period. 
            
        E.g., suppose a technology has the following estimated electricity 
            production (estimated endogenously and excluding demands for ENTC) 
            and associated value of msp_max (stored in the "Maximum Production 
            Increase Fraction to Satisfy MinShareProduction Electricity" 
            SISEPUEDE model variable):

            time_period     est. production     msp_max
                            implied by MSP     
            -----------     ---------------     -------
            0               10                  -999
            1               10.5                -999
            2               11                  -999
            3               11.5                -999
            4               12                  0
            .
            .
            .
            n - 2           23                  0
            n - 1           23.1                0

            Then the MSP for this technology would be adjusted to never exceed 
            the value of 11.5, which was found at time_period 3. msp_max = 0
            means that a 0% increase is allowable in the MSP passed to NemoMod,
            so the specified MSP trajectory (which is passed to NemoMod) is 
            adjusted to reflect this change.
        
        NOTE: Only the *first value* after that last non-specified time period
            affects this variable. Using the above table as an example, entering 
            0 in time_period 4 and 1 in time_period 5 means that 0 is used for 
            all time_periods on and after 4.
        

        Function Arguments
        ------------------
        - df_input: input data frame containing baseline trajectories

        Keyword Arguments
        -----------------
        - cats_to_cap: list of categories to cap using the transformation
            implementation vector self.vec_implementation_ramp. If None, 
            defaults to pp_hydropower
        - strat: strategy number to pass
        - **kwargs: passed to ade.transformations_general()
        """
       
        df_input = (
            self.base_inputs 
            if not isinstance(df_input, pd.DataFrame) 
            else df_input
        )

        # CHECK CATEGORIES TO CAP

        cats_to_cap = (
            ["pp_hydropower"]
            if cats_to_cap is None
            else cats_to_cap
        )
        cats_to_cap = [x for x in self.attribute_technology if x in cats_to_cap]
        if cats_to_cap is None:
            return None

        # build dictionary if valid
        dict_cat_to_vector = dict(
            (x, self.vec_msp_resolution_cap)
            for x in cats_to_cap
        )


        # SET UP INPUT DICTIONARY

        df_out = adt.transformation_entc_change_msp_max(
            df_input,
            dict_cat_to_vector,
            self.model_electricity,
            drop_flag = self.model_electricity.drop_flag_tech_capacities,
            strategy_id = strat,
            **kwargs
        )
 
        return df_out

    
    
    def transformation_entc_least_cost(self,
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
            strategy_id = strat,
        )

        return df_strat_cur



    def transformation_entc_renewables_target(self,
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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
        df_input: Union[pd.DataFrame, None] = None,
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