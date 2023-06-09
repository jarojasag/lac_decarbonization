###
###   DEVELOP SOME SIMPLE CLASSES THAT CODIFY SHARED FUNCTIONALITY AND SUPPORT DATA PIPELINE
###

import geopy.distance
from model_attributes import *
import numpy as np
import pandas as pd
import support_functions as sf


class Regions:
    """
    Leverage some simple region actions based on model attributes. Supports the
        following actions for data:

        * Aggregation by World Bank global region
        * Finding the closest region (by population centroid)
        * Shared replacement dictionaries (IEA/WB/UN)
        * And more

    The Regions class is designed to provide convenient support for batch 
        integration of global and regional datasets into the SISEPUEDE 
        framework.
    """
    def __init__(self,
        model_attributes: ModelAttributes,
    ):

        self._initialize_region_properties(model_attributes)

        # initialize some default data source properties
        self._initialize_defaults_iea()

        return None



    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_defaults_iea(self,
    ) -> None:
        """
        Sets the following default properties, associated with fields in IEA
            data tables:

            * self.dict_iea_countries_lc_to_regions
            * self.field_iea_balance
            * self.field_iea_country
            * self.field_iea_product
            * self.field_iea_time
            * self.field_iea_unit
            * self.field_iea_value
        """

        self.dict_iea_countries_lc_to_regions = {
            "chinese_taipei": "taiwan",
            "czech_republic": "czechia",
            "hong_kong_(china)": "hong_kong",
            "korea": "republic_of_korea",
            "people's_republic_of_china": "china",
            "republic_of_north_macedonia": "north_macedonia",
            "republic_of_turkiye": "turkey",
            "republic_of_türkiye": "turkey",
            "slovak_republic": "slovakia",
            "united_states": "united_states_of_america",
        }

        self.field_iea_balance = "Balance"
        self.field_iea_country = "Country"
        self.field_iea_product = "Product"
        self.field_iea_time = "Time"
        self.field_iea_unit = "Unit"
        self.field_iea_value = "Value"     

        return None

        

    def _initialize_region_properties(self,
        model_attributes: ModelAttributes,
        field_year: str = "year",
    ) -> None:
        """
        Set the following properties:

            * self.all_isos
            * self.all_region
            * self.all_wb_regions
            * self.attributes
            * self.dict_iso_to_region
            * self.dict_region_to_iso
            * self.dict_region_to_wb_region
            * self.dict_wb_region_to_region
            * self.field_iso
            * self.field_lat
            * self.field_lon
            * self.field_wb_global_region
            * self.key

        """
        # some fields
        field_iso = "iso_alpha_3"
        field_lat = "latitude_population_centroid_2020"
        field_lon = "longitude_population_centroid_2020"
        field_wb_global_region = "world_bank_global_region"

        # set attributes and some dictionaries
        attributes = model_attributes.dict_attributes.get(f"{model_attributes.dim_region}")

        # initialize ISO dictionaries
        dict_region_to_iso = attributes.field_maps.get(f"{attributes.key}_to_{field_iso}")
        dict_iso_to_region = attributes.field_maps.get(f"{field_iso}_to_{attributes.key}")
        all_isos = sorted(list(dict_iso_to_region.keys()))

        # WorldBank region dictionaries
        dict_wb_region_to_region = sf.group_df_as_dict(
            attributes.table,
            [field_wb_global_region],
            fields_out_set = attributes.key
        )
        dict_region_to_wb_region = attributes.field_maps.get(f"{attributes.key}_to_{field_wb_global_region}")
        all_wb_regions = sorted(list(dict_wb_region_to_region.keys()))

    
        # assign as properties
        self.all_isos = all_isos
        self.all_regions = attributes.key_values
        self.all_wb_regions = all_wb_regions
        self.attributes = attributes
        self.dict_region_to_iso = dict_region_to_iso
        self.dict_iso_to_region = dict_iso_to_region
        self.dict_region_to_wb_region = dict_region_to_wb_region
        self.dict_wb_region_to_region = dict_wb_region_to_region
        self.field_iso = field_iso
        self.field_lat = field_lat
        self.field_lon = field_lon
        self.field_wb_global_region = field_wb_global_region
        self.key = attributes.key

        return None




    ########################
    #    CORE FUNCTIONS    #
    ########################

    # 
    def aggregate_df_by_wb_global_region(self,
        df_in: pd.DataFrame,
        global_wb_region: str,
        fields_group: List[str],
        dict_agg: Dict[str, str],
        field_iso: Union[str, None] = None,
    ) -> pd.DataFrame:
        """
        Get a regional average (for WB global region) across ISOs for which
            production averages are available in df_in

        Function Arguments
        ------------------
        - df_in: input data frame
        - global_wb_region: World Bank global region to aggregate df_in to
        - fields_group: fields to group on (excluding region)
        - dict_agg: aggregation dictionary to use 

        Keyword Arguments
        -----------------
        - field_iso: field containing the ISO code. If None, defaults to 
            self.field_iso
        """
        
        field_iso = self.field_iso if (field_iso is None) else field_iso
        if global_wb_region not in self.all_wb_regions:
            return df_in

        regions_wb = [
            self.dict_region_to_iso.get(x) 
            for x in self.dict_wb_region_to_region.get(global_wb_region)
        ]
        df_filt = df_in[df_in[field_iso].isin(regions_wb)]
        
        # get aggregation
        df_filt = sf.simple_df_agg(
            df_filt, 
            fields_group,
            dict_agg
        )
        
        return df_filt
        

    
    def get_closest_region(self,
        region: str,
        missing_flag: float = -999,
        regions_valid: Union[List[str], None] = None,
        type_input: str = "region",
        type_return: str = "region",
    ) -> Union[str, None]:
        """
        Based on latitude/longitude of population centers, find the 
            closest neighboring region.
        

        Function Arguments
        ------------------
        - region: region to search for closest neighbor
        - attr_region: attribute table for regions
        
        Keyword Arguments
        -----------------
        - field_iso: iso field in attr_regin
        - field_lat: field storing latitude
        - field_lon: field storing longitude
        - missing_flag: flag indicating a missing value
        - regions_valid: optional list of regions to restrict search to. If None,
            searches through all regions specified in attr_region
        - type_input: input region type. Either "region" or "iso"
        - type_return: return type. Either "region" or "iso"
        """
        
        ##  INITIALIZATION
        attr_region = self.attributes
        type_return = "region" if (type_return not in ["region", "iso"]) else type_return
        type_input = "region" if (type_input not in ["region", "iso"]) else type_input
        
        # check region/lat/lon
        region = self.dict_iso_to_region.get(region) if (type_input == "iso") else region
        region = region if (region in attr_region.key_values) else None
        coords = self.get_coordinates(region)
        
        # return None if one of the dimensions is missing
        if (coords is None) or (region is None):
            return None

        lat, lon = coords
        
        
        ##  FILTER TABLE AND APPLY DISTANCES
        
        if (regions_valid is None):
            regions_valid = attr_region.key_values 
        else:
            regions_valid = (
                [x for x in attr_region.key_values if x in (regions_valid)]
                if type_input == "region"
                else [x for x in attr_region.key_values if self.dict_region_to_iso.get(x) in (regions_valid)]
            )
            
        df_regions = (
            attr_region.table[
                attr_region.table[attr_region.key].isin(regions_valid)
            ]
            .copy()
            .reset_index(drop = True)
        )
        
        # function to apply
        def f(
            tup: Tuple[float, float]
        ) -> float:
            y, x = tuple(tup)
            
            out = (
                -1.0
                if (min(y, lat) < -90) or (max(y, lat) > 90) or (min(x, lon) < -180) or (max(x, lon) > 180)
                else geopy.distance.geodesic((lat, lon), (y, x)).km
            )
            
            return out
        

        vec_dists = np.array(
            df_regions[[self.field_lat, self.field_lon]]
            .apply(f, raw = True, axis = 1)
        )
        valid_dists = vec_dists[vec_dists > 0.0]
        out = None
        
        if len(valid_dists) > 0:

            m = min(vec_dists)
            w = np.where(vec_dists == m)[0]

            out = (
                list(df_regions[attr_region.key])[w[0]]
                if len(w) > 0
                else None
            )
            out = self.dict_region_to_iso.get(out) if (type_return == "iso") else out


        return out



    def get_coordinates(self,
        region: Union[str, None],
    ) -> Union[Tuple[float, float], None]:
        """
        Return the latitude, longitude coordinates of the population centroid of
            region `region`. `region` can be entered as a region (one of the 
            self.attributes.key_values) or the ISO3 code. If neither is found, 
            returns None

        Function Arguments
        ------------------
        - region_str: region string; either region or ISO can be entered
        """
        
        dict_region_to_lat = self.attributes.field_maps.get(f"{self.attributes.key}_to_{self.field_lat}")
        dict_region_to_lon = self.attributes.field_maps.get(f"{self.attributes.key}_to_{self.field_lon}")

        # check region
        region = (
            self.dict_iso_to_region.get(region)
            if region not in self.all_regions
            else region
        )

        if region is None:
            return None

        # if valid, get coordinates
        tuple_out = (dict_region_to_lat.get(region), dict_region_to_lon.get(region))

        return tuple_out



    def get_world_bank_region(self,
        region: str
    ) -> Union[str, None]:
        """
        Retrieve the World Bank global region associated with region. Often used 
            for assigning regional averages.
        """
        region = self.return_region_or_iso(region, return_type = "region")
        out = self.dict_region_to_wb_region.get(region)

        return out



    def return_region_or_iso(self,
        region: str,
        return_type: str = "region",
    ) -> Union[str, None]:
        """
        Return region for region entered as region or ISO.

        Function Arguments
        ------------------
        - region: region or iso code

        Keyword Arguments
        -----------------
        return_type: "region" or "iso". Will return a region if set to "region" 
            or ISO if set to "iso"
        """
        return_type = "region" if (return_type not in ["region", "iso"]) else return_type
        dict_retrieve = self.dict_iso_to_region if (return_type == "region") else self.dict_region_to_iso
        all_vals = self.all_regions if (return_type == "region") else self.all_isos

        # check region
        region = (
            dict_retrieve.get(region)
            if region not in all_vals
            else region
        )

        return region




    ##  DATA SOURCE-SPECIFIC MODIFICATIONS

    def data_func_iea_get_isos_from_countries(self,
        df_in: Union[pd.DataFrame, List, np.ndarray, str],
        field_country: Union[str, None] = None,
        return_modified_df: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Map IEA countries in field_country to ISO codes contained in 
            df_in[field_country]. If field_country is None, defaults to 
            self.field_iea_country.

        Function Arguments
        ------------------
        - df_in: input data frame containing field country (if None, uses 
            self.field_iea_country) OR list/np.ndarray or input country strings
            OR string

        Keyword Arguments
        -----------------
        - field_country: field in df_in used to identify IEA countries if df_in
            is a DataFrame
        - return_modified_df: if True and df_in is a DataFrame, will return a 
            DataFrame modified to include the iso field
        """
       
        field_country = self.field_iea_country if (field_country is None) else field_country
        vec_iso = (
            list(df_in[field_country]) 
            if isinstance(df_in, pd.DataFrame) 
            else (
                [df_in] if isinstance(df_in, str) else df_in
            )
        )

        vec_iso = [x.lower().replace(" ", "_") for x in vec_iso]
        vec_iso = [self.dict_iea_countries_lc_to_regions.get(x, x) for x in vec_iso]
        vec_iso = [self.dict_region_to_iso.get(x, x) for x in vec_iso]
        
        out = np.array(vec_iso).astype(str)
        if isinstance(df_in, pd.DataFrame) & return_modified_df:
            df_in[self.field_iso] = vec_iso
            out = df_in

        return out







class TimePeriods:
    """
    Leverage some simple time period actions based on a model attributes. The 
        TimePeriods class provides a seamless method for converting years to 
        time periods in SISEPUEDE and can be expanded to integrate months (if
        modeling at that scale).
    """
    def __init__(self,
        model_attributes: ModelAttributes
    ):

        self._initialize_time_properties(model_attributes)

        return None
    


    def _initialize_time_properties(self,
        model_attributes: ModelAttributes,
        field_year: str = "year",
    ) -> None:
        """
        Set the following properties:

            * self.all_time_periods
            * self.all_years
            * self.attributes
            * self.dict_time_period_to_year
            * self.dict_year_to_time_period
            * self.field_time_period
            * self.field_year
            * self.min_year
        """

        attributes = model_attributes.dict_attributes.get(f"dim_{model_attributes.dim_time_period}")
        dict_year_to_time_period = attributes.field_maps.get(f"{field_year}_to_{attributes.key}")
        dict_time_period_to_year = attributes.field_maps.get(f"{attributes.key}_to_{field_year}")
        
        all_time_periods = attributes.key_values
        all_years = sorted(list(set([dict_time_period_to_year.get(x) for x in all_time_periods])))
        year_min, year_max = min(all_years), max(all_years)


        self.all_time_periods = all_time_periods
        self.all_years = all_years
        self.attributes = attributes
        self.dict_time_period_to_year = dict_time_period_to_year
        self.dict_year_to_time_period = dict_year_to_time_period
        self.field_time_period = attributes.key
        self.field_year = field_year
        self.year_max = year_max
        self.year_min = year_min

        return None



    def tp_to_year(self,
        time_period: int,
    ) -> int:
        """
        Convert time period to a year. If time_period is numeric, uses closest
            integer; otherwise, returns None
        """
        time_period = (
            time_period if isinstance(time_period, int) else (
                int(np.round(time_period))
                if isinstance(time_period, float)
                else None
            )     
        )

        if time_period is None:
            return None

        out = self.dict_time_period_to_year.get(
            time_period,
            time_period + self.year_min
        )

        return out


    
    def tps_to_years(self,
        vec_tps: Union[List, np.ndarray, pd.DataFrame, pd.Series],
        field_time_period: Union[str, None] = None,
        field_year: Union[str, None] = None,
    ) -> np.ndarray:
        """
        Convert a vector of years to time periods. 

        Function Arguments
        ------------------
        - vec_tps: List-like input including time periods to convert to years; 
            if DataFrame, will write to field_year (if None, default to
            self.field_year) and look for field_time_period (source time 
            periods, defaults to self.field_time_period)

        Keyword Arguments
        -----------------
        - field_time_period: optional specification of a field to store time 
            period. Only used if vec_years is a DataFrame.
        - field_year: optional specification of a field containing years. Only 
            used if vec_years is a DataFrame.
        """

        df_q = isinstance(vec_tps, pd.DataFrame)
        # check input if data frame
        if df_q:
            
            field_time_period = self.field_time_period if (field_time_period is None) else field_time_period
            field_year = self.field_year if (field_year is None) else field_year
            if field_time_period not in vec_tps.columns:
                return None

            vec = list(vec_tps[field_time_period])

        else:
            vec = list(vec_tps)

        out = np.array([self.tp_to_year(x) for x in vec])

        if df_q:
            df_out = vec_tps.copy()
            df_out[field_year] = out
            out = df_out

        return out
    


    def year_to_tp(self,
        year: int,
    ) -> Union[int, None]:
        """
        Convert a year to a time period. If year is numeric, uses closest
            integer; otherwise, returns None
        """
        year = (
            year if sf.isnumber(year, integer = True) else (
                int(np.round(year))
                if sf.isnumber(year)
                else None
            )     
        )

        if year is None:
            return None

        out = self.dict_year_to_time_period.get(
            year,
            year - self.year_min
        )

        return out



    def years_to_tps(self,
        vec_years: Union[List, np.ndarray, pd.DataFrame, pd.Series],
        field_time_period: Union[str, None] = None,
        field_year: Union[str, None] = None,
    ) -> np.ndarray:
        """
        Convert a vector of years to time periods. 

        Function Arguments
        ------------------
        - vec_years: List-like input including years to convert to time period;
            if DataFrame, will write to field_time_period (if None, default to
            self.field_time_period) and look for field_year (source years,
            defaults to self.field_year)

        Keyword Arguments
        -----------------
        - field_time_period: optional specification of a field to store time 
            period. Only used if vec_years is a DataFrame.
        - field_year: optional specification of a field containing years. Only 
            used if vec_years is a DataFrame.
        """

        df_q = isinstance(vec_years, pd.DataFrame)
        # check input if data frame
        if df_q:

            field_time_period = self.field_time_period if (field_time_period is None) else field_time_period
            field_year = self.field_year if (field_year is None) else field_year
            if field_year not in vec_years.columns:
                return None

            vec = list(vec_years[field_year])
        else:
            vec = list(vec_years)

        out = np.array([self.year_to_tp(x) for x in vec])

        if df_q:
            df_out = vec_years.copy()
            df_out[field_time_period] = out
            out = df_out

        return out






class Transformation:
    """
    Create a Transformation class to support construction in sectoral 
        transformations. 

    Initialization Arguments
    ------------------------
    - code: strategy code associated with the transformation. Must be defined in 
        attr_strategy.table[field_strategy_code]
    - func: the function associated with the transformation OR an ordered list 
        of functions representing compositional order, e.g., 

        [f1, f2, f3, ... , fn] -> fn(f{n-1}(...(f2(f1(x))))))

    - attr_strategy: AttributeTable usd to define strategies from 
        ModelAttributes

    Keyword Arguments
    -----------------
    - field_strategy_code: field in attr_strategy.table containing the strategy
        codes
    - field_strategy_name: field in attr_strategy.table containing the strategy
        name
    """
    
    def __init__(self,
        code: str,
        func: Union[Callable, List[Callable]],
        attr_strategy: Union[AttributeTable, None],
        field_strategy_code: str = "strategy_code",
        field_strategy_name: str = "strategy",
    ):
        
        self._initialize_function(func)
        self._initialize_code(
            code, 
            attr_strategy, 
            field_strategy_code,
            field_strategy_name
        )
        
    
    
    def __call__(self,
        *args,
        **kwargs
    ) -> Any:
        
        val = self.function(
            *args,
            strat = self.id,
            **kwargs
        )

        return val
    




    def _initialize_code(self,
        code: str,
        attr_strategy: Union[AttributeTable, None],
        field_strategy_code: str,
        field_strategy_name: str,
    ) -> None:
        """
        Initialize the transformation name. Sets the following
            properties:

            * self.baseline 
                - bool indicating whether or not it represents the baseline 
                    strategy
            * self.code
            * self.id
            * self.name
        """
        
        # initialize and check code/id num
        id_num = (
            attr_strategy.field_maps.get(f"{field_strategy_code}_to_{attr_strategy.key}")
            if attr_strategy is not None
            else None
        )
        id_num = id_num.get(code) if (id_num is not None) else -1

        if id_num is None:
            raise ValueError(f"Invalid strategy code '{code}' specified in support_classes.Transformation: strategy not found.")

        id_num = id_num if (id_num is not None) else -1

        # initialize and check name/id num
        name = (
            attr_strategy.field_maps.get(f"{attr_strategy.key}_to_{field_strategy_name}")
            if attr_strategy is not None
            else None
        )
        name = name.get(id_num) if (name is not None) else ""

        # check baseline
        baseline = (
            attr_strategy.field_maps.get(f"{attr_strategy.key}_to_baseline_{attr_strategy.key}")
            if attr_strategy is not None
            else None
        )
        baseline = (baseline.get(id_num, 0) == 1)


        ##  set properties

        self.baseline = bool(baseline)
        self.code = str(code)
        self.id = int(id_num)
        self.name = str(name)
        
        return None

    
    
    def _initialize_function(self,
        func: Union[Callable, List[Callable]],
    ) -> None:
        """
        Initialize the transformation function. Sets the following
            properties:

            * self.function
        """
        
        function = None

        if isinstance(func, list):

            func = [x for x in func if callable(x)]

            if len(func) > 0:  
                
                # define a dummy function and assign
                def function_out(
                    *args, 
                    **kwargs
                ) -> Any:
                    f"""
                    Composite Transformation function for {self.name}
                    """
                    out = None
                    if len(args) > 0:
                        out = (
                            args[0].copy() 
                            if isinstance(args[0], pd.DataFrame) | isinstance(args[0], np.ndarray)
                            else args[0]
                        )

                    for f in func:
                        out = f(out, **kwargs)

                    return out

                function = function_out

        elif callable(func):
            function = func

        # check if function assignment failed; if not, assign
        if function is None:
            raise ValueError(f"Invalid type {type(func)}: the object 'func' is not callable.")
        
        self.function = function

        return None
        
        



    
    
    