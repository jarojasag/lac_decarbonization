###
###    SET OF TOOLS FOR READING/WRITING FROM/TO SISEPUEDE DATA REPOSITORY
###

import itertools
import logging
import model_attributes as ma
import model_afolu as mafl
import model_ippu as mi
import model_circular_economy as mc
import model_electricity as ml
import model_energy as me
import model_socioeconomic as se
import numpy as np
import os, os.path
import pandas as pd
import re
import support_classes as sc
import support_functions as sf
import time
from typing import *
import warnings




class SISEPUEDEBatchDataRepository:
    """
    Interact with the sisepuede_data git hub repository (read and write) using
        SISEPUEDE model variables.

    Initialization Arguments
    ------------------------
    - dir_repository: path to repository containing all data
    - model_attributes: model_attributes.ModelAttributes object used to 
        coordinate and access variables
    """

    def __init__(self,
        dir_repository: str,
        model_attributes: ma.ModelAttributes,
    ): 

        self._initialize_attributes(model_attributes)
        self._initialize_fields()
        self._initialize_repository(dir_repository)




    #################################
    #   INITIALIZATION FUNCTIONS    #
    #################################

    def _initialize_attributes(self,
        model_attributes: ma.ModelAttributes,
    ) -> None:
        """
        Initialize model attributes and associated support classes. Initializes
            the following properties:

            * self.model_attributes
            * self.regions (sc.Regions)
            * self.time_periods (sc.TimePeriods)
        """

        self.model_attributes = model_attributes
        self.regions = sc.Regions(model_attributes)
        self.time_periods = sc.TimePeriods(model_attributes)

        return None



    def _initialize_fields(self,
    ) -> None:
        """
        Initialize fields that are used in repository as well as groups of 
            fields that can be ignored etc. Sets the following properties:

            * self.field_repo_iso
            * self.field_repo_location_code
            * self.field_repo_nation
            * self.field_repo_year

        """

        self.field_repo_country = "country"
        self.field_repo_iso = "iso_code3"
        self.field_repo_location_code = "location_code"
        self.field_repo_nation = "Nation"
        self.field_repo_year = "Year"

        return None



    def _initialize_repository(self,
        dir_repository: str,
    ) -> None:
        """
        Initialize the data repository; check structure, etc. Sets the following
            properties:

            * self.dict_sector_to_subdir
            * self.dir_repository
            * self.key_historical
            * self.key_projected
            * self.subdir_input_to_sisepuede
        """

        # dictionary mapping SISEPUEDE sector name to sisepuede_data repository directory
        dict_sector_to_subdir = {
            "AFOLU": "AFOLU",
            "Energy": "Energy",
            "IPPU": "IPPU", 
            "Socioeconomic": "SocioEconomic"
        }

        self.dict_sector_to_subdir = dict_sector_to_subdir
        self.dir_repository = sf.check_path(dir_repository, create_q = False)
        self.key_historical = "historical"
        self.key_projected = "projected"
        self.subdir_input_to_sisepuede = "input_to_sisepuede"

        return None



    

    ########################
    #    CORE FUNCTIONS    #
    ########################
    
    def file_to_sd_dirs(self,
        fp_csv: str,
        fields_ind: List[str],
        years_historical: List[int],
        dict_rename: Union[Dict[str, str], None] = None,
    ) -> Union[Dict[str, pd.DataFrame], None]:
        """
        Read a CSV file and return a dictionary of files to write
        
        Function Arguments
        ------------------
        - fp_csv: file path of CSV to read
        - fields_ind: fields to use as index (present in all output CSVs). 
            NOTE: if dict_rename is not None and any index fielids are 
            renamed by dict_rename, then fields_ind should specify renaming
            targets.
        - years_historical: years to consider as historical
        
        Keyword Arguments
        -----------------
        - dict_rename: optional dictionary used to rename fields that are read
            in.
            NOTE: Renaming occurs *before* extracting fields_ind, so fields_ind
                should reference target renamed fields if applicable
        """
        
        # some quick checks
        if fp_csv is None:
            return None
        
        if (not os.path.exists(fp_csv)) | (not sf.islistlike(years_historical)):
            return None
        
        # check time periods/regions
        regions = self.regions
        time_periods = self.time_periods


        if len(years_historical) == 0:
            return None
        
        # read and check time indexing--add years if only time period is included
        df_csv = pd.read_csv(fp_csv)
        if (time_periods.field_year not in df_csv.columns) and (time_periods.field_time_period not in df_csv.columns):
            return None
        df_csv = (
            time_periods.tps_to_years(df_csv)
            if time_periods.field_year not in df_csv.columns
            else df_csv
        )
        
        # rename csv 
        dict_rnm = {}
        if isinstance(dict_rename, dict):
            for k, v in dict_rename.items():
                dict_rnm.update({k: v}) if (k in df_csv.columns) else None
        df_csv.rename(columns = dict_rnm, inplace = True)
        field_year = dict_rnm.get(time_periods.field_year, time_periods.field_year)
        
        # get fields and return None if invalid
        fields_ind = [x for x in fields_ind if x in df_csv.columns]
        fields_dat = [x for x in df_csv.columns if x in self.model_attributes.all_variables]
        if min(len(fields_dat), len(fields_ind)) == 0:
            return None
        
        
        # initialize and write output
        dict_out = {}
        for fld in fields_dat:
            
            df_ext = df_csv[fields_ind + [fld]]
            
            df_ext_hist = df_ext[
                df_ext[field_year].isin(years_historical)
            ].reset_index(drop = True)
            
            df_ext_proj = df_ext[
                ~df_ext[field_year].isin(years_historical)
            ].reset_index(drop = True)
            
            dict_out.update(
                {
                    fld: {
                        self.key_historical: df_ext_hist,
                        self.key_projected: df_ext_proj
                    }
                }
            )
            
        return dict_out



    def field_to_path(self,
        fld: str,
        key_type: str,
    ) -> Union[str, None]:
        """
        Convert SISEPUEDE field `fld` to output path in sisepuede_data 
            repository

        Function Arguments
        ------------------
        - fld: valid SISEPUEDE field. If invalid, returns None
        - key_type: "historical" or "projected". If invalid, returns None
        """
        modvar = self.model_attributes.dict_variables_to_model_variables.get(fld)
        key = (
            self.key_historical 
            if (key_type in ["historical", self.key_historical])
            else (
                self.key_projected
                if (key_type in ["projected", self.key_projected])
                else None
            )
        )

        if (modvar is None) | (key is None):
            return None

        # otherwise, get sector info and outputs
        sector = self.model_attributes.get_variable_subsector(modvar)
        sector = self.model_attributes.get_subsector_attribute(sector, "sector")
        subdir_sector = self.dict_sector_to_subdir.get(sector, sector)

        # create outputs
        fp_out_base = os.path.join(self.dir_repository, subdir_sector, fld, self.subdir_input_to_sisepuede)
        fp_out = os.path.join(fp_out_base, key, f"{fld}.csv")

        return fp_out



    def write_from_rbd(self,
        dir_batch: str,
        dict_years_historical: Union[Dict[str, List[int]], List[int]],
        dirs_ignore: Union[list[str], None] = None,
        ext_read: str = "csv",
        field_iso_out: Union[str, None] = None,
        field_region_out: Union[str, None] = None,
        field_year_out: Union[str, None] = None,
        fps_ignore: Union[List[str], None] = None,
        key_historical: Union[str, None] = None,
        key_projected: Union[str, None] = None,
        write_q: bool = True,
    ) -> Tuple[Dict, Dict]:
        """
        Using directory dir_batch (in SISEPUEDE repository), generate inputs
            for sisepuede_data repo
            
        Function Arguments
        ------------------
        - dir_batch: directory storing batch data using lac_decarbonization 
            structure
        - dict_years_historical: dictionary mapping a file to years historical 
            OR a list of integer years to consider histroical
        
        Keyword Arguments
        -----------------
        - dirs_ignore: list of subdirectories to ignore
        - ext_read: extension of input files to read
        - fields_ignore: list of fields to ignore in each input file when 
            checking for fields that will be written to self.dir_repository
        - fps_ignore: optional file paths to ignore
        - key_historical: optional key to use for historical subdirectories. If
            None, defaults to SISEPUEDEBatchDataRepository.key_historical
        - key_projected: optional key to use for historical subdirectories. If
            None, defaults to SISEPUEDEBatchDataRepository.key_projected
        - write_q: write output data to files
        """

        # some field initialization
        field_iso_out = (
            self.field_repo_iso
            if field_iso_out is None
            else field_iso_out
        )
        field_region_out = (
            self.field_repo_nation
            if field_region_out is None
            else field_region_out
        )
        field_year_out = (
            self.field_repo_year
            if field_year_out is None
            else field_year_out
        )
        fields_ind = [field_year_out, field_iso_out]
        dict_rename = {
            self.model_attributes.dim_region: self.field_repo_nation,
            self.field_repo_country: self.field_repo_nation,
            self.field_repo_nation.lower(): self.field_repo_nation,
            self.regions.field_iso: self.field_repo_iso,
            self.time_periods.field_year: self.field_repo_year
        }

        # subdirectory keys
        key_historical = (
            self.key_historical
            if not isinstance(key_historical, str)
            else key_historical
        )

        key_projected = (
            self.key_projected
            if not isinstance(key_projected, str)
            else key_projected
        )
        
        # directory checks--make output if not exstis + loop through subdirectories to check for available data
        subdirs = (
            [x for x in os.listdir(dir_batch) if os.path.join(dir_batch, x) not in dirs_ignore]
            if sf.islistlike(dirs_ignore)
            else os.listdir(dir_batch)
        )

        
        dict_out = {}
        dict_paths = {}
        
        for subdir in subdirs:
            fp_subdir = os.path.join(dir_batch, subdir)

            if os.path.isdir(fp_subdir):
                
                fns_read = [x for x in os.listdir(fp_subdir) if x.endswith(f".{ext_read}")]
                
                for fn in fns_read:
                    years_historical = (
                        dict_years_historical.get(fn)
                        if isinstance(dict_years_historical, dict)
                        else dict_years_historical
                    )
                    
                    fp_read = os.path.join(fp_subdir, fn)
                    fp_read = None if (fp_read in fps_ignore) else fp_read
                    
                    dict_read = self.file_to_sd_dirs(
                        fp_read,
                        fields_ind,
                        years_historical,
                        dict_rename = dict_rename,
                    )
                    
                    # get variable information
                    if dict_read is not None:
                        
                        for fld in dict_read.keys():
                            fp_out_hist = self.field_to_path(fld, self.key_historical)
                            fp_out_proj = self.field_to_path(fld, self.key_projected)
                            
                            dict_paths.update(
                                {
                                    fld: {
                                        self.key_historical: fp_out_hist,
                                        self.key_projected: fp_out_proj
                                    }
                                }
                            )
                            
                        dict_out.update(dict_read) 


        # write outputs?
        if write_q:
            
            for fld in dict_out.keys():

                dict_dfs_cur = dict_out.get(fld)
                dict_paths_cur = dict_paths.get(fld)
                

                for key in [key_historical, key_projected]:
                    
                    # get df
                    df_write = dict_dfs_cur.get(key)
                    
                    if df_write is not None:
                        # check directory
                        fp = dict_paths_cur.get(key)
                        dir_base = os.path.dirname(fp)
                        os.makedirs(dir_base, exist_ok = True) if not os.path.exists(dir_base) else None
                        
                        df_write.to_csv(
                            fp, 
                            index = None,
                            encoding = "UTF-8"
                        )
                        
                        print(f"DataFrame successfully written to '{fp}'")
                
        
        return dict_out, dict_paths


        
    def read(self,
        dict_modvars: Dict[str, Union[List[str], None]],
        add_time_periods: bool = False,
    ) -> pd.DataFrame: 
        """
        Read inputs from the repository for use.
        
        Function Arguements
        -------------------
        - dict_modvars: dictionary with model variables as keys and a list of 
            categories to apply to (or None to read all applicable)
            
        Keyword Arguements
        ------------------
        - add_time_periods: add time periods to input?
        """
        
        # SOME INITIALIZATION

        # some needed dictionaries
        dict_sector_to_subdir = self.dict_sector_to_subdir
        dict_subsec_abv_to_sector = self.model_attributes.dict_attributes.get("abbreviation_subsector").field_maps.get("abbreviation_subsector_to_sector")
        dict_subsec_to_subsec_abv = self.model_attributes.dict_attributes.get("abbreviation_subsector").field_maps.get("subsector_to_abbreviation_subsector")
        dict_modvars = (
            dict((x, None) for x in self.model_attributes.all_model_variables)
            if not isinstance(dict_modvars, dict)
            else dict_modvars
        ) 

        # some fields
        field_iso = self.field_repo_iso.lower()
        field_year = self.field_repo_year.lower()
        fields_index = [field_iso, field_year] 
        fields_to_iso = [self.field_repo_location_code]
        
        # initialize output
        df_out = None
        df_index = None # used to govern merges
        dict_modvar_to_fields = {}
        dict_modvar_to_ordered_cats = {}
        
        modvars = list(dict_modvars.keys())
        
        for k, modvar in enumerate(modvars):
            
            cats_defined = self.model_attributes.get_variable_categories(modvar)
            cats = dict_modvars.get(modvar)
            cats = cats_defined if (cats is None) else cats
            cats = (
                [x for x in cats_defined if x in cats]
                if (cats_defined is not None)
                else [None]
            )
            
            subsec = self.model_attributes.get_variable_subsector(modvar)
            sector = dict_subsec_abv_to_sector.get(
                dict_subsec_to_subsec_abv.get(subsec)
            )
            sector_repo = dict_sector_to_subdir.get(sector)
            
            
            if (sector_repo is not None) and (len(cats) > 0):

                for cat in cats:
                    
                    restriction = None if (cat is None) else [cat]
                    var_name = self.model_attributes.build_varlist(
                        subsec,
                        modvar, 
                        restrict_to_category_values = restriction
                    )[0]

                    df_var = []
                    
                    for key in [self.key_historical, self.key_projected]:
                        
                        fp_read = self.field_to_path(var_name, key)

                        if os.path.exists(fp_read):
                            
                            try:
                                # read
                                df_var_cur = pd.read_csv(fp_read)
                                
                                # rename where necessary
                                dict_rnm_to_iso = dict(
                                    (x, field_iso) 
                                    for x in fields_to_iso
                                    if x in df_var_cur.columns
                                )
                                df_var_cur.rename(
                                    columns = dict_rnm_to_iso,
                                    inplace = True
                                )
                                
                                # clean the fields
                                dict_rnm = dict((x, x.lower()) for x in df_var_cur.columns)
                                df_var_cur.rename(
                                    columns = dict_rnm,
                                    inplace = True
                                )

                                # drop any unwanted columns
                                df_var_cur = df_var_cur[fields_index + [var_name]]
                                df_var_cur.set_index(fields_index, inplace = True)

                                if key == self.key_projected:
                                    inds_prev = df_var[0].index
                                    df_var_cur = df_var_cur[
                                        [(x not in inds_prev) for x in df_var_cur.index]
                                    ]
                                    df_var[0].reset_index(inplace = True)
                                    df_var_cur.reset_index(inplace = True)
                                
                                df_var.append(df_var_cur)

                            except Exception as e:
                                warnings.warn(f"Error trying to read {fp_read}: {e}")
                                
                        

                    # concatenate and sort
                    df_var = pd.concat(df_var, axis = 0) if (len(df_var) > 0) else None
                    
                    if ((fields_index is not None) and (df_var is not None)):                    
                        # get dictionaries
                        fields_add = sorted([x for x in df_var.columns if x not in fields_index])
                        fields_exist = dict_modvar_to_fields.get(modvar)
                        
                        (
                            dict_modvar_to_fields.update({modvar: fields_add}) 
                            if fields_exist is None
                            else dict_modvar_to_fields[modvar].extend(fields_add)
                        )
                        
                        (
                            dict_modvar_to_ordered_cats.update({modvar: [cat]}) 
                            if fields_exist is None
                            else dict_modvar_to_ordered_cats[modvar].append(cat)
                        )
                        
                        
                    if df_var is not None:

                        df_var.sort_values(by = fields_index, inplace = True)
                        #df_var.set_index(fields_index, inplace = True)
                        
                        df_var.reset_index(drop = True, inplace = True)

                        if (df_out is None):

                            df_out = [df_var]
                            df_index = df_var[fields_index].copy()

                        else:
                            #df_out.append(df_var)
                            
                            fold_q = (
                                True
                                if df_var[fields_index].shape != df_index.shape
                                else not all(df_var[fields_index] == df_index)
                            )

                            # setup indexing data frame
                            if fold_q:
                                df_out = pd.concat(df_out, axis = 1)

                                df_index = (
                                    pd.merge(
                                        df_index, 
                                        df_var[fields_index],
                                        how = "outer"
                                    )
                                    .sort_values(by = fields_index)
                                    .reset_index(drop = True)
                                )


                                df_out = (
                                    pd.merge(
                                        df_index, 
                                        df_out, 
                                        how = "left", 
                                        on = fields_index
                                    )
                                    .sort_index()
                                    .reset_index(drop = True)
                                )
                                df_out = [df_out]
                                df_var = (
                                    pd.merge(
                                        df_index, 
                                        df_var, 
                                        how = "left", 
                                        on = fields_index
                                    )
                                    .sort_index()
                                    .reset_index(drop = True)
                                )

                            #""";
                            # print(f"appended {var_name}")
                            # print(df_index.shape)
                            # print(df_out[0].shape)
                            # print(df_var.shape)
                            # print("\n")
                            df_out.append(df_var[[var_name]])


        if (df_out is not None) and (fields_index is not None):

            """
            df_out = (
                df_out[0].join(
                    df_out[1:],
                    how = "outer"
                )
                if len(df_out) > 1
                else df_out[0]
            )
            """;
            df_out = pd.concat(df_out, axis = 1)
            df_out.sort_values(by = fields_index, inplace = True) 
            df_out.reset_index(drop = True, inplace = True) 

        df_out = (
            self.time_periods.years_to_tps(df_out)
            if add_time_periods
            else df_out
        )
        # , dict_modvar_to_fields, dict_modvar_to_ordered_cats

        return df_out