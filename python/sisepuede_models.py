import logging
from model_attributes import ModelAttributes
from model_afolu import AFOLU
from model_circular_economy import CircularEconomy
from model_electricity import ElectricEnergy
from model_energy import NonElectricEnergy
from model_ippu import IPPU
from model_socioeconomic import Socioeconomic
import numpy as np
import os, os.path
import pandas as pd
import support_functions as sf
import sqlalchemy
import tempfile
from typing import *






class SISEPUEDEModels:
	"""
	Instantiate models for SISEPUEDE.

	Initialization Arguments
	------------------------
	- model_attributes: ModelAttributes object used to manage variables and
		coordination

	Optional Arguments
	------------------
	- allow_electricity_run: allow the electricity model to run (high-runtime
		model)
		* Generally should be left to True
	- fp_nemomod_reference_files: directory housing reference files called by
		NemoMod when running electricity model
		* REQUIRED TO RUN ELECTRICITY MODEL
	- fp_nemomod_temp_sqlite_db: optional file path to use for SQLite database
		used in Julia NemoMod Electricity model
		* If None, defaults to a temporary path sql database
	- logger: optional logging.Logger object used to log model events
	"""
	def __init__(self,
		model_attributes: ModelAttributes,
		allow_electricity_run: bool = True,
		fp_julia: Union[str, None] = None,
		fp_nemomod_reference_files: Union[str, None] = None,
		fp_nemomod_temp_sqlite_db: Union[str, None] = None,
		logger: Union[logging.Logger, None] = None
	):
		# initialize input objects
		self.logger = logger
		self.model_attributes = model_attributes

		# initialize sql path for electricity projection and path to electricity models
		self._initialize_path_nemomod_reference(allow_electricity_run, fp_nemomod_reference_files)
		self._initialize_path_nemomod_sql(fp_nemomod_temp_sqlite_db)
		# initialize last--depends on self.allow_electricity_run
		self._initialize_path_julia(fp_julia)

		# initialize models
		self._initialize_models()




	##############################################
	#	SUPPORT AND INITIALIZATION FUNCTIONS	#
	##############################################

	def _initialize_models(self,
	) -> None:
		"""
		Initialize the path to NemoMod reference files required for ingestion. Initializes
			the following properties:

			* self.allow_electricity_run
			* self.fp_nemomod_reference_files
		"""

		self.model_afolu = AFOLU(self.model_attributes)
		self.model_circecon = CircularEconomy(self.model_attributes)
		self.model_electricity = ElectricEnergy(
			self.model_attributes,
			self.fp_julia,
			self.fp_nemomod_reference_files,
			logger = self.logger
		) if self.allow_electricity_run else None
		self.model_energy = NonElectricEnergy(
			self.model_attributes,
			logger = self.logger
		)
		self.model_ippu = IPPU(self.model_attributes)
		self.model_socioeconomic = Socioeconomic(self.model_attributes)



	def _initialize_path_julia(self,
		fp_julia: Union[str, None]
	) -> None:
		"""
		Initialize the path to the NemoMod SQL database used to execute runs. Initializes
			the following properties:

			* self.fp_julia

			NOTE: Will set `self.allow_electricity_run = False` if the path is not found.
		"""

		self.fp_julia = None
		if isinstance(fp_julia, str):
			if os.path.exists(fp_julia):
				self.fp_julia = fp_julia
				self._log(f"Set Julia directory for modules and environment to '{self.fp_julia}'.", type_log = "info")
			else:
				self.allow_electricity_run = False
				self._log(f"Invalid path '{fp_julia}' specified for Julia reference modules and environment: the path does not exist. Setting self.allow_electricity_run = False.", type_log = "error")



	def _initialize_path_nemomod_reference(self,
		allow_electricity_run: bool,
		fp_nemomod_reference_files: Union[str, None]
	) -> None:
		"""
		Initialize the path to NemoMod reference files required for ingestion. Initializes
			the following properties:

			* self.allow_electricity_run
			* self.fp_nemomod_reference_files

		Function Arguments
		------------------
		- allow_electricity_run: exogenous specification of whether or not to allow the
			electricity model to run
		- fp_nemomod_reference_files: path to NemoMod reference files
		"""

		self.allow_electricity_run = False
		self.fp_nemomod_reference_files = None

		try:
			self.fp_nemomod_reference_files = sf.check_path(fp_nemomod_reference_files, False)
			self.allow_electricity_run = allow_electricity_run
		except Exception as e:
			self._log(f"Path to NemoMod reference files '{fp_nemomod_reference_files}' not found. The Electricity model will be disallowed from running.", type_log = "warning")



	def _initialize_path_nemomod_sql(self,
		fp_nemomod_temp_sqlite_db: Union[str, None]
	) -> None:
		"""
		Initialize the path to the NemoMod SQL database used to execute runs. 
			Initializes the following properties:

			* self.fp_nemomod_temp_sqlite_db
		"""

		valid_extensions = ["sqlite", "db"]

		# initialize as temporary
		fn_tmp = os.path.basename(tempfile.NamedTemporaryFile().name)
		fn_tmp = f"{fn_tmp}.sqlite"
		self.fp_nemomod_temp_sqlite_db = os.path.join(
			os.getcwd(),
			fn_tmp
		)

		if isinstance(fp_nemomod_temp_sqlite_db, str):
			try_endings = [fp_nemomod_temp_sqlite_db.endswith(x) for x in valid_extensions]

			if any(try_endings):
				self.fp_nemomod_temp_sqlite_db = fp_nemomod_temp_sqlite_db
				self._log(f"Successfully initialized NemoMod temporary database path as {self.fp_nemomod_temp_sqlite_db}.", type_log = "info")

			else:
				self._log(f"Invalid path '{fp_nemomod_temp_sqlite_db}' specified as fp_nemomod_temp_sqlite_db. Using temporary path {self.fp_nemomod_temp_sqlite_db}.", type_log = "info")


		# clear old temp database to prevent competing key information in sql schema
		os.remove(self.fp_nemomod_temp_sqlite_db) if os.path.exists(self.fp_nemomod_temp_sqlite_db) else None

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



	############################
	#    CORE FUNCTIONALITY    #
	############################

	def project(self,
		df_input_data: pd.DataFrame,
		include_electricity_in_energy: bool = True,
		models_run: Union[List[str], None] = None,
		regions: Union[List[str], str, None] = None,
		run_integrated: bool = True
	) -> pd.DataFrame:
		"""
		Run SISEPUEDE models in appropriate order.

		**WARNING** ONLY DESIGNED TO RUN ONE REGION AT A TIME RIGHT NOW. Will 
			design multi-region specification soon.


		Function Arguments
		------------------
		df_input_data: DataFrame containing SISEPUEDE inputs

		Optional Arguments
		------------------
		- models_run: list of sector models to run as defined in
			SISEPUEDEModels.model_attributes. Can include the following values:

			* AFOLU (or af)
			* Circular Economy (or ce)
			* IPPU (or ip)
			* Energy (or en)
				* Note: set include_electricity_in_energy = False to avoid
					running the electricity model with energy
			* Socioeconomic (or se)

		Keyword Arguments
		-----------------
		- include_electricity_in_energy: include the electricity model in runs
			of the energy model?
			* If False, runs without electricity (time intensive model)
		- regions: regions to run the model for (NEEDS ADDITIONAL WORK IN 
			NON-ELECTRICITY SECTORS)
		- run_integrated: run models as integrated collection?
			* If False, will run each model individually, without interactions
				(not recommended)
		"""

		df_return = []
		models_run = self.model_attributes.get_sector_list_from_projection_input(models_run)
		regions = self.model_attributes.get_region_list_filtered(regions)
		
		
		##  1. Run AFOLU and collect output

		if "AFOLU" in models_run:
			self._log("Running AFOLU model", type_log = "info")
			try:
				df_return.append(self.model_afolu.project(df_input_data))
				self._log(f"AFOLU model run successfully completed", type_log = "info")

			except Exception as e:
				self._log(f"Error running AFOLU model: {e}", type_log = "error")


		##  2. Run CircularEconomy and collect output - requires AFOLU to run integrated

		if "Circular Economy" in models_run:
			self._log("Running CircularEconomy model", type_log = "info")
			if run_integrated and set(["AFOLU"]).issubset(set(models_run)):
				df_input_data = self.model_attributes.transfer_df_variables(
					df_input_data,
					df_return[0],
					self.model_circecon.integration_variables
				)

			try:
				df_return.append(self.model_circecon.project(df_input_data))
				df_return = (
					[sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
					if run_integrated 
					else df_return
				)
				self._log(f"CircularEconomy model run successfully completed", type_log = "info")

			except Exception as e:
				self._log(f"Error running CircularEconomy model: {e}", type_log = "error")


		##  3. Run IPPU and collect output

		if "IPPU" in models_run:
			self._log("Running IPPU model", type_log = "info")
			if run_integrated and set(["Circular Economy"]).issubset(set(models_run)):
				df_input_data = self.model_attributes.transfer_df_variables(
					df_input_data,
					df_return[0],
					self.model_ippu.integration_variables
				)

			try:
				df_return.append(self.model_ippu.project(df_input_data))
				df_return = (
					[sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
					if run_integrated 
					else df_return
				)
				self._log(f"IPPU model run successfully completed", type_log = "info")

			except Exception as e:
				self._log(f"Error running IPPU model: {e}", type_log = "error")


		##  4. Run Non-Electric Energy (excluding Fugitive Emissions) and collect output

		if "Energy" in models_run:
			self._log("Running Energy model (NonElectricEnergy without Fugitive Emissions)", type_log = "info")
			if run_integrated and set(["IPPU", "AFOLU"]).issubset(set(models_run)):
				df_input_data = self.model_attributes.transfer_df_variables(
					df_input_data,
					df_return[0],
					self.model_energy.integration_variables_non_fgtv
				)

			try:
				df_return.append(self.model_energy.project(df_input_data))
				df_return = (
					[sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
					if run_integrated 
					else df_return
				)
				self._log(f"NonElectricEnergy without Fugitive Emissions model run successfully completed", type_log = "info")

			except Exception as e:
				self._log(f"Error running NonElectricEnergy without Fugitive Emissions: {e}", type_log = "error")


		##  5. Run Electricity and collect output

		if ("Energy" in models_run) and include_electricity_in_energy and self.allow_electricity_run:
			self._log("Running Energy model (Electricity and Fuel Production: trying to call Julia)", type_log = "info")
			if run_integrated and set(["Circular Economy", "AFOLU"]).issubset(set(models_run)):
				df_input_data = self.model_attributes.transfer_df_variables(
					df_input_data,
					df_return[0],
					self.model_electricity.integration_variables
				)

			# create the engine and try to run Electricity
			engine = sqlalchemy.create_engine(f"sqlite:///{self.fp_nemomod_temp_sqlite_db}")
			try:
				df_elec = self.model_electricity.project(
					df_input_data, 
					engine,
					regions = regions
				)
				df_return.append(df_elec)
				df_return = (
					[sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
					if run_integrated 
					else df_return
				)
				self._log(f"ElectricEnergy model run successfully completed", type_log = "info")

			except Exception as e:
				self._log(f"Error running ElectricEnergy model: {e}", type_log = "error")


		##  6. Add fugitive emissions from Non-Electric Energy and collect output

		if "Energy" in models_run:
			self._log("Running Energy (Fugitive Emissions)", type_log = "info")
			if run_integrated and set(["IPPU", "AFOLU"]).issubset(set(models_run)):
				df_input_data = self.model_attributes.transfer_df_variables(
					df_input_data,
					df_return[0],
					self.model_energy.integration_variables_fgtv
				)

			try:
				df_return.append(
					self.model_energy.project(
						df_input_data, 
						subsectors_project = self.model_attributes.subsec_name_fgtv
					)
				)
				df_return = (
					[sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
					if run_integrated 
					else df_return
				)
				self._log(f"Fugitive Emissions from Energy model run successfully completed", type_log = "info")

			except Exception as e:
				self._log(f"Error running Fugitive Emissions from Energy model: {e}", type_log = "error")


		##  7. Add Socioeconomic output at the end to avoid double-initiation throughout models

		if len(df_return) > 0:
			self._log("Appending Socioeconomic outputs", type_log = "info")

			try:
				df_return.append(
					self.model_socioeconomic.project(
						df_input_data, 
						project_for_internal = False
					)
				)
				df_return = (
					[sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate")] 
					if run_integrated 
					else df_return
				)
				self._log(f"Socioeconomic outputs successfully appended.", type_log = "info")

			except Exception as e:
				self._log(f"Error appending Socioeconomic outputs: {e}", type_log = "error")


		# build output data frame
		df_return = (
			sf.merge_output_df_list(df_return, self.model_attributes, merge_type = "concatenate") 
			if (len(df_return) > 0) 
			else pd.DataFrame()
		)

		return df_return
