from attribute_table import AttributeTable
from ingestion import *
import itertools
from lhs_design import LHSDesign
import logging
from model_attributes import ModelAttributes
from model_afolu import AFOLU
from model_circular_economy import CircularEconomy
from model_electricity import ElectricEnergy
from model_energy import NonElectricEnergy
from model_ippu import IPPU
from model_socioeconomic import Socioeconomic
import logging
import numpy as np
import os, os.path
import pandas as pd
from sampling_unit import FutureTrajectories
import support_functions as sf
import sqlalchemy
import tempfile
from typing import *
import warnings



class SISEPUEDEFileStructure:
	"""
	Create and verify the directory structure for SISEPUEDE.

	Optional Arguments
	------------------
	- dir_ingestion: directory containing templates for ingestion. The ingestion directory should include
		subdirectories for each template class that may be run, including:
			* calibrated: input variables that are calibrated for each region and sector
			* demo: demo parameters that are independent of region (default in quick start)
			* uncalibrated: preliminary input variables defined for each region that have not yet been
				calibrated

		The calibrated and uncalibrated subdirectories require separate subdrectories for each region, each
		of which contains an input template for each
	- fn_config: name of configuration file in SISEPUEDE directory
	- logger: optional logging.Logger object used for logging

	"""
	def __init__(self,
		dir_ingestion: Union[str, None] = None,
		fn_config: str = "sispuede.config",
		logger: Union[logging.Logger, None] = None
	):

		self.logger = logger

		# run checks of directories
		self._check_config(fn_config)
		self._check_critical_directories()
		self._check_ingestion(dir_ingestion)
		self._check_optional_directories()

		# initialize model attributes
		self._initialize_model_attributes()



	##############################
	#    SUPPORTING FUNCTIONS    #
	##############################

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



	##########################
	#    DIRECTORY CHECKS    #
	##########################

	def _check_config(self,
		fn_config: str
	) -> None:
		"""
		Check the configuration file name. Sets the following properties:

			* self.fn_config
		"""

		self.fn_config = "sisepuede.config"
		if isinstance(fn_config, str):
			self.fn_config = fn_config if fn_config.endswith(".config") else self.fn_config



	def _check_critical_directories(self,
	) -> None:
		"""
		Check directory structure for SISEPUEDE. Sets the following properties:

			*
		"""

		##  Initialize base paths

		self.dir_py = os.path.dirname(os.path.realpath(__file__))
		self.dir_proj = os.path.dirname(self.dir_py)
		# initialize error message
		count_errors = 0
		msg_error_dirs = ""


		##  Check configuration file

		self.fp_config = os.path.join(self.dir_proj, self.fn_config)
		if not os.path.exists(self.fp_config):
			count_errors += 1
			msg_error_dirs += f"\n\tConfiguration file '{self.fp_config}' not found"
			self.fp_config = None


		##  Check docs path

		self.dir_docs = os.path.join(os.path.dirname(self.dir_py), "docs", "source") if (self.dir_py is not None) else ""
		if not os.path.exists(self.dir_docs):
			count_errors += 1
			msg_error_dirs += f"\n\tDocs subdirectory '{self.dir_docs}' not found"
			self.dir_docs = None


		##  Check attribute tables path (within docs path)

		self.dir_attribute_tables = os.path.join(self.dir_docs, "csvs") if (self.dir_docs is not None) else ""
		if not os.path.exists(self.dir_attribute_tables):
			count_errors += 1
			msg_error_dirs += f"\n\tAttribute tables subdirectory '{self.dir_attribute_tables}' not found"
			self.dir_attribute_tables = None


		##  Check Julia directory

		self.dir_jl = os.path.join(self.dir_proj, "julia")
		if not os.path.exists(self.dir_jl):
			count_errors += 1
			msg_error_dirs += f"\n\tJulia subdirectory '{self.dir_jl}' not found"
			self.dir_jl = None


		##  Check reference directory

		self.dir_ref = os.path.join(self.dir_proj, "ref")
		if not os.path.exists(self.dir_ref):
			count_errors += 1
			msg_error_dirs += f"\n\tReference subdirectory '{self.dir_ref}' not found"
			self.dir_ref = None


		##  Check NemoMod reference directory (within reference directory)

		self.dir_ref_nemo = os.path.join(self.dir_ref, "nemo_mod") if (self.dir_ref is not None) else ""
		if not os.path.exists(self.dir_ref_nemo):
			count_errors += 1
			msg_error_dirs += f"\n\tNemoMod reference subdirectory '{self.dir_ref_nemo}' not found"
			self.dir_ref_nemo = None


		##  error handling
		if count_errors > 0:
			self._log(f"There were {count_errors} errors initializing the SISEPUEDE directory structure:{msg_error_dirs}", type_log = "error")

			raise RuntimeError("SISEPUEDE unable to initialize file directories. Check the log for more information.")



	def _check_ingestion(self,
		dir_ingestion: Union[str, None]
	) -> None:
		"""
		Check path to templates.

		Function Arguments
		------------------
		dir_ingestion: ingestion directory storing input templates for SISEPUEDE
			* If None, defaults to ..PATH_SISEPUEDE/ref/ingestion

		"""

		##  Check template ingestion path (within reference directory)

		# initialize
		self.dir_ingestion = os.path.join(self.dir_ref, "ingestion") if (self.dir_ref is not None) else None
		self.dir_parameters_calibrated = None
		self.dir_parameters_demo = None
		self.dir_parameters_uncalibrated = None

		# override if input path is specified
		if isinstance(dir_ingestion, str):
			if os.path.exists(dir_ingestion):
				self.dir_ingestion = dir_ingestion

		# check existence
		if not os.path.exists(self.dir_ingestion):
			self._log(f"\tIngestion templates subdirectory '{self.dir_ingestion}' not found")
			self.dir_ingestion = None
		else:
			# sheets with complete input variables and calibrated parameters by region
			dir_parameters_calibrated = os.path.join(self.dir_ingestion, "calibrated")
			self.dir_parameters_calibrated = dir_parameters_calibrated if os.path.exists(dir_parameters_calibrated) else self.dir_parameters_calibrated

			# demonstration input variables and parameters to facilitate quick start/demonstration
			dir_parameters_demo = os.path.join(self.dir_ingestion, "demo")
			self.dir_parameters_demo = dir_parameters_demo if os.path.exists(dir_parameters_demo) else self.dir_parameters_demo

			# sheets with complete or incomplete input variables and uncalibrated parameters by region
			dir_parameters_uncalibrated = os.path.join(self.dir_ingestion, "uncalibrated")
			self.dir_parameters_uncalibrated = dir_parameters_uncalibrated if os.path.exists(dir_parameters_uncalibrated) else self.dir_parameters_uncalibrated



	def _check_optional_directories(self,
	) -> None:
		"""
		Check directories that are not critical to SISEPUEDE functioning, including those that
			can be created if not found. Checks the following properties:

			* self.dir_out
			* self.dir_ref_batch_data
			* self.dir_ref_data_crosswalks
		"""

		##  Output and temporary directories (can be created)

		self.dir_out, self.dir_tmp = None, None
		if self.dir_proj is not None:
			self.dir_out = sf.check_path(os.path.join(self.dir_proj, "out"), True)
			self.dir_tmp = sf.check_path(os.path.join(self.dir_proj, "tmp"), True)


		##  Batch data directories (not required to run SISEPUEDE, but required for Data Generation notebooks and routines)

		self.dir_ref_batch_data, self.dir_ref_data_crosswalks = None, None
		if self.dir_ref is not None:
			self.dir_ref_batch_data = sf.check_path(os.path.join(self.dir_ref, "batch_data_generation"), True)
			self.dir_ref_data_crosswalks = sf.check_path(os.path.join(self.dir_ref, "data_crosswalks"), True)



	###############################################
	#    INITIALIZE FILES AND MODEL ATTRIBUTES    #
	###############################################

	def _initialize_model_attributes(self,
	) -> None:
		"""
		Initialize SISEPUEDE model attributes from directory structure. Sets the following
			properties:

			* self.model_attributes
		"""
		self.model_attributes = None
		if (self.dir_attribute_tables is not None) and (self.fp_config is not None):
			self.model_attributes = ma.ModelAttributes(self.dir_attribute_tables, self.fp_config)



	def _initialize_key_file_path_defaults(self,
	) -> None:
		"""
		Initialize key default file paths, including output and temporary files. Sets the
			following properties:

			*
		"""

class SISEPUEDEModels:
	"""
	Instantiate models based on

	Initialization Arguments
	------------------------
	- model_attributes: ModelAttributes object used to manage variables and coordination

	Optional Arguments
	------------------
	- fp_nemomod_reference_files: directory housing reference files called by NemoMod when running electricity model
		* REQUIRED TO RUN ELECTRICITY MODEL
	- fp_nemomod_temp_sqlite_db: optional file path to use for SQLite database used in Julia NemoMod Electricity model
		* If None, defaults to a temporary path sql database
	- logger: optional logging.Logger object used to log model events
	"""
	def __init__(self,
		model_attributes: ModelAttributes,
		fp_nemomod_reference_files: Union[str, None] = None, #sa.dir_ref_nemo
		fp_nemomod_temp_sqlite_db: Union[str, None] = None, #sa.fp_sqlite_nemomod_db_tmp
		logger: Union[logging.Logger, None] = None
	):
		# initialize input objects
		self.logger = logger
		self.model_attributes = model_attributes

		# initialize sql path for electricity projection and path to electricity models
		self._initialize_nemomod_reference_path(fp_nemomod_reference_files)
		self._initialize_nemomod_sql_path(fp_nemomod_temp_sqlite_db)

		# initialize models
		self._initialize_models()






	##############################################
	#    SUPPORT AND INITIALIZATION FUNCTIONS    #
	##############################################

	# get sector specifications
	def get_projection_sectors(self,
		sectors_project: Union[list, str, None] = None,
		delim: str = "|"
	) -> list:
		"""
			Check and retrieve valid projection subsectors to run in SISEPUEDEModels.project()

			Keyword Arguments
			------------------
			- sectors_project: list or string of sectors to run. If None, will run all valid sectors.
				* NOTE: sectors or sector abbreviations are accepted as valid inputs
			- delim: delimiter to use in input strings
		"""
		# get subsector attribute
		attr_sec = self.model_attributes.dict_attributes.get("abbreviation_sector")
		dict_map = attr_sec.field_maps.get(f"{attr_sec.key}_to_sector")
		valid_sectors_project = [dict_map.get(x) for x in attr_sec.key_values]

		# convert input to list
		if (sectors_project is None):
			list_out = valid_sectors_project
		elif isinstance(sectors_project, str):
			list_out = sectors_project.split(delim)
		elif isinstance(sectors_project, list) or isinstance(sectors_project, np.ndarray):
			list_out = list(sectors_project)
		# check values
		list_out = [dict_map.get(x, x) for x in list_out if dict_map.get(x, x) in valid_sectors_project]

		return list_out



	def _initialize_models(self,
	) -> None:
		"""
		Initialize the path to NemoMod reference files required for ingestion. Initializes
			the following properties:

			* self.allow_elecricity_run
			* self.fp_nemomod_reference_files
		"""

		self.model_afolu = AFOLU(self.model_attributes)
		self.model_circecon = CircularEconomy(self.model_attributes)
		self.model_electricity = ElectricEnergy(self.model_attributes, self.fp_nemomod_reference_files) if self.allow_elecricity_run else None
		self.model_energy = NonElectricEnergy(self.model_attributes)
		self.model_ippu = IPPU(self.model_attributes)
		self.model_socioeconomic = Socioeconomic(self.model_attributes)



	def _initialize_nemomod_reference_path(self,
		fp_nemomod_reference_files: Union[str, None]
	) -> None:
		"""
		Initialize the path to NemoMod reference files required for ingestion. Initializes
			the following properties:

			* self.allow_elecricity_run
			* self.fp_nemomod_reference_files
		"""

		self.allow_elecricity_run = False
		try:
			self.fp_nemomod_reference_files = sf.check_path(fp_nemomod_reference_files, False)
			self.allow_elecricity_run = True
		except Exception as e:
			self.fp_nemomod_reference_files = None
			self._log(f"Path to NemoMod reference files '{fp_nemomod_reference_files}' not found. The Electricity model will be disallowed from running.", type_log = "warning")



	def _initialize_nemomod_sql_path(self,
		fp_nemomod_temp_sqlite_db: Union[str, None]
	) -> None:
		"""
		Initialize the path to the NemoMod SQL database used to execute runs. Initializes
			the following properties:

			* self.fp_nemomod_temp_sqlite_db
		"""

		valid_extensions = ["sqlite", "db"]
		if isinstance(fp_nemomod_temp_sqlite_db, str):
			try_endings = [fp_nemomod_temp_sqlite_db.endswith(x) for x in valid_extensions]
			if any(try_endings):
				self.fp_nemomod_temp_sqlite_db = fp_nemomod_temp_sqlite_db
			else:
				fn_tmp = os.path.basename(tempfile.NamedTemporaryFile().name)
				fn_tmp = f"{fn_tmp}.sqlite"
				self.fp_nemomod_temp_sqlite_db = os.path.join(
					os.getcwd(),
					fn_tmp
				)

				self._log(f"Invalid path '{fp_nemomod_temp_sqlite_db}' specified as fp_nemomod_temp_sqlite_db. Using temporary path {self.fp_nemomod_temp_sqlite_db}.")



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
		run_integrated: bool = True
	) -> pd.DataFrame:
		"""
		Run SISEPUEDE models in appropriate order.

		Function Arguments
		------------------
		df_input_data: DataFrame containing SISEPUEDE inputs

		Optional Arguments
		------------------
		- models_run: list of sector models to run as defined in SISEPUEDEModels.model_attributes. Can
			include the following values:

			* AFOLU (or af)
			* Circular Economy (or ce)
			* IPPU (or ip)
			* Energy (or en)
				* Note: set include_electricity_in_energy = False to avoid running the electricity model with energy
			* Socioeconomic (or se)

		Keyword Arguments
		-----------------
		- include_electricity_in_energy: include the electricity model in runs of the energy model?
			* If False, runs without electricity (time intensive model)
		- run_integrated: run models as integrated collection?
			* If False, will run each model individually, without interactions (not recommended)
		"""

		df_return = []
		models_run = self.get_projection_sectors(models_run)

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
				df_return = [sf.merge_output_df_list(df_return, self.model_attributes, "concatenate")] if run_integrated else df_return
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
				df_return = [sf.merge_output_df_list(df_return, self.model_attributes, "concatenate")] if run_integrated else df_return
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
				df_return = [sf.merge_output_df_list(df_return, self.model_attributes, "concatenate")] if run_integrated else df_return
				self._log(f"NonElectricEnergy without Fugitive Emissions model run successfully completed", type_log = "info")

			except Exception as e:
				self._log(f"Error running NonElectricEnergy without Fugitive Emissions: {e}", type_log = "error")


		##  5. Run Electricity and collect output

		if ("Energy" in models_run) and include_electricity_in_energy and self.allow_elecricity_run:
			self._log("Running Energy model (Electricity: trying to call Julia)", type_log = "info")
			if run_integrated and set(["Circular Economy", "AFOLU"]).issubset(set(models_run)):
				df_input_data = self.model_attributes.transfer_df_variables(
					df_input_data,
					df_return[0],
					self.model_electricity.integration_variables
				)

			# create the engine and try to run Electricity
			engine = sqlalchemy.create_engine(f"sqlite:///{self.fp_nemomod_temp_sqlite_db}")
			try:
				df_elec = self.model_electricity.project(df_input_data, engine)
				df_return.append(df_elec)
				df_return = [sf.merge_output_df_list(df_return, self.model_attributes, "concatenate")] if run_integrated else df_return
				self._log(f"ElectricEnergy model run successfully completed", type_log = "info")

			except Exception as e:
				self._log(f"Error running ElectricEnergy model: {e}", type_log = "error")


		##  6. Finally, add fugitive emissions from Non-Electric Energy and collect output

		if "Energy" in models_run:
			print("\n\tRunning Energy (Fugitive Emissions)")
			if run_integrated and set(["IPPU", "AFOLU"]).issubset(set(models_run)):
				df_input_data = self.model_attributes.transfer_df_variables(
					df_input_data,
					df_return[0],
					self.model_energy.integration_variables_fgtv
				)

			try:
				df_return.append(self.model_energy.project(df_input_data, subsectors_project = self.model_attributes.subsec_name_fgtv))
				df_return = [sf.merge_output_df_list(df_return, self.model_attributes, "concatenate")] if run_integrated else df_return
				self._log(f"Fugitive Emissions from Energy model run successfully completed", type_log = "info")

			except Exception as e:
				self._log(f"Error running Fugitive Emissions from Energy model: {e}", type_log = "error")


		# build output data frame
		df_return = sf.merge_output_df_list(df_return, self.model_attributes, "concatenate") if (len(df_return) > 0) else pd.DataFrame()

		return df_return





class SISEPUEDEExperimentalManager:
	"""
	Launch and manage experiments based on LHS sampling over trajectories.


	Initialization Arguments
	------------------------
	- attribute_design: AttributeTable required to define experimental designs to run. Lever effects are
		evaluated using scalars `y`, which are derived from LHC samples that are subject to a linear
		transformation of the form

			`y = max(min(mx + b, sup), inf)`.

		For each design `_d`, the table passes information on the values of `m_d`, `b_d`, `sup_d` and
		`inf_d`. Each row of the table represents a different design.

	The table should include the following fields:
		* `linear_transform_l_b`: field containing `b`
		* `linear_transform_l_m`: field containing `m`
		* `linear_transform_l_inf`: field containing the infinum of lever effect scalar
		* `linear_transform_l_sup`: field containing the supremeum of lever effect scalar
		* `vary_l`: whether or not lever effects vary in the design (binary)
		* `vary_x`: whether or not exogenous uncertainties vary in the design (binary)

	- fp_templates: file path to directory containing input Excel templates
	- model_attributes: ModelAttributes class used to build baseline databases
	- regions: regions (degined in ModelAttributes) to run and build futures for.


	Optional Initialization Arguments
	---------------------------------
	- attribute_strategy: AttributeTable defining strategies. If not defined, strategies are inferred from
	 	templates.
	- demo_database_q: whether or not the input database is used as a demo
		* If run as demo, then `fp_templates` does not need to include subdirectories for each region
			specified
	- sectors: sectors to include
		* If None, then try to initialize all input sectors



	Notes
	-----
	-
	"""

	def __init__(self,
		attribute_design: AttributeTable,
		model_attributes: ModelAttributes,
		fp_templates: str,
		regions: Union[list, None],
		# lhs characteristics
		time_period_u0: int,
		n_trials: int,
		# optional/keyword arguments
		attribute_strategy: Union[AttributeTable, None] = None,
		demo_database_q: bool = True,
		sectors: Union[list, None] = None,

		base_future: Union[int, None] = None,
		fan_function_specification: str = "linear",
		field_uniform_scaling_q: str = "uniform_scaling_q",
		field_variable_trajgroup: str = "variable_trajectory_group",
		field_variable_trajgroup_type: str = "variable_trajectory_group_trajectory_type",
		field_variable: str = "variable",
		logger: Union[logging.Logger, None] = None,
		random_seed: Union[int, None] = None
	):

		self.model_attributes = model_attributes

		# initialize some key fields
		self.field_region = self.model_attributes.dim_region
		self.field_time_period = self.model_attributes.dim_time_period
		self.field_time_series_id = self.model_attributes.dim_time_series_id
		self.field_uniform_scaling_q = field_uniform_scaling_q
		self.field_variable = field_variable
		self.field_variable_trajgroup = field_variable_trajgroup
		self.field_variable_trajgroup_type = field_variable_trajgroup_type
		self.field_year = "year"

		# initialize keys--note: key_design is assigned in self._initialize_attribute_design
		self.key_future = self.model_attributes.dim_future_id
		self.key_primary = self.model_attributes.dim_primary_id
		self.key_strategy = self.model_attributes.dim_strategy_id

		# ordered by sort hierarchy
		self.sort_ordered_dimensions_of_analysis = self.model_attributes.sort_ordered_dimensions_of_analysis

		# initialize additional components
		self.fan_function_specification = fan_function_specification
		self.logger = logger
		self.n_trials = n_trials
		self.time_period_u0 = time_period_u0
		self.random_seed = random_seed

		# initialize some SQL information for restoration and/or archival
		self._initialize_archival_settings()

		# initialize key elements
		self._initialize_attribute_design(attribute_design)
		self._initialize_base_future(base_future)
		self._initialize_baseline_database(
			fp_templates,
			regions,
			demo_database_q
		)
		self._initialize_future_trajectories(
			fan_function_specification = self.fan_function_specification,
			field_time_period = self.field_time_period,
			field_uniform_scaling_q = self.field_uniform_scaling_q,
			field_variable = self.field_variable,
			field_variable_trajgroup = self.field_variable_trajgroup,
			field_variable_trajgroup_type = self.field_variable_trajgroup_type,
			key_future = self.key_future,
			key_strategy = self.key_strategy,
			logger = self.logger
		)
		self._initialize_lhs_design()

		# generate some elements
		self._generate_primary_keys_index()








	##################################
	#    INITIALIZATION FUNCTIONS    #
	##################################

	def _initialize_archival_settings(self,
	) -> None:
		"""
		Initialize key archival settings used to store necessary experimental parameters,
			Latin Hypercube Samples, ModelAttribute tables, and more. Sets the following
			properties:

			* self.

		"""

		self.archive_table_name_experimental_configuration = "EXPERIMENTAL_CONFIGURATION"
		self.archive_table_name_lhc_samples_l = "LHC_SAMPLES_LEVER_EFFECTS"
		self.archive_table_name_lhc_samples_x = "LHC_SAMPLES_EXOGENOUS_UNCERTAINTIES"



	def _initialize_attribute_design(self,
		attribute_design: AttributeTable,
		field_transform_b: str = "linear_transform_l_b",
		field_transform_m: str = "linear_transform_l_m",
		field_transform_inf: str = "linear_transform_l_inf",
		field_transform_sup: str = "linear_transform_l_sup",
		field_vary_l: str = "vary_l",
		field_vary_x: str = "vary_x",
		logger: Union[logging.Logger, None] = None
	) -> None:
		"""
		Verify AttributeTable attribute_design specified for the design and set properties if valid. Initializes
			the following properties if successful:

		* self.attribute_design
		* self.field_transform_b
		* self.field_transform_m
		* self.field_transform_inf
		* self.field_transform_sup
		* self.field_vary_l
		* self.field_vary_x
		* self.key_design


		Function Arguments
		------------------
		- attribute_design: AttributeTable used to define different designs

		Keyword Arguments
		-----------------
		- field_transform_b: field in attribute_design.table giving the value of `b` for each attribute_design.key_value
		- field_transform_m: field in attribute_design.table giving the value of `m` for each attribute_design.key_value
		- field_transform_inf: field in attribute_design.table giving the value of `inf` for each attribute_design.key_value
		- field_transform_sup: field in  attribute_design.table giving the value of `sup` for each attribute_design.key_value
		- field_vary_l: required field in attribute_design.table denoting whether or not LEs vary under the design
		- field_vary_x: required field in attribute_design.table denoting whether or not Xs vary under the design
		- logger: optional logging.Logger() object to log to
			* if None, warnings are sent to standard out

		"""

		# verify input type
		if not isinstance(attribute_design, AttributeTable):
			tp = str(type(attribute_design))
			self._log(f"Invalid type '{tp}' in specification of attribute_design: attribute_design should be an AttributeTable.")

		# check required fields (throw error if not present)
		required_fields = [
			field_transform_b,
			field_transform_m,
			field_transform_inf,
			field_transform_sup,
			field_vary_l,
			field_vary_x
		]
		sf.check_fields(attribute_design.table, required_fields)

		# if successful, set properties
		self.attribute_design = attribute_design
		self.field_transform_b = field_transform_b
		self.field_transform_m = field_transform_m
		self.field_transform_inf = field_transform_inf
		self.field_transform_sup = field_transform_sup
		self.field_vary_l = field_vary_l
		self.field_vary_x = field_vary_x
		self.key_design = attribute_design.key



	def _initialize_base_future(self,
		future: Union[int, None]
	) -> None:
		"""
		Set the baseline future. If None, defaults to 0. Initializes the following
			properties:

			* self.baseline_future
		"""

		self.baseline_future = int(min(future, 0)) if (future is not None) else 0



	def _initialize_baseline_database(self,
		fp_templates: str,
		regions: Union[List[str], None],
		demo_q: bool
	) -> None:
		"""
		Initialize the BaseInputDatabase class used to construct future trajectories.
			Initializes the following properties:

			* self.attribute_strategy
			* self.base_input_database
			* self.baseline_strategy


		Function Arguments
		------------------
		- fp_templates: path to templates (see ?BaseInputDatabase for more information)
		- regions: list of regions to run experiment for
			* If None, will attempt to initialize all regions defined in ModelAttributes
		- demo_q: import templates run as a demo (region-independent)?
		"""

		self._log("Initializing BaseInputDatabase", type_log = "info")

		try:
			self.base_input_database = BaseInputDatabase(
				fp_templates,
				self.model_attributes,
				regions,
				demo_q = demo_q,
				logger = self.logger
			)

			self.attribute_strategy = self.base_input_database.attribute_strategy
			self.baseline_strategy = self.base_input_database.baseline_strategy

		except Exception as e:
			msg = f"Error initializing BaseInputDatabase -- {e}"
			self._log(msg, type_log = "error")
			raise RuntimeError(msg)



	def _initialize_future_trajectories(self,
		**kwargs
	) -> None:
		"""
		Initialize the FutureTrajectories object for executing experiments. Initializes
			the following properties:

			* self.future_trajectories
			* self.n_factors
			* self.n_factors_l
			* self.n_factors_x
		"""

		self._log("Initializing FutureTrajectories", type_log = "info")

		try:
			self.future_trajectories = FutureTrajectories(
				self.base_input_database.database,
				{
					self.key_strategy: self.base_input_database.baseline_strategy
				},
				self.time_period_u0,
				**kwargs
			);

			self.n_factors = len(self.future_trajectories.all_sampling_units)
			self.n_factors_l = len(self.future_trajectories.all_sampling_units_l)
			self.n_factors_x = len(self.future_trajectories.all_sampling_units_x)


		except Exception as e:
			msg = f"Error initializing FutureTrajectories -- {e}"
			self._log(msg, type_log = "error")
			raise RuntimeError(msg)



	def _initialize_lhs_design(self,
	) -> None:
		"""
		Initializes LHS design and associated tables used in the Experiment. Creates the
			following properties:

			* self.lhs_design

		"""

		self._log("Initializing LHSDesign", type_log = "info")

		try:
			self.lhs_design = LHSDesign(
				self.attribute_design,
				self.key_future,
				n_factors_l = self.n_factors_l,
				n_factors_x = self.n_factors,
				n_trials = self.n_trials,
				random_seed = self.random_seed,
				fields_factors_l = self.future_trajectories.all_sampling_units_l,
				fields_factors_x = self.future_trajectories.all_sampling_units,
				logger = self.logger
			)

		except Exception as e:
			msg = f"Error initializing FutureTrajectories -- {e}"
			self._log(msg, type_log = "error")
			raise RuntimeError(msg)



	def _log(self,
		msg: str,
		type_log: str = "log",
		**kwargs
	) -> None:
		"""
		Clean implementation of sf._optional_log in-line using default logger. See
			?sf._optional_log for more information.

		Function Arguments
		------------------
		- msg: message to log

		Keyword Arguments
		-----------------
		- type_log: type of log to use
		- **kwargs: passed as logging.Logger.METHOD(msg, **kwargs)
		"""
		sf._optional_log(self.logger, msg, type_log = type_log, **kwargs)



	def _restore_from_database(self,
		table_name_experimental_configuration: Union[str, None] = None,
		table_name_lhs_l: Union[str, None] = None,
		table_name_lhs_x: Union[str, None] = None
	) -> None:
		"""
		Restore a SISEPUEDE Experimental Session from an SQL database containing the following tables:

			*
		-
		-
		"""


		return None




	##############################
	#	SUPPORTING FUNCTIONS	#
	##############################

	def generate_database(self,
		list_primary_keys: Union[list, None] = None
	) -> pd.DataFrame:
		"""
		Generate an data of inputs for primary keys specified in list_primary_keys.

		Optional Arguments
		------------------
		- list_primary_keys: list of primary keys to include in input database.
			* If None, uses
		"""
		return None



	def _generate_primary_keys_index(self,
	) -> None:
		"""
		Generate a data frame of primary scenario keys. Assigns the following
			properties:

			* self.all_
			* self.primary_key_database
		"""

		self._log(f"Generating primary keys (values of {self.key_primary})...", type_log = "info")

		# get all designs, strategies, and futures
		all_designs = self.attribute_design.key_values
		all_strategies = self.base_input_database.attribute_strategy.key_values
		all_futures = [self.baseline_future]
		all_futures += self.lhs_design.vector_lhs_key_values if (self.lhs_design.vector_lhs_key_values is not None) else []

		prods = [
			all_designs,
			all_strategies,
			all_futures
		]

		df_primary_keys = pd.DataFrame(
		 	list(itertools.product(*prods)),
			columns = [self.key_design, self.key_strategy, self.key_future]
		)

		df_primary_keys = sf.add_data_frame_fields_from_dict(
			df_primary_keys,
			{
				self.key_primary: range(len(df_primary_keys))
			}
		)

		self.primary_key_database = df_primary_keys






	############################
	#    CORE FUNCTIONALITY    #
	############################








# wrapper for ExperimentalManager and RunSisepuede
class SISEPUEDE:
	"""
	SISEPUEDE is a ...

	Includes following classes...


	See documentation at (LINK HERE)



	"""

	def __init__(self, ):
		return None
