from attribute_table import AttributeTable
import datetime
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
import re
from sampling_unit import FutureTrajectories
import support_functions as sf
import sqlalchemy
import tempfile
import time
from typing import *
import warnings



class SISEPUEDEAnalysisID:
	"""
	Create a unique ID for each session/set of runs. Can be instantiated using a
		string (from a previous run) or empty, which creates a new ID.

	Initialization Arguments
	------------------------
	- id_str: optional entry of a previous string containing an ID.
		* If None, creates a new ID based on time in isoformat
	- logger: optional log object to pass
	- regex_template: optional regular expression used to parse id
		* Should take form
			re.compile("TEMPLATE_STRING_HERE_(.+$)")
		where whatever is contained in (.+$) is assumed to be an isoformat time.


	"""
	def __init__(self,
		id_str: Union[str, None] = None,
		logger: Union[logging.Logger, None] = None,
		regex_template: Union[str, None] = None
	):
		self.logger = logger
		self._check_id(id_str)


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



	def _check_id(self,
		id_str: Union[str, None] = None,
		regex_template: Union[re.Pattern, None] = None
	) -> None:
		"""
		Set the SISEPUEDE runtime ID to distinguish between different analytical
		 	runs. Sets the following properties:

			* self.default_regex_template
			* self.regex_template
			* self.id
			* self.isoformat
			* self.year
			* self.month
			* self.day
			* self.hour
			* self.minute
			* self.second
			* self.microsecond
		"""

		self.isoformat = None
		self.default_regex_template = re.compile("sisepuede_run_(.+$)")
		self.regex_template = self.default_regex_template if not isinstance(regex_template, re.Pattern) else regex_template
		# get regex substitution
		str_regex_sub = [x for x in self.regex_template.split(self.regex_template.pattern) if (x != "")]
		str_regex_sub = str_regex_sub[0] if (len(str_regex_sub) > 0) else None
		date_info = None

		# try to initialize from string if specified
		if isinstance(id_str, str):
			match = self.regex_template.match(id_str)
			if match is not None:
				try:
					date_info = datetime.datetime.fromisoformat(match.groups()[0])
					self.isoformat = match.groups()[0]
					self.id = id_str

				except Exception as e:
					self._log(f"Error in SISEPUEDEAnalysisID trying to initialize ID '{id_str}': {e}.\n\tDefaulting new ID.", type_log = None)
					id_str = None
			else:
				id_str = None

		# otherwise, create a new one
		if id_str is None:
			date_info = datetime.datetime.now()
			self.isoformat = date_info.isoformat()
			self.id = self.regex_template.pattern.replace(str_regex_sub, self.isoformat) if (str_regex_sub is not None) else f"{self.regex_template.pattern}_{self.isoformat}"

		# set properties
		(
			self.year,
			self.month,
			self.day,
			self.hour,
			self.minute,
			self.second,
			self.microsecond
		) = (
			date_info.year,
			date_info.month,
			date_info.day,
			date_info.hour,
			date_info.minute,
			date_info.second,
			date_info.microsecond
		)

		# note the success
		self._log(f"Successfully initialized SISEPUEDE Analysis ID '{self.id}'", type_log = "info")



	def _set_file_string(self,
	) -> None:
		"""
		Set the file-system safe string. Sets the following properties:

		* self.id_fs_safe
		* self.dict_id_from_fs_safe_replacements
		* self.dict_id_to_fs_safe_replacements

		"""

		self.dict_id_to_fs_safe_replacements = {":": ";"}
		self.dict_id_from_fs_safe_replacements = sf.reverse_dict(self.dict_id_to_fs_safe_replacements)
		self.id_fs_safe = self.id_to_file_safe_id()


	########################################################################
	#    SOME FUNCTIONS FOR CONVERTING TO/FROM FILE SYSTEM-SAFE STRINGS    #
	########################################################################

	def id_from_file_safe_id(self,
		id: str,
		dict_replacements: Union[Dict, None] = None
	) -> str:
		"""
		Convert a file-system safe string to an ID string (invert invalid characters
			to support POSIX strings).

		Function Arguments
		------------------
		- id: file-system safe string to initialize as id

		Keyword Arguments
		-----------------
		- dict_replacements: dictionary to use to replace file-system safe substrings
			with ID-valid strings
		"""

		dict_replacements = self.dict_id_from_fs_safe_replacements if (dict_replacements is None) else dict_replacements

		return sf.str_replace(id, dict_replacements)



	def id_to_file_safe_id(self,
		id: Union[str, None] = None,
		dict_replacements: Union[Dict, None] = None
	) -> str:
		"""
		Convert an id to a file-system safe string (replace invalid characters).

		Keyword Arguments
		-----------------
		- id: POSIX-time based SISEPUEDEAnalysisID.id string to replace
		- dict_replacements: dictionary to use to replace substrings
		"""
		id = self.id if (id is None) else id
		dict_replacements = self.dict_id_to_fs_safe_replacements if (dict_replacements is None) else dict_replacements

		return sf.str_replace(id, dict_replacements)






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
	- id_str: Optional id_str used to create SISEPUEDEAnalysisID (see ?SISEPUEDEAnalysisID for more
		information on properties). Can be used to set outputs for a previous ID/restore a session.
		* If None, creates a unique ID for the session (used in output file names)
	- logger: optional logging.Logger object used for logging

	"""
	def __init__(self,
		dir_ingestion: Union[str, None] = None,
		fn_config: str = "sisepuede.config",
		id_str: Union[str, None] = None,
		logger: Union[logging.Logger, None] = None
	):

		self.logger = logger
		self._initialize_analysis_id(id_str)

		# run checks of directories
		self._check_config(fn_config)
		self._check_required_directories()
		self._check_ingestion(dir_ingestion)
		self._check_optional_directories()

		# initialize model attributes, set runtime id, then check/instantiate downstream file paths
		self._initialize_model_attributes()
		self._check_nemomod_reference_file_paths()
		self._initialize_file_path_defaults()



	##############################
	#	SUPPORTING FUNCTIONS	#
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
	#	DIRECTORY CHECKS	#
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



	def _check_required_directories(self,
	) -> None:
		"""
		Check directory structure for SISEPUEDE. Sets the following properties:

			* self.dir_attribute_tables
			* self.dir_docs
			* self.dir_jl
			* self.dir_proj
			* self.dir_py
			* self.dir_ref
			* self.dir_ref_nemo
			* self.fp_config
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
		else:
			self._log(f"Verification of SISEPUEDE directory structure completed successfully with 0 errors.", type_log = "info")



	def _check_ingestion(self,
		dir_ingestion: Union[str, None]
	) -> None:
		"""
		Check path to templates. Sets the following properties:

			* self.dir_ingestion
			* self.dict_data_mode_to_template_directory
			* self.valid_data_modes

		Function Arguments
		------------------
		dir_ingestion: ingestion directory storing input templates for SISEPUEDE
			* If None, defaults to ..PATH_SISEPUEDE/ref/ingestion
		"""

		##  Check template ingestion path (within reference directory)

		# initialize
		self.valid_data_modes = ["calibrated", "demo", "uncalibrated"]
		self.dir_ingestion = os.path.join(self.dir_ref, "ingestion") if (self.dir_ref is not None) else None
		self.dict_data_mode_to_template_directory = None

		# override if input path is specified
		if isinstance(dir_ingestion, str):
			if os.path.exists(dir_ingestion):
				self.dir_ingestion = dir_ingestion

		# check existence
		if not os.path.exists(self.dir_ingestion):
			self._log(f"\tIngestion templates subdirectory '{self.dir_ingestion}' not found", type_log = "error")
			self.dir_ingestion = None
		else:
			self.dict_data_mode_to_template_directory = dict(zip(
				self.valid_data_modes,
				[os.path.join(self.dir_ingestion, x) for x in self.valid_data_modes]
			))



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

	def _initialize_analysis_id(self,
		id_str: Union[str, None]
	) -> None:
		"""
		Initialize the session id. Initializes the following properties:

			* self.sisepuede_analysis_id (SISEPUEDEAnalysisID object)
			* self.analysis_id (shortcurt to self.sisepuede_analysis_id.id)
		"""
		self.sisepuede_analysis_id = SISEPUEDEAnalysisID(
			id_str = id_str,
			logger = self.logger
		)
		self.analysis_id = self.sisepuede_analysis_id.id



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



	def _check_nemomod_reference_file_paths(self,
	) -> None:
		"""
		Check and initiailize any NemoMod reference file file paths. Sets the following properties:

			* self.allow_electricity_run
			* self.required_reference_tables_nemomod
		"""

		# initialize
		self.allow_electricity_run = True
		self.required_reference_tables_nemomod = None

		# error handling
		count_errors = 0
		msg_error = ""

		if (self.dir_ref_nemo is not None) and (self.dir_jl is not None):

			# nemo mod input files - specify required, run checks
			model_electricity = ElectricEnergy(
				self.model_attributes,
				self.dir_jl,
				self.dir_ref_nemo,
				initialize_julia = False
			)
			self.required_reference_tables_nemomod = model_electricity.required_reference_tables

			# initialize dictionary of file paths
			dict_nemomod_reference_tables_to_fp_csv = dict(zip(
				self.required_reference_tables_nemomod,
				[None for x in self.required_reference_tables_nemomod]
			))

			# check all required tables
			for table in self.required_reference_tables_nemomod:
				fp_out = os.path.join(self.dir_ref_nemo, f"{table}.csv")
				if os.path.exists(fp_out):
					dict_nemomod_reference_tables_to_fp_csv.update({table: fp_out})
				else:
					count_errors += 1
					msg_error += f"\n\tNemoMod reference table '{table}' not found in directory {self.dir_ref_nemo}."
					self.allow_electricity_run = False
					del dict_nemomod_reference_tables_to_fp_csv[table]
		else:
			count_errors += 1
			msg_error = "\n\tNo NemoMod model refererence files were found."
			self.allow_electricity_run = False

		if msg_error != "":
			self._log(f"There were {count_errors} while trying to initialize NemoMod:{msg_error}\nThe electricity model cannot be run. Disallowing electricity model runs.", type_log = "error")
		else:
			self._log(f"NemoMod reference file checks completed successfully.", type_log = "info")



	def _initialize_file_path_defaults(self,
	) -> None:
		"""
		Initialize any default file paths, including output and temporary files. Sets the
			following properties:

			* self.fp_base_output_raw
			* self.fp_sqlite_tmp_nemomod_intermediate
		"""

		self.fp_base_output_raw = None
		self.fp_sqlite_tmp_nemomod_intermediate = None

		fbn_raw = f"{self.analysis_id}_outputs_raw".replace(":", ";")

		if self.dir_out is not None:
			# base output path for CSV or SQL--if CSVs, represents a directory. If SQLite, append .sqlite to get path
			self.fp_base_output_raw = os.path.join(self.dir_out, fbn_raw)
			# SQLite Database location for intermediate NemoMod calculations
			self.fp_sqlite_tmp_nemomod_intermediate = os.path.join(self.dir_tmp, "nemomod_intermediate_database.sqlite")





class SISEPUEDEModels:
	"""
	Instantiate models based on

	Initialization Arguments
	------------------------
	- model_attributes: ModelAttributes object used to manage variables and coordination

	Optional Arguments
	------------------
	- allow_electricity_run: allow the electricity model to run (high-runtime model)
		* Generally should be left to True
	- fp_nemomod_reference_files: directory housing reference files called by NemoMod when running electricity model
		* REQUIRED TO RUN ELECTRICITY MODEL
	- fp_nemomod_temp_sqlite_db: optional file path to use for SQLite database used in Julia NemoMod Electricity model
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
		self.model_energy = NonElectricEnergy(self.model_attributes)
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
		Initialize the path to the NemoMod SQL database used to execute runs. Initializes
			the following properties:

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
	#	CORE FUNCTIONALITY	#
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

		if ("Energy" in models_run) and include_electricity_in_energy and self.allow_electricity_run:
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
	Launch and manage experiments based on LHS sampling over trajectories. The SISEPUEDEExperimentalManager
		class reads in input templates to generate input databases, controls deployment, generation of
		multiple runs, writing output to applicable databases, and post-processing of applicable metrics.
		Users should use SISEPUEDEExperimentalManager to set the number of trials and the start year of
		uncertainty.


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
		self.demo_mode = demo_database_q
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
		# IMPORTANT: HEREHERE NEED TO INTEGRATE REGIONAL COMPONENT-
		# ONE FutureTrajectories OBJECT FOR EACH REGION?
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
			self._log(f"Invalid type '{tp}' in specification of attribute_design: attribute_design should be an AttributeTable.", type_log = "error")

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
			* self.regions


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
			self.regions = self.base_input_database.regions

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

			* self.dict_future_trajectories
			* self.dict_n_factors
			* self.dict_n_factors_l
			* self.dict_n_factors_x

			Additionally, can update

			* self.regions

			if any regions fail.
		"""

		self._log("Initializing FutureTrajectories", type_log = "info")

		self.dict_future_trajectories = {}
		self.dict_n_factors = {}
		self.dict_n_factors_l = {}
		self.dict_n_factors_x = {}

		drop_regions = []

		# filter base input database for each region to instantiate a new FutureTrajectories object
		for region in self.regions:

			region_print = self.get_output_region(region)

			try:
				df_input = self.base_input_database.database[
					self.base_input_database.database[self.field_region] == region
				].reset_index(drop = True)

				future_trajectories_cur = FutureTrajectories(
					df_input,
					{
						self.key_strategy: self.base_input_database.baseline_strategy
					},
					self.time_period_u0,
					**kwargs
				)

				self.dict_future_trajectories.update({
					region: future_trajectories_cur
				})
				self.dict_n_factors.update({
					region: len(future_trajectories_cur.all_sampling_units)
				})
				self.dict_n_factors_l.update({
					region: len(future_trajectories_cur.all_sampling_units_l)
				})
				self.dict_n_factors_x.update({
					region: len(future_trajectories_cur.all_sampling_units_x)
				})

				self._log(f"\tFutureTrajectories for '{region_print}' complete.", type_log = "info")

			except Exception as e:
				self._log(f"Error initializing FutureTrajectories for region {region_print} -- {e}.", type_log = "error")
				self._log(f"Dropping region '{region_print}' due to error in FutureTrajectories initialization.", type_log = "warning")
				drop_regions.append(region)

		# update regions if necessary
		self.regions = [x for x in self.regions if (x not in drop_regions)]
		if len(self.regions) == 0:
			raise RuntimeError(f"Error initializing SISEPUEDE: no regions left to instantiate.")



	def _initialize_lhs_design(self,
	) -> None:
		"""
		Initializes LHS design and associated tables used in the Experiment. Creates the
			following properties:

			* self.dict_lhs_design
			* self.vector_lhs_key_values

			Additionally, can update

			* self.regions

			if any regions fail.
		"""

		self._log("Initializing LHSDesign", type_log = "info")

		self.dict_lhs_design = {}
		self.vector_lhs_key_values = None

		drop_regions = []

		for region in self.regions:

			region_print = self.get_output_region(region)

			try:

				future_trajectories_cur = self.dict_future_trajectories.get(region)
				n_factors = self.dict_n_factors.get(region)
				n_factors_l = self.dict_n_factors_l.get(region)

				lhs_design_cur = LHSDesign(
					self.attribute_design,
					self.key_future,
					n_factors_l = n_factors_l,
					n_factors_x = n_factors,
					n_trials = self.n_trials,
					random_seed = self.random_seed,
					fields_factors_l = future_trajectories_cur.all_sampling_units_l,
					fields_factors_x = future_trajectories_cur.all_sampling_units,
					logger = self.logger
				)

				self.dict_lhs_design.update({
					region: lhs_design_cur
				})

				self.vector_lhs_key_values = lhs_design_cur.vector_lhs_key_values if (self.vector_lhs_key_values is None) else self.vector_lhs_key_values

				self._log(f"\tLHSDesign for region '{region_print}' complete.", type_log = "info")

			except Exception as e:
				self._log(f"Error initializing LHSDesign for region '{region_print}' -- {e}.", type_log = "error")
				self._log(f"Dropping region '{region_print}' due to error in LHSDesign initialization.", type_log = "warning")
				drop_regions.append(region)

		# update regions if necessary
		self.regions = [x for x in self.regions if (x not in drop_regions)]
		if len(self.regions) == 0:
			raise RuntimeError(f"Error initializing SISEPUEDE: no regions left to instantiate.")



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
		all_futures += self.vector_lhs_key_values if (self.vector_lhs_key_values is not None) else []

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



	def get_output_region(self,
		region: str,
		str_demo_region: str = "DEMO"
	) -> str:
		"""
		Retrieve a region for output tables

		Function Arguments
		------------------
		- region: input region to convert

		Keyword Arguments
		-----------------
		- str_demo_region: string specifying a region for a demo run
		"""

		out = str_demo_region if self.demo_mode else region

		return out





	############################
	#    CORE FUNCTIONALITY    #
	############################








# wrapper for ExperimentalManager and RunSisepuede
class SISEPUEDE:

	"""
	SISEPUEDE is a ...

	Includes following classes...


	See documentation at (LINK HERE)


	Initialization Arguments
	------------------------
	- data_mode: template class to initialize from. Three options are allowed:
		* calibrated
		* demo
		* uncalibrated


	Optional Arguments
	------------------

	Optional arguments are used to pass values to SISEPUEDE outside of the configuaration framework.
		This can be a desireable approach for iterative modeling or determining a suitable data
		pipeline.

	- attribute_design: optional AttributeTable object used to specify the design_id.
		* Note: If None, will attempt to find a table within the ModelAttributes object with key
		"dim_design_id". If none is found, will assume a single design where:
			(a) exogenous uncertainties vary at ranges specified in input templates; and
			(b) lever effects are fixed.

	- dir_ingestion: directory storing SISEPUEDE templates for ingestion
		* Note: if running outside of demo mode, the directory should contain subdirectories dor each
			region, with each region including input templates for each of the 5 SISEPUEDE sectors. For
			example, the directory should have the following tree structure:

			* dir_ingestion
				|_ calibrated
					|_ region_1
						|_ model_input_variables_af_demo.xlsx
						|_ model_input_variables_ce_demo.xlsx
						|_ model_input_variables_en_demo.xlsx
						|_ model_input_variables_ip_demo.xlsx
						|_ model_input_variables_se_demo.xlsx
					|_ region_2
						|_ model_input_variables_af_demo.xlsx
						.
						.
						.
				|_ demo
					|_ model_input_variables_af_demo.xlsx
					|_ model_input_variables_ce_demo.xlsx
					|_ model_input_variables_en_demo.xlsx
					|_ model_input_variables_ip_demo.xlsx
					|_ model_input_variables_se_demo.xlsx
				|_ uncalibrated
					|_ region_1
						|_ model_input_variables_af_demo.xlsx
						|_ model_input_variables_ce_demo.xlsx
						|_ model_input_variables_en_demo.xlsx
						|_ model_input_variables_ip_demo.xlsx
						|_ model_input_variables_se_demo.xlsx
					|_ region_2
						|_ model_input_variables_af_demo.xlsx
						.
						.
						.

	- logger: Optional logging.Logger object to use for logging
	- regions: list of regions to include in the experiment. [NOTE, in v1.0, this shuld be limited to a
		single region]

	-

	"""

	def __init__(self,
		data_mode: str,
		attribute_design: Union[AttributeTable, None] = None,
		dir_ingestion: Union[str, None] = None,
		logger: Union[logging.Logger, None]= None,
		regions: Union[List[str], None] = None
	):

		self.logger = logger
		self._initialize_file_structure(dir_ingestion)
		self._initialize_data_mode(data_mode)
		self._initialize_experimental_manager(regions = regions)
		self._initialize_models()

		self._initialize_function_aliases()



	##############################################
	#    SUPPORT AND INITIALIZATION FUNCTIONS    #
	##############################################

	def get_config_parameter(self,
		parameter: str
	) -> Union[int, List, str]:
		"""
		Retrieve a configuration parameter from self.model_attributes. Must be initialized
			after _initialize_file_structure()
		"""

		return self.model_attributes.configuration.get(parameter)



	def _initialize_data_mode(self,
		data_mode: Union[str, None] = None,
		default_mode: str = "demo"
	) -> None:
		"""
		Initialize mode of operation. Sets the following properties:

			* self.data_mode
			* self.demo_mode
			* self.dir_templates
			* self.valid_data_modes
		"""
		self.valid_data_modes = self.file_struct.valid_data_modes

		try:
			data_mode = default_mode if (data_mode is None) else data_mode
			self.data_mode = default_mode if (data_mode not in self.valid_data_modes) else data_mode
			self.demo_mode = (self.data_mode == "demo")
			self.dir_templates = self.file_struct.dict_data_mode_to_template_directory.get(self.data_mode) if (self.file_struct is not None) else None

			self._log(f"Running SISEPUEDE under template data mode '{self.data_mode}'.", type_log = "info")

		except Exception as e:
			self._log(f"Error in _initialize_data_mode(): {e}", type_log = "error")
			raise RuntimeError()



	def _initialize_experimental_manager(self,
		attribute_design: Union[AttributeTable, None] = None,
		key_config_n_lhs: str = "num_lhc_samples",
		key_config_random_seed: str = "random_seed",
		key_config_time_period_u0: str = "time_period_u0",
		key_model_attributes_design: str = "dim_design_id",
		num_trials: Union[int, None] = None,
		random_seed: Union[int, None] = None,
		regions: Union[List[str], None] = None,
		time_t0_uncertainty: Union[int, None] = None
	) -> None:
		"""
		Initialize the Experimental Manager for SISEPUEDE. The SISEPUEDEExperimentalManager
			class reads in input templates to generate input databases, controls deployment,
			generation of multiple runs, writing output to applicable databases, and
			post-processing of applicable metrics. Users should use SISEPUEDEExperimentalManager
			to set the number of trials and the start year of uncertainty. Sets the following
			properties:

			* self.attribute_primary
			* self.baseline_future
			* self.baseline_strategy
			* self.experimental_manager
			* self.key_design
			* self.key_future
			* self.key_primary
			* self.key_strategy
			* self.keys_index
			* self.n_trials
			* self.random_seed
			* self.regions
			* self.time_period_u0


		Keyword Arguments
		-----------------
		- attribute_design: AttributeTable used to specify designs.
			* If None, tries to access "dim_design_id" from ModelAttributes.dict_attributes
		- key_config_n_lhs: configuration key used to determine the number of LHC samples to
			generate
		- key_config_random_seed: configuration key used to set the random seed
		- key_config_time_period_u0: configuration key used to determine the time period of
			initial uncertainty in uncertainty assessment.
		- key_model_attributes_design: key in model_attributes.dict_attributes used to try and
			get design attribute.
			* If None, defaults to "dim_design_id"
		- num_trials: number if LHS trials to run.
			* If None, revert to configuration defaults from self.model_attributes
		- random_seed: random seed used to generate LHS samples
			* If None, revert to configuration defaults from self.model_attributes
		- regions: regions to initialize.
			* If None, initialize using all regions
		- time_t0_uncertainty: time where uncertainty starts
		"""

		num_trials = int(max(num_trials, 0)) if (isinstance(num_trials, int) or isinstance(num_trials, float)) else self.get_config_parameter(key_config_n_lhs)
		attribute_design = self.model_attributes.dict_attributes.get(key_model_attributes_design) if not isinstance(attribute_design, AttributeTable) else attribute_design

		self.experimental_manager = None
		self.n_trials = self.get_config_parameter(key_config_n_lhs)
		self.time_period_u0 = self.get_config_parameter(key_config_time_period_u0)
		self.random_seed = self.get_config_parameter(key_config_time_period_u0)

		try:
			self.experimental_manager = SISEPUEDEExperimentalManager(
				attribute_design,
				self.model_attributes,
				self.dir_templates,
				regions,
				self.time_period_u0,
				self.n_trials,
				demo_database_q = self.demo_mode,
				logger = self.logger,
				random_seed = self.random_seed
			)

			self._log(f"Successfully initialized SISEPUEDEExperimentalManager.", type_log = "info")

		except Exception as e:
			self._log(f"Error initializing the experimental manager in _initialize_experimental_manager(): {e}", type_log = "error")
			raise RuntimeError()

		self.attribute_primary = AttributeTable(
			self.experimental_manager.primary_key_database,
			self.experimental_manager.key_primary,
			[]
		)

		self.baseline_future = self.experimental_manager.baseline_future
		self.baseline_strategy = self.experimental_manager.baseline_strategy

		self.field_region = self.experimental_manager.field_region
		self.key_design = self.experimental_manager.key_design
		self.key_future = self.experimental_manager.key_future
		self.key_primary = self.experimental_manager.key_primary
		self.key_strategy = self.experimental_manager.key_strategy
		self.keys_index = [
			self.key_design,
			self.key_future,
			self.key_primary,
			self.key_strategy
		]

		self.n_trials = self.experimental_manager.n_trials
		self.random_seed = self.experimental_manager.random_seed
		self.regions = self.experimental_manager.regions
		self.time_period_u0 = self.experimental_manager.time_period_u0



	def _initialize_file_structure(self,
		dir_ingestion: Union[str, None] = None
	) -> None:

		"""
		Intialize the SISEPUEDEFileStructure object and model_attributes object. Initializes
			the following properties:

			* self.file_struct
			* self.model_attributes
		"""

		self.file_struct = None
		self.model_attributes = None

		try:
			self.file_struct = SISEPUEDEFileStructure(
				dir_ingestion = dir_ingestion
			)

			self._log(f"Successfully initialized SISEPUEDEFileStructure.", type_log = "info")

		except Exception as e:
			self._log(f"Error trying to initialize SISEPUEDEFileStructure: {e}", type_log = "error")
			raise RuntimeError()

		self.model_attributes = self.file_struct.model_attributes



	def _initialize_models(self,
		dir_jl: Union[str, None] = None,
		dir_nemomod_reference_files: Union[str, None] = None,
		fp_sqlite_tmp_nemomod_intermediate: Union[str, None] = None
	) -> None:
		"""
		Initialize models for SISEPUEDE. Sets the following properties:

			* self.dir_jl
			* self.dir_nemomod_reference_files
			* self.fp_sqlite_tmp_nemomod_intermediate
			* self.models

		Optional Arguments
		------------------
		For the following arguments, entering = None will return the SISEPUEDE default
		- dir_jl: file path to julia environment and supporting module directory
		- dir_nemomod_reference_files: directory containing NemoMod reference files
		- fp_nemomod_temp_sqlite_db: file name for temporary database used to run NemoMod
		"""

		dir_jl = self.file_struct.dir_jl if (dir_jl is None) else dir_jl
		dir_nemomod_reference_files = self.file_struct.dir_ref_nemo if (dir_nemomod_reference_files is None) else dir_nemomod_reference_files
		fp_sqlite_tmp_nemomod_intermediate = self.file_struct.fp_sqlite_tmp_nemomod_intermediate if (fp_sqlite_tmp_nemomod_intermediate is None) else fp_sqlite_tmp_nemomod_intermediate

		try:
			self.models = SISEPUEDEModels(
				self.model_attributes,
				allow_electricity_run = self.file_struct.allow_electricity_run,
				fp_julia = dir_jl,
				fp_nemomod_reference_files = dir_nemomod_reference_files,
				fp_nemomod_temp_sqlite_db = fp_sqlite_tmp_nemomod_intermediate,
				logger = self.logger
			)

			self._log(f"Successfully initialized SISEPUEDEModels.", type_log = "info")
			if not self.file_struct.allow_electricity_run:
				self._log(f"\tOne or more reference files are missing, and the electricity model cannot be run. This run will not include electricity results. Try locating the missing files and re-initializing SISEPUEDE to run the electricity model.", type_log = "warning")

		except Exception as e:
			self._log(f"Error trying to initialize models: {e}", type_log = "error")
			raise RuntimeError()

		self.dir_jl = dir_jl
		self.dir_nemomod_reference_files = dir_nemomod_reference_files
		self.fp_sqlite_tmp_nemomod_intermediate = fp_sqlite_tmp_nemomod_intermediate



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




	############################
	#    SHORTCUT FUNCTIONS    #
	############################

	def _initialize_function_aliases(self,
	) -> None:
		"""
		Initialize function aliases.
		"""

		self.get_output_region = self.experimental_manager.get_output_region



	############################
	#    CORE FUNCTIONALITY    #
	############################


	# ADD SQL RETURN TYUP
	def generate_scenario_database_from_primary_key(self,
		primary_key: Union[int, None],
		regions: Union[List[str], str, None] = None,
		**kwargs
	) -> Union[Dict[str, pd.DataFrame], None]:
		"""
		Generate an input database for SISEPUEDE based on the primary key.

		Function Arguments
		------------------
		- primary_key: primary key to generate input database for
			* returns None if primary key entered is invalid

		Keyword Arguments
		-----------------
		- regions: list of regions or string of a region to include.
			* If a list of regions or single region is entered, returns a
				dictionary of input databases of the form
				{region: df_input_region, ...}
			* Invalid regions return None
		- **kwargs: passed to SISEPUEDE.models.project(..., **kwargs)
		"""

		# check primary keys to run
		if (primary_key not in self.attribute_primary.key_values):
			self._log(f"Error in generate_scenario_database_from_primary_key: {self.key_primary} = {primary_key} not found.", type_log = "error")
			return None

		# check region
		regions = self.regions if (regions is None) else regions
		regions = [regions] if not isinstance(regions, list) else regions
		regions = [x for x in regions if x in self.regions]
		if len(regions) == 0:
			self._log(f"Error in generate_scenario_database_from_primary_key: no valid regions found in input.", type_log = "error")
			return None

		# get designs
		df_primary_keys = self.attribute_primary.table[
			self.attribute_primary.table[self.key_primary].isin([primary_key])
		]
		all_designs = sorted(list(set(df_primary_keys[self.key_design])))

		# initialize output (TEMPORARY)
		dict_return = {}

		for region in regions:

			# retrieve region specific future trajectories and lhs design
			future_trajectories_cur = self.experimental_manager.dict_future_trajectories.get(region)
			lhs_design_cur = self.experimental_manager.dict_lhs_design.get(region)
			region_out = self.get_output_region(region)


			##  GET DIMENSIONS

			df_primary_keys_cur_design = sf.subset_df(
				df_primary_keys,
				{
					self.key_primary: primary_key
				}
			)
			design = int(df_primary_keys_cur_design[self.key_design].iloc[0])
			future = int(df_primary_keys_cur_design[self.key_future].iloc[0])
			strategy = int(df_primary_keys_cur_design[self.key_strategy].iloc[0])


			##  GET LHS TABLES AND FILTER
			df_lhs_l, df_lhs_x = lhs_design_cur.retrieve_lhs_tables_by_design(design, return_type = pd.DataFrame)

			# reduce lhs tables - LEs
			df_lhs_l = df_lhs_l[
				df_lhs_l[self.key_future].isin([future])
			] if (df_lhs_l is not None) else df_lhs_l
			# Xs
			df_lhs_x = df_lhs_x[
				df_lhs_x[self.key_future].isin([future])
			] if (df_lhs_x is not None) else df_lhs_x


			##  GENERATE INPUT BY FUTURE

			# determine if baseline future and fetch lhs rows
			base_future_q = (future == self.baseline_future)
			lhs_l = df_lhs_l[df_lhs_l[self.key_future] == future].iloc[0] if ((df_lhs_l is not None) and not base_future_q) else None
			lhs_x = df_lhs_x[df_lhs_x[self.key_future] == future].iloc[0] if ((df_lhs_x is not None) and not base_future_q) else None

			# generate the futures and get available strategies
			df_input = future_trajectories_cur.generate_future_from_lhs_vector(
				lhs_x,
				df_row_lhc_sample_l = lhs_l,
				future_id = future,
				baseline_future_q = base_future_q
			)


			##  FILTER BY STRATEGY

			df_input = df_input[
				(df_input[self.key_strategy] == strategy)
			].sort_values(
				by = [self.model_attributes.dim_time_period]
			).drop(
				[x for x in df_input.columns if x in self.keys_index], axis = 1
			).reset_index(
				drop = True
			)


			##  ADD IDS AND RETURN

			sf.add_data_frame_fields_from_dict(
				df_input,
				{
					self.field_region: region_out,
					self.key_primary: primary_key
				},
				prepend_q = True
			)

			dict_return.update({region_out: df_input})

		return dict_return



	# ADD SQL RETURN TYUP
	def project_scenarios(self,
		primary_keys: Union[List[int], Dict[str, int], None],
		return_type: Union[type, str, None] = None,
		**kwargs
	) -> None:
		"""
		Project scenarios forward for a set of primary keys.

		Function Arguments
		------------------
		- primary_keys: list of primary keys to run OR dictionary of index keys (e.g., strategy_id, design_id)
			with scenarios associated as values (uses AND operation to filter scenarios). If None, returns
			all possible primary keys.

		Keyword Arguments
		-----------------
		- return_to: specifies how to return output dataframe
			* If pd.DataFrame, then will return a dataframe
			* If sqlalchemy.engine.Engine, will return to a database FIGURE THIS OUT
		- **kwargs: passed to SISEPUEDE.models.project(..., **kwargs)
		"""

		# check primary keys to run
		if isinstance(primary_keys, dict):
			primary_keys = sorted(list(
					sf.subset_df(
					self.attribute_primary.table,
					primary_keys
				)[self.attribute_primary.key]
			))
		elif isinstance(primary_keys, list):
			primary_keys = sorted([x for x in primary_keys if x in self.attribute_primary.key_values])
		elif primary_keys is None:
			primary_keys = self.attribute_primary.key_values

		# get designs
		df_primary_keys = self.attribute_primary.table[
			self.attribute_primary.table[self.key_primary].isin(primary_keys)
		]
		all_designs = sorted(list(set(df_primary_keys[self.key_design])))

		# initialize output (TEMPORARY)
		df_out = []


		for region in self.regions:

			# retrieve region specific future trajectories and lhs design
			future_trajectories_cur = self.experimental_manager.dict_future_trajectories.get(region)
			lhs_design_cur = self.experimental_manager.dict_lhs_design.get(region)
			region_out = self.get_output_region(region)

			for design in all_designs:

				df_lhs_l, df_lhs_x = lhs_design_cur.retrieve_lhs_tables_by_design(design, return_type = pd.DataFrame)

				# get reduced set of primary keys
				df_primary_keys_cur_design = df_primary_keys[
					df_primary_keys[self.key_design] == design
				]
				keep_futures = sorted(list(set(df_primary_keys_cur_design[self.key_future])))

				# reduce lhs tables - LEs
				df_lhs_l = df_lhs_l[
					df_lhs_l[self.key_future].isin(keep_futures)
				] if (df_lhs_l is not None) else df_lhs_l
				# Xs
				df_lhs_x = df_lhs_x[
					df_lhs_x[self.key_future].isin(keep_futures)
				] if (df_lhs_x is not None) else df_lhs_x


				# next, loop over futures
				#  Note that self.generate_future_from_lhs_vector() will return a table for all strategies
				#  associated with the future, so we can prevent redundant calls by running all strategies
				#  that need to be run for a given future

				for future in keep_futures:

					# determine if baseline future and fetch lhs rows
					base_future_q = (future == self.baseline_future)
					lhs_l = df_lhs_l[df_lhs_l[self.key_future] == future].iloc[0] if ((df_lhs_l is not None) and not base_future_q) else None
					lhs_x = df_lhs_x[df_lhs_x[self.key_future] == future].iloc[0] if ((df_lhs_x is not None) and not base_future_q) else None


					# generate the futures and get available strategies
					df_input = future_trajectories_cur.generate_future_from_lhs_vector(
						lhs_x,
						df_row_lhc_sample_l = lhs_l,
						future_id = future,
						baseline_future_q = base_future_q
					)
					all_strategies = sorted(list(
						set(df_input[self.key_strategy])
					))

					for strategy in all_strategies:

						# get primary id
						id_primary = df_primary_keys_cur_design[
							(df_primary_keys_cur_design[self.key_future] == future) &
							(df_primary_keys_cur_design[self.key_strategy] == strategy)
						][self.key_primary]

						id_primary = int(id_primary.iloc[0]) if (len(id_primary) > 0) else None

						if id_primary in primary_keys:

							# filter the data frame down
							df_input_cur = df_input[
								(df_input[self.key_strategy] == strategy)
							].copy().reset_index(
								drop = True
							).sort_values(
								by = [self.model_attributes.dim_time_period]
							).drop(
								[x for x in df_input.columns if x in self.keys_index], axis = 1
							)

							# try to run the model
							try:
								t0 = time.time()
								df_output = self.models.project(df_input_cur, **kwargs)
								df_output = sf.add_data_frame_fields_from_dict(
									df_output,
									{
										self.field_region: region_out,
										self.key_primary: id_primary
									},
									prepend_q = True
								)
								df_out.append(df_output)
								t_elapse = sf.get_time_elapsed(t0)

								self._log(f"Model run for {self.key_primary} = {id_primary} successfully completed in {t_elapse} seconds.", type_log = "info")

							except Exception as e:

								self._log(f"Model run for {self.key_primary} = {id_primary} failed with the following error: {e}", type_log = "error")

		df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)

		return df_out
