from attribute_table import *
import itertools
import logging
from model_attributes import ModelAttributes
import numpy as np
import os, os.path
import pandas as pd
from sisepuede_experimental_manager import *
from sisepuede_file_structure import *
from sisepuede_models import *
#from sisepuede_output_database import *
import sisepuede_output_database as sod
import support_functions as sf
import tempfile
import time
from typing import *






class SISEPUEDE:

	"""
	SISEPUEDE (SImulation of SEctoral Pathways and Uncertainty Exploration for
		DEcarbonization) is an integrated modeling framework (IMF) used to
		assess decarbonization pathways under deep uncertainty. SISEPUEDE
		estimates GHG emissions primarily using the IPCC Guidelines for
		Greenhouse Gas Inventories (2006 and 2019R) and further includes costs
		and benefits of transformation-level strategies across 4 emission
		sectors and 16 emission subsectors.

		The SISEPUEDE IMF includes the following components:

		* Integrated GHG Inventory Model (SISEPUEDEModels)
		* Economic assessment of technical costs and co-benefits
		* Uncertainty tools (SISEPUEDEExperimentalManager)
		* Flexible database management (SISEPUEDEOutputDatabase)
		* Automated data population using open access data sources
		10-20 pre-defined transformations per sector + cross sectoral strategies

	More on SISPUEDE, including model documentation, a description of sectors,
		and a quick start guide, can be found at the SISEPUEDE documentation,
		located at

		https://sisepuede.readthedocs.io



	Initialization Arguments
	------------------------
	- data_mode: template class to initialize from. Three options are allowed:
		* calibrated
		* demo
		* uncalibrated


	Optional Arguments
	------------------
	Optional arguments are used to pass values to SISEPUEDE outside of the
		configuaration framework. This can be a desireable approach for
		iterative modeling or determining a suitable data pipeline.

	- attribute_design: optional AttributeTable object used to specify the
		design_id.
		* Note: If None, will attempt to find a table within the ModelAttributes
			object with key "dim_design_id". If none is found, will assume a
			single design where:

			(a) exogenous uncertainties vary at ranges specified in input
				templates; and
			(b) lever effects are fixed.

	- dir_ingestion: directory storing SISEPUEDE templates for ingestion
		* Note: if running outside of demo mode, the directory should contain
			subdirectories dor each region, with each region including input
			templates for each of the 5 SISEPUEDE sectors. For example, the
			directory should have the following tree structure:

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
	- id_str: Optional id_str used to create AnalysisID (see ?AnalysisID for
		more information on properties). Can be used to set outputs for a
		previous ID/restore a session.
		* If None, creates a unique ID for the session (used in output file
			names)
	- logger: Optional logging.Logger object to use for logging
	- regions: list of regions to include in the experiment. [NOTE, in v1.0,
		this should be limited to a single region]
	- replace_output_dbs_on_init: default is set to false; if True, will
		destroy exisiting output tables if an AnalysisID is specified.
	- regex_template_prepend: string to prepend to output files tagged with the
		analysis id
	"""

	def __init__(self,
		data_mode: str,
		attribute_design: Union[AttributeTable, None] = None,
		dir_ingestion: Union[str, None] = None,
		id_str: Union[str, None] = None,
		logger: Union[logging.Logger, None] = None,
		regions: Union[List[str], None] = None,
		regex_template_prepend: str = "sisepuede_run",
		replace_output_dbs_on_init: bool = False
	):

		self.logger = logger

		self._initialize_file_structure(
			dir_ingestion = dir_ingestion,
			id_str = id_str,
			regex_template_prepend = regex_template_prepend
		)

		self._initialize_attribute_design(attribute_design)
		self._initialize_keys()
		self._initialize_output_database(replace_output_dbs_on_init = replace_output_dbs_on_init)
		self._initialize_data_mode(data_mode)
		self._initialize_experimental_manager(regions = regions)
		self._initialize_models()
		self._initialize_function_aliases()
		self._initialize_base_database_tables()


	##################################
	#    INITIALIZATION FUNCTIONS    #
	##################################

	def _initialize_attribute_design(self,
		attribute_design: Union[AttributeTable, None] = None,
		key_model_attributes_design: str = "dim_design_id"
	) -> None:
		"""
		Initialize and check the attribute design table. Sets the following
			properties:

			* self.attribute_design


		Keyword Arguments
		-----------------
		- attribute_design: AttributeTable used to specify designs.
			* If None, tries to access "dim_design_id" from
				ModelAttributes.dict_attributes
		- key_model_attributes_design: key in model_attributes.dict_attributes
			used to try and get design attribute.
			* If None, defaults to "dim_design_id"
		"""

		# initialize the attribute design table -- checks on the table are run when experimental manager is initialized
		self.attribute_design = self.model_attributes.dict_attributes.get(key_model_attributes_design) if not isinstance(attribute_design, AttributeTable) else attribute_design



	def get_config_parameter(self,
		parameter: str
	) -> Union[int, List, str]:
		"""
		Retrieve a configuration parameter from self.model_attributes. Must be initialized
			after _initialize_file_structure()
		"""

		return self.model_attributes.configuration.get(parameter)



	def _initialize_base_database_tables(self,
	) -> None:
		"""
		Initialize database tables that characterize the analytical
			configuration. Initializes the following tables:

			* self.database.table_name_analysis_metadata
			* self.database.table_name_attribute_design
			* self.database.table_name_attribute_lhs_l
			* self.database.table_name_attribute_lhs_x
			* self.database.table_name_attribute_strategy
			* self.database.table_name_base_input
		"""

		if not self.from_existing_analysis_id:

			# get some tables
			df_analysis_metadata = self.model_attributes.configuration.to_data_frame()
			df_attribute_design = self.attribute_design.table
			df_lhs_l, df_lhs_x = self.build_lhs_tables()
			df_attribute_strategy = self.attribute_strategy.table
			df_base_input = self.experimental_manager.base_input_database.database


			##  WRITE TABLES TO OUTPUT DATABASE

			self.database._write_to_table(
				self.database.table_name_analysis_metadata,
				df_analysis_metadata
			)

			self.database._write_to_table(
				self.database.table_name_attribute_design,
				df_attribute_design
			)

			self.database._write_to_table(
				self.database.table_name_attribute_lhs_l,
				df_lhs_l
			) if (df_lhs_l is not None) else None

			self.database._write_to_table(
				self.database.table_name_attribute_lhs_x,
				df_lhs_x
			) if (df_lhs_x is not None) else None

			self.database._write_to_table(
				self.database.table_name_attribute_strategy,
				df_attribute_strategy
			)

			self.database._write_to_table(
				self.database.table_name_base_input,
				df_base_input
			)



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
		key_config_n_lhs: str = "num_lhc_samples",
		key_config_random_seed: str = "random_seed",
		key_config_time_period_u0: str = "time_period_u0",
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

			* self.baseline_future
			* self.baseline_strategy
			* self.experimental_manager
			* self.n_trials
			* self.odpt_primary
			* self.random_seed
			* self.regions
			* self.time_period_u0


		Keyword Arguments
		-----------------
		- key_config_n_lhs: configuration key used to determine the number of LHC samples to
			generate
		- key_config_random_seed: configuration key used to set the random seed
		- key_config_time_period_u0: configuration key used to determine the time period of
			initial uncertainty in uncertainty assessment.
		- num_trials: number if LHS trials to run.
			* If None, revert to configuration defaults from self.model_attributes
		- random_seed: random seed used to generate LHS samples
			* If None, revert to configuration defaults from self.model_attributes
		- regions: regions to initialize.
			* If None, initialize using all regions
		- time_t0_uncertainty: time where uncertainty starts
		"""

		num_trials = int(max(num_trials, 0)) if (isinstance(num_trials, int) or isinstance(num_trials, float)) else self.get_config_parameter(key_config_n_lhs)

		self.experimental_manager = None
		self.n_trials = self.get_config_parameter(key_config_n_lhs)
		self.time_period_u0 = self.get_config_parameter(key_config_time_period_u0)
		self.random_seed = self.get_config_parameter(key_config_time_period_u0)

		try:
			self.experimental_manager = SISEPUEDEExperimentalManager(
				self.attribute_design,
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


		self.attribute_strategy = self.experimental_manager.attribute_strategy
		self.odpt_primary = self.experimental_manager.primary_key_database
		self.baseline_future = self.experimental_manager.baseline_future
		self.baseline_strategy = self.experimental_manager.baseline_strategy
		self.n_trials = self.experimental_manager.n_trials
		self.random_seed = self.experimental_manager.random_seed
		self.regions = self.experimental_manager.regions
		self.time_period_u0 = self.experimental_manager.time_period_u0



	def _initialize_file_structure(self,
		dir_ingestion: Union[str, None] = None,
		id_str: Union[str, None] = None,
		regex_template_prepend: str = "sisepuede_run"
	) -> None:

		"""
		Intialize the SISEPUEDEFileStructure object and model_attributes object.
			Initializes the following properties:

			* self.analysis_id
			* self.file_struct
			* self.fp_base_output_raw
			* self.id
			* self.id_fs_safe
			* self.model_attributes

		Optional Arguments
		------------------
		- dir_ingestion: directory containing templates for ingestion. The
			ingestion directory should include subdirectories for each template
			class that may be run, including:
				* calibrated: input variables that are calibrated for each
					region and sector
				* demo: demo parameters that are independent of region (default
					in quick start)
				* uncalibrated: preliminary input variables defined for each
					region that have not yet been calibrated
			The calibrated and uncalibrated subdirectories require separate
				subdrectories for each region, each of which contains an input
				template for each
		- id_str: Optional id_str used to create AnalysisID (see ?AnalysisID
			for more information on properties). Can be used to set outputs for
			a previous ID/restore a session.
			* If None, creates a unique ID for the session (used in output file
				names)
		"""

		self.file_struct = None
		self.model_attributes = None

		try:
			self.file_struct = SISEPUEDEFileStructure(
				dir_ingestion = dir_ingestion,
				id_str = id_str,
				regex_template_prepend = regex_template_prepend
			)

			self._log(f"Successfully initialized SISEPUEDEFileStructure.", type_log = "info")

		except Exception as e:
			self._log(f"Error trying to initialize SISEPUEDEFileStructure: {e}", type_log = "error")
			raise RuntimeError()

		self.analysis_id = self.file_struct.analysis_id
		self.fp_base_output_raw = self.file_struct.fp_base_output_raw
		self.from_existing_analysis_id = self.file_struct.from_existing_analysis_id
		self.id = self.file_struct.id
		self.id_fs_safe = self.file_struct.id_fs_safe
		self.model_attributes = self.file_struct.model_attributes



	def _initialize_keys(self,
		attribute_design: Union[AttributeTable, None] = None
	) -> None:
		"""
		Initialize scenario dimension keys that are shared for initialization.
			Initializes the followin properties:

			* self.key_design
			* self.key_future
			* self.key_primary
			* self.key_region
			* self.key_strategy
			* self.key_time_period
			* self.keys_index

		NOTE: these keys are initialized separately within
			SISEPUEDEExperimentalManager, but they depend on the same shared
			sources (attribute_design and self.model_attributes).
		"""

		# set keys
		self.key_design = self.attribute_design.key
		self.key_future = self.model_attributes.dim_future_id
		self.key_primary = self.model_attributes.dim_primary_id
		self.key_region = self.model_attributes.dim_region
		self.key_strategy = self.model_attributes.dim_strategy_id
		self.key_time_period = self.model_attributes.dim_time_period

		self.keys_index = [
			self.key_design,
			self.key_future,
			self.key_primary,
			self.key_region,
			self.key_strategy
		]


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



	def _initialize_output_database(self,
		config_key_output_method: str = "output_method",
		default_db_type: str = "sqlite",
		replace_output_dbs_on_init: bool = False
	) -> None:
		"""
		Initialize the SISEPUEDEOutputDatabase structure. Allows for quick
			reading and writing of data files. Sets the following properties:

			* self.database


		Keyword Arguments
		-----------------
		- config_key_output_method: configuration key to use to determine the
			method for the output database.
		- default_db_type: default type of output database to use if invalid
			entry found from config.
		- replace_output_dbs_on_init: replace output database tables on
			initialization if they exist? Only applies if loading from an
			existing dataset.
		"""
		# try getting the configuration parameter
		db_type = self.get_config_parameter(config_key_output_method)
		db_type = default_db_type if (db_type is None) else db_type
		self.database = None

		try:
			self.database = sod.SISEPUEDEOutputDatabase(
				db_type,
				{
					"design": self.key_design,
					"future": self.key_future,
					"primary": self.key_primary,
					"region": self.key_region,
					"strategy": self.key_strategy,
					"time_series": None
				},
				analysis_id = self.analysis_id,
				fp_base_output = self.fp_base_output_raw,
				create_dir_output = True,
				logger = self.logger,
				replace_on_init = False,

			)

		except Exception as e:
			msg = f"Error initializing SISEPUEDEOutputDatabase: {e}"
			self._log(msg, type_log = "error")


		if self.database is None:
			return None

		# log if successful
		self._log(
			f"Successfully initialized database with:\n\ttype:\t{db_type}\n\tanalysis id:\t{self.id}\n\tfp_base_output:\t{self.fp_base_output_raw}",
			type_log = "info"
		)


		##  COMPLETE SOME ADDITIONAL INITIALIZATIONS

		# remove the output database if specified
		if replace_output_dbs_on_init:
			tables_destroy = [
				self.database.table_name_output
			]

			for table in tables_destroy:
				self._destroy_table(table)



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

	def _destroy_table(self,
		table_name: Union[str, None]
	) -> None:
		"""
		Destroy a table (delete rows and reset columns) without removing from
			the database.
		"""
		if table_name is None:
			return None

		self.database.db._destroy(table_name)



	def get_primary_keys(self,
		primary_keys: Union[List[int], Dict[str, int], None]
	) -> List[int]:
		"""
		Based on list of primary keys or subsetting dictioary, get a list of
			primary keys. Used to support filtering in a number of contexts.


		Function Arguments
		------------------
		- primary_keys: list of primary keys to run OR dictionary of index keys
			(e.g., strategy_id, design_id) with scenarios associated as values
			(uses AND operation to filter scenarios). If None, returns all
			possible primary keys.
		"""

		if isinstance(primary_keys, dict):
			primary_keys = sorted(list(
				self.odpt_primary.get_indexing_dataframe(
					key_values = primary_keys,
					keys_return = [self.odpt_primary.key_primary]
				)[self.odpt_primary.key_primary]
			))
		elif isinstance(primary_keys, list):
			primary_keys = sorted([x for x in primary_keys if x in self.odpt_primary.range_key_primary])
		elif primary_keys is None:
			primary_keys = self.odpt_primary.range_key_primary

		return primary_keys



	def _initialize_function_aliases(self,
	) -> None:
		"""
		Initialize function aliases.
		"""

		self.get_output_region = self.experimental_manager.get_output_region



	def read_output(self,
		primary_keys: Union[List[int], Dict[str, int], None],
		**kwargs
	) -> pd.DataFrame:
		"""
		Read output data generated after running .project_scenarios.

		Function Arguments
		------------------
		- primary_keys: list of primary keys to run OR dictionary of index keys
			(e.g., strategy_id, design_id) with scenarios associated as values
			(uses AND operation to filter scenarios). If None, returns all
			possible primary keys.

		Optional Arguments
		------------------
		- dict_subset: dictionary with keys that are columns in the table and
			values, given as a list, to subset the table. dict_subset is written
			as:

			dict_subset = {
				field_a = [val_a1, val_a2, ..., val_am],
				field_b = [val_b1, val_b2, ..., val_bn],
				.
				.
				.
			}

			NOTE: dict_subset should NOT contain self.key_primary (it will be
			removed if passed in dict_subset) since these are passed in the
			`primary_keys` argument
		- fields_select: fields to read in. Reducing the number of fields to read
			can speed up the ingestion process and reduce the data frame's memory
			footprint.

		Keyword Arguments
		-----------------
		- drop_duplicates: drop duplicates in a CSV when reading? (only applies
			if the database is initialized using CSVs)
			* Default is False to improve speeds
			* Set to True to ensure that only unique rows are read in
		- query_logic: default is "and". Subsets table to as

			where field_a in (val_a1, val_a2, ..., val_am) ~ field_b in (val_b1, val_b2, ..., val_bn)...

			where `~ in ["and", "or"]`
		"""

		# get primary keys and initialize subset
		primary_keys = self.get_primary_keys(primary_keys)
		dict_subset = {
			self.key_primary: primary_keys
		} if not isinstance(primary_keys, range) else {}

		# check for additional arguments passed and remove the subset dictionary if it is passed
		dict_subset_kwargs = kwargs.get("dict_subset")
		if isinstance(dict_subset_kwargs, dict):
			dict_subset.update(
				dict(
					(k, v) for k, v in dict_subset_kwargs.items() if k not in dict_subset.keys()
				)
			)
		if dict_subset_kwargs is not None:
			del kwargs["dict_subset"]

		df_out = self.database.read_table(
			self.database.table_name_output,
			dict_subset = dict_subset,
			**kwargs
		)

		return df_out



	def _write_chunk_to_table(self,
		df_list: List[pd.DataFrame],
		check_duplicates: bool = False,
		table_name: Union[str, None] = None,
		**kwargs
	) -> pd.DataFrame:
		"""
		Write a chunk of data frames to output database.

		Function Arguments
		------------------
		- df_list: list of data frames to write

		Keyword Arguments
		-----------------
		= check_duplicates: check for duplicate rows?
		- table_name: table name to write to. Default is
			self.database.table_name_output
		- **kwargs: passed to IterativeDatabaseTable._write_to_table
		"""

		table_name = self.database.table_name_output if (table_name is None) else table_name

		df_out = pd.concat(df_list, axis = 0).reset_index(drop = True)
		df_out.drop_duplicates(inplace = True) if check_duplicates else None

		self.database._write_to_table(
			table_name,
			df_out,
			**kwargs
		)
		df_out = []

		return df_out




	#########################
	#    TABLE FUNCTIONS    #
	#########################

	def build_lhs_tables(self,
	) -> pd.DataFrame:
		"""
		Build LHS tables for export to database. Returns a tuple

			df_l, df_x

			where `df_l` is the database of lever effect LHC samples and `df_x`
			is the database of exogenous uncertainty LHC samples. Both are long
			by region and LHS key.
		"""

		# initialize output
		df_l = []
		df_x = []

		for region in self.regions:

			lhsd = self.experimental_manager.dict_lhs_design.get(region)
			df_lhs_l, df_lhs_x = lhsd.retrieve_lhs_tables_by_design(None, return_type = pd.DataFrame)
			region_out = self.get_output_region(region)

			# lever effect LHS table
			if (df_lhs_l is not None):
				df_lhs_l = sf.add_data_frame_fields_from_dict(
					df_lhs_l,
					{
						self.key_region: region_out
					}
				)
				df_l.append(df_lhs_l)

			# exogenous uncertainty LHS table
			if (df_lhs_x is not None):
				df_lhs_x = sf.add_data_frame_fields_from_dict(
					df_lhs_x,
					{
						self.key_region: region_out
					}
				)
				df_x.append(df_lhs_x)

		df_l = pd.concat(df_l, axis = 0).reset_index(drop = True) if (len(df_l) > 0) else None
		if (df_l is not None):
			df_l.columns = [str(x) for x in df_l.columns]
			fields_ord_l = [self.key_region, lhsd.field_lhs_key]
			fields_ord_l += sf.sort_integer_strings([x for x in df_l.columns if x not in fields_ord_l])
			df_l = df_l[fields_ord_l]

		df_x = pd.concat(df_x, axis = 0).reset_index(drop = True) if (len(df_x) > 0) else None
		if df_x is not None:
			df_x.columns = [str(x) for x in df_x.columns]
			fields_ord_x = [self.key_region, lhsd.field_lhs_key]
			fields_ord_x += sf.sort_integer_strings([x for x in df_x.columns if x not in fields_ord_x])
			df_x = df_x[fields_ord_x]

		return df_l, df_x






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
		if (primary_key not in self.odpt_primary.range_key_primary):
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
		dict_primary_keys = self.odpt_primary.get_dims_from_key(
			primary_key,
			return_type = "dict"
		)

		# initialize output (TEMPORARY)
		dict_return = {}

		for region in regions:

			# retrieve region specific future trajectories and lhs design
			future_trajectories_cur = self.experimental_manager.dict_future_trajectories.get(region)
			lhs_design_cur = self.experimental_manager.dict_lhs_design.get(region)
			region_out = self.get_output_region(region)


			##  GET DIMENSIONS

			design = dict_primary_keys.get(self.key_design) # int(df_primary_keys_cur_design[self.key_design].iloc[0])
			future = dict_primary_keys.get(self.key_future) # int(df_primary_keys_cur_design[self.key_future].iloc[0])
			strategy = dict_primary_keys.get(self.key_strategy) # int(df_primary_keys_cur_design[self.key_strategy].iloc[0])


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
					self.key_region: region_out,
					self.key_primary: primary_key
				},
				prepend_q = True
			)

			dict_return.update({region_out: df_input})

		return dict_return



	def project_scenarios(self,
		primary_keys: Union[List[int], Dict[str, int], None],
		chunk_size: int = 10,
		force_overwrite_existing_primary_keys: bool = False,
		**kwargs
	) -> List[int]:
		"""
		Project scenarios forward for a set of primary keys. Returns the set of
			primary keys that ran successfully.

		Function Arguments
		------------------
		- primary_keys: list of primary keys to run OR dictionary of index keys (e.g., strategy_id, design_id)
			with scenarios associated as values (uses AND operation to filter scenarios). If None, returns
			all possible primary keys.

		Keyword Arguments
		-----------------
		- chunk_size: size of chunk to use to write to IterativeDatabaseTable.
			If 1, updates table after every iteration; otherwise, stores chunks
			in memory, aggregates, then writes to IterativeDatabaseTable.
		- force_overwrite_existing_primary_keys: if the primary key is already found
			in the output database table, should it be overwritten? Default is
			False. It is recommended that iterations on the same scenarios be
			undertaken using different AnalysisID structures. Otherwise, defaults
			to initialization resolutsion (write_skip)
		- **kwargs: passed to SISEPUEDE.models.project(..., **kwargs)
		"""

		primary_keys = self.get_primary_keys(primary_keys)

		# get designs
		df_primary_keys = self.odpt_primary.get_indexing_dataframe(
			key_values = primary_keys
		)
		all_designs = sorted(list(set(df_primary_keys[self.key_design])))

		# initializations
		df_out = []
		df_out_primary = []
		dict_primary_keys_run = dict((x, [None for x in primary_keys]) for x in self.regions)
		iterate_outer = 0

		# available indices and resolution
		idt = self.database.db.dict_iterative_database_tables.get(
			self.database.table_name_output
		)
		index_conflict_resolution = None
		index_conflict_resolution = "write_replace" if (force_overwrite_existing_primary_keys or (idt.index_conflict_resolution == "write_replace")) else None
		set_available_ids = idt.available_indices

		for region in self.regions:

			iterate_inner = 0

			# retrieve region specific future trajectories and lhs design
			future_trajectories_cur = self.experimental_manager.dict_future_trajectories.get(region)
			lhs_design_cur = self.experimental_manager.dict_lhs_design.get(region)
			region_out = self.get_output_region(region)

			for design in all_designs:

				df_lhs_l, df_lhs_x = lhs_design_cur.retrieve_lhs_tables_by_design(
					design,
					return_type = pd.DataFrame
				)

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

						# get primary id info
						df_primary_keys_cur_design_fs = df_primary_keys_cur_design[
							(df_primary_keys_cur_design[self.key_future] == future) &
							(df_primary_keys_cur_design[self.key_strategy] == strategy)
						].reset_index(drop = True)

						id_primary = df_primary_keys_cur_design_fs[self.key_primary]
						id_primary = int(id_primary.iloc[0]) if (len(id_primary) > 0) else None
						write_q = ((region_out, id_primary) not in set_available_ids) or (index_conflict_resolution == "write_replace")
						tup = (region_out, id_primary)

						if (id_primary in primary_keys) and write_q:

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

							success = False

							# try to run the model
							try:
								t0 = time.time()
								df_output = self.models.project(df_input_cur, **kwargs)
								df_output = sf.add_data_frame_fields_from_dict(
									df_output,
									{
										self.key_region: region_out,
										self.key_primary: id_primary
									},
									prepend_q = True
								)
								df_out.append(df_output)
								t_elapse = sf.get_time_elapsed(t0)
								success = True

								self._log(f"Model run for {self.key_primary} = {id_primary} successfully completed in {t_elapse} seconds.", type_log = "info")

							except Exception as e:

								self._log(f"Model run for {self.key_primary} = {id_primary} failed with the following error: {e}", type_log = "error")


							# if the model run is successful and the chunk size is appropriate, update primary keys that ran successfully and write to output
							if success:

								df_out_primary.append(df_primary_keys_cur_design_fs)

								if (len(df_out)%chunk_size == 0) and (len(df_out) > 0):
									df_out = self._write_chunk_to_table(
										df_out,
										table_name = self.database.table_name_output,
										index_conflict_resolution = index_conflict_resolution
									)

								if (len(df_out_primary)%chunk_size == 0) and (len(df_out_primary) > 0):
									df_out_primary = self._write_chunk_to_table(
										df_out_primary,
										check_duplicates = True,
										table_name = self.database.table_name_attribute_primary,
										index_conflict_resolution = index_conflict_resolution
									)

								# append to output
								df_out_primary.append(df_primary_keys_cur_design_fs)

								dict_primary_keys_run[region][iterate_inner] = id_primary

								iterate_inner += 1 # number of iterations for this region
								iterate_outer += 1 # number of total iterations

			# reduce length after running
			dict_primary_keys_run[region] = dict_primary_keys_run[region][0:iterate_inner]

		# write tables to output
		self._write_chunk_to_table(
			df_out,
			table_name = self.database.table_name_output,
			index_conflict_resolution = index_conflict_resolution
		) if (len(df_out) > 0) else None

		df_out_primary = self._write_chunk_to_table(
			df_out_primary,
			check_duplicates = True,
			table_name = self.database.table_name_attribute_primary,
			index_conflict_resolution = index_conflict_resolution
		) if (len(df_out_primary) > 0) else None

		return dict_primary_keys_run
