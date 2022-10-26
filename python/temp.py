
class SISEPUEDEOutputDatabase:
	"""
	Manage output from SISEPUEDE in a cohesive and flexible database structure,
		including output to SQLite, CSVs, or remote SQL databases. The output
		database includes a number of tables and allows for the specification
		of optional post-processing functions and additional tables.

	The following is a list of table name keyword arguments for all default
		tables generated ny SISEPUEDE, along with the table's default name and
		a description of the table.

		* table_name_analysis_metadata -> "ANALYSIS_METADATA"
			* The analysis metadata table stores information associated with
				each unique SISEPUEDE run, including configuration options
				(both analytical and experimental parameters), time of run,
				analysis id, and more.

		* table_name_attribute_design -> "ATTRIBUTE_DESIGN"
			* The design attribute table stores the attribute_design table
				associated with the run.

		* table_name_attribute_lhs_l -> "ATTRIBUTE_LHC_SAMPLES_LEVER_EFFECTS"
			* The lever effect latin hypercube sample attribute table stores the
				latin hypercube samples associated with plausible future lever
				effects.

		* table_name_attribute_lhs_x -> "ATTRIBUTE_LHC_SAMPLES_EXOGENOUS_UNCERTAINTIES"
			* The exogenous uncertainties latin hypercube sample attribute table
				stores the latin hypercube samples associated with plausible
				future uncertainties.

		* table_name_attribute_primary -> "ATTRIBUTE_PRIMARY"
			* The primary key attribute table stores the attribute_primary table
				associated with the run.

		* table_name_attribute_strategy -> "ATTRIBUTE_STRATEGY"
			* The strategy attribute table stores the attribute_strategy table
				associated with the run, which governs information about
				strategies (across all sectors).

		* table_name_base_input -> "MODEL_BASE_INPUT_DATABASE"
			* The base input database is derived from input templates and is used
				as the basis for generating all future trajectories. It is stored
				in the SISEPUEDE class as `SISEPUEDE.base_input_datase.database`.

		* table_name_input -> "MODEL_INPUT"
			* The model input database is the entire range of futures (indexed by
				primary key). To avoid unecessary storage, this database, by default,
				is *not* written to a table. Note that individual futures (and the
				table itself) can be reproduced quickly using SISEPUEDE's internal
				functions in combination with LHS tables, which are saved by default.

			* Derivative tables that summarize inputs can be passed using
				the `dict_derivative_table_functions` initialization argument (see
				below).

			* NOTE: To save this table, ensure that "input" is *not* contained in
				the `tables_write_exclude` initialization argument list.

		* table_name_output -> "MODEL_OUTPUT"
			* The model output database includes all outputs across the entire
				range of futures (indexed by primary key). While large, this database
				is included in default outputs to save compute power/time.

			* Derivative tables that summarize outputs by primary key can be passed
				using the `dict_derivative_table_functions` initialization argument
				(see below).

			* NOTE: To *not* save this table, ensure that "output" is contained in
				the `tables_write_exclude` initialization argument list.



	Initialization Arguments
	------------------------
	- export_engine: string specifying output method, or, optionally, sqlalchemy
		engine connected to a database/schema used to output data. Options for
		export are:
		* string:
			* "csv": exports CSVs to subdirectory (associated with analysis run id)
				located in output directory
			* "sqlite": exports all tables to a SQL lite database located in the
				output directory
		* sqlalchemy.engine.Engine: sqlalchemy engine that specifies a database
			and schema to write output tables to. This engine can be used to write
			to a remote database service.
		* None:
			If `None`, defaults to SQLite in SISEPUEDE output subdirectory


	Optional Arguments
	------------------
	- analysis_run_id: optional specification of a SISEPUEDE analysis run id. Can be
		enetered in any of the following forms:
		* SISPUEDEAnalysisID: pass a SISPUEDEAnalysisID object to use
		* str: pass a string of a SISPUEDEAnalysisID; this will initialize a
			new SISPUEDEAnalysisID object within the database structure, but
			allows for connections with databases associated with the specified
			SISEPUEDEAnalysisID
		* None: initialize a new SISPUEDEAnalysisID for the database

	- dict_derivative_table_functions: optional dictionary used to specify additional
		tables (not a standard output). The dictionary maps new table names to a
		tuple, where the first element of the tuple, source_table, represents a
		source table used to calculate the derivative table and FUNCTION_APPLY_i gives
		a function that is applied to the source table (source_table) to develop the
		derivative table TABLE_NAME_i (dictionary key). The dictionary, therefore,
		should have the following form:

		dict_derivative_table_functions = {
			"TABLE_NAME_1": (source_table, FUNCTION_APPLY_1),
			.
			.
			.
			"TABLE_NAME_N": (source_table, FUNCTION_APPLY_N)
		}

		The functions specified in dict_derivative_table_functions can only be applied
			to SISEPUEDE inputs and outputs, i.e., specification of source_table can
			take the following values:
			* "input"
			* "output"

		The function to apply, FUNCTION_APPLY_i, requires the following positional
			arguments:

			(1) model_attributes:ModelAttributes
			(2) df_source:pd.DataFrame,

		Each function should return a data frame. In docstring form, the function
		would have the following form:

			FUNCTION_APPLY_i(
				model_attributes:ModelAttributes,
				df_source:pd.DataFrame
			) -> pd.DataFrame

		If dict_derivative_table_functions is None, or if the dictionary is empty, then
		no derivative tables are generated or written.
	- fp_base_output: output file path to write output to *excluding the file extension*.
		* If export_engine is an instance of sqlalchemy.engine.Engine, then
			fp_base_output is unused
		* If export_engine == "csv", then the tables are saved under the directory
			fp_base_output; e.g.,

			fp_base_output
			|_ table_1.csv
			|_ table_2.csv
			.
			.
			.
			|_ table_n.csv

		* if export_engine == "sqlite", then the tables are saved to an sqlite database at
			f"{fp_base_output}.sqlite"
		 * If None, defaults to paths in SISEPUEDE.file_struct

	- logger: optional log object to pass



	Keyword Arguments
	-----------------

	- create_dir_output: Create output directory implied by fp_base_output if it
		does not exist
	- tables_write_exclude: list of tables to exclude from writing to output. Default
		is ["inputs"] (unless extensive storage is available, writing raw inputs is not
		recommended.)


	The following are table names for output tables, which can be changed using keyword
		arguments. See the descriptions above of each for default names of tables. Note
		that the following lists include the "names" to use in `tables_write_exclude` to
		exclude that table from output.

	- table_name_analysis_metadata: table name to use for storing analysis metadata
		* To exclude from the output database, include "analysis_metadat" in
			`tables_write_exclude`

	- table_name_attribute_design: table name to use for storing the attribute table
		for the design key
		* To exclude from the output database, include "attribute_design" in
			`tables_write_exclude`

	- table_name_attribute_lhs_l: table name to use for storing the attribute table
		for lever effect Latin Hypercube samples
		* To exclude from the output database, include "attribute_lhs_l" in
			`tables_write_exclude`

	- table_name_attribute_lhs_x: table name to use for storing the attribute table
		for exogenous uncertainty Latin Hypercube samples
		* To exclude from the output database, include "attribute_lhs_x" in
			`tables_write_exclude`

	- table_name_attribute_primary: table name to use for storing the attribute table
		for the primary key
		* To exclude from the output database, include "attribute_primary" in
			`tables_write_exclude`

	- table_name_attribute_strategy: table name to use for storing the attribute table
		for the strategy key
		* To exclude from the output database, include "attribute_strategy" in
			`tables_write_exclude`

	- table_name_base_input: table name to use for storing the base input database
		used to input variables
		* To exclude from the output database, include "base_input" in
			`tables_write_exclude`

	- table_name_input: table name to use for storing the complete database of
		SISEPUEDE model inputs
		* To exclude from the output database, include "input" in
			`tables_write_exclude`

	- table_name_output: table name to use for storing the complete database of
		SISEPUEDE model outputs
		* To exclude from the output database, include "output" in
			`tables_write_exclude`


	"""
	def __init__(self,
		export_engine: Union[sqlalchemy.engine.Engine, str, None],
		analysis_run_id: Union[ssp.SISEPUEDEAnalysisID, str, None] = None, #HEREHERE delete ssp.
		fp_base_output: Union[str, None] = None,
		create_dir_output: bool = False,
		logger: Union[logging.Logger, None] = None,
		dict_optional_functions: Union[Dict[str, Tuple[str, Callable[[ModelAttributes, pd.DataFrame], pd.DataFrame]]], None] = None,
		table_name_analysis_metadata: str = "ANALYSIS_METADATA",
		table_name_attribute_design: str = "ATTRIBUTE_DESIGN",
		table_name_attribute_lhs_l: str = "ATTRIBUTE_LHC_SAMPLES_LEVER_EFFECTS",
		table_name_attribute_lhs_x: str = "ATTRIBUTE_LHC_SAMPLES_EXOGENOUS_UNCERTAINTIES",
		table_name_attribute_primary: str = "ATTRIBUTE_PRIMARY",
		table_name_attribute_strategy: str = "ATTRIBUTE_STRATEGY",
		table_name_base_input: str = "MODEL_BASE_INPUT_DATABASE",
		table_name_input: str = "MODEL_INPUT",
		table_name_output: str = "MODEL_OUTPUT",
		tables_write_exclude: Union[List[str], None] = ["input"]
	):
		self.logger = logger

		self._initialize_fp_base_output(
			export_engine,
			fp_base_output,
			create_dir_output = create_dir_output
		)




	##############################################
	#	INITIALIZATION AND SUPPORT FUNCTIONS	#
	##############################################


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



	def _initialize_analysis_id(self,
		analysis_run_id: Union[ssp.SISEPUEDEAnalysisID, str, None]
	) -> None:
		"""
		Initialize the session id. Initializes the following properties:

			* self.sisepuede_analysis_id (SISEPUEDEAnalysisID object)
			* self.analysis_id (shortcurt to self.sisepuede_analysis_id.id)
		"""
		if isinstance(analysis_run_id, SISEPUEDEAnalysisID):
			self.sisepuede_analysis_id = analysis_run_id
		elif isinstance(analysis_run_id, str):

			try:
				self.sisepuede_analysis_id = SISEPUEDEAnalysisID(
					id_str = analysis_run_id,
					logger = self.logger
				)

			except Exception as e:
				self._log(f"Invalid ", type_log = "warning")

		self.analysis_id = self.sisepuede_analysis_id.id



	def _initialize_fp_base_output(self,
		export_engine: Union[sqlalchemy.engine.Engine, str],
		fp_base_output: Union[str, None],
		create_dir_output: bool = False
	) -> None:
		"""
		Initialize the output directory. Sets the following properties:

			* self.fp_base_output
		"""
		self.dir_output = os.path.getcwd()

		if isinstance(fp_base_output, str):

			if os.path.exists(dir_output) or create_dir_output:
				self.dir_output = sf.check_path(dir_output, True)

		self._log(f"Setting output directory to {self.dir_output}", type_log = "info")



	def _initialize_output_database(self,
		export_engine: Union[sqlalchemy.engine.Engine, str, None],
		fn_out_base: Union[str, None] = None,
		default_engine_str: str = "sqlite"
	) -> None:
		"""
		Initialize the output database based on the export_engine. Sets the following
			properties:

			* self.fp_base_output


		Function Arguments
		------------------
		- export_engine: string specifying output method, or, optionally, sqlalchemy
		engine connected to a database/schema used to output data. Options for
		export are:
			* string:
				* "csv": exports CSVs to subdirectory (associated with analysis run id)
					located in output directory
				* "sqlite": exports all tables to a SQL lite database located in the
					output directory
			* sqlalchemy.engine.Engine: sqlalchemy engine that specifies a database
				and schema to write output tables to. This engine can be used to write
				to a remote database service.
			* None:
				If `None`, defaults to default string in SISEPUEDE output subdirectory
		- fn_out_base: based file path to use for output (appends .sqlite if writing
			to SQLite database, or is written to SISPUEDE output directory if )
		"""
		# set some baseline validity sets
		valid_engine_strs = ["csv", "sqlite"]
		valid_extensions_csv = ["csv"]
		valid_extensions_sql = ["sqlite", "db"]

		# check default engine spcification and engine
		default_engine_str = "sqlite" if (default_engine_str not in valid_engine_strs) else default_engine_str
		export_engine = default_engine_str if (export_engine is None) else export_engine

		# case where string
		if isinstance(export_engine, str):
			export_engine = default_engine_str if (export_engine not in valid_engine_strs) else export_engine
			fp_out_base =

			# if sql, convert to engine
			if

		if isinstance(fp_nemomod_temp_sqlite_db, str):
			try_endings = [export_engine.endswith(x) for x in valid_extensions_sql]
			if any(try_endings):
				self.fp_nemomod_temp_sqlite_db = fp_nemomod_temp_sqlite_db


	#
	#
	#
	def _check_output_database(self,

	):
		"""
		Initialize
		"""
		return None
