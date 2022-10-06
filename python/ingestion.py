import data_structures as ds
import numpy as np
import os, os.path
import pandas as pd
import pyDOE2 as pyd
import re
import setup_analysis as sa
import support_functions as sf
import time
from typing import Union
import warnings


#
class InputTemplate:
	"""
		The InputTemplate class is used to ingest an input data template and format it for the SISEPUEDE DAG.

		See https://sisepuede.readthedocs.io for more information on the input template.

		Initialization
		--------------
		- template: The InputTemplate can be initialized using a file path to an Excel file or a dictionary of
			* path: if initializing using a path, the template should point to an Excel workbook containing the input data template. A description of the workbook's format is found below under "Template Formatting".
			* dict: if initializing using a dictionary, the dictionary should have the following structure:
				{
					"strategy_id-X0": pd.DataFrame(),
					"strategy_id-X1": pd.DataFrame()...
				}

				I.e., keys should follow the

		- model_attributes: a ModelAttributes data structure used to coordinate variables and inputs

		Template Formatting
		-------------------

		(info here)


		Required Fields
		---------------

		(info here)



	"""

	def __init__(self,
		template: Union[str, dict],
		model_attributes: ds.ModelAttributes,
		subsec: Union[str, None],
		field_req_normalize_group: str = "normalize_group",
		field_req_subsector: str = "subsector",
		field_req_trajgroup_no_vary_q: str = "trajgroup_no_vary_q",
		field_req_uniform_scaling_q: str = "uniform_scaling_q",
		field_req_variable: str = "variable",
		field_req_variable_trajectory_group: str = "variable_trajectory_group",
		field_req_variable_trajectory_group_trajectory_type: str = "variable_trajectory_group_trajectory_type"
	):
		self.model_attributes = model_attributes

		# set characteristics of the template (can be modified if needed)
		self.field_req_normalize_group = field_req_normalize_group
		self.field_req_subsector = field_req_subsector
		self.field_req_trajgroup_no_vary_q = field_req_trajgroup_no_vary_q
		self.field_req_uniform_scaling_q = field_req_uniform_scaling_q
		self.field_req_variable = field_req_variable
		self.field_req_variable_trajectory_group = field_req_variable_trajectory_group
		self.field_req_variable_trajectory_group_trajectory_type = field_req_variable_trajectory_group_trajectory_type
		self.list_fields_required_base = [
			self.field_req_normalize_group,
			self.field_req_subsector,
			self.field_req_trajgroup_no_vary_q,
			self.field_req_uniform_scaling_q,
			self.field_req_variable,
			self.field_req_variable_trajectory_group,
			self.field_req_variable_trajectory_group_trajectory_type
		]
		self.list_fields_required_binary = [
			self.field_req_normalize_group,
			self.field_req_trajgroup_no_vary_q,
			self.field_req_uniform_scaling_q
		]

		#
		self.regex_sheet_name = self.set_regex_sheet_name()
		(
			self.dict_strategy_id_to_sheet,
			self.dict_strategy_id_to_strategy_sheet,
			self.field_max,
			self.field_min,
			self.fields_tp
		) = self.read_template(template)



	##################################
	#	INITIALIZATION FUNCTIONS	#
	##################################

	def set_regex_sheet_name(self
	) -> re.Pattern:
		"""
		Set the regular expression for input sheets to match
		"""
		return re.compile(f"{self.model_attributes.dim_strategy_id}-(\d*$)")



	############################
	#	TEMPLATE FUNCTIONS	#
	############################

	def get_sheet_strategy(self,
		sheet_name: str,
		return_type: type = int
	) -> Union[int, dict]:
		"""
			Get the strategy associated with a sheet.

			Function Arguments
			------------------
			- sheet_name: the name of the sheet to import

			Keyword Arguments
			-----------------
			- return_type: int or dict.
				* return_type = int will return the strategy number associated with a sheet.
				* return_type = dict will return a dictionary mapping the sheet name to the strategy number.
		"""
		out = self.regex_sheet_name.match(sheet_name)
		# update if there is a match
		if out is not None:
			id = int(out.groups()[0])
			out = {sheet_name: id} if (return_type == dict) else id

		return out



	# read an input template and gather key characteristics
	def read_template(self,
		template_input: Union[str, dict]
	) -> tuple:
		"""
			Import the InputTemplate, check strategies, and set characteristics

			Returns
			-------
			Returns a 5-tuple of the following order:
			- dict_outputs: dictionary {int: pd.DataFrame} mapping a strategy id to the associated dataframe
			- dict_sheet_to_strategy: dictionary mapping a sheet to a strategy name
			- field_max: field specifying the maximum scalar in the final time period
			- field_min: field specifying the minimum scalar in the final time period
			- fields_tp: fields specifying the time periods

			Function Arguments
			------------------
			- template_input: file path (str) to Excel Template or input dictionary (dict) with keys matching InputTemplate.regex_sheet_name
		"""

		if isinstance(template_input, str):
			fp_read = sf.check_path(template_input, False)
			dict_inputs = pd.read_excel(template_input, sheet_name = None)
		elif not isinstance(template_input, dict):
			return None
		else:
			dict_inputs = template_input

		# iteration initializations - objects used to check
		all_time_period_fields = None
		all_time_period_fields_max = None
		any_baseline = False
		strat_base = self.model_attributes.get_baseline_scenario_id(self.model_attributes.dim_strategy_id)

		# outputs and iterators
		dict_outputs = {}
		dict_strategy_to_sheet = {}
		sheets_iterate = list(dict_inputs.keys())

		for k in sheets_iterate:
			if self.regex_sheet_name.match(k) is not None:
				# get the strategy, test if it is baseline
				dict_strat_cur = self.get_sheet_strategy(k, return_type = dict)
				strat_cur = dict_strat_cur.get(k)
				baseline_q = (strat_cur == strat_base)
				any_baseline = any_baseline or baseline_q
				# get the data frame and check it
				df_template_sheet = dict_inputs.get(k)

				tup = self.verify_input_template_sheet(df_template_sheet, base_strategy_q = baseline_q)

				# note: error passing and descriptions are handled in verify_input_template_sheet
				if tup is not None:
					dict_field_tp_to_tp, df_template_sheet, field_min, field_max, fields_tp  = tup

					# check time period fields
					all_time_period_fields = set(fields_tp) if (all_time_period_fields is None) else (all_time_period_fields & set(fields_tp))
					if all_time_period_fields != set(fields_tp):
						raise KeyError(f"Error in sheet {k}: encountered inconsistent definition of time periods.")

					# check max time period fields
					all_time_period_fields_max = set([field_max]) if (all_time_period_fields_max is None) else (all_time_period_fields_max | set([field_max]))
					if len(all_time_period_fields_max) > 1:
						raise KeyError(f"Error in sheet {k}: encountered inconsistent definition of fields specifying maximum and minimum scalars.")

					# check binary fields HEREHERE
					for fld in self.list_fields_required_binary:
						df_template_sheet = sf.check_binary_fields(df_template_sheet, fld)

					# update outputs
					dict_strategy_to_sheet.update({strat_cur: k})
					df_template_sheet = df_template_sheet[self.list_fields_required_base + [field_min, field_max] + fields_tp]
					dict_outputs.update({strat_cur: df_template_sheet})



		if not any_baseline:
			warnings.warn(f"Note: no sheets associated with the baseline strategy {strat_base} were found in the input template. Check the template before proceeding to build the input database.")

		return dict_outputs, dict_strategy_to_sheet, field_max, field_min, fields_tp



	def build_inputs_by_strategy(self,
		dict_strategies_to_sheet: dict = None,
		strategies_include: list = None
	) -> pd.DataFrame:
		"""
		Built a sectoral input variable database for SISEPUEDE based on the input template. This database can be combined across multiple templtes and used to create a SampleUnit object to explore uncertainty.

		Function Arguments
		------------------

		Keyword Arguments
		-----------------
		- dict_strategies_to_sheet: dictionary of type {int -> pd.DataFrame} that maps a strategy id to its associated template sheet
			* If None (default), uses InputTemplate.dict_strategy_id_to_sheet
		- strategies_include: list or list-like (np.array) of strategy ids to include in the database (integer). If None, include all.

		"""
		dict_strategies_to_sheet = self.dict_strategy_id_to_sheet if (dict_strategies_to_sheet is None) else dict_strategies_to_sheet
		strat_base = self.model_attributes.get_baseline_scenario_id(self.model_attributes.dim_strategy_id)
		strats_all = sorted(list(dict_strategies_to_sheet.keys()))
		strategies_include = strats_all if (strategies_include is None) else [x for x in strategies_include if x in strats_all]
		if strat_base not in strategies_include:
			if strat_base in dict_strategies_to_sheet.keys():
				strategies_include = [strat_base] + [x for x in strategies_include if (x != strat_base)]
			else:
				raise KeyError(f"Error in build_inputs_by_strategy: key '{strat_base}' (baseline strategy) not found")
		else:
			strategies_include = [strat_base] + [x for x in strategies_include if (x != strat_base)]
		#
		df_out = []

		for strat in strategies_include:
			df_sheet = dict_strategies_to_sheet.get(strat)
			df_sheet = self.model_attributes.add_index_fields(
				df_sheet,
				strategy_id = strat
			)

			if strat != strat_base:
				#
				# strat_base is always the first
				#
				vars_cur = list(df_sheet[self.field_req_variable])
				df_sheet_base = df_out[0][~df_out[0][self.field_req_variable].isin(vars_cur)].copy()
				df_sheet_base[self.model_attributes.dim_strategy_id] = df_sheet_base[self.model_attributes.dim_strategy_id].replace({strat_base: strat})
				df_sheet = pd.concat([df_sheet, df_sheet_base[df_sheet.columns]], axis = 0).reset_index(drop = True)

			if len(df_out) == 0:
				df_out.append(df_sheet)
			else:
				df_out.append(df_sheet[df_out[0].columns])

		df_out = pd.concat(df_out, axis = 0)

		return df_out



	##  check specification of time periods on an input template sheet, then return any valid fields + a cleaned pd.DataFrame
	def verify_and_return_sheet_time_periods(self,
		df_in: pd.DataFrame,
		regex_max: re.Pattern = re.compile("max_(\d*$)"),
		regex_min: re.Pattern = re.compile("min_(\d*$)"),
		regex_tp: re.Pattern = re.compile("(\d*$)")
	) -> tuple:
		"""
		Get time periods in a sheet in addition to min/max specification fields.

		Returns
		-------

		Returns a 5-tuple in the following order:

		(dict_field_tp_to_tp, df_in, field_min, field_max, fields_tp)

		- dict_field_tp_to_tp:
		- df_in: cleaned DataFrame that excludes invalid time periods
		- field_min: field that stores the minimum scalar for the final time period in the template
		- field_max: field that stores the maximum scalar for the final time period in the template
		- fields_tp: fields denoting time periods


		Function Arguments
		------------------

		- df_in: Input data frame storing template values


		Keyword Arguments
		-----------------

		- regex_max: re.Pattern (compiled regular expression) used to match the field storing the maximum scalar values at the final time period
		- regex_min: re.Pattern used to match the field storing the minimum scalar values at the final time period
		- regex_tp: re.Pattern used to match the field storing data values for each time period

		"""

		##  GET MIN/MAX AT FINAL TIME PERIOD

		# determine max field/time period
		field_max = [regex_max.match(str(x)) for x in df_in.columns if (regex_max.match(str(x)) is not None)]

		if len(field_max) == 0:
			raise KeyError("No field associated with a maximum scalar value found in data frame.")
		elif len(field_max) > 1:
			fpl = sf.format_print_list(field_max)
			raise KeyError(f"Multiple maximum fields found in input DataFrame: {fpl} all satisfy the conditions. Choose one and retry.")
		else:
			field_max = field_max[0]
			tp_max = int(field_max.groups()[0])
			field_max = field_max.string

		# determine min field/time period
		field_min = [regex_min.match(str(x)) for x in df_in.columns if (regex_min.match(str(x)) is not None)]
		if len(field_min) == 0:
			raise KeyError("No field associated with a minimum scalar value found in data frame.")
		elif len(field_min) > 1:
			fpl = sf.format_print_list(field_min)
			raise KeyError(f"Multiple minimum fields found in input DataFrame: {fpl} all satisfy the conditions. Choose one and retry.")
		else:
			field_min = field_min[0]
			tp_min = int(field_min.groups()[0])
			field_min = field_min.string

		# check that min/max specify final time period
		if (tp_min != tp_max):
			raise ValueError(f"Fields '{field_min}' and '{field_max}' imply asymmetric final time periods.")


		##  GET TIME PERIODS

		# get initial information on time periods
		fields_tp = [regex_tp.match(str(x)) for x in df_in.columns if (regex_tp.match(str(x)) is not None)]
		# rename the dataframe to ensure fields are strings
		dict_rnm = dict([(x, str(x)) for x in df_in.columns if not isinstance(x, str)])
		df_in.rename(columns = dict_rnm, inplace = True)

		dict_field_tp_to_tp = dict([(x.string, int(x.groups()[0])) for x in fields_tp])

		# check fields for definition in attribute_time_period
		pydim_time_period = self.model_attributes.get_dimensional_attribute(self.model_attributes.dim_time_period, "pydim")
		attr_tp = self.model_attributes.dict_attributes.get(pydim_time_period)
		# fields to keep/drop

		fields_valid = [x.string for x in fields_tp if (dict_field_tp_to_tp.get(x.string) in attr_tp.key_values)]
		fields_invalid = [x.string for x in fields_tp if (x.string not in fields_valid)]
		defined_tp = [dict_field_tp_to_tp.get(x) for x in fields_valid]

		if (tp_max not in defined_tp):
			raise ValueError(f"Error trying to define template: the final time period {tp_max} defined in the input template does not exist in the {self.model_attributes.dim_time_period} attribute table at '{attr_tp.fp_table}'")

		if len(fields_invalid) > 0:
			flds_drop = sf.format_print_list([x for x in fields_invalid])
			warnings.warn(f"Dropping fields {flds_drop} from input template: the time periods are not defined in the {self.model_attributes.dim_time_period} attribute table at '{attr_tp.fp_table}'")
			df_in.drop(fields_invalid, axis = 1, inplace = True)

		return (dict_field_tp_to_tp, df_in, field_min, field_max, fields_valid)



	##  verify the structure of an input template sheet
	def verify_input_template_sheet(self,
		df_template_sheet: pd.DataFrame,
		base_strategy_q: bool = False,
		sheet_name: str = None
	) -> Union[tuple, None]:
		"""
		Verify the formatting of an input template sheet and retrieve information to verify all strategies

		Returns
		-------

		Returns a 5-tuple in the following order:

		(dict_field_tp_to_tp, df_in, field_min, field_max, fields_tp)

		- dict_field_tp_to_tp:
		- df_in: cleaned DataFrame that excludes invalid time periods
		- field_min: field that stores the minimum scalar for the final time period in the template
		- field_max: field that stores the maximum scalar for the final time period in the template
		- fields_tp: fields denoting time periods

		*NOTE*: returns None if errors occur trying to load, so return values should not be assigned as tuple elements


		Function Arguments
		------------------
		- df_template_sheet: A data frame representing the input template sheet by strategy


		Keyword Arguments
		-----------------
		- base_strategy_q: running the base strategy? If so, requirements for input variables are different.
		- sheet_name: name of the sheet passed for error handling and troubleshooting

		"""

		# check fields and retrieve information about time periods
		try:

			sf.check_fields(df_template_sheet, self.list_fields_required_base)
			(
				dict_field_tp_to_tp,
				df_template_sheet,
				field_min,
				field_max,
				fields_tp
			) = self.verify_and_return_sheet_time_periods(df_template_sheet)

		except Exception as e:
			sheet_str = f" '{sheet_name}'" if (sheet_name is not None) else ""
			warnings.warn(f"Trying to verify sheet{sheet_str} produced the following error in verify_input_template_sheet:\n\t{e}\nReturning None")
			return None

		return (dict_field_tp_to_tp, df_template_sheet, field_min, field_max, fields_tp)





class BaseInputDatabase:
	"""
		The BaseInputDatabase class is used to combine InputTemplates from multiple sectors into a single input for

		Initialization Arguments
		------------------------
		- fp_templates: file path to directory containing input Excel templates
		- model_attributes: ModelAttributes object used to define sectors and check templates
		- regions: regions to include
			* If None, then try to initialize all input regions

		Optional Arguments
		--------=---------
		- demo_q: whether or not the database is run as a demo
			* If run as demo, then `fp_templates` does not need to include subdirectories for each region specified
		- sectors: sectors to include
			* If None, then try to initialize all input sectors


	"""
	def __init__(self,
		fp_templates: str,
		model_attributes: ds.ModelAttributes,
		regions: Union[list, None],
		demo_q: bool = True,
		sectors: Union[list, None] = None
	):
		self.demo_q, self.fp_templates, self.model_attributes = demo_q, fp_templates, model_attributes
		self.regions = self.get_regions(regions)
		self.sectors = self.get_sectors(sectors)

		self.database = self.generate_database()


	##################################
	#	INITIALIZATION FUNCTIONS	#
	##################################

	def get_regions(self, regions: Union[str,  None]) -> list:
		"""
		Import regions for the BaseInputDatabase class from BaseInputDatabase
		"""
		attr_region = self.model_attributes.dict_attributes.get("region")

		if regions is None:
			regions_out = attr_region.key_values
		else:
			regions_out = [self.model_attributes.clean_region(region) for region in regions]
			regions_out = [region for region in regions_out if region in attr_region.key_values]

		if self.demo_q and len(regions_out) > 0:
			regions_out = [regions_out[0]]

		return regions_out



	def get_sectors(self, sectors: Union[str, None]) -> list:
		"""
		Import regions for the BaseInputDatabase class from BaseInputDatabase
		"""
		attr_sector = self.model_attributes.dict_attributes.get("abbreviation_sector")
		key_dict = f"{attr_sector.key}_to_sector"
		dict_conv = attr_sector.field_maps.get(key_dict)
		all_sectors = [dict_conv.get(x) for x in attr_sector.key_values]

		if sectors is None:
			sectors_out = all_sectors
		else:
			sectors_out = [sector for sector in sectors if sector in all_sectors]

		return sectors_out



	############################
	#	CORE FUNCTIONALITY	#
	############################

	def generate_database(self,
		regions: Union[list, None] = None,
		sectors: Union[list, None] = None,
		**kwargs
	) -> Union[pd.DataFrame, None]:
		"""
			Load templates and generate a base input database.
				* Returns None if no valid templates are found.

			Function Arguments
			------------------

			Keyword Arguments
			-----------------
			regions: List of regions to load. If None, load BaseInputDatabase.regions.
			sectors: List of sectors to load. If None, load BaseInputDatabase.sectors
			**kwargs: passed to BaseInputDatabase.get_template_path()
		"""
		# initialize output
		all_fields = None
		df_out = []
		regions = self.regions if (regions is None) else self.get_regions(regions)
		sectors = self.sectors if (sectors is None) else self.get_regions(sectors)

		for region in enumerate(regions):
			i, region = region

			df_out_region = []

			for sector in enumerate(sectors):
				j, sector = sector

				# read the input database for the sector
				try:
					template_cur = InputTemplate(
						self.get_template_path(region, sector, **kwargs),
						sa.model_attributes,
						None
					)

					df_template_db = template_cur.build_inputs_by_strategy()

				except Exception as e:
					warnings.warn(f"Warning in generate_database--template read for sector '{sector}' in region '{region}' failed. The following error was returned: {e}")
					df_template_db = None

				# check time period fields
				set_template_cols = set(df_template_db.columns)
				if all_fields is not None:
					if not set(df_template_db.columns).issubset(all_fields):
						warnings.warn(f"Error in sector '{sector}', region '{region}': encountered inconsistent definition of template fields. Dropping...")
						df_template_db = None
					else:
						fields_drop = list(set_template_cols - all_fields)
						df_template_db.drop(fields_drop, axis = 1, inplace = True) if (len(fields_drop) > 0) else None
				else:
					all_fields = set_template_cols

				# update dataframe list
				if (len(df_out_region) == 0) and (df_template_db is not None):
					df_out_region = [df_template_db for x in range(len(self.sectors))]
				elif len(df_out_region) > 0:
					df_out_region[j] = df_template_db

			# add region
			df_out_region = pd.concat(df_out_region, axis = 0).reset_index(drop = True) if (len(df_out_region) > 0) else None
			df_out_region = self.model_attributes.add_index_fields(
				df_out_region,
				region = region
			)

			# add to outer df
			if (len(df_out) == 0) and (df_out_region is not None):
				df_out = [df_out_region for x in range(len(self.regions))]
			elif len(df_out) > 0:
				df_out[i] = df_out_region

		df_out = pd.concat(df_out, axis = 0).reset_index(drop = True) if (len(df_out) > 0) else None

		return df_out



	def get_template_path(self,
		region: Union[str, None],
		sector: str,
		append_base_directory: bool = True,
		create_export_dir: bool = False,
		template_base_str: str = "model_input_variables"
	) -> str:
		"""
			Generate a path for an input template based on a sector, region, a database regime type, and a dictionary mapping different database regime types to input directories storing the input Excel templates.

			Function Arguments
			------------------
			- region: three-character region code
			- sector: the emissions sector (e.g., AFOLU, Circular Economy, etc.

			Keyword Arguments
			-----------------
			- append_base_directory: append the base directory name (basename = os.path.basename(self.fp_templates)) to the template? Default is true to reduce ambiguity.
				* if True, templates take form `model_input_variables_{region}_{abv_sector}_{basename}.xlsx`
				* if False, templates take form `model_input_variables_{region}_{abv_sector}.xlsx`
			- create_export_dir: boolean indicating whether or not to create a directory specified in dict_valid_types if it does not exist.
			- template_base_str: baseline string for naming templates
		"""

		attr_region = self.model_attributes.dict_attributes.get("region")

		# check sector
		if sector in self.model_attributes.all_sectors:
			abv_sector = self.model_attributes.get_sector_attribute(sector, "abbreviation_sector")
		else:
			valid_sectors = sf.format_print_list(self.model_attributes.all_sectors)
			raise ValueError(f"Invalid sector '{sector}' specified: valid sectors are {valid_sectors}.")

		# check region
		if not self.demo_q:
			# check
			if region is None:
				raise ValueError(f"Invalid specification of region: a region must be specified unless the database is initialized in demo mode.")

			region_lower = self.model_attributes.clean_region(region)
			# check region and create export directory if necessary
			if region_lower not in attr_region.key_values:
				valid_regions = sf.format_print_list(attr_region.key_values)
				raise ValueError(f"Invalid region '{region}' specified: valid regions are {valid_regions}.")

			dir_exp = sf.check_path(os.path.join(self.fp_templates, region_lower), create_export_dir)
			region_str = f"_{region_lower}"
		else:
			region_str = ""

		# check appendage
		if append_base_directory:
			append_str = os.path.basename(self.fp_templates)
			append_str = f"_{append_str}"
		else:
			append_str = ""


		fn_out = f"{template_base_str}{region_str}_{abv_sector}{append_str}.xlsx"

		return os.path.join(self.fp_templates, fn_out)
