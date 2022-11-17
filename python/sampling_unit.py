from attribute_table import AttributeTable
import itertools
import logging
import math
import numpy as np
import pandas as pd
import os, os.path
import re
import support_functions as sf
import time
from typing import *



##  class for sampling and generating experimental design
class SamplingUnit:
	"""
	Generate future trajectories based on an input database.

	Initialization Arguments
	------------------------
	- df_variable_definition: DataFrame used to define variable specifications
	- dict_baseline_ids: dictionary mapping a string of a baseline id field to a
		baseline id value (integer)
	- time_period_u0: first time period with uncertainty

	Keyword Arguments
	-----------------
	- fan_function_specification: type of uncertainty approach to use
		* linear: linear ramp to time time T - 1
		* sigmoid: sigmoid function that ramps to time T - 1
	- field_time_period: field used to denote the time period
	- field_uniform_scaling_q: field used to identify whether or not a variable
	- field_variable: field used to specify variables
	- field_variable_trajgroup: field used to identify the trajectory group
		(integer)
	- field_variable_trajgroup_type: field used to identify the trajectory group
		type (max, min, mix, or lhs)
	- key_strategy: field used to identify the strategy (int)
		* This field is important as uncertainty in strategies is assessed
			differently than uncetainty in other variables
	- missing_flag_trajgroup: flag used to identify null trajgroups (default is
		-999)
	- regex_id: regular expression used to identify id fields in the input
		template
	- regex_max: re.Pattern (compiled regular expression) used to match the
		field storing the maximum scalar values at the final time period
	- regex_min: re.Pattern used to match the field storing the minimum scalar
		values at the final time period
	- regex_tp: re.Pattern used to match the field storing data values for each
		time period
	"""
	def __init__(self,
		df_variable_definition: pd.DataFrame,
		dict_baseline_ids: Dict[str, int],
		time_period_u0: int,
		fan_function_specification: str = "linear",
		field_time_period: str = "time_period",
		field_trajgroup_no_vary_q: str = "trajgroup_no_vary_q",
		field_uniform_scaling_q: str = "uniform_scaling_q",
		field_variable_trajgroup: str = "variable_trajectory_group",
		field_variable_trajgroup_type: str = "variable_trajectory_group_trajectory_type",
		field_variable: str = "variable",
		key_strategy: str = "strategy_id",
		missing_trajgroup_flag: int = -999,
		regex_id: re.Pattern = re.compile("(\D*)_id$"),
		regex_max: re.Pattern = re.compile("max_(\d*$)"),
		regex_min: re.Pattern = re.compile("min_(\d*$)"),
		regex_tp: re.Pattern = re.compile("(\d*$)")
	):

		##  set some attributes

		# from function args
		self.field_time_period = field_time_period
		self.field_trajgroup_no_vary_q = field_trajgroup_no_vary_q
		self.field_uniform_scaling_q = field_uniform_scaling_q
		self.field_variable_trajgroup = field_variable_trajgroup
		self.field_variable_trajgroup_type = field_variable_trajgroup_type
		self.field_variable = field_variable
		self.key_strategy = key_strategy
		self.missing_trajgroup_flag = missing_trajgroup_flag
		self.time_period_end_certainty = self.check_time_start_uncertainty(time_period_u0)
		# others
		self._set_parameters()
		self._set_attributes_from_table(
			df_variable_definition,
			regex_id,
			regex_max,
			regex_min,
			regex_tp
		)

		# perform initializations
		self._initialize_uncertainty_functional_form(fan_function_specification)
		self._initialize_scenario_variables(dict_baseline_ids)
		self._initialize_trajectory_arrays()
		self._initialize_xl_type()





	##################################
	#	INITIALIZATION FUNCTIONS	#
	##################################

	# get attributes from the table
	def _set_attributes_from_table(self,
		df_variable_definition: pd.DataFrame,
		regex_id: re.Pattern,
		regex_max: re.Pattern,
		regex_min: re.Pattern,
		regex_tp: re.Pattern
	) -> None:
		"""
		Set a range of attributes derived the input df_variable_definition.
			Sets the following properties:

			* self.df_variable_definitions
			* self.fields_id
			* self.required_fields

		Function Arguments
		------------------
		- df_variable_definition: data frame used to set variable specifications
		- regex_id: regular expression used to identify id fields in the input
			template
		- regex_max: re.Pattern (compiled regular expression) used to match the
			field storing the maximum scalar values at the final time period
		- regex_min: re.Pattern used to match the field storing the minimum scalar
			values at the final time period
		- regex_tp: re.Pattern used to match the field storing data values for
			each time period

		"""
		self.df_variable_definitions, self.required_fields = self.check_input_data_frame(df_variable_definition)
		self.fields_id = self.get_id_fields(regex_id)
		self.field_min_scalar, self.field_max_scalar, self.time_period_scalar = self.get_scalar_time_period(
			regex_max,
			regex_min
		)
		self.fields_time_periods, self.time_periods = self.get_time_periods(regex_tp)
		self.variable_trajectory_group = self.get_trajgroup()
		self.variable_trajectory_group_vary_q = self.get_trajgroup_vary_q()



	def _set_parameters(self,
	) -> None:
		"""
		Set some key parameters.

		* self.dict_required_tg_spec_fields
		* self.key_mix_trajectory
		* self.key_inf_traj_boundary
		* self.key_sup_traj_boundary
		* self.primary_key_id_coordinates
		* self.required_tg_specs

		"""

		self.key_mix_trajectory = "mixing_trajectory"
		self.key_inf_traj_boundary = "trajectory_boundary_0"
		self.key_sup_traj_boundary = "trajectory_boundary_1"
		self.primary_key_id_coordinates = "primary_key_id_coord"

		# maps internal name (key) to classification in the input data frame (value)
		self.dict_required_tg_spec_fields = {
			self.key_mix_trajectory: "mix",
			self.key_inf_traj_boundary: "trajectory_boundary_0",
			self.key_sup_traj_boundary: "trajectory_boundary_1"
		}
		self.required_tg_specs = list(self.dict_required_tg_spec_fields.values())



	def check_input_data_frame(self,
		df_in: pd.DataFrame
	):
		"""
		Check df_in for required fields. Sets the following attributes:

			* self.df_variable_definitions
			* self.required_fields
		"""
		# some standardized fields to require
		fields_req = [
			self.key_strategy,
			self.field_trajgroup_no_vary_q,
			self.field_variable_trajgroup,
			self.field_variable_trajgroup_type,
			self.field_uniform_scaling_q,
			self.field_variable
		]

		if len(set(fields_req) & set(df_in.columns)) < len(set(fields_req)):
			fields_missing = list(set(fields_req) - (set(fields_req) & set(df_in.columns)))
			fields_missing.sort()
			str_missing = ", ".join([f"'{x}'" for x in fields_missing])
			raise ValueError(f"Error: one or more columns are missing from the data frame. Columns {str_missing} not found")

		elif (self.key_strategy in df_in.columns) and ("_id" not in self.key_strategy):
			raise ValueError(f"Error: the strategy field '{self.key_strategy}' must contain the substring '_id'. Check to ensure this substring is specified.")

		return df_in.drop_duplicates(), fields_req



	def _initialize_scenario_variables(self,
		dict_baseline_ids: Dict[str, int],
		df_in: Union[pd.DataFrame, None] = None,
		fields_id: Union[list, None] = None,
		field_merge_key: Union[str, None] = None
	) -> None:
		"""
		Check inputs of the input data frame and id fields. Sets the following
			properties:

			* self.data_table
			* self.df_id_coordinates
			* self.dict_baseline_ids
			* self.dict_id_values
			* self.dict_variable_info
			* self.id_coordinates
			* self.num_scenarios
			* self.variable_specifications


		Function Arguments
		------------------
		- dict_baseline_ids: dictionary mapping each dimensional key to nominal
			baseline

		Keyword Arguments
		-----------------
		- df_in: input data frame used to specify variables
		- fields_id: id fields included in df_in
		- field_merge_key: scenario key
		"""
		df_in = self.df_variable_definitions if not isinstance(df_in, pd.DataFrame) else df_in
		fields_id = self.fields_id if not isinstance(fields_id, list) else fields_id
		field_merge_key = self.primary_key_id_coordinates if (field_merge_key is None) else field_merge_key
		tups_id = set([tuple(x) for x in np.array(df_in[fields_id])])

		for tg_type in self.required_tg_specs:
			df_check = df_in[df_in[self.field_variable_trajgroup_type] == tg_type]
			for vs in list(df_check[self.field_variable].unique()):
				tups_id = tups_id & set([tuple(x) for x in np.array(df_check[df_check[self.field_variable] == vs][fields_id])])
		#
		df_scen = pd.DataFrame(tups_id, columns = fields_id)
		df_in = pd.merge(df_in, df_scen, how = "inner", on = fields_id)
		df_scen[field_merge_key] = range(len(df_scen))
		tups_id = sorted(list(tups_id))

		# id values and baseline ids
		dict_id_values, dict_baseline_ids = self.get_scenario_values(
			dict_baseline_ids,
			df_in = df_in,
			fields_id = fields_id
		)
		var_specs = self.get_all_vs(df_in)

		self.data_table = df_in
		self.df_id_coordinates = df_scen
		self.dict_baseline_ids = dict_baseline_ids
		self.dict_id_values = dict_id_values
		self.dict_variable_info = self.get_variable_dictionary(df_in, var_specs)
		self.id_coordinates = tups_id
		self.num_scenarios = len(tups_id)
		self.variable_specifications = var_specs




	def check_time_start_uncertainty(self,
		t0: int
	) -> int:
		return max(t0, 1)



	def generate_indexing_data_frame(self,
		df_id_coords: Union[pd.DataFrame, None] = None,
		dict_additional_fields: Dict[str, Union[float, int, str]] = None,
		field_primary_key_id_coords: Union[str, None] = None,
		field_time_period: Union[str, None] = None
	) -> pd.DataFrame:

		"""
		Generate an data frame long by time period and all id coordinates included in the sample unit.

		Keyword Arguments
		-----------------
		- df_id_coords: data frame containing id coordinates + primary key (in field_primary_key_id_coords)
			* If None, default to self.df_id_coordinates
		- dict_additional_fields: dictionary mapping additional fields to values to add
			* If None, no additional fields are added
		- field_primary_key_id_coords: field in df_id_coords denoting the primary key
			* If None, default to self.primary_key_id_coordinates
		- field_time_period: field to use for data frame
			* If None, default to self.field_time_period
		"""

		df_id_coords = self.df_id_coordinates if (df_id_coords is None) else df_id_coords
		field_primary_key_id_coords = self.primary_key_id_coordinates if (field_primary_key_id_coords is None) else field_primary_key_id_coords
		field_time_period = self.field_time_period if (field_time_period is None) else field_time_period

		# build array of coordinates x time periods
		df_coords_by_future = np.array([
			np.repeat(
				df_id_coords[field_primary_key_id_coords],
				len(self.time_periods)
			),
			np.concatenate(
				np.repeat(
					[self.time_periods],
					len(self.df_id_coordinates),
					axis = 0
				)
			)
		]).transpose()

		# convert to data frame
		df_coords_by_future = pd.DataFrame(
			df_coords_by_future,
			columns = [field_primary_key_id_coords, field_time_period]
		)

		df_coords_by_future = pd.merge(
			df_coords_by_future,
			df_id_coords,
			how = "left"
		).sort_values(
			by = [field_primary_key_id_coords, field_time_period]
		).reset_index(
			drop = True
		).drop(
			field_primary_key_id_coords, axis = 1
		)

		if dict_additional_fields is not None:
			df_coords_by_future = sf.add_data_frame_fields_from_dict(
				df_coords_by_future,
				dict_additional_fields
			)

		return df_coords_by_future



	def get_all_vs(self,
		df_in: pd.DataFrame
	) -> List:
		"""
		Get all variable schema associated with input template df_in
		"""
		if not self.field_variable in df_in.columns:
			raise ValueError(f"Field '{self.field_variable}' not found in data frame.")
		all_vs = sorted(list(df_in[self.field_variable].unique()))

		return all_vs



	def get_id_fields(self,
		regex_id: re.Pattern,
		df_in: Union[pd.DataFrame, None] = None
	) -> List:
		"""
		Get all id fields associated with input template df_in.

		Function Arguments
		------------------
		- regex_id: regular expression used to identify id fields

		Keyword Arguments
		-----------------
		- df_in: data frame to use to find id fields. If None, use
			self.df_variable_definitions

		"""

		if not isinstance(regex_id, re.Pattern):
			fields_out = []
		else:
			df_in = self.df_variable_definitions if (df_in is None) else df_in
			fields_out = sorted(
				[x for x in df_in.columns if (regex_id.match(x) is not None)]
			)

		if len(fields_out) == 0:
			raise ValueError(f"No id fields found in data frame.")

		return fields_out



	def _initialize_trajectory_arrays(self,
		df_in: Union[pd.DataFrame, None] = None,
		fields_id: Union[list, None] = None,
		fields_time_periods: Union[list, None] = None,
		variable_specifications: Union[list, None] = None
	) -> Dict[str, np.ndarray]:
		"""
		Order trajectory arrays by id fields; used for quicker lhs application
			across id dimensions. Sets the following properties:

			* self.ordered_trajectory_arrays
			* self.scalar_diff_arrays

		Keyword Arguments
		-----------------
		- df_in: variable specification data frame
		- fields_id: id fields included in the specification of variables
		- fields_time_periods: fields denoting time periods
		- variable_specifications: list of variable specifications included in
			df_in
		"""

		# initialize defaults
		df_in = self.data_table if not isinstance(df_in, pd.DataFrame) else df_in
		fields_id = self.fields_id if not isinstance(fields_id, list) else fields_id
		fields_time_periods = self.fields_time_periods if not isinstance(fields_time_periods, list) else fields_time_periods
		variable_specifications = self.variable_specifications if not isinstance(variable_specifications, list) else variable_specifications

		# set some other variables
		tp_end = self.fields_time_periods[-1]
		dict_scalar_diff_arrays = {}
		dict_ordered_traj_arrays = {}

		for vs in variable_specifications:

			df_cur_vs = df_in[df_in[self.field_variable].isin([vs])]
			dfs_cur = [(None, df_cur_vs)] if (self.variable_trajectory_group is None) else df_cur_vs.groupby([self.field_variable_trajgroup_type])

			for df_cur in dfs_cur:
				tgs, df_cur = df_cur
				df_cur.sort_values(by = fields_id, inplace = True)

				# ORDERED TRAJECTORY ARRAYS
				array_data = np.array(df_cur[fields_time_periods])
				coords_id = df_cur[fields_id]
				dict_ordered_traj_arrays.update(
					{
						(vs, tgs): {
							"data": np.array(df_cur[fields_time_periods]),
							"id_coordinates": df_cur[fields_id]
						}
					}
				)

				# SCALAR DIFFERENCE ARRAYS - order the max/min scalars
				var_info = self.dict_variable_info.get((vs, tgs))
				vec_scale_max = np.array([var_info.get("max_scalar")[tuple(x)] for x in np.array(coords_id)])
				vec_scale_min = np.array([var_info.get("min_scalar")[tuple(x)] for x in np.array(coords_id)])

				# difference, in final time period, between scaled value and baseline value-dimension is # of scenarios
				dict_tp_end_delta = {
					"max_tp_end_delta": array_data[:,-1]*(vec_scale_max - 1),
					"min_tp_end_delta": array_data[:,-1]*(vec_scale_min - 1)
				}

				dict_scalar_diff_arrays.update({(vs, tgs): dict_tp_end_delta})

		self.ordered_trajectory_arrays = dict_ordered_traj_arrays
		self.scalar_diff_arrays = dict_scalar_diff_arrays



	def get_scalar_time_period(self,
		regex_max: re.Pattern,
		regex_min: re.Pattern,
		df_in:Union[pd.DataFrame, None] = None
	) -> Tuple[str, str, int]:
		"""
		Determine final time period (tp_final) as well as the fields associated with the minimum
			and maximum scalars (field_min/field_max) using input template df_in. Returns a tuple
			with the following elements:

			* field_min
			* field_max
			* tp_final

		Function Arguments
		------------------
		- regex_max: re.Pattern (compiled regular expression) used to match the
			field storing the maximum scalar values at the final time period
		- regex_min: re.Pattern used to match the field storing the minimum scalar
			values at the final time period

		Keyword Arguments
		-----------------
		- df_in: input data frame defining variable specifications. If None,
			uses self.df_variable_definitions
		"""

		df_in = self.df_variable_definitions if (df_in is None) else df_in

		field_min = [x for x in df_in.columns if (regex_min.match(x) is not None)]
		if len(field_min) == 0:
			raise ValueError("No field associated with a minimum scalar value found in data frame.")
		else:
			field_min = field_min[0]

		# determine max field/time period
		field_max = [x for x in df_in.columns if (regex_max.match(x) is not None)]
		if len(field_max) == 0:
			raise ValueError("No field associated with a maximum scalar value found in data frame.")
		else:
			field_max = field_max[0]

		tp_min = int(field_min.split("_")[1])
		tp_max = int(field_max.split("_")[1])
		if (tp_min != tp_max) | (tp_min == None):
			raise ValueError(f"Fields '{tp_min}' and '{tp_max}' imply asymmetric final time periods.")
		else:
			tp_out = tp_min

		return (field_min, field_max, tp_out)



	def get_scenario_values(self,
		dict_baseline_ids: Dict[str, int],
		df_in: Union[pd.DataFrame, None] = None,
		fields_id: Union[list, None] = None,
	) -> Tuple[Dict, Dict]:
		"""
		Get scenario index values by scenario dimension and verifies baseline
			values. Returns a tuple:

			dict_id_values, dict_baseline_ids

		where `dict_id_values` maps each dimensional key (str) to a list of
			values and `dict_baseline_ids` maps each dimensional key (str) to a
			baseline scenario index

		Function Arguments
		------------------
		- df_in: data frame containing the input template
		- fields_id: list of id fields
		- dict_baseline_ids: dictionary mapping each dimensional key to nominal
			baseline

		Function Arguments
		------------------
		"""

		df_in = self.data_table if not isinstance(df_in, pd.DataFrame) else df_in
		fields_id = self.fields_id if not isinstance(fields_id, list) else fields_id
		#
		dict_id_values = {}
		dict_id_baselines = dict_baseline_ids.copy()

		for fld in fields_id:
			dict_id_values.update({fld: list(df_in[fld].unique())})
			dict_id_values[fld].sort()

			# check if baseline for field is determined
			if fld in dict_id_baselines.keys():
				bv = int(dict_id_baselines[fld])

				if bv not in dict_id_values[fld]:
					if fld == self.key_strategy:
						raise ValueError(f"Error: baseline {self.key_strategy} scenario index '{bv}' not found in the variable trajectory input sheet. Please ensure the basline strategy is specified correctly.")
					else:
						msg_warning = f"The baseline id for dimension {fld} not found. The experimental design will not include futures along this dimension of analysis."
						warnings.warn(msg_warning)
			else:
				# assume minimum >= 0
				bv = min([x for x in dict_id_values[fld] if x >= 0])
				dict_id_baselines.update({fld: bv})
				msg_warning = f"No baseline scenario index found for {fld}. It will be assigned to '{bv}', the lowest non-negative integer."
				warnings.warn(msg_warning)

		return dict_id_values, dict_id_baselines



	def get_time_periods(self,
		regex_tp: re.Pattern,
		df_in:Union[pd.DataFrame, None] = None
	) -> Tuple[List, List]:
		"""
		Get fields associated with time periods in the template as well as time
			periods defined in input template df_in. Returns the following
			elements:

			fields_time_periods, time_periods

			where

			* fields_time_periods: nominal fields in df_in containing time
				periods
			* time_periods: ordered list of integer time periods

		Function Arguments
		-----------------
		- regex_tp: re.Pattern used to match the field storing data values for
			each time period

		Keyword Arguments
		-----------------
		- df_in: input data frame defining variable specifications. If None,
			uses self.

		"""

		df_in = self.df_variable_definitions if (df_in is None) else df_in

		#fields_time_periods = [x for x in df_in.columns if x.isnumeric()]
		#fields_time_periods = [x for x in fields_time_periods if int(x) == float(x)]
		fields_time_periods = [str(x) for x in df_in.columns if (regex_tp.match(str(x)) is not None)]
		if len(fields_time_periods) == 0:
			raise ValueError("No time periods found in data frame.")

		time_periods = sorted([int(x) for x in fields_time_periods])
		fields_time_periods = [str(x) for x in time_periods]

		return fields_time_periods, time_periods



	def get_trajgroup(self,
		df_in:Union[pd.DataFrame, None] = None
	) -> Union[int, None]:
		"""
		Get the trajectory group for the sampling unit from df_in.

		Keyword Arguments
		-----------------
		- df_in: input data frame defining variable specifications. If None,
			uses self.
		"""

		df_in = self.df_variable_definitions if (df_in is None) else df_in

		if not self.field_variable_trajgroup in df_in.columns:
			raise ValueError(f"Field '{self.field_variable_trajgroup}' not found in data frame.")
		# determine if this is associated with a trajectory group
		if len(df_in[df_in[self.field_variable_trajgroup] > self.missing_trajgroup_flag]) > 0:
			return int(list(df_in[self.field_variable_trajgroup].unique())[0])
		else:
			return None



	def get_trajgroup_vary_q(self,
		df_in:Union[pd.DataFrame, None] = None
	) -> Union[int, None]:
		"""
		Get the trajectory group for the sampling unit from df_in.

		Keyword Arguments
		-----------------
		- df_in: input data frame defining variable specifications. If None,
			uses self.
		"""

		df_in = self.df_variable_definitions if (df_in is None) else df_in

		if not self.field_trajgroup_no_vary_q in df_in.columns:
			raise ValueError(f"Field '{self.field_trajgroup_no_vary_q}' not found in data frame.")
		# determine if this is associated with a trajectory group
		out = (len(df_in[df_in[self.field_trajgroup_no_vary_q] == 1]) == 0)

		return out



	def get_variable_dictionary(self,
		df_in: pd.DataFrame,
		variable_specifications: list,
		fields_id: list = None,
		field_max: str = None,
		field_min: str = None,
		fields_time_periods: list = None
	) -> None:
		"""
		Retrieve a dictionary mapping a vs, tg pair to a list of information.
			The variable dictionary includes information on sampling ranges for
			time period scalars, wether the variables should be scaled
			uniformly, and the trajectories themselves.

		"""
		fields_id = self.fields_id if (fields_id is None) else fields_id
		field_max = self.field_max_scalar if (field_max is None) else field_max
		field_min = self.field_min_scalar if (field_min is None) else field_min
		fields_time_periods = self.fields_time_periods if (fields_time_periods is None) else fields_time_periods

		dict_var_info = {}
		tgs_loops = self.required_tg_specs if (self.variable_trajectory_group is not None) else [None]

		for vs in variable_specifications:

			df_vs = sf.subset_df(df_in, {self.field_variable: vs})

			for tgs in tgs_loops:

				df_cur = df_vs if (tgs is None) else sf.subset_df(df_vs, {self.field_variable_trajgroup_type: tgs})
				dict_vs = {
					"max_scalar": sf.build_dict(df_cur[fields_id + [field_max]], force_tuple = True),
					"min_scalar": sf.build_dict(df_cur[fields_id + [field_min]], force_tuple = True),
					"uniform_scaling_q": sf.build_dict(df_cur[fields_id + [self.field_uniform_scaling_q]], force_tuple = True),
					"trajectories": sf.build_dict(df_cur[fields_id + fields_time_periods], (len(self.fields_id), len(self.fields_time_periods)), force_tuple = True)
				}

				dict_var_info.update({(vs, tgs): dict_vs})

		return dict_var_info



	# determine if the sampling unit represents a strategy (L) or an uncertainty (X)
	def _initialize_xl_type(self,
		thresh: float = (10**(-12))
	) -> Tuple[str, Dict[str, Any]]:
		"""
		Infer the sampling unit type--strategy (L) or an uncertainty (X)--by
			comparing variable specification trajectories across strategies.
			Sets the following properties:

			* self.fields_order_strat_diffs
			* self.xl_type, self.dict_strategy_info

		Keyword Arguments
		-----------------
		- thresh: threshold used to identify significant difference between
			variable specification trajectories across strategies. If a
			variable specification trajectory shows a difference of diff between
			any strategy of diff > thresh, it is defined to be a strategy.
 		"""

		# set some field variables
		fields_id_no_strat = [x for x in self.fields_id if (x != self.key_strategy)]
		fields_order_strat_diffs = fields_id_no_strat + [self.field_variable, self.field_variable_trajgroup_type]
		fields_merge = [self.field_variable, self.field_variable_trajgroup_type] + fields_id_no_strat
		fields_ext = fields_merge + self.fields_time_periods

		# some strategy distinctions -- baseline vs. non- baseline
		strat_base = self.dict_baseline_ids.get(self.key_strategy)
		strats_not_base = [x for x in self.dict_id_values.get(self.key_strategy) if (x != strat_base)]

		# pivot by strategy--sort by fields_order_strat_diffs for use in dict_strategy_info
		df_pivot = pd.pivot(
			self.data_table,
			fields_merge,
			[self.key_strategy],
			self.fields_time_periods
		).sort_values(by = fields_order_strat_diffs)
		fields_base = [(x, strat_base) for x in self.fields_time_periods]

		# get the baseline strategy specification + set a renaming dictionary for merges - pivot inclues columns names as indices, they resturn when calling to_flat_index()
		df_base = df_pivot[fields_base].reset_index()
		df_base.columns = [x[0] for x in df_base.columns.to_flat_index()]
		arr_base = np.array(df_pivot[fields_base])

		dict_out = {
			"baseline_strategy_data_table": df_base,
			"baseline_strategy_array": arr_base,
			"difference_arrays_by_strategy": {}
		}
		dict_diffs = {}
		strategy_q = False

		for strat in strats_not_base:

			arr_cur = np.array(df_pivot[[(x, strat) for x in self.fields_time_periods]])
			arr_diff = arr_cur - arr_base
			dict_diffs.update({strat: arr_diff})

			strategy_q = (max(np.abs(arr_diff.flatten())) > thresh) | strategy_q


		dict_out.update({"difference_arrays_by_strategy": dict_diffs}) if strategy_q else None
		type_out = "L" if strategy_q else "X"

		# set properties
		self.dict_strategy_info = dict_out
		self.fields_order_strat_diffs = fields_order_strat_diffs
		self.xl_type = type_out



	############################
	#    CORE FUNCTIONALITY    #
	############################

	def get_scalar_diff_arrays(self,
	) ->  None:
		"""
		Get the scalar difference arrays, which are arrays that are scaled
			and added to the baseline to represent changes in future
			trajectries. Sets the following properties:

			* self.
		"""

		tgs_loops = self.required_tg_specs if (self.variable_trajectory_group is not None) else [None]
		tp_end = self.fields_time_periods[-1]
		dict_out = {}

		for vs in self.variable_specifications:

			for tgs in tgs_loops:

				# get the vector (dim is by scenario, sorted by self.fields_id )
				vec_tp_end = self.ordered_trajectory_arrays.get((vs, tgs)).get("data")[:,-1]
				tups_id_coords = [tuple(x) for x in np.array(self.ordered_trajectory_arrays.get((vs, tgs))["id_coordinates"])]

				# order the max/min scalars
				vec_scale_max = np.array([self.dict_variable_info[(vs, tgs)]["max_scalar"][x] for x in tups_id_coords])
				vec_scale_min = np.array([self.dict_variable_info[(vs, tgs)]["min_scalar"][x] for x in tups_id_coords])

				# difference, in final time period, between scaled value and baseline value-dimension is # of scenarios
				dict_tp_end_delta = {
					"max_tp_end_delta": vec_tp_end*(vec_scale_max - 1),
					"min_tp_end_delta": vec_tp_end*(vec_scale_min - 1)
				}

				dict_out.update({(vs, tgs): dict_tp_end_delta})

		return dict_out



	def mix_tensors(self,
		vec_b0: np.ndarray,
		vec_b1: np.ndarray,
		vec_mix: np.ndarray,
		constraints_mix: tuple = (0, 1)
	) -> np.ndarray:

		v_0 = np.array(vec_b0)
		v_1 = np.array(vec_b1)
		v_m = np.array(vec_mix)

		if constraints_mix != None:
			if constraints_mix[0] >= constraints_mix[1]:
				raise ValueError("Constraints to the mixing vector should be passed as (min, max)")
			v_alpha = v_m.clip(*constraints_mix)
		else:
			v_alpha = np.array(vec_mix)

		if len(v_alpha.shape) == 0:
			v_alpha = float(v_alpha)
			check_val = len(set([v_0.shape, v_1.shape]))
		else:
			check_val = len(set([v_0.shape, v_1.shape, v_alpha.shape]))

		if check_val > 1:
			raise ValueError("Incongruent shapes in mix_tensors")

		return v_0*(1 - v_alpha) + v_1*v_alpha



	def ordered_by_ota_from_fid_dict(self, dict_in: dict, key_tuple: tuple):
		return np.array([dict_in[tuple(x)] for x in np.array(self.ordered_trajectory_arrays[key_tuple]["id_coordinates"])])


	## UNCERTAINY FAN FUNCTIONS

	 # construct the "ramp" vector for uncertainties
	def build_ramp_vector(self,
		tuple_param: Union[tuple, None] = None
	) -> np.ndarray:
		"""
		Convert tuple_param to a vector for characterizing uncertainty.

		Keyword Arguments
		-----------------
		- tuple_param: tuple of parameters to pass to f_fan

		"""
		tuple_param = self.get_f_fan_function_parameter_defaults(self.uncertainty_fan_function_type) if (tuple_param is None) else tuple_param

		if len(tuple_param) == 4:
			tp_0 = self.time_period_end_certainty
			n = len(self.time_periods) - tp_0 - 1

			return np.array([int(i > tp_0)*self.f_fan(i - tp_0 , n, *tuple_param) for i in range(len(self.time_periods))])
		else:
			raise ValueError(f"Error: tuple_param {tuple_param} in build_ramp_vector has invalid length. It should have 4 parameters.")


	# basic function that determines the shape; based on a generalization of the sigmoid (includes linear option)
	def f_fan(self, x, n, a, b, c, d):
		"""
		 *defaults*

		 for linear:
			set a = 0, b = 2, c = 1, d = n/2
		 for sigmoid:
			set a = 1, b = 0, c = math.e, d = n/2


		"""
		return (a*n + b*x)/(n*(1 + c**(d - x)))


	# parameter defaults for the fan, based on the number of periods n
	def get_f_fan_function_parameter_defaults(self,
		n: int,
		fan_type: str,
		return_type: str = "params"
	) -> list:

		dict_ret = {
			"linear": (0, 2, 1, n/2),
			"sigmoid": (1, 0, math.e, n/2)
		}

		if return_type == "params":
			return dict_ret.get(fan_type)
		elif return_type == "keys":
			return list(dict_ret.keys())
		else:
			str_avail_keys = ", ".join(list(dict_ret.keys()))
			raise ValueError(f"Error: invalid return_type '{return_type}'. Ensure it is one of the following: {str_avail_keys}.")



	# verify fan function parameters
	def _initialize_uncertainty_functional_form(self,
		fan_type: Union[str, Tuple[Union[float, int]]],
		default_fan_type: str = "linear"
	) -> None:
		"""
		Set function parameters surrounding fan function. Sets the following
			properties:

			* self.uncertainty_fan_function_parameters
			* self.uncertainty_fan_function_type
			* self.uncertainty_ramp_vector
			* self.valid_fan_type_strs

		Behavioral Notes
		----------------
		- Invalid types for fan_type will result in the default_fan_type.
		- The dead default (if default_fan_type is invalid as a keyword) is
			linear

		Function Arguments
		------------------
		- fan_type: string specifying fan type OR tuple (4 values) specifying
			arguments to

		Keyword Arguments
		-----------------
		- default_fan_type: default fan function to use to describe uncertainty
		"""

		self.uncertainty_fan_function_parameters = None
		self.uncertainty_fan_function_type = None
		self.uncertainty_ramp_vector = None
		self.valid_fan_type_strs = self.get_f_fan_function_parameter_defaults(0, "", return_type = "keys")

		default_fan_type = "linear" if (default_fan_type not in self.valid_fan_type_strs) else default_fan_type
		n_uncertain = len(self.time_periods) - self.time_period_end_certainty
		params = None

		# set to default if an invalid type is entered
		if not (isinstance(fan_type, str) or isinstance(fan_type, typle)):
			fan_type = default_fan_type

		if isinstance(fan_type, tuple):
			if len(fan_type) != 4:
				#raise ValueError(f"Error: fan parameter specification {fan_type} invalid. 4 Parameters are required.")
				fan_type = default_fan_type
			elif not all(set([isinstance(x, int) or isinstance(x, float) for x in fan_type])):
				#raise ValueError(f"Error: fan parameter specification {fan_type} contains invalid parameters. Ensure they are numeric (int or float)")
				fan_type = default_fan_type
			else:
				fan_type = "custom"
				params = fan_type

		# only implemented if not otherwise set
		if params is None:
			fan_type = default_fan_type if (fan_type not in self.valid_fan_type_strs) else fan_type
			params = self.get_f_fan_function_parameter_defaults(n_uncertain, fan_type, return_type = "params")

		# build ramp vector
		tp_0 = self.time_period_end_certainty
		n = n_uncertain - 1
		vector_ramp = np.array([int(i > tp_0)*self.f_fan(i - tp_0 , n, *params) for i in range(len(self.time_periods))])

		#
		self.uncertainty_fan_function_parameters = params
		self.uncertainty_fan_function_type = fan_type
		self.uncertainty_ramp_vector = vector_ramp



	def generate_future(self,
		lhs_trial_x: float,
		lhs_trial_l: float = 1.0,
		baseline_future_q: bool = False,
		constraints_mix_tg: tuple = (0, 1),
		flatten_output_array: bool = False,
		vary_q: Union[bool, None] = None
	) -> Dict[str, np.ndarray]:
		"""
		Generate a dictionary mapping each variable specification to futures ordered by self.ordered_trajectory_arrays((vs, tg))["id_coordinates"]

		Function Arguments
		------------------
		- lhs_trial_x: LHS trial used to generate uncertainty fan for base future

		Keyword Arguments
		------------------
		- lhs_trial_l: LHS trial used to modify strategy effect
		- baseline_future_q: generate a baseline future? If so, lhs trials do not apply
		- constraints_mix_tg: constraints on the mixing fraction for trajectory groups
		- flatten_output_array: return a flattened output array (apply np.flatten())
		- vary_q: does the future vary? if not, returns baseline
		"""

		vary_q = self.variable_trajectory_group_vary_q if not isinstance(vary_q, bool) else vary_q

		# clean up some cases for None entries
		baseline_future_q = True if (lhs_trial_x is None) else baseline_future_q
		lhs_trial_x = 1.0 if (lhs_trial_x is None) else lhs_trial_x
		lhs_trial_l = 1.0 if (lhs_trial_l is None) else lhs_trial_l

		# some additional checks for potential negative numbers
		baseline_future_q = True if (lhs_trial_x < 0) else baseline_future_q
		lhs_trial_x = 1.0 if (lhs_trial_x < 0) else lhs_trial_x
		lhs_trial_l = 1.0 if (lhs_trial_l < 0) else lhs_trial_l

		# set to baseline if not varying
		baseline_future_q = baseline_future_q | (not vary_q)

		# initialization
		all_strats = self.dict_id_values.get(self.key_strategy)
		n_strat = len(all_strats)
		strat_base = self.dict_baseline_ids.get(self.key_strategy)

		# index by variable_specification at keys
		dict_out = {}

		if self.variable_trajectory_group is not None:
			#list(set([x[0] for x in self.ordered_trajectory_arrays.keys()]))
			cat_mix = self.dict_required_tg_spec_fields.get("mixing_trajectory")
			cat_b0 = self.dict_required_tg_spec_fields.get("trajectory_boundary_0")
			cat_b1 = self.dict_required_tg_spec_fields.get("trajectory_boundary_1")

			# use mix between 0/1 (0 = 100% trajectory_boundary_0, 1 = 100% trajectory_boundary_1)
			for vs in self.variable_specifications:

				dict_ordered_traj_arrays = self.ordered_trajectory_arrays.get((vs, None))
				dict_scalar_diff_arrays = self.scalar_diff_arrays.get((vs, None))
				dict_var_info = self.dict_variable_info.get((vs, None))

				dict_arrs = {
					cat_b0: self.ordered_trajectory_arrays[(vs, cat_b0)].get("data"),
					cat_b1: self.ordered_trajectory_arrays[(vs, cat_b1)].get("data"),
					cat_mix: self.ordered_trajectory_arrays[(vs, cat_mix)].get("data")
				}

				# for trajectory groups, the baseline is the specified mixing vector
				mixer = dict_arrs[cat_mix] if baseline_future_q else lhs_trial_x
				arr_out = self.mix_tensors(dict_arrs[cat_b0], dict_arrs[cat_b1], mixer, constraints_mix_tg)

				if self.xl_type == "L":
					#
					# if the XL is an L, then we use the modified future as a base (reduce to include only baseline strategy), then add the uncertainty around the strategy effect
					#
					# get id coordinates( any of cat_mix, cat_b0, or cat_b1 would work -- use cat_mix)
					df_ids_ota = pd.concat([
						self.ordered_trajectory_arrays.get((vs, cat_mix))["id_coordinates"].copy().reset_index(drop = True),
						pd.DataFrame(arr_out, columns = self.fields_time_periods)],
						axis = 1
					)
					w = np.where(df_ids_ota[self.key_strategy] == strat_base)
					df_ids_ota = df_ids_ota.iloc[w[0].repeat(n_strat)].reset_index(drop = True)

					arr_out = np.array(df_ids_ota[self.fields_time_periods])
					arrs_strategy_diffs = self.dict_strategy_info.get("difference_arrays_by_strategy")
					df_baseline_strategy = self.dict_strategy_info.get("baseline_strategy_data_table")
					inds0 = set(np.where(df_baseline_strategy[self.field_variable] == vs)[0])
					l_modified_cats = []

					for cat_cur in [cat_b0, cat_b1, cat_mix]:

						# get the index for the current vs/cat_cur
						inds = np.sort(np.array(list(inds0 & set(np.where(df_baseline_strategy[self.field_variable_trajgroup_type] == cat_cur)[0]))))
						n_inds = len(inds)
						df_ids0 = df_baseline_strategy[[x for x in self.fields_id if (x != self.key_strategy)]].loc[inds.repeat(n_strat)].reset_index(drop = True)
						new_strats = list(np.zeros(len(df_ids0)).astype(int))

						# initialize as list - we only do this to guarantee the sort is correct
						df_future_strat = np.zeros((n_inds*n_strat, len(self.fields_time_periods)))
						ind_repl = 0

						# iterate over strategies
						for strat in all_strats:
							# replace strategy ids
							new_strats[ind_repl*n_inds:((ind_repl + 1)*n_inds)] = [strat for x in inds]
							# get the strategy difference that is adjusted by lhs_trial_x_delta; if baseline strategy, use 0s
							df_repl = np.zeros((n_inds, len(self.fields_time_periods))) if (strat == strat_base) else arrs_strategy_diffs[strat][inds, :]*lhs_trial_l
							np.put(
								df_future_strat,
								range(
									n_inds*len(self.fields_time_periods)*ind_repl,
									n_inds*len(self.fields_time_periods)*(ind_repl + 1)
								),
								df_repl
							)
							ind_repl += 1

						df_ids0[self.key_strategy] = new_strats
						df_future_strat = pd.concat([df_ids0, pd.DataFrame(df_future_strat, columns = self.fields_time_periods)], axis = 1).sort_values(by = self.fields_id).reset_index(drop = True)
						l_modified_cats.append(dict_arrs[cat_cur] + np.array(df_future_strat[self.fields_time_periods]))

					arr_out = self.mix_tensors(*l_modified_cats, constraints_mix_tg)

				# to compare the difference between the "L" design uncertainty and the baseline and add this to the uncertain future (final array)
				arr_out = arr_out.flatten() if flatten_output_array else arr_out
				dict_out.update({vs: arr_out})


		else:

			rv = self.uncertainty_ramp_vector

			for vs in self.variable_specifications:

				dict_ordered_traj_arrays = self.ordered_trajectory_arrays.get((vs, None))
				dict_scalar_diff_arrays = self.scalar_diff_arrays.get((vs, None))
				dict_var_info = self.dict_variable_info.get((vs, None))

				# order the uniform scaling by the ordered trajectory arrays
				vec_unif_scalar = self.ordered_by_ota_from_fid_dict(dict_var_info["uniform_scaling_q"], (vs, None))
				# gives 1s where we keep standard fanning (using the ramp vector) and 0s where we use uniform scaling
				vec_base = 1 - vec_unif_scalar
				#
				if max(vec_unif_scalar) > 0:
					vec_max_scalar = self.ordered_by_ota_from_fid_dict(dict_var_info["max_scalar"], (vs, None))
					vec_min_scalar = self.ordered_by_ota_from_fid_dict(dict_var_info["min_scalar"], (vs, None))
					vec_unif_scalar = vec_unif_scalar*(vec_min_scalar + lhs_trial_x*(vec_max_scalar - vec_min_scalar)) if not baseline_future_q else np.ones(vec_unif_scalar.shape)

				vec_unif_scalar = np.array([vec_unif_scalar]).transpose()
				vec_base = np.array([vec_base]).transpose()

				delta_max = dict_scalar_diff_arrays.get("max_tp_end_delta")
				delta_min = dict_scalar_diff_arrays.get("min_tp_end_delta")
				delta_diff = delta_max - delta_min
				delta_val = delta_min + lhs_trial_x*delta_diff

				# delta and uniform scalar don't apply if operating under baseline future
				delta_vec = 0.0 if baseline_future_q else (rv * np.array([delta_val]).transpose())

				arr_out = dict_ordered_traj_arrays.get("data") + delta_vec
				arr_out = arr_out*vec_base + vec_unif_scalar*dict_ordered_traj_arrays.get("data")

				if self.xl_type == "L":
					# get series of strategies
					series_strats = dict_ordered_traj_arrays.get("id_coordinates")[self.key_strategy]
					w = np.where(np.array(series_strats) == strat_base)[0]
					# get strategy adjustments
					lhs_mult_deltas = 1.0 if baseline_future_q else lhs_trial_l
					array_strat_deltas = np.concatenate(
						series_strats.apply(
							self.dict_strategy_info["difference_arrays_by_strategy"].get,
							args = (np.zeros((1, len(self.time_periods))), )
						)
					)*lhs_mult_deltas

					arr_out = (array_strat_deltas + arr_out[w, :]) if (len(w) > 0) else arr_out

				arr_out = arr_out.flatten() if flatten_output_array else arr_out
				dict_out.update({vs: arr_out})


		return dict_out






class FutureTrajectories:

	"""
	Create a collection of SampleUnit objects to use to generate futures.

	Initialization Arguments
	------------------------
	- df_input_database: DataFrame to use as database of baseline inputs (across strategies)
	- dict_baseline_ids: dictionary mapping a string of a baseline id field to a baseline id value (integer)
	- time_period_u0: first time period with uncertainty

	Keyword Arguments
	-----------------
	- dict_all_dims: optional dictionary defining all values associated with
		keys in dict_baseline_ids to pass to each SamplingUnit. If None
		(default), infers from df_input_database. Takes the form
		{
			index_0: [id_val_00, id_val_01,... ],
			index_1: [id_val_10, id_val_11,... ],
			.
			.
			.
		}
	- fan_function_specification: type of uncertainty approach to use
		* linear: linear ramp to time time T - 1
		* sigmoid: sigmoid function that ramps to time T - 1
	- field_sample_unit_group: field used to identify sample unit groups. Sample unit groups are composed of:
		* individual variable specifications
		* trajectory groups
	- field_time_period: field used to specify the time period
	- field_uniform_scaling_q: field used to identify whether or not a variable
	- field_variable: field used to specify variables
	- field_variable_trajgroup: field used to identify the trajectory group (integer)
	- field_variable_trajgroup_type: field used to identify the trajectory group type (max, min, mix, or lhs)
	- fan_function_specification: type of uncertainty approach to use
		* linear: linear ramp to time time T - 1
		* sigmoid: sigmoid function that ramps to time T - 1
	- key_future: field used to identify the future
	- key_strategy: field used to identify the strategy (int)
		* This field is important as uncertainty in strategies is assessed differently than uncetainty in other variables
	- logger: optional logging.Logger object used to track generation of futures
	- regex_id: regular expression used to identify id fields in the input template
	- regex_trajgroup: Regular expression used to identify trajectory group variables in `field_variable` of `df_input_database`
	- regex_trajmax: Regular expression used to identify trajectory maxima in variables and trajgroups specified in `field_variable` of `df_input_database`
	- regex_trajmin: Regular expression used to identify trajectory minima in variables and trajgroups specified in `field_variable` of `df_input_database`
	- regex_trajmix: Regular expression used to identify trajectory baseline mix (fraction maxima) in variables and trajgroups specified in `field_variable` of `df_input_database`

	"""
	def __init__(self,
		df_input_database: pd.DataFrame,
		dict_baseline_ids: Dict[str, int],
		time_period_u0: int,
		dict_all_dims: Union[Dict[str, List[int]], None] = None,
		fan_function_specification: str = "linear",
		field_sample_unit_group: str = "sample_unit_group",
		field_time_period: str = "time_period",
		field_uniform_scaling_q: str = "uniform_scaling_q",
		field_trajgroup_no_vary_q: str = "trajgroup_no_vary_q",
		field_variable: str = "variable",
		field_variable_trajgroup: str = "variable_trajectory_group",
		field_variable_trajgroup_type: str = "variable_trajectory_group_trajectory_type",
		key_future: str = "future_id",
		key_strategy: str = "strategy_id",
		# optional logger
		logger: Union[logging.Logger, None] = None,
		# regular expressions used to define trajectory group components in input database and
		regex_id: re.Pattern = re.compile("(\D*)_id$"),
		regex_max: re.Pattern = re.compile("max_(\d*$)"),
		regex_min: re.Pattern = re.compile("min_(\d*$)"),
		regex_tp: re.Pattern = re.compile("(\d*$)"),
		regex_trajgroup: re.Pattern = re.compile("trajgroup_(\d*)-(\D*$)"),
		regex_trajmax: re.Pattern = re.compile("trajmax_(\D*$)"),
		regex_trajmin: re.Pattern = re.compile("trajmin_(\D*$)"),
		regex_trajmix: re.Pattern = re.compile("trajmix_(\D*$)"),
		# some internal vars
		specification_tgt_lhs: str = "lhs",
		specification_tgt_max: str = "trajectory_boundary_1",
		specification_tgt_min: str = "trajectory_boundary_0"
	):

		##  INITIALIZE PARAMETERS

		# dictionary of baseline ids and fan function
		self.dict_baseline_ids = dict_baseline_ids
		self.fan_function_specification = fan_function_specification
		# set default fields
		self.key_future = key_future
		self.field_sample_unit_group = field_sample_unit_group
		self.key_strategy = key_strategy
		self.field_time_period = field_time_period
		self.field_trajgroup_no_vary_q = field_trajgroup_no_vary_q
		self.field_uniform_scaling_q = field_uniform_scaling_q
		self.field_variable = field_variable
		self.field_variable_trajgroup = field_variable_trajgroup
		self.field_variable_trajgroup_type = field_variable_trajgroup_type
		# logging.Logger
		self.logger = logger
		# missing values flag
		self.missing_flag_int = -999
		# default regular expressions
		self.regex_id = regex_id
		self.regex_max = regex_max
		self.regex_min = regex_min
		self.regex_tp = regex_tp
		self.regex_trajgroup = regex_trajgroup
		self.regex_trajmax = regex_trajmax
		self.regex_trajmin = regex_trajmin
		self.regex_trajmix = regex_trajmix
		# some default internal specifications used in templates
		self.specification_tgt_lhs = specification_tgt_lhs
		self.specification_tgt_max = specification_tgt_max
		self.specification_tgt_min = specification_tgt_min
		# first period with uncertainty
		self.time_period_u0 = time_period_u0


		##  KEY INITIALIZATIONS

		self._initialize_input_database(df_input_database)
		self._initialize_dict_all_dims(dict_all_dims)
		self._initialize_sampling_units()
		self._set_xl_sampling_units()



	###########################################################
	#	SOME BASIC INITIALIZATIONS AND INTERNAL FUNCTIONS	#
	###########################################################

	def _initialize_dict_all_dims(self,
		dict_all_dims: Union[Dict, None]
	) -> None:
		"""
		Initialize the dictionary of all dimensional values to accomodate
			for each sampling unit--ensures that each SamplingUnit has the
			same dimensional values (either strategy or discrete baselines).
			Sets the following properties:

			* self.dict_all_dimensional_values

		Function Arguments
		------------------
		- dict_all_dims: dictionary of all dimensional values to preserve.
			Takes the form

			{
				index_0: [id_val_00, id_val_01,... ],
				index_1: [id_val_10, id_val_11,... ],
				.
				.
				.
			}
		"""

		dict_all_dims_out = {}
		self.dict_all_dimensional_values = None

		# infer from input database if undefined
		if dict_all_dims is None:
			for k in self.dict_baseline_ids:
				dict_all_dims_out.update({
					k: sorted(list(self.input_database[k].unique()))
				})

		else:
			# check that each dimension is defined in the baseline ids
			for k in dict_all_dims.keys():
				if k in self.dict_baseline_ids.keys():
					dict_all_dims_out.update({k: dict_all_dims.get(k)})

		self.dict_all_dimensional_values = dict_all_dims_out



	def _log(self,
		msg: str,
		type_log: str = "log",
		**kwargs
	):
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



	def _set_xl_sampling_units(self,
	) -> None:
		"""
		Determine X/L sampling units--sets three properties:

		- all_sampling_units_l
		- all_sampling_units_x
		- dict_sampling_unit_to_xl_type
		"""
		all_sampling_units_l = []
		all_sampling_units_x = []
		dict_sampling_unit_to_xl_type = {}

		for k in self.dict_sampling_units.keys():
			xl_type = self.dict_sampling_units.get(k).xl_type
			dict_sampling_unit_to_xl_type.update({k: xl_type})
			all_sampling_units_x.append(k) if (xl_type == "X") else all_sampling_units_l.append(k)

		self.all_sampling_units_l = all_sampling_units_l
		self.all_sampling_units_x = all_sampling_units_x
		self.dict_sampling_unit_to_xl_type = dict_sampling_unit_to_xl_type



	def get_df_row_element(self,
		row: Union[pd.Series, pd.DataFrame, None],
		index: Union[int, float, str],
		return_def: Union[int, float, str, None] = None
	) -> float:
		"""
		Support for self.generate_future_from_lhs_vector. Read an element from a named series or DataFrame.

		Function Arguments
		------------------
		- row: Series or DataFrame. If DataFrame, only reads the first row
		- index: column index in input DataFrame to read (field)

		Keyword Arguments
		-----------------
		- return_def: default return value if row is None
		"""
		out = return_def
		if isinstance(row, pd.DataFrame):
			out = float(row[index].iloc[0]) if (index in row.columns) else out
		elif isinstance(row, pd.Series):
			out = float(row[index]) if (index in row.index) else out

		return out



	####################################
	#    PREPARE THE INPUT DATABASE    #
	####################################

	def clean_sampling_unit_input_df(self,
		df_su_input: pd.DataFrame,
		dict_all_dims: Union[Dict[str, List], None] = None,
		dict_baseline_ids: Union[Dict[str, int], None] = None,
		dict_expand_vars: Union[Dict[str, List[str]], None] = None,
		sample_unit_id: Any = None
	) -> pd.DataFrame:
		"""
		Prepare an input data frame for initializing SamplingUnit within
			FutureTrajectories. Ensures that all dimensions that are specified
			in the global database are defined in the Sampling Unit. Replaces
			missing dimensions with core baseline (e.g., (0, 0, 0)).

		Function Arguments
		------------------
		- df_su_input: input data frame to SamplingUnit
		- dict_all_dims: optional dictionary to pass that contains dimensions.
			This dictionary is used to determine which dimensions *should* be
			present. If any of the dimension sets specified in dict_all_dims
			are not found in df_in, their values are replaced with associated
			baselines.

		Keyword Arguments
		-----------------
		- dict_baseline_ids: dictionary mapping index fields to baseline
			values
		- dict_expand_vars: dictionary of variable specification components to
			expand. If none, expands along all uniquely defined values for
			self.field_variable_trajgroup_type and self.variable
		- sample_unit_id: optional id to pass for error troubleshooting
		"""

		dict_all_dims = self.dict_all_dimensional_values if (dict_all_dims is None) else dict_all_dims
		if not isinstance(dict_all_dims, dict):
			return df_su_input


		##  CHECK BASELINES DEFINED

		dict_baseline_ids = self.dict_baseline_ids if not isinstance(dict_baseline_ids, dict) else dict_baseline_ids
		df_su_base = sf.subset_df(
			df_su_input,
			dict_baseline_ids
		).drop_duplicates()

		# add expansion variables
		dict_expand_vars = {
			self.field_variable_trajgroup_type: list(df_su_input[self.field_variable_trajgroup_type].unique()),
			self.field_variable: list(df_su_input[self.field_variable].unique())
		} if (dict_expand_vars is None) else dict_expand_vars
		dict_all_dims.update(dict_expand_vars)

		n_req_baseline = np.prod([len(v) for v in dict_expand_vars.values()])
		if len(df_su_base) != n_req_baseline:
			sg = "" if (sample_unit_id is not None) else f" {sample_unit_id}"
			msg = f"Unable to initialize sample group{sg}: one or more variables and/or variable trajectory group types are missing. Check the input data frame."
			self._log(msg, type_log = "error")
			raise RuntimeError(msg)


		##  BUILD EXPANDED DATAFRAME, MERGE IN AVAILABLE, THEN FILL IN ROWS

		dims_expand = [x for x in df_su_input.columns if x in dict_all_dims.keys()]
		vals_expand = [dict_all_dims.get(x) for x in dims_expand]
		df_dims_req = pd.DataFrame(
			list(itertools.product(*vals_expand)),
			columns = dims_expand
		)
		df_dims_req = pd.merge(df_dims_req, df_su_input, how = "left")

		# get merge fields (expansion fields that exclude baseline ids) and subset fields (data fields to replace)
		fields_merge = list(dict_expand_vars.keys())
		fields_subset = [x for x in df_dims_req.columns if x not in dims_expand]
		df_out = sf.fill_df_rows_from_df(
			df_dims_req,
			df_su_base,
			fields_merge,
			fields_subset
		)

		return df_out[df_su_input.columns]



	def generate_future_from_lhs_vector(self,
		df_row_lhc_sample_x: Union[pd.Series, pd.DataFrame],
		df_row_lhc_sample_l: Union[pd.Series, pd.DataFrame, None] = None,
		future_id: Union[int, None] = None,
		baseline_future_q: bool = False,
		dict_optional_dimensions: Dict[str, int] = {}
	) -> pd.DataFrame:
		"""
		Build a data frame of a single future for all sample units

		Function Arguments
		------------------
		- df_row_lhc_sample_x: data frame row with column names as sample groups for all sample groups to vary with uncertainties

		Keyword Arguments
		-----------------
		- df_row_lhc_sample_l: data frame row with column names as sample groups for all sample groups to vary with uncertainties
			* If None, lhs_trial_l = 1 in all samples (constant strategy effect across all futures)
		- future_id: optional future id to add to the dataframe using self.future_id
		- baseline_future_q: generate the dataframe for the baseline future?
		- dict_optional_dimensions: dictionary of optional dimensions to pass to the output data frame (form: {key_dimension: id_value})
		"""

		# check the specification of
		if not (isinstance(df_row_lhc_sample_x, pd.DataFrame) or isinstance(df_row_lhc_sample_x, pd.Series) or (df_row_lhc_sample_x is None)):
			tp = str(type(df_row_lhc_sample_x))
			self._log(f"Invalid input type {tp} specified for df_row_lhc_sample_x in generate_future_from_lhs_vector: pandas Series or DataFrames (first row) are acceptable inputs. Returning baseline future.", type_log = "warning")
			df_row_lhc_sample_x = None

		# initialize outputs and iterate
		dict_df = {}
		df_out = []
		for k in enumerate(self.all_sampling_units):
			k, su = k
			samp = self.dict_sampling_units.get(su)

			if samp is not None:

				# get LHC samples for X and L
				lhs_x = self.get_df_row_element(df_row_lhc_sample_x, su)
				lhs_l = self.get_df_row_element(df_row_lhc_sample_l, su, 1.0)

				# note: if lhs_x is None, returns baseline future no matter what,
				dict_fut = samp.generate_future(
					lhs_x,
					lhs_l,
					baseline_future_q = baseline_future_q
				)

				dict_df.update(
					dict((key, value.flatten()) for key, value in dict_fut.items())
				)

				# initialize indexing if necessary
				if len(df_out) == 0:
					dict_fields = None if (future_id is None) else {self.key_future: future_id}
					df_out.append(
						samp.generate_indexing_data_frame(
							dict_additional_fields = dict_fields
						)
					)

		df_out.append(pd.DataFrame(dict_df))
		df_out = pd.concat(df_out, axis = 1).reset_index(drop = True)
		df_out = sf.add_data_frame_fields_from_dict(df_out, dict_optional_dimensions) if isinstance(dict_optional_dimensions, dict) else df_out

		return df_out



	def get_trajgroup_and_variable_specification(self,
		input_var_spec: str,
		regex_trajgroup: Union[re.Pattern, None] = None,
		regex_trajmax: Union[re.Pattern, None] = None,
		regex_trajmin: Union[re.Pattern, None] = None,
		regex_trajmix: Union[re.Pattern, None] = None
	) -> Tuple[str, str]:
		"""
		Derive a trajectory group and variable specification from variables in an input variable specification

		Function Arguments
		------------------
		- input_var_spec: variable specification string

		Keyword Arguments
		-----------------
		- regex_trajgroup: Regular expression used to identify trajectory group variables in `field_variable` of `df_input_database`
		- regex_trajmax: Regular expression used to identify trajectory maxima in variables and trajgroups specified in `field_variable` of `df_input_database`
		- regex_trajmin: Regular expression used to identify trajectory minima in variables and trajgroups specified in `field_variable` of `df_input_database`
		- regex_trajmix: Regular expression used to identify trajectory baseline mix (fraction maxima) in variables and trajgroups specified in `field_variable` of `df_input_database`
		"""
		input_var_spec = str(input_var_spec)
		regex_trajgroup = self.regex_trajgroup if (regex_trajgroup is None) else regex_trajgroup
		regex_trajmax = self.regex_trajmax if (regex_trajmax is None) else regex_trajmax
		regex_trajmin = self.regex_trajmin if (regex_trajmin is None) else regex_trajmin
		regex_trajmix = self.regex_trajmix if (regex_trajmix is None) else regex_trajmix

		tg = None
		var_spec = None

		# check trajgroup match
		trajgroup_match = regex_trajgroup.match(input_var_spec)
		if trajgroup_match is not None:
			tg = int(trajgroup_match.groups()[0])
			check_spec_string = str(trajgroup_match.groups()[1])
		else:
			check_spec_string = input_var_spec

		# check trajectory max/min/mix
		if regex_trajmax.match(check_spec_string) is not None:
			var_spec = "max"
		elif regex_trajmin.match(check_spec_string) is not None:
			var_spec = "min"
		elif regex_trajmix.match(check_spec_string) is not None:
			var_spec = "mix"
		elif (check_spec_string == "lhs") and (tg is not None):
			var_spec = "lhs"

		return (tg, var_spec)



	def _initialize_input_database(self,
		df_in: pd.DataFrame,
		field_sample_unit_group: Union[str, None] = None,
		field_variable: Union[str, None] = None,
		field_variable_trajgroup: Union[str, None] = None,
		field_variable_trajgroup_type: Union[str, None] = None,
		missing_trajgroup_flag: Union[int, None] = None,
		regex_trajgroup: Union[re.Pattern, None] = None,
		regex_trajmax: Union[re.Pattern, None] = None,
		regex_trajmin: Union[re.Pattern, None] = None,
		regex_trajmix: Union[re.Pattern, None] = None
	) -> None:
		"""
		Prepare the input database for sampling by adding sample unit group,
			cleaning up trajectory groups, etc. Sets the following properties:

			* self.input_database

		Function Arguments
		------------------
		- df_in: input database to use to generate SampleUnit objects

		Keyword Arguments
		-----------------
		- field_sample_unit_group: field used to identify groupings of sample units
		- field_variable: field in df_in used to denote the database
		- field_variable_trajgroup: field denoting the variable trajectory group
		- field_variable_trajgroup_type: field denoting the type of the variable within a variable trajectory group
		- missing_trajgroup_flag: missing flag for trajectory group values
		- regex_trajgroup: regular expression used to match trajectory group variable specifications
		- regex_trajmax: regular expression used to match the maximum trajectory component of a trajectory group variable element
		- regex_trajmin: regular expression used to match the minimum trajectory component of a trajectory group variable element
		- regex_trajmix: regular expression used to match the mixing component of a trajectory group variable element
		"""

		# input dataframe
		#df_in = self.df_input_database if (df_in is None) else df_in

		# key fields
		field_sample_unit_group = self.field_sample_unit_group if (field_sample_unit_group is None) else field_sample_unit_group
		field_variable = self.field_variable if (field_variable is None) else field_variable
		field_variable_trajgroup = self.field_variable_trajgroup if (field_variable_trajgroup is None) else field_variable_trajgroup
		field_variable_trajgroup_type = self.field_variable_trajgroup_type if (field_variable_trajgroup_type is None) else field_variable_trajgroup_type
		# set the missing flag
		missing_flag = self.missing_flag_int if (missing_trajgroup_flag is None) else int(missing_trajgroup_flag)
		# regular expressions
		regex_trajgroup = self.regex_trajgroup if (regex_trajgroup is None) else regex_trajgroup
		regex_trajmax = self.regex_trajmax if (regex_trajmax is None) else regex_trajmax
		regex_trajmin = self.regex_trajmin if (regex_trajmin is None) else regex_trajmin
		regex_trajmix = self.regex_trajmix if (regex_trajmix is None) else regex_trajmix

		##  split traj groups
		new_col_tg = []
		new_col_spec_type = []

		# split out traj group and variable specification
		df_add = df_in[field_variable].apply(
			self.get_trajgroup_and_variable_specification,
			args = (regex_trajgroup,
			regex_trajmax,
			regex_trajmin,
			regex_trajmix)
		)

		# add the variable trajectory group
		df_add = pd.DataFrame([np.array(x) for x in df_add], columns = [field_variable_trajgroup, field_variable_trajgroup_type])
		df_add[field_variable_trajgroup] = df_add[field_variable_trajgroup].replace({None: missing_flag})
		df_add[field_variable_trajgroup] = df_add[field_variable_trajgroup].astype(int)
		df_in = pd.concat([
				df_in.drop([field_variable_trajgroup, field_variable_trajgroup_type], axis = 1),
				df_add
			],
			axis = 1
		)

		##  update trajgroups to add dummies
		new_tg = df_in[df_in[field_variable_trajgroup] >= 0][field_variable_trajgroup]
		new_tg = 1 if (len(new_tg) == 0) else max(np.array(new_tg)) + 1
		tgs = list(df_in[field_variable_trajgroup].copy())
		tgspecs = list(df_in[field_variable_trajgroup_type].copy())

		# initialization outside of the iteration
		var_list = list(df_in[field_variable].copy())
		dict_parameter_to_tg = {}
		dict_repl_tgt = {"max": self.specification_tgt_max, "min": self.specification_tgt_min}

		for i in range(len(df_in)):
			# get trajgroup, trajgroup type, and variable specification for current row
			tg = int(df_in[field_variable_trajgroup].iloc[i])
			tgspec = str(df_in[field_variable_trajgroup_type].iloc[i])
			vs = str(df_in[field_variable].iloc[i])

			if (tgspec != "<NA>") & (tgspec != "None"):
				new_tg_q = True

				if tg > 0:
					# drop the group/remove the trajmax/min/mix
					vs = regex_trajgroup.match(vs).groups()[0]
					new_tg_q = False

				# check for current trajgroup type
				for regex in [regex_trajmax, regex_trajmin, regex_trajmix]:
					matchstr = regex.match(vs)
					vs = matchstr.groups()[0] if (matchstr is not None) else vs

				# update the variable list
				var_list[i] = vs

				# update indexing of trajectory groups
				if new_tg_q:
					if vs in dict_parameter_to_tg.keys():
						tgs[i] = int(dict_parameter_to_tg.get(vs))
					else:
						dict_parameter_to_tg.update({vs: new_tg})
						tgs[i] = new_tg
						new_tg += 1

		# update outputs
		df_in[field_variable] = var_list
		df_in[field_variable_trajgroup] = tgs
		df_in = df_in[~df_in[field_variable].isin([self.specification_tgt_lhs])].reset_index(drop = True)
		df_in[field_variable_trajgroup_type].replace(dict_repl_tgt, inplace = True)

		# add sample_unit_group field
		dict_var_to_su = sf.build_dict(df_in[df_in[field_variable_trajgroup] > 0][[field_variable, field_variable_trajgroup]].drop_duplicates())
		vec_vars_to_assign = sorted(list(set(df_in[df_in[field_variable_trajgroup] <= 0][field_variable])))
		min_val = (max(dict_var_to_su.values()) + 1) if (len(dict_var_to_su) > 0) else 1
		dict_var_to_su.update(
			dict(zip(
				vec_vars_to_assign,
				list(range(min_val, min_val + len(vec_vars_to_assign)))
			))
		)
		df_in[field_sample_unit_group] = df_in[field_variable].replace(dict_var_to_su)

		self.input_database = df_in



	def _initialize_sampling_units(self,
		df_in: Union[pd.DataFrame, None] = None,
		dict_all_dims: Union[Dict[str, List[int]], None] = None,
		fan_function: Union[str, None] = None,
		**kwargs
	) -> None:
		"""
		Instantiate all defined SamplingUnits from input database. Sets the
			following properties:

			* self.n_su
			* self.all_sampling_units
			* self.dict_sampling_units

		Behavioral Notes
		----------------
		- _initialize_sampling_units() will try to identify the availablity of
			dimensions specified in dict_all_dims within df_in. If a dimension
			specified in dict_all_dims is not found within df_in, the funciton
			will replace the value with the baseline strategy.
		- If a product specified within self.dict_baseline_ids is missing in
			the dataframe, then an error will occur.

		Keword Arguments
		-----------------
		- df_in: input database used to identify sampling units. Must include
			self.field_sample_unit_group
		- dict_all_dims: optional dictionary to pass that contains dimensions.
			This dictionary is used to determine which dimensions *should* be
			present. If any of the dimension sets specified in dict_all_dims
			are not found in df_in, their values are replaced with associated
			baselines.
		- fan_function: function specification to use for uncertainty fans
		- **kwargs: passed to SamplingUnit initialization
		"""

		# get some defaults
		df_in = self.input_database if (df_in is None) else df_in
		fan_function = self.fan_function_specification if (fan_function is None) else fan_function
		dict_all_dims = self.dict_all_dimensional_values if not isinstance(dict_all_dims, dict) else dict_all_dims

		# setup inputs
		kwarg_keys = list(kwargs.keys())
		field_time_period = self.field_time_period if ("field_time_period" not in kwarg_keys) else kwargs.get("field_time_period")
		field_uniform_scaling_q = self.field_uniform_scaling_q if ("field_uniform_scaling_q" not in kwarg_keys) else kwargs.get("field_uniform_scaling_q")
		field_trajgroup_no_vary_q = self.field_trajgroup_no_vary_q if ("field_trajgroup_no_vary_q" not in kwarg_keys) else kwargs.get("field_trajgroup_no_vary_q")
		field_variable_trajgroup = self.field_variable_trajgroup if ("field_variable_trajgroup" not in kwarg_keys) else kwargs.get("field_variable_trajgroup")
		field_variable_trajgroup_type = self.field_variable_trajgroup_type if ("field_variable_trajgroup_type" not in kwarg_keys) else kwargs.get("field_variable_trajgroup_type")
		field_variable = self.field_variable if ("field_variable" not in kwarg_keys) else kwargs.get("field_variable")
		key_strategy = self.key_strategy if ("key_strategy" not in kwarg_keys) else kwargs.get("key_strategy")
		regex_id = self.regex_id if ("regex_id" not in kwarg_keys) else kwargs.get("regex_id")
		regex_max = self.regex_max if ("regex_max" not in kwarg_keys) else kwargs.get("regex_max")
		regex_min = self.regex_min if ("regex_min" not in kwarg_keys) else kwargs.get("regex_min")
		regex_tp = self.regex_tp if ("regex_tp" not in kwarg_keys) else kwargs.get("regex_tp")

		dict_sampling_units = {}

		dfgroup_sg = df_in.groupby(self.field_sample_unit_group)
		all_sample_groups = sorted(list(set(df_in[self.field_sample_unit_group])))
		n_sg = len(dfgroup_sg)
		if isinstance(dict_all_dims, dict):
			dict_all_dims = dict((k, v) for k, v in dict_all_dims.items() if k in self.dict_baseline_ids.keys())


		##  GENERATE SamplingUnit FROM DATABASE

		self._log(f"Instantiating {n_sg} sampling units.", type_log = "info")
		t0 = time.time()

		for iterate in enumerate(dfgroup_sg):

			i, df_sg = iterate
			df_sg = df_sg[1]
			sg = int(df_sg[self.field_sample_unit_group].iloc[0])

			# fill in missing values from baseline if dims are missing in input database
			df_sg = self.clean_sampling_unit_input_df(df_sg, dict_all_dims = dict_all_dims) if (dict_all_dims is not None) else df_sg

			samp = SamplingUnit(
				df_sg.drop(self.field_sample_unit_group, axis = 1),
				self.dict_baseline_ids,
				self.time_period_u0,
				fan_function_specification = fan_function,
				field_time_period = field_time_period,
				field_uniform_scaling_q = field_uniform_scaling_q,
				field_trajgroup_no_vary_q = field_trajgroup_no_vary_q,
				field_variable_trajgroup = field_variable_trajgroup,
				field_variable_trajgroup_type = field_variable_trajgroup_type,
				field_variable = field_variable,
				key_strategy = key_strategy,
				missing_trajgroup_flag = self.missing_flag_int,
				regex_id = regex_id,
				regex_max = regex_max,
				regex_min = regex_min,
				regex_tp = regex_tp
			)

			dict_sampling_units = dict(zip(all_sample_groups, [samp for x in range(n_sg)])) if (i == 0) else dict_sampling_units
			dict_sampling_units.update({sg: samp})

			self._log(f"Iteration {i} complete.", type_log = "info") if (i%250 == 0) else None

		t_elapse = sf.get_time_elapsed(t0)
		self._log(f"\t{n_sg} sampling units complete in {t_elapse} seconds.", type_log = "info")

		self.n_su = n_sg
		self.all_sampling_units = all_sample_groups
		self.dict_sampling_units = dict_sampling_units
