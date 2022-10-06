import math
import numpy as np
import pandas as pd
import os, os.path
import re
import support_functions as sf
from typing import *



##  class for sampling and generating experimental design
class SamplingUnit:
	"""
	Generate future trajectories based on an input database.

	Initialization Arguments
	------------------------
	- df_variable_definition: DataFrame to
	- dict_baseline_ids: dictionary mapping a string of a baseline id field to a baseline id value (integer)
	- time_period_u0: first time period with uncertainty

	Keyword Arguments
	-----------------
	- fan_function_specification: type of uncertainty approach to use
		* linear: linear ramp to time time T - 1
		* sigmoid: sigmoid function that ramps to time T - 1
	- field_strategy_id: field used to identify the strategy (int)
		* This field is important as uncertainty in strategies is assessed differently than uncetainty in other variables
	- field_time_period: field used to denote the time period
	- field_uniform_scaling_q: field used to identify whether or not a variable
	- field_variable: field used to specify variables
	- field_variable_trajgroup: field used to identify the trajectory group (integer)
	- field_variable_trajgroup_type: field used to identify the trajectory group type (max, min, mix, or lhs)
	- missing_flag_trajgroup: flag used to identify null trajgroups (default is -999)
	"""
	def __init__(self,
		df_variable_definition: pd.DataFrame,
		dict_baseline_ids: Dict[str, int],
		time_period_u0: int,
		fan_function_specification: str = "linear",
		field_strategy_id: str = "strategy_id",
		field_time_period: str = "time_period",
		field_uniform_scaling_q: str = "uniform_scaling_q",
		field_variable_trajgroup: str = "variable_trajectory_group",
		field_variable_trajgroup_type: str = "variable_trajectory_group_trajectory_type",
		field_variable: str = "variable",
		missing_trajgroup_flag: int = -999
	):

		##  set some attributes

		# from function args
		self.field_strategy_id = field_strategy_id
		self.field_time_period = field_time_period
		self.field_uniform_scaling_q = field_uniform_scaling_q
		self.field_variable_trajgroup = field_variable_trajgroup
		self.field_variable_trajgroup_type = field_variable_trajgroup_type
		self.field_variable = field_variable
		self.missing_trajgroup_flag = missing_trajgroup_flag
		self.time_period_end_certainty = self.check_time_start_uncertainty(time_period_u0)
		# others
		self.set_parameters()

		# derive additional attributes
		self.df_variable_definitions = self.check_input_data_frame(df_variable_definition)
		self.fields_id = self.get_id_fields(self.df_variable_definitions)
		self.field_min_scalar, self.field_max_scalar, self.time_period_scalar = self.get_scalar_time_period(self.df_variable_definitions)
		self.fields_time_periods, self.time_periods = self.get_time_periods(self.df_variable_definitions)
		self.variable_trajectory_group = self.get_trajgroup(self.df_variable_definitions)
		self.uncertainty_fan_function_parameters = self.get_fan_function_parameters(fan_function_specification)
		self.uncertainty_ramp_vector = self.build_ramp_vector(self.uncertainty_fan_function_parameters)

		self.data_table, self.df_id_coordinates, self.id_coordinates = self.check_scenario_variables(self.df_variable_definitions, self.fields_id)
		self.dict_id_values, self.dict_baseline_ids = self.get_scenario_values(self.data_table, self.fields_id, dict_baseline_ids)
		self.num_scenarios = len(self.id_coordinates)
		self.variable_specifications = self.get_all_vs(self.data_table)
		self.dict_variable_info = self.get_variable_dictionary(
			self.data_table,
			self.variable_specifications
		)
		self.ordered_trajectory_arrays = self.get_ordered_trajectory_arrays(self.data_table, self.fields_id, self.fields_time_periods, self.variable_specifications)
		self.scalar_diff_arrays = self.get_scalar_diff_arrays()

		# important components for different design ids + assessing uncertainty in lever acheivement
		self.fields_order_strat_diffs = [x for x in self.fields_id if (x != self.field_strategy_id)] + [self.field_variable, self.field_variable_trajgroup_type]
		self.xl_type, self.dict_strategy_info = self.infer_sampling_unit_type()



	##################################
	#	INITIALIZATION FUNCTIONS	#
	##################################

	def set_parameters(self,):

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

		return None



	def check_input_data_frame(self, df_in: pd.DataFrame):

		# some standardized fields to require
		fields_req = [self.field_strategy_id, self.field_variable_trajgroup, self.field_variable_trajgroup_type, self.field_uniform_scaling_q, self.field_variable]
		if len(set(fields_req) & set(df_in.columns)) < len(set(fields_req)):
			fields_missing = list(set(fields_req) - (set(fields_req) & set(df_in.columns)))
			fields_missing.sort()
			str_missing = ", ".join([f"'{x}'" for x in fields_missing])
			raise ValueError(f"Error: one or more columns are missing from the data frame. Columns {str_missing} not found")
		elif (self.field_strategy_id in df_in.columns) and ("_id" not in self.field_strategy_id):
			raise ValueError(f"Error: the strategy field '{self.field_strategy_id}' must contain the substring '_id'. Check to ensure this substring is specified.")

		return df_in.drop_duplicates()



	def check_scenario_variables(self,
		df_in: pd.DataFrame,
		fields_id: list,
		field_merge_key: Union[str, None] = None
	) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple]]:

		field_merge_key = self.primary_key_id_coordinates if (field_merge_key is None) else field_merge_key
		tups_id = set([tuple(x) for x in np.array(df_in[fields_id])])

		for tg_type in self.required_tg_specs:
			df_check = df_in[df_in[self.field_variable_trajgroup_type] == tg_type]
			for vs in list(df_check[self.field_variable].unique()):
				tups_id = tups_id & set([tuple(x) for x in np.array(df_check[df_check[self.field_variable] == vs][fields_id])])

		df_scen = pd.DataFrame(tups_id, columns = fields_id)
		df_in = pd.merge(df_in, df_scen, how = "inner", on = fields_id)
		df_scen[field_merge_key] = range(len(df_scen))

		return (df_in, df_scen, tups_id)



	def check_time_start_uncertainty(self, t0: int):
		return max(t0, 1)



	def generate_indexing_data_frame(self,
		df_id_coords: Union[pd.DataFrame, None] = None,
		field_primary_key_id_coords: Union[str, None] = None
		field_time_period: Union[str, None] = None
	) -> pd.DataFrame:

	"""
	Generate an data frame long by time period and all id coordinates included in the sample unit.

	Function Arguments
	------------------


	Keyword Arguments
	-----------------
	- df_id_coords: data frame containing id coordinates + primary key (in field_primary_key_id_coords)
		* If None, default to self.df_id_coordinates
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

	return df_coords_by_future



	def get_all_vs(self, df_in: pd.DataFrame):
		if not self.field_variable in df_in.columns:
			raise ValueError(f"Field '{self.field_variable}' not found in data frame.")
		all_vs = list(df_in[self.field_variable].unique())
		all_vs.sort()
		return all_vs



	def get_id_fields(self, df_in: pd.DataFrame):
		fields_out = [x for x in df_in.columns if ("_id" in x)]
		fields_out.sort()
		if len(fields_out) == 0:
			raise ValueError(f"No id fields found in data frame.")

		return fields_out



	def get_ordered_trajectory_arrays(self,
		df_in: pd.DataFrame,
		fields_id: list,
		fields_time_periods: list,
		variable_specifications: list
	) -> Dict[str, np.ndarray]:
		# order trajectory arrays by id fields; used for quicker lhs application across id dimensions
		dict_out = {}
		for vs in variable_specifications:
			df_cur_vs = df_in[df_in[self.field_variable].isin([vs])].sort_values(by = fields_id)

			if self.variable_trajectory_group == None:
				dict_out.update({(vs, None): {"data": np.array(df_cur_vs[fields_time_periods]), "id_coordinates": df_cur_vs[fields_id]}})
			else:
				for tgs in self.required_tg_specs:
					df_cur = df_cur_vs[df_cur_vs[self.field_variable_trajgroup_type] == tgs]
					dict_out.update({(vs, tgs): {"data": np.array(df_cur[fields_time_periods]), "id_coordinates": df_cur[fields_id]}})
		return dict_out



	def get_scalar_time_period(self, df_in:pd.DataFrame):
		# determine min field/time period
		field_min = [x for x in df_in.columns if "min" in x]
		if len(field_min) == 0:
			raise ValueError("No field associated with a minimum scalar value found in data frame.")
		else:
			field_min = field_min[0]

		# determine max field/time period
		field_max = [x for x in df_in.columns if "max" in x]
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



	def get_scenario_values(self, df_in: pd.DataFrame, fields_id: list, dict_baseline_ids: dict):
		# get scenario index values by scenario dimension
		dict_id_values = {}
		dict_id_baselines = dict_baseline_ids.copy()

		for fld in fields_id:
			dict_id_values.update({fld: list(df_in[fld].unique())})
			dict_id_values[fld].sort()

			# check if baseline for field is determined
			if fld in dict_id_baselines.keys():
				bv = int(dict_id_baselines[fld])

				if bv not in dict_id_values[fld]:
					if fld == self.field_strategy_id:
						raise ValueError(f"Error: baseline {self.field_strategy_id} scenario index '{bv}' not found in the variable trajectory input sheet. Please ensure the basline strategy is specified correctly.")
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



	def get_time_periods(self, df_in: pd.DataFrame):
		fields_time_periods = [x for x in df_in.columns if x.isnumeric()]
		fields_time_periods = [x for x in fields_time_periods if int(x) == float(x)]
		if len(fields_time_periods) == 0:
			raise ValueError("No time periods found in data frame.")
		else:
			time_periods = [int(x) for x in fields_time_periods]

		time_periods.sort()
		fields_time_periods = [str(x) for x in time_periods]

		return (fields_time_periods, time_periods)



	# get the trajectory group for the sampling unit
	def get_trajgroup(self, df_in: pd.DataFrame):
		if not self.field_variable_trajgroup in df_in.columns:
			raise ValueError(f"Field '{self.field_variable_trajgroup}' not found in data frame.")
		# determine if this is associated with a trajectory group
		if len(df_in[df_in[self.field_variable_trajgroup] > self.missing_trajgroup_flag]) > 0:
			return int(list(df_in[self.field_variable_trajgroup].unique())[0])
		else:
			return None



	# the variable dictionary includes information on sampling ranges for time period scalars, wether the variables should be scaled uniformly, and the trajectories themselves
	def get_variable_dictionary(self,
		df_in: pd.DataFrame,
		variable_specifications: list,
		fields_id: list = None,
		field_max: str = None,
		field_min: str = None,
		fields_time_periods: list = None
	):
		"""
		Retrieve a dictionary mapping a vs, tg pair to a list of
		"""
		fields_id = self.fields_id if (fields_id is None) else fields_id
		field_max = self.field_max_scalar if (field_max is None) else field_max
		field_min = self.field_min_scalar if (field_min is None) else field_min
		fields_time_periods = self.fields_time_periods if (fields_time_periods is None) else fields_time_periods

		dict_var_info = {}

		if self.variable_trajectory_group != None:
			tgs_loops = self.required_tg_specs
		else:
			tgs_loops = [None]

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
	def infer_sampling_unit_type(self, thresh: float = (10**(-12))):
		fields_id_no_strat = [x for x in self.fields_id if (x != self.field_strategy_id)]

		strat_base = self.dict_baseline_ids[self.field_strategy_id]
		strats_not_base = [x for x in self.dict_id_values[self.field_strategy_id] if (x != strat_base)]
		fields_ext = [self.field_variable, self.field_variable_trajgroup_type] + self.fields_id + self.fields_time_periods
		fields_ext = [x for x in fields_ext if (x != self.field_strategy_id)]
		fields_merge = [x for x in fields_ext if (x not in self.fields_time_periods)]
		# get the baseline strategy specification + set a renaming dictionary for merges
		df_base = self.data_table[self.data_table[self.field_strategy_id] == strat_base][fields_ext].sort_values(by = self.fields_order_strat_diffs).reset_index(drop = True)
		arr_base = np.array(df_base[self.fields_time_periods])
		fields_base = list(df_base.columns)

		dict_out = {
			"baseline_strategy_data_table": df_base,
			"baseline_strategy_array": arr_base,
			"difference_arrays_by_strategy": {}
		}

		dict_diffs = {}
		strategy_q = False

		for strat in strats_not_base:

			df_base = pd.merge(df_base, self.data_table[self.data_table[self.field_strategy_id] == strat][fields_ext], how = "inner", on = fields_merge, suffixes = (None, "_y"))
			df_base.sort_values(by = self.fields_order_strat_diffs, inplace = True)
			arr_cur = np.array(df_base[[(x + "_y") for x in self.fields_time_periods]])
			arr_diff = arr_cur - arr_base
			dict_diffs.update({strat: arr_diff})
			df_base = df_base[fields_base]

			if max(np.abs(arr_diff.flatten())) > thresh:
				strategy_q = True

		if strategy_q:
			dict_out.update({"difference_arrays_by_strategy": dict_diffs})
			type_out = "L"
		else:
			type_out = "X"

		return type_out, dict_out



	############################
	#	CORE FUNCTIONALITY	#
	############################

	def get_scalar_diff_arrays(self):

		tp_end = self.fields_time_periods[-1]
		dict_out = {}

		for vs in self.variable_specifications:
			if self.variable_trajectory_group != None:
				tgs_loops = self.required_tg_specs
			else:
				tgs_loops = [None]
			for tgs in tgs_loops:

				# get the vector (dim is by scenario, sorted by self.fields_id )
				vec_tp_end = self.ordered_trajectory_arrays[(vs, tgs)]["data"][:,-1]
				tups_id_coords = [tuple(x) for x in np.array(self.ordered_trajectory_arrays[(vs, tgs)]["id_coordinates"])]

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
	def build_ramp_vector(self, tuple_param):

		if tuple_param == None:
			tuple_param = self.get_f_fan_function_parameter_defaults(self.uncertainty_fan_function_type)

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
	def get_f_fan_function_parameter_defaults(self, n: int, fan_type: str, return_type: str = "params"):
		dict_ret = {
			"linear": (0, 2, 1, n/2),
			"sigmoid": (1, 0, math.e, n/2)
		}

		if return_type == "params":
			return dict_ret[fan_type]
		elif return_type == "keys":
			return list(dict_ret.keys())
		else:
			str_avail_keys = ", ".join(list(dict_ret.keys()))
			raise ValueError(f"Error: invalid return_type '{return_type}'. Ensure it is one of the following: {str_avail_keys}.")



	# verify fan function parameters
	def get_fan_function_parameters(self, fan_type):

		if type(fan_type) == str:
			n = len(self.time_periods) - self.time_period_end_certainty
			keys = self.get_f_fan_function_parameter_defaults(n, fan_type, "keys")
			if fan_type in keys:
				return self.get_f_fan_function_parameter_defaults(n, fan_type, "params")
			else:
				str_avail_keys = ", ".join(keys)
				raise ValueError(f"Error: no defaults specified for uncertainty fan function of type {fan_type}. Use a default or specify parameters a, b, c, and d. Default functional parameters are available for each of the following: {str_avail_keys}")
		elif type(fan_type) == tuple:
			if len(fan_type) == 4:
				if set([type(x) for x in fan_type]).issubset({int, float}):
					return fan_type
				else:
					raise ValueError(f"Error: fan parameter specification {fan_type} contains invalid parameters. Ensure they are numeric (int or float)")
			else:
				raise ValueError(f"Error: fan parameter specification {fan_type} invalid. 4 Parameters are required.")



	def build_futures(self,
		n_samples: int,
		random_seed: int
	):
		print(f"sampling {self.id_values}")



	def generate_future(self,
		lhs_trial_x: float,
		lhs_trial_l: float = 1.0,
		constraints_mix_tg: tuple = (0, 1),
		baseline_future_q: bool = False
	) -> Dict[str, np.ndarray]:
		"""
		Generate a dictionary mapping each variable specification to futures ordered by self.ordered_trajectory_arrays((vs, tg))["id_coordinates"]

		Function Arguments
		------------------
		- lhs_trial_x: LHS trial used to generate uncertainty fan for base future

		Keyword Arguments
		------------------
		- lhs_trial_l: LHS trial used to modify strategy effect
		- constraints_mix_tg: constraints on the mixing fraction for trajectory groups
		- baseline_future_q: generate a baseline future? If so, lhs trials do not apply.
		"""

		# some checks
		if (lhs_trial_x < 0):
				raise ValueError(f"The value of lhs_trial_x = {lhs_trial_x} is invalid. lhs_trial_x must be >= 0.")
		if (lhs_trial_l < 0):
				raise ValueError(f"The value of lhs_trial_l = {lhs_trial_l} is invalid. lhs_trial_l must be >= 0.")

		# initialization
		all_strats = self.dict_id_values.get(self.field_strategy_id)
		n_strat = len(all_strats)
		strat_base = self.dict_baseline_ids.get(self.field_strategy_id)

		# index by variable_specification at keys
		dict_out = {}

		if self.variable_trajectory_group is not None:
			#list(set([x[0] for x in self.ordered_trajectory_arrays.keys()]))
			cat_mix = self.dict_required_tg_spec_fields.get("mixing_trajectory")
			cat_b0 = self.dict_required_tg_spec_fields.get("trajectory_boundary_0")
			cat_b1 = self.dict_required_tg_spec_fields.get("trajectory_boundary_1")

			# use mix between 0/1 (0 = 100% trajectory_boundary_0, 1 = 100% trajectory_boundary_1)
			for vs in self.variable_specifications:

				dict_arrs = {
					cat_b0: self.ordered_trajectory_arrays[(vs, cat_b0)]["data"],
					cat_b1: self.ordered_trajectory_arrays[(vs, cat_b1)]["data"],
					cat_mix: self.ordered_trajectory_arrays[(vs, cat_mix)]["data"]
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
						self.ordered_trajectory_arrays[(vs, cat_mix)]["id_coordinates"].copy().reset_index(drop = True),
						pd.DataFrame(arr_out, columns = self.fields_time_periods)],
						axis = 1
					)
					w = np.where(df_ids_ota[self.field_strategy_id] == strat_base)
					df_ids_ota = df_ids_ota.iloc[w[0].repeat(n_strat)].reset_index(drop = True)

					arr_out = np.array(df_ids_ota[self.fields_time_periods])
					l_modified_cats = []
					inds0 = set(np.where(self.dict_strategy_info["baseline_strategy_data_table"][self.field_variable] == vs)[0])

					for cat_cur in [cat_b0, cat_b1, cat_mix]:

						# get the index for the current vs/cat_cur
						inds = np.sort(np.array(list(inds0 & set(np.where(self.dict_strategy_info["baseline_strategy_data_table"][self.field_variable_trajgroup_type] == cat_cur)[0]))))
						n_inds = len(inds)
						df_ids0 = self.dict_strategy_info["baseline_strategy_data_table"][[x for x in self.fields_id if (x != self.field_strategy_id)]].loc[inds.repeat(n_strat)].reset_index(drop = True)
						new_strats = list(np.zeros(len(df_ids0)).astype(int))

						# initialize as list - we only do this to guarantee the sort is correct
						df_future_strat = np.zeros((n_inds*n_strat, len(self.fields_time_periods)))
						ind_repl = 0

						##  start loop
						for strat in all_strats:
							# strategy ids
							new_strats[ind_repl*n_inds:((ind_repl + 1)*n_inds)] = [strat for x in inds]
							# get the strategy difference that is adjusted by lhs_trial_x_delta; if baseline strategy, use 0s
							df_repl = np.zeros((n_inds, len(self.fields_time_periods))) if (strat == strat_base) else self.dict_strategy_info["difference_arrays_by_strategy"][strat][inds, :]*lhs_trial_l
							np.put(df_future_strat, range(n_inds*len(self.fields_time_periods)*ind_repl, n_inds*len(self.fields_time_periods)*(ind_repl + 1)), df_repl)
							ind_repl += 1

						df_ids0[self.field_strategy_id] = new_strats
						df_future_strat = pd.concat([df_ids0, pd.DataFrame(df_future_strat, columns = self.fields_time_periods)], axis = 1).sort_values(by = self.fields_id).reset_index(drop = True)
						l_modified_cats.append(dict_arrs[cat_cur] + np.array(df_future_strat[self.fields_time_periods]))

					arr_out = self.mix_tensors(*l_modified_cats, constraints_mix_tg)

					#
					# one option for this approach is to compare the difference between the "L" design uncertainty and the baseline and add this to the uncertain future (final array)
					#

				dict_out.update({vs: arr_out})


		else:

			rv = self.uncertainty_ramp_vector

			for vs in self.variable_specifications:
				# order the uniform scaling by the ordered trajectory arrays
				vec_unif_scalar = self.ordered_by_ota_from_fid_dict(self.dict_variable_info[(vs, None)]["uniform_scaling_q"], (vs, None))
				# gives 1s where we keep standard fanning (using the ramp vector) and 0s where we use uniform scaling
				vec_base = 1 - vec_unif_scalar
				#
				if max(vec_unif_scalar) > 0:
					vec_max_scalar = self.ordered_by_ota_from_fid_dict(self.dict_variable_info[(vs, None)]["max_scalar"], (vs, None))
					vec_min_scalar = self.ordered_by_ota_from_fid_dict(self.dict_variable_info[(vs, None)]["min_scalar"], (vs, None))
					vec_unif_scalar = vec_unif_scalar*(vec_min_scalar + lhs_trial_x*(vec_max_scalar - vec_min_scalar))

				vec_unif_scalar = np.array([vec_unif_scalar]).transpose()
				vec_base = np.array([vec_base]).transpose()

				delta_max = self.scalar_diff_arrays[(vs, None)]["max_tp_end_delta"]
				delta_min = self.scalar_diff_arrays[(vs, None)]["min_tp_end_delta"]
				delta_diff = delta_max - delta_min
				delta_val = delta_min + lhs_trial_x*delta_diff

				dict_ota = self.ordered_trajectory_arrays.get((vs, None))
				delta_vec = 0.0 if baseline_future_q else (rv * np.array([delta_val]).transpose())
				array_out = dict_ota.get("data") + delta_vec
				array_out = array_out*vec_base + vec_unif_scalar*self.ordered_trajectory_arrays[(vs, None)]["data"]

				if self.xl_type == "L":
					# get series of strategies
					series_strats = self.ordered_trajectory_arrays.get((vs, None))["id_coordinates"][self.field_strategy_id]
					w = np.where(np.array(series_strats) == strat_base)[0]
					# get strategy adjustments
					lhs_mult_deltas = 1.0 if baseline_future_q else lhs_trial_l
					array_strat_deltas = np.concatenate(
						series_strats.apply(
							self.dict_strategy_info["difference_arrays_by_strategy"].get,
							args = (np.zeros((1, len(self.time_periods))),)
						)
					)*lhs_mult_deltas
					array_out = (array_strat_deltas + array_out[w, :]) if (len(w) > 0) else array_out

				dict_out.update({vs: array_out})

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
	- fan_function_specification: type of uncertainty approach to use
		* linear: linear ramp to time time T - 1
		* sigmoid: sigmoid function that ramps to time T - 1
	- field_sample_unit_group: field used to identify sample unit groups. Sample unit groups are composed of:
		* individual variable specifications
		* trajectory groups
	- field_strategy_id: field used to identify the strategy (int)
		* This field is important as uncertainty in strategies is assessed differently than uncetainty in other variables
	- field_uniform_scaling_q: field used to identify whether or not a variable
	- field_variable: field used to specify variables
	- field_variable_trajgroup: field used to identify the trajectory group (integer)
	- field_variable_trajgroup_type: field used to identify the trajectory group type (max, min, mix, or lhs)
	- fan_function_specification: type of uncertainty approach to use
		* linear: linear ramp to time time T - 1
		* sigmoid: sigmoid function that ramps to time T - 1
	- regex_trajgroup: Regular expression used to identify trajectory group variables in `field_variable` of `df_input_database`
	- regex_trajmax: Regular expression used to identify trajectory maxima in variables and trajgroups specified in `field_variable` of `df_input_database`
	- regex_trajmin: Regular expression used to identify trajectory minima in variables and trajgroups specified in `field_variable` of `df_input_database`
	- regex_trajmix: Regular expression used to identify trajectory baseline mix (fraction maxima) in variables and trajgroups specified in `field_variable` of `df_input_database`

	"""
	def __init__(self,
		df_input_database: pd.DataFrame,
		dict_baseline_ids: Dict[str, int],
		time_period_u0: int,

		fan_function_specification: str = "linear",
		field_sample_unit_group: str = "sample_unit_group",
		field_strategy_id: str = "strategy_id",
		field_uniform_scaling_q: str = "uniform_scaling_q",
		field_variable: str = "variable",
		field_variable_trajgroup: str = "variable_trajectory_group",
		field_variable_trajgroup_type: str = "variable_trajectory_group_trajectory_type",

		regex_trajgroup: re.Pattern = re.compile("trajgroup_(\d*)-(\D*$)"),
		regex_trajmax: re.Pattern = re.compile("trajmax_(\D*$)"),
		regex_trajmin: re.Pattern = re.compile("trajmin_(\D*$)"),
		regex_trajmix: re.Pattern = re.compile("trajmix_(\D*$)"),

		specification_tgt_lhs: str = "lhs",
		specification_tgt_max: str = "trajectory_boundary_1",
		specification_tgt_min: str = "trajectory_boundary_0"
	):

		##  INITIALIZE PARAMETERS

		self.df_input_database = df_input_database
		# set default fields
		self.field_sample_unit_group = field_sample_unit_group
		self.field_strategy_id = field_strategy_id
		self.field_uniform_scaling_q = field_uniform_scaling_q
		self.field_variable = field_variable
		self.field_variable_trajgroup = field_variable_trajgroup
		self.field_variable_trajgroup_type = field_variable_trajgroup_type
		# missing values flag
		self.missing_flag_int = -999
		# default regular expressions
		self.regex_trajgroup = regex_trajgroup
		self.regex_trajmax = regex_trajmax
		self.regex_trajmin = regex_trajmin
		self.regex_trajmix = regex_trajmix
		# some default internal specifications used in templates
		self.specification_tgt_lhs = specification_tgt_lhs
		self.specification_tgt_max = specification_tgt_max
		self.specification_tgt_min = specification_tgt_min



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
		- regex_trajgroup:

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



	def prepare_input_database(self,
		df_in: pd.DataFrame = None,
		field_sample_unit_group: Union[str, None] = None,
		field_variable: Union[str, None] = None,
		field_variable_trajgroup: Union[str, None] = None,
		field_variable_trajgroup_type: Union[str, None] = None,
		missing_trajgroup_flag: Union[int, None] = None,
		regex_trajgroup: Union[re.Pattern, None] = None,
		regex_trajmax: Union[re.Pattern, None] = None,
		regex_trajmin: Union[re.Pattern, None] = None,
		regex_trajmix: Union[re.Pattern, None] = None
	):
		"""
		Prepare the input database for sampling by adding sample unit group, cleaning up trajectory groups, etc.

		Keyword Arguments
		-----------------
		- df_in: input database to use to generate SampleUnit objects
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
		df_in = self.df_input_database if (df_in is None) else df_in

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

		return df_in
