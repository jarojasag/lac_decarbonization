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
		self.list_required_base_fields = [
			self.field_req_normalize_group,
			self.field_req_subsector,
			self.field_req_trajgroup_no_vary_q,
			self.field_req_uniform_scaling_q,
			self.field_req_variable,
			self.field_req_variable_trajectory_group,
			self.field_req_variable_trajectory_group_trajectory_type
		]

        self.regex_sheet_name = self.set_regex_sheet_name()







    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def set_regex_sheet_name(self,
    ) -> re.Pattern:
        """
            Set the regular expression for input sheets to match
        """
        return re.compile(f"{self.model_attributes.dim_strategy_id}-(\d*$)")



    ############################
    #    TEMPLATE FUNCTIONS    #
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
    ) -> tuple[list, dict[str, int], dict[str, pd.DataFrame]]:
        """
            Import the InputTemplate, check strategies, and set characteristics

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

        all_strategies = []
        dict_outputs = {}
        dict_sheet_to_strategy = {}
        sheets_iterate = list(dict_inputs.keys())

        for k in sheets_iterate:
            if self.regex_sheet_name.match(k) is not None:
				# get the strategy, test if it is baseline
				dict_strat_cur = get_sheet_strategy(k, return_type = dict)
				baseline_q = (dict_strat_cur.get(k) == self.model_attributes.get_baseline_scenario_id(self.model_attributes.dim_strategy_id))

                # get the data frame and check it
				df_template_sheet = dict_inputs.get(k)
				tup = verify_input_template_sheet(df_template_sheet, base_strategy_q = baseline_q)

				# note: error passing and descriptions are handled in verify_input_template_sheet
				if tup is not None:
					dict_field_tp_to_tp, df_template_sheet, field_min, field_max, fields_tp  = tup

					# need to build function for adding index fields (e.g., strategy id, etc.)
					df_template_sheet

					#dict_sheet_to_strategy.update()
					all_strategies.append(dict_sheet_to_strategy.get(k))

            else:
                del dict_inputs[k]

        return all_strategies, dict_sheet_to_strategy, dict_inputs




    def build_modvar_input_db(self,
        repl_missing_with_base: bool = True,
        strat_base: int = 0
    ) -> pd.DataFrame:

        sheet_base = f"{model_attributes.dim_strategy_id}-{strat_base}"
        strats_eval = [strat_base]

        df_out = []

        for sec in sectors:
            fp_templ = sa.excel_template_path(sec, region, template_type, True)
            if not os.path.exists(fp_templ):
                raise ValueError(f"Error: path '{fp_templ}' to template not found.")
            # check available sheets and ensure baseline is available
            sheets_avail = pd.ExcelFile(fp_templ).sheet_names
            if sheet_base not in sheets_avail:
                 raise ValueError(f"Baseline strategy sheet {sheet_base} not found in '{fp_templ}'. The template must have a sheet for the baseline strategy.")


            for strat in strats_eval:
                sheet = f"{model_attributes.dim_strategy_id}-{strat}"
                if not sheet in sheets_avail:
                    msg = f"Sheet {sheet} not found in '{fp_templ}'. Check the template."
                    if repl_missing_with_base:
                        warnings.warn(f"{msg}. The baseline strategy will be used.")
                        sheet = sheet_base
                    else:
                        raise ValueError(msg)

                #
                df_tmp = pd.read_excel(fp_templ, sheet_name = sheet)
                df_tmp[model_attributes.dim_strategy_id] = strat

                #
                #   ADD CHECKS FOR TIME PERIODS
                #


                #
                #   ADD DIFFERENT STEPS FOR NON-BASELINE STRATEGY
                #

                if len(df_out) == 0:
                    df_out.append(df_tmp)
                else:
                    df_out.append(df_tmp[df_out[0].columns])

        df_out = pd.concat(df_out, axis = 0).sort_values(by = ["subsector", "variable"]).reset_index(drop = True)

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
        field_max = [regex_max.match(x) for x in df_in.columns if (regex_max.match(x) is not None)]
        if len(field_min) == 0:
			raise KeyError("No field associated with a maximum scalar value found in data frame.")
		elif len(field_min) > 1:
			fpl = sf.format_print_list(field_max)
			raise KeyError(f"Multiple maximum fields found in input DataFrame: {fpl} all satisfy the conditions. Choose one and retry.")
        else:
            field_max = field_max[0]
			tp_max = Int(field_max.groups()[0])
			field_max = field_max.string

        # determine min field/time period
        field_min = [regex_min.match(x) for x in df_in.columns if (regex_min.match(x) is not None)]
        if len(field_min) == 0:
			raise KeyError("No field associated with a minimum scalar value found in data frame.")
		elif len(field_min) > 1:
			fpl = sf.format_print_list(field_min)
			raise KeyError(f"Multiple minimum fields found in input DataFrame: {fpl} all satisfy the conditions. Choose one and retry.")
        else:
            field_min = field_min[0]
			tp_min = Int(field_min.groups()[0])
			field_min = field_min.string

		# check that min/max specify final time period
        if (tp_min != tp_max):
            raise ValueError(f"Fields '{tp_min}' and '{tp_max}' imply asymmetric final time periods.")


		##  GET TIME PERIODS

		# get initial information on time periods
		fields_tp = [regex_tp.match(x) for x in df_in.columns if (regex_tp.match(x) is not None)]
		dict_field_tp_to_tp = dict([(x.string, Int(x.groups()[0])) for x in fields_tp])

		# check fields for definition in attribute_time_period
		pydim_time_period = self.model_attributes.get_dimensional_attribute(self.model_attributes.dim_time_period, "pydim")
        attr_tp = self.dict_attributes.get(pydim_time_period)
		# fields to keep/drop
		fields_valid = [x for x in fields_tp if (dict_field_tp_to_tp.get(x) in attr_tp.key_values)]
		fields_invalid = [x for x in fields_tp if (x not in fields_valid)]
		defined_tp = [dict_field_tp_to_tp.get(x) for x in fields_valid]

		if !(tp_max in defined_tp):
			raise ValueError(f"Error trying to define template: the final time period {tp_max} defined in the input template does not exist in the {self.model_attributes.dim_time_period} attribute table at '{attr_tp.fp_table}'")

		if length(fields_invalid) > 0:
			flds_drop = sf.format_print_list(fields_invalid)
			warnings.warning(f"Dropping fields {flds_drop} from input template: the time periods are not defined in the {self.model_attributes.dim_time_period} attribute table at '{attr_tp.fp_table}'")
			df_in.drop(fields_invalid, axis = 1, inplace = True)

			fields_tp = fields_valid

        return (dict_field_tp_to_tp, df_in, field_min, field_max, fields_tp)



	##  verify the structure of an input template sheet
	def verify_input_template_sheet(self,
		df_template_sheet: pd.DataFrame,
		base_strategy_q: bool = False,
		sheet_name: str = None
	) -> Union[tuple[dict, pd.DataFrame, str, str, list[str]], None]:
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

			sf.check_fields(df_template_sheet, self.list_required_base_fields)
			(
				dict_field_tp_to_tp,
				df_template_sheet,
				field_min,
				field_max,
				fields_tp
			) = self.verify_and_return_sheet_time_periods(df_template_sheet)

		except Exception as e:
			sheet_str = f" '{sheet_name}'" if (sheet_name is not None) else ""
			warning(f"Trying to verify sheet{sheet_str} produced the following error in verify_input_template_sheet:\n\t{e}\nReturning None")
			return None

		return (dict_field_tp_to_tp, df_template_sheet, field_min, field_max, fields_tp)
