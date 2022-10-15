#
# SAVE AFOLU COMPONENT
#
# USED IN SOC AND F_SOM

# loop over dry/wet to estimate carbon stocks in crops
for modvar in self.modvar_list_agrc_frac_drywet:
    # soil category
    cat_soil = ma.clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
    ind_soil = attr_soil.get_key_value_index(cat_soil)
    arr_soil_soc_crop_drywet_cur = (arr_agrc_crop_area*dict_arrs_agrc_frac_drywet[modvar]).transpose()
    # add component to EF1 estimate for F_SOM
    vec_soil_ef1_soc_est += np.sum(arr_soil_soc_crop_drywet_cur, axis = 0)*arr_soil_ef1_organic[:, ind_soil]/vec_soil_area_crop_pasture
    # then, modify the soc array and estimate contribution to w
    arr_soil_soc_crop_drywet_cur *= arr_soil_organic_c_stocks[:, ind_soil]*arr_lndu_factor_soil_carbon[:, self.ind_lndu_crop]*(1 - vec_soil_soc_lost_in_cropland)
    vec_soil_soc_total_cur = np.sum(arr_soil_soc_crop_drywet_cur, axis = 0)
    vec_soil_soc_total += vec_soil_soc_total_cur
    vec_soil_soc_total_mineral += vec_soil_soc_total_cur*arr_lndu_frac_mineral_soils[:, self.ind_lndu_crop]

# loop over tropical/temperate cropland to get soil carbon for organic drained soils
for modvar in self.modvar_list_agrc_frac_temptrop:
    # soil category
    cat_soil = ma.clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
    ind_soil = attr_soil.get_key_value_index(cat_soil)
    # get land use category for soil carbon facto
    arr_soil_soc_crop_temptrop_cur = (arr_agrc_crop_area*dict_arrs_agrc_frac_temptrop[modvar]).transpose()
    arr_soil_soc_crop_temptrop_cur *= arr_lndu_frac_organic_soils[:, self.ind_lndu_crop]
    # get SOC totals and integrate land-use specific mineral fractions
    vec_soil_emission_co2_soil_carbon_organic += np.sum(arr_soil_soc_crop_temptrop_cur*arr_soil_ef_c_organic_cultivated_soils[:, ind_soil], axis = 0)

# loop over dry/wet to estimate carbon stocks in grassland
for modvar in self.modvar_list_lndu_frac_drywet:
    # soil category
    cat_soil = ma.clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
    ind_soil = attr_soil.get_key_value_index(cat_soil)
    vec_soil_soc_pstr_drywet_cur = (arr_land_use*dict_arrs_lndu_frac_drywet[modvar])[:, self.ind_lndu_grass]
    # add component to EF1 estimate for F_SOM
    vec_soil_ef1_soc_est += vec_soil_soc_pstr_drywet_cur.copy()*arr_soil_ef1_organic[:, ind_soil]/vec_soil_area_crop_pasture
    vec_soil_soc_pstr_drywet_cur *= arr_soil_organic_c_stocks[:, ind_soil]*arr_lndu_factor_soil_carbon[:, self.ind_lndu_grass]
    vec_soil_soc_total += vec_soil_soc_pstr_drywet_cur
    vec_soil_soc_total_mineral += vec_soil_soc_pstr_drywet_cur*arr_lndu_frac_mineral_soils[:, self.ind_lndu_grass]


# loop over tropical/temperate NP/temperate NR
for modvar in self.modvar_list_frst_frac_temptrop:
    # soil category
    cat_soil = ma.clean_schema(self.model_attributes.get_variable_attribute(modvar, pycat_soil))
    ind_soil = attr_soil.get_key_value_index(cat_soil)
    # get land use category for soil carbon facto
    cats_lndu = [ma.clean_schema(x) for x in self.model_attributes.get_ordered_category_attribute(self.subsec_name_frst, pycat_lndu)]
    inds_lndu = [attr_lndu.get_key_value_index(x) for x in cats_lndu]
    arr_soil_soc_frst_temptrop_cur = (arr_area_frst*dict_arrs_frst_frac_temptrop[modvar]*arr_lndu_factor_soil_carbon[:, inds_lndu]).transpose()
    arr_soil_soc_frst_temptrop_cur *= arr_soil_organic_c_stocks[:, ind_soil]
    # get SOC totals and integrate land-use specific mineral fractions
    vec_soil_soc_total_cur = np.sum(arr_soil_soc_frst_temptrop_cur, axis = 0)
    vec_soil_soc_total += vec_soil_soc_total_cur
    vec_soil_soc_total_mineral += np.sum(arr_soil_soc_frst_temptrop_cur.transpose()*arr_lndu_frac_mineral_soils[:, inds_lndu], axis = 1)
