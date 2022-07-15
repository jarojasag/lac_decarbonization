# agricultural model variables
self.modvar_agrc_area_prop_calc = "Cropland Area Proportion"
self.modvar_agrc_area_prop_init = "Initial Cropland Area Proportion"
self.modvar_agrc_area_crop = "Crop Area"
self.modvar_agrc_ef_ch4 = ":math:\\text{CH}_4 Crop Anaerobic Decomposition Emission Factor"
self.modvar_agrc_ef_co2_biomass = ":math:\\text{CO}_2 Crop Biomass Emission Factor"
self.modvar_agrc_ef_co2_soil_carbon = ":math:\\text{CO}_2 Crop Soil Carbon Emission Factor"
self.modvar_agrc_ef_n2o_burning = ":math:\\text{N}_2\\text{O} Crop Biomass Burning Emission Factor"
self.modvar_agrc_ef_n2o_fertilizer = ":math:\\text{N}_2\\text{O} Crop Fertilizer and Lime Emission Factor"
self.modvar_agrc_elas_crop_demand_income = "Crop Demand Income Elasticity"
self.modvar_agrc_emissions_ch4_rice = ":math:\\text{CH}_4 Emissions from Rice"
self.modvar_agrc_emissions_co2_soil_carbon = ":math:\\text{CO}_2 Emissions from Soil Carbon"
self.modvar_agrc_emissions_n2o_biomass_burning = ":math:\\text{N}_2\\text{O} Emissions from Biomass Burning"
self.modvar_agrc_frac_animal_feed = "Crop Fraction Animal Feed"
self.modvar_agrc_frac_dry = "Agriculture Fraction Dry"
self.modvar_agrc_frac_temperate = "Agriculture Fraction Temperate"
self.modvar_agrc_frac_tropical = "Agriculture Fraction Tropical"
self.modvar_agrc_frac_wet = "Agriculture Fraction Wet"
self.modvar_agrc_net_imports = "Change to Net Imports of Crops"
self.modvar_agrc_yf = "Crop Yield Factor"
self.modvar_agrc_yield = "Crop Yield"
# additional lists
self.modvar_list_agrc_frac_drywet = [
    self.modvar_agrc_frac_dry,
    self.modvar_agrc_frac_wet
]
self.modvar_list_agrc_frac_temptrop = [
    self.modvar_agrc_frac_temperate,
    self.modvar_agrc_frac_tropical
]

# forest model variables
self.modvar_frst_average_fraction_burned_annually = "Average Fraction of Forest Burned Annually"
self.modvar_frst_ef_fires = "Forest Fire Emission Factor"
self.modvar_frst_ef_ch4 = "Forest Methane Emissions"
self.modvar_frst_emissions_sequestration = ":math:\\text{CO}_2 Emissions from Forest Sequestration"
self.modvar_frst_emissions_methane = ":math:\\text{CH}_4 Emissions from Forests"
self.modvar_frst_frac_temperate_nutrient_poor = "Forest Fraction Temperate Nutrient Poor"
self.modvar_frst_frac_temperate_nutrient_rich = "Forest Fraction Temperate Nutrient Rich"
self.modvar_frst_frac_tropical = "Forest Fraction Tropical"
self.modvar_frst_sq_co2 = "Forest Sequestration Emission Factor"
self.modvar_frst_init_per_hh_wood_demand = "Initial Per Household Wood Demand"
#additional lists
self.modvar_list_frst_frac_temptrop = [
    self.modvar_frst_frac_temperate_nutrient_poor,
    self.modvar_frst_frac_temperate_nutrient_rich,
    self.modvar_frst_frac_tropical
]

self.modvar_ippu_average_construction_materials_required_per_household = "Average per Household Demand for Construction Materials"
self.modvar_ippu_ratio_of_production_to_harvested_wood = "Ratio of Production to Harvested Wood Demand"




# land use model variables
self.modvar_lndu_area_by_cat = "Land Use Area"
self.modvar_lndu_area_converted_from_type = "Area of Land Use Area Conversion Away from Type"
self.modvar_lndu_area_converted_to_type = "Area of Land Use Area Conversion To Type"
self.modvar_lndu_ef_co2_conv = ":math:\\text{CO}_2 Land Use Conversion Emission Factor"
self.modvar_lndu_emissions_conv = ":math:\\text{CO}_2 Emissions from Land Use Conversion"
self.modvar_lndu_emissions_ch4_from_wetlands = ":math:\\text{CH}_4 Emissions from Wetlands"
self.modvar_lndu_emissions_n2o_from_pastures = ":math:\\text{N}_2\\text{O} Emissions from Pastures"
self.modvar_lndu_emissions_co2_from_pastures = ":math:\\text{CO}_2 Emissions from Pastures"
self.modvar_lndu_frac_dry = "Land Use Fraction Dry"
self.modvar_lndu_frac_temperate = "Land Use Fraction Temperate"
self.modvar_lndu_frac_tropical = "Land Use Fraction Tropical"
self.modvar_lndu_frac_wet = "Land Use Fraction Wet"
self.modvar_lndu_initial_frac = "Initial Land Use Area Proportion"
self.modvar_lndu_ef_ch4_boc = "Land Use BOC :math:\\text{CH}_4 Emission Factor"
self.modvar_lndu_ef_co2_soilcarb = "Land Use Soil Carbon :math:\\text{CO}_2 Emission Factor"
self.modvar_lndu_prob_transition = "Unadjusted Land Use Transition Probability"
self.modvar_lndu_reallocation_factor = "Land Use Yield Reallocation Factor"
self.modvar_lndu_vdes = "Vegetarian Diet Exchange Scalar"
# additional lists
self.modvar_list_lndu_frac_drywet = [
    self.modvar_lndu_frac_dry,
    self.modvar_lndu_frac_wet
]
self.modvar_list_lndu_frac_temptrop = [
    self.modvar_lndu_frac_temperate,
    self.modvar_lndu_frac_tropical
]

# livestock model variables
self.modvar_lvst_animal_weight = "Animal Weight"
self.modvar_lvst_carrying_capacity_scalar = "Carrying Capacity Scalar"
self.modvar_lvst_dry_matter_consumption = "Daily Dry Matter Consumption"
self.modvar_lvst_ef_ch4_ef = ":math:\\text{CH}_4 Enteric Fermentation Emission Factor"
self.modvar_lvst_elas_lvst_demand = "Elasticity of Livestock Demand to GDP per Capita"
self.modvar_lvst_emissions_ch4_ef = ":math:\\text{CH}_4 Emissions from Livestock Enteric Fermentation"
self.modvar_lvst_frac_exc_n_in_dung = "Fraction Nitrogen Excretion in Dung"
self.modvar_lvst_frac_mm_anaerobic_digester = "Livestock Manure Management Fraction Anaerobic Digester"
self.modvar_lvst_frac_mm_anaerobic_lagoon = "Livestock Manure Management Fraction Anaerobic Lagoon"
self.modvar_lvst_frac_mm_composting = "Livestock Manure Management Fraction Composting"
self.modvar_lvst_frac_mm_daily_spread = "Livestock Manure Management Fraction Daily Spread"
self.modvar_lvst_frac_mm_deep_bedding = "Livestock Manure Management Fraction Deep Bedding"
self.modvar_lvst_frac_mm_dry_lot = "Livestock Manure Management Fraction Dry Lot"
self.modvar_lvst_frac_mm_incineration = "Livestock Manure Management Fraction Incineration"
self.modvar_lvst_frac_mm_liquid_slurry = "Livestock Manure Management Fraction Liquid Slurry"
self.modvar_lvst_frac_mm_poultry_manure = "Livestock Manure Management Fraction Poultry Manure"
self.modvar_lvst_frac_mm_ppr = "Livestock Manure Management Fraction Paddock Pasture Range"
self.modvar_lvst_frac_mm_solid_storage = "Livestock Manure Management Fraction Solid Storage"
self.modvar_lvst_genfactor_nitrogen = "Daily Nitrogen Generation Factor"
self.modvar_lvst_genfactor_volatile_solids = "Daily Volatile Solid Generation Factor"
self.modvar_lvst_b0_manure_ch4 = "Maximum Manure :math:\\text{CH}_4 Generation Capacity"
self.modvar_lvst_net_imports = "Change to Net Imports of Livestock"
self.modvar_lvst_pop = "Livestock Head Count"
self.modvar_lvst_pop_init = "Initial Livestock Head Count"
self.modvar_lvst_total_animal_mass = "Total Domestic Animal Mass"
self.modvar_list_lvst_mm_fractions = [
    self.modvar_lvst_frac_mm_anaerobic_digester,
    self.modvar_lvst_frac_mm_anaerobic_lagoon,
    self.modvar_lvst_frac_mm_composting,
    self.modvar_lvst_frac_mm_daily_spread,
    self.modvar_lvst_frac_mm_deep_bedding,
    self.modvar_lvst_frac_mm_dry_lot,
    self.modvar_lvst_frac_mm_incineration,
    self.modvar_lvst_frac_mm_liquid_slurry,
    self.modvar_lvst_frac_mm_poultry_manure,
    self.modvar_lvst_frac_mm_ppr,
    self.modvar_lvst_frac_mm_solid_storage
]
# some categories
self.cat_lsmm_incineration = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_lsmm, {"incineration_category": 1})[0]
self.cat_lsmm_pasture = self.model_attributes.get_categories_from_attribute_characteristic(self.subsec_name_lsmm, {"pasture_category": 1})[0]

# manure management variables
self.modvar_lsmm_dung_incinerated = "Dung Incinerated"
self.modvar_lsmm_ef_direct_n2o = ":math:\\text{N}_2\\text{O} Manure Management Emission Factor"
self.modvar_lsmm_emissions_ch4 = ":math:\\text{CH}_4 Emissions from Manure Management"
self.modvar_lsmm_emissions_n2o = ":math:\\text{N}_2\\text{O} Emissions from Manure Management"
self.modvar_lsmm_frac_loss_leaching = "Fraction of Nitrogen Lost to Leaching"
self.modvar_lsmm_frac_loss_volatilisation = "Fraction of Nitrogen Lost to Volatilisation"
self.modvar_lsmm_frac_n_available_used = "Fraction of Nitrogen Used in Fertilizer"
self.modvar_lsmm_mcf_by_pathway = "Manure Management Methane Correction Factor"
self.modvar_lsmm_n_from_bedding = "Nitrogen from Bedding per Animal"
self.modvar_lsmm_n_from_codigestates = "Nitrogen from Co-Digestates Factor"
self.modvar_lsmm_n_to_pastures = "Total Nitrogen to Pastures"
self.modvar_lsmm_n_to_fertilizer= "Nitrogen Available for Fertilizer"
self.modvar_lsmm_n_to_fertilizer_agg_dung = "Total Nitrogen Available for Fertilizer from Dung"
self.modvar_lsmm_n_to_fertilizer_agg_urea = "Total Nitrogen Available for Fertilizer from Urea"
self.modvar_lsmm_ratio_n2_to_n2o = "Ratio of :math:\\text{N}_2 to :math:\\text{N}_2\\text{O}"
self.modvar_lsmm_rf_biogas = "Biogas Recovery Factor at LSMM Anaerobic Digesters"
self.modvar_lsmm_rf_biogas_recovered = "LSMM Biogas Recovered from Anaerobic Digesters"

# soil management variables
self.modvar_soil_demscalar_fertilizer = "Fertilizer N Demand Scalar"
self.modvar_soil_demscalar_liming = "Liming Demand Scalar"
self.modvar_soil_ef1_n_managed_soils_rice = ":math:\\text{EF}_{1FR} - \\text{N}_2\\text{O} Rice Fields"
self.modvar_soil_ef1_n_managed_soils_org_fert = ":math:\\text{EF}_1 - \\text{N}_2\\text{O} Organic Amendments and Fertilizer"
self.modvar_soil_ef1_n_managed_soils_syn_fert = ":math:\\text{EF}_1 - \\text{N}_2\\text{O} Synthetic Fertilizer"
self.modvar_soil_ef2_n_organic_soils = ":math:\\text{EF}_2 - \\text{N}_2\\text{O} Emissions from Drained and Managed Organic Soils"
self.modvar_soil_ef3_n_prp = "EF3 N Pasture Range and Paddock"
self.modvar_soil_ef4_n_volatilisation = "EF4 N Volatilisation and Re-Deposition Emission Factor"
self.modvar_soil_ef5_n_leaching = "EF5 N Leaching and Runoff Emission Factor"
self.modvar_soil_ef_c_liming_dolomite = "C Liming Emission Factor Dolomite"
self.modvar_soil_ef_c_liming_limestone = "C Liming Emission Factor Limestone"
self.modvar_soil_ef_c_urea = "C Urea Emission Factor"
self.modvar_soil_emissions_co2_lime_urea = ":math:\\text{CO}_2 Emissions from Lime and Urea"
self.modvar_soil_emissions_n2o_crop_residues = ":math:\\text{N}_2\\text{O} Emissions from Crop Residues"
self.modvar_soil_emissions_n2o_fertilizer = ":math:\\text{N}_2\\text{O} Emissions from Fertilizer Use"
self.modvar_soil_emissions_n2o_mineral_soils = ":math:\\text{N}_2\\text{O} Emissions from Mineral Soils"
self.modvar_soil_emissions_n2o_organic_soils = ":math:\\text{N}_2\\text{O} Emissions from Organic Soils"
self.modvar_soil_emissions_n2o_ppr = ":math:\\text{N}_2\\text{O} Emissions from Paddock Pasture and Range"
self.modvar_soil_frac_n_lost_leaching = "Leaching Fraction of N Lost"
self.modvar_soil_frac_n_lost_volatilisation_on = "Volatilisation Fraction from Organic Amendments and Fertilizers"
self.modvar_soil_frac_n_lost_volatilisation_sn_non_urea = "Volatilisation Fraction from Non-Urea Synthetic Fertilizers"
self.modvar_soil_frac_n_lost_volatilisation_sn_urea = "Volatilisation Fraction from Urea Synthetic Fertilizers"
self.modvar_soil_frac_organic_soils_drained = "Fraction of Organic Soils Drained"
self.modvar_soil_frac_soc_lost = "Fraction of SOC Lost in Cropland"
self.modvar_soil_frac_synethic_fertilizer_urea = "Fraction Synthetic Fertilizer Use Urea"
self.modvar_soil_fertuse_synthetic = "Initial Synthetic Fertilizer Use"
self.modvar_soil_organic_c_stocks = "Soil Organic C Stocks"
self.modvar_soil_ratio_c_to_n_soil_organic_matter = "C to N Ratio of Soil Organic Matter"
self.modvar_soil_qtyinit_liming_dolomite = "Initial Liming Dolomite Applied to Soils"
self.modvar_soil_qtyinit_liming_limestone = "Initial Liming Limestone Applied to Soils"

self.modvar_soil_emissions_
self.modvar_soil_ef_c_liming_dolomite = "C Liming Emission Factor Dolomite"
self.modvar_soil_ef_c_liming_limestone = "C Liming Emission Factor Limestone"
self.modvar_soil_qtyinit_liming_dolomite = "Initial Liming Dolomite Applied to Soils"
self.modvar_soil_qtyinit_liming_limestone = "Initial Liming Limestone Applied to Soils"
self.modvar_soil_demscalar_liming = "Liming Demand Scalar"

self.modvar_agrc_frac_dry_matter_in_crop = "Dry Matter Fraction of Harvested Crop"
self.modvar_agrc_n_content_of_above_ground_residues = "N Content of Above Ground Residues"
self.modvar_agrc_n_content_of_below_ground_residues = "N Content of Below Ground Residues"
self.modvar_agrc_ratio_above_ground_residue_to_harvested_yield = "Ratio of Above Ground Residue to Harvested Yield"
self.modvar_agrc_ratio_below_ground_biomass_to_above_ground_biomass = "Ratio of Below Ground Biomass to Above Ground Biomass"
self.modvar_agrc_fraction_residues_removed = "Fraction of Residues Removed"
self.modvar_agrc_fraction_residues_burned = "Fraction of Residues Burned"
self.modvar_agrc_regression_m_above_ground_residue = "Above Ground Residue Dry Matter Slope"
self.modvar_agrc_regression_b_above_ground_residue = "Above Ground Residue Dry Matter Intercept"
