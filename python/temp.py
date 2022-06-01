# Stationary Combustion and Other Energy variables
self.modvar_scoe_deminit_energy_demand_per_hh_elec = "SCOE Initial Per Household Demand for Electric Appliances"
self.modvar_scoe_deminit_energy_demand_per_hh_heat = "SCOE Initial Per Household Demand for Heat Energy"
self.modvar_scoe_deminit_energy_demand_per_mmmgdp_elec = "SCOE Initial Per GDP Demand for Electric Appliances"
self.modvar_scoe_deminit_energy_demand_per_mmmgdp_heat = "SCOE Initial Per GDP Demand for Heat Energy"
self.modvar_scoe_efficiency_fact_heat_en_coal = "SCOE Efficiency Factor for Heat Energy from Coal"
self.modvar_scoe_efficiency_fact_heat_en_diesel = "SCOE Efficiency Factor for Heat Energy from Diesel"
self.modvar_scoe_efficiency_fact_heat_en_electricity = "SCOE Efficiency Factor for Heat Energy from Electricity"
self.modvar_scoe_efficiency_fact_heat_en_gasoline = "SCOE Efficiency Factor for Heat Energy from Gasoline"
self.modvar_scoe_efficiency_fact_heat_en_hydrogen = "SCOE Efficiency Factor for Heat Energy from Hydrogen"
self.modvar_scoe_efficiency_fact_heat_en_kerosene = "SCOE Efficiency Factor for Heat Energy from Kerosene"
self.modvar_scoe_efficiency_fact_heat_en_natural_gas = "SCOE Efficiency Factor for Heat Energy from Natural Gas"
self.modvar_scoe_efficiency_fact_heat_en_pliqgas = "SCOE Efficiency Factor for Heat Energy from Petroleum Liquid Gas"
self.modvar_scoe_efficiency_fact_heat_en_solid_biomass = "SCOE Efficiency Factor for Heat Energy from Solid Biomass"
self.modvar_scoe_elasticity_hh_energy_demand_electric_to_gdppc = "SCOE Elasticity of Per Household Electrical Applicance Demand to GDP Per Capita"
self.modvar_scoe_elasticity_hh_energy_demand_heat_to_gdppc = "SCOE Elasticity of Per Household Heat Energy Demand to GDP Per Capita"
self.modvar_scoe_elasticity_mmmgdp_energy_demand_elec_to_gdppc = "SCOE Elasticity of Per GDP Electrical Applicance Demand to GDP Per Capita"
self.modvar_scoe_elasticity_mmmgdp_energy_demand_heat_to_gdppc = "SCOE Elasticity of Per GDP Heat Energy Demand to GDP Per Capita"
self.modvar_scoe_emissions_ch4 = ":math:\\text{CH}_4 Emissions from SCOE"
self.modvar_scoe_emissions_co2 = ":math:\\text{CO}_2 Emissions from SCOE"
self.modvar_scoe_emissions_n2o = ":math:\\text{N}_2\text{O} Emissions from SCOE"
self.modvar_scoe_energy_demand_electricity = "Total Electrical Energy Demand from SCOE"
self.modvar_scoe_energy_demand_heat_agg = "Total Non-Electrical Heat Energy Demand from SCOE"
self.modvar_scoe_energy_demand_heat_coal = "SCOE Heat Energy Demand Coal"
self.modvar_scoe_energy_demand_heat_diesel = "SCOE Heat Energy Demand Diesel"
self.modvar_scoe_energy_demand_heat_electricity = "SCOE Heat Energy Demand Electricity"
self.modvar_scoe_energy_demand_heat_gasoline = "SCOE Heat Energy Demand Gasoline"
self.modvar_scoe_energy_demand_heat_hydrogen = "SCOE Heat Energy Demand Hydrogen"
self.modvar_scoe_energy_demand_heat_kerosene = "SCOE Heat Energy Demand Kerosene"
self.modvar_scoe_energy_demand_heat_natural_gas = "SCOE Heat Energy Demand Natural Gas"
self.modvar_scoe_energy_demand_heat_pliq_gas = "SCOE Heat Energy Demand Petroleum Liquid Gas"
self.modvar_scoe_energy_demand_heat_biomass = "SCOE Heat Energy Demand Solid Biomass"
self.modvar_scoe_frac_heat_en_coal = "SCOE Fraction Heat Energy Demand Coal"
self.modvar_scoe_frac_heat_en_diesel = "SCOE Fraction Heat Energy Demand Diesel"
self.modvar_scoe_frac_heat_en_electricity = "SCOE Fraction Heat Energy Demand Electricity"
self.modvar_scoe_frac_heat_en_gasoline = "SCOE Fraction Heat Energy Demand Gasoline"
self.modvar_scoe_frac_heat_en_hydrogen = "SCOE Fraction Heat Energy Demand Hydrogen"
self.modvar_scoe_frac_heat_en_kerosene = "SCOE Fraction Heat Energy Demand Kerosene"
self.modvar_scoe_frac_heat_en_natural_gas = "SCOE Fraction Heat Energy Demand Natural Gas"
self.modvar_scoe_frac_heat_en_pliqgas = "SCOE Fraction Heat Energy Demand Petroleum Liquid Gas"
self.modvar_scoe_frac_heat_en_solid_biomass = "SCOE Fraction Heat Energy Demand Solid Biomass"
# fuel fractions to check summation over (keys)
self.modvar_scoe_dict_fuel_fractions_to_efficiency_factors = [
    self.modvar_scoe_frac_heat_en_coal: self.modvar_scoe_efficiency_fact_heat_en_coal,
    self.modvar_scoe_frac_heat_en_diesel: self.modvar_scoe_efficiency_fact_heat_en_diesel,
    self.modvar_scoe_frac_heat_en_electricity: self.modvar_scoe_efficiency_fact_heat_en_electricity,
    self.modvar_scoe_frac_heat_en_gasoline: self.modvar_scoe_efficiency_fact_heat_en_gasoline,
    self.modvar_scoe_frac_heat_en_hydrogen: self.modvar_scoe_efficiency_fact_heat_en_hydrogen,
    self.modvar_scoe_frac_heat_en_kerosene: self.modvar_scoe_efficiency_fact_heat_en_kerosene,
    self.modvar_scoe_frac_heat_en_natural_gas: self.modvar_scoe_efficiency_fact_heat_en_natural_gas,
    self.modvar_scoe_frac_heat_en_pliqgas: self.modvar_scoe_efficiency_fact_heat_en_pliqgas,
    self.modvar_scoe_frac_heat_en_solid_biomass: self.modvar_scoe_efficiency_fact_heat_en_solid_biomass
]
