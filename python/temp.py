self.modvar_fgtv_ef_ch4_distribution = ":math:\\text{CH}_4 FGTV Distribution Emission Factor"
self.modvar_fgtv_ef_ch4_production_flaring = ":math:\\text{CH}_4 FGTV Production Flaring Emission Factor"
self.modvar_fgtv_ef_ch4_production_fugitive = ":math:\\text{CH}_4 FGTV Production Fugitive Emission Factor"
self.modvar_fgtv_ef_ch4_production_venting = ":math:\\text{CH}_4 FGTV Production Venting Emission Factor"
self.modvar_fgtv_ef_ch4_transmission = ":math:\\text{CH}_4 FGTV Transmission Emission Factor"
self.modvar_fgtv_ef_co2_distribution = ":math:\\text{CO}_2 FGTV Distribution Emission Factor"
self.modvar_fgtv_ef_co2_production_flaring = ":math:\\text{CO}_2 FGTV Production Flaring Emission Factor"
self.modvar_fgtv_ef_co2_production_fugitive = ":math:\\text{CO}_2 FGTV Production Fugitive Emission Factor"
self.modvar_fgtv_ef_co2_production_venting = ":math:\\text{CO}_2 FGTV Production Venting Emission Factor"
self.modvar_fgtv_ef_co2_transmission = ":math:\\text{CO}_2 FGTV Transmission Emission Factor"
self.modvar_fgtv_ef_n2o_production_flaring = ":math:\\text{N}_2\\text{O} FGTV Production Flaring Emission Factor"
self.modvar_fgtv_ef_n2o_production_fugitive = ":math:\\text{N}_2\\text{O} FGTV Production Fugitive Emission Factor"
self.modvar_fgtv_ef_n2o_production_venting = ":math:\\text{N}_2\\text{O} FGTV Production Venting Emission Factor"
self.modvar_fgtv_ef_n2o_transmission = ":math:\\text{N}_2\\text{O} FGTV Transmission Emission Factor"
self.modvar_fgtv_ef_nmvoc_distribution = "NMVOC FGTV Distribution Emission Factor"
self.modvar_fgtv_ef_nmvoc_production_flaring = "NMVOC FGTV Production Flaring Emission Factor"
self.modvar_fgtv_ef_nmvoc_production_fugitive = "NMVOC FGTV Production Fugitive Emission Factor"
self.modvar_fgtv_ef_nmvoc_production_venting = "NMVOC FGTV Production Venting Emission Factor"
self.modvar_fgtv_ef_nmvoc_transmission = "NMVOC FGTV Transmission Emission Factor"
self.modvar_fgtv_emissions_ch4 = ":math:\\text{CH}_4 Fugitive Emissions"
self.modvar_fgtv_emissions_co2 = ":math:\\text{CO}_2 Fugitive Emissions"
self.modvar_fgtv_emissions_n2o = ":math:\\text{N}_2\\text{O} Fugitive Emissions"
self.modvar_fgtv_emissions_nmvoc = "NMVOC Fugitive Emissions"
self.modvar_fgtv_frac_non_fugitive_flared = "Fraction Non-Fugitive :math:\\text{CH}_4 Flared"
self.modvar_fgtv_frac_reduction_fugitive_leaks = "Reduction in Fugitive Leaks"


dict_emission_to_components = {
        self.modvar_fgtv_emissions_ch4: {
                "distribution": self.modvar_fgtv_ef_ch4_distribution ,
                "production_flaring": self.modvar_fgtv_ef_ch4_production_flaring,
                "production_fugitive": self.modvar_fgtv_ef_ch4_production_fugitive,
                "production_venting": self.modvar_fgtv_ef_ch4_production_venting,
                "transmission": self.modvar_fgtv_ef_ch4_transmission
        },

        self.modvar_fgtv_emissions_co2: {
                "distribution": self.modvar_fgtv_ef_co2_distribution,
                "production_flaring":self.modvar_fgtv_ef_co2_production_flaring,
                "production_fugitive": self.modvar_fgtv_ef_co2_production_fugitive,
                "production_venting": self.modvar_fgtv_ef_co2_production_venting,
                "transmission": self.modvar_fgtv_ef_co2_transmission
        },

        self.modvar_fgtv_emissions_n2o: {
                "distribution": None,
                "production_flaring": self.modvar_fgtv_ef_n2o_production_flaring,
                "production_fugitive": self.modvar_fgtv_ef_n2o_production_fugitive,
                "production_venting": self.modvar_fgtv_ef_n2o_production_venting,
                "transmission": self.modvar_fgtv_ef_n2o_transmission
        },

        self.modvar_fgtv_emissions_nmvoc: {
        "distribution": self.modvar_fgtv_ef_nmvoc_distribution,
        "production_flaring": self.modvar_fgtv_ef_nmvoc_production_flaring,
        "production_fugitive": self.modvar_fgtv_ef_nmvoc_production_fugitive,
        "production_venting": self.modvar_fgtv_ef_nmvoc_production_venting,
        "transmission": self.modvar_fgtv_ef_nmvoc_transmission
        }
}
