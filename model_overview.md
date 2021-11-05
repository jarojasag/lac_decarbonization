#How to Use this Documentation

This documentation provides a detailed look at data requirements and interactions between models in the Latin American—Integrated Decarbonization Pathways Model (LAC-IDPM).

### Metavariables and Constructing Input Parameters
This document makes use of the \$VARNAME\$ notation to denote metavariables for parameter and model input variable name.

For example, model input variables used to denote agricutlural activity emission factors by crop type and gas in `model_input_variables.csv` may have the following structure:
`$CAT-CROP$_ef_$UNIT-MASS$_$EMISSION-GAS$_$UNIT-AREA$`, where

- `$CAT-CROP$` is the categorical crop type (e.g., pineapple, banana, coffee, etc.);
- `$EMISSION-GAS$` is the greenhouse gas that is being emitted (e.g. CO$_2$, N$_2$O, CH$_4$)
- `$UNIT-MASS$` is the unit of mass for gas emission (e.g., `kg` for kilograms; some sector variables may use `gg` for gigagrams);
- `$UNIT-AREA$` is the area unit (e.g., `ha` is hectares).

As a tangible example, the CO$_2$ emission factor for banana production, which captures crop burning, decomposition, and other factors, would be entered as `banana_ef_kg_co2_ha` since, in this case, `$CAT-CROP$ = banana`, `$EMISSION-GAS$ = co2`, `$UNIT-AREA$ = ha`, `$UNIT-MASS$ = kg`. Similarly, the N$_2$O factor, which includes crop liming and fertilization, would be captured as `banana_ef_kg_n2o_ha`.

These are _metavariables_, which characterize and describe the notation for naming model input variables. Each variable is associated with some _schema_. In the following sections, data requirements and the associated naming schema for `model_input_variables.csv` are 


#Preliminary Variable Estimates for Calibration

### continue description here....
<br><br>

#Entering Variable Trajectories for Model Runs

The input sheet `model_input_variables.csv` stores input variables trajectories for all variables and parameters that are included in the 
### continue description here....
<br><br>

#Data Requirements


## Emission Attributes and Information

| Gas | Name | CO2 equivalent factor | `$EMISSION-GAS$` |
| --------- | --------- | --------- | ----------- |
| CH4 | Methane | 25 | `ch4` |
| CO2 | Carbon Dioxide | 1 | `co2` |
| N2O | Nitrus Oxide | 310 | `n2o` |
| Refrigerants... | | | |
### continue description here and more gasses....
<br>


## General Data (Cross Sector) by Country

### General Country Attributes and Information

The following variables are required for each country.

| Variable | Information | Variable Schema |
| --------- | --------- | ----------- |
| Area of Country | Units: Hectares (ha) | `area_country_ha` |
| Urban Population | Units: # of people | `population_urban` |
| Rural Population | Units: # of people | `population_rural` |
### continue description here and more attributes....
<br>

### Economic Attributes and Information

| Variable | Information | Variable Schema |
| --------- | --------- | ----------- |
| Gross Domestic Product | Units: Billion USD (2020\$) | `gdp_mmm_usd` |
| Value Added — ### | Units: Billion USD (2020\$) | `va_industry_mmm_usd` |
| Value Added — ### | Units: Billion USD (2020\$) | `va_e###` |

<br><br>






## AFOLU

### <u>Agriculture</u>

#### Categories
Agriculture is divided into the following categories (crops), given by `$CAT-AGRICULTURE$`. If a crop is not present in a country, set all associated factors to 0. 

| Category Name | `$CAT-AGRICULTURE$` | Description |
| --------- | --------- | ----------- |
| Banana (platano) | `banana` | Banana crops |
| Coffee (café) | `coffee` | Coffee crops |
| Grapes () | | Grape crops |
| Palm Plantations (palma) | `palm_oil` | Palm plantations for palm oil |
| Pineapple (piña) | `pineapple` | Pineapple crops |
| Rice (arroz) | `rice` | Rice crops |
| Sugar Cane (cano) | `sugar_cane` | Sugar cane crops|
| Soybeans () | `soy` | Soy crops|
| Wheat (harina) | `wheat` | Wheat crops|

### continue additional crop types, will look through FAO


#### Information by Category

For each agricultural category, trajectories of the following variables are needed. 

| Variable | Information | Variable Schema |
| --------- | --------- | ----------- |
| Area proportion | Fraction of total country area (%) | `frac_lu_$CAT-AGRICULTURE$` |
| CH4 Emission Factor | Annual average CH4 (methane) emitted per ha of crop grown. <b>RICE is the only crop this is needed for. This will be 0 for most crops (or negligible).</b> | `$CAT-FOREST$_ef_seq_$UNIT-MASS$_co2_$UNIT-AREA$` (`$EMISSION-GAS$ = co2`)|
| CO2 Emission Factor | Annual average CO2 (carbon dioxide) emitted per ha of crop; for the purposes of accounting and calibration, this includes the following categories: crop burning, ##CONTINUE LISTING HERE## | `$CAT-CROP$_ef_ff_$UNIT-MASS$_co2_$UNIT-AREA$` (`$EMISSION-GAS$ = co2`)|
| N20 Emission Factor | Annual average N2O (nitrous oxide) emitted per ha due to forest fires (≥ 0) | `$CAT-CROP$_ef_ff_$UNIT-MASS$_co2_$UNIT-AREA$` (`$EMISSION-GAS$ = co2`)|
<br>

### <u>Forestry</u>

#### Categories
Forest should divided into the following categories, given by `$CAT-FOREST$`.

| Category Name | `$CAT-FOREST$` | Definition |
| --------- | --------- | ----------- |
| Mangroves | `mangroves` | |
| Primary Wet Forest | `primary_wet` | |
| Primary Dry Forest | `primary_dry` | |
| Secondary Wet Forest | `secondary_wet` | |
| Secondary Dry Forest | `secondary_dry` | |


#### Data Requirements by Category
For each forest category, the following variables are needed. 

| Variable | Information | Variable Schema |
| --------- | --------- | ----------- |
| Area proportion | Fraction of total country area (%) | `frac_lu_$CAT-FOREST$` |
| Sequestration Emission Factor | Annual average CO2 emitted per ha from sequestration (< 0 – this is a negative number) | `$CAT-FOREST$_ef_seq_$UNIT-MASS$_co2_$UNIT-AREA$` (`$EMISSION-GAS$ = co2`)|
| Forest Fire Emission Factor | Annual average CO2 emitted per ha due to forest fires (≥ 0) | `$CAT-CROP$_ef_ff_$UNIT-MASS$_co2_$UNIT-AREA$` (`$EMISSION-GAS$ = co2`)|
<br>

### <u>Land Use</u>

#### Categories
Land use should be divided into the following categories, given by `$CAT-LANDUSE$`. Note that the sum of land use area across all land use, forestry, agriculture, and livestock land use categories should equal the total amount of land available in the country. 

| Category Name | `$CAT-LANDUSE$` | Definition |
| --------- | --------- | ----------- |
| Settlements | `settlement` | Area of land devoted to urban/suburban development |
| Other | `other` | Other land use categories |
| Wetlands | `wetlands` | |
| Mangroves | `secondary_dry` | |


#### Data Requirements by Category
For each forest category, the following variables are needed. 

| Variable | Information | Variable Schema |
| --------- | --------- | ----------- |
| Area proportion | Fraction of total country area (%) | `frac_lu_$CAT-FOREST$` |
| Sequestration Emission Factor | Annual average CO2 emitted per ha from sequestration (< 0 – this is a negative number) | `$CAT-FOREST$_ef_seq_$UNIT-MASS$_co2_$UNIT-AREA$` (`$EMISSION-GAS$ = co2`)|
| Forest Fire Emission Factor | Annual average CO2 emitted per ha due to forest fires (≥ 0) | `$CAT-CROP$_ef_ff_$UNIT-MASS$_co2_$UNIT-AREA$` (`$EMISSION-GAS$ = co2`)|
<br>

### <u>Livestock</u>

`https://www.epa.gov/sites/default/files/2020-10/documents/ag_module_users_guide.pdf`
<br><br>

## Energy
Same thing as AFLO, but for energy

### <u>Buildings</u>
<br>

### <u>Electricity Generation</u>
<br>

### <u>Industrial Energy</u>
<br>

### <u>Transportation</u>
<br><br>


## Industrial Processes and Product Use

### <u>Industrial Processes and Product Use</u>

<br><br>

## Circular Economy

### <u>Domestic Liquid Waste</u>
<br>

### <u>Domestic Solid Waste</u>
<br>

### <u>Industrial Liquid Waste</u>
<br>

### <u> Industrial Solid Waste</u>

<br><br>




#Data Glossary

### Aggregate, easy to read variable table here for quick reference... See below...

## Summary of metavariables

etc.

* `EMISSION-GAS`
* `FUEL`
* `CAT-AGRICULTURE`
* `CAT-FOREST`
* `CAT-INDUSTRY`
* `CAT-LANDUSE`
* `CAT-LIVESTOCK`
* `CAT-WASTE-LIQUID`
* `CAT-WASTE-SOLID`
* `TECHNOLOGY`
* `UNIT-AREA`
* `UNIT-MASS `

## Summary of variables required

| Sector | Subsector | Variable | Information | Variable Schema | Varies by |
| --------- | --------- | --------- | --------- | ----------- | ----------- |
| All | - | Area of Country | Units: Hectares (ha) | `area_country_ha` | - |
| All | - | Urban Population | Units: # of people | `population_urban` | - |
| All | - | Rural Population | Units: # of people | `population_rural` | - |
| AFOLU | Forestry | Area proportion by crop type | Fraction of total country area (%) | `frac_lu_$CAT-FOREST$` | `$CAT-FOREST$` |
| AFOLU | Forestry | Sequestration Emission Factor | Annual average CO2 emitted per ha from sequestration (< 0 – this is a negative number) | `$CAT-FOREST$_ef_seq_$UNIT-MASS$_co2_$UNIT-AREA$` (`$EMISSION-GAS$ = co2`)| `$CAT-FOREST$` |
| AFOLU | Forestry | Forest Fire Emission Factor | Annual average CO2 emitted per ha due to forest fires (≥ 0) | `$CAT-CROP$_ef_ff_$UNIT-MASS$_co2_$UNIT-AREA$` (`$EMISSION-GAS$ = co2`)| `$CAT-FOREST$` |