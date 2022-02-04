General Data
============

Gasses
------
.. csv-table:: Gasses potentially included in LAC-IDPM and their CO2 equivalent
   :file: ./csvs/attribute_gas.csv
   :widths: 25, 35, 20, 20
   :header-rows: 1

Sectors and Subsectors
----------------------
LAC-IPDM models emissions in four key sectors: AFOLU, Circular Economy, Energy, and IPPU.

.. csv-table:: Emissions sectors in LAC-IDPM
   :file: ./csvs/attribute_sector.csv
   :widths: 20, 20, 20, 40
   :header-rows: 1

Each of the four key sectors are divided into several subsectors, detailed below.

.. csv-table:: Subsectors modeled in LAC-IDPM
   :file: ./csvs/attribute_subsector.csv
   :widths: 20, 20, 10, 40, 10
   :header-rows: 1

Regions (Countries)
-------------------

The **MODELNAMEHERE** encompasses 26 countries, or, more generally, regions. These regions are associated with different NDCs, power grids, governmental structures and political regimes. Each region can be run independently for all python models, though the NemoMod model, which is designed to incorporate regional power sharing, has to be run at once.

.. csv-table:: The following REGION dimensions are specified for the **MODELNAMEHERE** NemoMod model.
   :file: ./csvs/attribute_cat_region.csv
   :widths: 20, 20, 20, 20, 20
   :header-rows: 1


General Variables by Region
----------------------------
The following variables are required for each region (or country).

.. csv-table:: General variables required for each county
   :file: ./csvs/table_varreqs_by_country_general.csv
   :widths: 15, 20, 15, 10, 20, 10, 10
   :header-rows: 1


Economic Variables by Country
-----------------------------

.. csv-table:: Economic variables required for each county
   :file: ./csvs/table_varreqs_by_country_economic.csv
   :widths: 15, 20, 15, 10, 20, 10, 10
   :header-rows: 1
