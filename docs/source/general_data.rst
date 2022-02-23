============
General Data
============


Gasses
------
Emissions are calculated in a unit of mass (default MT) for each relevant gas included. For :math:`\text{CO}_2\text{e}` conversions, the default Global Warming Potential (GWP) time horizon is 100 years. However, the GWP time horizon can be changed in the `Analytical Parameters <../analytical_parameters.html>`_ configuration file. The GWP conversion factors below are taken from `IPCC AR6 WG1 Chapter 7 - Table 7.SM.7 <https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter_07_Supplementary_Material.pdf>`_.

.. csv-table:: Gasses potentially included in LAC-IDPM and their CO2 equivalent
   :file: ./csvs/attribute_gas.csv
   :header-rows: 1

Units - Mass
------------
The emissions accounting mass can be set in the configuration file (see the in the `Analytical Parameters <../analytical_parameters.html>`_ section for more information about configuration parameters). The default unit of mass for reporting emissions is MT (megatons).

   .. csv-table:: Mass units used in the model and relationships between them.
      :file: ./csvs/attribute_mass.csv
      :header-rows: 1


Sectors and Subsectors
----------------------
LAC-IPDM models emissions in four key sectors: AFOLU, Circular Economy, Energy, and IPPU. Additional, emissions are driven by activity in the Socioeconomic sector.

.. csv-table:: Emissions sectors in LAC-IDPM
   :file: ./csvs/attribute_sector.csv
   :header-rows: 1

Each of the four key emissions sectors and the socioeconomic sector are divided into several subsectors, which are detailed below.

.. csv-table:: Subsectors modeled in LAC-IDPM
   :file: ./csvs/attribute_subsector.csv
   :header-rows: 1

Regions (Countries)
-------------------

The **MODELNAMEHERE** encompasses 26 countries, or, more generally, regions. These regions are associated with different NDCs, power grids, governmental structures and political regimes. Each region can be run independently for all python models, though the NemoMod model, which is designed to incorporate regional power sharing, has to be run at once.

.. csv-table:: The following REGION dimensions are specified for the **MODELNAMEHERE** NemoMod model.
   :file: ./csvs/attribute_cat_region.csv
   :header-rows: 1
