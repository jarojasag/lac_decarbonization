===============================
Energy - Electricity Generation
===============================

The electricity sector is modeled using `NemoMod <https://sei-international.github.io/NemoMod.jl/stable/>`_ (`access the Julia GitHub repository here <https://github.com/sei-international/NemoMod.jl/>`_), an energy framework developed by the `Stockholm Environmental Institute <https://www.sei.org>`_. However, the SISEPUEDE model introduces a basic model, guided by simple assumptions, that can be built and improved upon by countries with deeper systemic knowledge at a later date. In general, SISEPUEDE acts as a wrapper for Julia, formatting input data and integrating uncertainty into an integrated modeling framework for NemoMod.

NemoMod requires several dimensions of data; these data include:

* EMISSIONS
* FUEL
* MODE
* STORAGE
* TECHNOLOGY
* TIMESLICE
* TSGROUP1
* TSGROUP2
* YEARS

These dimensions are associated with attribute tables in SISEPUEDE scripts and the ModelAttributes class, and some--such as FUEL, STORAGE, and TECHNOLOGY--are subsectors in SISEPUEDE.

.. note::
   Most of the variables that are required by category are explained in further detail in the `NemoMod Parameter Documentation <https://sei-international.github.io/NemoMod.jl/stable/parameters/>`_. For example, if it is unclear what the *Capacity Factor* is (see Categories - TECHNOLOGY below), the NemoMod parameter documentation can provide additional information.

Categories - Fuel
-----------------

Fuel is cross-cutting, affecting all energy sectors. It is

.. csv-table:: The following FUEL dimensions are specified for the SISEPUEDE NemoMod model.
   :file: ./csvs/attribute_cat_fuel.csv
   :header-rows: 1

Variables by Categories - FUEL
------------------------------

The following variables are required for each category ``$CAT-FUEL$``.

----


Categories - Region
-----------------------

NemoMod allows users to specify regions, and policies can be modeled that represent cross-regional power transfers, storage, etc. In the SISEPUEDE NemoMod implementation, each country is treated as a region.



Variables by Categories - Region
------------------------------------

here

----


Categories - Storage
-----------------------

Storage
.. csv-table:: The following STORAGE dimensions are specified for the SISEPUEDE NemoMod model.
   :file: ./csvs/attribute_cat_storage.csv
   :header-rows: 1

Variables by Categories - STORAGE
------------------------------------



Categories - TECHNOLOGY
-----------------------

The SISEPUEDE model (v1.0) uses NemoMod *only* to model the electricity sector. Therefore, technologies are limited to power generation (power plants) and storage.

.. csv-table:: The following TECHNOLOGY dimensions are specified for the SISEPUEDE NemoMod model.
   :file: ./csvs/attribute_cat_technology.csv
   :header-rows: 1

Variables by Categories - TECHNOLOGY
------------------------------------

The following variables are required for each category ``$CAT-TECHNOLOGY$``. Note that these technologies represent consumers of fuel (which includes electricity) and are generally power plants and storage.

----
