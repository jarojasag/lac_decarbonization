======
Energy
======


Energy includes a range of variables and categories. Given the integrated nature of the energy sector, there are several "cross-subsector" categories required to construct the NemoMod energy model. These categories include Fuels and Technologies. The dimensions required for the NemoMod framework are available from the `NemoMod Categories Documentation <https://sei-international.github.io/NemoMod.jl/stable/dimensions/>`_.


Electricity Generation
======================

The electricity sector is represented one of the most complex modeling approaches include in the **MODELNAMEHERE** framework. It is modeled using `NemoMod <https://sei-international.github.io/NemoMod.jl/stable/>`_ (`access the Julia GitHub repository here <https://github.com/sei-international/NemoMod.jl/>`_), an energy framework developed by the `Stockholm Environmental Institute <https://www.sei.org>`_. However, the **MODELNAMEHERE** model introduces a basic model, guided by simple assumptions, that can be built and improved upon by countries with deeper systemic knowledge at a later date.

NemoMod requires several dimensions of data; these data include FUEL, TECHNOLOGY, EMISSIONS, YEARS, STORAGE, TIMESLICE.

.. note::
   Most of the variables that are required by category are explained in further detail in the `NemoMod Parameter Documentation <https://sei-international.github.io/NemoMod.jl/stable/parameters/>`_. For example, if it is unclear what the *Capacity Factor* is (see Categories - TECHNOLOGY below), the NemoMod parameter documentation can provide additional information.

Categories - FUEL
-----------------

.. csv-table:: The following FUEL dimensions are specified for the **MODELNAMEHERE** NemoMod model.
   :file: ./csvs/attribute_cat_fuel.csv
   :header-rows: 1

Variables by Categories - FUEL
------------------------------

The following variables are required for each category ``$CAT-FUEL$``.

----

Categories - REGION
-----------------------

NemoMod allows users to specify regions, and policies can be modeled that represent cross-regional power transfers, storage, etc. In the **MODELNAMEHERE** NemoMod implementation, each country is treated as a region.



Variables by Categories - REGION
------------------------------------

here

----


Categories - STORAGE
-----------------------

.. csv-table:: The following STORAGE dimensions are specified for the **MODELNAMEHERE** NemoMod model.
   :file: ./csvs/attribute_cat_storage.csv
   :header-rows: 1

Variables by Categories - STORAGE
------------------------------------



Categories - TECHNOLOGY
-----------------------

The **MODELNAMEHERE** model (v1.0) uses NemoMod *only* to model the electricity sector. Therefore, technologies are limited to power generation (power plants) and storage.

.. csv-table:: The following TECHNOLOGY dimensions are specified for the **MODELNAMEHERE** NemoMod model.
   :file: ./csvs/attribute_cat_technology.csv
   :header-rows: 1

Variables by Categories - TECHNOLOGY
------------------------------------

The following variables are required for each category ``$CAT-TECHNOLOGY$``. Note that these technologies represent consumers of fuel (which includes electricity) and are generally power plants and storage.





Other Energy: Stationary Emissions and Carbon Capture and Sequestration (CCS)
=============================================================================

Variables by Partial Category
-----------------------------

OESC (**O**\ther **E**\nergy: **S**\tationary emissions and **C**\arbon capture and sequestration) captures energy emissions and sequestration not captured elsewhere in energy, including stationary emissions in buildings (split out by differing drivers) and carbon capture and sequestration technology. OESC requires the following variables.

.. csv-table:: For different OESC categories, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_partial_category_en_oesc.csv
   :header-rows: 1


Categories
----------

OESC is divided into the following categories, which

.. csv-table:: Other categories (``$CAT-OESC$`` attribute table)
   :file: ./csvs/attribute_cat_oesc.csv
   :header-rows: 1

----




Variables by Category
---------------------

Categories
----------


----

Industrial Energy
=================

Variables by Category
---------------------

For each industrial category ``$CAT-INDUSTRY$``, the following variables are required.

Categories
----------
Industrial categories are described in `Industial Processes and Product Use (IPPU) <../ippu.html>`_.

----

Transportation
==============

Variables by Category
---------------------

Categories
----------
