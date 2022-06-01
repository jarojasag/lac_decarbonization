======
Energy
======


Energy includes a range of variables and categories. Given the integrated nature of the energy sector, there are several "cross-subsector" categories required to construct the NemoMod energy model. These categories include Fuels and Technologies. The dimensions required for the NemoMod framework are available from the `NemoMod Categories Documentation <https://sei-international.github.io/NemoMod.jl/stable/dimensions/>`_.


Electricity Generation
======================

The electricity sector is represented one of the most complex modeling approaches include in the **MODELNAMEHERE** framework. It is modeled using `NemoMod <https://sei-international.github.io/NemoMod.jl/stable/>`_ (`access the Julia GitHub repository here <https://github.com/sei-international/NemoMod.jl/>`_), an energy framework developed by the `Stockholm Environmental Institute <https://www.sei.org>`_. However, the **MODELNAMEHERE** model introduces a basic model, guided by simple assumptions, that can be built and improved upon by countries with deeper systemic knowledge at a later date.

NemoMod requires several dimensions of data; these data include FUEL, TECHNOLOGY, EMISSIONS, YEARS, STORAGE, TIMESLICE. These dimensions are treated as subsectors in **MODELNAMEHERE** in scripts and attribute table structure.

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





Stationary Combustion and Other Energy (SCOE)
=============================================


.. note:: | Energy efficiency factor relative for delivery of heat energy using coal versus delivery using electricity. Represents technological efficiency for the system of heat energy delivery.
          |
          | For example, a value of 0.8 would indicate that 80% of the input energy results in output energy (e.g., 1.25 TJ becomes 1 TJ) by final delivery, while a value of 1 would indicate perfect efficiency (1 TJ in  :math:`\implies` 1 TJ out)
          |
          | At time :math:`t = 0`, the efficiencies are used to calculate an end-user demand for energy, which elasticities are applied to to estimate an output demand. In subsequent time steps, as the mix of energy use changes, input energy demands are calculated using the efficiency factors of different mixes of fuels.


Variables by Partial Category
-----------------------------

SCOE (**S**\tationary **C**\tombustion and **O**\tther **E**\tnergy) captures stationary emissions in buildings (split out by differing drivers) and other emissions not captured elsewhere. SCOE requires the following variables.

.. csv-table:: For different SCOE categories, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_partial_category_en_scoe.csv
   :header-rows: 1


Categories
----------

SCOE is divided into the following categories, which

.. csv-table:: Other categories (``$CAT-SCOE$`` attribute table)
   :file: ./csvs/attribute_cat_scoe.csv
   :header-rows: 1

----




Variables by Category
---------------------

Categories
----------


----

Industrial Energy
=================

Industrial energy includes emission from **DESCRIPTION**

Variables by Category
---------------------

For each industrial category ``$CAT-INDUSTRY$``, the following variables are required.

.. csv-table:: For different SCOE categories, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_category_en_inen.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------


.. csv-table:: For different Industrial categories, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_partial_category_en_inen.csv
   :header-rows: 1


Categories
----------
Industrial categories are described in `Industial Processes and Product Use (IPPU) <../ippu.html>`_.

----

Transportation
==============

Known Issues
------------

**Discuss how variables that are set in Transportation have to be added to the NonElectricEnergy class as well**


Variables by Category
---------------------

.. note::
   :math:`\text{CH}_4` and :math:`\text{N}_4\text{O}` emissions from mobile combustion of fuels are highly dependent on the technologies (e.g., types of cars) that use the fuels. Therefore, emission factors for mobile combustion of fuels are contained in the Transportation subsector instead of the Energy Fuels subsector. See Section Volume 2, Chapter 3, Section 3.2.1.2 of the `2006 IPCC Guidelines for National Greenhouse Gas Inventories <https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_3_Ch3_Mobile_Combustion.pdf>`_ for more information.

For each industrial category ``$CAT-TRANSPORTATION$``, the following variables are required.

.. csv-table:: For different SCOE categories, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_category_en_trns.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------


.. csv-table:: For different Transportation categories, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_partial_category_en_trns.csv
   :header-rows: 1


Categories
----------



Transportation Demand
=====================

Transportation demand is broken into its own subsector given some of the complexities that drive transportation demand. The **MODELNAME** transportation demand subsector allows for more complex interactions--e.g., interactions with industrial production, growth in tourism, waste collection, and imports and exports--to be integrated, though these are not dealt with explicitly at this time.

Variables by Category
---------------------

Categories
----------
