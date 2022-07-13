======
Energy
======

Energy includes a range of variables and categories. Given the integrated nature of the energy sector, there are several "cross-subsector" categories required to construct the NemoMod energy model. These categories include Fuels and Technologies. The dimensions required for the NemoMod framework are available from the `NemoMod Categories Documentation <https://sei-international.github.io/NemoMod.jl/stable/dimensions/>`_.


Modeling Concepts and Important Notes
=====================================

Fuels and Heat Energy
---------------------

In general, energy is produced by stationary or mobile combustion of different fuels. The combustion of fuels releases :math:`\text{CO}_2`, :math:`\text{CH}_4`, and :math:`\text{N}_2\text{O}` (and other gasses, which may not be captured). These fuels are utilized by different technologies, which may use fuels at different efficiencies. Energy can also be stored (specifically electricity). The term **fuel** is explicitly used in all energy subsectors, while **technology** and **storage** are used in the NemoMod electricity model.

The combination of fuel and efficiency is an important concept for entering input data. Energy use in SCOE and CCSQ both use a fraction of energy *demand at point-of-use* to project future changes in fuel mixtures. However, many empirical data that are used rely on energy *consumption*. In general, where :math:`D_t` is the demand at time :math:`t`, :math:`C_t` is consumption at time :math:`t`, :math:`\alpha_t \in \mathbb{R}^n` is the vector of fuel mix fractions of *demand* at time :math:`t` for :math:`n` fuels, and :math:`\e_t \in \mathbb{R}^n` is the vector of fuel-technology average efficiencies at time :math:`t` for :math:`n` fuels, the demand is

:math:`D_t = C_t(\alpha_t \cdot e_t)`.

Therefore, where :math:`\hat{\alpha}_t \in \mathbb{R}^n` is the vector of fuel mix fractions of *consumption* at time :math:`t` for :math:`n` fuels, the fraction at point-of-use demand for fuel :math:`i` should be entered as

:math:`\alpha_{ti} = \frac{\alpha_{ti}\e_{ti}}{\alpha_t \cdot e_t}`,

i.e., the point-of-use demand is the efficiency-weighted fraction of consumption. For more information on the energy models' mathematical specification, see the `Mathematical Documentation of Energy Models <./mathdoc_energy.html>`_.

----


Electricity Generation
======================

The electricity sector is represented one of the most complex modeling approaches include in the SISEPUEDE framework. It is modeled using `NemoMod <https://sei-international.github.io/NemoMod.jl/stable/>`_ (`access the Julia GitHub repository here <https://github.com/sei-international/NemoMod.jl/>`_), an energy framework developed by the `Stockholm Environmental Institute <https://www.sei.org>`_. However, the SISEPUEDE model introduces a basic model, guided by simple assumptions, that can be built and improved upon by countries with deeper systemic knowledge at a later date.

NemoMod requires several dimensions of data; these data include FUEL, TECHNOLOGY, EMISSIONS, YEARS, STORAGE, TIMESLICE. These dimensions are treated as subsectors in SISEPUEDE in scripts and attribute table structure.

.. note::
   Most of the variables that are required by category are explained in further detail in the `NemoMod Parameter Documentation <https://sei-international.github.io/NemoMod.jl/stable/parameters/>`_. For example, if it is unclear what the *Capacity Factor* is (see Categories - TECHNOLOGY below), the NemoMod parameter documentation can provide additional information.

Categories - FUEL
-----------------

.. csv-table:: The following FUEL dimensions are specified for the SISEPUEDE NemoMod model.
   :file: ./csvs/attribute_cat_fuel.csv
   :header-rows: 1

Variables by Categories - FUEL
------------------------------

The following variables are required for each category ``$CAT-FUEL$``.

----


Categories - REGION
-----------------------

NemoMod allows users to specify regions, and policies can be modeled that represent cross-regional power transfers, storage, etc. In the SISEPUEDE NemoMod implementation, each country is treated as a region.



Variables by Categories - REGION
------------------------------------

here

----


Categories - STORAGE
-----------------------

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



Stationary Combustion and Other Energy (SCOE)
=============================================

SCOE (**S**\tationary **C**\tombustion and **O**\tther **E**\tnergy) captures stationary emissions in buildings (split out by differing drivers) and other emissions not captured elsewhere. SCOE requires the following variables.

.. note:: | Energy efficiency factor represents the technological efficiency for the system of heat energy delivery. Some system/fuels may conserve energy more efficiently than others.
          |
          | For example, a value of 0.8 would indicate that 20% (1 - 0.8) of the input energy to the system (e.g., for heating, cooking, water heaters, etc.) is lost (e.g., 1.25 TJ of input energy satisfies 1 TJ of end-use demand), while a value of 1 would indicate perfect efficiency (1 TJ in :math:`\implies` 1 TJ out)
          |
          | At time :math:`t = 0`, the efficiencies are used to calculate an end-user demand for energy, which elasticities are applied to to estimate a point-of-use demand. In subsequent time steps, as the mix of energy use changes, input energy demands are calculated using the efficiency factors of different mixes of fuels.


Variables by Category
---------------------

For each SCOE category ``$CAT-SCOE$``, the following variables are required.

.. csv-table:: For different SCOE categories, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_category_en_scoe.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------

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

Transportation consists of different categories (or modes) of transportation that are used to satisfy different types of demand. In general

Known Issues
------------

**Discuss how variables that are set in Transportation have to be added to the NonElectricEnergy class as well**


Variables by Category
---------------------

.. note::
   :math:`\text{CH}_4` and :math:`\text{N}_4\text{O}` emissions from mobile combustion of fuels are highly dependent on the technologies (e.g., types of cars) that use the fuels. Therefore, emission factors for mobile combustion of fuels are contained in the Transportation subsector instead of the Energy Fuels subsector. See Section Volume 2, Chapter 3, Section 3.2.1.2 of the `2006 IPCC Guidelines for National Greenhouse Gas Inventories <https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_3_Ch3_Mobile_Combustion.pdf>`_ for more information.

For each transportation category ``$CAT-TRANSPORTATION$``, the following variables are required.

.. csv-table:: For different Transportation categories, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_en_trns.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------

.. csv-table:: For different Transportation categories, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_partial_category_en_trns.csv
   :header-rows: 1


Categories
----------

Transportation is divided into the following categories. These categories are associated with different transportation demand categories (see below), which govern mode-shifting.

.. csv-table:: Other categories (``$CAT-TRANSPORTATION$`` attribute table)
   :file: ./csvs/attribute_cat_transportation.csv
   :header-rows: 1

----



Transportation Demand
=====================

Transportation demand is broken into its own subsector given some of the complexities that drive transportation demand (unlike other subsectors, like SCOE, that do not contain categorical mode-shifting within demands). The **MODELNAME** transportation demand subsector allows for more complex interactions--e.g., interactions with industrial production, growth in tourism, waste collection, and imports and exports--to be integrated, though these are not dealt with explicitly at this time.

For each transportation demand category ``$CAT-TRANSPORTATION-DEMAND$``, the following variables are required.

.. csv-table:: For different Transportation categories, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_en_trde.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------

.. csv-table:: For different Transportation Demand categories, trajectories of the following variables are needed. The category for which variables are required is denoted in the *categories* column.
   :file: ./csvs/table_varreqs_by_partial_category_en_trde.csv
   :header-rows: 1


Categories
----------

Transportation demand is divided into the following categories. These categories are associated with different allowable mode shifts between vehicle types.

.. csv-table:: Transportation Demand categories (``$CAT-TRANSPORTATION-DEMAND$`` attribute table)
   :file: ./csvs/attribute_cat_transportation_demand.csv
   :header-rows: 1
