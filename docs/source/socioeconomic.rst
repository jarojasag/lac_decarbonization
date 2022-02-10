=============
Socioeconomic
=============

The socioeconomic sector represents the primary drivers of emissions. These include economic and demographic factors that influence all emissions sectors of the economy. These factors are treated as exogenous uncertainties.

General
=======

Variables by Category
---------------------

.. csv-table:: For each general category category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_se_gnrl.csv
   :widths: 15, 20, 15, 10, 20, 10, 10
   :header-rows: 1

Variables by Partial Category
-----------------------------

The general socioeconomic subsector includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column. If the variable applies to no specific categorical value, the entry will read **none**.

.. csv-table:: Trajectories of the following variables are needed for **some** (or no) general categories.
   :file: ./csvs/table_varreqs_by_partial_category_se_gnrl.csv
   :widths: 15, 15, 20, 10, 10, 10, 10, 10
   :header-rows: 1

Categories
----------

General should be divided into the following categories, given by ``$CAT-GENERAL$``.

.. csv-table:: General categories (``$CAT-GENERAL$`` attribute table)
   :file: ./csvs/attribute_cat_general.csv
   :widths: 15,15,30,15,10,15
   :header-rows: 1

----


Economy
=======

Variables by Category
---------------------

.. csv-table:: For each general exogenous economic category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_se_econ.csv
   :widths: 15, 20, 15, 10, 20, 10, 10
   :header-rows: 1

Variables by Partial Category
-----------------------------

The economy socioeconomic subsector includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column. If the variable applies to no specific categorical value, the entry will read **none**.

.. csv-table:: Trajectories of the following variables are needed for **some** (or no) general categories.
   :file: ./csvs/table_varreqs_by_partial_category_se_econ.csv
   :widths: 15, 15, 20, 10, 10, 10, 10, 10
   :header-rows: 1

Categories
----------

General should be divided into the following categories, given by ``$CAT-ECONOMY$``.

.. csv-table:: Economy categories (``$CAT-ECONOMY$`` attribute table)
   :file: ./csvs/attribute_cat_economy.csv
   :widths: 15,15,30,15,10,15
   :header-rows: 1
