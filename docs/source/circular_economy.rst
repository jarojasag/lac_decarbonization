================
Circular Economy
================

TEMPORARY:
mcf - methane correction factor

ef for recycled materials (this is a production emission factor)

for domestic solid waste:
kg_dbo5_per_capita_per_anho
kg_n2_per_kg_proteina
kg_n2o_per_kg_n
kg_por_persona_proteina_per_anho
max_ch4_per_dbo
max_ch4_per_dco
mcf_ch4_dbo5

- waste generated per capita
- fraction of waste by solid waste category

- DOC (degradable organic carbon)
- k (check)

by applicable type (subcategories):
- fraction of waste recycled
- fraction of waste composted

for REMAINING WASTE:
- fraction of TOTAL waste landfilled
- fraction of TOTAL waste burned
- fraction of TOTAL waste uncaptured

-------

for each liquid waste type:
- fraction that is not sewered
- fraction that is sewered and not treated
- fraction that is sewered + treated



Liquid Waste
============

Liquid waste include...

Variables by Category
---------------------

.. csv-table:: The following variables are required for each category ``$CAT-WASTE-LIQUID$`` of liquid waste.
   :file: ./csvs/table_varreqs_by_category_ce_wali.csv
   :widths: 20, 30, 30, 10, 10
   :header-rows: 1


Categories
----------

Liquid waste is divided into the following categories, denotes as ``$CAT-WASTE-LIQUID$``.

.. csv-table:: Land Use categories (``$CAT-WASTE-LIQUID$`` attribute table)
   :file: ./csvs/attribute_cat_liquid_waste.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------

Liquid waste includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column.

.. csv-table:: Trajectories of the following variables are needed for **some** liquid waste categories.
   :file: ./csvs/table_varreqs_by_partial_category_ce_wali.csv
   :header-rows: 1

----

Solid Waste
===========

Variables by Category
---------------------

.. csv-table:: The following variables are required for each category ``$CAT-WASTE-SOLID$`` of solid waste.
   :file: ./csvs/table_varreqs_by_category_ce_waso.csv
   :header-rows: 1

Categories
----------

Solid waste is divided into the following categories, denoted as ``$CAT-WASTE-SOLID$``.

.. csv-table:: Solid waste categories (``$CAT-WASTE-SOLID$`` attribute table)
   :file: ./csvs/attribute_cat_solid_waste.csv
   :header-rows: 1


Variables by Partial Category
-----------------------------

Solid waste includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column.

.. csv-table:: Trajectories of the following variables are needed for **some** solid waste categories.
   :file: ./csvs/table_varreqs_by_partial_category_ce_waso.csv
   :header-rows: 1
