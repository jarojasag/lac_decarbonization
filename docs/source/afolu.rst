===========================================
Agriculture, Forestry, and Land Use (AFOLU)
===========================================


Agriculture
===========

The **Agriculture** subsector is used to quantify emissions associated with growing crops, including emissions from the release of soil carbon, fertilizer applications and crop liming, crop burning, methane emissions from paddy rice fields, **AND MORE;CONTINUE**. Agriculture is divided into the following categories (crops), given by the metavariable ``$CAT-AGRICULTURE$``. Each crop should be associated an FAO classifications. `See the FAO <https://www.fao.org/waicent/faoinfo/economic/faodef/annexe.htm>`_ for the source of these classifications and a complete mapping of crop types to categories. On the git, the table ``ingestion/FAOSTAT/ref/attribute_fao_crop.csv`` contains the information mapping each crop to this crop type. Note, this table can be used to merge and aggregate data from FAO into these categories. If a crop type is not present in a country, set the associated area as a fraction of crop area to 0.

Variables by Category
---------------------

Agriculture requires the following variables.

.. csv-table:: For each agricultural category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_af_agrc.csv
   :header-rows: 1
.. :widths: 20, 30, 30, 10, 10


Categories
----------

Agriculture is divided into the following categories.

.. csv-table:: Agricultural categories (``$CAT-AGRICULTURE$`` attribute table)
   :file: ./csvs/attribute_cat_agriculture.csv
   :header-rows: 1
..   :widths: 15,15,30,15,10,15



**Costs to be added**

----

Forestry
========

Variables by Category
---------------------

.. csv-table:: For each forest category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_af_frst.csv
   :header-rows: 1

Variables by Partial Category
-----------------------------

Forestry includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column.

.. csv-table:: Trajectories of the following variables are needed for **some** forest categories. If they are independent of categories, the category will show up as **none**.
   :file: ./csvs/table_varreqs_by_partial_category_af_frst.csv
   :header-rows: 1

Categories
----------

Forestry is divided into the following categories. These categories reflect an aggregation of forestry types into emission-relevant categories. Note that areas of forested land are determined in the **Land Use** subsector. The land use at time *t* is determined by an ergodic Markov Chain (probabilities are set in the variable input table and subject to uncertainty using the mixing approach)

.. csv-table:: Forest categories (``$CAT-FOREST$`` attribute table)
   :file: ./csvs/attribute_cat_forest.csv
   :header-rows: 1
..   :widths: 15,15,30,15,10,15


----

Land Use
========

Land use projections are driven by a Markov Chain, represented by a transition matrix :math:`Q(t)` (the matrix is specified for each time period in the ``model_input_variables.csv`` file). The model requires initial states (entered as a fraction of total land area) for all land use categories ``$CAT-LANDUSE$``

.. note::
   The entries :math:`Q_{ij}(t)` give the transition probability of land use category :math:`i` to land use category :math:`j`. :math:`Q` is row stochastic, so that :math:`\sum_{j}Q_{ij}(t) = 1` for each land use category :math:`i` and time period :math:`t`. To preserve row stochasticity, it is highly recommended that strategies and uncertainty be represented using the trajectory mixing approach, where bounding trajectories on transitions probabilities are specified and uncertainty exploration gives a mix between them.

Variables by Category
---------------------

.. csv-table:: For each land use category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_af_lndu.csv
   :header-rows: 1

Variables by Partial Category
-----------------------------

Land use includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column.

.. note::
   Note that the sum of all initial fractions of area across land use categories *u* should be should equal 1to , i.e. :math:`\sum_u \varphi_u = 1`, where :math:`\varphi_{\text{$CAT-LANDUSE$}} \to` ``frac_lu_$CAT-LANDUSE$`` at period *t*.

.. csv-table:: Trajectories of the following variables are needed for **some** land use categories.
   :file: ./csvs/table_varreqs_by_partial_category_af_lndu.csv
   :header-rows: 1
.. :widths: 15, 15, 20, 10, 10, 10, 10, 10

Categories
----------

Land use should be divided into the following categories, given by ``$CAT-LANDUSE$``.

.. csv-table:: Land Use categories (``$CAT-LANDUSE$`` attribute table)
   :file: ./csvs/attribute_cat_land_use.csv
   :header-rows: 1

----


Livestock
=========

For each category, the following variables are needed. Information on enteric fermentation can be found from `the EPA <https://www3.epa.gov/ttnchie1/ap42/ch14/final/c14s04.pdf>`_ and **ADDITIONAL LINKS HERE**.

Variables by Category
---------------------

.. csv-table:: For each livestock category, trajectories of the following variables are needed.
   :file: ./csvs/table_varreqs_by_category_af_lvst.csv
   :header-rows: 1

Variables by Partial Category
-----------------------------

Livestock includes some variables that apply only to a subset of categories. These variables are described below. The categories that variables apply to are described in the ``category`` column.

.. csv-table:: Trajectories of the following variables are needed for **some** livestock categories.
   :file: ./csvs/table_varreqs_by_partial_category_af_lvst.csv
   :header-rows: 1

Categories
----------

Livestock should be divided into the following categories, given by ``$CAT-LIVESTOCK$``.

.. csv-table:: Livestock categories (``$CAT-LIVESTOCK$`` attribute table)
   :file: ./csvs/attribute_cat_livestock.csv
   :header-rows: 1
