=======================
SISEPUEDE Documentation
=======================

SISEPUEDE (**SI**\tmulating **SE**\tctoral **P**\tathways and **U**\tncertainty **E**\txploration for **DE**\tcarbonization) is an integrated Python/Julia modeling system used to facilitate exploratory analyses of decarbonization transformations too sectors at the country level. It includes several key components:

- A data pipeline management system

- Uncertainty specification and trajectory sampling mechanism

- Sectoral models of emissions


Check out the :doc:`General Data <../general_data.htm>` section to get started.

About the Model
===============

SISEPUEDE is a compartmentalized, sector-based model of emissions based primarily on two key publications from the IPCC:

#. `2006 IPCC Guidelines for National Greenhouse Gas Inventories <https://www.ipcc-nggip.iges.or.jp/public/2006gl/index.html>`_ and

#.  `2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories <https://www.ipcc-nggip.iges.or.jp/public/2006gl/index.html>`_

These two documents are often abbreviated as **V##, C## IPCC GNGHGI** in attribute tables. In this notation, **V##** gives the volume number, while **C##** gives the chapter number. For example, V5, C6 refers to Volume 5, Chapter 6 (*Wastewaster Treatment and Discharge*).

**EXPAND**


Metavariables and Constructing Input Parameters
-----------------------------------------------

This document makes use of the ``$VARNAME$`` notation to denote metavariables as components of variable schema. These variable schema are used to store data in the ``model_input_variables.csv`` file.

For example, model input variables used to denote agricultural activity emission factors by crop type and gas in ``model_input_variables.csv`` may have the following structure:
``ef_agactivity_$CAT-AGRICULTURE$_$UNIT-MASS$_$EMISSION-GAS$_$UNIT-AREA$``, where

- ``$CAT-AGRICULTURE$`` is the categorical crop type (e.g., cereals, oil crops, pulses, etc.);
- ``$EMISSION-GAS$`` is the greenhouse gas that is being emitted (e.g. ``co2``, ``ch4``, or ``n2o``)
- ``$UNIT-MASS$`` is the unit of mass for gas emission (e.g., ``kg`` for kilograms; some sector variables may use ``gg`` for gigagrams);
- ``$UNIT-AREA$`` is the area unit (e.g., ``ha`` is hectares).

These components are referred to as *metavariables*--they characterize and describe the notation for naming model input variables. Each variable is associated with some naming *schema*, which presents a standardized format for variable entry depending on the relevant metavariables.

.. note::
   Example: the :math:`\text{CO}_2` emission factor for maize crop production, which captures crop burning, decomposition, and other factors, would be entered as ``ef_agactivity_maize_kg_co2_ha`` since, in this case, ``$CAT-AGRICULTURE$ = maize``, ``$EMISSION-GAS$ = co2``, ``$UNIT-AREA$ = ha``, and ``$UNIT-MASS$ = kg``. Similarly, the :math:`\text{N}_2\text{O}` factor, which includes crop liming and fertilization, would be captured as ``ef_agactivity_maize_kg_n2o_ha``.



Contents
--------
.. toc example struct from https://github.com/readthedocs/sphinx_rtd_theme/blob/c9b1bde560d8ee31400e4e4f92f2e8d7a42265ce/docs/index.rst
.. https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html

.. toctree::
   :caption:  Getting Started
   :hidden:

   quick_start
   analytical_parameters
   dimensions_of_analysis
   running_models

.. toctree::
   :caption: Variables, Categories, and Data
   :hidden:

   general_data
   afolu
   circular_economy
   energy_electric
   energy_non_electric
   ippu
   socioeconomic

.. toctree::
   :caption: Mathematical Specifications
   :hidden:

   mathdoc_afolu
   mathdoc_circular_economy
   mathdoc_economic_impact
   mathdoc_energy
   mathdoc_ippu
