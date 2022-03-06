===================================
MODELNAMEHERE Documentation
===================================

**MODELNAMEHERE** (Some Acronym Depends) is an
integrated Python/Julia model used to facilitate exploratory analyses of decarbonization
transformations too sectors at the country level. It

Check out the :doc:`General Data <../general_data>` section to get started.

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

.. toctree::
   analytical_parameters
   dimensions_of_analysis
   general_data
   economic_impact
   afolu
   circular_economy
   energy
   ippu
   socioeconomic
