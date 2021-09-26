============
Examples
============

These examples illustrate the main features and applications of the NEXTorch. 

Specifically these tutorials fall into three categories:

- `Basic API Usage`_ 
- `Applications in Reaction Engineering`_ 
- `Mixed Type Parameters`_
- `Multi-Objective Optimization(MOO)`_ 

Basic API Usage
---------------

Basic usage of the NEXTorch API

- Example 1 and 2: Implementation of NEXTorch for single-objective optimization
- Example 12 and 13: different DOE methods and comparing NEXTorch to Ax syntax
.. nbgallery::
    :name: basic-api-usage
    :glob:

    ../examples/01_simple_1d
    ../examples/02_sin_1d
    ../examples/12_PFR_yield_extension
    ../examples/13_Using_Ax


Applications in Reaction Engineering
------------------------------------

Applications of NEXTorch in the real-world problems, such as 
catalyst synthesis, reaction condition optimizations, and reactor design

- Example 3, 5, and 8: Automated optimization loop for computational simulations
- Example 4: Human-in-the-loop optimization for laboratory experiments
  

.. nbgallery::
    :name: app-in-react-eng
    :glob:

    ../examples/03_LH_mechanism
    ../examples/04_NDC_catalyst
    ../examples/05_PFR_yield
    ../examples/08_Stub_tuner


Mixed Type Parameters
--------------------------------------------

Demo for systems with parameters of mixed types 

.. nbgallery::
  :name: mixed-type
  :glob:

  ../examples/10_PFR_mixed_type_inputs


Multi-Objective Optimization(MOO)
--------------------------------------------

Demo for MOO systems 

- Example 6, 7 and 9: MOO with the weighted sum method
- Example 11: MOO with EHVI


.. nbgallery::
    :name: app-in-moo
    :glob:

    ../examples/06_ellipse_MOO
    ../examples/07_PFR_MOO
    ../examples/09_Stub_tuner_MOO
    ../examples/11_PFR_EHVI_MOO