============
Examples
============

These examples illustrate the main features and applications of the NEXTorch. Specifically these tutorials include:

- `Basic API Usage`_ demonstrates the whole process of using NEXTorch to do the optimizations, how does NEXTorch deal with 
  different parameter types, and the basic usage of the NEXTorch API.
- `Applications in Reaction Engineering`_ demonstrates the applications of NEXTorch in the real-world problem, such as 
  catalyst synthesis, reaction condition optimizations, and reactor design. Moreover, the human-in-the-loop optimization using 
  NEXTorch and the integration of NEXTorch with computational simulations for automatic optimizations are demonstrated.
- `Applications in Multi-Objective Optimization`_ shows the capability of doing multi-objective optimizations using NEXTorch. 
  Two different ways of performing multi-objective optimization are employed to optimize the reaction conditions and design the 
  reactors.

Basic API Usage
---------------
.. nbgallery::
    :caption: Gallery
    :name: rst-gallery
    :glob:

    ../examples/01_simple_1d
    ../examples/02_sin_1d
    ../examples/10_PFR_mixed_type_inputs

Applications in Reaction Engineering
------------------------------------
.. nbgallery::
    :caption: Gallery
    :name: rst-gallery
    :glob:

    ../examples/03_LH_mechanism
    ../examples/04_NDC_catalyst
    ../examples/05_PFR_yield
    ../examples/08_stub_tuner

Applications in Multi-Objective Optimization
--------------------------------------------
.. nbgallery::
    :caption: Gallery
    :name: rst-gallery
    :glob:

    ../examples/06_ellipse_MOO
    ../examples/07_PFR_MOO
    ../examples/09_Stub_tuner_MOO
    ../examples/11_PFR_EHVI_MOO