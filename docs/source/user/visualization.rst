===============
Visualization
===============

.. contents:: Table of Contents
    :depth: 2
    
.. currentmodule:: nextorch.plotting

NEXTorch offers a variety of visualization functions in :code:`nextorch.plotting` module.
The plots are rendered using matplotlib_ as a backend.
It is an active area of development for NEXTorch.

We provide two forms of the visualization functions depending on their inputs format: 
(1) :code:`X` or :code:`Y` matrics or (2) an :code:`Experiment` object (functions that end with :code:`_exp`). 

Parity Plots
--------------

.. autosummary::
    :nosignatures:

    parity
    parity_exp
    parity_with_ci
    parity_with_ci_exp


Discovery Plots
----------------

.. autosummary::
    :nosignatures:

    opt_per_trial
    opt_per_trial_exp


Functions in 1-dimension Parameter Space
-------------------------------------------

.. autosummary::
    :nosignatures:

    acq_func_1d
    acq_func_1d_exp
    response_1d
    response_1d_exp


Functions in 2-dimensional Parameter Spaces
---------------------------------------------

.. autosummary::
    :nosignatures:

    sampling_2d
    sampling_2d_exp
    response_heatmap
    response_heatmap_exp
    objective_heatmap
    objective_heatmap_exp
    response_heatmap_err_exp
    response_surface
    response_surface_exp
    objective_surface
    objective_surface_exp
    

Functions in 3-dimensional Parameter Spaces
--------------------------------------------
.. autosummary::
    :nosignatures:

    sampling_3d
    sampling_3d_exp
    response_scatter_exp


Pareto Front
-------------------
.. autosummary::
    :nosignatures:

    pareto_front
    pareto_front_exp



.. _matplotlib: https://matplotlib.org/stable/index.html