=====================
Design of Experiment
=====================

.. currentmodule:: nextorch.doe

We use design of experiments (DOE) to generate the initial sample plan (:code:`X_init`).
Common methods include general full factorial design, completely randomized design and Latin hypercube sampling (LHS). 
The formula can be found in standard statistics books.
We use LHS heavily since its near-random design and the efficient space-filling abilities.

.. note::

    :code:`X_init` is always in a unit scale. 

    Setting the :code:`seed` parameter in random designs could make sure that the same set of data are generated every time.


Example 
----------
The examples are for a 3-dimensional system.
Generate a full factorial design with 5 levels in each dimension. 

.. code-block:: python

    from nextorch import doe

    n_ff_level = 5
    X_init_ff = doe.full_factorial([n_ff_level, n_ff_level, n_ff_level])



Generate a LHS design with 10 initial points. 

.. code-block:: python

    from nextorch import doe

    X_init_lhs = doe.latin_hypercube(n_dim=3, n_points=10, seed=1)
    

Generate a completely random design with 50 initial points.

.. code-block:: python

    from nextorch import doe

    X_init_random = doe.randomized_design(n_dim=3, n_points=50, seed=1)


Here is a list of DOE functions in :code:`nextorch.doe` module.

.. autosummary::
    :nosignatures:

    full_factorial
    latin_hypercube
    randomized_design
    randomized_design_w_levels
