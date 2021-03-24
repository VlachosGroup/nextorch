=====================
Design of Experiment
=====================

.. currentmodule:: nextorch.doe

Common methods include general full factorial design, completely randomized design and Latin hypercube sampling (LHS). 
The formula can be found in standard statistics books.
We use LHS heavily since its near-random design and the efficient space-filling abilities.


.. autosummary::
    :nosignatures:

    full_factorial
    latin_hypercube
    randomized_design
    randomized_design_w_levels
