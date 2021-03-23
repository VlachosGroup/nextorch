============
Parameter
============

.. currentmodule:: nextorch.utils

Parameters and the data associated with them are generally of three types: 

- Continuous: variables that are numerical and can take any real value in a range
- Categorical: variables that are non-numeric, denoted by words, text, or symbols.
- Ordinal: variables that are numerical and can take ordered discrete values


Traditionally, BO is designed for systems with all continuous parameters. 
However, depending on the problem, the parameters can also be of other types, and the resulting design space can be discrete or mixed (continuous-discrete). 
If the number of discrete combinations is low, one possible solution is to enumerate the values and optimize the continuous parameters for each. 


Without loss of generality, the default setting in NEXTorch is to use a continuous relaxation approach where the acquisition function is optimized in the relaxed space and rounded to the available values. 
For ordinal parameters, these values are the ordered discrete values. 
For categorical parameters, we encode the categories with integers from 0 to n_category-1 and then perform continuous relaxation in the encoding space. 
Since a parameter can be approximated as continuous given a high order discretization, this approach usually works well for problems with high discrete combinations. 

In NEXTorch, a `Parameter` class stores the range and type of each parameter. A `ParameterSpace` class consists of all Parameter classes. 

.. autosummary::
    :toctree: parameter
    :nosignatures:

    Parameter
    ParameterSpace