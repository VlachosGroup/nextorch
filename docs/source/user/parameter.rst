============
Parameter
============

.. currentmodule:: nextorch.parameter

Parameters and the data associated with them are generally of three types: 

- Continuous: variables that are numerical and can take any real value in a range
- Categorical: variables that are non-numeric, denoted by words, text, or symbols.
- Ordinal: variables that are numerical and can take ordered discrete values


Traditionally, BO is designed for systems with all continuous parameters `[1]`_. 
However, depending on the problem, the parameters can also be of other types, and the resulting design space can be discrete or mixed (continuous-discrete). 
If the number of discrete combinations is low, one possible solution is to enumerate the values and optimize the continuous parameters for each. 


Without loss of generality, the default setting in NEXTorch is to use a continuous relaxation approach to treat other types of parameters. 
The acquisition function is optimized in the relaxed space and rounded to the available values. 
For ordinal parameters, these values are the ordered discrete values. 
For categorical parameters, we encode the categories with integers from 0 to :math:`n_{category}-1` and then perform continuous relaxation in the encoding space. 
Since a parameter can be approximated as continuous given a high order discretization, this approach usually works well for problems with high discrete combinations. 

In NEXTorch, a :code:`Parameter` class stores the range and type of each parameter. A :code:`ParameterSpace` class consists of all Parameter classes. 

.. note::
    The default parameter type is continuous. If all parameters are continuous, it is not required to initialize any :code:`Parameter` or :code:`ParameterSpace` class. 
    We can skip the :code:`define_space` step and use :code:`input_data` to input the data directly into an :code:`Experiment` class.

Example
------------
Set the type and range for each parameter. Input their type, range or interval.

.. code-block:: python

    from nextorch import parameter

    parameter_1 = parameter.Parameter(x_type = 'ordinal', x_range=[140, 200], interval = 5)
    parameter_2 = parameter.Parameter(x_type = 'categorical', values=['low', 'medium low', 'medium', 'medium high', 'high'])
    parameter_3 = parameter.Parameter(x_type = 'continuous', x_range=[-2, 2])

Input a list of :code:`Parameter` into a :code:`ParameterSpace` class.

.. code-block:: python
    
    from nextorch import parameter

    parameters = [parameter_1, parameter_2, parameter_3]
    parameter_space = parameter.ParameterSpace(parameters)

Set the parameter space for an :code:`Experiment` class.

.. code-block:: python
    
    parameters = [parameter_1, parameter_2, parameter_3]    
    Experiment.define_space(parameters)


For more details, see :code:`nextorch.parameter`.

.. autosummary::
    :nosignatures:

    Parameter
    ParameterSpace

-----------------

Reference
----------

`[1]`_ Frazier, P. I. A Tutorial on Bayesian Optimization. 2018.


.. _[1]: https://arxiv.org/abs/1807.02811