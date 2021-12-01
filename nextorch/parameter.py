"""
Creates Parameters and ParameterSpace defined by their types

"""

import numpy as np
import copy
import torch
from torch import Tensor, tensor 
from typing import Optional, TypeVar, Union, Tuple, List

# NEED TO EXPLAIN THESE IN DOCS
# Create a type variable for 1D arrays from numpy, np.ndarray
Array = TypeVar('Array')
# Create a type variable for 2D arrays from numpy, np.ndarray, and call it as a matrix
Matrix = TypeVar('Matrix')

# Create a type variable which is array like (1D) including list, array, 1d tensor
ArrayLike1d = Union[list, Array, Tensor]
# Create a type variable which is matrix like (2D) including matrix, tensor, 2d list
# This also includes ArrayList1d types
MatrixLike2d = Union[list, Matrix, Tensor]

#%% Parameter space classes 
class Parameter():
    """
    Parameter class,
    stores properties for a single parameter
    """
    def __init__(
        self, 
        name: Optional[str] = None,
        x_type: Optional[str] = None, 
        x_range: Optional[ArrayLike1d] = None, 
        values: Optional[ArrayLike1d] = None,
        interval: Optional[ArrayLike1d] = None,
    ):
        """Define the properties of parameters

        Parameters
        ----------
        name : Optional[str], optional
            Parameter name, by default None
        x_type : Optional[str], optional
            Parameter type, must be continuous, ordinal, 
            or categorical, by default None
        x_range : Optional[ArrayLike1d], optional
            Parameter range, by default None
        values : Optional[ArrayLike1d], optional
            Specific parameter values, by default None
        interval : Optional[ArrayLike1d], optional
            Intervals for ordinal parameter, by default None

        Raises
        ------
        ValueError
            Invalid input x_type
        ValueError
            Invalid input for ordinal parameter
        ValueError
            Invalid input for categorical parameter
        """
        # the allowed default parameters types
        default_types = ['continuous', 'ordinal', 'categorical']
        encoding = None

        # if no input, use continuous
        if x_type is None:
            x_type = 'continuous'
        # check if input type is valid
        else:
            if x_type not in default_types:
                raise ValueError('Input type is not allowed. Please input either \
                    continuous, ordinal, or categorical.')

        # Define the properties
        if x_type == 'continuous':
            # Set default x_range
            if x_range is None: 
                x_range = [0, 1]
        elif x_type == 'ordinal':
            # Case 1, values are specified, use the values as encoding
            if values is not None:  # use the input values
                x_range = [np.min(values), np.max(values)]
                encoding = values 
            # Case 2, interval and x_range are specified
            if (interval is not None) and (x_range is not None):
                n_points = int((x_range[1] - x_range[0])/interval) + 1
                values = np.linspace(x_range[0], x_range[1], n_points)
                encoding = values 
            else: 
                raise ValueError('Ordinal parameter: \
                    must either input a list of values or the interval and range.')
        else: # categorical
            if values is None:
                raise ValueError('Categorical parameter: must input a list of values.')
            # Encodings are [0, 1, ... n_categories-1]
            n_categories = len(values)
            x_range = [0, n_categories-1]
            encoding = np.linspace(x_range[0], x_range[1], n_categories)

        # Assign to self
        self.name = name
        self.x_type = x_type
        self.x_range = x_range # real ranges
        self.values = values
        self.encoding = encoding 
        

class ParameterSpace():
    """
    ParameterSpace class,
    define the parameter space for an experiment
    """
    def __init__(self, parameters: Union[Parameter, List[Parameter]]):
        """[summary]

        Parameters
        ----------
        parameters : Union[Parameter, List[Parameter]]
            A single or a list of parameters 
        """
        # Convert to a list for a single parameter
        if not isinstance(parameters, list):
            parameters = [parameters]

        # Collect properties from each parameter
        self.names = [pi.name for pi in parameters] 
        self.X_types = [pi.x_type for pi in parameters] 
        self.X_ranges = [pi.x_range for pi in parameters] 
        self.values_2D = [pi.values for pi in parameters]
        self.encodings = [pi.encoding for pi in parameters]

        # Check if all parameters are continuous
        self.all_continuous = True
        if ('ordinal' in self.X_types) or ('categorical' in self.X_types):
            self.all_continuous = False
