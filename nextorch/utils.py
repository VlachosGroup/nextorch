"""
nextorch.utils

Utility functions for Bayesian Optimization
"""

import numpy as np
import copy
import torch
from torch import Tensor 

from typing import Optional, TypeVar, Union, Tuple
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

dtype = torch.float
torch.set_default_dtype(dtype)


#%% Scaling helper functions 
def get_ranges_X(X: MatrixLike2d) -> list:
    """Calculate the ranges for X matrix

    Parameters
    ----------
    X : MatrixLike2d
        matrix or tensor

    Returns
    -------
    list
        2D list of ranges: [left bound, right bound]
    """
    if len(X.shape)<2:
        X = copy.deepcopy(X)
        X = np.array([X])
        
    X_ranges = []
    n_dim = X.shape[1]

    for i in range(n_dim):
        X_ranges.append([np.min(X[:,i]), np.max(X[:,i])])
    
    return X_ranges


def unitscale_xv(xv: ArrayLike1d, xi_range: ArrayLike1d) -> ArrayLike1d:
    """
    Takes in an x array in a real scale
    and converts it to a unit scale

    Parameters
    ----------
    xv : ArrayLike1d
        original x array
    xi_range : ArrayLike1d
        range of x, [left bound, right bound]

    Returns
    -------
    xunit: ArrayLike1d, same type as xv
        normalized x in a unit scale
    """    
    xunit = copy.deepcopy(xv)
    lb = xi_range[0] #the left bound
    rb = xi_range[1] #the right bound
    xunit = (xv - lb)/(rb - lb)
    
    return xunit

def unitscale_X(
    X: MatrixLike2d,  
    X_ranges: Optional[MatrixLike2d] = None, 
    unit_flag: Optional[bool] = False,
    log_flags: Optional[list] = None, 
    decimals: Optional[int] = None
) -> Matrix:
    """Takes in a matrix in a real scale
    and converts it into a unit scale

    Parameters
    ----------
    X : MatrixLike2d
        original matrix in a real scale
    X_ranges : Optional[MatrixLike2d], optional
        list of x ranges, by default None
    unit_flag: Optional[bool], optional,
        by default, False 
        If true, the X is in a unit scale so
        the function is used to scale X to a log scale
    log_flags : Optional[list], optional
        list of boolean flags
        True: use the log scale on this dimensional
        False: use the normal scale 
        by default []
    decimals : Optional[int], optional
        Number of decimal places to keep
        by default None, i.e. no rounding up 

    Returns
    -------
    Xunit: numpy matrix
        matrix scaled to a unit scale
    """
     #If 1D, make it 2D a matrix
    if len(X.shape)<2:
        X = copy.deepcopy(X)
        X = np.array([X])
        
    n_dim = X.shape[1] #the number of column in X
    
    if unit_flag: #the X are in unit scale
        X_ranges = [[0,1]] * n_dim
    else:
        if X_ranges is None: # X_ranges not defined
            X_ranges = get_ranges_X(X)
    
    if log_flags is None: log_flags = [False] * n_dim
    
    # Initialize with a zero matrix
    Xunit = np.zeros((X.shape[0], X.shape[1]))
    for i, xi in enumerate(np.transpose(X)):
        if log_flags[i]:
            Xunit[:,i] =  np.log10(unitscale_xv(xi, X_ranges[i]))
        else:
            Xunit[:,i] =  unitscale_xv(xi, X_ranges[i])
    
    # Round up if necessary
    if not decimals == None:
        Xunit = np.around(Xunit, decimals = decimals)  
    
    return Xunit


def inverse_unitscale_xv(xv: ArrayLike1d, xi_range: ArrayLike1d) -> ArrayLike1d:    
    """
    Takes in an x array in a unit scale
    and converts it to a real scale

    Parameters
    ----------
    xv : ArrayLike1d
        x array in a unit scale
    xi_range : ArrayLike1d
        range of x, [left bound, right bound]

    Returns
    -------
    xv: ArrayLike1d, same type as xv
        x in a real scale
    """ 
    xreal = copy.deepcopy(xv)
    lb = xi_range[0] #the left bound
    rb = xi_range[1] #the right bound
    xreal = lb + (rb-lb)*xv
    
    return xreal


def inverse_unitscale_X(
    X: MatrixLike2d, 
    X_ranges: Optional[MatrixLike2d]= None,
    unit_flag: Optional[bool] = False,
    log_flags: Optional[list] = None, 
    decimals: Optional[int] = None
) -> Matrix:
    """Takes in a matrix in a unit scale
    and converts it into a real scale

    Parameters
    ----------
    X : MatrixLike2d
        original matrix in a unit scale
    X_ranges : Optional[MatrixLike2d], optional
        list of x ranges, by default None
    unit_flag: Optional[bool], optional,
        by default, False 
        If true, the X is in a unit scale so
        the function is used to scale X to a log scale
    log_flags : Optional[list], optional
        list of boolean flags
        True: use the log scale on this dimensional
        False: use the normal scale 
        by default None
    decimals : Optional[int], optional
        Number of decimal places to keep
        by default None, i.e. no rounding up 

    Returns
    -------
    Xunit: numpy matrix
        matrix scaled to a real scale
    """
    if len(X.shape)<2:
        X = copy.deepcopy(X)
        X = np.array([X]) #If 1D, make it 2D array
    
    n_dim = X.shape[1]  #the number of column in X
    
    if unit_flag: #the X are in unit scale
        X_ranges = [[0,1]] * n_dim
    else:
        if X_ranges is None: # X_ranges not defined
            X_ranges = get_ranges_X(X)
    
    if log_flags is None: log_flags = [False] * n_dim
    
    Xreal = np.zeros((X.shape[0], X.shape[1]))
    for i, xi in enumerate(np.transpose(X)):
        if log_flags[i]:
            Xreal[:,i] =  10**(inverse_unitscale_xv(xi, X_ranges[i]))
        else:
            Xreal[:,i] =  inverse_unitscale_xv(xi, X_ranges[i])

    # Round up if necessary
    if not decimals == None:
        Xreal = np.around(Xreal, decimals = decimals)  
    
    return Xreal


def standardize_X(
    X: MatrixLike2d, 
    X_mean: Optional[ArrayLike1d] = None, 
    X_std: Optional[ArrayLike1d] = None,
    return_type: Optional[str] = 'tensor'
) -> MatrixLike2d:
    """Takes in an array/matrix X 
    and returns the standardized data with zero mean and a unit variance

    Parameters
    ----------
    X : MatrixLike2d
        the original matrix or array
    X_mean : Optional[ArrayLike1d], optional
        same type as X
        mean of each column in X, 
        by default None, it will be computed here
    X_std : Optional[ArrayLike1d], optional
        same type as X
        stand deviation of each column in X, 
        by default None, it will be computed here
    return_type: Optional[str], optional
        either 'tensor' or 'np'

    Returns
    -------
    X_standard: MatrixLike2d, set by return_type
        Standardized X matrix
    
    Raises
    ------
    ValueError
        if input return_type not defined
    """    
    # Compute the mean and std if not provided

    if X_mean is None:
        X_mean = X.mean(axis = 0)
        X_std = X.std(axis = 0)
    if return_type == 'tensor':
        X = np_to_tensor(X)
        X_mean = np_to_tensor(X_mean)
        X_std = np_to_tensor(X_std)
    elif return_type == 'np':
        X = tensor_to_np(X)
        X_mean = tensor_to_np(X_mean)
        X_std = tensor_to_np(X_std)
    
    else: 
        raise ValueError('return_type must be either tensor or np')
        
    return (X - X_mean) / X_std



def inverse_standardize_X(
    X: MatrixLike2d, 
    X_mean: ArrayLike1d, 
    X_std: ArrayLike1d,
    return_type: Optional[str] = 'tensor'
) -> MatrixLike2d:
    """Takes in an arrary/matrix/tensor X 
    and returns the data in the real scale

    Parameters
    ----------
    X : MatrixLike2d
        the original matrix or array
    X_mean : Optional[ArrayLike1d], optional
        same type as X
        mean of each column in X, 
        by default None, it will be computed here
    X_std : Optional[ArrayLike1d], optional
        same type as X
        stand deviation of each column in X, 
        by default None, it will be computed here
    return_type: Optional[str], optional
        either 'tensor' or 'np'

    Returns
    -------
    X_real: MatrixLike2d, set by return_type
        in real scale

    Raises
    ------
    ValueError
        if input return_type not defined
    """    
    if return_type == 'tensor':
        X = np_to_tensor(X)
        X_mean = np_to_tensor(X_mean)
        X_std = np_to_tensor(X_std)
    elif return_type == 'np':
        X = tensor_to_np(X)
        X_mean = tensor_to_np(X_mean)
        X_std = tensor_to_np(X_std)
    else: 
        raise ValueError('return_type must be either tensor or np')

    if isinstance(X, Tensor):
        X_real = X * X_std +  X_mean
    else:
        X_real = np.multiply(X, X_std) +  X_mean # element by element multiplication
    
    return X_real



#%% 2-dimensional system specific functions
def create_2D_mesh_X(mesh_size = 41) -> Tuple[Matrix, Matrix, Matrix]:   
    """Create 2D mesh for testing

    Parameters
    ----------
    mesh_size : int, optional
        mesh size, by default 41

    Returns
    -------
    X_test: Matrix
        X in 2D mesh
    X1: Matrix
        X1
    X2: Matrix
        X2
    """
    nx1, nx2 = (mesh_size, mesh_size)
    x1 = np.linspace(0, 1, nx1)
    x2 = np.linspace(0, 1, nx2)
    # Use Cartesian indexing, the matrix indexing is wrong
    X1, X2 = np.meshgrid(x1, x2,  indexing='xy') 
    
    X_test = []
    for i in range(nx1):
        for j in range(nx2):
            X_test.append([X1[i,j], X2[i,j]])
    
    X_test = np.array(X_test)
    
    return X_test, X1, X2

def transform_2D_X(X1: Matrix, X2: Matrix, X_ranges: Matrix
) -> Tuple[Matrix, Matrix]:
    """Transform X1 and X2 in unit scale to real scales for plotting

    Parameters
    ----------
    X1 : Matrix
        X for variable 1
    X2 : Matrix
        X for variable 2
    X_range : Matrix
        the ranges of two variables 

    Returns
    -------
    X1: Matrix
        X1 in a real scale
    X2: Matrix
        X2 in a real scale
    """
    X_ranges = np.array(X_ranges).T
    X1 = inverse_unitscale_xv(X1, X_ranges[0])
    X2 = inverse_unitscale_xv(X2, X_ranges[1])
    
    return X1, X2
  
def transform_2D_mesh_Y(X: Tensor, X_mean: Tensor, X_std: Tensor, mesh_size: int
) -> Matrix:
    """takes in 1 column of X tensor 
    predict the Y values
    convert to real units and return a 2D numpy array 
    in the size of mesh_size*mesh_size

    Parameters
    ----------
    X : Tensor
        1d tensor
    X_mean : Tensor
        means 
    X_std : Tensor
        standard deviations
    mesh_size : int
        mesh size

    Returns
    -------
    X_plot2D: numpy matrix
        in real units for plotting
        
    """
    X = X.clone()
    # Inverse the standardization
    X_real = inverse_standardize_X(X, X_mean, X_std)

    # Convert to numpy for plotting
    X_plot2D = np.reshape(X_real.detach().numpy(), (mesh_size, mesh_size))
    
    return X_plot2D
    

def np_to_tensor(X: MatrixLike2d) -> Tensor:
    """Converts numpy objects to tensor objects
    Returns a copy

    Parameters
    ----------
    X : MatrixLike2D
        numpy objects

    Returns
    -------
    X: Tensor
        tensor objects
    """
    if not isinstance(X, Tensor): 
        X = torch.tensor(X, dtype= dtype)
    else:
        X = X.detach().clone()

    return X


def tensor_to_np(X: MatrixLike2d) -> Matrix:
    """Convert tensor objects to numpy array objects
    Returns a copy with no gradient information
    Parameters
    ----------
    X : MatrixLike2d
        tensor objects

    Returns
    -------
    Matrix
        numpy objects
    """
    if not isinstance(X, np.ndarray):
        X = X.detach().cpu().numpy()
    else: 
        X = X.copy()

    return X



# experiments are in tensor
# plot in numpy
# util functions works for both tensor and numpy