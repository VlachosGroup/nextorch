"""
nextorch.utils

Utility functions for Bayesian Optimization
"""

import numpy as np
import copy
import torch

from typing import Optional, TypeVar, Union, Tuple
# Create a type variable for 1D arrays from numpy
array = TypeVar('array')
# Create a type variable for 2D arrays from numpy and call it as a matrix
matrix = TypeVar('matrix')
# Create a type variable for torch tensor, it can 1D, 2D arrays in torch
tensor = TypeVar('tensor')


# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

#%%
'''
Normalization functions 
Work for numpy arrays 
'''

def norm_xv(xv: array, xi_range: list) -> array:
    """
    Takes in an x array in a real scale
    and converts it to a unit scale

    Parameters
    ----------
    xv : array
        original x array
    xi_range : list
        range of x, [left bound, right bound]

    Returns
    -------
    xv: array
        normalized x in a unit scale
    """    
    xunit = copy.deepcopy(xv)
    lb = xi_range[0] #the left bound
    rb = xi_range[1] #the right bound
    xunit = (xv - lb)/(rb - lb)
    
    return xunit

def norm_X(
    X: matrix,  
    X_range: Optional[list] = [], 
    log_flags: Optional[list] = [], 
    decimals: Optional[int] = None
) -> matrix:
    """Takes in a matrix in a real scale
    and converts it into a unit scale

    Parameters
    ----------
    X : matrix
        original matrix in a real scale
    X_range : Optional[list], optional
        list of x ranges, by default []
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
    Xunit: matrix
        matrix scaled to a unit scale
    """
     #If 1D, make it 2D a matrix
    if len(X.shape)<2:
        X = copy.deepcopy(X)
        X = np.array([X])
        
    dim = X.shape[1] #the number of column in X
    
    if X_range == []: X_range = [[0,1]] * dim
    else: X_range = np.transpose(X_range)
    
    if log_flags == []: log_flags = [False] * dim
    
    # Initialize with a zero matrix
    Xunit = np.zeros((X.shape[0], X.shape[1]))
    for i, xi in enumerate(np.transpose(X)):
        if log_flags[i]:
            Xunit[:,i] =  np.log10(norm_xv(xi, X_range[i]))
        else:
            Xunit[:,i] =  norm_xv(xi, X_range[i])
    
    # Round up if necessary
    if not decimals == None:
        Xunit = np.around(Xunit, decimals = decimals)  
    
    return Xunit


def inversenorm_xv(xv: array, xi_range: list) -> array:    
    """
    Takes in an x array in a unit scale
    and converts it to a real scale

    Parameters
    ----------
    xv : array
        x array in a unit scale
    xi_range : list
        range of x, [left bound, right bound]

    Returns
    -------
    xv: array
        x in a real scale
    """ 
    xreal = copy.deepcopy(xv)
    lb = xi_range[0] #the left bound
    rb = xi_range[1] #the right bound
    xreal = lb + (rb-lb)*xv
    
    return xreal


def inversenorm_X(
    X: matrix, 
    X_range: Optional[list]= [], 
    log_flags: Optional[list] = [], 
    decimals: Optional[int] = None
) -> matrix:
    """Takes in a matrix in a unit scale
    and converts it into a real scale

    Parameters
    ----------
    X : matrix
        original matrix in a unit scale
    X_range : Optional[list], optional
        list of x ranges, by default []
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
    Xunit: matrix
        matrix scaled to a real scale
    """
    if len(X.shape)<2:
        X = copy.deepcopy(X)
        X = np.array([X]) #If 1D, make it 2D array
    
    dim = X.shape[1]  #the number of column in X
    
    if X_range == []: X_range = [[0,1]] * dim
    else: X_range = np.transpose(X_range)
    
    if log_flags == []: log_flags = [False] * dim
    
    Xreal = np.zeros((X.shape[0], X.shape[1]))
    for i, xi in enumerate(np.transpose(X)):
        if log_flags[i]:
            Xreal[:,i] =  10**(inversenorm_xv(xi, X_range[i]))
        else:
            Xreal[:,i] =  inversenorm_xv(xi, X_range[i])

    # Round up if necessary
    if not decimals == None:
        Xreal = np.around(Xreal, decimals = decimals)  
    
    return Xreal


def standardize_X(
    X: Union[array, matrix, tensor], 
    X_mean: Optional[matrix] = None, 
    X_std: Optional[matrix] = None
) -> matrix:
    """Takes in an array/matrix X 
    and returns the standardized data with zero mean and a unit variance

    Parameters
    ----------
    X : matrix or array
        the original matrix or array
    X_mean : Optional[list], optional
        mean of each column in X, 
        by default None, it will be computed here
    X_std : list, optional
        stand deviation of each column in X, 
        by default None, it will be computed here

    Returns
    -------
    X_standard: matrix
        Standardized X matrix
    """    
    # Compute the mean and std if not provided
    if X_mean is None:
        X_mean = X.mean(axis = 0)
        X_std = X.std(axis = 0)
        
    return (X - X_mean) / X_std



def inversestandardize_X(X, X_mean, X_std):
    '''
    Takes in a vector/matrix X and returns the data in the real scale
    ''' 
    if type(X) == torch.Tensor:
        X_real = X * X_std +  X_mean
    else:
        X_real = np.multiply(X, X_std) +  X_mean # element by element multiplication
    
    return X_real
    
    

def eval_test_function(X_unit, X_range, test_function):
    '''
    Input the X matrix (in unit range) in tensor or numpy
    and a test function which evaluate np arrays
    evaluate the test function and return y in tensor
    '''
    # Convert matrix type from tensor to numpy array
    if type(X_unit) ==  torch.Tensor:
        X_unit_np = X_unit.cpu().numpy()
    else:
        X_unit_np = X_unit.copy()
        
    if type(X_range) ==  torch.Tensor:
        X_range_np = X_range.cpu().numpy()
    else:
        X_range_np = X_range.copy()
        
        
    X_real = inversenorm_X(X_unit_np, X_range_np)
    
    #print(X_real)
    # evaluate y
    y = test_function(X_real)
    # Convert to tensor
    y_tensor = torch.tensor(y, dtype = dtype)

    return y_tensor
    
    

        
def predict_surrogate(model, test_x):
    '''
    Input a Gaussian process model 
    return the mean, lower confidence interval and upper confidence intervale 
    from the postierior
    '''
    
    test_x = copy.deepcopy(test_x)
    test_x = torch.tensor(test_x, dtype = dtype)
    
    posterior = model.posterior(test_x)
    
    test_y_mean = posterior.mean
    test_y_lower, test_y_upper = posterior.mvn.confidence_region()
    
    return test_y_mean, test_y_lower, test_y_upper

#%%
'''
2D specific functions
'''


def create_2D_mesh(mesh_size = 41):   
    '''
    Create 2D mesh for testing
    return x1 and x2 used for for-loops
    '''
    nx1, nx2 = (mesh_size, mesh_size)
    x1 = np.linspace(0, 1, nx1)
    x2 = np.linspace(0, 1, nx2)
    # Use Cartesian indexing, the matrix indexing is wrong
    x1, x2 = np.meshgrid(x1, x2,  indexing='xy') 
    
    test_x = []
    for i in range(nx1):
        for j in range(nx2):
            test_x.append([x1[i,j], x2[i,j]])
    
    test_x = np.array(test_x)
    
    return test_x, x1, x2

  
def transform_plot2D_Y(X, X_mean, X_std, mesh_size):
    '''
    takes in 1 column of tensor 
    convert to real units and return a 2D numpy array 
    '''
    X = X.clone()
    # Inverse the standardization
    X_real = inversestandardize_X(X, X_mean, X_std)

    # Convert to numpy for plotting
    X_plot2D = np.reshape(X_real.detach().numpy(), (mesh_size, mesh_size))
    
    return X_plot2D
    

def transform_plot2D_X(X1, X2, X_range):
    
    X_range = np.array(X_range).T
    X1 = inversenorm_xv(X1, X_range[0])
    X2 = inversenorm_xv(X2, X_range[1])
    
    return X1, X2
    