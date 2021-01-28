"""
nextorch.doe

Generates initial design of experiment (DOE)
Includes full factorial design, latin hypercube design and random design
"""
import itertools
import numpy as np
import scipy
from typing import Optional, TypeVar, List
# Create a type variable for 1D arrays from numpy
Array = TypeVar('Array')
# Create a type variable for 2D arrays from numpy and call it as a matrix
Matrix = TypeVar('Matrix')


import pyDOE2 as DOE 
import nextorch.utils as ut


def full_factorial(levels: List[int]) -> Matrix:    
    """Generates full factorial design 

    Parameters
    ----------
    levels : list of int
        Each number is a discrete level of each independent variable
        m is the number of variables or the size of the list

    Returns
    -------
    X_norm: Matrix
        Normalized sampling plan with the shape of prod(level_i) * m
    """    
    # Import DOE function
    DOE_function = DOE.fullfact
    X_real = DOE_function(levels)

    # Normailize X_real
    X_ranges = np.transpose([[0, i-1] for i in levels]) #-1 for python index
    X_norm = ut.norm_X(X_real, X_range = X_ranges)
    
    return X_norm



def latin_hypercube(
    n_dim: int, 
    n_points: int, 
    random_seed: Optional[int] = None, 
    criterion: Optional[str] = None
) -> Matrix:
    """Generates latin hypercube design

    Parameters
    ----------
    n_dim : int
        Number of independent variables
    n_points : int
        Total number of points in the design
    random_seed : Optional[int], optional
        Random seed, by default None
    criterion : Optional[str], optional
        String that tells lhs how to sample the points, by default None 
        which simply randomizes the points within the intervals.
        Other options: “center”, “maximin”, “centermaximin”, or “correlation” 
        See https://pythonhosted.org/pyDOE/randomized.html for details. 

    Returns
    -------
    X_norm: Matrix
        Normalized sampling plan with the shape of n_point * n_dim
    """    
    X_norm = DOE.lhs(n_dim, samples = n_points, criterion = criterion, random_state= random_seed)

    return X_norm


def randomized_design(
    n_dim: int, 
    n_points: int, 
    seed: Optional[int] = None
) -> Matrix:
    """Generates randomized design

    Parameters
    ----------
    n_dim : int
        Number of independent variables
    n_points : int
        Total number of points in the design
    seed : Optional[int], optional
            Random seed, by default None

    Returns
    -------
    X_norm: Matrix
        Normalized sampling plan with the shape of n_point * n_dim
    """    
    # Set the random state
    if seed is None:
        random_state = np.random.RandomState()
    else:
        random_state = np.random.RandomState(seed)
    X_norm = random_state.rand(n_points, n_dim)
    
    return X_norm


def randomized_design_w_levels(
    levels: List[int], 
    seeds: Optional[List[int]] = None
) -> Matrix:
    """Generates randomized design with levels in each dimension

    Parameters
    ----------
    levels : list of int
        Each number is a discrete level of each independent variable
        m is the number of variables or the size of the list
    seeds : Optional[list of int], optional
        List of random seeds, same size as levels, by default None

    Returns
    -------
    X_norm: Matrix
        Normalized sampling plan with the shape of prod(level_i) * m
    """    
    n_dim = len(levels)
    n_points = np.prod(levels)  # number of points
    X_norm = np.zeros((n_points, n_dim))
    x_vectors = []
    # Each dimension has 1d random design of level_i points
    for i, level_i in enumerate(levels):
        if seeds is None:
            x_vector_i = randomized_design(1, level_i)
        else:
            x_vector_i = randomized_design(1, level_i, seeds[i])
        x_vectors.append(x_vector_i.flatten())
    # Combination and assign back to X
    combo = list(itertools.product(*x_vectors))
    for i, ci in enumerate(combo):
        X_norm[i,:] = np.array(ci)

    return X_norm


def norm_transform(X_norm: Matrix, means: List[float], stdvs: List[float]) -> Matrix:
    """Transform designs into normal distribution

    Parameters
    ----------
    X_norm : Matrix
        Original design matrix with the shape of n_points*n_dim
    means : list of float
        Means of the target distributions, size of n_dim
    stdvs : list of float
        Standard deviations of the target distributions, size of n_dim

    Returns
    -------
    X_transformed: Matrix
        Transformed sampling plan with the shape of n_points*n_dim
    """     
    n_dim = X_norm.shape[1] # number of independent variables
    X_transformed = np.zeros(X_norm.shape) # copy

    for i in range(n_dim):
        X_transformed[:, i] = scipy.stats.norm(loc=means[i], scale=stdvs[i]).ppf(X_norm[:, i])

    return X_transformed


'''
Other designs such as:
- Fractional-Factorial
- Plackett-Burman
- Box-Behnken designs
- Central composite designs
can be generated by pyDOE2 and hence used by nextorch
'''

