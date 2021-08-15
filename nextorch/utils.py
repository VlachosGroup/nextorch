"""
Utility functions for Bayesian Optimization

- PyTorch tensor to numpy array conversion
- (Inverse) normalization
- (Inverse) standardization
- Mesh test points generation
- Ordinal, categorical variable encoding/decoding


"""

#from nextorch.bo import ParameterSpace
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

dtype = torch.float
torch.set_default_dtype(dtype)


from nextorch.parameter import ParameterSpace
#%% Type conversion

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


def expand_list(list_1d: list) -> list:
    """Expand 1d list to 2d

    Parameters
    ----------
    list_1d : list
        input list

    Returns
    -------
    list_2d: list
        output 2d list
    """
    list_2d = copy.deepcopy(list_1d)
    if not isinstance(list_1d[0], list):
        list_2d = [list_2d]

    return list_2d

def expand_ranges_X(X_ranges: MatrixLike2d) -> list:
    """Expand 1d X_range to 2d list

    Parameters
    ----------
    X_ranges : MatrixLike2d
        list of x ranges (in 1d)

    Returns
    -------
    X_ranges: list
        X ranges in 2d list

    Raises
    ------
    ValueError
        Input type other than tensor/list/numpy matrix
    """
    # if tensor, convert to numpy matrix first
    if isinstance(X_ranges, Tensor):
        X_ranges = tensor_to_np(X_ranges)

    if isinstance(X_ranges, np.ndarray):
        if len(X_ranges.shape)<2:
            X_ranges = copy.deepcopy(X_ranges)
            X_ranges = np.array([X_ranges])
    elif isinstance(X_ranges, list):
        X_ranges = expand_list(X_ranges)
    else:
        raise ValueError("Input type not allowed, must be a Tensor/list/numpy matrix")

    return X_ranges
    
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
        X = np.expand_dims(X, axis=1) #If 1D, make it 2D array
        
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
        matrix scaled to a unit scale
    """
    
    #If 1D, make it 2D a matrix
    if len(X.shape)<2:
        X = copy.deepcopy(X)
        X = np.expand_dims(X, axis=1) # If 1D, make it 2D array
        
    n_dim = X.shape[1] #the number of column in X

    if X_ranges is None: # X_ranges not defined
        X_ranges = get_ranges_X(X)
    X_ranges = expand_ranges_X(X_ranges) #expand to 2d
    
    if log_flags is None: log_flags = [False] * n_dim
    
    # Initialize with a zero matrix
    Xunit = np.zeros((X.shape[0], X.shape[1]))
    for i in range(n_dim):
        xi = X[:,i]
        Xunit[:,i] =  unitscale_xv(xi, X_ranges[i])
        # Operate on a log scale
        if log_flags[i]:
            Xunit[:,i] =  np.log10(Xunit[:,i])
            
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
        X = np.expand_dims(X, axis=1) #If 1D, make it 2D array
    
    n_dim = X.shape[1]  #the number of column in X
    
    if X_ranges is None: # X_ranges not defined
        X_ranges = get_ranges_X(X)
    X_ranges = expand_ranges_X(X_ranges) #expand to 2d
    
    if log_flags is None: log_flags = [False] * n_dim
    
    # Initialize with a zero matrix
    Xreal = np.zeros((X.shape[0], X.shape[1]))
    for i in range(n_dim):
        xi = X[:,i]
        Xreal[:,i] =  inverse_unitscale_xv(xi, X_ranges[i])
        # Operate on a log scale
        if log_flags[i]:
            Xreal[:,i] =  10**(Xreal[:,i])
            
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
def create_X_mesh_2d(mesh_size: Optional[int] = 41
) -> Tuple[Matrix, Matrix, Matrix]:   
    """Create 2D mesh for testing

    Parameters
    ----------
    mesh_size : int, optional
        mesh size, by default 41

    Returns
    -------
    X_test: Matrix
        X in 2D mesh, 2 columns
    X1: Matrix
        X1 for surface plots
    X2: Matrix
        X2 for surface plots
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

  
def transform_Y_mesh_2d(X: ArrayLike1d, mesh_size: Optional[int] = 41
) -> Matrix:
    """takes in 1 column of X tensor 
    predict the Y values
    convert to real units and return a 2D numpy array 
    in the size of mesh_size*mesh_size

    Parameters
    ----------
    X : ArrayLike1d
        1d tensor or numpy array
    mesh_size : Optional[int], optional
        mesh size, by default 41
        
    Returns
    -------
    X_plot2D: numpy matrix
        in real units for plotting
        
    """
    X = tensor_to_np(X)

    # Convert to numpy for plotting
    X_plot2D = np.reshape(X, (mesh_size, mesh_size))
    
    return X_plot2D
    


def transform_X_2d(X1: Matrix, X2: Matrix, X_ranges: Matrix
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
    X1 = inverse_unitscale_xv(X1, X_ranges[0])
    X2 = inverse_unitscale_xv(X2, X_ranges[1])
    
    return X1, X2


def prepare_full_X_unit(
    X_test: MatrixLike2d, 
    n_dim: int, 
    x_indices: Union[List[int]],
    fixed_values: Union[List[float]],
) -> MatrixLike2d:
    """Create a full X_test matrix given with varying x at dimensions 
    defined by x_indices and fixed values at the rest dimensions

    Parameters
    ----------
    X_test: MatrixLike2d
        X_test as mesh points, in a unit scale 
    n_dim : int
        Dimensional of X, i.e., number of columns 
    x_indices : Union[List[int]]
        indices of varying x variables
    fixed_values : Union[List[float]]
        fixed values in other dimensions, 
        in a unit scale

    Returns
    -------
    X_full: MatrixLike2d
        Test X in a unit scale
    """
    n_points = X_test.shape[0]
    xi_list = [] # a list of the columns
    di_fixed = 0 # index for fixed value dimensions
    di_test = 0 # index for varying x dimensions

    for di in range(n_dim):
        # Get a column from X_test
        if di in x_indices:
            xi = X_test[:, di_test]
            di_test += 1
        # Create a column of fix values
        else:
            fix_value_i = fixed_values[di_fixed]
            xi = np.ones((n_points, 1)) * fix_value_i
            di_fixed += 1

        xi_list.append(xi)
    # Stack the columns into a matrix
    X_full = np.column_stack(xi_list)

    return X_full


def prepare_full_X_real(
    X_test: MatrixLike2d, 
    X_ranges: MatrixLike2d, 
    x_indices: Union[List[int]],
    fixed_values_real: Union[List[float]],
) -> MatrixLike2d:
    """Create a full X_test matrix given with varying x at dimensions 
    defined by x_indices and fixed values at the rest dimensions

    Parameters
    ----------
    X_test : MatrixLike2d
        X as mesh points, in a unit scale
    X_ranges : MatrixLike2d
        list of x ranges
    x_indices : Union[List[int], int]
        indices of varying x variables
    fixed_values_real : Union[List[float], float]]
        fixed values in other dimensions, 
        in a real scale

    Returns
    -------
    X_full: MatrixLike2d
        Test X in a real scale
    """
    # get the dimensions of the system
    n_dim = len(X_ranges)
    n_points = X_test.shape[0]
    xi_list = [] # a list of the columns
    di_fixed = 0 # index for fixed value dimensions
    di_test = 0 # index for varying x dimensions

    for di in range(n_dim):
        if di in x_indices:
            # Get a column from X_test
            xi = X_test[:, di_test]
            # Convert xi to a real scale
            xi = inverse_unitscale_xv(xi, X_ranges[di])
            di_test += 1
        else:
            # No need to scale for given fixed values 
            fix_value_i = fixed_values_real[di_fixed]
            xi = np.ones((n_points, 1)) * fix_value_i
            di_fixed += 1

        xi_list.append(xi)
    # Stack the columns into a matrix
    X_full = np.column_stack(xi_list)

    return X_full


def get_baseline_unit(
    n_dim: int,
    baseline: str
) -> List[float]:
    """Get the baseline values from X_ranges 
    in a unit scale

    Parameters
    ----------
    X_ranges : MatrixLike2d
        list of x ranges
    baseline : str
        the choice of baseline

    Returns
    -------
    List[float]
        a list of baseline values

    Raises
    ------
    ValueError
        if the input baseline is not in the default values
    """

    default_baselines = ['left', 'right', 'center']
    if baseline not in default_baselines:
        raise ValueError('The baseline must be left, right or center.')

    baseline_values = []
    if baseline == 'left':
        baseline_values = [0] * n_dim
    if baseline == 'right':
        baseline_values = [1] * n_dim
    if baseline == 'center':
        baseline_values = [0.5] * n_dim

    return baseline_values


def fill_full_X_test(
    X_test_varying: MatrixLike2d,
    X_ranges: MatrixLike2d, 
    x_indices: Optional[List[int]] = [0, 1],
    fixed_values: Optional[Union[ArrayLike1d, float]] = None,
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = None,
    baseline: Optional[str] = 'left',
) -> Matrix:
    """Choose certain dimensions defined by x_indices, 
    fill them with mesh test points and keep the rest
    as fixed values. If no fixed value input, the baseline values
    are used at those dimensions 

    Parameters
    ----------
    X_test_varying : MatrixLike2d
        mesh test points at dimensions defined by x_indices
    X_ranges : MatrixLike2d
        list of x ranges
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    fixed_values : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a unit scale, by default None
    fixed_values_real : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a real scale, by default None
    baseline : Optional[str], optional
        the choice of baseline, must be left, right or center

    Returns
    -------
    X_test: MatrixLike2d
        Test X in a unit scale

    Raises
    ------
    ValueError
        if the total dimensions do not add up to the system dimensionality
    """
    n_dim = len(X_ranges)

    # conversion in a unit scale
    unit_scale_flag = True
    
     # Convert to a list given a single value input
    if not isinstance(x_indices, list):
        x_indices = [x_indices]

    # if no input fix_values, use baseline values
    if (fixed_values is None) and (fixed_values_real is None):
        fixed_values = get_baseline_unit(n_dim - len(x_indices), baseline)
    
    # update fixed_values if the real values are the input
    if fixed_values_real is not None:
        fixed_values = fixed_values_real
        unit_scale_flag = False

    # Convert to a list given a single value input
    if not isinstance(fixed_values, list):
        fixed_values = [fixed_values]

    # check if all dimensions are provided
    if not n_dim == (len(x_indices)+ len(fixed_values)):
        raise ValueError('The sum of varying dimensions and fixed dimensions \
            should be equal to the dimensionality of the system. Check input.')

    if unit_scale_flag:
        X_test = prepare_full_X_unit(X_test=X_test_varying,
                                    n_dim = n_dim,
                                    x_indices = x_indices,
                                    fixed_values=fixed_values)
    else:
        X_test = prepare_full_X_real(X_test=X_test_varying,
                                    X_ranges=X_ranges,
                                    x_indices = x_indices,
                                    fixed_values_real=fixed_values)
        X_test = unitscale_X(X_test, X_ranges=X_ranges)        

    return X_test


def create_full_X_test_2d(
    X_ranges: MatrixLike2d, 
    x_indices: Optional[List[int]] = [0, 1],
    fixed_values: Optional[Union[ArrayLike1d, float]] = None,
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = None,
    baseline: Optional[str] = 'left',
    mesh_size: Optional[int] = 41
) -> Tuple[Matrix, Matrix, Matrix]:
    """Choose two dimensions, create 2D mesh and keep the rest
    as fixed values. If no fixed value input, the baseline values
    are used at those dimensions 

    Parameters
    ----------
    X_ranges : MatrixLike2d
        list of x ranges
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    fixed_values : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a unit scale, by default None
    fixed_values_real : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a real scale, by default None
    baseline : Optional[str], optional
        the choice of baseline, must be left, right or center
    mesh_size : int, optional
        mesh size, by default 41

    Returns
    -------
    X_test: MatrixLike2d
        Test X in a unit scale
    X1: Matrix
        X1 for surface plots
    X2: Matrix
        X2 for surface plots
    """
    # Create 2d mesh test points  
    X_test_2d, X1, X2 = create_X_mesh_2d(mesh_size)
    # Get full X with fixed values at the given dimensions 
    X_test = fill_full_X_test(X_test_varying=X_test_2d,
                              X_ranges=X_ranges, 
                              x_indices=x_indices,
                              fixed_values=fixed_values,
                              fixed_values_real=fixed_values_real,
                              baseline=baseline)

    return X_test, X1, X2



def create_full_X_test_1d(
    X_ranges: MatrixLike2d, 
    x_index: Optional[int] = 0,
    fixed_values: Optional[Union[ArrayLike1d, float]] = None,
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = None,
    baseline: Optional[str] = 'left',
    mesh_size: Optional[int] = 41
) -> Matrix:
    """Choose two dimensions, create 2D mesh and keep the rest
    as fixed values. If no fixed value input, the baseline values
    are used at those dimensions 

    Parameters
    ----------
    X_ranges : MatrixLike2d
        list of x ranges
    x_index : Optional[int], optional
        index of two x variables, by default [0]
    fixed_values : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a unit scale, by default None
    fixed_values_real : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a real scale, by default None
    baseline : Optional[str], optional
        the choice of baseline, must be left, right or center
    mesh_size : int, optional
        mesh size, by default 41

    Returns
    -------
    X_test: MatrixLike2d
        Test X in a unit scale
    """
    # Create 1d mesh test points  
    X_test_1d = np.linspace(0, 1, mesh_size)
    # Expand to 2d
    X_test_1d = np.expand_dims(X_test_1d, axis=1)
    
    # Get full X with fixed values at the given dimensions 
    X_test = fill_full_X_test(X_test_varying=X_test_1d,
                              X_ranges=X_ranges, 
                              x_indices=x_index,
                              fixed_values=fixed_values,
                              fixed_values_real=fixed_values_real,
                              baseline=baseline)

    return X_test


#%% Encoding and decoding functions 
def binary_search(nums: ArrayLike1d, target: float) -> int:
    """Modified binary search
    return the index in the original array
    where all values of the elements to its left 
    and itself are smaller than or equal to the target

    Parameters
    ----------
    nums : ArrayLike1d
        sorted 1d array
    target : float
        target value

    Returns
    -------
    ans: int
        target index
    """
    
    l = 0
    r = len(nums)
    
    while (l < r): 
        m = int(l + (r-l)/2)
        if (nums[m] > target): r = m
        else: l = m +1 

    ans = l-1

    return ans


def find_nearest_value(values: ArrayLike1d, x: float) -> Tuple[float, int]:
    """find the nearest value in an array
    given a target value

    Parameters
    ----------
    values : ArrayLike1d
        sorted array, in ascending order
    x : float
        original value

    Returns
    -------
    ans: float
        Nearest value to the original
    index_target: int
        index of the target in the given array
    """

    n = len(values)
    index_left = binary_search(values, x) 
    index_target = index_left

    # check the right side
    if index_left < n-1:
        index_right = index_left + 1
        diff_left = np.abs(values[index_left] - x)
        diff_right = np.abs(values[index_right] - x)
        
        if diff_left > diff_right:
            index_target = index_right

    ans = values[index_target]

    return ans, index_target


def encode_xv(xv: ArrayLike1d, encoding: ArrayLike1d) -> ArrayLike1d:
    """Convert original data to encoded data

    Parameters
    ----------
    xv : ArrayLike1d
        original x array
    encoding : ArrayLike1d
        encoding array

    Returns
    -------
    xv_encoded: ArrayLike1d
        encoded data array
    """

    xv_encoded =  copy.deepcopy(xv)
    xv_encoded = tensor_to_np(xv_encoded)

    for i in range(len(xv)):
        xv_encoded[i], _ = find_nearest_value(encoding, xv[i])
    
    return xv_encoded


def decode_xv(xv_encoded: ArrayLike1d, 
              encoding: ArrayLike1d,
              values: ArrayLike1d) -> ArrayLike1d:
    """Decoded the data to ordinal or categorical values

    Parameters
    ----------
    xv_encoded : ArrayLike1d
        encoded data array
    encoding : ArrayLike1d
        encoding array
    values : ArrayLike1d
        ordinal or categorical values

    Returns
    -------
    xv_decoded: ArrayLike1d
        data decoded, with the original 
        ordinal or categorical values
    """

    xv_decoded =  copy.deepcopy(xv_encoded)
    xv_decoded = tensor_to_np(xv_decoded)
    # Set array type as object for str values
    xv_decoded = xv_decoded.astype(object)

    for i in range(len(xv_encoded)):
        _, index_i = find_nearest_value(encoding, xv_encoded[i])
        xv_decoded[i] = values[index_i]
    
    return xv_decoded


def real_to_encode_X(
    X: MatrixLike2d, 
    X_types: List[str],   
    encodings: MatrixLike2d,
    X_ranges: Optional[MatrixLike2d] = None,  
    log_flags: Optional[list] = None, 
    decimals: Optional[int] = None
) -> Matrix:
    """Takes in a matrix in a real scale
    from the relaxed continuous space,
    rounds (encodes) to the available values,
    and converts it into a unit scale

    Parameters
    ----------
    X : MatrixLike2d
        original matrix in a real scale
    X_types : List[str]
        list of parameter types
    encodings : MatrixLike2d
        encoding Matrix
    X_ranges : Optional[MatrixLike2d], optional
        list of x ranges, by default None
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
    Xencode: numpy matrix
        matrix scaled to a unit scale 
    """
    #If 1D, make it 2D a matrix
    if len(X.shape)<2:
        X = copy.deepcopy(X)
        X = np.expand_dims(X, axis=1) #If 1D, make it 2D array
        
    n_dim = X.shape[1] #the number of column in X

    if X_ranges is None: # X_ranges not defined
        X_ranges = get_ranges_X(X)
    X_ranges = expand_ranges_X(X_ranges) #expand to 2d
    
    if log_flags is None: log_flags = [False] * n_dim
    
    # Initialize with a zero matrix
    Xencode = np.zeros((X.shape[0], X.shape[1]))
    for i in range(n_dim):
        xi = X[:,i]
        # scale based on the type
        if X_types[i] == 'continuous':  
            Xencode[:,i] =  unitscale_xv(xi, X_ranges[i])
        else: #categorical and oridinal
            encoding_unit = unitscale_xv(encodings[i], X_ranges[i])
            Xencode[:, i] = encode_xv(xi, encoding_unit)

        # Operate on a log scale
        if log_flags[i]: 
            Xencode[:,i] =  np.log10(Xencode[:, i])
    
    # Round up if necessary
    if not decimals == None:
        Xencode = np.around(Xencode, decimals = decimals)  
    
    return Xencode


def unit_to_encode_X(
    X: MatrixLike2d, 
    X_types: List[str],   
    encodings: MatrixLike2d,
    X_ranges: Optional[MatrixLike2d] = None,  
    log_flags: Optional[list] = None, 
    decimals: Optional[int] = None
) -> Matrix:
    """Takes in a matrix in a unit scale
    from the relaxed continuous space,
    rounds (encodes) to the available values

    Parameters
    ----------
    X : MatrixLike2d
        original matrix in a real scale
    X_types : List[str]
        List of parameter types
    encodings : MatrixLike2d
        encoding Matrix
    X_ranges : Optional[MatrixLike2d], optional
        list of x ranges, by default None
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
    Xencode: numpy matrix
        matrix scaled to a unit scale
    """
    #If 1D, make it 2D a matrix
    if len(X.shape)<2:
        X = copy.deepcopy(X)
        X = np.expand_dims(X, axis=1) #If 1D, make it 2D array
        
    n_dim = X.shape[1] #the number of column in X

    if X_ranges is None: # X_ranges not defined
        X_ranges = [0,1] * n_dim
    
    if log_flags is None: log_flags = [False] * n_dim
    
    # Initialize with a zero matrix
    Xencode = np.zeros((X.shape[0], X.shape[1]))
    for i in range(n_dim):
        xi = X[:,i]
        # scale based on the type
        if X_types[i] == 'continuous':  
            Xencode[:, i] = xi
        else: #categorical and oridinal
            encoding_unit = unitscale_xv(encodings[i], X_ranges[i])
            Xencode[:, i] = encode_xv(xi, encoding_unit)

        # Operate on a log scale
        if log_flags[i]: 
            Xencode[:,i] =  np.log10(Xencode[:, i])
    
    # Round up if necessary
    if not decimals == None:
        Xencode = np.around(Xencode, decimals = decimals)  
    
    return Xencode



def encode_to_real_X(
    X: MatrixLike2d, 
    X_types: List[str],   
    encodings: MatrixLike2d,
    values_2D: MatrixLike2d,
    all_continuous: Optional[bool] = True, 
    X_ranges: Optional[MatrixLike2d] = None, 
    log_flags: Optional[list] = None, 
    decimals: Optional[int] = None
) -> Matrix:
    """Takes in a matrix in a unit scale
    from the encoding space,
    converts it into a real scale

    Parameters
    ----------
    X : MatrixLike2d
        original matrix in a real scale
    X_types : List[str]
        list of parameter types
    encodings : MatrixLike2d
        encoding Matrix
    values_2D: MatrixLike2d
        list of values for ordinal or categorical parameters
    all_continuous: Optional[bool], Optional
        flag for all continuous parameters
    X_ranges : Optional[MatrixLike2d], optional
        list of x ranges, by default None
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
    Xreal: numpy matrix
        matrix scaled to a unit scale
    """
    #If 1D, make it 2D a matrix
    if len(X.shape)<2:
        X = copy.deepcopy(X)
        X = np.expand_dims(X, axis=1) #If 1D, make it 2D array
        
    n_dim = X.shape[1] #the number of column in X

    if X_ranges is None: # X_ranges not defined
        X_ranges = get_ranges_X(X)
    X_ranges = expand_ranges_X(X_ranges) #expand to 2d
    
    if log_flags is None: log_flags = [False] * n_dim
    
    # Initialize with a zero matrix
    Xreal = np.zeros((X.shape[0], X.shape[1]), dtype=object)

    if all_continuous:
        Xreal = np.zeros((X.shape[0], X.shape[1]))
    
    for i in range(n_dim):
        xi = X[:,i]
        # scale based on the type
        if X_types[i] == 'continuous':
            Xreal[:,i] =  inverse_unitscale_xv(xi, X_ranges[i])
        else: #categorical and oridinal
            encoding_unit = unitscale_xv(encodings[i], X_ranges[i])
            Xreal[:, i] = decode_xv(xi, encoding_unit, values_2D[i])
        # Operate on a log scale
        if log_flags[i]:
            Xreal[:,i] =  10**(Xreal[:,i])
    
    # Round up if necessary
    if not decimals == None:
        Xreal = np.around(Xreal, decimals = decimals)  
    
    return Xreal


def real_to_encode_ParameterSpace(
    X: MatrixLike2d, 
    parameter_space: ParameterSpace,
    log_flags: Optional[list] = None, 
    decimals: Optional[int] = None
) -> Matrix:
    """Takes in a matrix in a real scale
    from the relaxed continuous space,
    rounds (encodes) to the available values,
    and converts it into a unit scale.
    Using ParameterSpace object

    Parameters
    ----------
    X : MatrixLike2d
        original matrix in a real scale
    parameter_space : ParameterSpace
        ParameterSpace object
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
    Xencode: numpy matrix
        matrix scaled to a unit scale 
    """
    Xencode = real_to_encode_X(X=X, 
                           X_types=parameter_space.X_types,
                           encodings=parameter_space.encodings,
                           X_ranges=parameter_space.X_ranges,
                           log_flags=log_flags, 
                           decimals=decimals)
    return Xencode


def unit_to_encode_ParameterSpace(
    X: MatrixLike2d, 
    parameter_space: ParameterSpace,
    log_flags: Optional[list] = None, 
    decimals: Optional[int] = None
) -> Matrix:
    """Takes in a matrix in a unit scale
    from the relaxed continuous space,
    rounds (encodes) to the available values
    Using ParameterSpace object

    Parameters
    ----------
    X : MatrixLike2d
        original matrix in a real scale
    parameter_space : ParameterSpace
        ParameterSpace object
    log_flags : Optional[list], optional
        list of boolean flags
        True: use the log scale on this dimensional
        False: use the normal scale 
        by default None
    decimals : Optional[int], optional
        Number of decimal places to keep
        by default None, i.e. no rounding up 
    """
    Xencode = unit_to_encode_X(X=X, 
                           X_types=parameter_space.X_types,
                           encodings=parameter_space.encodings,
                           X_ranges=parameter_space.X_ranges,
                           log_flags=log_flags, 
                           decimals=decimals)
    return Xencode


def encode_to_real_ParameterSpace(
    X: MatrixLike2d, 
    parameter_space: ParameterSpace,
    log_flags: Optional[list] = None, 
    decimals: Optional[int] = None
) -> Matrix:
    """Takes in a matrix in a unit scale
    from the encoding space,
    converts it into a real scale
    Using ParameterSpace object

    Parameters
    ----------
    X : MatrixLike2d
        original matrix in a real scale
    parameter_space : ParameterSpace
        ParameterSpace object
    log_flags : Optional[list], optional
        list of boolean flags
        True: use the log scale on this dimensional
        False: use the normal scale 
        by default None
    decimals : Optional[int], optional
        Number of decimal places to keep
        by default None, i.e. no rounding up 
    """
    Xreal = encode_to_real_X(X=X, 
                           X_types=parameter_space.X_types,
                           encodings=parameter_space.encodings,
                           values_2D=parameter_space.values_2D,
                           all_continuous=parameter_space.all_continuous,
                           X_ranges=parameter_space.X_ranges,
                           log_flags=log_flags, 
                           decimals=decimals)
    return Xreal





