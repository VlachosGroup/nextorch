"""
Handles input and output
"""

import os
import pandas as pd 
from pandas import DataFrame

import numpy as np
import copy
from typing import Optional, TypeVar, Union, Tuple, List

from torch import var
from nextorch.utils import Array, Matrix, ArrayLike1d, MatrixLike2d



def read_excel(
    file_path: str, 
    sheet_name: Optional[str] = 0, 
    var_names: Optional[List[str]] = None, 
    skiprows: Optional[list] = None,
    index_col: Optional[int] = 0, 
    verbose: Optional[bool] = True
) -> Tuple[DataFrame, DataFrame]:
    """Reads an excel file and returns the data in pandas Dataframe

    Parameters
    ----------
    file_path : str
        Path of the excel spreedsheet
    sheet_name : Optional[str], optional
        Name of the excel sheet, by default 0, i.e., the first sheet
    var_names : Optional[List[str]], optional
        Names of variables to include, by default None, 
    skiprows : Optional[list], optional
        Rows to skip at the beginning (0-indexed)
        by default None, i.e. import all
    index_col : Optional[int], optional
        Column (0-indexed) to use as the row labels of the DataFrame
        by default 0 assuming the indices are saved in the first column
    verbose : Optional[bool], optional
        Flag whether to output print statements, by default True

    Returns
    -------
    data: `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        Input data
    data_full: `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        Full data from the file
    
    """  

    data_full = pd.read_excel(file_path, 
                              sheet_name= sheet_name, 
                              skiprows = skiprows,
                              index_col = index_col)
    # Select the variables
    if var_names is not None:
        data = data_full[var_names]
    else:
        data = data_full

    # Print statments regards the input
    if verbose: 
        var_names_in = list(data.columns)
        n_vars = len(var_names_in) 
        n_points = data.shape[0]
        var_names_s = ''
        print('\nInput data contains {} points, {} variables:'.format(n_points, n_vars))
        for vi in var_names_in:
            var_names_s += vi + ', '
        print('\t{}'.format(var_names_s.strip(', ')))

    return data, data_full


def read_csv(
    file_path: str, 
    var_names: Optional[List[str]] = None, 
    skiprows: Optional[list] = None,
    index_col: Optional[int] = 0, 
    verbose: Optional[bool] = True
) -> Tuple[DataFrame, DataFrame]:
    """Reads a csv file and returns the data in pandas Dataframe

    Parameters
    ----------
    file_path : str
        Path of the csv file
    var_names : Optional[List[str]], optional
        Names of variables to include, by default None, 
    skiprows : Optional[list], optional
        Rows to skip at the beginning (0-indexed)
        by default None, i.e. import all
    index_col : Optional[int], optional
        Column (0-indexed) to use as the row labels of the DataFrame
        by default 0 assuming the indices are saved in the first column
    verbose : Optional[bool], optional
        Flag whether to output print statements, by default True

    Returns
    -------
    data: `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        Input data 
    data_full: `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        Full data from the file

    """   
    data_full = pd.read_csv(file_path, 
                            skiprows = skiprows,
                            index_col = index_col)
    # Select the variables
    if var_names is not None:
        data = data_full[var_names]
    else:
        data = data_full

    # Print statments regards the input
    if verbose: 
        var_names_in = list(data.columns)
        n_vars = len(var_names_in) 
        n_points = data.shape[0]
        var_names_s = ''
        print('\nInput data contains {} points, {} variables:'.format(n_points, n_vars))
        for vi in var_names_in:
            var_names_s += vi + ', '
        print('\t{}'.format(var_names_s.strip(', ')))

    return data, data_full

def split_X_y(
    data: DataFrame,
    Y_names: Union[str, List[str]], 
    X_names: Optional[List[str]] = None
) -> Tuple[Matrix, Matrix, List[str], List[str]]:
    """Splits the data into independent (X) 
    and dependent (y) varibles 

    Parameters
    ----------
    data : `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        Input dataframe
    Y_names : Union[str, List[str]]
        Name(s) of dependent variables, can be a single str or list of str
    X_names : Optional[List[str]], optional
        Names of independent variables, by default None, i.e. select all

    Returns
    -------
    X: Matrix
        Independent variable matrix
    Y: Matrix 
        Dependent variable matrix
    X_names: List[str]
        Independent variables names
    Y_names: List[str]
        Dependent variable matrix
    
    """    
    if isinstance(Y_names, str): 
        Y_names = [Y_names]

    var_names = list(data.columns)
    # Select the rest if X_names are not specified
    if X_names is None:
        X_names = [ni for ni in var_names if ni not in Y_names]

    X = np.array(data[X_names])
    Y = np.array(data[Y_names])

    return X,  Y, X_names, Y_names


def np_to_dataframe(
    X: Union[Matrix, list],
    var_names: Optional[Union[str, List[str]]] = None,
    n: Optional[int] = 1,
) -> DataFrame:
    """Convert a list numpy matrices to a single dataframe

    Parameters
    ----------
    X : Union[Matrix, list]
        List numpy matrices or a single matrix
    var_names : Optional[Union[str, List[str]]], optional
        Names of variables, by default None
    n : Optional[int], optional
        number of rows in output dataframe, by default 1

    Returns
    -------
    data: `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
        Output data 

    """
    # Input is a list of matrices 
    if isinstance(X, list):
        X_all = [] # copy of all X 

        for Xi in X: 
            Xi = np.array(Xi)
            if len(Xi.shape) == 0: # if scalar, add one more dimension
                Xi = np.array(Xi)[np.newaxis]
            if len(Xi.shape) < 2: #If 1D, make it 2D array
                Xi = copy.deepcopy(Xi)
                Xi = np.reshape(Xi, (n, -1))             
            X_all.append(Xi)
            
        # Concatenate along column wise
        X_all = np.concatenate(X_all, axis=1)

    # Input is a single matrix 
    else:
        if len(X.shape)<2:
            X_all = copy.deepcopy(X)
            X_all = np.reshape(X_all, (n, -1))   # X_all = np.expand_dims(X, axis=1) #If 1D, make it 2D array
        else:
            X_all = X.copy()

    # Get the number of columns
    n_col = X_all.shape[1]
    # Set default variable names
    if var_names is None:
        var_names = ['x' + str(i+1) for i in range(n_col)]
    if isinstance(var_names, str):
        var_names = [var_names]
    
    data = pd.DataFrame(X_all, columns=var_names)

    return data


"""
GUI module
"""

    