"""
nextorch.io

Handles input and output files
"""

import os
import pandas as pd 
import numpy as np
from typing import Optional, TypeVar, Union, Tuple
# Create a type variable for 1D arrays from numpy
array = TypeVar('array')
# Create a type variable for 2D arrays from numpy and call it as a matrix
matrix = TypeVar('matrix')
# Create a type variable for pandas dataframe
dataframe = TypeVar('dataframe')

def read_excel(
    file_path: str, 
    sheet_name: Optional[str] = 0, 
    var_names: Optional[list] = None, 
    skiprows: Optional[list] = None,
    index_col: Optional[int] = 0, 
    verbose: Optional[bool] = True,
) -> dataframe:
    """Reads an excel file and returns the data in pandas Dataframe

    Parameters
    ----------
    file_path : str
        Path of the excel spreedsheet
    sheet_name : Optional[str], optional
        Name of the excel sheet, by default 0, i.e., the first sheet
    var_names : Optional[list], optional
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
    data: dataframe
        Input data saved in pandas dataframe
    """  

    data = pd.read_excel(file_path, 
                        sheet_name= sheet_name, 
                        names = var_names,
                        skiprows = skiprows,
                        index_col = index_col)
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

    return data


def read_csv(
    file_path: str, 
    var_names: Optional[list] = None, 
    skiprows: Optional[list] = None,
    index_col: Optional[int] = 0, 
    verbose: Optional[bool] = True,
) -> dataframe:
    """Reads a csv file and returns the data in pandas Dataframe

    Parameters
    ----------
    file_path : str
        Path of the csv file
    var_names : Optional[list], optional
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
    data: dataframe
        Input data saved in pandas dataframe
    """   
    data = pd.read_csv(file_path, 
                        names = var_names,
                        skiprows = skiprows,
                        index_col = index_col)
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

    return data

def split_X_y(
    data: dataframe,
    y_names: Union[str, list], 
    X_names: Optional[list] = None
) -> Tuple[matrix, list, matrix, list]:
    """Splits the data into independent (X) 
    and dependent (y) varibles 

    Parameters
    ----------
    data : dataframe
        Input dataframe
    y_names : Union[str, list]
        Name(s) of dependent variables, can be a single str or list of str
    X_names : Optional[list], optional
        Names of independent variables, by default None, i.e. select all

    Returns
    -------
    X, y: Tuple[matrix, list, matrix, list]
        Independent variable matrix, names and
        dependent variable matrix, names 
    """    
    if isinstance(y_names, str): 
        y_names = [y_names]

    var_names = list(data.columns)
    # Select the rest if X_names are not specified
    if X_names is None:
        X_names = [ni for ni in var_names if ni not in y_names]

    X = np.array(data[X_names])
    y = np.array(data[y_names])

    return X, X_names, y, y_names



"""
GUI module
"""

    