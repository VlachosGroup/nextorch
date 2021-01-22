"""
nextorch.plotting

Creates 2-dimensional and 3-dimensional visualizations 
The plots are rendered using matplotlib as a backend
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes._axes import Axes
import numpy as np

import torch
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model

from typing import Optional, TypeVar, Union, Tuple, List
from nextorch.utils import Array, Matrix, ArrayLike1d, MatrixLike2d
from nextorch.utils import tensor_to_np, np_to_tensor, standardize_X
from nextorch.bo import eval_acq_func, \
    eval_objective_func, predict_model, predict_real, Experiment


# Set matplotlib default values
font = {'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.major.width'] = 2


def add_2D_x_slice(
    ax: Axes, 
    xvalue: float, 
    yrange: List[float], 
    zrange: List[float], 
    mesh_size: Optional[int] = 100
) -> Axes:
    """Adds a 2-dimensional plane on x axis, parallel to y-z plane
    in the 3-dimensional (x, y, z) space

    Parameters
    ----------
    ax :  `matplotlib.axes.Axes.axis`_
        Ax of the plot
    xvalue : float
        the value on x axis which the slice is made
    yrange : list of float
        [left bound, right bound] of y value
    zrange : list of float
        [left bound, right bound] of z value
    mesh_size : Optional[int], optional
        mesh size on the slice, by default 100

    Returns
    -------
    ax : `matplotlib.axes.Axes.axis`_
        Axes of the plots

    .. _`matplotlib.axes.Axes.axis`: https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axis.html
    """    
    colormap = cm.summer
    Y, Z = np.meshgrid(np.linspace(yrange[0], yrange[1], mesh_size), np.linspace(zrange[0], zrange[1], mesh_size), indexing = 'ij')
    X = xvalue * np.ones((mesh_size, mesh_size))
    ax.plot_surface(X, Y, Z,  cmap=colormap, rstride=1 , cstride=1, shade=False, alpha = 0.7)

    return ax


def add_2D_y_slice(
    ax: Axes, 
    yvalue: float, 
    xrange: List[float], 
    zrange: List[float], 
    mesh_size: Optional[int] = 100
) -> Axes:
    """Adds a 2-dimensional plane on y axis, parallel to x-z plane
    in the 3-dimensional (x, y, z) space

    Parameters
    ----------
    ax :  `matplotlib.axes.Axes.axis`_
        Ax of the plot
    yvalue : float
        the value on y axis which the slice is made
    xrange : list of float
        [left bound, right bound] of x value
    zrange : list of float
        [left bound, right bound] of z value
    mesh_size : Optional[int], optional
        mesh size on the slice, by default 100

    Returns
    -------
    ax : `matplotlib.axes.Axes.axis`_
        Axes of the plots

    .. _`matplotlib.axes.Axes.axis`: https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axis.html
    """    
    colormap = cm.summer
    Y, Z = np.meshgrid(np.linspace(xrange[0], xrange[1], mesh_size), np.linspace(zrange[0], zrange[1], mesh_size), indexing = 'ij')
    X = yvalue * np.ones((mesh_size, mesh_size))
    ax.plot_surface(X, Y, Z,  cmap=colormap, rstride=1 , cstride=1, shade=False, alpha = 0.7)

    return ax


def add_2D_z_slice(
    ax: Axes, 
    zvalue: float, 
    xrange: List[float], 
    yrange: List[float], 
    mesh_size: Optional[int] = 100
) -> Axes:
    """Adds a 2-dimensional plane on z axis, parallel to x-y plane
    in the 3-dimensional (x, y, z) space

    Parameters
    ----------
    ax :  `matplotlib.axes.Axes.axis`_
        Ax of the plot
    zvalue : float
        the value on z axis which the slice is made
    xrange : list of float
        [left bound, right bound] of x value
    yrange : list of float
        [left bound, right bound] of y value
    mesh_size : Optional[int], optional
        mesh size on the slice, by default 100

    Returns
    -------
    ax : `matplotlib.axes.Axes.axis`_
        Axes of the plots

    .. _`matplotlib.axes.Axes.axis`: https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.axis.html
    """    
    colormap = cm.summer
    X, Y = np.meshgrid(np.linspace(xrange[0], xrange[1], mesh_size), np.linspace(yrange[0], yrange[1], mesh_size), indexing = 'ij')
    Z = zvalue * np.ones((mesh_size, mesh_size))
    ax.plot_surface(X, Y, Z,  cmap=colormap, rstride=1 , cstride=1, shade=False, alpha = 0.7)

    return ax


def plot_acq_func_1d(
    acq_func: AcquisitionFunction, 
    X_test: MatrixLike2d, 
    X_train: Optional[MatrixLike2d] = None, 
    X_new: Optional[MatrixLike2d] = None):
    """Plot 1-dimensional acquision function 

    Parameters
    ----------
    acq_func : 'botorch.acquisition.AcquisitionFunction'_
        the acquision function object
    X_test : MatrixLike2d
        Test data points for plotting
    X_train : Optional[MatrixLike2d], optional
        Training data points, by default None
    X_new : Optional[MatrixLike2d], optional
        The next data point, i.e the infill points,
        by default None

    .._'botorch.acquisition.AcquisitionFunction': https://botorch.org/api/acquisition.html
    """
    # compute acquicision function values at X_test and X_train
    acq_val_test = eval_acq_func(acq_func, X_test)
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(X_test, acq_val_test, 'b-', label = 'Acquisition')
    # Plot training points as black 
    if X_train is not None:
        acq_val_train = eval_acq_func(acq_func, X_train)
        ax.scatter(X_train, acq_val_train, s = 120, c= 'k', marker = '*', label = 'Initial Data')
    # Plot the new infill points as red stars
    if X_new is not None:
        acq_val_new = eval_acq_func(acq_func, X_new)
        ax.scatter(X_new, acq_val_new,  s = 120, c ='r', marker = '*', label = 'Infill Data')
    
    ax.ticklabel_format(style = 'sci', axis = 'y', scilimits = (-2,2) )
    ax.set_xlabel('x')
    ax.set_ylabel(r'$ \alpha$')    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout()
    plt.show()


def plot_objective_func_1d(
    model: Model, 
    X_test: MatrixLike2d, 
    Y_test: Optional[MatrixLike2d] = None, 
    X_train: Optional[MatrixLike2d] = None,
    Y_train: Optional[MatrixLike2d] = None, 
    X_new: Optional[MatrixLike2d] = None,
    Y_new: Optional[MatrixLike2d] = None,
    plot_real: Optional[bool] = False,
    Y_mean: Optional[MatrixLike2d] = None,
    Y_std: Optional[MatrixLike2d] = None):
    """Plot objective function along 1 dimension
    Input X variables are in a unit scale and
    Input Y variables are in a real scale

    Parameters
    ----------
    model : 'botorch.models.model.Model'_
        A GP model
    X_test : MatrixLike2d
        Test X data points for plotting
    Y_test : Optional[MatrixLike2d], optional
        Test Y data if the objective function is known, 
        by default None
    X_train : Optional[MatrixLike2d], optional
        Training X data points, by default None
    Y_train : Optional[MatrixLike2d], optional
        Training Y data points, by default None
    X_new : Optional[MatrixLike2d], optional
        The next X data point, i.e the infill points,
        by default None
    Y_new : Optional[MatrixLike2d], optional
        The next Y data point, i.e the infill points,
        by default None
    plot_real : Optional[bool], optional
        if true plot in the real scale for Y, 
        by default False
    Y_mean : MatrixLike2d
        The mean of initial Y set
    Y_std : MatrixLike2d
        The std of initial Y set

    Raises
    ------
    ValueError
        if X_train is provided but Y_train is not
    ValueError
        if X_new is provided but Y_new is not
    ValueError
        if plot in the real scale but Y_mean or Y_std is not provided

    :_'botorch.models.model.Model': https://botorch.org/api/models.html
    """    
    # handle the edge cases
    if (X_train is not None) and (Y_train is None):
        raise ValueError("Plot X_train, must also input Y_train")
    if (X_new is not None) and (Y_new is None):
        raise ValueError("Plot X_new, must also input Y_new")
    if not plot_real and (Y_mean is None or Y_std is None):
        raise ValueError("Plot in the standard scale, must supply the mean and std of Y set")

    if plot_real: # Y in a real scale
        Y_test_pred, Y_test_lower_pred, Y_test_upper_pred = predict_real(model, 
                                                                        X_test, 
                                                                        Y_mean, 
                                                                        Y_std, 
                                                                        return_type= 'np')
    else: # Y in a standardized scale
        Y_test = standardize_X(Y_test, Y_mean, Y_std, return_type= 'np') #standardize Y_test
        Y_train = standardize_X(Y_train, Y_mean, Y_std, return_type= 'np') 
        Y_new = standardize_X(Y_new, Y_mean, Y_std, return_type= 'np') 

        Y_test_pred, Y_test_lower_pred, Y_test_upper_pred = predict_model(model, 
                                                                          X_test, 
                                                                          return_type= 'np')
    # reduce the dimension to 1d arrays
    X_test = np.squeeze(tensor_to_np(X_test))
    Y_test_pred = np.squeeze(Y_test_pred)
    Y_test_lower_pred = np.squeeze(Y_test_lower_pred)
    Y_test_upper_pred = np.squeeze(Y_test_upper_pred)

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot the groud truth Y_test if provided
    if Y_test is not None:
        Y_test = np.squeeze(tensor_to_np(Y_test))
        ax.plot(X_test, Y_test, 'k--', label = 'Objective f(x)')

    # Plot posterior means as blue line
    ax.plot(X_test, Y_test_pred, 'b', label = 'Posterior Mean')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(X_test, Y_test_lower_pred, Y_test_upper_pred, alpha=0.5, label = 'Confidence')

    # Plot training points as black stars
    if X_train is not None:
        X_train = np.squeeze(tensor_to_np(X_train))
        Y_train = np.squeeze(tensor_to_np(Y_train))
        ax.scatter(X_train, Y_train, s =120, c= 'k', marker = '*', label = 'Initial Data')
    # Plot the new infill points as red stars
    if X_new is not None:    
        X_new = np.squeeze(tensor_to_np(X_new))
        Y_new = np.squeeze(tensor_to_np(Y_new))
        ax.scatter(X_new, Y_new, s = 120, c = 'r', marker = '*', label = 'Infill Data')
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout()
    plt.show()


def plot_exp_objective_func_1d(
    Exp: Experiment,
    X_test: MatrixLike2d, 
    Y_test: Optional[MatrixLike2d] = None, 
    X_new: Optional[MatrixLike2d] = None,
    Y_new: Optional[MatrixLike2d] = None,
    plot_real: Optional[bool] = False):
    """Plot objective function along 1 dimension
       using Experiment object

    Parameters
    ----------
    Exp : Experiment
        Experiment object
    X_test : MatrixLike2d
        Test X data points for plotting
    Y_test : Optional[MatrixLike2d], optional
        Test Y data if the objective function is known, 
        by default None
    X_new : Optional[MatrixLike2d], optional
        The next X data point, i.e the infill points,
        by default None
    Y_new : Optional[MatrixLike2d], optional
        The next Y data point, i.e the infill points,
        by default None
    plot_real : Optional[bool], optional
        if true plot in the real scale for Y, 
        by default False
    """
    # if no Y_test input, generate Y_test from objective function
    if (Y_test is None) and (Exp.objective_func is not None):
        Y_test = eval_objective_func(X_test, Exp.X_ranges, Exp.objective_func)

    # if no Y_new input, generate Y_test from objective function
    if (X_new is not None) and (Exp.objective_func is not None) and (Y_new is None):
        Y_new = eval_objective_func(X_new, Exp.X_ranges, Exp.objective_func)

    plot_objective_func_1d(model = Exp.model, 
                           X_test = X_test,
                           Y_test = Y_test,
                           X_train = Exp.X,
                           Y_train = Exp.Y_real, #be sure to use Y_real
                           X_new = X_new,
                           Y_new = Y_new,
                           plot_real = plot_real,
                           Y_mean = Exp.Y_mean,
                           Y_std= Exp.Y_std)
