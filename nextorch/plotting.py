"""
nextorch.plotting

Creates 2-dimensional and 3-dimensional visualizations 
The plots are rendered using matplotlib as a backend
"""
import os, sys
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# Set global colormap
colormap = cm.jet


#%% Parity plots
def parity(
    y1: MatrixLike2d, 
    y2: MatrixLike2d, 
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None, 
    i_iter: Optional[Union[str, int]] = ''):
    """Plot parity plot comparing the ground true 
    objective function values against predicted model mean

    Parameters
    ----------
    y1 : MatrixLike2d
        Ground truth values
    y2 : MatrixLike2d
        Model predicted values
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory
    i_iter: Optional[Union[str, int]], optional
        Iteration number to add to the figure name
        by default ''
    """
    y1 = np.squeeze(tensor_to_np(y1))
    y2 = np.squeeze(tensor_to_np(y2))

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y1, y2, s=60, alpha = 0.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")

    lims = [
        np.min([y1.min(), y2.min()]),  # min of both axes
        np.max([y1.max(), y2.max()]),  # max of both axes
    ]
    # number of sections in the axis
    nsections = 5
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticks(np.around(np.linspace(lims[0], lims[1], nsections), 2))
    ax.set_yticks(np.around(np.linspace(lims[0], lims[1], nsections), 2))
    ax.set_xticklabels(np.around(np.linspace(lims[0], lims[1], nsections), 2))
    ax.set_yticklabels(np.around(np.linspace(lims[0], lims[1], nsections), 2))

    plt.show()

    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        fig.savefig(os.path.join(save_path, 'parity_'+ str(i_iter) + '.png'), 
                    bbox_inches="tight")


def parity_exp(Exp: Experiment, 
               save_fig: Optional[bool] = False, 
               design_name: Optional[Union[str, int]] = 'final'):
    """Plot parity plot comparing the ground true 
    objective function values against predicted model mean
    Using Experiment object

    Parameters
    ----------
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory
    design_name : Optional[Union[str, int]], optional
        Design name to add to the figure name
        by default 'final'
    """
    
    Y_real_pred = Exp.validate_training(show_confidence=False)

    parity(y1=Exp.Y_real, 
           y2=Y_real_pred,
           save_fig=save_fig,
           save_path=Exp.exp_path,
           i_iter = design_name)
    
def parity_with_ci(
    y1: MatrixLike2d, 
    y2: MatrixLike2d, 
    y2_lower: MatrixLike2d,
    y2_upper: MatrixLike2d,
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None, 
    i_iter: Optional[Union[str, int]] = ''):
    """Plot parity plot comparing the ground true 
    objective function values against predicted model mean
    with predicted confidence interval as error bars 

    Parameters
    ----------
    y1 : MatrixLike2d
        Ground truth values
    y2 : MatrixLike2d
        Model predicted values
    y2_lower: MatrixLike2d
    y2_upper: MatrixLike2d

    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory
    i_iter: Optional[Union[str, int]], optional
        Iteration number to add to the figure name
        by default ''
    """
    y1 = np.squeeze(tensor_to_np(y1))
    y2 = np.squeeze(tensor_to_np(y2))
    y2_lower = np.squeeze(tensor_to_np(y2_lower))
    y2_upper = np.squeeze(tensor_to_np(y2_upper))
    # calculate the error margin
    y2err = np.row_stack((np.abs(y2_lower - y2), np.abs(y2_upper - y2))) 
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.errorbar(y1, y2, yerr = y2err, fmt = 'o', capsize = 2, alpha = 0.5)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    
    lims = [
        np.min([y1.min(), y2.min()]),  # min of both axes
        np.max([y1.max(), y2.max()]),  # max of both axes
    ]
    # number of sections in the axis
    nsections = 5
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticks(np.around(np.linspace(lims[0], lims[1], nsections), 2))
    ax.set_yticks(np.around(np.linspace(lims[0], lims[1], nsections), 2))
    ax.set_xticklabels(np.around(np.linspace(lims[0], lims[1], nsections), 2))
    ax.set_yticklabels(np.around(np.linspace(lims[0], lims[1], nsections), 2))

    plt.show()
    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        fig.savefig(os.path.join(save_path, 'parity_w_ci_'+ str(i_iter) + '.png'), 
                    bbox_inches="tight")


def parity_with_ci_exp(Exp: Experiment, 
                       save_fig: Optional[bool] = False, 
                       design_name: Optional[Union[str, int]] = 'final'):
    """Plot parity plot comparing the ground true 
    objective function values against predicted model mean
    with predicted confidence interval as error bars 
    Using Experiment object

    Parameters
    ----------
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory
    design_name : Optional[Union[str, int]], optional
        Design name to add to the figure name
        by default 'final'
    """
    
    Y_real_pred, Y_lower_real_pred, Y_upper_real_pred = \
        Exp.validate_training(show_confidence=True)

    parity_with_ci(y1=Exp.Y_real, 
                   y2=Y_real_pred,
                   y2_lower=Y_lower_real_pred,
                   y2_upper=Y_upper_real_pred,
                   save_fig=save_fig,
                   save_path=Exp.exp_path,
                   i_iter = design_name)


#%% Functions for 1 dimensional systems
def acq_func_1d(
    acq_func: AcquisitionFunction, 
    X_test: MatrixLike2d, 
    X_train: Optional[MatrixLike2d] = None, 
    X_new: Optional[MatrixLike2d] = None, 
    X_name: Optional[str] = 'x',
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None, 
    i_iter: Optional[Union[str, int]] = ''):
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
    X_name: Optional[str], optional
        Name of X varibale shown as x-label
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory
    i_iter: Optional[Union[str, int]], optional
        Iteration index to add to the figure name
        by default ''

    .._'botorch.acquisition.AcquisitionFunction': https://botorch.org/api/acquisition.html
    """
    # compute acquicision function values at X_test and X_train
    acq_val_test = eval_acq_func(acq_func, X_test, return_type='np')
    X_test = np.squeeze(tensor_to_np(X_test))
    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(X_test, acq_val_test, 'b-', label = 'Acquisition')
    # Plot training points as black 
    if X_train is not None:
        acq_val_train = eval_acq_func(acq_func, X_train, return_type='np')
        X_train = np.squeeze(tensor_to_np(X_train))
        ax.scatter(X_train, acq_val_train, s = 120, c= 'k', marker = '*', label = 'Initial Data')
    # Plot the new infill points as red stars
    if X_new is not None:
        acq_val_new = eval_acq_func(acq_func, X_new, return_type='np')
        X_new = np.squeeze(tensor_to_np(X_new))
        ax.scatter(X_new, acq_val_new,  s = 120, c ='r', marker = '*', label = 'Infill Data')
    
    ax.ticklabel_format(style = 'sci', axis = 'y', scilimits = (-2,2) )
    ax.set_xlabel(X_name)
    ax.set_ylabel(r'$ \alpha$')    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        fig.savefig(os.path.join(save_path, 'acq_func_i'+ str(i_iter) + '.png'), 
                    bbox_inches="tight")



def acq_func_1d_exp(Exp: Experiment,
    X_test: MatrixLike2d, 
    X_new: Optional[MatrixLike2d] = None,
    X_name: Optional[str] = 'x', 
    save_fig: Optional[bool] = False):
    """Plot 1-dimensional acquision function 
    Using Experiment object

    Parameters
    ----------
    Exp : Experiment
        Experiment object
    X_test : MatrixLike2d
        Test data points for plotting
    X_new : Optional[MatrixLike2d], optional
        The next data point, i.e the infill points,
        by default None
    X_name: Optional[str], optional
        Name of X varibale shown as x-label
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    """
    acq_func_1d(acq_func = Exp.acq_func_current, 
                X_test=X_test, 
                X_train=Exp.X,
                X_new=X_new,
                X_name = X_name, 
                save_fig= save_fig,
                save_path=Exp.exp_path,
                i_iter = Exp.n_points - Exp.n_points_init)

def objective_func_1d(
    model: Model, 
    X_test: MatrixLike2d, 
    Y_test: Optional[MatrixLike2d] = None, 
    X_train: Optional[MatrixLike2d] = None,
    Y_train: Optional[MatrixLike2d] = None, 
    X_new: Optional[MatrixLike2d] = None,
    Y_new: Optional[MatrixLike2d] = None,
    plot_real: Optional[bool] = False,
    Y_mean: Optional[MatrixLike2d] = None,
    Y_std: Optional[MatrixLike2d] = None, 
    X_name: Optional[str] = 'x', 
    Y_name: Optional[str] = 'y',
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None, 
    i_iter: Optional[Union[str, int]] = ''):
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
    X_name: Optional[str], optional
        Name of X varibale shown as x-label
    Y_name: Optional[str], optional
        Name of Y varibale shown as y-label
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory
    i_iter: Optional[str], optional
        Iteration number to add to the figure name
        by default ''
    

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
        
    ax.set_xlabel(X_name)
    ax.set_ylabel(Y_name)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        fig.savefig(os.path.join(save_path, 'objective_func_i'+ str(i_iter) + '.png'), 
                    bbox_inches="tight")


def objective_func_1d_exp(
    Exp: Experiment,
    X_test: MatrixLike2d, 
    Y_test: Optional[MatrixLike2d] = None, 
    X_new: Optional[MatrixLike2d] = None,
    Y_new: Optional[MatrixLike2d] = None,
    plot_real: Optional[bool] = False, 
    X_name: Optional[str] = 'x', 
    Y_name: Optional[str] = 'y', 
    save_fig: Optional[bool] = False):
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
    X_name: Optional[str], optional
        Name of X varibale shown as x-label
    Y_name: Optional[str], optional
        Name of Y varibale shown as y-label
    """
    # if no Y_test input, generate Y_test from objective function
    if (Y_test is None) and (Exp.objective_func is not None):
        Y_test = eval_objective_func(X_test, Exp.X_ranges, Exp.objective_func)

    # if no Y_new input, generate Y_test from objective function
    if (X_new is not None) and (Exp.objective_func is not None) and (Y_new is None):
        Y_new = eval_objective_func(X_new, Exp.X_ranges, Exp.objective_func)

    objective_func_1d(model = Exp.model, 
                     X_test = X_test,
                     Y_test = Y_test,
                     X_train = Exp.X,
                     Y_train = Exp.Y_real, #be sure to use Y_real
                     X_new = X_new,
                     Y_new = Y_new,
                     plot_real = plot_real,
                     Y_mean = Exp.Y_mean,
                     Y_std= Exp.Y_std, 
                     X_name= X_name, 
                     Y_name=Y_name, 
                     save_fig= save_fig,
                     save_path=Exp.exp_path,
                     i_iter = Exp.n_points - Exp.n_points_init)


#%% Functions for 2 dimensional problems
def set_axis_values(
    xi_range: ArrayLike1d, 
    n_sections: Optional[int] = 2, 
    decimals: Optional[int] = 1
) -> ArrayLike1d:
    """Divide xi_range into n_sections

    Parameters
    ----------
    xi_range : ArrayLike1d
        range of x, [left bound, right bound]
    n_sections : Optional[int], optional
        number of sections, by default 2
    decimals : Optional[int], optional
        number of decimal places to keep, by default 1

    Returns
    -------
    axis_values: ArrayLike1d
        axis values with rounding up
        Number of values is n_sections + 1
    """
    lb = xi_range[0]
    rb = xi_range[1] + (xi_range[1]-xi_range[0])/n_sections
    interval = (xi_range[1]-xi_range[0])/n_sections
    axis_values = np.arange(lb, rb, interval)
    axis_values = np.around(axis_values, decimals = decimals)

    return axis_values


def sampling_2d(
    Xs: Union[MatrixLike2d, List[MatrixLike2d]], 
    X_ranges: Optional[MatrixLike2d] = None,
    X_names: Optional[List[str]] = None,
    design_names: Optional[Union[str, List[str]]] = None,
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None, 
    ):
    """Plot sampling plan(s) in 2 dimensional space
    X must be 2 dimensional

    Parameters
    ----------
    Xs : Union[MatrixLike2d, List[MatrixLike2d]]
        The set of sampling plans,
        Can be a list of matrices or one matrix
    X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
    X_name: Optional[str], optional
        Names of X varibale shown as x,y-labels
    design_names : Optional[List[str]], optional
        Names of the designs, by default None
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory
    """
    # if only one set of design is input, convert to list
    if not isinstance(Xs, list):
        Xs = [Xs]
    # set default design names if none
    if design_names is None:
        design_names = ['design' + str(i) for i in range(len(Xs))]
    # set the file name
    # if only one set of design, use that design name
    # else use comparison in the name
    file_name = 'sampling_2d_'
    if not isinstance(design_names, list):
        file_name += design_names
    else:
        file_name += 'comparison'  
    # Set default axis names 
    n_dim = Xs.shape[1]
    if X_names is None:
            X_names = ['x' + str(i+1) for i in range(n_dim)]
    # Set default [0,1] range for a unit scale
    if X_ranges is None:
        X_ranges = [[0,1]] * n_dim

    # set the colors
    colors = colormap(np.linspace(0, 1, len(Xs)))
    # make the plot
    fig,ax = plt.subplots(figsize=(6, 6))
    for Xi, ci, name_i in zip(Xs, colors, design_names):
        ax.scatter(Xi[:,0], Xi[:,1], c = ci , s = 60, label = name_i)
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.axis('square')
    ax.axis([0, 1, 0, 1])
    ax.set_xlabel(X_names[0])
    ax.set_ylabel(X_names[1])
    ax.set_xticks(set_axis_values(X_ranges, 4))
    ax.set_yticks(set_axis_values(X_ranges, 4))
    plt.show()

    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        fig.savefig(os.path.join(save_path, file_name + '.png'), 
                    bbox_inches="tight")

def sampling_2d_exp():
    pass


def add_x_slice_2d(
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


def add_y_slice_2d(
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


def add_z_slice_2d(
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


def sampling_3d(
    Xs: Union[MatrixLike2d, List[MatrixLike2d]], 
    X_ranges: Optional[MatrixLike2d] = None,
    X_names: Optional[List[str]] = None, 
    slice_axis: Optional[str] = None, 
    slice_value: Optional[float] = None, 
    design_names: Optional[Union[str, List[str]]] = None,
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None):
    """Plot sampling plan(s) in 3 dimensional space
    X must be 3 dimensional

    Parameters
    ----------
    Xs : Union[MatrixLike2d, List[MatrixLike2d]]
        The set of sampling plans,
        Can be a list of matrices or one matrix
    X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
    X_name: Optional[List(str)], optional
        Names of X varibale shown as x,y,z-labels
    slice_axis : Optional[str], optional
        axis where a 2d slice is made, by default None
    slice_value : Optional[float], optional
        value on the axis where a 2d slide is made, 
        by default None 
    design_names : Optional[List[str]], optional
        Names of the designs, by default None
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory

    Raises
    ------
    ValueError
        if input axis is defined but the value is not given
    ValueError
        if input axis name is not x, y or z
    """
    
    # if only one set of design is input, convert to list
    if not isinstance(Xs, list):
        Xs = [Xs]
    # set default design names if none
    if design_names is None:
        design_names = ['design' + str(i) for i in range(len(Xs))]
    # set the file name
    # if only one set of design, use that design name
    # else use comparison in the name
    file_name = 'sampling_3d_'
    if not isinstance(design_names, list):
        file_name += design_names # for a single design, include its name
    else:
        file_name += 'comparison' # for multiple designs, use "comparison"
    # Set default axis names 
    n_dim = Xs.shape[1]
    if X_names is None:
            X_names = ['x' + str(i+1) for i in range(n_dim)]
    # Set default [0,1] range for a unit scale
    if X_ranges is None:
        X_ranges = [[0,1]] * n_dim
    
    # set the colors
    colors = colormap(np.linspace(0, 1, len(Xs)))
    # Visualize sampling plan - a 3D scatter plot
    fig  = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')
    for Xi, ci, name_i in zip(Xs, colors, design_names):
        ax.scatter(Xi[:,0], Xi[:,1], Xi[:,2], \
            c=ci, marker='o', s = 50, alpha = 0.6, label = name_i)
    # set axis labels and ticks
    ax.set_xlabel(X_names[0], labelpad= 15)
    ax.set_ylabel(X_names[1],labelpad= 15)
    ax.set_zlabel(X_names[2],labelpad=3)
    ax.set_xticks(set_axis_values(X_ranges[0]))
    ax.set_yticks(set_axis_values(X_ranges[1]))
    ax.set_zticks(set_axis_values(X_ranges[2]))
    ax.view_init(30, 45)
    # Add a 2d slide if required
    if slice_axis is not None:
        if slice_value is None:
            raise ValueError("Input a slice value")
        if slice_axis == 'x': 
            add_x_slice_2d(ax, slice_value, X_ranges[1], X_ranges[2])
            file_name += '_slice_x'
        if slice_axis == 'y': 
            add_y_slice_2d(ax, X_ranges[0], slice_value, X_ranges[2])
            file_name += '_slice_y'
        if slice_axis == 'z': 
            add_z_slice_2d(ax, X_ranges[0], X_ranges[1], slice_value)
            file_name += '_slice_z'
        else: 
            raise ValueError("Input slice_axis is not valid, must be x, y or z")
    plt.show()
    
    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        fig.savefig(os.path.join(save_path, file_name + '.png'), 
                    bbox_inches="tight")


def sampling_3d_exp():
    # we better distinguish initial and infill points by default
    pass

#%%
def response_heatmap(
    Y_real: MatrixLike2d,  
    Y_real_range: ArrayLike1d, 
    Y_name: Optional[str] = '',
    X_ranges: Optional[MatrixLike2d] = None,
    X_names: Optional[List[str]] = None, 
    X_train: Optional[MatrixLike2d] = None, 
    variable_indices: Optional[List[int]] = [0, 1],
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None,
    i_iter: Optional[Union[str, int]] = '', 
    log_flag: Optional[bool] = False):  
    '''
    Takes in the function value X, sampling plan X
    Makes heat map and show the locations of sampling points
    all input are numpy matrices
    '''  

    # Extract two variable indices for plotting
    variable_indices = sorted(variable_indices) 
    x_index = variable_indices[0]
    y_index = variable_indices[1]
    # Set default axis names 
    if X_names is None:
            X_names = ['x' + str(i+1) for i in variable_indices]
    # set the file name
    filename = 'heatmap_'+ Y_name + str(x_index) + str(y_index) + '_i_' + str(i_iter) 
    
    # Set default [0,1] range for a unit scale
    if X_ranges is None:
        X_ranges = [[0,1]] * 2
    # Set default number of sections
    n_tick_sections  = 4
    n_x = Y_real.shape[0]
    n_y = Y_real.shape[1]
    # Visualize response - a 2D heatmap
    fig,ax = plt.subplots(figsize=(6, 6))
    
    if log_flag:
        Yreal_log = np.log10(abs(Y_real))
        im = ax.imshow(Yreal_log, cmap = 'jet', interpolation = 'gaussian', \
             vmin = Y_real_range[0], vmax = Y_real_range[1], origin = 'lower') 
    else:
        im = ax.imshow(Y_real, cmap = 'jet', interpolation = 'gaussian', \
            vmin = Y_real_range[0], vmax = Y_real_range[1], origin = 'lower') 

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax = cax)
    # Viusalize the sampling points as scatter points 
    if X_train is not None:
        ax.scatter(X_train[:,0], X_train[:,1], c = 'white', edgecolors= 'k' )
    # set axis labels and ticks   
    ax.set_xlabel(X_names[x_index])
    ax.set_ylabel(X_names[y_index])
    ax.set_xticks(np.arange(0, n_x, n_tick_sections+1))
    ax.set_xticklabels(set_axis_values(X_ranges[x_index], n_tick_sections))
    ax.set_yticks(np.arange(0, n_y, n_tick_sections+1))
    ax.set_yticklabels(set_axis_values(X_ranges[y_index], n_tick_sections))
    
    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        fig.savefig(os.path.join(save_path, filename + '.png'), 
                    bbox_inches="tight")
    

def response_heatmap_exp():
    pass

def response_heatmap_err_exp():
    '''
    Takes in the function value X, sampling plan X
    Makes heat map of error and show the locations of sampling points
    the error is not normalized (future work)
    ''' 
    pass


def response_surface(
    X1_test: MatrixLike2d,
    X2_test: MatrixLike2d,
    Y_real: MatrixLike2d,  
    Y_real_range: ArrayLike1d,
    Y_real_lower: Optional[MatrixLike2d] = None, 
    Y_real_upper: Optional[MatrixLike2d] = None, 
    X_ranges: Optional[MatrixLike2d] = None,
    X_names: Optional[List[str]] = None, 
    variable_indices: Optional[List[int]] = [0, 1],
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None,
    i_iter: Optional[Union[str, int]] = '', 
    log_flag: Optional[bool] = False):

    '''
    Takes in the function value X, sampling plan X
    Makes heat map and show the locations of sampling points
    all input are numpy matrices
    '''  
    n_tick_sections  = 4
    
    fig  = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(test_x1, test_x2, Y_real, cmap = 'jet', vmin = 0, vmax = 50) 


    if not type(Y_lower) == list:
        ax.plot_surface(test_x1, test_x2, Y_lower, cmap = 'Blues', alpha = 0.7, vmin = 0, vmax = 50) 
    if not type(Y_upper) == list:
        ax.plot_surface(test_x1, test_x2, Y_upper, cmap = 'Reds', alpha = 0.7, vmin = 0, vmax = 50, ) 

    
    ax.set_xlabel(X_name_with_unit[plotting_axis[0]], labelpad=15)
    ax.set_ylabel(X_name_with_unit[plotting_axis[1]], labelpad=15)
    ax.set_zlabel(Y_name_with_unit, labelpad=10)
    
    ax.set_xticks(np.linspace(0, 1, n_tick_sections+1))
    ax.set_xticklabels(set_range(Xranges[plotting_axis[0]], n_tick_sections))
    
    ax.set_yticks(np.linspace(0, 1, n_tick_sections+1))
    ax.set_yticklabels(set_range(Xranges[plotting_axis[1]], n_tick_sections))
    
    ax.view_init(30, 45)

    return fig   

def _plot_for_testing(test_Y_real, test_Y_obj, Xreal_range, savepath, iteration_no = 0, 
                      make_heatmap = True, make_error = True, make_surface = True):
    '''
    Generate three types of plots for testing
    '''
        
    if make_heatmap:
        fig = heatmap(test_Y_real, Xreal_range)
        fig.savefig(os.path.join(savepath, 'heatmap_' + str(iteration_no) + '.png'))
        
    if make_error:
        fig = heatmap_error(test_Y_real- test_Y_obj, Xreal_range)
        fig.savefig(os.path.join(savepath, 'heatmap_err_' + str(iteration_no) + '.png'))
        
    if make_surface:
        fig = surfaceplots(test_Y_real, Xreal_range)
        fig.savefig(os.path.join(savepath, 'surface_' + str(iteration_no) + '.png'))