"""
Creates 1-dimensional, 2-dimensional and 3-dimensional visualizations 
The plots are rendered using matplotlib_ as a backend

.. _matplotlib: https://matplotlib.org/stable/index.html
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
from nextorch.utils import  ArrayLike1d, MatrixLike2d, create_full_X_test_1d, create_full_X_test_2d
from nextorch.utils import tensor_to_np, standardize_X, transform_Y_mesh_2d, unitscale_xv
from nextorch.bo import eval_acq_func, eval_objective_func, \
    model_predict, model_predict_real, Experiment, EHVIMOOExperiment, WeightedMOOExperiment


# # Set matplotlib default values
# font = {'size'   : 20}

# matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.linewidth'] = 1.5
# matplotlib.rcParams['xtick.major.size'] = 8
# matplotlib.rcParams['xtick.major.width'] = 2
# matplotlib.rcParams['ytick.major.size'] = 8
# matplotlib.rcParams['ytick.major.width'] = 2
# matplotlib.rcParams["figure.dpi"] = 100
# matplotlib.rcParams['savefig.dpi'] = 600

# Set global plotting variables
colormap = cm.jet
figformat = 'png'
backgroundtransparency = False

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
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")

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
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, 'parity_'+ str(i_iter) + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)


def parity_exp(
    Exp: Experiment, 
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
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    
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
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, 'parity_w_ci_'+ str(i_iter) + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)


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

#%% Discovery plots
def opt_per_trial(
    Ys: Union[list, ArrayLike1d],
    maximize: Optional[bool] = True,
    Y_real_range: Optional[ArrayLike1d] = None, 
    Y_name: Optional[str] = None,
    log_flag: Optional[bool] = False,
    design_names: Optional[Union[str, List[str]]] = None,
    save_fig: Optional[bool] = False, 
    save_path: Optional[str] = None,
):
    """Discovery plot  
    show the optimum value performance versus the trial number
    i.e. the index of training data

    Parameters
    ----------
    Ys : Union[list, ArrayLike1d]
        Response of each design in a real scale
    maximize : Optional[bool], optional
        by default True, maximize the objective function
        Otherwise False, minimize the objective function
    Y_real_range : ArrayLike1d
        Ranges of the response, [lb, rb]
        to show on the plot, by default None
    Y_name : Optional[str], optional
        Name of Y variable, by default None
    log_flag : Optional[bool], optional
        flag to plot in a log scale, by default False
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
    if not isinstance(Ys, list):
        Ys = [Ys]
    # set default design names if none
    if design_names is None:
        design_names = ['design' + str(i) for i in range(len(Ys))]
    if not isinstance(design_names, list):
        design_names = [design_names]
    # Set Y_name in file name
    if Y_name is None:
        Y_name = ''
    # set the file name
    # if only one set of design, use that design name
    # else use comparison in the name
    file_name = 'opt_per_trial_' + Y_name + '_'
    if not isinstance(design_names, list):
        file_name += design_names
    else:
        file_name += 'comparison'  

    # set the colors
    colors = colormap(np.linspace(0, 1, len(Ys)))
    # make the plot
    fig,ax = plt.subplots(figsize=(8, 6))

    for yi, ci, name_i in zip(Ys, colors, design_names):
        if log_flag:
            yi = np.log10(abs(yi))
        if maximize:
            opt_yi = np.maximum.accumulate(yi)
        else:
            opt_yi = np.minimum.accumulate(yi)
        ax.plot(np.arange(len(yi)), opt_yi,  '-o', color = ci, \
            label = name_i, markersize=5, linewidth = 3, markerfacecolor="None")
    if Y_real_range is not None:
        ax.set_ylim(Y_real_range)
    ax.set_xlabel('Trial Index')
    ax.set_ylabel('Best Observed '+ Y_name)
    ax.legend()

    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, file_name + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)


def opt_per_trial_exp(
    Exp: Experiment, 
    Y_real_range: Optional[ArrayLike1d] = None, 
    log_flag: Optional[bool] = False,
    save_fig: Optional[bool] = False):
    """Discovery plot  
    show the optimum value performance versus the trial number
    i.e. the index of training data
    Using the experiment object

    Parameters
    ----------
    Exp : Experiment
        Experiment object
    Y_real_range : ArrayLike1d
        Ranges of the response, [lb, rb]
        to show on the plot, by default None
    log_flag : Optional[bool], optional
        flag to plot in a log scale, by default False
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    """
    opt_per_trial(Ys=Exp.Y_real,
                  maximize=Exp.maximize,
                  Y_name=Exp.Y_names[0],
                  Y_real_range=Y_real_range,
                  log_flag=log_flag,
                  save_fig=save_fig,
                  save_path=Exp.exp_path,
                  design_names='final')

#%% Functions for 1 dimensional systems
def acq_func_1d(
    acq_func: AcquisitionFunction, 
    X_test: MatrixLike2d, 
    n_dim: Optional[int] = 1,
    X_ranges: Optional[MatrixLike2d] = None,
    x_index: Optional[int] = 0,
    X_train: Optional[MatrixLike2d] = None, 
    X_new: Optional[MatrixLike2d] = None, 
    X_names: Optional[str] = None,
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None, 
    i_iter: Optional[Union[str, int]] = ''):
    """Plot 1-dimensional acquision function 
    at the given dimension defined by x_index

    Parameters
    ----------
    acq_func : 'botorch.acquisition.AcquisitionFunction'_
        the acquision function object
    X_test : MatrixLike2d
        Test data points for plotting
    n_dim : Optional[int], optional
        Dimensional of X, i.e., number of columns 
        by default 1
    X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
    x_index : Optional[int], optional
        index of the x variable, by default 0
    X_train : Optional[MatrixLike2d], optional
        Training data points, by default None
    X_new : Optional[MatrixLike2d], optional
        The next data point, i.e the infill points,
        by default None
    X_names : Optional[str], optional
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
    # Set default axis names 
    if X_names is None:
        if (n_dim == 1): X_name = 'x'
        else: X_name = 'x' + str(x_index + 1) 
    else: 
        X_name = X_names[x_index]

    # Set default [0,1] range for a unit scale
    if X_ranges is None:
        X_ranges = [[0,1]] * n_dim
    # Set default number of sections
    n_tick_sections  = 5

    # compute acquicision function values at X_test and X_train
    acq_val_test = eval_acq_func(acq_func, X_test, return_type='np')
    # Select the given dimension
    x_test_1d = np.squeeze(tensor_to_np(X_test)[:, x_index])

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_test_1d, acq_val_test, 'b-', label = 'Acquisition')

    # Plot training points as black 
    # Only for 1-dimensional system
    if (n_dim == 1) and (X_train is not None):
        acq_val_train = eval_acq_func(acq_func, X_train, return_type='np')
        x_train = np.squeeze(tensor_to_np(X_train)[:, x_index])
        ax.scatter(x_train, acq_val_train, s = 120, c= 'k', marker = '*', label = 'Initial Data')
    # Plot the new infill points as red stars
    # Only for 1-dimensional system
    if (n_dim == 1) and (X_new is not None):
        acq_val_new = eval_acq_func(acq_func, X_new, return_type='np')
        x_new = np.squeeze(tensor_to_np(X_new)[:, x_index])
        ax.scatter(x_new, acq_val_new,  s = 120, c ='r', marker = '*', label = 'Infill Data')
    
    ax.ticklabel_format(style = 'sci', axis = 'y' )#, scilimits = (-2,2) )
    ax.set_xlabel(X_name)
    xlim_plot = list(ax.set_xlim((0,1))) 
    
    ax.set_xticks(set_axis_values(xlim_plot, n_tick_sections))
    ax.set_xticklabels(set_axis_values(X_ranges[x_index], n_tick_sections))
    ax.set_ylabel(r'$ \alpha$')    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, 'acq_func_i'+ str(i_iter) + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)



def acq_func_1d_exp(Exp: Experiment,
    X_new: Optional[MatrixLike2d] = None,
    x_index: Optional[int] = 0,
    fixed_values: Optional[Union[ArrayLike1d, float]] = None,
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = None,
    baseline: Optional[str] = 'left',
    mesh_size: Optional[int] = 41,
    save_fig: Optional[bool] = False):
    """Plot 1-dimensional acquision function 
    at the given dimension defined by x_index
    Using Experiment object

    Parameters
    ----------
    Exp : Experiment
        Experiment object
    X_new : Optional[MatrixLike2d], optional
        The next data point, i.e the infill points,
        by default None
    x_index : Optional[int], optional
        index of two x variables, by default 0
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
    """
    # Create 1d mesh test points 
    X_test_1d = create_full_X_test_1d(X_ranges=Exp.X_ranges, 
                                      x_index=x_index,
                                      fixed_values=fixed_values,
                                      fixed_values_real=fixed_values_real,
                                      baseline=baseline,
                                      mesh_size=mesh_size) 

    acq_func_1d(acq_func = Exp.acq_func_current, 
                X_test=X_test_1d, 
                n_dim=Exp.n_dim,
                X_ranges=Exp.X_ranges,
                x_index=x_index,
                X_train=Exp.X,
                X_new=X_new,
                X_names=Exp.X_names, 
                save_fig= save_fig,
                save_path=Exp.exp_path,
                i_iter = Exp.n_points - Exp.n_points_init)


def response_1d(
    model: Model, 
    X_test: MatrixLike2d, 
    n_dim: Optional[int] = 1,
    X_ranges: Optional[MatrixLike2d] = None,
    x_index: Optional[int] = 0,
    Y_test: Optional[MatrixLike2d] = None, 
    X_train: Optional[MatrixLike2d] = None,
    Y_train: Optional[MatrixLike2d] = None, 
    X_new: Optional[MatrixLike2d] = None,
    Y_new: Optional[MatrixLike2d] = None,
    negate_Y: Optional[bool] = False,
    plot_real: Optional[bool] = False,
    Y_mean: Optional[MatrixLike2d] = None,
    Y_std: Optional[MatrixLike2d] = None, 
    X_names: Optional[str] = None, 
    Y_name: Optional[str] = None,
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None, 
    i_iter: Optional[Union[str, int]] = ''):
    """Plot response values
    at the given dimension defined by x_index
    Input X variables are in a unit scale and
    Input Y variables are in a real scale

    Parameters
    ----------
    model : 'botorch.models.model.Model'_
        A GP model
    X_test : MatrixLike2d
        Test data points for plotting
    n_dim : Optional[int], optional
        Dimensional of X, i.e., number of columns 
        by default 1
    X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
    x_index : Optional[int], optional
        index of the x variable, by default 0
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
    X_names: Optional[str], optional
        Name of X varibales shown as x-label
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
    
    # Set default axis names 
    if X_names is None:
        if (n_dim == 1): X_name = 'x'
        else: X_name = 'x' + str(x_index + 1) 
    else: 
        X_name = X_names[x_index]

    # Set default axis names 
    if Y_name is None:
         Y_name = 'y'
    # Set default [0,1] range for a unit scale
    if X_ranges is None:
        X_ranges = [[0,1]] * n_dim
    # Set default number of sections
    n_tick_sections  = 5

    if plot_real: # Y in a real scale
        Y_test_pred, Y_test_lower_pred, Y_test_upper_pred = model_predict_real(model=model, 
                                                                                X_test=X_test, 
                                                                                Y_mean=Y_mean, 
                                                                                Y_std=Y_std, 
                                                                                return_type= 'np', 
                                                                                negate_Y=negate_Y)
    else: # Y in a standardized scale
        Y_test = standardize_X(Y_test, Y_mean, Y_std, return_type= 'np') #standardize Y_test
        Y_train = standardize_X(Y_train, Y_mean, Y_std, return_type= 'np') 
        Y_new = standardize_X(Y_new, Y_mean, Y_std, return_type= 'np') 

        Y_test_pred, Y_test_lower_pred, Y_test_upper_pred = model_predict(model=model, 
                                                                          X_test=X_test, 
                                                                          return_type= 'np',
                                                                          negate_Y=negate_Y)
    # Select the given dimension
    x_test_1d = tensor_to_np(X_test)[:, x_index]
    # reduce the dimension to 1d arrays
    y_test_pred = Y_test_pred
    y_test_lower_pred = Y_test_lower_pred
    y_test_upper_pred = Y_test_upper_pred

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot model predicted posterior means as blue line
    ax.plot(x_test_1d, y_test_pred, 'b', label = 'Posterior Mean')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(x_test_1d, y_test_lower_pred, y_test_upper_pred, alpha=0.5, label = 'Confidence')

    # Plot the groud truth Y_test if provided
    # Only for 1-dimensional system
    if (n_dim == 1) and (Y_test is not None):
        y_test = np.squeeze(tensor_to_np(Y_test))
        ax.plot(x_test_1d, y_test, 'k--', label = 'Objective f(x)')

    # Plot training points as black stars
    # Only for 1-dimensional system
    if (n_dim == 1) and (X_train is not None):
        x_train = np.squeeze(tensor_to_np(X_train)[:, x_index])
        y_train = np.squeeze(tensor_to_np(Y_train))
        ax.scatter(x_train, y_train, s =120, c= 'k', marker = '*', label = 'Initial Data')

    # Plot the new infill points as red stars
    # Only for 1-dimensional system
    if (n_dim == 1) and (X_new is not None):    
        x_new = np.squeeze(tensor_to_np(X_new)[:, x_index])
        y_new = np.squeeze(tensor_to_np(Y_new))
        ax.scatter(x_new, y_new, s = 120, c = 'r', marker = '*', label = 'Infill Data')
        
    ax.set_xlabel(X_name)
    xlim_plot = list(ax.set_xlim(0, 1))
    ax.set_xticks(set_axis_values(xlim_plot, n_tick_sections))
    ax.set_xticklabels(set_axis_values(X_ranges[x_index], n_tick_sections))
    ax.set_ylabel(Y_name)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, 'objective_func_i'+ str(i_iter) + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)


def response_1d_exp(
    Exp: Experiment,
    X_new: Optional[MatrixLike2d] = None,
    Y_new: Optional[MatrixLike2d] = None,
    x_index: Optional[int] = 0,
    y_index: Optional[int] = 0,
    fixed_values: Optional[Union[ArrayLike1d, float]] = None,
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = None,
    baseline: Optional[str] = 'left',
    mesh_size: Optional[int] = 41,
    plot_real: Optional[bool] = False, 
    save_fig: Optional[bool] = False):
    """Plot reponse valus
    at the given dimension defined by x_index
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
    x_index : Optional[int], optional
        index of two x variables, by default 0
    y_index : Optional[int], optional
        index of the y variables, by default 0
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
    plot_real : Optional[bool], optional
        if true plot in the real scale for Y, 
        by default False
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    """
    # Create 1d mesh test points 
    X_test = create_full_X_test_1d(X_ranges=Exp.X_ranges, 
                                    x_index=x_index,
                                    fixed_values=fixed_values,
                                    fixed_values_real=fixed_values_real,
                                    baseline=baseline,
                                    mesh_size=mesh_size) 

    # if no Y_test input, generate Y_test from objective function
    Y_test = None
    if Exp.objective_func is not None:
        Y_test = eval_objective_func(X_test, Exp.X_ranges, Exp.objective_func)

    # if no Y_new input, generate Y_test from objective function
    if (Exp.objective_func is not None) and (Y_new is None):
        Y_new = eval_objective_func(X_new, Exp.X_ranges, Exp.objective_func)

    response_1d(model = Exp.model, 
                X_test = X_test,
                n_dim=Exp.n_dim,
                x_index=x_index,
                X_ranges=Exp.X_ranges,
                Y_test = Y_test,
                X_train = Exp.X,
                Y_train = Exp.Y_real, #be sure to use Y_real
                X_new = X_new,
                Y_new = Y_new,
                negate_Y = Exp.negate_Y,
                plot_real = plot_real,
                Y_mean = Exp.Y_mean,
                Y_std= Exp.Y_std, 
                X_names=Exp.X_names, 
                Y_name=Exp.Y_names[y_index], 
                save_fig= save_fig,
                save_path=Exp.exp_path,
                i_iter = Exp.n_points - Exp.n_points_init)


#%% Functions for 2 dimensional systems on sampling
def set_axis_values(
    xi_range: ArrayLike1d, 
    n_sections: Optional[int] = 2, 
    decimals: Optional[int] = 2
) -> ArrayLike1d:
    """Divide xi_range into n_sections

    Parameters
    ----------
    xi_range : ArrayLike1d
        range of x, [left bound, right bound]
    n_sections : Optional[int], optional
        number of sections, by default 2
    decimals : Optional[int], optional
        number of decimal places to keep, by default 2

    Returns
    -------
    axis_values: ArrayLike1d
        axis values with rounding up
        Number of values is n_sections + 1
    """
    lb = xi_range[0]
    rb = xi_range[1]
    
    axis_values = np.linspace(lb, rb, n_sections+1, endpoint = True)
    axis_values = np.around(axis_values, decimals = decimals)

    return axis_values



def sampling_2d(
    Xs: Union[MatrixLike2d, List[MatrixLike2d]], 
    X_ranges: Optional[MatrixLike2d] = None,
    x_indices: Optional[List[int]] = [0, 1],
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
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
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
    
    # Extract two variable indices for plotting
    x_indices = sorted(x_indices) 
    index_0 = x_indices[0]
    index_1 = x_indices[1]

    # Set default axis names 
    n_dim = Xs[0].shape[1]
    if X_names is None:
        X_names = ['x' + str(xi+1) for xi in x_indices]

    # Set default [0,1] range for a unit scale
    if X_ranges is None:
        X_ranges = [[0,1]] * n_dim
    # Set default number of sections
    n_tick_sections  = 5

    # set the colors
    colors = colormap(np.linspace(0, 1, len(Xs)))
    # make the plot
    fig,ax = plt.subplots(figsize=(6, 6))
    for Xi, ci, name_i in zip(Xs, colors, design_names):
        ax.scatter(Xi[:, index_0], Xi[:, index_1], color = ci , s = 60, label = name_i,  alpha = 0.6)
    
    # Get axes limits
    xlim_plot = list(ax.set_xlim(0, 1))
    ylim_plot = list(ax.set_ylim(0, 1))
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.axis('square')
    #ax.axis([0, 1, 0, 1])
    ax.set_xlabel(X_names[index_0])
    ax.set_ylabel(X_names[index_1])
    ax.set_xticks(set_axis_values(xlim_plot, n_tick_sections))
    ax.set_xticklabels(set_axis_values(X_ranges[index_0], n_tick_sections))
    ax.set_yticks(set_axis_values(ylim_plot, n_tick_sections))
    ax.set_yticklabels(set_axis_values(X_ranges[index_1], n_tick_sections))
    plt.show()

    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, file_name + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)

def sampling_2d_exp(
    Exp: Experiment,
    x_indices: Optional[List[int]] = [0, 1],
    design_names: Optional[Union[str, List[str]]] = None,
    save_fig: Optional[bool] = False):
    """Plot sampling plan(s) in 2 dimensional space
    X must be 2 dimensional
    Using the experiment object

    Parameters
    ----------
    Exp : Experiment
        Experiment object
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    design_names : Optional[List[str]], optional
        Names of the designs, by default None
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    """
    # Set the initial X set as the first design
    X_init = tensor_to_np(Exp.X_init)
    Xs = [X_init]
    # Set the default design name
    if design_names is None:
        design_names = ['Initial']
    # If there are infill points
    if Exp.n_points > Exp.n_points_init:
        X_infill = Exp.X[Exp.n_points_init:,:]
        X_infill = tensor_to_np(X_infill)
        Xs.append(X_infill)
        design_names.append('Infill')
    
    sampling_2d(Xs = Xs, 
                X_ranges=Exp.X_ranges,
                x_indices=x_indices,
                X_names=Exp.X_names,
                design_names=design_names,
                save_fig=save_fig,
                save_path=Exp.exp_path)


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
    X, Z = np.meshgrid(np.linspace(xrange[0], xrange[1], mesh_size), np.linspace(zrange[0], zrange[1], mesh_size), indexing = 'ij')
    Y = yvalue * np.ones((mesh_size, mesh_size))
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

    #return ax


def sampling_3d(
    Xs: Union[MatrixLike2d, List[MatrixLike2d]], 
    X_ranges: Optional[MatrixLike2d] = None,
    x_indices: Optional[List[int]] = [0, 1, 2],
    X_names: Optional[List[str]] = None, 
    slice_axis: Optional[Union[str, int]] = None, 
    slice_value: Optional[float] = None, 
    slice_value_real: Optional[float] = None, 
    design_names: Optional[Union[str, List[str]]] = None,
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None):
    """Plot sampling plan(s) in 3 dimensional space
    X must be 3 dimensional

    Parameters
    ----------
    Xs : Union[MatrixLike2d, List[MatrixLike2d]]
        The set of sampling plans in a unit scale,
        Can be a list of matrices or one matrix
    X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
    x_indices : Optional[List[int]], optional
        indices of three x variables, by default [0, 1, 2]
    X_name: Optional[List(str)], optional
        Names of X varibale shown as x,y,z-labels
    slice_axis : Optional[Union[str, int]], optional
        axis where a 2d slice is made, by default None
    slice_value : Optional[float], optional
        value on the axis where a 2d slide is made, 
        in a unit scale, by default None 
    slice_value_real : Optional[float], optional
        value on the axis where a 2d slide is made, 
        in a real scale, by default None 
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
        if input axis name is not x, y or z, or 0, 1, 2
    """
    
    # if only one set of design is input, convert to list
    if not isinstance(Xs, list):
        Xs = [Xs]
    # set default design names if none
    if design_names is None:
        design_names = ['design' + str(i) for i in range(len(Xs))]
    if not isinstance(design_names, list):
        design_names = [design_names]
    # set the file name
    # if only one set of design, use that design name
    # else use comparison in the name
    file_name = 'sampling_3d_'
    if not isinstance(design_names, list):
        file_name += design_names # for a single design, include its name
    else:
        file_name += 'comparison' # for multiple designs, use "comparison"

    # Extract two variable indices for plotting
    x_indices = sorted(x_indices) 
    index_0 = x_indices[0]
    index_1 = x_indices[1]
    index_2 = x_indices[2]

    # Set default axis names 
    n_dim = Xs[0].shape[1]
    if X_names is None:
        X_names = ['x' + str(i+1) for i in range(n_dim)]
    # Set default [0,1] range for a unit scale
    if X_ranges is None:
        X_ranges = [[0,1]] * n_dim
    # Set default number of sections
    n_tick_sections  = 5
    
    # set the colors
    colors = colormap(np.linspace(0, 1, len(Xs)))
    # Visualize sampling plan - a 3D scatter plot
    fig  = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')
    for Xi, ci, name_i in zip(Xs, colors, design_names):
        ax.scatter(Xi[:, index_0], Xi[:, index_1], Xi[:, index_2], \
            color=ci, marker='o', s = 60, alpha = 0.6, label = name_i)
    # Get axes limits
    xlim_plot = list(ax.set_xlim(0, 1))
    ylim_plot = list(ax.set_ylim(0, 1))
    zlim_plot = list(ax.set_zlim(0, 1))
    
    # Add a 2d slide if required
    if slice_axis is not None:
        if (slice_value is None) and (slice_value_real is None):
            raise ValueError("Input a slice value")
        if (slice_axis == 'x') or (slice_axis == 0): 
            if slice_value is None: # convert the slice value into a unit scale
                slice_value = unitscale_xv(slice_value_real, X_ranges[0])
            add_x_slice_2d(ax, slice_value, [0, 1], [0, 1])
            file_name += '_slice_x'
        elif (slice_axis == 'y') or (slice_axis == 1): 
            if slice_value is None:
                slice_value = unitscale_xv(slice_value_real, X_ranges[1])
            add_y_slice_2d(ax, slice_value, [0, 1], [0, 1])
            file_name += '_slice_y'
        elif (slice_axis == 'z') or (slice_axis == 2): 
            if slice_value is None:
                slice_value = unitscale_xv(slice_value_real, X_ranges[2])
            add_z_slice_2d(ax, slice_value, [0, 1], [0, 1])
            file_name += '_slice_z'
        else: 
            raise ValueError("Input slice_axis is not valid, must be x, y or z, or 0, 1, 2")
    
    # set axis labels and ticks
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.set_xlabel(X_names[index_0], labelpad= 15)
    ax.set_ylabel(X_names[index_1],labelpad= 15)
    ax.set_zlabel(X_names[index_2],labelpad=3)
    ax.set_xticks(set_axis_values(xlim_plot, n_tick_sections))
    ax.set_xticklabels(set_axis_values(X_ranges[index_0], n_tick_sections))
    ax.set_yticks(set_axis_values(ylim_plot, n_tick_sections))
    ax.set_yticklabels(set_axis_values(X_ranges[index_1], n_tick_sections))
    ax.set_zticks(set_axis_values(zlim_plot, n_tick_sections))
    ax.set_zticklabels(set_axis_values(X_ranges[index_2], n_tick_sections))
    ax.view_init(30, 45)
    plt.show()
    
    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, file_name + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)


def sampling_3d_exp(
    Exp: Experiment,
    x_indices: Optional[List[int]] = [0, 1, 2],
    slice_axis: Optional[str] = None, 
    slice_value: Optional[float] = None,
    slice_value_real: Optional[float] = None, 
    design_names: Optional[Union[str, List[str]]] = None,
    save_fig: Optional[bool] = False):
    """Plot sampling plan(s) in 3 dimensional space
    X must be 3 dimensional
    Using the experiment object

    Parameters
    ----------
    Exp : Experiment
        Experiment object
    x_indices : Optional[List[int]], optional
        indices of three x variables, by default [0, 1, 2]
    slice_axis : Optional[str], optional
        axis where a 2d slice is made, by default None
    slice_value : Optional[float], optional
        value on the axis where a 2d slide is made, 
        by default None 
    slice_value_real : Optional[float], optional
        value on the axis where a 2d slide is made, 
        in a real scale, by default None 
    design_names : Optional[List[str]], optional
        Names of the designs, by default None
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    """
    # Set the initial X set as the first design
    X_init = tensor_to_np(Exp.X_init)
    Xs = [X_init]
    # Set the default design name
    if design_names is None:
        design_names = ['Initial']
    # If there are infill points
    if Exp.n_points > Exp.n_points_init:
        X_infill = Exp.X[Exp.n_points_init:,:]
        X_infill = tensor_to_np(X_infill)
        Xs.append(X_infill)
        design_names.append('Infill')
    
    ax= sampling_3d(Xs=Xs, 
                    X_ranges=Exp.X_ranges,
                    x_indices=x_indices,
                    X_names=Exp.X_names,
                    slice_axis=slice_axis,
                    slice_value=slice_value,
                    slice_value_real=slice_value_real,
                    design_names=design_names,
                    save_fig=save_fig,
                    save_path=Exp.exp_path)


#%% Functions for 2 dimensional systems on response heatmaps
def response_heatmap(
    Y_real: MatrixLike2d,  
    Y_real_range: Optional[ArrayLike1d] = None, 
    Y_name: Optional[str] = None,
    log_flag: Optional[bool] = False,
    n_dim: Optional[int] = 2,
    x_indices: Optional[List[int]] = [0, 1],
    X_ranges: Optional[MatrixLike2d] = None,
    X_names: Optional[List[str]] = None, 
    X_train: Optional[MatrixLike2d] = None, 
    X_new: Optional[MatrixLike2d] = None,
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None,
    i_iter: Optional[Union[str, int]] = ''):  
    """Show a heat map for the response in a real scale

    Parameters
    ----------
    Y_real : MatrixLike2d
        Response in a real scale
    Y_real_range : ArrayLike1d
        Ranges of the response, [lb, rb]
        to show on the plot, by default None
    Y_name : Optional[str], optional
        Names of Y variable, by default None
    log_flag : Optional[bool], optional
        flag to plot in a log scale, by default False
    n_dim : Optional[int], optional
        Dimensional of X, i.e., number of columns 
        by default 2
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
    X_name: Optional[List(str)], optional
        Names of X varibale shown as x,y,z-labels
        by default None
    X_train : Optional[MatrixLike2d], optional
        Data points used in training, by default None
    X_new : Optional[MatrixLike2d], optional
        The next data point, i.e the infill points,
        by default None
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory
    i_iter: Optional[str], optional
        Iteration number to add to the figure name
        by default '''
    """
    '''
    Takes in the function value X, sampling plan X
    Makes heat map and show the locations of sampling points
    all input are numpy matrices
    '''  
    # Preprocess Y_real
    Y_real = tensor_to_np(Y_real)

    # Set default Y_real_range
    if Y_real_range is None:
        Y_real_range = [np.min(Y_real), np.max(Y_real)]
    if log_flag:
        Y_real = np.log10(abs(Y_real))

    # Extract two variable indices for plotting
    x_indices = sorted(x_indices) 
    index_0 = x_indices[0]
    index_1 = x_indices[1]
    
    # Set default axis names 
    if X_names is None:
        X_names = ['x' + str(xi + 1) for xi in range(n_dim)]

    # Set Y_name in file name
    if Y_name is None: 
        Y_name = 'y'

    # set the file name
    filename = 'heatmap_'+ Y_name + '_' + str(index_0) +\
         str(index_1) + '_i_' + str(i_iter) 
    

    # Set default [0,1] range for a unit scale
    if X_ranges is None:
        X_ranges = [[0,1]] * n_dim
    # Set default number of sections
    n_tick_sections  = 5

    # Visualize response - a 2D heatmap
    fig,ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(Y_real, cmap = 'jet', interpolation = 'gaussian', \
            vmin = Y_real_range[0], vmax = Y_real_range[1], origin = 'lower', \
            extent = (0,1,0,1))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax = cax)

    # Viusalize the sampling points as hollow black scatter points 
    # Only for 2-dimensional system
    if (n_dim == 2) and (X_train is not None) :
        ax.scatter(X_train[:,index_0], X_train[:,index_1], s = 60, \
            c = 'white', edgecolors= 'k', alpha = 0.6)

    # Viusalize the infill points as hollow red scatter points 
    # Only for 2-dimensional system
    if (n_dim == 2) and (X_new is not None) :
        ax.scatter(X_new[:,index_0], X_new[:,index_1], s = 60, \
            c = 'white', edgecolors= 'r', alpha = 0.6)

    #  Obtain axes limits
    xlim_plot = list(ax.set_xlim((0,1)))
    ylim_plot = list(ax.set_ylim((0,1)))
    # set axis labels and ticks   
    ax.set_xlabel(X_names[index_0])
    ax.set_ylabel(X_names[index_1])
    ax.set_xticks(set_axis_values(xlim_plot, n_tick_sections))
    ax.set_xticklabels(set_axis_values(X_ranges[index_0], n_tick_sections))
    ax.set_yticks(set_axis_values(ylim_plot, n_tick_sections))
    ax.set_yticklabels(set_axis_values(X_ranges[index_1], n_tick_sections))
     
    plt.show()

    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, filename + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)
    

def response_heatmap_exp(
    Exp: Experiment,
    X_new: Optional[MatrixLike2d] = None,
    Y_real_range: Optional[ArrayLike1d] = None, 
    log_flag: Optional[bool] = False,
    x_indices: Optional[List[int]] = [0, 1],
    fixed_values: Optional[Union[ArrayLike1d, float]] = [],
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = [],
    baseline: Optional[str] = 'left',
    mesh_size: Optional[int] = 41,
    show_samples: Optional[bool] = True,
    save_fig: Optional[bool] = False):
    """Show a heat map for the response in a real scale
    Using the experiment object
    Parameters
    ----------
    Exp : Experiment
        Experiment object
    X_new : Optional[MatrixLike2d], optional
        The next data point, i.e the infill points,
        by default None
    Y_real_range : Optional[ArrayLike1d], optional
        Ranges of the response, [lb, rb]
        to show on the plot, by default None
    log_flag : Optional[bool], optional
        flag to plot in a log scale
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    fixed_values : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a unit scale, by default []
    fixed_values_real : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a real scale, by default []
    baseline : Optional[str], optional
        the choice of baseline, must be left, right or center
    mesh_size : Optional[int], optional
        mesh size, by default 41
    show_samples: Optional[bool], optional
        if true show the sample points   
        by default True
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False

    """    
    # Create 2D mesh test points  
    X_test, _, _ = create_full_X_test_2d(X_ranges=Exp.X_ranges, 
                                         x_indices=x_indices,
                                         fixed_values=fixed_values,
                                         fixed_values_real=fixed_values_real,
                                         baseline=baseline,
                                         mesh_size=mesh_size) 
                                     
    # Make prediction using the GP model
    Y_test = Exp.predict_real(X_test)
    Y_test_2d = transform_Y_mesh_2d(Y_test, mesh_size=mesh_size)
    # select the sample points
    X_train = None
    if show_samples: X_train = Exp.X

    response_heatmap(Y_real=Y_test_2d,
                     Y_real_range = Y_real_range,
                     Y_name=Exp.Y_names[0],
                     log_flag=log_flag,
                     n_dim=Exp.n_dim,
                     x_indices=x_indices,
                     X_ranges=Exp.X_ranges,
                     X_names=Exp.X_names,
                     X_train=X_train,
                     X_new=X_new,
                     save_fig=save_fig,
                     save_path=Exp.exp_path,
                     i_iter=Exp.n_points - Exp.n_points_init)
    

def objective_heatmap_exp(
    Exp: Experiment,
    X_new: Optional[MatrixLike2d] = None,
    Y_real_range: Optional[ArrayLike1d] = None, 
    log_flag: Optional[bool] = False,
    x_indices: Optional[List[int]] = [0, 1],
    fixed_values: Optional[Union[ArrayLike1d, float]] = [],
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = [],
    baseline: Optional[str] = 'left',
    mesh_size: Optional[int] = 41,
    show_samples: Optional[bool] = True,
    save_fig: Optional[bool] = False):
    """Show a heat map for objective function in a real scale
    Using the experiment object
    Parameters
    ----------
    Exp : Experiment
        Experiment object
    X_new : Optional[MatrixLike2d], optional
        The next data point, i.e the infill points,
        by default None
    Y_real_range : Optional[ArrayLike1d], optional
        Ranges of the response, [lb, rb]
        to show on the plot, by default None
    log_flag : Optional[bool], optional
        flag to plot in a log scale
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    fixed_values : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a unit scale, by default []
    fixed_values_real : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a real scale, by default []
    baseline : Optional[str], optional
        the choice of baseline, must be left, right or center
    mesh_size : Optional[int], optional
        mesh size, by default 41
    show_samples: Optional[bool], optional
        if true show the sample points   
        by default True
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    """
    # Create 2D mesh test points  
    X_test, _, _ = create_full_X_test_2d(X_ranges=Exp.X_ranges, 
                                         x_indices=x_indices,
                                         fixed_values=fixed_values,
                                         fixed_values_real=fixed_values_real,
                                         baseline=baseline,
                                         mesh_size=mesh_size) 
    # Calculate objective function value 
    Y_obj_test = eval_objective_func(X_test, Exp.X_ranges, Exp.objective_func)
    Y_obj_test_2d = transform_Y_mesh_2d(Y_obj_test, mesh_size=mesh_size)

    # select the sample points
    X_train = None
    if show_samples: X_train = Exp.X
    
    response_heatmap(Y_real=Y_obj_test_2d,
                    Y_real_range = Y_real_range,
                    Y_name=Exp.Y_names[0],
                    log_flag= log_flag,
                    n_dim=Exp.n_dim,
                    x_indices=x_indices,
                    X_ranges=Exp.X_ranges,
                    X_names=Exp.X_names,
                    X_train=X_train,
                    X_new=X_new,
                    save_fig=save_fig,
                    save_path=Exp.exp_path,
                    i_iter='objective')


def objective_heatmap(
    objective_func: object,
    X_ranges: MatrixLike2d,
    Y_name: Optional[str] = None,
    Y_real_range: Optional[ArrayLike1d] = None, 
    log_flag: Optional[bool] = False,
    x_indices: Optional[List[int]] = [0, 1],
    fixed_values: Optional[Union[ArrayLike1d, float]] = [],
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = [],
    X_names: Optional[List[str]] = None, 
    X_train: Optional[MatrixLike2d] = None, 
    X_new: Optional[MatrixLike2d] = None,
    baseline: Optional[str] = 'left',
    mesh_size: Optional[int] = 41,
    save_fig: Optional[bool] = False, 
    name: Optional[str] = 'simple_experiment'
):
    """Show a 3-dimensional response surface 
    in a real scale 
    Using the experiment object

    Parameters
    ----------
    objective_func : function object
        a objective function to optimize
    X_ranges : MatrixLike2d, 
            list of x ranges
    Y_name : Optional[str], optional
        Name of Y variable, by default None
    Y_real_range : Optional[ArrayLike1d], optional
        Ranges of the response, [lb, rb]
        to show on the plot, by default None
    log_flag : Optional[bool], optional
        flag to plot in a log scale
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    fixed_values : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a unit scale, by default []
    fixed_values_real : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a real scale, by default []
    baseline : Optional[str], optional
        the choice of baseline, must be left, right or center
    X_name: Optional[List(str)], optional
        Names of X varibale shown as x,y,z-labels
        by default None
    X_train : Optional[MatrixLike2d], optional
        Data points used in training, by default None
    X_new : Optional[MatrixLike2d], optional
        The next data point, i.e the infill points,
        by default None
    mesh_size : Optional[int], optional
        mesh size, by default 41
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    name : Optional[str], optional
            Name of the objective function, 
            by default 'simple_experiment'
    """
    n_dim = len(X_ranges)
    # Create 2D mesh test points  
    X_test, _, _ = create_full_X_test_2d(X_ranges=X_ranges, 
                                         x_indices=x_indices,
                                         fixed_values=fixed_values,
                                         fixed_values_real=fixed_values_real,
                                         baseline=baseline,
                                         mesh_size=mesh_size) 
    # Calculate objective function value 
    Y_obj_test = eval_objective_func(X_test, X_ranges, objective_func)
    Y_obj_test_2d = transform_Y_mesh_2d(Y_obj_test, mesh_size=mesh_size)

    # Set up the path to save graphical results
    parent_dir = os.getcwd()
    exp_path = os.path.join(parent_dir, name)

    response_heatmap(Y_real=Y_obj_test_2d,  
                    Y_real_range=Y_real_range,
                    Y_name=Y_name,
                    log_flag= log_flag,
                    n_dim=n_dim,
                    x_indices=x_indices,
                    X_ranges=X_ranges,
                    X_names=X_names,
                    X_train=X_train,
                    X_new=X_new,
                    save_fig=save_fig,
                    save_path=exp_path,
                    i_iter='objective')
    

def response_heatmap_err_exp(
    Exp: Experiment,
    X_new: Optional[MatrixLike2d] = None,
    Y_real_range: Optional[ArrayLike1d] = None, 
    log_flag: Optional[bool] = False,
    x_indices: Optional[List[int]] = [0, 1],
    fixed_values: Optional[Union[ArrayLike1d, float]] = [],
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = [],
    baseline: Optional[str] = 'left',
    mesh_size: Optional[int] = 41,
    save_fig: Optional[bool] = False):
    """Show a heat map for percentage error 
    (objective - response)/objective in a real scale
    Using the experiment object
    Parameters
    ----------
    Exp : Experiment
        Experiment object
    X_new : Optional[MatrixLike2d], optional
        The next data point, i.e the infill points,
        by default None
    Y_real_range : Optional[ArrayLike1d], optional
        Ranges of the response, [lb, rb]
        to show on the plot, by default None
    log_flag : Optional[bool], optional
        flag to plot in a log scale
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    fixed_values : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a unit scale, by default []
    fixed_values_real : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a real scale, by default []
    baseline : Optional[str], optional
        the choice of baseline, must be left, right or center
    mesh_size : Optional[int], optional
        mesh size, by default 41
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    """
    # Create 2D mesh test points  
    X_test, _, _ = create_full_X_test_2d(X_ranges=Exp.X_ranges, 
                                         x_indices=x_indices,
                                         fixed_values=fixed_values,
                                         fixed_values_real=fixed_values_real,
                                         baseline=baseline,
                                         mesh_size=mesh_size) 
    # Make prediction using the GP model
    Y_test = Exp.predict_real(X_test)
    Y_test_2d = transform_Y_mesh_2d(Y_test, mesh_size=mesh_size)

    # Calculate objective function value 
    Y_obj_test = eval_objective_func(X_test, Exp.X_ranges, Exp.objective_func)
    Y_obj_test_2d = transform_Y_mesh_2d(Y_obj_test, mesh_size=mesh_size)

    # Calculate the percentage errors
    Y_err_2d = np.abs((Y_obj_test_2d - Y_test_2d)/Y_obj_test_2d)
    response_heatmap(Y_real=Y_err_2d,
                    Y_real_range = Y_real_range,
                    Y_name = Exp.Y_names[0]+'_error',
                    log_flag= log_flag,
                    n_dim=Exp.n_dim,
                    x_indices=x_indices,
                    X_ranges=Exp.X_ranges,
                    X_names=Exp.X_names,
                    X_train=Exp.X,
                    X_new=X_new,
                    save_fig=save_fig,
                    save_path=Exp.exp_path,
                    i_iter=Exp.n_points - Exp.n_points_init)
    
#%% Functions for 2 dimensional systems on response sufaces
def response_surface(
    X1_test: MatrixLike2d,
    X2_test: MatrixLike2d,
    Y_real: MatrixLike2d,  
    Y_real_lower: Optional[MatrixLike2d] = None, 
    Y_real_upper: Optional[MatrixLike2d] = None, 
    Y_real_range: Optional[ArrayLike1d] = None,
    Y_name: Optional[str] = None,
    n_dim: Optional[int] = 2,
    log_flag: Optional[bool] = False,
    x_indices: Optional[List[int]] = [0, 1],
    X_ranges: Optional[MatrixLike2d] = None,
    X_names: Optional[List[str]] = None, 
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None,
    i_iter: Optional[Union[str, int]] = ''):
    """Plot a response surface in 3-dimensional space

    Parameters
    ----------
    X1_test : MatrixLike2d
        [description]
    X2_test : MatrixLike2d
        [description]
    Y_real : MatrixLike2d
        Response in a real scale
    Y_real_lower : Optional[MatrixLike2d], optional
        Model predicted lower bound in a real scale, 
        by default None
    Y_real_upper : Optional[MatrixLike2d], optional
        Model predicted lower bound in a real scale, , by default None
    Y_real_range : ArrayLike1d
        Ranges of the response, [lb, rb]
    Y_name : Optional[str], optional
        Name of Y variable, by default None
    n_dim : Optional[int], optional
        Dimensional of X, i.e., number of columns 
    log_flag : Optional[bool], optional
        flag to plot in a log scale
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
    X_name: Optional[List(str)], optional
        Names of X varibale shown as x,y,z-labels
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory
    i_iter: Optional[str], optional
        Iteration number to add to the figure name
        by default '''
    """
    # Preprocess Y_real
    X1_test = tensor_to_np(X1_test)
    X2_test = tensor_to_np(X2_test)
    Y_real = tensor_to_np(Y_real)

    if Y_real_lower is not None:
        Y_real_lower = tensor_to_np(Y_real_lower)
    if Y_real_upper is not None:
        Y_real_upper = tensor_to_np(Y_real_upper)
    # Set default Y_real_range
    if Y_real_range is None:
        Y_real_range = [np.min(Y_real), np.max(Y_real)]
    if log_flag:
        Y_real = np.log10(abs(Y_real))
        if Y_real_lower is not None:
            Y_real_lower = np.log10(abs(Y_real_lower))
        if Y_real_upper is not None:
            Y_real_upper = np.log10(abs(Y_real_upper))
    
    # Extract two variable indices for plotting
    x_indices = sorted(x_indices) 
    index_0 = x_indices[0]
    index_1 = x_indices[1]
    
    # Set default axis names 
    if X_names is None:
            X_names = ['x' + str(xi + 1) for xi in range(n_dim)]
    # Set Y_name in file name
    if Y_name is None:
        Y_name = 'y'
        Y_name_plot = 'y'
    else: 
        Y_name_plot = Y_name
    # set the file name
    filename = 'surface_'+ Y_name + '_' + str(index_0) +\
         str(index_1) + '_i_' + str(i_iter) 
    
    # Set default [0,1] range for a unit scale
    if X_ranges is None:
        X_ranges = [[0,1]] * n_dim
    # Set default number of sections
    n_tick_sections  = 5

    # Visualize response - a 3D surfaceplot
    fig  = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1_test, X2_test, Y_real, cmap = 'jet', \
        vmin = Y_real_range[0], vmax = Y_real_range[1]) 
    #  Obtain axes limits
    xlim_plot = list(ax.set_xlim(0, 1))
    ylim_plot = list(ax.set_ylim(0, 1))
    zlim_plot = list(ax.set_zlim(Y_real_range))
    if Y_real_lower is not None:
        ax.plot_surface(X1_test, X2_test, Y_real_lower, cmap = 'Blues', alpha = 0.7, \
            vmin = Y_real_range[0], vmax = Y_real_range[1]) 
    if Y_real_upper is not None:
        ax.plot_surface(X1_test, X2_test, Y_real_upper, cmap = 'Reds', alpha = 0.7, \
            vmin = Y_real_range[0], vmax = Y_real_range[1]) 
    
    # set axis labels and ticks   
    ax.set_xlabel(X_names[index_0], labelpad=15)
    ax.set_ylabel(X_names[index_1], labelpad=15)
    ax.set_zlabel(Y_name_plot, labelpad=10)
    ax.set_xticks(set_axis_values(xlim_plot, n_tick_sections))
    ax.set_xticklabels(set_axis_values(X_ranges[index_0], n_tick_sections))
    ax.set_yticks(set_axis_values(ylim_plot, n_tick_sections))
    ax.set_yticklabels(set_axis_values(X_ranges[index_1], n_tick_sections))
    ax.set_zticks(set_axis_values(zlim_plot, n_tick_sections, 1))
    ax.set_zticklabels(set_axis_values(zlim_plot, n_tick_sections, 1))

    ax.view_init(30, 45)

    plt.show()
    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()

        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, filename + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)


def response_scatter_exp(
    Exp: Experiment,
    Y_real_range: Optional[ArrayLike1d] = None,
    Y_name: Optional[str] = None,
    n_dim: Optional[int] = 3,
    log_flag: Optional[bool] = False,
    x_indices: Optional[List[int]] = [0, 1, 2],
    X_ranges: Optional[MatrixLike2d] = None,
    X_names: Optional[List[str]] = None, 
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None,
    i_iter: Optional[Union[str, int]] = ''):
    """Plot a response surface in 3-dimensional space

    Parameters
    ----------
    Y_real_range : ArrayLike1d
        Ranges of the response, [lb, rb]
    Y_name : Optional[str], optional
        Name of Y variable, by default None
    n_dim : Optional[int], optional
        Dimensional of X, i.e., number of columns 
    log_flag : Optional[bool], optional
        flag to plot in a log scale
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    X_ranges : MatrixLike2d, optional
            list of x ranges, by default None
    X_names: Optional[List(str)], optional
        Names of X varibale shown as x,y,z-labels
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory
    i_iter: Optional[str], optional
        Iteration number to add to the figure name
        by default '''
    """

    Y_real = Exp.Y_real
    # Set default Y_real_range
    if Y_real_range is None:
        Y_real_range = [np.min(Y_real), np.max(Y_real)]
    if log_flag:
        Y_real = np.log10(abs(Y_real))
    
    # Extract two variable indices for plotting
    x_indices = sorted(x_indices) 
    index_0 = x_indices[0]
    index_1 = x_indices[1]
    index_2 = x_indices[2]
    
    # Set default axis names 
    if X_names is None:
            X_names = ['x' + str(xi + 1) for xi in range(n_dim)]
    # Set Y_name in file name
    if Y_name is None:
        Y_name = 'y'
        Y_name_plot = 'y'
    else: 
        Y_name_plot = Y_name
    # set the file name
    filename = 'scatter_'+ Y_name + '_' + str(index_0) +\
         str(index_1) + str(index_2) + '_i_' + str(i_iter) 
    
    # Set default X_ranges 
    if X_ranges is None:
        X_ranges = [[np.min(Exp.X_real[:, index_0]), np.max(Exp.X_real[:, index_0])], 
                    [np.min(Exp.X_real[:, index_1]), np.max(Exp.X_real[:, index_1])], 
                    [np.min(Exp.X_real[:, index_2]), np.max(Exp.X_real[:, index_2])]]
    # Set default number of sections
    n_tick_sections  = 5

    # Visualize response - a 3D surfaceplot
    fig  = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')
    im = ax.scatter(Exp.X_real[:, index_0], Exp.X_real[:, index_1], Exp.X_real[:, index_2], 
                      vmin=Y_real_range[0], vmax=Y_real_range[1], linewidths=1, alpha=0.7, 
                      edgecolor='k', s=60, c=Y_real)

    #  Obtain axes limits
    xlim_plot = list(ax.set_xlim(X_ranges[index_0]))
    ylim_plot = list(ax.set_ylim(X_ranges[index_1]))
    zlim_plot = list(ax.set_zlim(X_ranges[index_2]))
    
    # set axis labels and ticks   
    ax.set_xlabel(X_names[index_0], labelpad=15)
    ax.set_ylabel(X_names[index_1], labelpad=15)
    ax.set_zlabel(X_names[index_2], labelpad=15)
    ax.set_xticks(set_axis_values(xlim_plot, n_tick_sections))
    ax.set_xticklabels(set_axis_values(X_ranges[index_0], n_tick_sections))
    ax.set_yticks(set_axis_values(ylim_plot, n_tick_sections))
    ax.set_yticklabels(set_axis_values(X_ranges[index_1], n_tick_sections))
    ax.set_zticks(set_axis_values(ylim_plot, n_tick_sections))
    ax.set_zticklabels(set_axis_values(X_ranges[index_2], n_tick_sections))

    # set colorbar for response
    cbar = fig.colorbar(im, ax=ax).set_label(label=Y_name, rotation=270, labelpad=20)

    ax.view_init(30, 45)

    plt.show()
    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()

        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, filename + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)

def response_surface_exp(
    Exp: Experiment,
    Y_real_range: Optional[ArrayLike1d] = None, 
    log_flag: Optional[bool] = False,
    x_indices: Optional[List[int]] = [0, 1],
    fixed_values: Optional[Union[ArrayLike1d, float]] = [],
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = [],
    baseline: Optional[str] = 'left',
    show_confidence: Optional[bool] = False,
    mesh_size: Optional[int] = 41,
    save_fig: Optional[bool] = False):
    """Show a 3-dimensional response surface 
    in a real scale 
    Using the experiment object

    Parameters
    ----------
    Exp : Experiment
        Experiment object
    Y_real_range : Optional[ArrayLike1d], optional
        Ranges of the response, [lb, rb]
        to show on the plot, by default None
    log_flag : Optional[bool], optional
        flag to plot in a log scale
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    fixed_values : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a unit scale, by default []
    fixed_values_real : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a real scale, by default []
    baseline : Optional[str], optional
        the choice of baseline, must be left, right or center
    mesh_size : Optional[int], optional
        mesh size, by default 41
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    """
    # Create 2D mesh test points  
    X_test, X1_test, X2_test = create_full_X_test_2d(X_ranges=Exp.X_ranges, 
                                                     x_indices=x_indices,
                                                     fixed_values=fixed_values,
                                                     fixed_values_real=fixed_values_real,
                                                     baseline=baseline,
                                                     mesh_size=mesh_size) 
    # Make predictions using the GP model
    if show_confidence:
        Y_test, Y_test_lower, Y_test_upper = Exp.predict_real(X_test, show_confidence = True)
        Y_test_2D = transform_Y_mesh_2d(Y_test, mesh_size)
        Y_test_lower_2D = transform_Y_mesh_2d(Y_test_lower, mesh_size)
        Y_test_upper_2D = transform_Y_mesh_2d(Y_test_upper, mesh_size)
    else:
        Y_test = Exp.predict_real(X_test)
        Y_test_2D = transform_Y_mesh_2d(Y_test, mesh_size)
        Y_test_lower_2D, Y_test_upper_2D = None, None
        
    response_surface(X1_test=X1_test,
                     X2_test=X2_test,
                     Y_real=Y_test_2D,  
                     Y_real_lower=Y_test_lower_2D, 
                     Y_real_upper=Y_test_upper_2D, 
                     Y_real_range=Y_real_range,
                     Y_name=Exp.Y_names[0],
                     n_dim=Exp.n_dim,
                     log_flag=log_flag,
                     x_indices=x_indices,
                     X_ranges=Exp.X_ranges,
                     X_names=Exp.X_names,
                     save_fig=save_fig,
                     save_path=Exp.exp_path,
                     i_iter=Exp.n_points - Exp.n_points_init)


def objective_surface_exp(
    Exp: Experiment,
    Y_real_range: Optional[ArrayLike1d] = None, 
    log_flag: Optional[bool] = False,
    x_indices: Optional[List[int]] = [0, 1],
    fixed_values: Optional[Union[ArrayLike1d, float]] = [],
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = [],
    baseline: Optional[str] = 'left',
    mesh_size: Optional[int] = 41,
    save_fig: Optional[bool] = False):
    """Show a 3-dimensional response surface 
    in a real scale 
    Using the experiment object

    Parameters
    ----------
    Exp : Experiment
        Experiment object
    Y_real_range : Optional[ArrayLike1d], optional
        Ranges of the response, [lb, rb]
        to show on the plot, by default None
    log_flag : Optional[bool], optional
        flag to plot in a log scale
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    fixed_values : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a unit scale, by default []
    fixed_values_real : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a real scale, by default []
    baseline : Optional[str], optional
        the choice of baseline, must be left, right or center
    mesh_size : Optional[int], optional
        mesh size, by default 41
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    """
    # Create 2D mesh test points  
    X_test, X1_test, X2_test = create_full_X_test_2d(X_ranges=Exp.X_ranges, 
                                                     x_indices=x_indices,
                                                     fixed_values=fixed_values,
                                                     fixed_values_real=fixed_values_real,
                                                     baseline=baseline,
                                                     mesh_size=mesh_size) 
    # Calculate objective function value 
    Y_obj_test = eval_objective_func(X_test, Exp.X_ranges, Exp.objective_func)
    Y_obj_test_2D = transform_Y_mesh_2d(Y_obj_test, mesh_size)
    Y_obj_lower_2D, Y_obj_upper_2D = None, None
    
    response_surface(X1_test=X1_test,
                     X2_test=X2_test,
                     Y_real=Y_obj_test_2D,  
                     Y_real_lower=Y_obj_lower_2D, 
                     Y_real_upper=Y_obj_upper_2D, 
                     Y_real_range=Y_real_range,
                     Y_name=Exp.Y_names[0],
                     n_dim=Exp.n_dim,
                     log_flag=log_flag,
                     x_indices=x_indices,
                     X_ranges=Exp.X_ranges,
                     X_names=Exp.X_names[x_indices], 
                     save_fig=save_fig,
                     save_path=Exp.exp_path,
                     i_iter='objective')


def objective_surface(
    objective_func: object,
    X_ranges: MatrixLike2d,
    Y_name: Optional[str] = None,
    Y_real_range: Optional[ArrayLike1d] = None, 
    log_flag: Optional[bool] = False,
    x_indices: Optional[List[int]] = [0, 1],
    fixed_values: Optional[Union[ArrayLike1d, float]] = [],
    fixed_values_real: Optional[Union[ArrayLike1d, float]] = [],
    baseline: Optional[str] = 'left',
    X_names: Optional[List[str]] = None, 
    mesh_size: Optional[int] = 41,
    save_fig: Optional[bool] = False, 
    name: Optional[str] = 'simple_experiment'
):
    """Show a 3-dimensional response surface 
    in a real scale 
    Using the experiment object

    Parameters
    ----------
    objective_func : function object
        a objective function to optimize
    X_ranges : MatrixLike2d, 
            list of x ranges
    Y_name : Optional[str], optional
        Name of Y variable, by default None
    Y_real_range : Optional[ArrayLike1d], optional
        Ranges of the response, [lb, rb]
        to show on the plot, by default None
    log_flag : Optional[bool], optional
        flag to plot in a log scale
    x_indices : Optional[List[int]], optional
        indices of two x variables, by default [0, 1]
    fixed_values : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a unit scale, by default []
    fixed_values_real : Optional[Union[ArrayLike1d, float]], optional
        fixed values in other dimensions, 
        in a real scale, by default []
    baseline : Optional[str], optional
        the choice of baseline, must be left, right or center
    X_name: Optional[List(str)], optional
        Names of X varibale shown as x,y,z-labels
        by default None
    mesh_size : Optional[int], optional
        mesh size, by default 41
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    name : Optional[str], optional
            Name of the objective function, 
            by default 'simple_experiment'
    """
    n_dim = len(X_ranges)
    # Create 2D mesh test points  
    X_test, X1_test, X2_test = create_full_X_test_2d(X_ranges=X_ranges, 
                                                     x_indices=x_indices,
                                                     fixed_values=fixed_values,
                                                     fixed_values_real=fixed_values_real,
                                                     baseline=baseline,
                                                     mesh_size=mesh_size) 
    # Calculate objective function value 
    Y_obj_test = eval_objective_func(X_test, X_ranges, objective_func)
    Y_obj_test_2D = transform_Y_mesh_2d(Y_obj_test, mesh_size)
    Y_obj_lower_2D, Y_obj_upper_2D = None, None


    # Set up the path to save graphical results
    parent_dir = os.getcwd()
    exp_path = os.path.join(parent_dir, name)

    response_surface(X1_test=X1_test,
                     X2_test=X2_test,
                     Y_real=Y_obj_test_2D,  
                     Y_real_lower=Y_obj_lower_2D, 
                     Y_real_upper=Y_obj_upper_2D, 
                     Y_real_range=Y_real_range,
                     Y_name=Y_name,
                     n_dim=n_dim,
                     log_flag=log_flag,
                     x_indices=x_indices,
                     X_ranges= X_ranges,
                     X_names= X_names, 
                     save_fig=save_fig,
                     save_path=exp_path,
                     i_iter='objective')


#%% Functions for Pareto front visualization
def pareto_front(
    y1: MatrixLike2d, 
    y2: MatrixLike2d, 
    Y_names: Optional[List[str]] = None, 
    fill: Optional[bool] = True,
    diagonal: Optional[bool] = True,
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
    fill: Optional[bool], optional
        if true fill the space enclosed by the points 
        by default True 
    diagonal: Optional[bool], optional
        if true plot the y = x line
        by default True 
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
    # Set default axis names 
    if Y_names is None:
            Y_names = ['y1', 'y2']

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y1, y2, s=60, alpha = 0.5)
    if fill:
        ax.fill_between(y1, y2, color = 'steelblue', alpha=0.3)
    lims = [
        np.min([y1.min(), y2.min()]),  # min of both axes
        np.max([y1.max(), y2.max()]),  # max of both axes
    ]
    # number of sections in the axis
    nsections = 5
    # now plot both limits against eachother
    if diagonal:
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlim([y1.min(), y1.max()]) #ax.set_xlim(lims)
    ax.set_ylim([y2.min(), y2.max()]) #ax.set_ylim(lims)
    ax.set_xticks(np.around(np.linspace(lims[0], lims[1], nsections), 2))
    ax.set_yticks(np.around(np.linspace(lims[0], lims[1], nsections), 2))
    ax.set_xticklabels(np.around(np.linspace(lims[0], lims[1], nsections), 2))
    ax.set_yticklabels(np.around(np.linspace(lims[0], lims[1], nsections), 2))
    ax.set_xlabel(Y_names[0])
    ax.set_ylabel(Y_names[1])

    plt.show()

    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, 'pareto_'+ str(i_iter) + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)



def pareto_front_exp(
    Exp: Union[WeightedMOOExperiment, EHVIMOOExperiment], 
    fill: Optional[bool] = True,
    diagonal: Optional[bool] = True,
    save_fig: Optional[bool] = False, 
    design_name: Optional[Union[str, int]] = 'final'):
    """Plot parity plot comparing the ground true 
    objective function values against predicted model mean
    Using MOOExperiment object

    Parameters
    ---------
    Exp: Union[WeightedMOOExperiment, EHVIMOOExperiment]
        MOOExperiment object
    fill: Optional[bool], optional
        if true fill the space enclosed by the points 
        by default True 
    diagonal: Optional[bool], optional
        if true plot the y = x line
        by default True 
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
    Y_real_opts = Exp.Y_real_opts

    pareto_front(y1=Y_real_opts[:, 0], 
                 y2=Y_real_opts[:, 1],
                 Y_names=Exp.Y_names,
                 fill=fill,
                 diagonal=diagonal,
                 save_fig=save_fig,
                 save_path=Exp.exp_path,
                 i_iter = design_name)

        




