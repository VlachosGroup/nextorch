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
from nextorch.utils import tensor_to_np, np_to_tensor

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
    X_train: MatrixLike2d, 
    X_new: Optional[MatrixLike2d] = None):
    """Plot 1-dimensional acquision function 

    Parameters
    ----------
    acq_func : AcquisitionFunction
        the acquision function object
    X_test : MatrixLike2d
        Test data points for plotting
    X_train : MatrixLike2d
        Training data points
    X_new : Optional[MatrixLike2d], optional
        The next data point, i.e the infill points,
        by default None
    """
    n_dim = 1
    # compute acquicision function values at X_test and X_train
    test_acq_val = acq_func(np_to_tensor(X_test).view((X_test.shape[0],1, n_dim)))
    train_acq_val = acq_func(np_to_tensor(X_train).view((X_train.shape[0],1,n_dim)))
    test_acq_val = tensor_to_np(test_acq_val)
    train_acq_val = tensor_to_np(train_acq_val)


    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(X_test, test_acq_val, 'b-', label = 'Acquisition')
    # Plot training points as black stars
    ax.scatter(X_train, train_acq_val, s = 120, c= 'k', marker = '*', label = 'Initial Data')
        # Plot the new infill points as red stars
    if X_new is not None:
        new_acq_val = acq_func(np_to_tensor(X_new).view((X_new.shape[0],1,n_dim)))
        new_acq_val = tensor_to_np(new_acq_val)
        ax.scatter(X_new, new_acq_val,  s = 120, c ='r', marker = '*', label = 'Infill Data')
    
    ax.ticklabel_format(style = 'sci', axis = 'y', scilimits = (-2,2) )
    ax.set_xlabel('x')
    ax.set_ylabel(r'$ \alpha$')    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout()
    plt.show()



#%% Not finished yet 
def plot_objective_func_1d(
    model: Model, 
    X_test: MatrixLike2d,
    Y_test: MatrixLike2d, 
    X_train: MatrixLike2d, 
    Y_train: MatrixLike2d,  
    X_new: MatrixLike2d, 
    Y_new: MatrixLike2d):
    '''
    Test the surrogate model with model, test_X and new_X
    '''
    # compute posterior
    posterior = model.posterior(np_to_tensor(X_test))
    # Get upper and lower confidence bounds (2 standard deviations from the mean)
    lower, upper = posterior.mvn.confidence_region()

    

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))
        
    # Plot the groud truth Y_test if provided
    ax.plot(X_test.cpu().numpy(), Y_test.cpu().numpy(), 'k--', label = 'Objective f(x)')
    # Plot posterior means as blue line
    ax.plot(X_test.cpu().numpy(), posterior.mean.cpu().numpy(), 'b', label = 'Posterior Mean')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(X_test.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5, label = 'Confidence')
    
    # Plot training points as black stars
    ax.scatter(X_train.cpu().numpy(), Y_train.cpu().numpy(), s =120, c= 'k', marker = '*', label = 'Initial Data')
        # Plot the new infill points as red stars
    if not X_new is None:    
        ax.scatter(X_new.cpu().numpy(), Y_new.cpu().numpy(), s = 120, c = 'r', marker = '*', label = 'Infill Data')
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout()
    plt.show()

