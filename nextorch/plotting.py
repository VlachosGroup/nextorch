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
import types
from typing import Optional, TypeVar, Union, Tuple, List

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


#%% Not finished yet 
def plot_testing(model, test_X, train_X, train_Y,  test_Y = None, new_X = None, new_Y = None):
    '''
    Test the surrogate model with model, test_X and new_X
    '''

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))

    with torch.no_grad():
        # compute posterior
        posterior = model.posterior(test_X)
        # Get upper and lower confidence bounds (2 standard deviations from the mean)
        lower, upper = posterior.mvn.confidence_region()
        
        # Plot the groud truth test_Y if provided
        ax.plot(test_X.cpu().numpy(), test_Y.cpu().numpy(), 'k--', label = 'Objective f(x)')
        # Plot posterior means as blue line
        ax.plot(test_X.cpu().numpy(), posterior.mean.cpu().numpy(), 'b', label = 'Posterior Mean')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_X.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5, label = 'Confidence')
        
        # Plot training points as black stars
        ax.scatter(train_X.cpu().numpy(), train_Y.cpu().numpy(), s =120, c= 'k', marker = '*', label = 'Initial Data')
         # Plot the new infill points as red stars
        if not type(new_X) == type(None):    
            ax.scatter(new_X.cpu().numpy(), new_Y.cpu().numpy(), s = 120, c = 'r', marker = '*', label = 'Infill Data')
        
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout()
    plt.show()

def plot_acq_func(acq_func, test_X, train_X, new_X = None):
    # compute acquicision function values at test_X
    test_acq_val = acq_func(test_X.view((test_X.shape[0],1,dim)))
    train_acq_val = acq_func(train_X.view((train_X.shape[0],1,dim)))

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))
    with torch.no_grad():
        ax.plot(test_X.cpu().numpy(), test_acq_val.detach(), 'b-', label = 'Acquistion (EI)')
        # Plot training points as black stars
        ax.scatter(train_X.cpu().numpy(), train_acq_val.detach(), s = 120, c= 'k', marker = '*', label = 'Initial Data')
         # Plot the new infill points as red stars
        if not type(new_X) == type(None):
            new_acq_val = acq_func(new_X.view((new_X.shape[0],1,dim)))
            ax.scatter(new_X.cpu().numpy(), new_acq_val.detach(),  s = 120, c ='r', marker = '*', label = 'Infill Data')
    
    ax.ticklabel_format(style = 'sci', axis = 'y', scilimits = (-2,2) )
    ax.set_xlabel('x')
    ax.set_ylabel(r'$ \alpha$')    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout()
    plt.show()