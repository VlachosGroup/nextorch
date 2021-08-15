# -*- coding: utf-8 -*-
"""
Test on 1d function
"""

import warnings
warnings.filterwarnings("ignore")

import pytest
import numpy as np
import matplotlib
matplotlib.use('agg')

from nextorch import plotting, bo, utils
from nextorch.parameter import Parameter

def simple_1d(X):
    """1D function y = (6x-2)^2 * sin(12x-4)

    Parameters
    ----------
    X : numpy array or a list
        1D independent variable

    Returns
    -------
    y: numpy array
        1D dependent variable
    """
    try:
        X.shape[1]
    except:
        X = np.array(X)
    if len(X.shape)<2:
        X = np.array([X])
        
    y = np.array([],dtype=float)
    
    for i in range(X.shape[0]):
        ynew = (X[i]*6-2)**2*np.sin((X[i]*6-2)*2) 
        y = np.append(y, ynew)
    y = y.reshape(X.shape)
    
    return y

objective_func = simple_1d

# Create a grid with a 0.25 interval
X_init = np.array([[0, 0.25, 0.5, 0.75]]).T

# Get the initial responses
Y_init = objective_func(X_init)

# Initialize an Experiment object Exp
# Set its name, the files will be saved under the folder with the same name
Exp = bo.Experiment('test_out_simple_1d') 

# Define parameter space
parameter = Parameter()
Exp.define_space(parameter)

# Import the initial data
# Set unit_flag to true since the X is in a unit scale
Exp.input_data(X_init, Y_init, unit_flag = True)

# Set the optimization specifications 
# Here we set the objective function, minimization as the goal
Exp.set_optim_specs(objective_func = objective_func, maximize = False)

# Set a flag for saving png figures
save_fig_flag = True

# Set the number of iterations  
n_trials = 10

# Optimization loop
for i in range(n_trials):
    # Generate the next experiment point
    # X_new is in a unit scale
    # X_new_real is in a real scale defined in X_ranges
    # Select EI as the acquisition function 
    X_new, X_new_real, acq_func = Exp.generate_next_point(acq_func_name = 'EI')
    # Get the reponse at this point
    Y_new_real = objective_func(X_new_real)
    
    # Plot the objective functions, and acqucision function
    plotting.response_1d_exp(Exp, mesh_size = 1000, X_new = X_new, plot_real = True, save_fig = save_fig_flag)
    plotting.acq_func_1d_exp(Exp, mesh_size = 1000,X_new = X_new, save_fig = save_fig_flag)
    
    # Input X and Y of the next point into Exp object
    # Retrain the model 
    Exp.run_trial(X_new, X_new_real, Y_new_real)

# Obtain the optimum
y_opt, X_opt, index_opt = Exp.get_optim()

# Make a parity plot comparing model predictions versus ground truth values
plotting.parity_exp(Exp, save_fig = save_fig_flag)
# Make a parity plot with the confidence intervals on the predictions
plotting.parity_with_ci_exp(Exp, save_fig = save_fig_flag)

# switch back to interactive mode
# matplotlib.use('TkAgg')


def test_input():
    # Test on input X, Y
    assert np.all(Exp.X_real[:4, :] == X_init)
    assert np.all(Exp.Y_real[:4, :] == Y_init)
    
def test_opt():
    # Test on optimal X and Y
    expected_X_opt = pytest.approx(0.75, abs=0.01)
    expected_Y_opt = pytest.approx(-6.02, abs=0.01)
    assert X_opt[0] == expected_X_opt 
    assert y_opt == expected_Y_opt  


