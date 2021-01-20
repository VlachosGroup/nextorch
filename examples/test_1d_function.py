# -*- coding: utf-8 -*-
"""
Example of Bayesian Optimization for 1D example
Use Gaussian Process regression to fit the model 
Use one type of acqucision functions to predict the next infill point
For minmization
"""
import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

from nextorch import plotting
from nextorch import bo
from nextorch import utils
import torch
import numpy as np

from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement

def simple_1d(X):
    """1D function f(x) = (6x-2)^2 * sin(12x-4)

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

# Define the objective function
objective_func = simple_1d


X_init = np.array([[0, 0.25, 0.5, 0.75]]).T
Y_init = objective_func(X_init)

X_test = np.linspace(0, 1, 1000)
Y_test = objective_func(X_test)


Exp = bo.Experiment('simple_1d')
Exp.input_data(X_init, Y_init,  unit_flag = True)
Exp.set_optim_specs(objective_func = simple_1d)

# Optimization loop
n = 10
for i in range(n):
    X_new, _ = Exp.generate_next_point()
    Exp.run_trial(X_new)

