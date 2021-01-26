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

from nextorch import plotting, bo
import numpy as np

#%% Define the objective function
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

objective_func = simple_1d

#%% Define the initial sampling scheme
# Assume X is already in a unit scale in [0, 1]
X_init = np.array([[0, 0.25, 0.5, 0.75]]).T
# Get the initial responses
Y_init = objective_func(X_init)

# Initialize an Experiment object
# Set its name, the files will be saved under the folder with the same name
Exp = bo.Experiment('simple_1d') 
# Import the initial data
Exp.input_data(X_init, Y_init,  unit_flag = True)
# Set the optimization specifications 
# here we set the objective function, minimization by default
Exp.set_optim_specs(objective_func = simple_1d)

# Create test data points for plotting
X_test = np.linspace(0, 1, 1000)

#%% Optimization loop
# Set the number of iterations  
n = 10
for i in range(n):
    # Generate the next experiment point
    X_new, X_new_real, acq_func = Exp.generate_next_point()
    # Get the reponse at this point
    Y_new_real = objective_func(X_new_real)
    
    # Plot the objective functions, and acqucision function
    plotting.objective_func_1d_exp(Exp, X_test = X_test, X_new = X_new)
    plotting.acq_func_1d_exp(Exp, X_test = X_test, X_new = X_new)
    
    # Retrain the model by input the next point into Exp object
    Exp.run_trial(X_new, X_new_real, Y_new_real)


#%%
# Validation 
y_opt, X_opt, index_opt = Exp.get_optim()
plotting.parity_exp(Exp, save_fig = True)
plotting.parity_with_ci_exp(Exp)