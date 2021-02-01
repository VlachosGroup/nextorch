"""
Example 2

Goal: maximization
Objective function: simple nonlinear
    Input (X) dimension: 1
    Output (Y) dimension: 1
    Analytical form available: Yes
Acqucision function: the default, expected improvement (EI)
Initial Sampling: random
Input X scale: unit

"""

import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

import numpy as np
from nextorch import plotting, bo, utils, doe

#%% Define the objective function
# Objective function
objective_func = lambda x: np.sin(x)

# Set the ranges
X_range = [0, np.pi*2]

#%% Initial Sampling 
# Randomly choose some points
# Sampling X is in a unit scale in [0, 1]
X_init = doe.randomized_design(n_dim = 1, n_points = 4, seed = 0)
#or we can do np.random.rand(4,1)

# Get the initial responses
Y_init_real = bo.eval_objective_func(X_init, X_range, objective_func)

# X in real scale can be obtained via
X_init_real = utils.inverse_unitscale_X(X_init, X_range)
# The reponse can be obtained via
# Y_init_real = objective_func(X_init_real)

#%% Initialize an Experiment object
# Set its name, the files will be saved under the folder with the same name
Exp = bo.Experiment('sin_1d') 
# Import the initial data
Exp.input_data(X_init, Y_init_real, unit_flag = True, X_ranges = X_range)

# Set the optimization specifications 
# here we set the objective function, minimization by default
Exp.set_optim_specs(objective_func = objective_func, minimize = False)

# Create test data points for plotting
X_test = np.linspace(0, 1, 1000)

# Set a flag for saving png figures
save_fig_flag = True

#%% Optimization loop
# Set the number of iterations  
n_trials = 10
for i in range(n_trials):
    # Generate the next experiment point
    X_new, X_new_real, acq_func = Exp.generate_next_point()
    # Get the reponse at this point
    Y_new_real = objective_func(X_new_real)
    
    # Plot the objective functions, and acqucision function
    plotting.objective_func_1d_exp(Exp, X_test = X_test, X_new = X_new, save_fig = save_fig_flag)
    plotting.acq_func_1d_exp(Exp, X_test = X_test, X_new = X_new, save_fig = save_fig_flag)
    
    # Retrain the model by input the next point into Exp object
    Exp.run_trial(X_new, X_new_real, Y_new_real)


#%%
# Validation 
y_opt, X_opt, index_opt = Exp.get_optim()
plotting.parity_exp(Exp, save_fig = save_fig_flag)
plotting.parity_with_ci_exp(Exp, save_fig = save_fig_flag)