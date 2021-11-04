"""
Template to run lab experiments (human-in-the-loop)
Use this for the initial set of experimental design
"""

#%% 1. Import NEXTorch and other packages
import os, sys
import numpy as np
from nextorch import plotting, bo, doe, io, utils

# set a random seed
r_seed = 25
np.random.seed(r_seed)

#%% 2. Define the design space 
# the number of dimensions (n_dim) is equal to the number of input parameters
# Set the input names and units
X_name_list = ['T', 'Heating rate', 'Time']
X_units = ['C', 'C/min', 'hr']
# Add the units
X_name_with_unit = []
for i, var in enumerate(X_name_list):
    if not X_units[i]  == '':
        var = var + ' ('+ X_units[i] + ')'
    X_name_with_unit.append(var)

# Set the output names
Y_name_with_unit = 'N_Content %'

# combine X and Y names
var_names = X_name_with_unit + [Y_name_with_unit]

# Set the operating range for each parameter
X_ranges = [[300, 500], 
            [3, 8], 
            [2, 6]] 

# Set the reponse plotting range
Y_plot_range = [0, 2.5]

# Get the information of the design space
n_dim = len(X_name_list) # the dimension of inputs
n_objective = 1 # the dimension of outputs


#%% 3. Define the initial sampling plan
# Select a design of experimental method first
# Assume we run Latin hypder cube to create the initial samplinmg
# Set the initial sampling points, approximately 5*n_dim
n_init_lhs = 16
X_init_lhs = doe.latin_hypercube(n_dim=n_dim, n_points=n_init_lhs, seed=r_seed)

# Convert the sampling plan to a unit scale
X_init_real = utils.inverse_unitscale_X(X_init_lhs, X_ranges)

# Visualize the sampling plan,
# Sampling_3d takes in X in unit scales
plotting.sampling_3d(X_init_lhs,
                     X_names = X_name_with_unit,
                     X_ranges = X_ranges,
                     design_names = 'LHS')

print('The predicted new data points in the initial set are:')
print(io.np_to_dataframe(X_init_real, X_name_with_unit, n = len(X_init_real)))

# Round up these numbers if needed
# Now it's time to run those experiments and gather the Y (reponse) values!