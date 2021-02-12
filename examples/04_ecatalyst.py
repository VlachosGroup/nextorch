"""
Example 4

Goal: maximization
Objective function: simple nonlinear
    Input (X) dimension: 4
    Output (Y) dimension: 1
    Analytical form available: No
Acqucision function: the default, expected improvement (EI)
Initial Sampling: Latin Hypercube
"""

import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

import numpy as np
from nextorch import plotting, bo, doe, io, utils


# Three input final temperature, heating rate, hold time
X_name_list = ['T', 'Heating rate', 'Time']
X_units = ['C', 'C/min', 'hr']
X_units_plot = [r'$\rm ^oC $', r'$\rm ^oC/min $', 'hr']


X_name_with_unit = []
for i, var in enumerate(X_name_list):
    if not X_units[i]  == '':
        var = var + ' ('+ X_units[i] + ')'
    X_name_with_unit.append(var)
    
X_name_with_unit_plot = []
for i, var in enumerate(X_name_list):
    if not X_units_plot[i]  == '':
        var = var + ' ('+ X_units_plot[i] + ')'
    X_name_with_unit_plot.append(var)
    
n_dim = len(X_name_list) # the dimension of inputs
n_objective = 1 # the dimension of outputs
n_trials = 4 # number of experiment iterations


# One output
Y_name_with_unit = 'N_Content'
Y_name_with_unit_plot = r'$\rm N_{Content}%$'

# combine X and Y names
var_names = X_name_with_unit + [Y_name_with_unit]

# Import data
file_path = os.path.join(project_path, 'examples', 'ecatalyst', 'synthesis_data.csv')
data, data_full = io.read_csv(file_path, var_names = var_names)

trial_no = data_full['Trial']

X_real, Y_real = io.split_X_y(data, Y_names = Y_name_with_unit)

# Initial Data 
X_ranges = [[300, 500], [3, 8], [2, 6]]
X_init_real = X_real[trial_no==0]
Y_init_real = Y_real[trial_no==0]


# assume we run Latin hypder cube
#n_init_lhc = 10
#X_init_lhc = doe.latin_hypercube(n_dim = n_dim, n_points = n_init_lhc)

# Compare the two sampling plans
X_init = utils.unitscale_X(X_init_real, X_ranges)
# sampling_3d takes in X in unit scales 
plotting.sampling_3d(X_init, 
                     X_names = X_name_with_unit,
                     X_ranges = X_ranges,
                     design_names = 'LHC')

X_per_trial = [X_init]
#%% Initialize an experimental object 
# Set its name, the files will be saved under the folder with the same name
Exp = bo.Experiment('ecatalyst') 
# Import the initial data
Exp.input_data(X_init_real, Y_init_real, X_names = X_name_with_unit_plot, Y_names = Y_name_with_unit_plot, \
               X_ranges = X_ranges, unit_flag = False)
# Set the optimization specifications 
# here we set the objective function, minimization by default
Exp.set_optim_specs(maximize =  True)

# 2D surface for variable 1 and 3 with variable 2 fixed 
x1_fixed_real = 300 # fixed x1 value
x2_fixed_real = 8 # fixed x2 value
x3_fixed_real = np.mean(X_ranges[2]) # fixed x3 value

Y_plot_range = [0, 2.5]

plotting.sampling_3d_exp(Exp, slice_axis = 'x', slice_value_real = x1_fixed_real)    
plotting.response_heatmap_exp(Exp, Y_real_range = Y_plot_range, x_indices = [1, 2],fixed_values_real = x1_fixed_real)

plotting.sampling_3d_exp(Exp, slice_axis = 'y', slice_value_real = x2_fixed_real)    
plotting.response_heatmap_exp(Exp, Y_real_range = Y_plot_range, x_indices = [0, 2],fixed_values_real = x2_fixed_real)

plotting.sampling_3d_exp(Exp, slice_axis = 'z', slice_value_real = x3_fixed_real)    
plotting.response_heatmap_exp(Exp, Y_real_range = Y_plot_range, x_indices = [0, 1],fixed_values_real = x3_fixed_real)



for i in range(1, n_trials+1):
    # Generate the next three experiment point
    # X_new, X_new_real, acq_func = Exp.generate_next_point(3)
    X_new_real = X_real[trial_no == i]
    X_new = utils.unitscale_X(X_new_real, X_ranges)
    X_per_trial.append(X_new)
    # Get the reponse at this point
    # run the experiments and get the data
    Y_new_real = Y_real[trial_no == i]
    
    # Retrain the model by input the next point into Exp object
    Exp.run_trial(X_new, X_new_real, Y_new_real)

    plotting.response_heatmap_exp(Exp, Y_real_range = Y_plot_range, x_indices = [1, 2], fixed_values_real = x1_fixed_real)
    plotting.response_heatmap_exp(Exp, Y_real_range = Y_plot_range, x_indices = [0, 2], fixed_values_real = x2_fixed_real)
    plotting.response_heatmap_exp(Exp, Y_real_range = Y_plot_range, x_indices = [0, 1], fixed_values_real = x3_fixed_real)
    
#%%
labels = ['trial_' + str(i) for i in range(n_trials+1)]
plotting.sampling_3d(X_per_trial, X_names = X_name_with_unit,  X_ranges = X_ranges, design_names = labels)