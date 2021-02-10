"""
Example 4

Goal: maximization
Objective function: PFR reaction model, yield
    Input (X) dimension: 3
    Output (Y) dimension: 1
    Analytical form available: Yes
Acqucision function: the default, expected improvement (EI)
Initial Sampling: full factorial and Latin Hypercube

"""

import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

import numpy as np
from nextorch import plotting, bo, doe, utils


#%% Define the objective function
from fructose_pfr_model_function import Reactor

# Three input temperature C, pH, log10(residence time)
X_name_list = ['T', 'pH', r'$\rm log_{10}(tf_{min})$']
X_units = [r'$\rm ^oC $', '', '']

n_dim = len(X_name_list) # the dimension of inputs
n_objective = 1 # the dimension of outputs

X_name_with_unit = []
for i, var in enumerate(X_name_list):
    if not X_units[i]  == '':
        var = var + ' ('+ X_units[i] + ')'
    X_name_with_unit.append(var)
    
# One output
Y_name_with_unit = 'Yield %'

# Specify range     
X_ranges =  [[140, 200], [0, 1], [-2, 2]]

def PFR_yield(X_real):
    
    if len(X_real.shape) < 2:
        X_real = np.expand_dims(X_real, axis=1) #If 1D, make it 2D array
        
    Y_real = []
    for i, xi in enumerate(X_real):
        Conditions = {'T_degC (C)': xi[0], 'pH': xi[1], 'tf (min)' : 10**xi[2]}
        yi, _ = Reactor(**Conditions)        
        Y_real.append(yi)
            
    Y_real = np.array(Y_real)
    # Put y in a column
    Y_real = np.expand_dims(Y_real, axis=1)
        
    return Y_real # HMF yield, HMF selectivity

# Objective function
objective_func = PFR_yield

#%% Initial Sampling 

# Full factorial design 
n_ff_level = 4
n_ff = n_ff_level**n_dim
X_init_ff = doe.full_factorial([n_ff_level, n_ff_level, n_ff_level])
# Get the initial responses
Y_init_ff = bo.eval_objective_func(X_init_ff, X_ranges, objective_func)

# Latin hypercube design with 10 initial points
n_init_lhc = 10
X_init_lhc = doe.latin_hypercube(n_dim = n_dim, n_points = n_init_lhc, seed= 1)
# Get the initial responses
Y_init_lhc = bo.eval_objective_func(X_init_lhc, X_ranges, objective_func)

# Compare the two sampling plans
plotting.sampling_3d([X_init_ff, X_init_lhc], 
                     X_names = X_name_with_unit,
                     X_ranges = X_ranges,
                     design_names = ['Full Fatorial', 'LHC'])


#%% Initialize an Experiment object

# Set its name, the files will be saved under the folder with the same name
Exp_ff = bo.Experiment('PFR_yield_ff') 
# Import the initial data
Exp_ff.input_data(X_init_ff, Y_init_ff, X_ranges = X_ranges, unit_flag = True)
# Set the optimization specifications 
# here we set the objective function, minimization by default
Exp_ff.set_optim_specs(objective_func = objective_func, 
                        maximize =  True)


# Set its name, the files will be saved under the folder with the same name
Exp_lhc = bo.Experiment('PFR_yield_lhc') 
# Import the initial data
Exp_lhc.input_data(X_init_lhc, Y_init_lhc, X_ranges = X_ranges, unit_flag = True)
# Set the optimization specifications 
# here we set the objective function, minimization by default
Exp_lhc.set_optim_specs(objective_func = objective_func, 
                        maximize =  True)


#%% Optimization loop

# Set the number of iterations  
n_trials_lhc = n_ff - n_init_lhc
for i in range(n_trials_lhc):
    # Generate the next experiment point
    X_new, X_new_real, acq_func = Exp_lhc.generate_next_point()
    # Get the reponse at this point
    Y_new_real = objective_func(X_new_real)

    # Retrain the model by input the next point into Exp object
    Exp_lhc.run_trial(X_new, X_new_real, Y_new_real)
    
#%% plots 
# Check the sampling points
# Final lhc Sampling
x2_fixed_real = 0.7 # fixed x2 value
Y_plot_range = [0, 50]
x_indices = [0, 2]
plotting.sampling_3d_exp(Exp_lhc, 
                         slice_axis = 'y', 
                         slice_value_real = x2_fixed_real)    
# Compare to full factorial
plotting.sampling_3d([Exp_ff.X, Exp_lhc.X], 
                     X_ranges = X_ranges,
                     design_names = ['Full Fatorial', 'LHC'],
                     slice_axis = 'y', 
                     slice_value_real = x2_fixed_real)

# Reponse heatmaps

# Objective function heatmap
#(this takes a long time)
#plotting.objective_heatmap(objective_func, 
#                           X_ranges, 
#                           Y_real_range = Y_plot_range, 
#                           x_indices = x_indices, 
#                           fixed_values_real = x2_fixed_real)
# LHC heatmap 
plotting.response_heatmap_exp(Exp_lhc, 
                              Y_real_range = Y_plot_range, 
                              x_indices = x_indices, 
                              fixed_values_real = x2_fixed_real)
# full factorial heatmap
plotting.response_heatmap_exp(Exp_ff, 
                              Y_real_range = Y_plot_range,
                              x_indices = x_indices, 
                              fixed_values_real = x2_fixed_real)


#%%
# Suface plots   
# Objective function surface plot  
#(this takes a long time)
#plotting.objective_surface(objective_func, 
#                           X_ranges, 
#                           Y_real_range = Y_plot_range, 
#                           x_indices = x_indices, 
#                           fixed_values_real = x2_fixed_real)
# LHC heatmap
plotting.response_surface_exp(Exp_lhc, 
                              Y_real_range = Y_plot_range,
                              x_indices = x_indices, 
                              fixed_values_real = x2_fixed_real)
# full fatorial error heatmap
plotting.response_surface_exp(Exp_ff, 
                              Y_real_range = Y_plot_range,
                              x_indices = x_indices, 
                              fixed_values_real = x2_fixed_real)


# Compare two plans in terms optimum in each trial
plotting.opt_per_trial([Exp_ff.Y_real, Exp_lhc.Y_real], 
                       design_names = ['Full Fatorial', 'LHC'])

