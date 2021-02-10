"""
Example 3

Goal: maximization 
Objective function: 2D Langmuir Hinshelwood mechanism
    Input (X) dimension: 2
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
from nextorch import plotting, bo, doe


# Set a flag for saving png figures
save_fig_flag = False

#%% Define the objective function
def rate(P): 
    '''
    langmuir hinshelwood mechanism
    X is a matrix of [P1, P2] in real units
    return r is the reactio rate
    '''
    # kinetic constants
    K1 = 1
    K2 = 10
    
    krds = 100
    
    if len(P.shape) < 2:
        P = np.array([P])
        
    r = np.zeros(P.shape[0])

    for i in range(P.shape[0]):
        P1, P2 = P[i][0], P[i][1]
        r[i] = krds*K1*K2*P1*P2/((1+K1*P1+K2*P2)**2)
    
    # Put y in a column
    r = np.expand_dims(r, axis=1)
    
    return r
# Objective function
objective_func = rate

# Set the ranges
X_ranges = [[1, 10], [1, 10]]

#%% Initial Sampling 
n_ff_level = 5
n_ff = n_ff_level**2
# Full factorial design 
X_init_ff = doe.full_factorial([n_ff_level, n_ff_level])
# Get the initial responses
Y_init_ff = bo.eval_objective_func(X_init_ff, X_ranges, objective_func)

n_init_lhc = 10
# Latin hypercube design with 10 initial points
X_init_lhc = doe.latin_hypercube(n_dim = 2, n_points = n_init_lhc, seed= 1)
# Get the initial responses
Y_init_lhc = bo.eval_objective_func(X_init_lhc, X_ranges, objective_func)

# Compare the two sampling plans
plotting.sampling_2d([X_init_ff, X_init_lhc], 
                     X_ranges = X_ranges,
                     design_names = ['Full Fatorial', 'LHC'])


#%% Initialize an Experiment object

# Set its name, the files will be saved under the folder with the same name
Exp_ff = bo.Experiment('LH_mechanism_ff') 
# Import the initial data
Exp_ff.input_data(X_init_ff, Y_init_ff, X_ranges = X_ranges, unit_flag = True)
# Set the optimization specifications 
# here we set the objective function, minimization by default
Exp_ff.set_optim_specs(objective_func = objective_func, 
                        maximize =  True)


# Set its name, the files will be saved under the folder with the same name
Exp_lhc = bo.Experiment('LH_mechanism_lhc') 
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
plotting.sampling_2d_exp(Exp_lhc)    
# Compare to full factorial
plotting.sampling_2d([Exp_ff.X, Exp_lhc.X], 
                     X_ranges = X_ranges,
                     design_names = ['Full Fatorial', 'LHC'])

# Reponse heatmaps
# Objective function heatmap
plotting.objective_heatmap(objective_func, X_ranges, Y_real_range = [0, 25])
# LHC heatmap
plotting.response_heatmap_exp(Exp_lhc, Y_real_range = [0, 25])
# full factorial heatmap
plotting.response_heatmap_exp(Exp_ff, Y_real_range = [0, 25])


# LHC error heatmap
plotting.response_heatmap_err_exp(Exp_lhc, Y_real_range = [0, 5])
# full fatorial error heatmap
plotting.response_heatmap_err_exp(Exp_ff, Y_real_range = [0, 5])


# Suface plots   
# Objective function surface plot  
plotting.objective_surface(objective_func, X_ranges, Y_real_range = [0, 25])
# LHC heatmap
plotting.response_surface_exp(Exp_lhc, Y_real_range = [0, 25])
# full fatorial error heatmap
plotting.response_surface_exp(Exp_ff, Y_real_range = [0, 25])


# Compare two plans in terms optimum in each trial
plotting.opt_per_trial([Exp_ff.Y_real, Exp_lhc.Y_real], 
                       design_names = ['Full Fatorial', 'LHC'])
