"""
Example 6

Goal: Multi-objective maximization
Objective function: PFR reaction model, yield and selectivity
    Input (X) dimension: 3
    Output (Y) dimension: 2
    Analytical form available: Yes
Acqucision function: the default, expected improvement (EI)
Initial Sampling: full factorial and Latin Hypercube

"""
import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
objective_path = os.path.join(project_path, 'examples', 'PFR')
sys.path.insert(0, project_path)
sys.path.insert(0, objective_path)

import numpy as np
from nextorch import plotting, bo, doe

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
Y_name_with_unit = ['Yield %', 'Selectivity %']

# Specify range     
X_ranges =  [[140, 200], [0, 1], [-2, 2]]

def PFR(X_real):
    
    if len(X_real.shape) < 2:
        X_real = np.expand_dims(X_real, axis=1) #If 1D, make it 2D array
        
    Y_real = []
    for i, xi in enumerate(X_real):
        Conditions = {'T_degC (C)': xi[0], 'pH': xi[1], 'tf (min)' : 10**xi[2]}
        yi = Reactor(**Conditions)        
        Y_real.append(yi)
            
    Y_real = np.array(Y_real)
        
    return Y_real # HMF yield, HMF selectivity

# Objective function
objective_func = PFR



#%% Initial Sampling 
# Latin hypercube design with 10 initial points
n_init_lhc = 10
X_init_lhc = doe.latin_hypercube(n_dim = n_dim, n_points = n_init_lhc, seed= 1)
# Get the initial responses
Y_init_lhc = bo.eval_objective_func(X_init_lhc, X_ranges, objective_func)

# Compare the two sampling plans
plotting.sampling_3d(X_init_lhc, 
                     X_names = X_name_with_unit,
                     X_ranges = X_ranges,
                     design_names = 'LHC')

#%% Initialize an multi-objective Experiment object
# Set its name, the files will be saved under the folder with the same name
Exp_lhc = bo.MOOExperiment('PFR_yield_MOO')  
# Import the initial data
Exp_lhc.input_data(X_init_lhc, 
                   Y_init_lhc, 
                   X_ranges = X_ranges, 
                   X_names = X_name_with_unit,
                   Y_names = Y_name_with_unit,
                   unit_flag = True)
# Set the optimization specifications 
# here we set the objective function, minimization by default
# Set a weight vector for objective 1
weights_vector = np.linspace(0, 1, 10)
Exp_lhc.set_optim_specs(objective_func = objective_func, 
                        maximize = True, 
                        weights = weights_vector)
#%%
# Set the number of iterations for each experiments
n_trials_lhc = 30
Exp_lhc.run_exp_auto(n_trials_lhc)

# Make the pareto plots 
plotting.pareto_front_exp(Exp_lhc, fill = False)
