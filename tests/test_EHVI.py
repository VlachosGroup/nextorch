# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Example 11 - Multi-objective optimization for plug flow reactor
# 
# In [Example 7](07_PFR_MOO.ipynb), we demonstrated how Bayesian Optimization can perform multi-objective optimization (MOO) and create a Pareto front using weighted objectives. In this example we will demonstrate how to use the acquisition function, Expected Hypervolume Improvement (EHVI) and its Monte Carlo variant (q-EHVI),[[1]](https://arxiv.org/abs/2006.05078) to perform the MOO. The interested reader is referred to reference 1-3 for more information. 
# 
# We will use the same PFR model. Two output variables will be generated from the model: yield (`y1`) and selectivity (`y2`). Unfortunately, these two variables cannot be maximized simultaneously. An increase in yield would lead to a decrease in selectivity, and vice versa. 
# 
# The details of this example is summarized in the table below:
# 
# | Key Item      | Description |
# | :----------------- | :----------------- |
# | Goal | Maximization, two objectives |
# | Objective function | PFR model |
# | Input (X) dimension | 3 |
# | Output (Y) dimension | 2 |
# | Analytical form available? | Yes |
# | Acqucision function | q-Expected Hypervolume improvement (qEHVI) |
# | Initial Sampling | Latin hypercube | 
# 
# Next, we will go through each step in Bayesian Optimization.
# %% [markdown]
# ## 1. Import `nextorch` and other packages
# %% [markdown]
# ## 2. Define the objective function and the design space
# We import the PFR model, and wrap it in a Python function called `PFR` as the objective function `objective_func`. 
# 
# The ranges of the input X are specified. 

# %%
import warnings
warnings.filterwarnings("ignore")

import time
from IPython.display import display

import numpy as np
from nextorch import plotting, bo, doe, utils, io

import os
import sys
PFR_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'examples', 'PFR'))
sys.path.insert(0, PFR_path)
# %%
#%% Define the objective function
from fructose_pfr_model_function import Reactor

def PFR(X_real):
    """PFR model

    Parameters
    ----------
    X_real : numpy matrix
        reactor parameters: 
        T, pH and tf in real scales

    Returns
    -------
    Y_real: numpy matrix
        reactor yield and selectivity
    """
    if len(X_real.shape) < 2:
        X_real = np.expand_dims(X_real, axis=1) #If 1D, make it 2D array
        
    Y_real = []
    for i, xi in enumerate(X_real):
        Conditions = {'T_degC (C)': xi[0], 'pH': xi[1], 'tf (min)' : 10**xi[2]}
        yi = Reactor(**Conditions)        
        Y_real.append(yi)
            
    Y_real = np.array(Y_real)
        
    return Y_real # yield, selectivity


# Objective function
objective_func = PFR


#%% Define the design space
# Three input temperature C, pH, log10(residence time)
X_name_list = ['T', 'pH', r'$\rm log_{10}(tf_{min})$']
X_units = [r'$\rm ^{o}C $', '', '']

# Add the units
X_name_with_unit = []
for i, var in enumerate(X_name_list):
    if not X_units[i]  == '':
        var = var + ' ('+ X_units[i] + ')'
    X_name_with_unit.append(var)
    
# two outputs
Y_name_with_unit = ['Yield %', 'Selectivity %']

# combine X and Y names
var_names = X_name_with_unit + Y_name_with_unit

# Set the operating range for each parameter
X_ranges =  [[140, 200], # Temperature ranges from 140-200 degree C
             [0, 1], # pH values ranges from 0-1 
             [-2, 2]] # log10(residence time) ranges from -2-2  


# Get the information of the design space
n_dim = len(X_name_list) # the dimension of inputs
n_objective = len(Y_name_with_unit) # the dimension of outputs

# %% [markdown]
# ## 3. Define the initial sampling plan
# Here we use LHC design with 10 points for the initial sampling. The initial reponse in a real scale `Y_init_real` is computed from the objective function.

# %%
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

# %% [markdown]
# ## 4. Initialize an `Experiment` object 
# 
# In this example, we use an `EHVIMOOExperiment` object, a class designed for multi-objective optimization using `Expected Hypervolume Improvement` as aquisition funciton. It can handle multiple weight combinations, perform the scalarized objective optimization automatically, and construct the entire Pareto front. 
# 
# An `EHVIMOOExperiment` is a subclass of `Experiment`. It requires all key components as `Experiment`. Additionally, `ref_point` is required for `EHVIMOOExperiment.set_ref_point` function. It defines a list of values that are slightly worse than the lower bound of objective values, where the lower bound is the minimum acceptable value of interest for each objective. It would be helpful if the user know the rough values using domain knowledge prior to optimization. 

# %%
#%% Initialize an multi-objective Experiment object
# Set its name, the files will be saved under the folder with the same name
Exp_lhc = bo.EHVIMOOExperiment('PFR_yield_MOO_EHVI')  
# Import the initial data
Exp_lhc.input_data(X_init_lhc, 
                   Y_init_lhc, 
                   X_ranges = X_ranges, 
                   X_names = X_name_with_unit,
                   Y_names = Y_name_with_unit,
                   unit_flag = True)

# Set the optimization specifications 
# here we set the reference point
ref_point = [10.0, 10.0]

# Set a timer
start_time = time.time()
Exp_lhc.set_ref_point(ref_point)
Exp_lhc.set_optim_specs(objective_func = objective_func, 
                        maximize = True)
end_time = time.time()
print('Initializing the experiment takes {:.2f} minutes.'.format((end_time-start_time)/60))

# %% [markdown]
# ## 5. Run trials 
# 
# `EHVIMOOExperiment.run_trials_auto` can run these tasks automatically by specifying the acqucision function, q-Expected Hypervolume Improvement (`qEHVI`). In this way, we generate one point per iteration in default. Alternatively, we can manually specify the number of next points we would like to obtain.
# 
# Some progress status will be printed out during the training. It takes 0.2 miuntes to obtain the whole front, much shorter than the 12.9 minutes in the weighted method. 

# %%
# Set the number of iterations for each experiments
n_trials_lhc = 30 

# run trials
# Exp_lhc.run_trials_auto(n_trials_lhc, 'qEHVI')

for i in range(n_trials_lhc):
    # Generate the next experiment point
    X_new, X_new_real, acq_func = Exp_lhc.generate_next_point(n_candidates=4)
    # Get the reponse at this point
    Y_new_real = objective_func(X_new_real) 
    # or 
    # Y_new_real = bo.eval_objective_func(X_new, X_ranges, objective_func)

    # Retrain the model by input the next point into Exp object
    Exp_lhc.run_trial(X_new, X_new_real, Y_new_real)

end_time = time.time()
print('Optimizing the experiment takes {:.2f} minutes.'.format((end_time-start_time)/60))

# %% [markdown]
# ## 6. Visualize the Pareto front
# We can get the Pareto set directly from the `EHVIMOOExperiment` object by using `EHVIMOOExperiment.get_optim`.
# 
# To visualize the Pareto front, `y1` values are plotted against `y2` values. The region below $ y=x $ is infeasible for the PFR model and we have no Pareto points fall below the line, incidating the method is validate. Besides, all sampling points are shown as well.

# %%
Y_real_opts, X_real_opts = Exp_lhc.get_optim()

# Parse the optimum into a table
data_opt = io.np_to_dataframe([X_real_opts, Y_real_opts], var_names)
display(data_opt.round(decimals=2))

# Make the pareto front
plotting.pareto_front(Y_real_opts[:, 0], Y_real_opts[:, 1], Y_names=Y_name_with_unit, fill=False)

# All sampling points
plotting.pareto_front(Exp_lhc.Y_real[:, 0], Exp_lhc.Y_real[:, 1], Y_names=Y_name_with_unit, fill=False)

# %% [markdown]
# ## References:
# 
# 1. Daulton, S.; Balandat, M.; Bakshy, E. Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization. arXiv 2020, No. 1, 1–30.
# 
# 2. BoTorch tutorials using qHVI: https://botorch.org/tutorials/multi_objective_bo; https://botorch.org/tutorials/constrained_multi_objective_bo
# 
# 3. Ax tutorial using qHVI: https://ax.dev/versions/latest/tutorials/multiobjective_optimization.html
# 
# 4. Desir, P.; Saha, B.; Vlachos, D. G. Energy Environ. Sci. 2019.
# 
# 5. Swift, T. D.; Bagia, C.; Choudhary, V.; Peklaris, G.; Nikolakis, V.; Vlachos, D. G. ACS Catal. 2014, 4 (1), 259–267
# 
# 6. The PFR model can be found on GitHub: https://github.com/VlachosGroup/Fructose-HMF-Model
# 
# 

