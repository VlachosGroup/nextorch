# # Example 6 - Multi-objective optimization for an ellipse function
# 
# In this example, we will demonstrate how Bayesian Optimization can perform multi-objective optimization (MOO) and create a Pareto front. We will use a hyperthetical function which has a shape of an ellispe:
# 
# $ y_1 = x $ and $ y_2 = \sqrt{(1 - x^2/4)} $
# 
# where $ y_1^2 + y_2^2/4 = 1 $. `x` is the only input parameter. `y1` and `y2` are two output reponses which cannot be optimized jointly. 
# 
# Multi-objective optimization derives a set of solutions that define the tradeoff between competing objectives. The boundary defined by the entire feasible solution set is called the Pareto front. 
# 
# In `nextorch`, we implement weighted sum method to construct the Pareto front. It is commonly used for convex problems. A set of objectives are scalarized to a single objective by adding each objective pre-multiplied by a user-supplied weight. The weight of an objective is chosen in proportion to its relative importance. The optimization is simply performed with respected to the scalarized objective. By varying the weight combinations, we can construct the whole Pareto front. 
# 
# For this example, the scalarized objective can be written as,
# $$ y = w_1 y_1 + w_2 y_2 $$
# where the weights $ w_1, w_2 \in [0, 1] $ and $w_1 + w_2 = 1 $.
# 
# The details of this example is summarized in the table below:
# 
# | Key Item      | Description |
# | :----------------- | :----------------- |
# | Goal | Maximization, two objectives |
# | Objective function | Ellipse function |
# | Input (X) dimension | 1 |
# | Output (Y) dimension | 2 |
# | Analytical form available? | Yes |
# | Acqucision function | Expected improvement (EI) |
# | Initial Sampling | Latin hypercube | 
# 
# Next, we will go through each step in Bayesian Optimization.
# %% [markdown]
# ## 1. Import `nextorch` and other packages

# %%
import os
import sys
import time
from IPython.display import display

project_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
sys.path.insert(0, project_path)

import numpy as np
from nextorch import plotting, bo, doe, utils, io

# %% [markdown]
# ## 2. Define the objective function and the design space
# We import the PFR model, and wrap it in a Python function called `PFR` as the objective function `objective_func`. 
# 
# The ranges of the input X are specified. 

# %%
#%% Define the objective function
def ellipse(X_real):
    """ellipse function

    Parameters
    ----------
    X_real : numpy matrix
        input parameter

    Returns
    -------
    Y_real: numpy matrix
        y1 and y2
    """
    if len(X_real.shape) < 2:
        X_real = np.expand_dims(X_real, axis=1) #If 1D, make it 2D array
        
    y1 = X_real.copy()
    y2 = np.sqrt(1 - X_real**2/4)
    
    Y_real = np.concatenate((y1, y2), axis = 1)
        
    return Y_real # y1, y2


# Objective function
objective_func = ellipse


#%% Define the design space
X_name = ['x']
    
# two outputs
Y_names = [r'$\rm y_1$', r'$\rm y_2$']

# combine X and Y names
var_names = X_name + Y_names

# Set the operating range for each parameter
X_ranges =  [[0, 2]]

# Get the information of the design space
n_dim = 1 # the dimension of inputs
n_objective = 2 # the dimension of outputs

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

# %% [markdown]
# ## 4. Initialize an `Experiment` object 
# 
# In this example, we use an `MOOExperiment` object, a class designed for multi-objective optimization. It can handle multiple weight combinations, perform the scalarized objective optimization automatically, and construct the entire Pareto front. 
# 
# An `MOOExperiment` is a subclass of `Experiment`. It requires all key components as `Experiment`:
# - Name of the experiment, used for output folder name
# - Input independent variables X: `X_init` or `X_init_real`
# - List of X ranges: `X_ranges`
# - Output dependent variables Y: `Y_init` or `Y_init_real`
# 
# Optional:
# - `unit_flag`: `True` if the input X matrix is a unit scale, else `False`
# - `objective_func`: Used for test plotting
# - `maximize`: `True` if we look for maximum, else `False` for minimum
# 
# Additionally, `weights` is required for `MOOExperiment.set_optim_specs` function. It defines a list of weights for objective 1. The weights of objective 2 is 1 minus that of objective 1. Under the hood, each weight combination correponds to a single `Experiment` object, each with a different scalarized objective. 
# 
# Some progress status will be printed out while initializing all single `Experiment` objects.

# %%
#%% Initialize an multi-objective Experiment object
# Set its name, the files will be saved under the folder with the same name
Exp_lhc = bo.MOOExperiment('ellipse_MOO')  
# Import the initial data
Exp_lhc.input_data(X_init_lhc, 
                   Y_init_lhc, 
                   X_ranges = X_ranges, 
                   X_names = X_name,
                   Y_names = Y_names,
                   unit_flag = False)

# Set the optimization specifications 
# here we set the objective function, minimization by default
# 10 weights, 10 Experiments
n_exp = 21 # number of single Experiments

# Set a weight vector for objective 1
weights_obj_1 = np.linspace(0, 1, n_exp)
weights_obj_2 = 1 - weights_obj_1

# Set a timer
start_time = time.time()
Exp_lhc.set_optim_specs(objective_func = objective_func, 
                        maximize = True, 
                        weights = weights_obj_1)
end_time = time.time()
print('Initializing {} Experiments takes {:.2f} minutes.'.format(n_exp, (end_time-start_time)/60))

# %% [markdown]
# ## 5. Run trials 
# 
# At each weight combinations, we perform an optimization task for the scalarized objective (a single `Experiment`). `MOOExperiment.run_exp_auto` run these tasks automatically by using the default choice of acqucision function, Expected improvement (EI). It takes in the number of trials required for each `Experiment`. The number of trials needs to be large enough which allows Bayesian Optimization algorithm to converge to the optimum. Nevertheless, the optimization of `y1` and `y2` are rather trivial due to their simple analytical expression. We will do 10 trials for each experiment. The total number of calls for the objective function is `n_trails` * `n_exp` (=210). 
# 
# Some progress status will be printed out during the training.

# %%
# Set the number of iterations for each experiments
n_trials_lhc = 10 
# Set a timer
start_time = time.time()
Exp_lhc.run_exp_auto(n_trials_lhc)

end_time = time.time()
print('Optimizing {} Experiments takes {:.2f} minutes.'.format(n_exp, (end_time-start_time)/60))

# %% [markdown]
# ## 6. Visualize the Pareto front
# We can get the Pareto set directly from the `MOOExperiment` object by using `MOOExperiment.get_optim`.
# 
# To visualize the Pareto front, `y1` values are plotted against `y2` values. The scatter points resemble an ellispe shape, incidating the method is able to map out the entire front. 

# %%
# Extract the set of optimal solutions
Y_real_opts, X_real_opts = Exp_lhc.get_optim()
weight_names = [r'$\rm w_1$', r'$\rm w_2$'] 

# Parse the optimum into a table
data_opt = io.np_to_dataframe([weights_obj_1, weights_obj_2, X_real_opts, Y_real_opts], weight_names + var_names, n = n_exp)
display(data_opt.round(decimals=2))

# Make the pareto plots 
plotting.pareto_front_exp(Exp_lhc, fill = True, diagonal = False)


