"""
Example 5

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
from nextorch import plotting, bo, doe, io
import pandas as pd


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
    
# One output
Y_name_with_unit = 'N_Content'
Y_name_with_unit_plot = r'$\rm N_{Content}%$'

# combine X and Y names
var_names = X_name_with_unit + [Y_name_with_unit]

# Import data
file_path = os.path.join(project_path, 'examples', 'ecatalyst', 'synthesis_data.csv')
data, data_full = io.read_csv(file_path, var_names = var_names)

iter_no = data_full['Iteration']

X_real, Y_real = io.split_X_y(data, Y_names = Y_name_with_unit)

# Initial Data 
X_ranges = [[300, 500], [3, 8], [2, 6]]
X_init = np.array(X_real[iter_no==0])


# Latin hypder cube
# Compare the two sampling plans
plotting.sampling_3d(X_init, 
                     X_names = X_name_with_unit,
                     X_ranges = X_ranges)
