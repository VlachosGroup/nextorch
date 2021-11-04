#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:40:16 2021

@author: yifanwang
"""

import numpy as np
from nextorch import plotting, bo, doe, io, utils


# Three input final temperature, heating rate, hold time
X_name_list = ['T', 'Catalyst loading', 'Time', 'Molar ratio']
X_units = ['C', 'mmol H+', 'hr', 'mol/mol']
#X_units_plot = [r'$\rm ^oC $', r'$\rm ^oC/min $', 'hr']

# Add the units
X_name_with_unit = []
for i, var in enumerate(X_name_list):
    if not X_units[i]  == '':
        var = var + ' ('+ X_units[i] + ')'
    X_name_with_unit.append(var)


# One output
Y_name_with_unit = 'Yield'

# combine X and Y names
var_names = X_name_with_unit + [Y_name_with_unit]

# Set the operating range for each parameter
X_ranges = [[40, 100], 
            [0.05, 0.3], 
            [1, 15],
            [1, 4]] 

# Set the reponse range
Y_plot_range = [0, 2.5]

# Get the information of the design space
n_dim = len(X_name_list) # the dimension of inputs
n_objective = 1 # the dimension of outputs
n_trials = 4 # number of experiment iterations




# Assume we run Latin hypder cube to create the initial samplinmg
n_init_lhc = 16
X_init_lhc = doe.latin_hypercube(n_dim = n_dim, n_points = n_init_lhc)

# Convert the sampling plan to a unit scale
X_init_real = utils.inverse_unitscale_X(X_init_lhc, X_ranges)



# Visualize the sampling plan,
# Sampling_3d takes in X in unit scales
plotting.sampling_3d(X_init_lhc,
                     X_names = X_name_with_unit,
                     X_ranges = X_ranges,
                     design_names = 'LHC')
