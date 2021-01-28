"""
Example 3

Goal: minimization
Objective function: 2D Langmuir Hinshelwood mechanism
    Input (X) dimension: 2
    Output (Y) dimension: 1
    Analytical form available: Yes
Acqucision function: the default, expected improvement (EI)
Initial Sampling: full factorial and Latin Hypercube
Input X scale: unit

"""

import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

import numpy as np
from nextorch import plotting, bo

#%% Define the objective function
def rate(P): 
    '''
    langmuir hinshelwood mechanism
    X is a matrix of [P1, P2] in real units
    return r is the reactio rate
    '''
    # Constants in the function
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

objective_func = rate
