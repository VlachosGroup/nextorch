# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:44:03 2021

@author: yifan
"""

import numpy as np
import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

from nextorch import io, bo, doe
import nextorch.utils as ut
from nextorch.parameter import Parameter, ParameterSpace

Xunit = doe.full_factorial([5,5])
xi = Xunit[:,0]

Y = np.array([1,2,3])
#%%
# test on ordinal parameters
p1 = Parameter(x_type = 'ordinal', interval = 1, x_range = [0,2])
p1_encoding_unit = ut.unitscale_xv(p1.encoding, p1.x_range)

def test_parameter():
    p1 = Parameter(x_type = 'ordinal', interval = 1, x_range = [0,2])
    p1_encoding_unit = ut.unitscale_xv(p1.encoding, p1.x_range)
    p1_real = [0, 0.5, 1]
    assert p1_encoding_unit == p1_real


# encoding
xi_encoded_1 = ut.encode_xv(xi, p1_encoding_unit)
# decoding
xi_values_1 = ut.decode_xv(xi_encoded_1, p1_encoding_unit, p1.values)


#%%
# test on categorical parameters
p2_values = ['low', 'medium low', 'medium', 'medium high', 'high']
p2 = Parameter(x_type = 'categorical', values = p2_values)
p2_encoding_unit = ut.unitscale_xv(p2.encoding, p2.x_range)

# encoding
xi_encoded_2 = ut.encode_xv(xi,p2_encoding_unit )
# decoding
xi_values_2 = ut.decode_xv(xi_encoded_2, p2_encoding_unit , p2.values)


#%% Use ParameterSpace
ps = ParameterSpace([p1, p2])
X_encode = ut.unit_to_encode_ParameterSpace(Xunit, ps)
X_real = ut.encode_to_real_ParameterSpace(X_encode, ps)



#%% Use database copy
ds = bo.Database()
ds.define_space([p1, p2])

ds.input_data(Xunit, Y, X_names = ['lol', 'ox'], standardized=True)



#%% Test on continuous
p3 = Parameter()
ps2 = ParameterSpace([p3, p3])

X_encode = ut.unit_to_encode_ParameterSpace(Xunit, ps2)
