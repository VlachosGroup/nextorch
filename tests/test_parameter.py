# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:44:03 2021

@author: yifan
"""
import os
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

from nextorch import io, bo, doe
import nextorch.utils as ut


X = doe.full_factorial([5,5])
xi = X[:,0]


# test on ordinal parameters
p1 = bo.Parameter(x_type = 'ordinal', interval = 1, x_range = [0,2])

xi_encoded_1 = ut.encode_xv(xi, p1.encoding)
xi_values_1 = ut.decode_xv(xi_encoded_1, p1.encoding, p1.values)


#%%
# test on categorical parameters
p2_values = ['low', 'medium low', 'medium', 'medium high', 'high']
p2 = bo.Parameter(x_type = 'categorical', values = p2_values)


xi_encoded_2 = ut.encode_xv(xi, p2.encoding)
xi_values_2 = ut.decode_xv(xi_encoded_2, p2.encoding, p2.values)