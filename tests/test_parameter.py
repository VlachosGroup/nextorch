# -*- coding: utf-8 -*-
import numpy as np

from nextorch import io, bo, doe
import nextorch.utils as ut
from nextorch.parameter import Parameter, ParameterSpace

Xunit = doe.full_factorial([5,5])
xi = Xunit[:,0]

Y = np.array([1,2,3])

# p1 - ordinal
p1 = Parameter(x_type = 'ordinal', interval = 1, x_range = [0,2])
p1_encoding_unit = ut.unitscale_xv(p1.encoding, p1.x_range)

# p2 - categorical
p2_values = ['low', 'medium low', 'medium', 'medium high', 'high']
p2 = Parameter(x_type = 'categorical', values = p2_values)


def test_ordinal_parameter():
    # test on ordinal parameters
    p1_real = [0, 0.5, 1]
    assert p1_encoding_unit == p1_real


def test_categorical_parameter():
    # test on categorical parameters
    
    p2_encoding_unit = ut.unitscale_xv(p2.encoding, p2.x_range)
    assert p2_encoding_unit == [0, 0.25, 0.5, 0.75, 1]

#%% Use ParameterSpace
ps = ParameterSpace([p1, p2])
X_encode = ut.unit_to_encode_ParameterSpace(Xunit, ps)
X_real = ut.encode_to_real_ParameterSpace(X_encode, ps)

#%% Use database copy
ds = bo.Database()
ds.define_space([p1, p2])
ds.input_data(Xunit, Y, X_names = ['lol', 'ox'], standardized=True)

